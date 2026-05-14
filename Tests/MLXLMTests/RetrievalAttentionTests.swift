// SPDX-License-Identifier: Apache-2.0
// Unit tests for RetrievalAttention dedupe + config invariants.
// Pure-logic; no MLX dispatch yet. Mirrors the Python reference at
// research/retrieval_attention/test_dedupe.py.

import Foundation
@testable import MLXLMCommon
import Testing

@Suite("RetrievalAttention dedupe + config")
struct RetrievalAttentionTests {

    // MARK: - Config sanity

    @Test func defaultConfigMatchesPRDV5() {
        let c = RetrievalAttentionConfig()
        #expect(c.staticInit == 128)
        #expect(c.slidingWindow == 2048)
        #expect(c.fineBlockSize == 64)
        #expect(c.fineTopK == 32)
        #expect(c.coarseBlockSize == 1024)
        #expect(c.coarseTopK == 2)
        #expect(c.contentDim == 16)
        #expect(c.trigDim == 16)
        #expect(c.selectorDim == 32)
        #expect(c.lambdaPos == 0.5)
        #expect(c.recencyAlpha == 0.0)
        #expect(c.denseFirstN == 4)
        #expect(c.denseLastN == 4)
        #expect(c.coarseRescueEnabled)
        #expect(!c.sentinelEnabled)  // [F-05]: off by default
    }

    @Test func preDedupeBudgetIs6272() {
        // PRD line 127: 128 + 2048 + 32·64 + 2·1024 = 6272.
        let c = RetrievalAttentionConfig()
        #expect(retrievalAttentionPreDedupeBudget(config: c) == 6272)
    }

    @Test func coarseDisableBudgetDrops() {
        var c = RetrievalAttentionConfig()
        c.coarseRescueEnabled = false
        #expect(retrievalAttentionPreDedupeBudget(config: c) == 6272 - 2048)
    }

    // MARK: - Layer hybrid policy (Decision 9)

    @Test func denseLayerGate() {
        let c = RetrievalAttentionConfig()
        // Llama-3-8B: 32 layers. first-4 + last-4 dense → middle 24 sparse.
        let total = 32
        for i in 0..<32 {
            let isSparse = c.isSparseLayer(layerIdx: i, totalLayers: total)
            if i < 4 || i >= total - 4 {
                #expect(!isSparse, "layer \(i) should be dense")
            } else {
                #expect(isSparse, "layer \(i) should be sparse")
            }
        }
    }

    @Test func tinyModelAllDense() {
        // 8-layer model with first-4 + last-4 dense = all dense.
        let c = RetrievalAttentionConfig()
        for i in 0..<8 {
            #expect(!c.isSparseLayer(layerIdx: i, totalLayers: 8))
        }
    }

    // MARK: - Dedupe correctness (mirror of Python test_dedupe.py)

    @Test func dedupeStaticWindowOnly() {
        // Week 1 Day 1: 128 init + 2048 sliding, no blocks. seq_len 10K.
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 10_000,
            fineBlockStarts: [],
            coarseBlockStarts: [],
            config: c)
        // Static head fully covered.
        for p in 0..<128 { #expect(idx.contains(p)) }
        // Sliding starts at 10_000 - 2048 = 7952.
        #expect(idx.contains(7952))
        #expect(idx.contains(9999))
        // Gap in the middle.
        #expect(!idx.contains(4000))
        // Total = 128 + 2048 = 2176 (regions don't overlap at 10K seqLen).
        #expect(idx.count == 2176)
    }

    @Test func dedupeShortContextRegionsCollapseToFullCover() {
        // At seqLen ≤ static + sliding, the union covers all positions.
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 2000,
            fineBlockStarts: [],
            coarseBlockStarts: [],
            config: c)
        // Sliding window floors at 0; static [0..127] ∪ sliding [0..1999] = [0..1999].
        #expect(idx.count == 2000)
        #expect(idx.first == 0)
        #expect(idx.last == 1999)
    }

    @Test func dedupeFineOverlappingSliding() {
        // seqLen 10K: sliding is [7952..9999]. Fine block at 9000
        // overlaps. Each position appears once; total = 2176 (no growth
        // because the fine block sits entirely inside sliding).
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 10_000,
            fineBlockStarts: [9000],
            coarseBlockStarts: [],
            config: c)
        #expect(idx.count == 2176)  // no duplicates added
        #expect(Set(idx).count == idx.count)  // no duplicates anywhere
    }

    @Test func dedupeCoarseAbsorbsFine() {
        // Fine block at 5000 (64 tokens) fits inside coarse block at
        // 4500 (1024 tokens). The intersection collapses correctly.
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 20_000,
            fineBlockStarts: [5000],
            coarseBlockStarts: [4500],
            config: c)
        // static 128 + sliding 2048 = 2176, plus coarse 1024 (entirely
        // outside static and sliding) = 3200. Fine 5000..5063 is fully
        // inside coarse 4500..5523 → no growth.
        #expect(idx.count == 2176 + 1024)
    }

    @Test func dedupeTwoFineBlocksOverlapping() {
        // Fine [5000..5063] and fine [5050..5113] overlap on [5050..5063]
        // → union = [5000..5113] = 114 tokens.
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 20_000,
            fineBlockStarts: [5000, 5050],
            coarseBlockStarts: [],
            config: c)
        #expect(idx.count == 2176 + 114)
    }

    @Test func dedupeSortedAscending() {
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 100_000,
            fineBlockStarts: [50_000, 30_000, 10_000],  // unsorted
            coarseBlockStarts: [70_000, 20_000],  // unsorted
            config: c)
        for i in 1..<idx.count { #expect(idx[i - 1] < idx[i]) }
    }

    @Test func dedupeAllUnderSeqLen() {
        let c = RetrievalAttentionConfig()
        let idx = retrievalAttentionGatherIndices(
            seqLen: 200,
            fineBlockStarts: [150],  // would naively span 150..213
            coarseBlockStarts: [100],  // 100..1123 capped to 100..199
            config: c)
        for p in idx {
            #expect(p >= 0)
            #expect(p < 200)
        }
    }

    @Test func dedupeNoCoarseWhenDisabled() {
        var c = RetrievalAttentionConfig()
        c.coarseRescueEnabled = false
        let idx = retrievalAttentionGatherIndices(
            seqLen: 100_000,
            fineBlockStarts: [10_000],
            coarseBlockStarts: [5000],  // should be ignored
            config: c)
        // 5000 not in result (coarse disabled), 10000..10063 is.
        #expect(idx.contains(10_000))
        #expect(!idx.contains(5000))
    }

    @Test func dedupe1MFullBudget() {
        // PRD example: 1M context, 32 fine blocks + 2 coarse, none
        // overlapping. Result should equal exactly 6272.
        let c = RetrievalAttentionConfig()
        let seqLen = 1_000_000
        let slidingStart = seqLen - c.slidingWindow
        // Fine blocks scattered, all before sliding.
        var fineStarts: [Int] = []
        for i in 0..<32 { fineStarts.append(10_000 + i * 20_000) }
        precondition(fineStarts.max()! + c.fineBlockSize < slidingStart)
        let coarseStarts = [800_000, 900_000]
        let idx = retrievalAttentionGatherIndices(
            seqLen: seqLen,
            fineBlockStarts: fineStarts,
            coarseBlockStarts: coarseStarts,
            config: c)
        #expect(idx.count == 6272)
    }
}
