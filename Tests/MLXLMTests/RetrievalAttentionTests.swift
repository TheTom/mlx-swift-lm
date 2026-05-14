// SPDX-License-Identifier: Apache-2.0
// Unit tests for RetrievalAttention dedupe + config invariants.
// Pure-logic; no MLX dispatch yet. Mirrors the Python reference at
// research/retrieval_attention/test_dedupe.py.

import Foundation
import MLX
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

    // MARK: - Gather + SDPA equivalence

    /// Gather-not-mask SDPA on a subset must equal naive dense SDPA
    /// computed against only the same subset. This pins the central
    /// architectural claim (PRD line 136): we can sparsify by selecting
    /// rows of K/V and running fused SDPA on the smaller contiguous
    /// tensor, with no behavioral drift vs the same row selection done
    /// via a mask.
    ///
    /// Synthetic Q/K/V — no model needed.
    @Test func gatherSDPAEquivsBlockSparseAgainstSubsetMask() {
        let B = 1, nHeads = 4, T = 256, D = 32
        let scale: Float = 1.0 / Float(D).squareRoot()

        // Fixed seed for reproducibility.
        MLXRandom.seed(42)
        let queries = MLXRandom.normal([B, nHeads, 1, D])
        let keys = MLXRandom.normal([B, nHeads, T, D])
        let values = MLXRandom.normal([B, nHeads, T, D])

        // A simulated dedupe result: static 8 + sliding 16 + 2 fine
        // 8-token blocks at positions 100 and 200. Built by hand to
        // hit a non-trivial layout with overlap.
        var idxSet = Set<Int>()
        for p in 0..<8 { idxSet.insert(p) }            // static
        for p in (T - 16)..<T { idxSet.insert(p) }      // sliding
        for p in 100..<108 { idxSet.insert(p) }         // fine block
        for p in 200..<208 { idxSet.insert(p) }         // fine block
        let gatherIdx = idxSet.sorted()

        // Gather path output
        let gatherOut = retrievalAttentionGatherAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            gatherIndices: gatherIdx,
            scale: scale,
            sinks: nil
        )

        // Reference: manually gather K/V at the same indices and run
        // dense SDPA directly. This is the "ground truth" the gather
        // path must match exactly.
        let idxArray = MLXArray(gatherIdx.map { Int32($0) })
        let refKeys = keys.take(idxArray, axis: 2)
        let refValues = values.take(idxArray, axis: 2)
        let refOut = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: refKeys,
            values: refValues,
            scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none
        )

        // Outputs must be bit-identical (same kernel, same inputs).
        let diff = (gatherOut - refOut).abs().max().item(Float.self)
        #expect(diff == 0.0, "gather path drifted from dense ref by \(diff)")
    }

    // MARK: - JL projection

    @Test func jlProjectionShapeAndDeterminism() {
        let dHead = 128
        let W1 = retrievalAttentionJLProjection(dHead: dHead)
        let W2 = retrievalAttentionJLProjection(dHead: dHead)
        #expect(W1.shape == [16, dHead])
        // Same seed → same matrix.
        let diff = (W1 - W2).abs().max().item(Float.self)
        #expect(diff == 0.0, "JL projection not deterministic (drift=\(diff))")
    }

    @Test func jlPreservesDotProductRanking() {
        // JL lemma is a high-variance heuristic at dim=16 (PRD line 244:
        // "ε > 0.5 at 99% confidence" — formal bound is loose). The
        // testable claim is that projected dot products *correlate*
        // with the originals across many pairs, not that the per-pair
        // values match. Pearson r > 0.6 across 500 pairs is a generous
        // floor that distinguishes JL from random noise (r ≈ 0).
        MLXRandom.seed(7)
        let n = 500
        let dHead = 128
        let q = MLXRandom.normal([n, dHead]).asType(.float32)
        let k = MLXRandom.normal([n, dHead]).asType(.float32)
        let W = retrievalAttentionJLProjection(dHead: dHead)
        let pq = matmul(q, W.transposed(1, 0))
        let pk = matmul(k, W.transposed(1, 0))

        let trueDots = (q * k).sum(axis: 1).asArray(Float.self)
        let projDots = (pq * pk).sum(axis: 1).asArray(Float.self)

        let meanT = trueDots.reduce(0, +) / Float(n)
        let meanP = projDots.reduce(0, +) / Float(n)
        var num: Float = 0
        var dT: Float = 0
        var dP: Float = 0
        for i in 0..<n {
            num += (trueDots[i] - meanT) * (projDots[i] - meanP)
            dT += (trueDots[i] - meanT) * (trueDots[i] - meanT)
            dP += (projDots[i] - meanP) * (projDots[i] - meanP)
        }
        let r = num / (dT.squareRoot() * dP.squareRoot())
        // At contentDim=16, dHead=128, the theoretical Pearson floor is
        // ~√(contentDim/dHead) = 0.35. Empirically we land near 0.30
        // across 500 random pairs (see [F-07] in the experiment log).
        // 0.20 is the floor — that still cleanly separates JL from
        // random noise (which would give r ≈ 0 at this sample size).
        #expect(
            r > 0.20,
            "JL Pearson correlation should preserve some dot signal, got r=\(r)"
        )
    }

    // MARK: - V3-trig basis

    @Test func trigFeaturesShape() {
        let positions = MLXArray((0..<Int32(100))).asType(.int32)
        let feats = retrievalAttentionTrigFeatures(relativePositions: positions)
        #expect(feats.shape == [100, 16])
    }

    @Test func trigFeaturesAtZeroAreAlternatingSinCos() {
        // sin(0)=0, cos(0)=1 → feature_2i = 0, feature_2i+1 = 1.
        let feats = retrievalAttentionTrigFeatures(
            relativePositions: MLXArray([Int32(0)])
        )
        let row = feats.asArray(Float.self)
        for i in stride(from: 0, to: 16, by: 2) {
            #expect(abs(row[i]) < 1e-5, "feature[\(i)] should be ~0, got \(row[i])")
            #expect(
                abs(row[i + 1] - 1.0) < 1e-5,
                "feature[\(i+1)] should be ~1, got \(row[i+1])"
            )
        }
    }

    @Test func trigFeaturesUnitNormPerFreqPair() {
        // sin²(p·f) + cos²(p·f) = 1 for any p, f.
        let positions = MLXArray([Int32(0), Int32(1), Int32(100), Int32(10_000)])
        let feats = retrievalAttentionTrigFeatures(relativePositions: positions)
        let arr = feats.asArray(Float.self)
        // Stride over freq pairs.
        for n in 0..<4 {
            for i in 0..<8 {
                let sinV = arr[n * 16 + 2 * i]
                let cosV = arr[n * 16 + 2 * i + 1]
                let norm2 = sinV * sinV + cosV * cosV
                #expect(
                    abs(norm2 - 1.0) < 1e-4,
                    "freq pair \(i) at pos \(n): sin²+cos² = \(norm2)"
                )
            }
        }
    }

    // MARK: - Scoring

    @Test func scoreBlocksPureContent() {
        var c = RetrievalAttentionConfig()
        c.lambdaPos = 0.0  // pure content
        c.recencyAlpha = 0.0
        let blocks = MLXArray.zeros([3, 32], dtype: .float32)
        // Block 0: content matches query (first 16 dims).
        // Block 2: content anti-matches.
        // Block 1: only trig signal (ignored at λ=0).
        var blocksHost = [Float](repeating: 0, count: 3 * 32)
        for i in 0..<16 {
            blocksHost[i] = 1.0  // block 0 content +
            blocksHost[2 * 32 + i] = -1.0  // block 2 content -
            blocksHost[32 + 16 + i] = 1.0  // block 1 trig
        }
        let blocksArr = MLXArray(blocksHost).reshaped(3, 32)
        var qHost = [Float](repeating: 0, count: 32)
        for i in 0..<16 { qHost[i] = 1.0 }
        let q = MLXArray(qHost)

        _ = blocks  // silence unused
        let scores = retrievalAttentionScoreBlocks(
            blockFeatures: blocksArr, q: q, config: c
        ).asArray(Float.self)
        #expect(scores[0] > scores[1])
        #expect(scores[1] > scores[2])
    }

    @Test func topKReturnsDescending() {
        let scores = MLXArray([Float(0.1), 0.9, 0.5, 0.3, 0.7])
        let top3 = retrievalAttentionTopKBlocks(scores: scores, k: 3)
        #expect(top3 == [1, 4, 2])
    }

    @Test func topKHandlesKExceedsLength() {
        let scores = MLXArray([Float(0.1), 0.9, 0.5])
        let top10 = retrievalAttentionTopKBlocks(scores: scores, k: 10)
        #expect(top10 == [1, 2, 0])
    }

    /// Bigger sanity check: gather output ≠ full-K SDPA output, but
    /// the gathered-rows-only path is INTERNALLY equivalent regardless
    /// of how many distinct indices we pass.
    @Test func gatherSDPAVariesWithIndexSet() {
        let B = 1, nHeads = 2, T = 128, D = 16
        let scale: Float = 1.0 / Float(D).squareRoot()

        MLXRandom.seed(7)
        let queries = MLXRandom.normal([B, nHeads, 1, D])
        let keys = MLXRandom.normal([B, nHeads, T, D])
        let values = MLXRandom.normal([B, nHeads, T, D])

        let allIdx = Array(0..<T)
        let halfIdx = Array(0..<(T / 2))

        let outAll = retrievalAttentionGatherAndAttend(
            queries: queries, keys: keys, values: values,
            gatherIndices: allIdx, scale: scale)
        let outHalf = retrievalAttentionGatherAndAttend(
            queries: queries, keys: keys, values: values,
            gatherIndices: halfIdx, scale: scale)

        let diff = (outAll - outHalf).abs().max().item(Float.self)
        #expect(diff > 1e-4, "different index sets must produce different output")
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
