// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// RetrievalAttention — decode-side block-sparse attention backend
// for mlx-swift-lm. Spec 034. See [[Open Sparse Stack PRD]] for the
// full design rationale.
//
// What this is, in 30 seconds:
//   - Cache stores post-RoPE rotated K (verified [F-01] on Llama / Qwen2 /
//     Qwen3 / Qwen3.5: all rotate before `attentionWithCacheUpdate`).
//   - Per query, score 64-token blocks of f(K) embeddings against f(Q).
//   - Score = (1−λ)·content_score + λ·trig_score + recency_bias.
//   - Pick top-32 fine blocks (2048 tokens) + top-2 coarse blocks
//     (2048 tokens) from a separate 1024-token-block index.
//   - Add static [0..128] + sliding [L−2048..L].
//   - Deduplicate the union to ≤6272 unique positions.
//   - `mx.take` (Swift: `take(_:axis:)`) the deduplicated K/V into a
//     contiguous tensor; run dense `scaledDotProductAttention` on it.
//   - Stays on the fused Metal kernel fast path. No sparse mask.
//
// v0: scaffolding + dedupe + config. Math + tests next.

import Foundation
import MLX
import MLXNN

// MARK: - Config

/// PRD Decisions 6, 7, 8, 13, 14, 19 baked into a single config struct.
/// All defaults match the PRD execution-ready (v5) numbers.
public struct RetrievalAttentionConfig: Sendable {

    // ----- Static-window (Decision 6: 128 init + 2048 sliding)

    /// Number of initial tokens always attended (PRD line 122).
    public var staticInit: Int = 128

    /// Number of trailing tokens always attended (PRD line 123).
    public var slidingWindow: Int = 2048

    // ----- Fine retrieved blocks (Decision 8)

    /// Tokens per fine block (PRD line 124). Block-level retrieval makes
    /// the gather memory-coalesced; 64 is the published sweet spot.
    public var fineBlockSize: Int = 64

    /// How many fine blocks to retrieve per query (PRD line 124).
    public var fineTopK: Int = 32

    // ----- Coarse rescue (Decision 14)

    /// Tokens per coarse rescue block (PRD line 125).
    public var coarseBlockSize: Int = 1024

    /// How many coarse blocks to retrieve per query (PRD line 125).
    public var coarseTopK: Int = 2

    /// Whether the coarse-rescue branch is enabled. PRD ships this on by
    /// default; Week 2 A/B 4 toggles for ablation (PRD line 302).
    public var coarseRescueEnabled: Bool = true

    // ----- Selector dim (Decision 7)

    /// JL content projection output dim (PRD line 159).
    public var contentDim: Int = 16

    /// V3-trig basis output dim (PRD line 160). 8 freq × 2 phases.
    public var trigDim: Int = 16

    /// Total selector embedding dim = contentDim + trigDim.
    public var selectorDim: Int { contentDim + trigDim }

    // ----- Scoring (Decisions 4, 19; PRD line 161)

    /// Content vs position blend. 0=pure content, 1=pure position.
    /// PRD default 0.5; Week 2 A/B 1 sweep ∈ {0.25, 0.5, 0.75}.
    public var lambdaPos: Float = 0.5

    /// ALiBi-style recency bias coefficient (PRD line 166).
    /// score += -alpha · log(distance_in_blocks + 1).
    /// Default 0.0 (off until Week 2 A/B 3 sweep).
    public var recencyAlpha: Float = 0.0

    // ----- GQA aggregation (Decision 19)

    public enum GQAAggregation: String, Sendable {
        /// Each KV group picks based on the strongest Q-head signal.
        /// NSA convention. Default v1.
        case perKVGroupMax = "per_kv_group_max"

        /// Each Q head picks its own top-K; union the 4 sets per KV
        /// group, cap at budget. Week 2 A/B 6.
        case cappedUnion = "capped_union"
    }

    public var gqaAggregation: GQAAggregation = .perKVGroupMax

    // ----- Per-layer hybrid (Decision 9)

    /// Number of initial dense layers (first-N). PRD line 396.
    public var denseFirstN: Int = 4

    /// Number of trailing dense layers (last-N).
    public var denseLastN: Int = 4

    /// Returns true if this layer index runs RetrievalAttention,
    /// false if it should fall back to dense attention.
    public func isSparseLayer(layerIdx: Int, totalLayers: Int) -> Bool {
        guard layerIdx >= denseFirstN else { return false }
        guard layerIdx < totalLayers - denseLastN else { return false }
        return true
    }

    // ----- Sentinel embedding (Decision adjacent, A/B 7)

    /// Whether to ALSO compute max-norm sentinel per block and score
    /// against max(score_mean, score_sentinel). [F-05] shows this
    /// HURTS on synthetic Gaussian K; only enable for real-K tests
    /// where magnitude correlates with semantic specificity.
    public var sentinelEnabled: Bool = false

    public init() {}
}

// MARK: - Dedupe (PRD line 129-134, validated by [F-01] dedupe.py tests)

/// Build a sorted, deduplicated array of token positions to gather.
///
/// The four RA regions can overlap (static + sliding touch at small
/// contexts; fine top-k may pick blocks inside sliding; coarse may
/// overlap fine). Without dedupe, a token attended twice gets its
/// softmax weight effectively doubled — wrong attention semantics.
///
/// - Parameters:
///   - seqLen: current cache length (number of populated KV positions).
///   - fineBlockStarts: token positions for each fine block's start.
///   - coarseBlockStarts: token positions for each coarse block's start.
///   - config: backend config (controls all four region sizes).
/// - Returns: sorted unique positions, all in `[0, seqLen)`.
public func retrievalAttentionGatherIndices(
    seqLen: Int,
    fineBlockStarts: [Int],
    coarseBlockStarts: [Int],
    config: RetrievalAttentionConfig
) -> [Int] {
    precondition(seqLen >= 0, "seqLen must be non-negative")

    var indices = Set<Int>()
    indices.reserveCapacity(
        config.staticInit + config.slidingWindow
            + config.fineTopK * config.fineBlockSize
            + (config.coarseRescueEnabled
                ? config.coarseTopK * config.coarseBlockSize : 0))

    // Static initial window: positions 0 ..< min(staticInit, seqLen).
    let staticEnd = min(config.staticInit, seqLen)
    for p in 0..<staticEnd { indices.insert(p) }

    // Sliding window: trailing `slidingWindow` tokens, floored at 0.
    let slidingStart = max(0, seqLen - config.slidingWindow)
    if slidingStart < seqLen {
        for p in slidingStart..<seqLen { indices.insert(p) }
    }

    // Fine retrieved blocks.
    for s in fineBlockStarts {
        precondition(
            s >= 0 && s < seqLen,
            "fine block start \(s) out of [0, \(seqLen))")
        let end = min(s + config.fineBlockSize, seqLen)
        for p in s..<end { indices.insert(p) }
    }

    // Coarse rescue blocks.
    if config.coarseRescueEnabled {
        for s in coarseBlockStarts {
            precondition(
                s >= 0 && s < seqLen,
                "coarse block start \(s) out of [0, \(seqLen))")
            let end = min(s + config.coarseBlockSize, seqLen)
            for p in s..<end { indices.insert(p) }
        }
    }

    return indices.sorted()
}

/// Pre-dedupe budget exposed for memory math + diagnostics. Does NOT
/// account for overlap collapse. PRD line 127.
public func retrievalAttentionPreDedupeBudget(
    config: RetrievalAttentionConfig
) -> Int {
    config.staticInit
        + config.slidingWindow
        + config.fineTopK * config.fineBlockSize
        + (config.coarseRescueEnabled ? config.coarseTopK * config.coarseBlockSize : 0)
}
