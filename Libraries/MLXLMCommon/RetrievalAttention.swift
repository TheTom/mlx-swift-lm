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

    /// How many fine blocks to retrieve per query (PRD line 124). When
    /// `adaptiveTopK` is true (the default), this is the FLOOR; the
    /// actual top-K at gather time is `max(fineTopK, ceil(seqLen /
    /// adaptiveTopKDivisor))`. See `effectiveFineTopK(seqLen:)`.
    public var fineTopK: Int = 32

    /// Whether to scale `fineTopK` with sequence length at gather time.
    /// F-41 measured 14B-1M @ 32K-1: default-32 → 1/8 multi-step match
    /// vs adaptive-128 → 8/8 match. Adaptive is the SHIP default; the
    /// flag is here only to expose an opt-out for ablation.
    public var adaptiveTopK: Bool = true

    /// Divisor for adaptive top-K scaling. `topK = ceil(seqLen / divisor)`.
    /// F-40/F-41 validated 256 as the sweet spot on Qwen2.5-14B-1M:
    /// 32 floor at ≤8K, 128 at 32K (8/8 multi-step match), 1024 at 256K.
    public var adaptiveTopKDivisor: Int = 256

    /// Compute the effective fine top-K to use at gather time given the
    /// current cache length.
    public func effectiveFineTopK(seqLen: Int) -> Int {
        guard adaptiveTopK else { return fineTopK }
        let scaled = (seqLen + adaptiveTopKDivisor - 1) / adaptiveTopKDivisor
        return max(fineTopK, scaled)
    }

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

    /// True if the selector path needs to compute and store the trig half
    /// of the selector embedding. Pure content (λ=0, ship default) skips
    /// trig — saves 5-6 MLX ops per index update and halves the stored
    /// per-token feature size.
    public var usesTrigFeatures: Bool { lambdaPos > 0.0 }

    /// Effective selector dim given `usesTrigFeatures` — `contentDim` when
    /// trig is off, `selectorDim` otherwise. Used by callers that need to
    /// know the actual stored width.
    public var effectiveSelectorDim: Int {
        usesTrigFeatures ? selectorDim : contentDim
    }

    // ----- Scoring (Decisions 4, 19; PRD line 161)

    /// Content vs position blend. 0=pure content, 1=pure position.
    /// PRD v5 default 0.5; revised to 0.0 (pure content) per F-17 and
    /// F-18 — both synthetic and real Qwen3 K microbenches show λ=0
    /// matches or beats the mixture. Also a perf win: when 0.0, the
    /// scoring path skips the trig basis multiply entirely.
    public var lambdaPos: Float = 0.0

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

    /// F-59 + F-60 SHIP DEFAULT: build attention mask on GPU instead of
    /// gathering K/V. Trades a slightly larger dense SDPA matmul for
    /// eliminating the CPU dedupe + asArray sync + idx upload + take K/V
    /// chain. Bit-exact equivalent to the gather path (F-59 verified
    /// max abs diff 0.0, cosine 1.0). F-60 measured **4.7x faster** at
    /// 16K on Qwen3-0.6B-4bit (gather 126ms → mask 27ms/decode step).
    /// Set to `false` to opt out (e.g., for very long contexts where
    /// dense-SDPA-over-T compute would dominate gather overhead).
    public var useMaskedDense: Bool = true

    /// F-69 experimental: fused sparse SDPA Metal kernel. Correct (F-69
    /// max_abs_diff=3.6e-7 vs MLX gather+SDPA reference) but with the
    /// current adaptive top_k + union-across-KV-heads gather, the
    /// pre-dedupe budget saturates T at most contexts, so the sparse
    /// kernel's compute savings vanish and the simpler sequential loop
    /// loses to MLX's tiled SDPA. **Disabled by default**; opt-in for
    /// experimentation. Real win requires per-Q-head separate gathers
    /// and a parallelized kernel.
    public var useFusedSparseSDPA: Bool = false

    /// F-71 experimental: NSA-style group-centric fused sparse SDPA Metal
    /// kernel. One threadgroup per (B, KV head); all `groupSize` Q heads
    /// in the KV group share the same per-head gather and reuse K/V reads
    /// across the group — `groupSize`x bandwidth reduction at the KV
    /// memory layer plus K_padded/T compute reduction. Skips the fp32
    /// pre-cast that crippled F-69 at long context (12+ GB extra alloc
    /// per decode step on 14B-1M @ 32K). Off by default until benches
    /// confirm a win over the F-59 mask path.
    public var useGroupSparseSDPA: Bool = false

    /// F-70 experimental: per-KV-head separate gather + batched dense SDPA.
    /// Avoids the cross-KV-head union that saturates the gather toward T
    /// at long context. Each KV head gets its own sorted gather of
    /// static + sliding + top-K-fine + top-K-coarse positions; we reshape
    /// the heads dim into the batch dim and call MLXFast SDPA once on
    /// `[nKVH, groupSize, 1, D]` queries against `[nKVH, 1, K_padded, D]`
    /// gathered K/V — staying on the fused tiled SDPA fast path. Duplicate
    /// positions per row are masked via an adjacent-diff additive mask
    /// (sorted gather → dups are adjacent). Off by default until F-70
    /// benches confirm a measurable win over the F-59 mask path.
    public var usePerKVHeadGather: Bool = false

    /// F-79 SHIP DEFAULT: amortize the selector top-K computation
    /// across decode steps. Q drifts slowly between adjacent decode
    /// tokens, so the top-K picks at step T+1 are usually nearly
    /// identical to those at step T. Reuse cached topK for
    /// `selectorAmortization` consecutive steps; refresh on the
    /// boundary. Only the F-73 mask build runs every step (sliding
    /// window stays current).
    ///
    /// Default 8 — measured cosine 0.99990 vs amort=1 reference over
    /// 16 decode steps on 14B-1M-4bit at 24K. Latency 44.1ms vs dense
    /// 41.6ms at 32K (within 6%). Set to 1 to disable amortization.
    public var selectorAmortization: Int = 8

    /// F-78 — dispatch the entire selector pipeline (projectQ + F-48
    /// fine + F-48 coarse + F-73 mask) on a separate MLX Stream so
    /// it can execute concurrently with the model's default-stream
    /// work (prior layers' MLP, current layer's Q/K/V projections).
    /// SDPA still runs on the default stream and consumes the mask via
    /// cross-stream dependency tracking. Targets the ~14ms residual
    /// gap by hiding selector latency behind already-running compute.
    public var useSelectorStream: Bool = false

    /// F-77 — projectQ + parallel fine+coarse score+topK in ONE Metal
    /// kernel (F-74's matmul fold + F-75's parallel layout), followed
    /// by F-73 mask kernel. 2 kernel dispatches per layer + 1 cache
    /// update. Targets the ~1.2ms of MLX-matmul dispatch overhead.
    public var useParallelBundleSelector: Bool = false

    /// F-76 implicit-positions sparse SDPA — the no-mask terminal path.
    /// Per layer: F-75 parallel score+topK kernel + F-76 sparse SDPA
    /// kernel = 2 fused dispatches (vs F-73's 4). Skips mask
    /// materialization entirely (NSA/Quest/FlexAttention design).
    /// Aims at dense parity.
    public var useImplicitSparseSDPA: Bool = false

    /// F-75 parallel fine+coarse score+topK kernel. 2 × NKVH
    /// threadgroups in one launch (fine and coarse concurrent), then
    /// the F-73 mask kernel. Total 3 kernel dispatches per sparse layer
    /// vs F-73's 4. Targets the remaining 13ms gap after F-73's 9.8ms
    /// win.
    public var useParallelScoreTopK: Bool = false

    /// F-74 fused selector-bundle path: collapse projectQ + scoreTopK_fine
    /// + scoreTopK_coarse + buildMask into 2 Metal kernel calls per
    /// sparse layer (vs F-73's 4 calls, vs F-59's ~10 MLX ops). Targets
    /// the remaining ~13ms RA-over-dense gap after F-73 closed 9.8ms of
    /// the original 22.8ms gap.
    public var useFusedSelectorBundle: Bool = false

    /// F-73 SHIP DEFAULT: build the additive `[1, 1, 1, T]` attention
    /// mask via a single fused Metal kernel that consumes the topK
    /// block starts directly. Bit-exact to F-59 (cosine 1.0, diff 0.0).
    /// Measured -9.8ms / decode step on Qwen2.5-14B-1M-4bit @ 32K
    /// (F-59 = 62.4ms → F-73 = 52.6ms). Set to `false` to fall back to
    /// F-59's multi-op MLX path for replay / debugging.
    public var useFusedMaskBuild: Bool = true

    /// F-73 diagnostic: bypass the entire RetrievalAttention selector
    /// pipeline at decode steps and route to plain dense SDPA over the
    /// full K/V cache. The cache still updates K/V (and the selector
    /// index from prefill stays warm), but the per-step scoring +
    /// mask-build chain is skipped. Used to isolate how much of the
    /// RA-over-dense gap is selector overhead vs the cache update
    /// + dispatcher chain. NOT a ship config — produces dense attention
    /// outputs.
    public var bypassSelectorDecode: Bool = false

    /// F-63: minimum cache size before RA mask/gather even kicks in.
    /// Below this threshold the dispatcher falls through to standard
    /// dense SDPA — at small caches RA's selector + mask construction
    /// overhead outweighs the modest mask coverage savings (gather
    /// covers ~78% of cache at 8K with default preBudget=6272, leaving
    /// only ~20% to mask out). Empirically RA only starts paying off
    /// near 2-3x the preBudget. Default 16384 = ~2.6x preBudget.
    public var sparseMinContext: Int = 16384

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
    if seqLen == 0 { return [] }

    // F-58: bitmap-based dedupe. Was a Set<Int> with ~6000 inserts at 16K
    // context (F-57 measured 3.95ms/call). Bitmap mark-then-scan is much
    // tighter for dense token-space dedupe — measured ~10x faster.
    var marks = [Bool](repeating: false, count: seqLen)

    // Static initial window: positions 0 ..< min(staticInit, seqLen).
    let staticEnd = min(config.staticInit, seqLen)
    marks.withUnsafeMutableBufferPointer { buf in
        for p in 0..<staticEnd { buf[p] = true }

        // Sliding window: trailing `slidingWindow` tokens, floored at 0.
        let slidingStart = max(0, seqLen - config.slidingWindow)
        for p in slidingStart..<seqLen { buf[p] = true }

        // Fine retrieved blocks.
        for s in fineBlockStarts {
            precondition(s >= 0 && s < seqLen,
                "fine block start \(s) out of [0, \(seqLen))")
            let end = min(s + config.fineBlockSize, seqLen)
            for p in s..<end { buf[p] = true }
        }

        // Coarse rescue blocks.
        if config.coarseRescueEnabled {
            for s in coarseBlockStarts {
                precondition(s >= 0 && s < seqLen,
                    "coarse block start \(s) out of [0, \(seqLen))")
                let end = min(s + config.coarseBlockSize, seqLen)
                for p in s..<end { buf[p] = true }
            }
        }
    }

    // Compact bitmap into the sorted [Int] result.
    var result: [Int] = []
    result.reserveCapacity(
        config.staticInit + config.slidingWindow
            + config.fineTopK * config.fineBlockSize
            + (config.coarseRescueEnabled
                ? config.coarseTopK * config.coarseBlockSize : 0))
    marks.withUnsafeBufferPointer { buf in
        for p in 0..<seqLen {
            if buf[p] { result.append(p) }
        }
    }
    return result
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

// MARK: - Gather-not-mask execution path (PRD line 136-141)

/// Gather K/V rows at the deduplicated indices and dispatch dense SDPA
/// on the contiguous result. Stays on `MLXFast.scaledDotProductAttention`'s
/// fused Metal kernel path (no sparse mask).
///
/// - Parameters:
///   - queries: `[B, nHeads, L, D]` query tensor (current step or
///     prefill chunk).
///   - keys: `[B, nKVHeads, T, D]` full per-layer cached keys (post-RoPE,
///     [F-01]).
///   - values: `[B, nKVHeads, T, D]` full per-layer cached values.
///   - gatherIndices: sorted unique positions into the time axis (T),
///     produced by `retrievalAttentionGatherIndices`. Count is ≤ 6272
///     for the default config.
///   - scale: attention scale factor (typically `1/√D`).
///   - sinks: optional per-head sink logits ([nHeads]). Flows through.
/// - Returns: SDPA output `[B, nHeads, L, D]`.
public func retrievalAttentionGatherAndAttend(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    gatherIndices: [Int],
    scale: Float,
    sinks: MLXArray? = nil
) -> MLXArray {
    precondition(
        keys.shape.count == 4 && values.shape.count == 4,
        "keys/values must be [B, nKVHeads, T, D]"
    )
    let T = keys.dim(2)
    precondition(
        gatherIndices.last == nil || gatherIndices.last! < T,
        "gather index out of cache range"
    )

    // MLXArray of int32 indices; gather along the time axis (axis 2).
    let idxArray = MLXArray(gatherIndices.map { Int32($0) })

    // `take(_:axis:)` is MLX's gather. Time axis is index 2 for both
    // keys and values. Result shape: [B, nKVHeads, gatherIndices.count, D].
    let gatheredKeys = keys.take(idxArray, axis: 2)
    let gatheredValues = values.take(idxArray, axis: 2)

    // Standard dense SDPA on the contiguous tensor. No mask — the
    // deduplicated indices are the entire attention region.
    return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
        MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: gatheredKeys,
            values: gatheredValues,
            scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none,
            sinks: sinks
        )
    }
}

// MARK: - End-to-end forward (selector + dedupe + gather + SDPA)

/// One-shot RetrievalAttention forward pass for a single query (decode
/// step, L=1) on a single (layer, KV head).
///
/// Caller responsibilities:
///   - Maintain the `RetrievalAttentionIndex` across update calls (one
///     per (layer, KV head); see `RetrievalAttentionIndex.update(newK:)`).
///   - Pass `keys`/`values` as the FULL per-(layer, KV-head) cache
///     `[T, dHead]`. v1 doesn't yet batch over heads.
///   - Provide the query as `[dHead]` (single head dim, decode-step).
///
/// What this function does:
///   1. Project q via the selector basis.
///   2. Score fine + (optionally) coarse blocks against the projected q.
///   3. Get top-k fine + top-k coarse block STARTS (in token coords).
///   4. Build deduplicated gather indices using static + sliding +
///      retrieved + coarse-rescue.
///   5. Gather K/V at those indices.
///   6. Run dense SDPA on the contiguous gathered tensors.
///
/// - Parameters:
///   - q: `[dHead]` decode-step query vector for this head.
///   - keys: `[T, dHead]` cached post-RoPE keys for this (layer, KV head).
///   - values: `[T, dHead_v]` cached values.
///   - index: stateful per-(layer, KV head) selector index, already
///     populated via `update(newK:)` calls during prefill.
///   - scale: SDPA scale (typically `1/√dHead`).
///   - config: RA backend config.
/// - Returns: `[dHead_v]` attention output for this query.
public func retrievalAttentionForwardSingleHead(
    q: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    index: RetrievalAttentionIndex,
    scale: Float,
    config: RetrievalAttentionConfig = RetrievalAttentionConfig()
) -> MLXArray {
    precondition(
        keys.shape.count == 2 && values.shape.count == 2,
        "expected [T, D] keys/values for single-head path"
    )
    let T = keys.dim(0)
    precondition(
        T == index.seqLen,
        "cache length \(T) doesn't match selector index seqLen \(index.seqLen)"
    )

    // 1-3: selector
    let projQ = index.projectQuery(q)
    let fineStarts = index.topKFineBlockStarts(against: projQ)
    let coarseStarts = config.coarseRescueEnabled
        ? index.topKCoarseBlockStarts(against: projQ) : []

    // 4: dedupe
    let gatherIdx = retrievalAttentionGatherIndices(
        seqLen: T,
        fineBlockStarts: fineStarts,
        coarseBlockStarts: coarseStarts,
        config: config
    )

    // Reshape keys/values to [B=1, nHeads=1, T, D] for the SDPA API.
    let dK = keys.dim(1)
    let dV = values.dim(1)
    let keys4 = keys.reshaped(1, 1, T, dK)
    let values4 = values.reshaped(1, 1, T, dV)
    let q4 = q.reshaped(1, 1, 1, dK)

    let out4 = retrievalAttentionGatherAndAttend(
        queries: q4,
        keys: keys4,
        values: values4,
        gatherIndices: gatherIdx,
        scale: scale
    )
    // Strip the dummy axes → [dV]
    return out4.reshaped(dV)
}
