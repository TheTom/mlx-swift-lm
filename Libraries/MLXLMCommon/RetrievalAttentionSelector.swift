// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// RetrievalAttention selector — content + V3-trig mixture, block
// pooling, scoring, top-k. Spec 034 (Open Sparse Stack PRD v5,
// Decisions 4 / 7 / 14 / 19, A/B 1-7).
//
// Decoupled from the gather + SDPA path so the selector math can be
// unit-tested without an MLX dispatch. Mirrors the Python reference
// at `research/retrieval_attention/selector.py` (tested in
// test_selector.py).

import Foundation
import MLX

extension RetrievalAttentionConfig {

    /// Deterministic seed for the JL random projection (PRD line 159).
    /// Same matrix for all prefill / decode / chunk boundaries —
    /// critical for the dot-product preservation argument.
    public static var jlSeed: UInt32 { 0x4F50_4E53 }  // "OPNS"
}

// MARK: - JL random projection (Decision 4, PRD line 159)

/// Build the JL random-projection matrix used to compress d_head → contentDim.
///
/// Entries are i.i.d. `N(0, 1/contentDim)`. This scaling preserves
/// `E[||Wx||²] = ||x||²` per the Johnson-Lindenstrauss construction.
/// The same matrix is used for both Q and K so that
/// `(Wq) · (Wk) = q · (WᵀW) · k ≈ q · k` in expectation.
///
/// PRD line 156-159: total selector dim is 32 (contentDim + trigDim = 16+16).
///
/// - Parameters:
///   - dHead: head dim of the underlying model.
///   - config: backend config (for `contentDim` and the JL seed).
/// - Returns: `[contentDim, dHead]` fp32 array.
public func retrievalAttentionJLProjection(
    dHead: Int,
    config: RetrievalAttentionConfig = RetrievalAttentionConfig()
) -> MLXArray {
    // Use a private MLXRandom.RandomState seeded with the JL seed so we
    // don't perturb the caller's global PRNG. (Sampling with `key:` is
    // the idiomatic way to scope randomness in MLX-Swift.)
    let key = MLXRandom.key(UInt64(RetrievalAttentionConfig.jlSeed))
    let scale = Float(1.0 / Float(config.contentDim).squareRoot())
    let W = MLXRandom.normal([config.contentDim, dHead], key: key) * scale
    return W.asType(.float32)
}

// MARK: - V3-trig basis (Decision 4, PRD line 160)

/// 16-dim V3-trig features for an array of relative positions.
/// 8 frequencies × {sin, cos}, interleaved as
/// `[sin(f0·p), cos(f0·p), sin(f1·p), cos(f1·p), …]`.
///
/// The base must match the model's `rope_theta` (PRD line 167-168 /
/// PRD line 260 — "consume `rope_scaling` factors"). Llama-3 uses 500K,
/// Llama-3.1 uses 10K (RoPE-NTK extended), Qwen2.5 uses 1M. The
/// trig basis must align with whatever the model's RoPE actually uses
/// or the positional manifold won't line up.
///
/// - Parameters:
///   - relativePositions: `[N]` int32 array of relative positions.
///   - base: RoPE theta. Default 10_000.
///   - config: backend config (for `trigDim`).
/// - Returns: `[N, trigDim]` fp32 array.
public func retrievalAttentionTrigFeatures(
    relativePositions: MLXArray,
    base: Float = 10_000.0,
    config: RetrievalAttentionConfig = RetrievalAttentionConfig()
) -> MLXArray {
    precondition(
        config.trigDim % 2 == 0,
        "trigDim must be even (half sin, half cos)"
    )
    let half = config.trigDim / 2

    // freq_i = base^(-2i / trigDim) for i in [0..half).
    let exponents = MLXArray(0..<Int32(half)).asType(.float32)
        * (-2.0 / Float(config.trigDim))
    let freqs = exp(exponents * log(base))  // base^(...) via exp/log

    // angles[n, i] = relativePositions[n] * freqs[i]
    let pos = relativePositions.asType(.float32).reshaped(-1, 1)  // [N, 1]
    let f = freqs.reshaped(1, half)  // [1, half]
    let angles = pos * f  // [N, half]

    let sinPart = sin(angles)  // [N, half]
    let cosPart = cos(angles)  // [N, half]

    // Interleave: out[..., 2i] = sin, out[..., 2i+1] = cos.
    let stacked = stacked([sinPart, cosPart], axis: -1)  // [N, half, 2]
    return stacked.reshaped(-1, config.trigDim)  // [N, trigDim]
}

// MARK: - Block pooling (Decision adjacent, PRD line 164)

/// Mean-pool a `[T, ...]` array into blocks of `blockSize`. Last block
/// is mean over `≤ blockSize` tokens.
///
/// For selector index construction (PRD line 164): pool per-block over
/// the f(K) embeddings so scoring is cheap (n_blocks × selectorDim
/// instead of seqLen × selectorDim).
///
/// - Parameters:
///   - features: `[T, ...]` array.
///   - blockSize: PRD default 64.
/// - Returns: `[ceil(T / blockSize), ...]`.
public func retrievalAttentionBlockMeanPool(
    _ features: MLXArray,
    blockSize: Int
) -> MLXArray {
    let T = features.dim(0)
    precondition(T > 0, "empty feature array")
    let nBlocks = (T + blockSize - 1) / blockSize
    let paddedT = nBlocks * blockSize

    // Pad with zeros to the next multiple of blockSize so we can use a
    // single reshape + mean. The final block's mean is computed over
    // blockSize entries (some of which are pad zeros) — we correct by
    // multiplying by `blockSize / actual_count` at the boundary.
    //
    // Trick: pad with the per-feature mean of the tail so the final
    // block's mean is unchanged. Simpler: just pad with zeros and
    // rescale the final block.
    let padCount = paddedT - T
    let result: MLXArray
    if padCount == 0 {
        result = features
            .reshaped([nBlocks, blockSize] + features.shape.dropFirst())
            .mean(axis: 1)
    } else {
        // Build a [paddedT, ...] array by concatenating with zeros.
        var padShape = features.shape
        padShape[0] = padCount
        let pad = MLXArray.zeros(padShape, dtype: features.dtype)
        let padded = concatenated([features, pad], axis: 0)
        let pooled = padded
            .reshaped([nBlocks, blockSize] + features.shape.dropFirst())
            .mean(axis: 1)
        // Last block was averaged over blockSize but only the first
        // (T % blockSize) entries were real → scale up.
        let tail = T - (nBlocks - 1) * blockSize
        let correction = Float(blockSize) / Float(tail)
        // pooled[nBlocks-1] *= correction (broadcasting over feature dims)
        var corrected = pooled
        // Build a mask of [nBlocks, 1, 1, ...] that is `correction` at
        // the last block and `1.0` elsewhere.
        var maskShape = Array(repeating: 1, count: pooled.shape.count)
        maskShape[0] = nBlocks
        let maskArr = MLXArray.ones(maskShape, dtype: pooled.dtype)
        // Set the last row to `correction`.
        // Use scatter-like indexing — `maskArr[nBlocks-1] = correction`
        // is awkward in MLX-Swift. Easier: build [nBlocks] vector then
        // reshape.
        var oneNB = Array(repeating: Float(1.0), count: nBlocks)
        oneNB[nBlocks - 1] = correction
        let oneNBArr = MLXArray(oneNB).reshaped(maskShape)
        corrected = pooled * oneNBArr.asType(pooled.dtype)
        return corrected
    }
    return result
}

// MARK: - Scoring + top-k (PRD line 165-167)

/// Score blocks for one query against per-block pooled f(K).
///
/// `score = (1 − λ) · content_score + λ · trig_score`
///
/// where:
///   - `content_score[b] = blockFeatures[b, 0..<contentDim] · q[0..<contentDim]`
///   - `trig_score[b] = blockFeatures[b, contentDim..] · q[contentDim..]`
///
/// Recency bias (PRD line 166): `score -= alpha · log(distance + 1)`.
///
/// - Parameters:
///   - blockFeatures: `[nBlocks, selectorDim]` block-pooled f(K).
///   - q: `[selectorDim]` query projected through the same selector
///     basis as the keys.
///   - config: blends + recency from `config.lambdaPos` / `config.recencyAlpha`.
/// - Returns: `[nBlocks]` scalar scores.
public func retrievalAttentionScoreBlocks(
    blockFeatures: MLXArray,
    q: MLXArray,
    config: RetrievalAttentionConfig = RetrievalAttentionConfig()
) -> MLXArray {
    precondition(
        blockFeatures.dim(1) == config.selectorDim,
        "blockFeatures dim 1 (\(blockFeatures.dim(1))) != selectorDim (\(config.selectorDim))"
    )
    precondition(
        q.dim(0) == config.selectorDim,
        "q dim 0 (\(q.dim(0))) != selectorDim (\(config.selectorDim))"
    )

    let qContent = q[0..<config.contentDim]
    let qTrig = q[config.contentDim..<config.selectorDim]
    let bContent = blockFeatures[0..., 0..<config.contentDim]
    let bTrig = blockFeatures[0..., config.contentDim..<config.selectorDim]

    let contentScore = matmul(bContent, qContent)
    let trigScore = matmul(bTrig, qTrig)

    var score =
        (1.0 - config.lambdaPos) * contentScore
        + config.lambdaPos * trigScore

    if config.recencyAlpha != 0.0 {
        let nBlocks = blockFeatures.dim(0)
        let blockIdx = MLXArray(0..<Int32(nBlocks)).asType(.float32)
        let distance = Float(nBlocks - 1) - blockIdx
        let penalty = MLXArray(config.recencyAlpha) * log(distance + 1.0)
        score = score - penalty
    }

    return score
}

/// Indices of the top-k blocks (descending score).
///
/// Returns at most `min(k, nBlocks)` indices.
public func retrievalAttentionTopKBlocks(
    scores: MLXArray,
    k: Int
) -> [Int] {
    let nBlocks = scores.dim(0)
    let take = min(k, nBlocks)
    if take == 0 { return [] }

    // Pull scores into the CPU side for partial sort. Partial-sort over
    // 16K blocks is trivially fast on CPU; the alternative
    // (Metal-side argpartition) is a v2 optimization if profiling shows
    // CPU sort dominates decode latency.
    let arr = scores.asArray(Float.self)
    let indexed = arr.enumerated().map { ($0.offset, $0.element) }
    let sorted = indexed.sorted { $0.1 > $1.1 }
    return sorted.prefix(take).map { $0.0 }
}
