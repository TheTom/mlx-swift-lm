// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Sidecar engine for retrieval attention.
///
/// Drives the same set of sparse prefill / decode paths that the
/// monolithic `RetrievalAttentionKVCache` exposes, but takes a plain
/// `StandardKVCache` plus a `RetrievalAttentionContext` instead of
/// wrapping the cache. The K/V cache is unchanged — model and dispatcher
/// see a vanilla `.raw` cache, so the case `.raw` arm runs as if RA
/// weren't there. Selector state lives in `ctx` and the sparse paths
/// fetch it on demand.
///
/// v1 — plain dense SDPA path only (used to validate the architecture
/// against `mlx-lm` Python at 128K). Sparse decode + sparse prefill move
/// in next as the existing wrapper helpers are migrated.
public func retrievalAttentionStep(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: StandardKVCache,
    ctx: RetrievalAttentionContext,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    BenchmarkSignpost.end(updH)

    // Prefill chunk side-channel: populate the selector index from
    // post-RoPE K for sparse-eligible layers. Decode (L==1) skips —
    // matches the F-72 behavior on the wrapper.
    if ctx.isSparseEligible {
        ctx.prefillUpdate(keys: keys)
    }

    // Decode-step (L == 1) sparse routing through F-73 fused-mask path:
    // run the selector against the sidecar context's selector index,
    // build a [1,1,1,T] additive mask on GPU, then call MLXFast SDPA
    // with that mask. Skips the 57 ms `RetrievalAttentionKVCache`
    // wrapper tax that V14/V16 measured. Falls through to plain SDPA
    // when: layer is dense-band, cache hasn't grown past the threshold,
    // sinks are required (not yet wired into the masked path), or the
    // selector index isn't populated yet (no L>1 chunks have run).
    let L = queries.dim(2)
    let T = cachedKeys.dim(2)
    let preBudget = retrievalAttentionPreDedupeBudget(config: ctx.raConfig)
    let threshold = max(preBudget, ctx.raConfig.sparseMinContext)
    let canGather = L == 1 && ctx.isSparseEligible && T > threshold
    if canGather && sinks == nil
        && !ctx.raConfig.bypassSelectorDecode
        && ctx.batchedIndex != nil
    {
        let qFlat = queries[0, 0..., 0, 0...]
        // F-84 block-gather: cross-KV-head UNION of top-K positions →
        // single 1D `take(axis: 2)` gathers [B, nKVH, k_padded, D] in
        // one coalesced kernel. Avoids F-70's per-row `takeAlong`
        // pitfall (no coalesce → 57 ms loss at 128 K). Targets long-ctx
        // bandwidth ceiling: dense reads 25 GB K/V at 128 K, blockGather
        // reads ~1.2 GB → 20× faster gather + tiny SDPA matmul.
        if ctx.raConfig.useBlockGather {
            return ctx.blockGatherAttend(
                queries: queries, keys: cachedKeys, values: cachedValues,
                qHeads: qFlat, scale: scale, offset: cache.offset)
        }
        // F-70 per-KV-head gather is the path that ACTUALLY reduces
        // K/V bandwidth: SDPA shape collapses from [1,nQH,1,T] (~131K
        // K positions @128K) to [1,nQH,1,K_padded] (~2k positions).
        // F-73 mask path only saves compute via -inf skip but still
        // loads all K/V.
        if ctx.raConfig.usePerKVHeadGather {
            return ctx.perKVHeadGatherAndAttend(
                queries: queries, keys: cachedKeys, values: cachedValues,
                qHeads: qFlat, scale: scale, offset: cache.offset)
        }
        if ctx.raConfig.useFusedMaskBuild {
            let raMask = ctx.buildAttentionMaskFusedKernel(
                q: qFlat, dtype: cachedKeys.dtype, T: T, offset: cache.offset)
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: .array(raMask), sinks: sinks
                )
            }
        }
    }

    // Default: plain dense SDPA. Used for dense-band layers, short
    // contexts, sinks-using models, or when sparse path opts itself
    // out via `bypassSelectorDecode`.
    return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask, sinks: sinks
        )
    }
}
