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

    // v1: plain SDPA. Sparse decode path will land alongside this once
    // the F-73 / F-79 helpers are migrated off the wrapper. The dense
    // path here exists primarily to prove out the architecture's perf
    // parity with the bare StandardKVCache → ~78 ms/step at 128K vs the
    // wrapper's ~135 ms.
    return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: cachedKeys, values: cachedValues,
            scale: scale, mask: mask, sinks: sinks
        )
    }
}
