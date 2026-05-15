import Foundation
import MLX
import MLXNN

/// F-80 cliff investigation: deterministic matmul-based SDPA fallback.
/// Set `MLX_DETERMINISTIC_SDPA=1` to route ALL attention calls through
/// explicit matmul + softmax + matmul instead of `MLXFast.scaledDotProductAttention`.
/// Used to isolate whether the long-context decode cliff is in MLX's
/// SDPA kernels or elsewhere in the pipeline.
nonisolated(unsafe) private let _deterministicSDPAEnabled: Bool = {
    if let env = ProcessInfo.processInfo.environment["MLX_DETERMINISTIC_SDPA"] {
        return env == "1" || env.lowercased() == "true"
    }
    return false
}()

/// Hand-rolled matmul-based attention. Bypasses `MLXFast.scaledDotProductAttention`.
/// Expected to be bit-deterministic across launches if matmul + softmax are themselves
/// deterministic (Tom's earlier quant-matmul cold test confirms matmul is deterministic).
/// - q: [B, nHeads, L, D]
/// - k: [B, nKVHeads, S, D]
/// - v: [B, nKVHeads, S, D]
/// - mask: optional additive mask, broadcastable to [B, nHeads, L, S]
/// Returns: [B, nHeads, L, D]
private func matmulSDPA(
    queries q: MLXArray,
    keys k: MLXArray,
    values v: MLXArray,
    scale: Float,
    mask: MLXArray?
) -> MLXArray {
    let B = q.dim(0)
    let nHeads = q.dim(1)
    let L = q.dim(2)
    let D = q.dim(3)
    let nKVHeads = k.dim(1)
    let S = k.dim(2)
    let vD = v.dim(3)

    // GQA-aware matmul without materializing nHeads-broadcast K/V.
    // Trick: reshape q to [B, nKVHeads, factor*L, D], matmul with k.T
    // (no broadcasting needed), reshape result back.
    let factor = nHeads / nKVHeads
    let qR = q.reshaped([B, nKVHeads, factor * L, D])
    let kT = k.swappedAxes(-2, -1)  // [B, nKVHeads, D, S]
    var scoresR = MLX.matmul(qR, kT) * scale  // [B, nKVHeads, factor*L, S]
    if let mask = mask {
        // Mask shape varies. For [B, nHeads, L, S] mask, reshape to
        // [B, nKVHeads, factor*L, S]. For [B, 1, L, S] or [1, 1, L, S],
        // broadcast handles it.
        if mask.dim(1) == nHeads {
            scoresR = scoresR + mask.reshaped([B, nKVHeads, factor * L, S])
        } else {
            scoresR = scoresR + mask
        }
    }
    let probsR = MLX.softmax(scoresR, axis: -1, precise: true)
    let outR = MLX.matmul(probsR, v)  // [B, nKVHeads, factor*L, vD]
    return outR.reshaped([B, nHeads, L, vD])
}

/// Convert an SDPA mask mode to an additive mask MLXArray, or nil for `.none`.
/// - q: query tensor for shape context
/// - S: total key sequence length
private func sdpaMaskToAdditive(
    _ mode: MLXFast.ScaledDotProductAttentionMaskMode,
    qShape: (Int, Int, Int, Int),
    S: Int,
    dtype: DType
) -> MLXArray? {
    switch mode {
    case .none:
        return nil
    case .causal:
        // [B, nHeads, L, S] additive mask — -inf above diagonal aligned to S.
        let (_, _, L, _) = qShape
        let qIdx = MLXArray(0..<Int32(L)).reshaped([L, 1])
        let kIdx = MLXArray(0..<Int32(S)).reshaped([1, S])
        let offset = S - L
        let causal = (kIdx .> (qIdx + Int32(offset)))
        let inf = MLXArray(Float(-1e30)).asType(dtype)
        let zero = MLXArray(Float(0)).asType(dtype)
        return MLX.`where`(causal, inf, zero)
    case .array(let m):
        return m
    case .arrays(let arr):
        return arr.first
    @unknown default:
        return nil
    }
}

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update.
///
/// Routes to the right backend based on the cache type:
/// - `TurboQuantizedKVCache` (default A path): raw-FP16 cache + standard
///   `MLXFast.scaledDotProductAttention(... sinks:)`. The TurboQuant rotation
///   is bypassed at decode (SDPA is invariant to a fixed orthogonal Π applied
///   to both Q and K), so `prepareQueries`/`inverseRotateOutput` are no-ops
///   and `updateAndDequant` keeps appending to the raw prefill buffer.
/// - `TurboQuantizedKVCache` with `useCompressedAttention = true` (B opt-in): runs
///   `compressedAttention` directly on the packed buffer (sinks unsupported)
/// - `AffineQuantizedKVCache`: affine quantized SDPA (sinks unsupported)
/// - any other cache: standard `MLXFast.scaledDotProductAttention(... sinks:)`
///
/// `sinks` defaults to `nil`; non-sinks-using models can omit it.
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
///   - sinks: Optional per-head attention-sink logits ([nHeads]) — flows through
///     SDPA. fatalErrors if combined with a cache type that doesn't support sinks
///     (affine quantized, B compressed TurboQuant).
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
    raContext: RetrievalAttentionContext? = nil
) -> MLXArray {
    // Sidecar retrieval-attention path. When a context is supplied AND the
    // cache is a plain StandardKVCache (case `.raw`), the dispatcher hands
    // off to `retrievalAttentionStep` which carries selector state through
    // the context instead of forcing all callers onto a wrapper KV cache.
    // Path is exercised by F-83 benches; default-nil keeps every other
    // caller on the existing dispatch.
    if let ctx = raContext, let std = cache as? StandardKVCache {
        return retrievalAttentionStep(
            queries: queries, keys: keys, values: values,
            cache: std, ctx: ctx, scale: scale, mask: mask, sinks: sinks
        )
    }
    guard let cache else {
        // Cache-less path (rare). Wrap the SDPA call so it shows up in traces
        // alongside cache-backed paths for comparability.
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        }
    }
    // Dispatch on `storageKind` rather than `as?` downcasts (spec 006). The
    // typed enum is extensible (new storage kinds don't touch this switch's
    // consumers) and self-documenting. The downcast inside each arm is a
    // class-identity assertion: storageKind is defined to mirror the concrete
    // class, so a mismatch indicates a programming error, not a runtime case.
    switch cache.storageKind {
    case .turboCompressed:
        guard let turboCache = cache as? TurboQuantizedKVCache else {
            fatalError(
                "storageKind .turboCompressed but cache is not TurboQuantizedKVCache: \(type(of: cache))"
            )
        }
        let L = queries.dim(2)
        if L > 1 {
            // Prefill (L>1): raw update + standard SDPA. Zero overhead.
            let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
            let (cachedKeys, cachedValues) = turboCache.update(keys: keys, values: values)
            BenchmarkSignpost.end(updH)
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: mask, sinks: sinks
                )
            }
        }
        // B (default): compressed-domain dequant + matrix-engine SDPA.
        // `compressedAttention` emits its own tq_encode/tq_score/tq_value/tq_rotate
        // sub-phase signposts internally — don't double-wrap here.
        // Sinks-using models (GPT-OSS family) auto-fallback to A — the
        // compressed-attention pass2 kernel doesn't yet incorporate the
        // sink-token logits in its online softmax (tracked in PR #99).
        if turboCache.useCompressedAttention && sinks == nil {
            return turboCache.compressedAttention(
                queries: queries, keys: keys, values: values,
                scale: scale, mask: mask
            )
        }
        // A path: raw-FP16 cache + standard SDPA(... sinks:). Used when the
        // user opts out via `TURBO_COMPRESSED_ATTENTION=0` /
        // `useCompressedAttention=false`, or when the model uses attention
        // sinks.
        // updateAndDequant returns raw K/V; prepareQueries/inverseRotateOutput
        // are no-ops in A — SDPA is invariant to the codec's orthogonal rotation
        // applied to both Q and K, so we skip the rotation entirely.
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (rotKeys, rotValues) = turboCache.updateAndDequant(keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        let rotQueries = turboCache.prepareQueries(queries)
        let rotOutput = BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: rotQueries, keys: rotKeys, values: rotValues,
                scale: scale, mask: mask, sinks: sinks
            )
        }
        return turboCache.inverseRotateOutput(rotOutput)

    case .affineQuantized:
        guard let quantizedKVCache = cache as? AffineQuantizedKVCache else {
            fatalError(
                "storageKind .affineQuantized but cache is not AffineQuantizedKVCache: \(type(of: cache))"
            )
        }
        if sinks != nil {
            fatalError("Affine quantized attention does not support non-zero sinks.")
        }
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.qsdpa) {
            quantizedScaledDotProductAttention(
                queries: queries,
                quantizedKeys: quantizedKeys,
                quantizedValues: quantizedValues,
                scale: scale,
                mask: mask,
                groupSize: quantizedKVCache.groupSize,
                bits: quantizedKVCache.bits,
                mode: quantizedKVCache.mode
            )
        }

    case .retrievalSparse:
        guard let raCache = cache as? RetrievalAttentionKVCache else {
            fatalError(
                "storageKind .retrievalSparse but cache is not RetrievalAttentionKVCache: \(type(of: cache))"
            )
        }
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (cachedKeys, cachedValues) = raCache.update(keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        let L = queries.dim(2)
        // Gather path: decode-step (L==1), sparse-eligible layer, and the
        // cache has enough tokens that a gather is materially smaller than
        // the full K/V. Below the budget threshold, sparse and dense agree
        // exactly (the union of static + sliding == [0, T)), so fall through
        // to dense and skip the selector overhead.
        let preBudget = retrievalAttentionPreDedupeBudget(config: raCache.raConfig)
        let threshold = max(preBudget, raCache.raConfig.sparseMinContext)
        let canGather = L == 1 && raCache.isSparseEligible && cachedKeys.dim(2) > threshold

        // F-83 — sparse prefill (chunked attention). Engages when L > 1
        // and the prior cache portion is long enough that gather pays
        // off vs dense O(L * T) compute. Sinks-using models would need
        // an online merge with the sink token logits — skip for V1.
        let priorLen = cachedKeys.dim(2) - L
        if L > 1 && raCache.isSparseEligible
            && raCache.raConfig.sparsePrefillEnabled
            && priorLen > raCache.raConfig.sparsePrefillMinContext
            && sinks == nil
        {
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                raCache.prefillSparseAttend(
                    queries: queries,
                    cachedKeys: cachedKeys,
                    cachedValues: cachedValues,
                    scale: scale
                )
            }
        }
        // F-73 diagnostic: bypass selector pipeline entirely at decode.
        // Falls through to the dense `MLXFast.scaledDotProductAttention`
        // below. Lets us isolate selector vs cache-update overhead.
        if canGather && raCache.raConfig.bypassSelectorDecode {
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: mask, sinks: sinks
                )
            }
        }
        if canGather {
            let qFlat = queries[0, 0..., 0, 0...]
            let D = cachedKeys.dim(3)
            let supportsFusedSparse = [32, 64, 96, 128, 256].contains(D)
            if raCache.raConfig.useGroupSparseSDPA && supportsFusedSparse && sinks == nil {
                // F-71 NSA-style group-centric fused kernel. K/V read once
                // per KV group, shared across all groupSize Q heads.
                return raCache.groupSparseSDPA(
                    queries: queries,
                    keys: cachedKeys,
                    values: cachedValues,
                    qHeads: qFlat,
                    scale: scale
                )
            }
            if raCache.raConfig.usePerKVHeadGather && sinks == nil {
                // F-70 per-KV-head batched SDPA. Skips cross-head union
                // so each head's gather is bounded by preBudget (not T).
                return raCache.perKVHeadGatherAndAttend(
                    queries: queries,
                    keys: cachedKeys,
                    values: cachedValues,
                    qHeads: qFlat,
                    scale: scale
                )
            }
            if raCache.raConfig.useFusedSparseSDPA && supportsFusedSparse && sinks == nil {
                // F-69 fused sparse SDPA: actually skips attention compute
                // over masked positions. Wins vs mask path at long context.
                let gatherMLX = raCache.gatherIndicesGPU(q: qFlat)
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    retrievalAttentionFusedSparseSDPA(
                        queries: queries,
                        keys: cachedKeys,
                        values: cachedValues,
                        gatherIndices: gatherMLX,
                        scale: scale
                    )
                }
            }
            if raCache.raConfig.useSelectorStream {
                // F-78 — dispatch the F-73 selector pipeline on a
                // separate MLX stream so it overlaps with the model's
                // default-stream work. MLX tracks the cross-stream
                // dependency: the SDPA call below waits on the mask.
                let T = cachedKeys.dim(2)
                let raMask = MLX.Stream.withStream(
                    retrievalAttentionSelectorStream()
                ) {
                    raCache.buildAttentionMaskFusedKernel(
                        q: qFlat, dtype: cachedKeys.dtype, T: T)
                }
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: cachedKeys, values: cachedValues,
                        scale: scale, mask: .array(raMask), sinks: sinks
                    )
                }
            }
            if raCache.raConfig.useParallelBundleSelector {
                // F-77 — projectQ folded into parallel fine+coarse score
                // kernel, then F-73 mask. 2 dispatches per sparse layer.
                let T = cachedKeys.dim(2)
                let raMask = raCache.buildAttentionMaskFusedParallelBundleKernel(
                    q: qFlat, dtype: cachedKeys.dtype, T: T
                )
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: cachedKeys, values: cachedValues,
                        scale: scale, mask: .array(raMask), sinks: sinks
                    )
                }
            }
            if raCache.raConfig.useImplicitSparseSDPA && sinks == nil {
                // F-76 — sparse SDPA with implicit position expansion.
                // 2 kernel dispatches per layer (F-75 selector + F-76 SDPA),
                // no mask, no gather array.
                if let out = raCache.implicitSparseSDPA(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    qHeads: qFlat, scale: scale
                ) {
                    return out
                }
            }
            if raCache.raConfig.useParallelScoreTopK {
                // F-75 — parallel fine+coarse score+topK kernel + F-73
                // mask kernel. 3 kernel dispatches per sparse layer.
                let T = cachedKeys.dim(2)
                let raMask = raCache.buildAttentionMaskFusedParallelKernel(
                    q: qFlat, dtype: cachedKeys.dtype, T: T
                )
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: cachedKeys, values: cachedValues,
                        scale: scale, mask: .array(raMask), sinks: sinks
                    )
                }
            }
            if raCache.raConfig.useFusedSelectorBundle {
                // F-74 — fused selector bundle (projectQ + scoreTopK_fine
                // + scoreTopK_coarse) followed by F-73 mask build.
                // 2 Metal kernels per layer instead of F-73's 5 MLX ops.
                let T = cachedKeys.dim(2)
                let raMask = raCache.buildAttentionMaskFusedBundleKernel(
                    q: qFlat, dtype: cachedKeys.dtype, T: T
                )
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: cachedKeys, values: cachedValues,
                        scale: scale, mask: .array(raMask), sinks: sinks
                    )
                }
            }
            if raCache.raConfig.useFusedMaskBuild {
                // F-73 — single Metal kernel writes the mask in one
                // launch (vs F-59's ~6 MLX ops). Selector pipeline is
                // the entire 22.9ms / decode-step gap to dense.
                let T = cachedKeys.dim(2)
                let raMask = raCache.buildAttentionMaskFusedKernel(
                    q: qFlat, dtype: cachedKeys.dtype, T: T
                )
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: cachedKeys, values: cachedValues,
                        scale: scale, mask: .array(raMask), sinks: sinks
                    )
                }
            }
            if raCache.raConfig.useMaskedDense {
                // F-59 mask-not-gather path. Build [1, 1, 1, T] attention
                // mask on GPU (0 at gather positions, -inf elsewhere) and
                // run dense SDPA. Skips CPU dedupe + asArray sync entirely.
                let T = cachedKeys.dim(2)
                let raMask = raCache.buildAttentionMaskGPU(
                    q: qFlat, dtype: cachedKeys.dtype, T: T
                )
                return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                    MLXFast.scaledDotProductAttention(
                        queries: queries, keys: cachedKeys, values: cachedValues,
                        scale: scale, mask: .array(raMask), sinks: sinks
                    )
                }
            }
            let gather = raCache.gatherIndicesForDecode(q: qFlat)
            return retrievalAttentionGatherAndAttend(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                gatherIndices: gather,
                scale: scale,
                sinks: sinks
            )
        }
        // Dense fallback — prefill / first-N or last-N dense layers /
        // pre-budget contexts. Index still got updated above so the
        // selector is warm by the time we transition into the gather band.
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: mask, sinks: sinks
            )
        }

    case .raw, .ssm, .composite:
        // Standard path — raw FP16/BF16 K/V (StandardKVCache), SSM caches
        // (SSMStateCache — not actually K/V but routed through the same
        // default-update path; layer code typically doesn't call
        // attentionWithCacheUpdate for SSM layers anyway), and composite
        // (CacheList — same reasoning).
        let updH = BenchmarkSignpost.begin(BenchmarkSignpost.PhaseLabel.kvUpdate)
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        BenchmarkSignpost.end(updH)
        if _deterministicSDPAEnabled && queries.dim(2) == 1 {
            // F-80 cliff investigation. Replace MLXFast.SDPA with matmul+softmax+matmul
            // only at decode (L=1). Prefill uses steel attention which is deterministic.
            // sinks unsupported on this path (no Qwen2 uses sinks).
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                let qS = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
                let addMask = sdpaMaskToAdditive(
                    mask, qShape: qS, S: cachedKeys.dim(2), dtype: queries.dtype
                )
                return matmulSDPA(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: addMask
                )
            }
        }
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: cachedKeys,
                values: cachedValues,
                scale: scale,
                mask: mask,
                sinks: sinks
            )
        }
    }
}
