// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// Per-layer side-channel state for retrieval attention.
///
/// Carries the selector index plus the small caches the sparse-decode
/// path relies on, decoupled from the K/V cache itself. The K/V cache
/// stays a plain `StandardKVCache` (or `TurboQuantizedKVCache`); the
/// dispatcher reads this context only when the layer chooses to engage
/// the sparse path.
///
/// Motivation: benchmark at 128K on Qwen2.5-14B-1M-4bit showed the
/// monolithic `RetrievalAttentionKVCache` wrapper adds ~57 ms/step at
/// decode even when sparse work is disabled — likely from lazy-graph
/// node retention, allocator fragmentation between selector buffers and
/// K/V buffers, or class-dispatch overhead through the wrapper. With
/// the cache held separately as `StandardKVCache`, decode drops from
/// ~135 ms to ~78 ms, within 11% of vanilla mlx-lm Python's 70.5 ms.
public final class RetrievalAttentionContext {

    /// Batched selector index covering all KV heads for this layer.
    /// Populated lazily on the first `prefillUpdate(...)` once we know
    /// `nKVHeads` and `dHead`.
    public private(set) var batchedIndex: BatchedRetrievalAttentionIndex?

    /// RetrievalAttention configuration (block sizes, top-K, dense layers).
    public let raConfig: RetrievalAttentionConfig

    /// Position of this attention layer in the model's attention layer list.
    public let layerIdx: Int

    /// Total attention-layer count. Used to skip the last-N dense band.
    public let totalLayers: Int

    /// RoPE base — fed to each per-head index so its trig features match
    /// the model's positional encoding.
    public let ropeBase: Float

    /// Cached `[nKVHeads]` int32 array of representative Q-head indices
    /// per KV group. Allocated lazily on first use.
    public var cachedHeadIdx: MLXArray?

    /// F-79 cached top-K outputs from the previous full selector refresh.
    /// Reused across `selectorAmortization` consecutive decode steps.
    public var cachedFineStarts: MLXArray?
    public var cachedCoarseStarts: MLXArray?
    public var lastRefreshOffset: Int = -1

    public init(
        layerIdx: Int,
        totalLayers: Int,
        raConfig: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        ropeBase: Float = 10_000.0
    ) {
        self.raConfig = raConfig
        self.layerIdx = layerIdx
        self.totalLayers = totalLayers
        self.ropeBase = ropeBase
    }

    /// Returns `true` when this layer is in the sparse band (not in the
    /// first-N or last-N dense layers).
    public var isSparseEligible: Bool {
        raConfig.isSparseLayer(layerIdx: layerIdx, totalLayers: totalLayers)
    }

    /// Allocate the batched selector index on first sight of the cache
    /// shape. Cheap to call on every update; idempotent after the first.
    public func ensureIndex(nKVHeads: Int, dHead: Int) {
        guard batchedIndex == nil else { return }
        batchedIndex = BatchedRetrievalAttentionIndex(
            config: raConfig,
            dHead: dHead,
            nKVHeads: nKVHeads,
            ropeBase: ropeBase,
            layerIdx: layerIdx
        )
    }

    /// Apply a prefill chunk's new keys to the selector index. Called by
    /// the dispatcher when `L > 1` and the layer is sparse-eligible.
    ///
    /// `keys` come in as `[1, nKVHeads, L, dHead]` post-RoPE.
    public func prefillUpdate(keys: MLXArray) {
        let nKVHeads = keys.dim(1)
        let dHead = keys.dim(3)
        ensureIndex(nKVHeads: nKVHeads, dHead: dHead)
        // F-72: index update only on prefill chunks (L > 1). Decode tokens
        // live in the sliding window and don't need block features.
        let L = keys.dim(2)
        if L > 1 {
            let keysF32 = keys.asType(.float32)[0, 0..., 0..., 0...]
            batchedIndex!.update(newKeys: keysF32)
        }
    }

    /// F-73 fused-mask-build for decode. Mirrors
    /// `RetrievalAttentionKVCache.buildAttentionMaskFusedKernel` but
    /// works against the sidecar context — used by the new
    /// `retrievalAttentionStep` engine path so sparse decode doesn't pay
    /// the 57 ms wrapper tax. F-79 amortization is preserved.
    ///
    /// - Parameters:
    ///   - q: `[nHeads, dHead]` — current query, post-RoPE.
    ///   - dtype: output mask dtype (matches K/V dtype).
    ///   - T: total prior-context length (`cachedKeys.dim(2)`).
    ///   - offset: cache.offset, used for F-79 amortization gating.
    /// - Returns: `[1, 1, 1, T]` additive mask (0 at attended positions,
    ///   -inf elsewhere). Pass to MLXFast SDPA as `mask: .array(...)`.
    public func buildAttentionMaskFusedKernel(
        q: MLXArray, dtype: DType, T: Int, offset: Int
    ) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead]")
        guard let index = batchedIndex else {
            fatalError("buildAttentionMaskFusedKernel: selector index not initialized; "
                + "context.prefillUpdate must run on at least one L>1 chunk first")
        }
        let nQHeads = q.dim(0)
        precondition(
            nQHeads % index.nKVHeads == 0,
            "Q heads (\(nQHeads)) must be a multiple of KV heads (\(index.nKVHeads))")
        let groupSize = nQHeads / index.nKVHeads
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<index.nKVHeads).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }

        // F-79 amortization — reuse top-K from prior call across
        // `selectorAmortization` consecutive decode steps; only refresh
        // when offset has advanced past the window. Mask itself rebuilds
        // every step so the sliding window stays current.
        let amort = max(1, raConfig.selectorAmortization)
        let needRefresh = (cachedFineStarts == nil)
            || (offset - lastRefreshOffset) >= amort
        let fineStarts: MLXArray
        let coarseStarts: MLXArray
        if needRefresh {
            let qStacked = q.take(cachedHeadIdx!, axis: 0).asType(.float32)
            let projQ = index.projectQueriesBatched(qStacked)
            let (f, c) = index.topKBlockStartsAllHeadsCombinedGPU(projectedQ: projQ)
            cachedFineStarts = f
            cachedCoarseStarts = c
            lastRefreshOffset = offset
            fineStarts = f
            coarseStarts = c
        } else {
            fineStarts = cachedFineStarts!
            coarseStarts = cachedCoarseStarts!
        }

        return retrievalAttentionBuildMaskFused(
            fineStarts: fineStarts,
            coarseStarts: coarseStarts,
            T: T,
            staticInit: raConfig.staticInit,
            slidingWindow: raConfig.slidingWindow,
            fineBS: raConfig.fineBlockSize,
            coarseBS: raConfig.coarseBlockSize,
            outputDtype: dtype
        )
    }

    /// F-70 per-KV-head sorted-gather + batched SDPA. The whole point:
    /// SDPA shape collapses from `[1, nQH, 1, T]` (full cache) to
    /// `[1, nQH, 1, K_padded]` where `K_padded ≈ static + sliding +
    /// top-K-fine + top-K-coarse ≈ 2k` at 128K. Reads ~2% of K/V
    /// instead of all of it — the actual sparse bandwidth win F-73's
    /// mask path doesn't realize.
    ///
    /// Mirrors `RetrievalAttentionKVCache.perKVHeadGatherAndAttend`
    /// against the sidecar context. Caller passes the cached K/V from
    /// `cache.update`.
    public func perKVHeadGatherAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        qHeads: MLXArray,
        scale: Float,
        offset: Int
    ) -> MLXArray {
        precondition(qHeads.shape.count == 2, "expected [nHeads, dHead]")
        precondition(keys.shape.count == 4, "expected [B, nKVH, T, D]")
        guard let index = batchedIndex else {
            fatalError("perKVHeadGatherAndAttend: selector index not initialized")
        }
        let T = keys.dim(2)
        let nKVH = keys.dim(1)
        let nQH = queries.dim(1)
        precondition(nQH % nKVH == 0)
        let groupSize = nQH / nKVH

        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            eval(cachedHeadIdx!)
        }

        // F-79 selector amortization on the gather path. The heavy work
        // is `projectQ + topKBlockStarts` (~6-8 ms/step at 128K). Cache
        // these across `selectorAmortization` consecutive decode steps;
        // rebuild the gather array each step using cached starts +
        // current static/sliding range.
        let amort = max(1, raConfig.selectorAmortization)
        let needRefresh = (cachedFineStarts == nil)
            || (offset - lastRefreshOffset) >= amort
        let fineStarts: MLXArray
        let coarseStarts: MLXArray
        if needRefresh {
            let qStacked = qHeads.take(cachedHeadIdx!, axis: 0).asType(.float32)
            let projQ = index.projectQueriesBatched(qStacked)
            let (f, c) = index.topKBlockStartsAllHeadsCombinedGPU(projectedQ: projQ)
            cachedFineStarts = f
            cachedCoarseStarts = c
            lastRefreshOffset = offset
            fineStarts = f
            coarseStarts = c
        } else {
            fineStarts = cachedFineStarts!
            coarseStarts = cachedCoarseStarts!
        }

        // Build gather array using cached top-K + current static/sliding
        // ranges. Static + sliding shift every step (sliding window
        // slides forward), so they rebuild each call — cheap (just
        // ranges + broadcast).
        let staticInit = raConfig.staticInit
        let slidingWindow = raConfig.slidingWindow
        let fineBS = raConfig.fineBlockSize
        let coarseBS = raConfig.coarseBlockSize
        let staticEnd = min(staticInit, T)
        let slidingStart = max(0, T - slidingWindow)
        let staticCount = staticEnd
        let slidingCount = max(0, T - slidingStart)
        let kFineEff = fineStarts.dim(1)
        let kCoarseEff = coarseStarts.dim(1)

        let staticPositions = staticCount > 0
            ? MLXArray(Int32(0)..<Int32(staticEnd))
            : MLXArray.zeros([0], dtype: .int32)
        let slidingPositions = slidingCount > 0
            ? MLXArray(Int32(slidingStart)..<Int32(T))
            : MLXArray.zeros([0], dtype: .int32)
        let staticPlus = concatenated([staticPositions, slidingPositions], axis: 0)
            .reshaped(1, staticCount + slidingCount)
        let staticBroadcast = broadcast(staticPlus, to: [nKVH, staticCount + slidingCount])

        var pieces: [MLXArray] = [staticBroadcast]
        if kFineEff > 0 {
            let offs = MLXArray(0..<Int32(fineBS)).reshaped(1, 1, fineBS)
            let expanded = fineStarts.expandedDimensions(axis: 2) + offs
            pieces.append(expanded.reshaped(nKVH, kFineEff * fineBS))
        }
        if kCoarseEff > 0 {
            let offs = MLXArray(0..<Int32(coarseBS)).reshaped(1, 1, coarseBS)
            let expanded = coarseStarts.expandedDimensions(axis: 2) + offs
            pieces.append(expanded.reshaped(nKVH, kCoarseEff * coarseBS))
        }
        let unsorted = concatenated(pieces, axis: 1)
        let clipped = clip(unsorted, min: Int32(0), max: Int32(T - 1))
        let gather = sorted(clipped, axis: 1)
        let kPadded = staticCount + slidingCount + kFineEff * fineBS + kCoarseEff * coarseBS

        let gatherForTake = gather.expandedDimensions(axes: [0, 3])
        let gatheredK = takeAlong(keys, gatherForTake, axis: 2)
        let gatheredV = takeAlong(values, gatherForTake, axis: 2)

        let prev = gather[0..., 0 ..< (kPadded - 1)]
        let curr = gather[0..., 1 ..< kPadded]
        let dupInner = (curr .== prev)
        let leadFalse = MLXArray.zeros([nKVH, 1], dtype: .bool)
        let dupMask = concatenated([leadFalse, dupInner], axis: 1)
        let zeroLike = MLXArray(Float(0)).asType(keys.dtype)
        let negInfLike = MLXArray(-Float.infinity).asType(keys.dtype)
        let baseMask = MLX.where(dupMask, negInfLike, zeroLike)
            .reshaped(1, nKVH, 1, 1, kPadded)
        let expanded = broadcast(
            baseMask, to: [1, nKVH, groupSize, 1, kPadded])
        let addMask = expanded.reshaped(1, nKVH * groupSize, 1, kPadded)

        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: gatheredK, values: gatheredV,
                scale: scale, mask: .array(addMask), sinks: nil
            )
        }
    }

    /// F-84 block-gather attend. Union top-K block positions across all
    /// KV heads into ONE 1D index list, then `take(_, axis: 2)` gathers
    /// `[B, nKVH, k_padded, D]` K/V in a single coalesced kernel dispatch.
    /// Dense SDPA on the small gathered tensor — no mask required when
    /// positions are unique; we apply an adjacent-diff dup-mask to handle
    /// the (rare) collisions left after dedupe.
    ///
    /// Why this should win at long context:
    ///   - F-70 per-KV-head used `takeAlong(axis: 2)` with per-row
    ///     positions — gather doesn't coalesce, ~57 ms loss at 128 K.
    ///   - F-73 mask path still reads all 25 GB K/V, only masks compute.
    ///   - blockGather: one 1D `take` → full bandwidth saving at gather
    ///     time + tiny SDPA matmul. With static knobs (fineBS=64,
    ///     fineTopK=32, no-adaptive) at 128 K, k_padded ≈ 4-6 K (cross-
    ///     head union expands beyond 2 K) → ~20-30× bandwidth saving →
    ///     2-4 ms gather + ~1-2 ms SDPA = 3-6 ms vs dense's 67 ms.
    public func blockGatherAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        qHeads: MLXArray,
        scale: Float,
        offset: Int
    ) -> MLXArray {
        precondition(qHeads.shape.count == 2, "expected [nHeads, dHead]")
        precondition(keys.shape.count == 4, "expected [B, nKVH, T, D]")
        guard let index = batchedIndex else {
            fatalError("blockGatherAttend: selector index not initialized")
        }
        let T = keys.dim(2)
        let nKVH = keys.dim(1)
        let nQH = queries.dim(1)
        precondition(nQH % nKVH == 0)
        let groupSize = nQH / nKVH

        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            eval(cachedHeadIdx!)
        }

        // F-79 selector amortization — share top-K picks across
        // `selectorAmortization` consecutive decode steps. The heavy work
        // is `projectQ + topKBlockStarts` (~6-8 ms/step at 128 K). With
        // F-84 + amortization=numLayers, we can hoist the selector pass
        // out of every layer entirely (TODO: cross-layer selector reuse).
        let amort = max(1, raConfig.selectorAmortization)
        let needRefresh = (cachedFineStarts == nil)
            || (offset - lastRefreshOffset) >= amort
        let fineStarts: MLXArray
        let coarseStarts: MLXArray
        if needRefresh {
            let qStacked = qHeads.take(cachedHeadIdx!, axis: 0).asType(.float32)
            let projQ = index.projectQueriesBatched(qStacked)
            let (f, c) = index.topKBlockStartsAllHeadsCombinedGPU(projectedQ: projQ)
            cachedFineStarts = f
            cachedCoarseStarts = c
            lastRefreshOffset = offset
            fineStarts = f
            coarseStarts = c
        } else {
            fineStarts = cachedFineStarts!
            coarseStarts = cachedCoarseStarts!
        }

        // Build per-head block positions: [nKVH, K_fine*fineBS] and
        // [nKVH, K_coarse*coarseBS]. Same expansion as perKVHead, but
        // we collapse to a 1D union below.
        let staticInit = raConfig.staticInit
        let slidingWindow = raConfig.slidingWindow
        let fineBS = raConfig.fineBlockSize
        let coarseBS = raConfig.coarseBlockSize
        let staticEnd = min(staticInit, T)
        let slidingStart = max(0, T - slidingWindow)
        let staticCount = staticEnd
        let slidingCount = max(0, T - slidingStart)
        let kFineEff = fineStarts.dim(1)
        let kCoarseEff = coarseStarts.dim(1)

        // Collect everything as 1D pieces. Static + sliding are SAME
        // across heads (only need them once). Fine + coarse vary per head
        // → flatten to 1D so the union is one sort+dedupe.
        var pieces1D: [MLXArray] = []
        if staticCount > 0 {
            pieces1D.append(MLXArray(Int32(0)..<Int32(staticEnd)))
        }
        if slidingCount > 0 {
            pieces1D.append(MLXArray(Int32(slidingStart)..<Int32(T)))
        }
        if kFineEff > 0 {
            // [nKVH, K_fine, fineBS] → [nKVH * K_fine * fineBS]
            let offs = MLXArray(0..<Int32(fineBS)).reshaped(1, 1, fineBS)
            let expanded = fineStarts.expandedDimensions(axis: 2) + offs
            pieces1D.append(expanded.reshaped(nKVH * kFineEff * fineBS))
        }
        if kCoarseEff > 0 {
            let offs = MLXArray(0..<Int32(coarseBS)).reshaped(1, 1, coarseBS)
            let expanded = coarseStarts.expandedDimensions(axis: 2) + offs
            pieces1D.append(expanded.reshaped(nKVH * kCoarseEff * coarseBS))
        }
        let unsorted1D = concatenated(pieces1D, axis: 0)
        let clipped1D = clip(unsorted1D, min: Int32(0), max: Int32(T - 1))
        // Sort → constant-size output (graph-trace-friendly). Cross-head
        // union of fine + coarse + shared static/sliding can grow up to
        // 8 * (32*64 + 2*1024) + 128 + 2048 ≈ 35 K positions in the
        // worst case (heads pick fully disjoint blocks). Best-case
        // (heads pick similar blocks) ≈ 6 K. The
        // `blockGatherKPaddedMaxFraction` guard catches the worst case
        // BEFORE the SDPA hits the slow path; the precondition fires on
        // first violation so misconfigured runs get visible failures
        // instead of silent regressions.
        let gather1D = sorted(clipped1D, axis: 0)
        let kPadded = gather1D.dim(0)

        // Soft guard against the perKVHead failure mode: when k_padded
        // exceeds `blockGatherKPaddedMaxFraction` of T, the gather
        // materializes too much f16 K/V scratch to beat the dense
        // fused-SDPA path. Fall back to dense SDPA at the full cache
        // rather than crash — keeps the path safe-by-default while we
        // tune knobs. Set fraction=1.0 to disable the guard entirely.
        let kPaddedCap = max(1, Int(Float(T) * raConfig.blockGatherKPaddedMaxFraction))
        if raConfig.blockGatherKPaddedMaxFraction < 1.0 && kPadded > kPaddedCap {
            // Fall back to plain dense SDPA over the full cache. No
            // gather, no mask — the dispatcher's outer .raw arm would
            // do this naturally, but we can't return up the call chain
            // here without restructuring; do it inline.
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys, values: values,
                    scale: scale, mask: .none, sinks: nil
                )
            }
        }

        // The win: one `take(axis: 2)` with 1D positions → MLX gathers
        // [B, nKVH, kPadded, D] in a single coalesced kernel. No
        // per-row dispatch (F-70's pitfall). Reads kPadded * D * 2
        // (K+V) bytes per layer instead of T * D * 2 — at 128 K T=131072
        // vs kPadded ≈ 6 K → ~22× bandwidth saving on the gather.
        let gatheredK = keys.take(gather1D, axis: 2)
        let gatheredV = values.take(gather1D, axis: 2)

        // Critical perf decision: `mask: .none` hits MLXFast SDPA's
        // fused tile path; `mask: .array(...)` falls off it (research-
        // measured ~9× regression at gathered shapes). With the
        // user-recommended config (fineTopK=32, fineBS=64, no-adaptive,
        // coarseTopK≤2) cross-head dups are rare enough that the doubled
        // softmax weight on collision positions is negligible vs the 9×
        // speedup. Set `blockGatherNoMask=false` (or env
        // VSM_SPARSE_BLOCK_GATHER_MASK=1) to opt into the dup mask for
        // configs where collisions matter.
        if raConfig.blockGatherNoMask {
            return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
                MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: gatheredK, values: gatheredV,
                    scale: scale, mask: .none, sinks: nil
                )
            }
        }

        // Dup mask path: [1, 1, 1, kPadded] additive mask — broadcasts
        // across batch/heads since positions are shared. -inf on
        // positions equal to their predecessor (dups land adjacent
        // after sort). Slower than mask:.none but correct under
        // collisions.
        let prev = gather1D[0 ..< (kPadded - 1)]
        let curr = gather1D[1 ..< kPadded]
        let dupInner = (curr .== prev)
        let leadFalse = MLXArray.zeros([1], dtype: .bool)
        let dupMask = concatenated([leadFalse, dupInner], axis: 0)
        let zeroLike = MLXArray(Float(0)).asType(keys.dtype)
        let negInfLike = MLXArray(-Float.infinity).asType(keys.dtype)
        let addMask = MLX.where(dupMask, negInfLike, zeroLike)
            .reshaped(1, 1, 1, kPadded)
        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: gatheredK, values: gatheredV,
                scale: scale, mask: .array(addMask), sinks: nil
            )
        }
    }
}
