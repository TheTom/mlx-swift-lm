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
}
