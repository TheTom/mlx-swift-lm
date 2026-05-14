// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// RetrievalAttentionKVCache — KV cache wrapper that maintains a per-KV-head
// selector index alongside raw K/V storage, for use by the dispatcher in
// `attentionWithCacheUpdate`. Spec 034 / PRD v5 Decision 5.
//
// Composition (not inheritance): an inner `StandardKVCache` holds the raw
// post-RoPE K/V; this wrapper holds an array of `RetrievalAttentionIndex`,
// one per KV head, that gets updated in lockstep on every `update(...)`.
//
// Dispatch contract: `storageKind == .retrievalSparse`. The dispatcher in
// `attentionWithCacheUpdate` looks at this kind, plus `isSparseLayer`, plus
// the L==1 decode-step check, to decide whether to route to the gather
// path. Prefill / multi-token / dense layers fall through to standard SDPA
// on the full K/V (which lets the index populate while keeping prefill
// correctness exact).
//
// v1 GQA aggregation: each KV head's index picks its own top-K block set;
// the dispatcher unions the gather positions across KV heads into a single
// dense SDPA. This is the simplest correct path — softmax still normalizes
// per-query, so attending over a per-head superset costs efficiency, not
// correctness. PRD Decision 19's perKVGroupMax mode is the natural next
// step (each head sees its own gather), and lands once we benchmark the
// union baseline.

import Foundation
import MLX

/// KV cache that backs `RetrievalAttention` block-sparse decode.
public final class RetrievalAttentionKVCache: BaseKVCache, CustomDebugStringConvertible {

    /// Inner raw K/V cache — same storage layout as `StandardKVCache`.
    public let inner: StandardKVCache

    /// Batched selector index covering all KV heads for this layer.
    /// Populated lazily on the first `update(...)` once we know
    /// `nKVHeads` and `dHead`. Replaces the per-head index array — F-43
    /// surfaced the per-head Swift loop as the dominant decode-step
    /// overhead.
    public private(set) var batchedIndex: BatchedRetrievalAttentionIndex?

    /// RetrievalAttention configuration (block sizes, top-K, dense layers).
    public let raConfig: RetrievalAttentionConfig

    /// Position of this attention layer in the model's attention layer list.
    /// Used together with `totalLayers` and `raConfig.denseFirstN/denseLastN`
    /// to decide whether sparse routing is allowed.
    public let layerIdx: Int

    /// Total attention-layer count. Used to skip the last-N dense band.
    public let totalLayers: Int

    /// RoPE base — fed to each per-head index so its V3-trig features match
    /// the model's positional encoding.
    public let ropeBase: Float

    /// Cached [nKVHeads] int32 array of representative Q-head indices per
    /// KV group (used for the strided gather of q in gatherIndicesForDecode).
    /// Allocated lazily on first use.
    private var cachedHeadIdx: MLXArray?

    /// Returns `true` when this layer is in the sparse band (not in the
    /// first-N or last-N dense layers).
    public var isSparseEligible: Bool {
        raConfig.isSparseLayer(layerIdx: layerIdx, totalLayers: totalLayers)
    }

    public override var storageKind: KVStorageKind { .retrievalSparse }

    public override var offset: Int {
        get { inner.offset }
        set { inner.offset = newValue }
    }

    public override var maxSize: Int? { inner.maxSize }

    public override var state: [MLXArray] {
        get { inner.innerState() }
        set {
            if !newValue.isEmpty {
                fatalError("state set on RetrievalAttentionKVCache (not supported)")
            }
        }
    }

    public override func innerState() -> [MLXArray] { inner.innerState() }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        inner.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    public override func peek() -> (MLXArray, MLXArray)? { inner.peek() }

    public init(
        layerIdx: Int,
        totalLayers: Int,
        raConfig: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        ropeBase: Float = 10_000.0,
        innerEviction: KVEviction = .unbounded,
        step: Int = 256
    ) {
        self.inner = StandardKVCache(eviction: innerEviction, step: step)
        self.raConfig = raConfig
        self.layerIdx = layerIdx
        self.totalLayers = totalLayers
        self.ropeBase = ropeBase
        super.init()
    }

    /// Allocate the batched selector index on first sight of the cache
    /// shape. Cheap to call on every update; idempotent after the first.
    private func ensureIndex(nKVHeads: Int, dHead: Int) {
        guard batchedIndex == nil else { return }
        batchedIndex = BatchedRetrievalAttentionIndex(
            config: raConfig,
            dHead: dHead,
            nKVHeads: nKVHeads,
            ropeBase: ropeBase,
            layerIdx: layerIdx
        )
    }

    /// Update inner storage AND (for sparse-eligible layers) the
    /// batched selector index.
    ///
    /// `keys` / `values` come in as `[B, nKVHeads, L, D]` (post-RoPE for K).
    /// v1 supports B == 1. Multi-B caches would need per-request indices.
    ///
    /// Dense-band layers (first-N / last-N) skip the index update entirely.
    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        precondition(keys.dim(0) == 1, "RA cache supports B=1 only in v1")
        let nKVHeads = keys.dim(1)
        let dHead = keys.dim(3)

        let (cachedK, cachedV) = inner.update(keys: keys, values: values)

        if !isSparseEligible {
            return (cachedK, cachedV)
        }

        ensureIndex(nKVHeads: nKVHeads, dHead: dHead)

        // [1, nKVHeads, L, D] → [nKVHeads, L, D]
        let keysF32 = keys.asType(.float32)[0, 0..., 0..., 0...]
        batchedIndex!.update(newKeys: keysF32)
        return (cachedK, cachedV)
    }

    /// Compute the union of per-head fine + coarse top-K block start positions.
    ///
    /// - Parameter q: `[nHeads, dHead]` — current query, post-RoPE, for the
    ///   decode step.
    /// - Returns: deduplicated, sorted gather indices (static + sliding +
    ///   fine top-K + coarse top-K) ready for `keys.take(_, axis: 2)`.
    public func gatherIndicesForDecode(q: MLXArray) -> [Int] {
        precondition(q.shape.count == 2, "expected [nHeads, dHead], got \(q.shape)")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        let nQHeads = q.dim(0)
        precondition(
            nQHeads % index.nKVHeads == 0,
            "Q heads (\(nQHeads)) must be a multiple of KV heads (\(index.nKVHeads))"
        )
        let groupSize = nQHeads / index.nKVHeads

        // Pick the representative Q head per KV group (head index = h * groupSize).
        // Single strided gather → [nKVHeads, dHead]. headIdx is cached on
        // the cache instance since it never changes for a given model.
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<index.nKVHeads).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = q.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)

        // Fine + coarse topK in a single GPU op chain with ONE asArray.
        // Halves the per-sparse-layer sync count vs the prior pattern of
        // calling topKFineBlockStartsAllHeads + topKCoarseBlockStartsAllHeads
        // (which did two separate asArrays).
        let (fineStartsPerHead, coarseStartsPerHead) =
            index.topKBlockStartsAllHeadsCombined(projectedQ: projQ)

        var allFine = Set<Int>()
        var allCoarse = Set<Int>()
        for h in 0..<index.nKVHeads {
            for s in fineStartsPerHead[h] { allFine.insert(s) }
            for s in coarseStartsPerHead[h] { allCoarse.insert(s) }
        }
        return retrievalAttentionGatherIndices(
            seqLen: self.offset,
            fineBlockStarts: Array(allFine),
            coarseBlockStarts: Array(allCoarse),
            config: raConfig
        )
    }

    /// F-59 mask-not-gather path: build a [1, 1, 1, T] additive attention
    /// mask on GPU with 0 at gathered positions and -inf elsewhere.
    /// Scatter is idempotent for setting to 0, so duplicate positions
    /// across heads collapse naturally — no CPU dedupe needed.
    public func buildAttentionMaskGPU(
        q: MLXArray, dtype: DType, T: Int
    ) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead], got \(q.shape)")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        let nQHeads = q.dim(0)
        precondition(
            nQHeads % index.nKVHeads == 0,
            "Q heads (\(nQHeads)) must be a multiple of KV heads (\(index.nKVHeads))"
        )
        let groupSize = nQHeads / index.nKVHeads
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<index.nKVHeads).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = q.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)

        // GPU-side expanded top-K positions (may overlap with static/sliding
        // and across heads; scatter dedupes for us).
        let topKPositions = index.expandedTopKPositionsGPU(projectedQ: projQ)

        // Static + sliding ranges.
        let staticEnd = min(raConfig.staticInit, T)
        let staticPositions = staticEnd > 0
            ? MLXArray(Int32(0) ..< Int32(staticEnd))
            : MLXArray.zeros([0], dtype: .int32)
        let slidingStart = max(0, T - raConfig.slidingWindow)
        let slidingPositions = slidingStart < T
            ? MLXArray(Int32(slidingStart) ..< Int32(T))
            : MLXArray.zeros([0], dtype: .int32)

        let allPositions = concatenated(
            [staticPositions, slidingPositions, topKPositions], axis: 0
        )
        // Clip to [0, T) — defensive against any small overrun from block
        // expansion at sequence tail.
        let clipped = clip(allPositions, min: Int32(0), max: Int32(T - 1))

        // Build mask [1, 1, 1, T] init to -inf, scatter 0 at gather positions.
        let negInf: Float = -.infinity
        let fillValue = MLXArray(negInf)
        let mask = MLXArray.full([1, 1, 1, T], values: fillValue).asType(dtype)
        mask[0, 0, 0, clipped] = MLXArray(Float(0)).asType(dtype)
        return mask
    }

    public var debugDescription: String {
        "RetrievalAttentionKVCache(layer=\(layerIdx)/\(totalLayers), "
            + "offset=\(offset), heads=\(batchedIndex?.nKVHeads ?? 0), "
            + "sparseEligible=\(isSparseEligible))"
    }
}
