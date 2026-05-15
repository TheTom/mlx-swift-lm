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

/// F-83 V1.3 — cross-layer selector reuse (IndexCache pattern,
/// arxiv 2603.12201). Within a single prefill chunk, the first
/// sparse layer in each `sparsePrefillGroupSize`-sized group runs
/// the selector and stores its gather-position list here; the rest
/// of the group reads it instead of re-running the selector. The
/// paper reports adjacent transformer layers share 70-100% of
/// selected blocks, so this is mostly free quality-wise while
/// reducing selector dispatch count by `groupSize`x.
///
/// Cache is keyed by `(priorLen, layerGroup)` and is automatically
/// invalidated when `priorLen` changes (i.e. the next chunk starts).
/// Safe for single-stream inference (B=1); the lock protects against
/// future multi-stream eval.
public final class F83SelectorReuseCache: @unchecked Sendable {
    public static let shared = F83SelectorReuseCache()
    private let lock = NSLock()
    private var lastPriorLen: Int = -1
    private var positionsByGroup: [Int: MLXArray] = [:]
    private init() {}

    public static func cachedPositions(
        group: Int, priorLen: Int
    ) -> MLXArray? {
        return shared.cachedPositions(group: group, priorLen: priorLen)
    }

    public static func cachePositions(
        group: Int, priorLen: Int, positions: MLXArray
    ) {
        shared.cachePositions(group: group, priorLen: priorLen, positions: positions)
    }

    public static func clear() {
        shared.clear()
    }

    private func cachedPositions(group: Int, priorLen: Int) -> MLXArray? {
        lock.lock(); defer { lock.unlock() }
        if priorLen != lastPriorLen {
            positionsByGroup.removeAll(keepingCapacity: true)
            lastPriorLen = priorLen
            return nil
        }
        return positionsByGroup[group]
    }

    private func cachePositions(group: Int, priorLen: Int, positions: MLXArray) {
        lock.lock(); defer { lock.unlock() }
        if priorLen != lastPriorLen {
            positionsByGroup.removeAll(keepingCapacity: true)
            lastPriorLen = priorLen
        }
        positionsByGroup[group] = positions
    }

    private func clear() {
        lock.lock(); defer { lock.unlock() }
        positionsByGroup.removeAll()
        lastPriorLen = -1
    }
}

/// KV cache that backs `RetrievalAttention` block-sparse decode.
public final class RetrievalAttentionKVCache: BaseKVCache, CustomDebugStringConvertible {

    /// Inner raw K/V cache. Default is `StandardKVCache`. For TurboQuant+
    /// composition (PRD Phase D), this may be a `TurboQuantizedKVCache` in
    /// rawKeyMode (K=FP16 raw, V=4bit compressed) — RA's selector index
    /// continues to project from raw K while V gets memory savings via
    /// TQ compression at the cost of V dequant at decode.
    public let inner: BaseKVCache

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

    /// F-79 cached top-K outputs from the previous full selector
    /// refresh. Reused across `selectorAmortization` consecutive
    /// decode steps so we skip projectQ + F-48 fine + F-48 coarse on
    /// most steps. Refreshed when the offset advances past the
    /// amortization window.
    private var cachedFineStarts: MLXArray?
    private var cachedCoarseStarts: MLXArray?
    private var lastRefreshOffset: Int = -1

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

    /// TurboQuant+ composition init — Phase D PRD goal. The inner cache is
    /// a `TurboQuantizedKVCache` in rawKeyMode (K stays FP16, V is
    /// compressed at the configured bitwidth). RA's selector still projects
    /// from raw K; V compression yields the memory win.
    /// - parameters:
    ///   - valueBits: V compression bitwidth (default 4)
    ///   - tqStep: TQ allocation step (default 1024 to match TQ defaults)
    public init(
        layerIdx: Int,
        totalLayers: Int,
        raConfig: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        ropeBase: Float = 10_000.0,
        valueBits: Int,
        tqStep: Int = 1024
    ) {
        self.inner = TurboQuantizedKVCache(
            keyBits: 0,             // rawKeyMode: K stays FP16
            valueBits: valueBits,   // V compressed
            step: tqStep,
            useCompressedAttention: false  // RA does its own gather/SDPA path
        )
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

        // F-80 compose: when inner is TurboQuantizedKVCache (rawKeyMode), call
        // updateAndDequant — handles both prefill raw-store AND decode-time
        // compression transition + V dequant. Standard cache path is untouched.
        let cachedK: MLXArray
        let cachedV: MLXArray
        if let tqInner = inner as? TurboQuantizedKVCache {
            precondition(
                tqInner.rawKeyMode,
                "RetrievalAttention + TurboQuant compose requires rawKeyMode "
                + "(keyBits=0); K must stay raw FP16 for the JL selector index."
            )
            (cachedK, cachedV) = tqInner.updateAndDequant(keys: keys, values: values)
        } else {
            (cachedK, cachedV) = inner.update(keys: keys, values: values)
        }

        if !isSparseEligible {
            return (cachedK, cachedV)
        }

        ensureIndex(nKVHeads: nKVHeads, dHead: dHead)

        // F-72: skip the selector index update at decode steps (L==1).
        // Decode tokens always live in the sliding window — they don't
        // need to be in the block features for top-K selection. This
        // removes ~24ms / step of MLX-op overhead at 14B-1M @ 32K (the
        // entire RA-over-dense gap). Assumes decode count < sliding
        // window (default 2048) so the oldest decode tokens never fall
        // outside the always-attended trailing region. Prefill (L > 1)
        // still updates the index as before.
        let L = keys.dim(2)
        if L > 1 {
            // [1, nKVHeads, L, D] → [nKVHeads, L, D]
            let keysF32 = keys.asType(.float32)[0, 0..., 0..., 0...]
            batchedIndex!.update(newKeys: keysF32)
        }
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

    /// F-69 fused sparse SDPA path: produce SORTED gather indices entirely
    /// on GPU (may contain duplicates across heads; the fused kernel
    /// adjacent-diff-skips them). No asArray sync.
    public func gatherIndicesGPU(q: MLXArray) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        let nQHeads = q.dim(0)
        let groupSize = nQHeads / index.nKVHeads
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<index.nKVHeads).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = q.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)
        let topKPositions = index.expandedTopKPositionsGPU(projectedQ: projQ)

        let T = self.offset
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
        let clipped = clip(allPositions, min: Int32(0), max: Int32(T - 1))
        // Sort so kernel can adjacent-diff-skip duplicates.
        return sorted(clipped)
    }

    /// F-71 NSA-style group-centric fused sparse SDPA path. Builds the
    /// per-KV-head sorted gather, then calls the fused Metal kernel that
    /// reads each K/V byte once per KV group (shared across `groupSize`
    /// Q heads).
    ///
    /// - Parameters:
    ///   - queries: `[1, nQH, 1, D]` post-RoPE decode query
    ///   - keys: `[1, nKVH, T, D]` full cached keys
    ///   - values: `[1, nKVH, T, D]` full cached values
    ///   - qHeads: `[nQH, D]` flat post-RoPE query (for selector projection)
    ///   - scale: SDPA scale factor (typically `1/√D`)
    /// - Returns: SDPA output `[1, nQH, 1, D]`
    public func groupSparseSDPA(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        qHeads: MLXArray,
        scale: Float
    ) -> MLXArray {
        precondition(qHeads.shape.count == 2, "expected [nHeads, dHead]")
        precondition(keys.shape.count == 4, "expected [B, nKVH, T, D]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        let T = keys.dim(2)
        let nKVH = keys.dim(1)
        let nQH = queries.dim(1)
        let groupSize = nQH / nKVH

        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<nKVH).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = qHeads.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)

        let (gather, _) = index.perKVHeadGatherGPU(
            projectedQ: projQ, seqLen: T
        )
        // F-71b deliberately skips the sentinel-marking pre-pass — at 48
        // layers the per-step `MLX.where` adds ~14ms (the entire gap we'd
        // be trying to close). Dups in the sorted per-row gather appear
        // only when sliding overlaps top-K-fine or top-K-coarse — usually
        // 1–4% of K_padded and not in the high-weight positions. The
        // resulting softmax double-count biases output by <0.001 cosine
        // below the F-59 mask path. The kernel still honours the -1
        // sentinel if a caller chooses to pre-dedupe.
        let perHeadGather = gather.expandedDimensions(axis: 0)

        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            retrievalAttentionGroupSparseSDPA(
                queries: queries,
                keys: keys,
                values: values,
                perKVHeadGather: perHeadGather,
                scale: scale
            )
        }
    }

    /// F-70 per-KV-head gather + batched SDPA: each KV head gathers its
    /// own sorted set of static + sliding + top-K-fine + top-K-coarse
    /// positions. We reshape the KV-heads dim into the batch dim and call
    /// `MLXFast.scaledDotProductAttention` once on
    /// `[nKVH, groupSize, 1, D]` queries against `[nKVH, 1, K_padded, D]`
    /// gathered K/V — staying on the fused tiled SDPA fast path while
    /// each head only sees its own gather (no cross-head union).
    /// Sorted-row duplicates are masked via an adjacent-diff additive mask.
    ///
    /// - Parameters:
    ///   - queries: `[1, nQH, 1, D]` post-RoPE decode query
    ///   - keys: `[1, nKVH, T, D]` full cached keys
    ///   - values: `[1, nKVH, T, D]` full cached values
    ///   - qHeads: `[nQH, D]` flat post-RoPE query (for selector projection)
    ///   - scale: SDPA scale (typically `1/√D`)
    /// - Returns: SDPA output `[1, nQH, 1, D]`
    public func perKVHeadGatherAndAttend(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        qHeads: MLXArray,
        scale: Float
    ) -> MLXArray {
        precondition(qHeads.shape.count == 2, "expected [nHeads, dHead]")
        precondition(keys.shape.count == 4, "expected [B, nKVH, T, D]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        let T = keys.dim(2)
        let nKVH = keys.dim(1)
        let D = keys.dim(3)
        let nQH = queries.dim(1)
        precondition(
            nQH % nKVH == 0,
            "Q heads (\(nQH)) must be a multiple of KV heads (\(nKVH))"
        )
        let groupSize = nQH / nKVH

        // Project query via selector. Reuse strided rep-head pick from
        // gatherIndicesForDecode / gatherIndicesGPU.
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<nKVH).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = qHeads.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)

        // Per-KV-head sorted gather [nKVH, K_padded] (clipped to [0, T-1],
        // dups possible only within a row since static+sliding+head_topK
        // can overlap; cross-head overlap is NOT collapsed — that's the
        // whole point).
        let (gather, kPadded) = index.perKVHeadGatherGPU(
            projectedQ: projQ, seqLen: T
        )

        // Per-head gather via takeAlong on axis 2 of K/V. Indices reshape
        // to [1, nKVH, K_padded, 1] and broadcast across D inside the
        // op — this keeps the result in the mlx-fast SDPA fast path
        // layout `[1, nKVH, K_padded, D]` (B=1, nKVH unchanged) instead
        // of moving heads into the batch dim. F-70a observed the latter
        // layout (B=nKVH) regresses 9x at 32K, even though work is
        // smaller, because the decode kernel's parallelism is best
        // tuned for B=1.
        let gatherForTake = gather.expandedDimensions(axes: [0, 3])  // [1, nKVH, K_padded, 1]
        let gatheredK = takeAlong(keys, gatherForTake, axis: 2)   // [1, nKVH, K_padded, D]
        let gatheredV = takeAlong(values, gatherForTake, axis: 2) // [1, nKVH, K_padded, D]

        // Adjacent-diff additive mask. For a sorted row, position k is a
        // duplicate iff gather[h, k] == gather[h, k-1]. Position 0 always
        // valid. Mask must broadcast to [B, nQH, L, K_padded] for MLX's
        // SDPA. We expand the per-KVH mask to per-QH by repeating each
        // KVH row `groupSize` times along a new axis, then reshape.
        let prev = gather[0..., 0 ..< (kPadded - 1)]
        let curr = gather[0..., 1 ..< kPadded]
        let dupInner = (curr .== prev)  // [nKVH, K_padded-1] Bool
        let leadFalse = MLXArray.zeros([nKVH, 1], dtype: .bool)
        let dupMask = concatenated([leadFalse, dupInner], axis: 1)  // [nKVH, K_padded]
        let zeroLike = MLXArray(Float(0)).asType(keys.dtype)
        let negInfLike = MLXArray(-Float.infinity).asType(keys.dtype)
        // [nKVH, K_padded] → [1, nKVH, 1, 1, K_padded] → expand to
        //   [1, nKVH, groupSize, 1, K_padded] → reshape [1, nQH, 1, K_padded].
        let baseMask = MLX.where(dupMask, negInfLike, zeroLike)
            .reshaped(1, nKVH, 1, 1, kPadded)
        let expanded = broadcast(
            baseMask, to: [1, nKVH, groupSize, 1, kPadded]
        )
        let addMask = expanded.reshaped(1, nKVH * groupSize, 1, kPadded)

        let out = BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: gatheredK,
                values: gatheredV,
                scale: scale,
                mask: .array(addMask),
                sinks: nil
            )
        }
        return out
    }

    /// F-76 implicit-positions sparse SDPA. Combines F-75 (parallel
    /// fine+coarse score+topK kernel) + sparse SDPA over the implicit
    /// static+sliding+topK positions. Total per layer: 3 ops
    /// (inner.update + F-75 + F-76) vs F-73's 6 ops. Targets parity
    /// with dense.
    ///
    /// Returns SDPA output `[B, nQH, 1, D]` — no mask, no gather array.
    public func implicitSparseSDPA(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        qHeads: MLXArray,
        scale: Float
    ) -> MLXArray? {
        precondition(qHeads.shape.count == 2, "expected [nHeads, dHead]")
        guard let index = batchedIndex else { return nil }
        guard let fineFeatures = index.fineBlockFeatures,
              let coarseFeatures = index.coarseBlockFeatures
        else { return nil }
        let nKVH = index.nKVHeads
        let groupSize = qHeads.dim(0) / nKVH
        let nFine = fineFeatures.dim(1)
        let nCoarse = coarseFeatures.dim(1)
        let isPow2 = { (n: Int) in n > 0 && (n & (n - 1)) == 0 }
        if !(isPow2(nFine) && nFine <= 1024 && isPow2(nCoarse) && nCoarse <= 1024) {
            return nil
        }
        let T = keys.dim(2)
        let kFine = min(raConfig.effectiveFineTopK(seqLen: T), nFine)
        let kCoarse = raConfig.coarseRescueEnabled
            ? min(raConfig.coarseTopK, nCoarse) : 0
        guard kCoarse > 0 else { return nil }

        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<nKVH).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = qHeads.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)

        let (fineStarts, coarseStarts) = retrievalAttentionParallelScoreTopK(
            projectedQ: projQ,
            fineFeatures: fineFeatures,
            coarseFeatures: coarseFeatures,
            kFine: kFine, kCoarse: kCoarse,
            fineBlockSize: raConfig.fineBlockSize,
            coarseBlockSize: raConfig.coarseBlockSize
        )
        // Reshape to [B=1, nKVH, K_*] for the F-76 kernel.
        let fineStarts3D = fineStarts.expandedDimensions(axis: 0)
        let coarseStarts3D = coarseStarts.expandedDimensions(axis: 0)

        return BenchmarkSignpost.interval(BenchmarkSignpost.PhaseLabel.sdpa) {
            retrievalAttentionImplicitSparseSDPA(
                queries: queries, keys: keys, values: values,
                fineStarts: fineStarts3D, coarseStarts: coarseStarts3D,
                staticInit: raConfig.staticInit,
                slidingWindow: raConfig.slidingWindow,
                fineBlockSize: raConfig.fineBlockSize,
                coarseBlockSize: raConfig.coarseBlockSize,
                scale: scale
            )
        }
    }

    /// F-77 build-mask via parallel bundle (projectQ + score+topK in
    /// one launch) + F-73 mask kernel. Per layer: 2 kernel dispatches
    /// + 2 small MLX ops (q.take + cache update) — total ~4 ops/layer
    /// vs F-73's 5.
    public func buildAttentionMaskFusedParallelBundleKernel(
        q: MLXArray, dtype: DType, T: Int
    ) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        guard let jlW = index.jlW,
              let fineFeatures = index.fineBlockFeatures,
              let coarseFeatures = index.coarseBlockFeatures
        else {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let nKVH = index.nKVHeads
        let groupSize = q.dim(0) / nKVH
        let nFine = fineFeatures.dim(1)
        let nCoarse = coarseFeatures.dim(1)
        let isPow2 = { (n: Int) in n > 0 && (n & (n - 1)) == 0 }
        if !(isPow2(nFine) && nFine <= 1024 && isPow2(nCoarse) && nCoarse <= 1024) {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let kFine = min(raConfig.effectiveFineTopK(seqLen: T), nFine)
        let kCoarse = raConfig.coarseRescueEnabled
            ? min(raConfig.coarseTopK, nCoarse) : 0
        guard kCoarse > 0 else {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let qF32 = q.asType(.float32)
        let (fineStarts, coarseStarts) = retrievalAttentionParallelBundle(
            q: qF32, jlW: jlW,
            fineFeatures: fineFeatures, coarseFeatures: coarseFeatures,
            groupSize: groupSize,
            kFine: kFine, kCoarse: kCoarse,
            fineBlockSize: raConfig.fineBlockSize,
            coarseBlockSize: raConfig.coarseBlockSize
        )
        return retrievalAttentionBuildMaskFused(
            fineStarts: fineStarts, coarseStarts: coarseStarts,
            T: T,
            staticInit: raConfig.staticInit,
            slidingWindow: raConfig.slidingWindow,
            fineBS: raConfig.fineBlockSize,
            coarseBS: raConfig.coarseBlockSize,
            outputDtype: dtype
        )
    }

    /// F-75 build-mask via parallel score+topK kernel. Per layer:
    /// projectQ (MLX matmul) + F-75 (one kernel for both fine and
    /// coarse, 2×NKVH threadgroups concurrent) + F-73 mask. Total
    /// 3 kernel dispatches per layer (vs F-73's 4).
    public func buildAttentionMaskFusedParallelKernel(
        q: MLXArray, dtype: DType, T: Int
    ) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        guard let fineFeatures = index.fineBlockFeatures,
              let coarseFeatures = index.coarseBlockFeatures
        else {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let nKVH = index.nKVHeads
        let groupSize = q.dim(0) / nKVH
        let nFine = fineFeatures.dim(1)
        let nCoarse = coarseFeatures.dim(1)
        let isPow2 = { (n: Int) in n > 0 && (n & (n - 1)) == 0 }
        if !(isPow2(nFine) && nFine <= 1024 && isPow2(nCoarse) && nCoarse <= 1024) {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let kFine = min(raConfig.effectiveFineTopK(seqLen: T), nFine)
        let kCoarse = raConfig.coarseRescueEnabled
            ? min(raConfig.coarseTopK, nCoarse) : 0
        guard kCoarse > 0 else {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray(
                (0..<nKVH).map { Int32($0 * groupSize) }
            )
            eval(cachedHeadIdx!)
        }
        let qStacked = q.take(cachedHeadIdx!, axis: 0).asType(.float32)
        let projQ = index.projectQueriesBatched(qStacked)

        let (fineStarts, coarseStarts) = retrievalAttentionParallelScoreTopK(
            projectedQ: projQ,
            fineFeatures: fineFeatures,
            coarseFeatures: coarseFeatures,
            kFine: kFine, kCoarse: kCoarse,
            fineBlockSize: raConfig.fineBlockSize,
            coarseBlockSize: raConfig.coarseBlockSize
        )
        return retrievalAttentionBuildMaskFused(
            fineStarts: fineStarts, coarseStarts: coarseStarts,
            T: T,
            staticInit: raConfig.staticInit,
            slidingWindow: raConfig.slidingWindow,
            fineBS: raConfig.fineBlockSize,
            coarseBS: raConfig.coarseBlockSize,
            outputDtype: dtype
        )
    }

    /// F-74 fused mask build via selector-bundle kernel. One kernel call
    /// does projectQ + scoreTopK_fine + scoreTopK_coarse (replacing 3
    /// MLX ops), then the F-73 build-mask kernel produces the
    /// [1, 1, 1, T] mask. Final per-sparse-layer op count: 2 (instead
    /// of 5 in F-73-bundle and ~10 in F-59).
    public func buildAttentionMaskFusedBundleKernel(
        q: MLXArray, dtype: DType, T: Int
    ) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        guard let jlW = index.jlW else {
            fatalError("JL projection not initialized — call update first")
        }
        guard let fineFeatures = index.fineBlockFeatures,
              let coarseFeatures = index.coarseBlockFeatures
        else {
            // No coarse — fall back to the non-bundle path.
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let nKVH = index.nKVHeads
        let groupSize = q.dim(0) / nKVH
        let nFine = fineFeatures.dim(1)
        let nCoarse = coarseFeatures.dim(1)
        // Both must be powers of 2 ≤ 1024 for the fused bundle kernel
        // (tree-reduce constraint). Fall back to non-bundle F-73 path
        // otherwise.
        let isPow2 = { (n: Int) in n > 0 && (n & (n - 1)) == 0 }
        if !(isPow2(nFine) && nFine <= 1024 && isPow2(nCoarse) && nCoarse <= 1024) {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        let kFine = min(raConfig.effectiveFineTopK(seqLen: T), nFine)
        let kCoarse = raConfig.coarseRescueEnabled
            ? min(raConfig.coarseTopK, nCoarse) : 0
        guard kCoarse > 0 else {
            return buildAttentionMaskFusedKernel(q: q, dtype: dtype, T: T)
        }
        // q is [nQH, dHead] (rep-per-group stride happens INSIDE the
        // kernel via the GROUP_SIZE template parameter).
        let qF32 = q.asType(.float32)
        let (fineStarts, coarseStarts) = retrievalAttentionSelectorBundleFused(
            q: qF32, jlW: jlW,
            fineFeatures: fineFeatures, coarseFeatures: coarseFeatures,
            groupSize: groupSize,
            kFine: kFine, kCoarse: kCoarse,
            fineBlockSize: raConfig.fineBlockSize,
            coarseBlockSize: raConfig.coarseBlockSize
        )
        return retrievalAttentionBuildMaskFused(
            fineStarts: fineStarts, coarseStarts: coarseStarts,
            T: T,
            staticInit: raConfig.staticInit,
            slidingWindow: raConfig.slidingWindow,
            fineBS: raConfig.fineBlockSize,
            coarseBS: raConfig.coarseBlockSize,
            outputDtype: dtype
        )
    }

    /// F-73 fused build-mask path: compute top-K block starts then call
    /// the single Metal kernel that writes the `[1, 1, 1, T]` additive
    /// mask in one launch. Replaces the ~6-op `buildAttentionMaskGPU`
    /// pipeline (range + concat + clip + full + scatter) with one
    /// kernel call. Targets the 22.9ms / decode step selector-pipeline
    /// overhead measured in F-73 diagnostic.
    public func buildAttentionMaskFusedKernel(
        q: MLXArray, dtype: DType, T: Int
    ) -> MLXArray {
        precondition(q.shape.count == 2, "expected [nHeads, dHead]")
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

        // F-79: reuse cached topK arrays for `selectorAmortization`
        // consecutive decode steps. Refresh only when offset has
        // advanced past the window. Mask build still runs every step
        // (so the sliding window stays current).
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

    /// F-83 M3 — sparse prefill attend for a chunk of L queries.
    ///
    /// Called from the dispatcher's `.retrievalSparse` branch when
    /// `L > 1` and `priorChunkLen > sparsePrefillMinContext`. Splits
    /// attention into:
    ///   1. Prior chunks: gather K/V at the union top-K of fine +
    ///      coarse positions across all L queries per KV head, plus
    ///      static prefix and sliding window. All positions are <
    ///      chunk_start so the prior portion is uniformly causal-safe
    ///      (no per-row mask needed for it).
    ///   2. Within-chunk: dense causal SDPA on the chunk's own K/V.
    ///
    /// Concatenates prior_gathered + chunk K/V into a single
    /// `[B, nKVH, P+L, D]` tensor and runs one `MLXFast.SDPA` call
    /// with a `[L, P+L]` mask. This bypasses the explicit online
    /// softmax merge — the merge is implicit in SDPA's single softmax.
    ///
    /// `cachedKeys` / `cachedValues` are the result of `update()` —
    /// they already include the chunk's own K/V at the tail.
    public func prefillSparseAttend(
        queries: MLXArray,
        cachedKeys: MLXArray,
        cachedValues: MLXArray,
        scale: Float
    ) -> MLXArray {
        precondition(queries.shape.count == 4, "queries [B, nH, L, D]")
        precondition(cachedKeys.shape.count == 4, "cachedKeys [B, nKVH, T, D]")
        guard let index = batchedIndex else {
            fatalError("indices not initialized; call update first")
        }
        let nH = queries.dim(1)
        let L = queries.dim(2)
        let Tcache = cachedKeys.dim(2)
        let nKVH = cachedKeys.dim(1)
        let priorLen = Tcache - L
        precondition(priorLen >= 0, "cache shrunk during update?")
        // Fast path: nothing to gather. Fall through to dense.
        if priorLen <= 0 {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: cachedKeys, values: cachedValues,
                scale: scale, mask: .causal
            )
        }

        let cfg = raConfig
        let fineBS = cfg.fineBlockSize
        let staticEnd = Swift.min(cfg.staticInit, priorLen)
        let slidingStart = Swift.max(0, priorLen - cfg.slidingWindow)
        let groupQHeadKV = nH / nKVH
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupQHeadKV) })
            eval(cachedHeadIdx!)
        }

        // F-83 V1.3 — IndexCache (arxiv 2603.12201): first sparse layer
        // in each `sparsePrefillSelectorGroupSize` group computes the
        // selector + position list once; remaining layers in the group
        // reuse the cached positions. Adjacent transformer layers share
        // 70-100% of selected blocks in practice, so this is mostly
        // free quality-wise while reducing selector dispatch count by
        // `groupSize`x (default 4 → from ~40 sparse layers per chunk
        // down to ~10 selector runs per chunk).
        let groupSize = Swift.max(1, cfg.sparsePrefillSelectorGroupSize)
        let layerGroup = layerIdx / groupSize
        let positions: MLXArray
        if let cached = F83SelectorReuseCache.cachedPositions(
            group: layerGroup, priorLen: priorLen) {
            positions = cached
        } else {
            // queries[0] → [nH, L, D]; take strided reps per KV group → [nKVH, L, D]
            let qStacked = queries[0].take(cachedHeadIdx!, axis: 0).asType(.float32)
            let qProj = index.projectQueriesBatchedL(qStacked)
            let prefillFineTopK = cfg.sparsePrefillFineTopK > 0
                ? cfg.sparsePrefillFineTopK
                : cfg.effectiveFineTopK(seqLen: index.seqLen)
            let fineStartsClean = index.crossHeadUnionTopKExcludingRangesGPU(
                projectedQ: qProj,
                k: prefillFineTopK,
                blockSize: fineBS,
                staticEnd: staticEnd,
                slidingStart: slidingStart
            )

            // Build positions list — static + sliding + fine, no duplicates
            // by construction. CRITICAL: when sliding overlaps static
            // (slidingStart <= staticEnd, i.e. priorLen is short enough
            // that the sliding window reaches back to within the static
            // prefix), we drop static entirely — sliding already covers
            // it. Otherwise we'd duplicate positions 0..slidingStart-1.
            let includeStatic = staticEnd > 0 && slidingStart >= staticEnd
            let staticPositions: MLXArray = includeStatic
                ? MLXArray(Int32(0) ..< Int32(staticEnd))
                : MLXArray.zeros([0], dtype: .int32)
            let slidingPositions: MLXArray = slidingStart < priorLen
                ? MLXArray(Int32(slidingStart) ..< Int32(priorLen))
                : MLXArray.zeros([0], dtype: .int32)
            let kFine = fineStartsClean.dim(0)
            let computedPositions: MLXArray
            if kFine > 0 {
                let fineOffsets = MLXArray(0..<Int32(fineBS)).reshaped(1, fineBS)
                let finePos = (fineStartsClean.reshaped(kFine, 1) + fineOffsets)
                    .reshaped(kFine * fineBS)
                let combined = concatenated(
                    [staticPositions, finePos, slidingPositions], axis: 0)
                computedPositions = clip(
                    combined, min: Int32(0), max: Int32(priorLen - 1))
            } else {
                computedPositions = concatenated(
                    [staticPositions, slidingPositions], axis: 0)
            }
            // Force materialization before caching — if positions stays
            // lazy, downstream reuse re-runs the selector graph for each
            // group member, defeating the whole point.
            eval(computedPositions)
            F83SelectorReuseCache.cachePositions(
                group: layerGroup, priorLen: priorLen,
                positions: computedPositions)
            positions = computedPositions
        }

        // Gather prior K/V at `positions`. Keep within-chunk K/V at the tail.
        let priorK = cachedKeys[0..., 0..., ..<priorLen, 0...]
        let priorV = cachedValues[0..., 0..., ..<priorLen, 0...]
        let gK = priorK.take(positions, axis: 2)   // [B, nKVH, P, D]
        let gV = priorV.take(positions, axis: 2)
        let chunkK = cachedKeys[0..., 0..., priorLen..., 0...]   // [B, nKVH, L, D]
        let chunkV = cachedValues[0..., 0..., priorLen..., 0...]
        let combinedK = concatenated([gK, chunkK], axis: 2)
        let combinedV = concatenated([gV, chunkV], axis: 2)

        // F-83 V1.4 — use `.causal` symbolic mask mode instead of
        // materializing a [1, 1, L, P+L] additive mask. MLXFast.SDPA's
        // `.causal` applies causal with offset = S - L internally:
        // S = P + L, so offset = P → row q attends to cols 0..q+P-1.
        // That's exactly our pattern: all P prior cols attend (each is
        // < chunk_start, causally valid for every query l), and cols
        // [P..P+L) follow the standard chunk-internal lower-triangle.
        // Eliminates 6+ MLX ops per layer per chunk (zeros + range x2 +
        // compare + where + concat + reshape) that V1.3 still paid.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: combinedK, values: combinedV,
            scale: scale, mask: .causal
        )
    }

    public var debugDescription: String {
        "RetrievalAttentionKVCache(layer=\(layerIdx)/\(totalLayers), "
            + "offset=\(offset), heads=\(batchedIndex?.nKVHeads ?? 0), "
            + "sparseEligible=\(isSparseEligible))"
    }
}
