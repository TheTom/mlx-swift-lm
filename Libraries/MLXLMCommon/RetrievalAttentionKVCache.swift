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

    /// Per-KV-head selector indices. Populated lazily on the first
    /// `update(...)` once we know `nKVHeads` and `dHead`.
    public private(set) var perHeadIndex: [RetrievalAttentionIndex] = []

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

    /// Allocate one selector index per KV head on first sight of the cache
    /// shape. Cheap to call on every update; idempotent after the first.
    private func ensureIndices(nKVHeads: Int, dHead: Int) {
        guard perHeadIndex.isEmpty else { return }
        perHeadIndex = (0..<nKVHeads).map { _ in
            RetrievalAttentionIndex(
                config: raConfig,
                dHead: dHead,
                ropeBase: ropeBase,
                layerIdx: layerIdx
            )
        }
    }

    /// Update inner storage AND (for sparse-eligible layers) every
    /// per-head selector index.
    ///
    /// `keys` / `values` come in as `[B, nKVHeads, L, D]` (post-RoPE for K).
    /// v1 supports B == 1. Multi-B caches would need per-request indices.
    ///
    /// Dense-band layers (first-N / last-N) skip the index update entirely
    /// — F-43 showed the selector update is the dominant per-step cost,
    /// and dense layers never query the index. 8 of 48 layers on 14B-1M
    /// → ~17% of the wasted overhead avoided "for free".
    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        precondition(keys.dim(0) == 1, "RA cache supports B=1 only in v1")
        let nKVHeads = keys.dim(1)
        let dHead = keys.dim(3)

        // Update the inner cache first — returns the *full* cached K/V tensor.
        let (cachedK, cachedV) = inner.update(keys: keys, values: values)

        // Dense-band layer? Skip the index entirely.
        if !isSparseEligible {
            return (cachedK, cachedV)
        }

        ensureIndices(nKVHeads: nKVHeads, dHead: dHead)

        // Update per-head indices with just the new rows (L tokens of K).
        // K is post-RoPE here (verified F-01).
        // Casting to float32 keeps the selector math stable; the index is fp32.
        let keysF32 = keys.asType(.float32)
        for h in 0..<nKVHeads {
            let newRows = keysF32[0, h, 0..., 0...]
            perHeadIndex[h].update(newK: newRows)
        }
        return (cachedK, cachedV)
    }

    /// Compute the union of per-head fine + coarse top-K block start positions
    /// for a given decode-step query. Each head's index sees the same query;
    /// downstream SDPA still runs per-head, so this is a gather-set hint only.
    ///
    /// - Parameter q: `[nHeads, dHead]` — current query, post-RoPE, for the
    ///   decode step.
    /// - Returns: deduplicated, sorted gather indices (static + sliding +
    ///   fine top-K + coarse top-K) ready for `keys.take(_, axis: 2)`.
    public func gatherIndicesForDecode(q: MLXArray) -> [Int] {
        precondition(q.shape.count == 2, "expected [nHeads, dHead], got \(q.shape)")
        let nQHeads = q.dim(0)
        let nKVHeads = perHeadIndex.count
        precondition(nKVHeads > 0, "indices not initialized; call update first")
        precondition(
            nQHeads % nKVHeads == 0,
            "Q heads (\(nQHeads)) must be a multiple of KV heads (\(nKVHeads))"
        )
        let groupSize = nQHeads / nKVHeads

        let seqLen = self.offset
        var allFine = Set<Int>()
        var allCoarse = Set<Int>()
        for h in 0..<nKVHeads {
            // perKVGroupMax: score each Q-head in the group, keep the one
            // with the highest top-of-list block, then take its top-K set.
            // v1 simplification: just take the GQA-group max scorer instead
            // of properly maxing scores. The first iteration we have the
            // Q head index simply == h * groupSize (representative head).
            // Future: do score(blockFeatures, q_i) for each q_i in group
            // and elementwise max before topK.
            let qHead = q[h * groupSize, 0...]
            let projQ = perHeadIndex[h].projectQuery(qHead.asType(.float32))
            let fineStarts = perHeadIndex[h].topKFineBlockStarts(against: projQ)
            for s in fineStarts { allFine.insert(s) }
            if raConfig.coarseRescueEnabled {
                let coarseStarts = perHeadIndex[h].topKCoarseBlockStarts(against: projQ)
                for s in coarseStarts { allCoarse.insert(s) }
            }
        }
        return retrievalAttentionGatherIndices(
            seqLen: seqLen,
            fineBlockStarts: Array(allFine),
            coarseBlockStarts: Array(allCoarse),
            config: raConfig
        )
    }

    public var debugDescription: String {
        "RetrievalAttentionKVCache(layer=\(layerIdx)/\(totalLayers), "
            + "offset=\(offset), heads=\(perHeadIndex.count), "
            + "sparseEligible=\(isSparseEligible))"
    }
}
