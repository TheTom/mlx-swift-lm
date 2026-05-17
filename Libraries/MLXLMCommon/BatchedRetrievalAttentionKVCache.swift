// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedRetrievalAttentionKVCache — F-85.
//
// Caveman: hold shared K/V rectangular for B slots. each slot pick own
// top-K via BatchedRetrievalAttentionIndexB. feed F-71b kernel direct
// — kernel already eat [B, nQH, 1, D] / [B, nKVH, T, D] / [B, nKVH, K].
//
// Composition: an inner BatchedKVCache holds the rectangular fp16
// K/V. The batched selector index sits alongside. At decode L=1:
//   1. cache.update writes new K/V into the rectangular buffer
//   2. index.update folds the new K (post-RoPE) into per-slot block
//      features
//   3. sparseAttend projects per-(B, KV) Q → top-K → gather list
//      [B, nKVH, K_padded] → calls retrievalAttentionGroupSparseSDPA
//
// v1: assumes rectangular T (all slots same offset). Continuous-batching
// ragged-T is a v2 concern.

import Foundation
import MLX

public final class BatchedRetrievalAttentionKVCache {

    /// Inner batched K/V cache. We hold a strong ref but don't own
    /// lifecycle exclusively — the model's per-layer cache list owns it.
    public let inner: BatchedKVCache

    /// Per-(B, KV-head) selector index.
    public let index: BatchedRetrievalAttentionIndexB

    public let raConfig: RetrievalAttentionConfig
    public let layerIdx: Int
    public let totalLayers: Int
    public let ropeBase: Float

    /// Cached `[B, nKVH]` int32 array of representative Q-head indices per
    /// KV group. Built lazily on first sparseAttend.
    private var cachedHeadIdx: MLXArray?

    public var isSparseEligible: Bool {
        raConfig.isSparseLayer(layerIdx: layerIdx, totalLayers: totalLayers)
    }

    public init(
        inner: BatchedKVCache,
        B: Int,
        nKVHeads: Int,
        dHead: Int,
        layerIdx: Int,
        totalLayers: Int,
        raConfig: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        ropeBase: Float = 10_000.0
    ) {
        self.inner = inner
        self.layerIdx = layerIdx
        self.totalLayers = totalLayers
        self.raConfig = raConfig
        self.ropeBase = ropeBase
        self.index = BatchedRetrievalAttentionIndexB(
            config: raConfig, B: B, dHead: dHead, nKVHeads: nKVHeads,
            ropeBase: ropeBase, layerIdx: layerIdx)
    }

    /// Apply a new K chunk to the selector index. Caller has ALREADY
    /// written K/V into the inner cache; this only updates selector state.
    /// Decode-step (L=1) is the common path; prefill chunks (L>1) also.
    public func updateIndex(newKeys: MLXArray) {
        guard isSparseEligible else { return }
        // Skip index updates on decode tokens — they live in the sliding
        // window so block-feature membership doesn't help selection. Matches
        // F-72 behavior in the single-batch wrapper cache.
        let L = newKeys.dim(2)
        if L > 1 {
            index.update(newKeys: newKeys.asType(.float32))
        }
        // (no L==1 update — sliding window covers it)
    }

    /// Sparse attend at decode step (L=1).
    ///
    /// - Parameters:
    ///   - queries: `[B, nQH, 1, D]` post-RoPE Q (already broadcasted with
    ///     GQA group expansion).
    ///   - scale: SDPA scale (1/sqrt(D)).
    /// - Returns: `[B, nQH, 1, D]` attention output.
    ///
    /// Path:
    ///   1. Slice per-(B, KV-head) representative Q via the GQA stride.
    ///   2. Project to selector content space.
    ///   3. Per-(B, nKVH) top-K + sorted gather list `[B, nKVH, K_padded]`.
    ///   4. Hand off to F-71b kernel `retrievalAttentionGroupSparseSDPA`.
    public func sparseAttend(
        queries: MLXArray, scale: Float
    ) -> MLXArray {
        precondition(queries.shape.count == 4, "queries must be [B, nQH, 1, D]")
        precondition(queries.dim(2) == 1, "decode-step L=1 only")
        let B = queries.dim(0)
        let nQH = queries.dim(1)
        let D = queries.dim(3)
        precondition(B == index.B, "B mismatch")

        // Pull rectangular K/V at the current offset. All slots assumed
        // to share offset (rectangular T) — caller responsible if not.
        let off = inner.offsets[0]
        let cachedK = inner.keys[..<B, 0..., ..<off, 0...]
        let cachedV = inner.values[..<B, 0..., ..<off, 0...]
        let nKVH = cachedK.dim(1)
        let T = cachedK.dim(2)
        precondition(nQH % nKVH == 0)
        let groupSize = nQH / nKVH

        // Build rep-per-group Q indices once. shape [nKVH] ints; we reuse
        // the same indices for every slot (Q heads laid out groupSize-contig).
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            eval(cachedHeadIdx!)
        }
        // queries[:, headIdx, 0, :]  → [B, nKVH, D]
        // queries is [B, nQH, 1, D]; squeeze L → [B, nQH, D]; take heads axis 1.
        let qSqueezed = queries[0..., 0..., 0, 0...]  // [B, nQH, D]
        let qRep = qSqueezed.take(cachedHeadIdx!, axis: 1).asType(.float32)
        // [B, nKVH, D]
        let projQ = index.projectQueriesBatched(qRep)
        let (gather, _) = index.perKVHeadGatherBatched(projectedQ: projQ, seqLen: T)

        // F-71b kernel direct. Returns [B, nQH, 1, D] in queries.dtype.
        return retrievalAttentionGroupSparseSDPA(
            queries: queries,
            keys: cachedK,
            values: cachedV,
            perKVHeadGather: gather,
            scale: scale
        )
    }
}
