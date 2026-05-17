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

    /// F-85 v2/v3 — env-gated kernel selection. Read once at module load.
    /// Values:
    ///   "f73"           (default) — F-73 batched mask kernel + MLXFast SDPA.
    ///   "f73loop"                 — Swift loop over single-batch F-73 mask + SDPA.
    ///   "f71b"                    — original v1 fused F-71b sparse SDPA.
    ///   "composegather" (F-85 v3) — per-(B, nKVH) gather K/V into [B,nKVH,K_padded,D]
    ///                               then dense MLXFast SDPA on the small slab.
    ///                               Theory: at B=8 ctx=32K K_padded≈2K, KV BW
    ///                               saving ≈16× vs F-73 mask which reads full T.
    public static let envBatchedKernel: String = {
        ProcessInfo.processInfo.environment["VSM_SPARSE_BATCHED_KERNEL"]
            ?? "f73"
    }()

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
    /// Dispatches by `VSM_SPARSE_BATCHED_KERNEL`:
    /// - `f73`           (default) — F-73 batched mask kernel + MLXFast SDPA.
    /// - `f73loop`                 — Swift loop over single-batch F-73 mask + SDPA.
    /// - `f71b`                    — original v1 F-71b custom kernel.
    /// - `composegather` (v3)      — per-(B, nKVH) gather → small dense SDPA.
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

        // Build rep-per-group Q indices once.
        if cachedHeadIdx == nil {
            cachedHeadIdx = MLXArray((0..<nKVH).map { Int32($0 * groupSize) })
            eval(cachedHeadIdx!)
        }
        let qSqueezed = queries[0..., 0..., 0, 0...]                         // [B, nQH, D]
        let qRep = qSqueezed.take(cachedHeadIdx!, axis: 1).asType(.float32)  // [B, nKVH, D]
        let projQ = index.projectQueriesBatched(qRep)                        // [B, nKVH, contentDim]

        let kernelChoice = BatchedRetrievalAttentionKVCache.envBatchedKernel
        switch kernelChoice {
        case "f71b":
            // v1 path — F-71b custom kernel. Retained for A/B regression check.
            let (gather, _) = index.perKVHeadGatherBatched(projectedQ: projQ, seqLen: T)
            return retrievalAttentionGroupSparseSDPA(
                queries: queries,
                keys: cachedK,
                values: cachedV,
                perKVHeadGather: gather,
                scale: scale
            )

        case "f73loop":
            // Path B — Swift loop over single-batch F-73 mask + SDPA. Slow
            // fallback for A/B vs the batched-kernel path.
            return sparseAttendF73Loop(
                queries: queries, projQ: projQ, K: cachedK, V: cachedV,
                T: T, scale: scale)

        case "composegather":
            // Path C (F-85 v3) — per-(B, nKVH) gather K/V into a small
            // [B, nKVH, K_padded, D] slab via takeAlong, then dense MLXFast
            // SDPA with mask:.none on the slab.
            return sparseAttendComposeGather(
                queries: queries, projQ: projQ, K: cachedK, V: cachedV,
                T: T, scale: scale)

        default:  // "f73" + anything else
            // Path A — single batched F-73 mask kernel + MLXFast SDPA.
            return sparseAttendF73Batched(
                queries: queries, projQ: projQ, K: cachedK, V: cachedV,
                T: T, scale: scale)
        }
    }

    /// F-85 v2 path A — one batched F-73 mask kernel launch + MLXFast SDPA.
    private func sparseAttendF73Batched(
        queries: MLXArray, projQ: MLXArray,
        K: MLXArray, V: MLXArray, T: Int, scale: Float
    ) -> MLXArray {
        // Per-slot per-KV-head top-K block starts. Both shapes [B, nKVH, K].
        let fineStarts = index.topKFineBlockStarts(projectedQ: projQ)
        let coarseStarts = index.topKCoarseBlockStarts(projectedQ: projQ)
        let mask = retrievalAttentionBuildMaskFusedBatched(
            fineStarts: fineStarts,
            coarseStarts: coarseStarts,
            T: T,
            staticInit: raConfig.staticInit,
            slidingWindow: raConfig.slidingWindow,
            fineBS: raConfig.fineBlockSize,
            coarseBS: raConfig.coarseBlockSize,
            outputDtype: K.dtype
        )
        // mask is [B, 1, 1, T] — broadcasts over nQH and L=1.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: K, values: V,
            scale: scale, mask: .array(mask)
        )
    }

    /// F-85 v2 path B — fallback. B Swift-side launches of single-batch
    /// F-73 mask + per-slot MLXFast SDPA. Only for A/B sanity vs the
    /// batched-kernel path; expected slower at B>1.
    private func sparseAttendF73Loop(
        queries: MLXArray, projQ: MLXArray,
        K: MLXArray, V: MLXArray, T: Int, scale: Float
    ) -> MLXArray {
        let B = queries.dim(0)
        let fineStarts = index.topKFineBlockStarts(projectedQ: projQ)
        let coarseStarts = index.topKCoarseBlockStarts(projectedQ: projQ)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(B)
        for b in 0..<B {
            let fineB = fineStarts[b, 0..., 0...]    // [nKVH, K_fine]
            let coarseB = coarseStarts[b, 0..., 0...]
            let mask = retrievalAttentionBuildMaskFused(
                fineStarts: fineB,
                coarseStarts: coarseB,
                T: T,
                staticInit: raConfig.staticInit,
                slidingWindow: raConfig.slidingWindow,
                fineBS: raConfig.fineBlockSize,
                coarseBS: raConfig.coarseBlockSize,
                outputDtype: K.dtype
            )  // [1, 1, 1, T]
            let qSlot = queries[b ..< (b + 1), 0..., 0..., 0...]  // [1, nQH, 1, D]
            let kSlot = K[b ..< (b + 1), 0..., 0..., 0...]
            let vSlot = V[b ..< (b + 1), 0..., 0..., 0...]
            let oSlot = MLXFast.scaledDotProductAttention(
                queries: qSlot, keys: kSlot, values: vSlot,
                scale: scale, mask: .array(mask)
            )
            outputs.append(oSlot)
        }
        return concatenated(outputs, axis: 0)
    }

    /// F-85 v3 path C — compose-gather: build per-(B, nKVH) gather index
    /// `[B, nKVH, K_padded]`, take K/V along axis 2 to materialize
    /// `[B, nKVH, K_padded, D]` slabs, then call dense MLXFast SDPA with
    /// `mask: .none` on the small slab. GQA broadcast (nKVH → nQH) handled
    /// internally by MLXFast.
    ///
    /// Bandwidth theory (B=8, T=32K, K_padded≈2K, nKVH=8, D=128):
    ///   dense reads:   B*nKVH*T*D*2B   = 268M elem × 2B = 536 MB/layer
    ///   compose reads: B*nKVH*K_pad*D*2B = 33 MB/layer
    ///   = 16× KV BW saving. Add SDPA on small slab + take overhead.
    ///
    /// Compose-gather vs F-84 blockGather: F-84 is B=1 with cross-head 1D
    /// union (small T). Here we keep per-(B, nKVH) gather since selector is
    /// already that shape, AND the union-across-heads-across-batch math gets
    /// silly at B=8 (worst-case 8 × 8 × 32 fully disjoint blocks = 16K of
    /// 32K which is close to dense anyway). Per-(B, nKVH) keeps the slab
    /// fully addressed and stays in fast path.
    private func sparseAttendComposeGather(
        queries: MLXArray, projQ: MLXArray,
        K: MLXArray, V: MLXArray, T: Int, scale: Float
    ) -> MLXArray {
        // Reuse existing selector — already returns [B, nKVH, K_padded] int32
        // gather indices sorted + clipped to [0, T-1].
        let (gather, kPadded) = index.perKVHeadGatherBatched(
            projectedQ: projQ, seqLen: T)
        precondition(gather.shape.count == 3, "gather must be [B, nKVH, K_padded]")
        precondition(gather.dim(0) == queries.dim(0), "gather B mismatch")
        // K/V shape [B, nKVH, T, D]. takeAlong expects indices broadcastable
        // to K excluding axis 2 → [B, nKVH, K_padded, 1] broadcasts D.
        let gExp = gather.expandedDimensions(axis: 3)        // [B, nKVH, K_padded, 1]
        let kSmall = takeAlong(K, gExp, axis: 2)             // [B, nKVH, K_padded, D]
        let vSmall = takeAlong(V, gExp, axis: 2)             // [B, nKVH, K_padded, D]
        _ = kPadded  // silence unused warning; kPadded == kSmall.dim(2)
        // Dense SDPA on the small slab. mask:.none hits the MLXFast fused
        // tile path (F-84 measured ~9× faster than mask:.array at gathered
        // shapes). GQA broadcasts nKVH up to nQH automatically.
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: kSmall, values: vSmall,
            scale: scale, mask: .none
        )
    }
}
