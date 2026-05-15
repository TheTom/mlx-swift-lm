// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedRetrievalAttentionIndex — batched-across-KV-heads version of
// RetrievalAttentionIndex. F-43 measured the per-head Swift loop in
// gatherIndicesForDecode as the dominant ~1.5s/step overhead at 24K
// context (8 heads × 4 GPU sync points × 48 layers × 32 steps ≈ 49K
// syncs). This class slashes that to ~4 sync points per layer per step
// by stacking all KV-head feature tensors and doing one batched matmul
// for project + one batched matmul for score.
//
// Shape conventions:
//   perTokenFeatures:    [nKVHeads, T, selectorDim]
//   fineBlockFeatures:   [nKVHeads, nFineBlocks, selectorDim]
//   coarseBlockFeatures: [nKVHeads, nCoarseBlocks, selectorDim]
//   q (input to score):  [nKVHeads, dHead]

import Foundation
import MLX

public final class BatchedRetrievalAttentionIndex {

    public let config: RetrievalAttentionConfig
    public let dHead: Int
    public let nKVHeads: Int
    public let ropeBase: Float
    public let layerIdx: Int

    private var jlMatrix: MLXArray?

    /// Read-only accessor for the cached transposed JL projection matrix
    /// (shape `[dHead, contentDim]`). Used by F-74 fused selector-bundle
    /// kernel to do projectQ inside the kernel as `Q @ W^T`. Nil if
    /// `update(...)` hasn't run yet.
    public var jlW: MLXArray? { jlMatrixT }
    /// Pre-transposed JL matrix `[dHead, contentDim]` for matmul. Cached
    /// once so we don't transpose on every update.
    private var jlMatrixT: MLXArray?

    /// Pre-allocated per-token features buffer. Grown in `featureChunkSize`
    /// chunks to avoid per-step concat reallocation. Only positions
    /// `[0..<populated]` are valid; positions `[populated..<buffer.dim(1)]`
    /// are uninitialised reservation.
    private(set) public var perTokenFeatures: MLXArray?
    private var populated: Int = 0
    private static let featureChunkSize: Int = 1024

    /// Pre-allocated block feature buffers (mirrors perTokenFeatures
    /// pattern). Block means are written in-place at their slot index;
    /// `effectiveFine/CoarseBlockFeatures` exposes the valid prefix
    /// `[0..<currentFine/CoarseBlocks]` for scoring.
    private var fineBlockBuffer: MLXArray?
    private var coarseBlockBuffer: MLXArray?
    private var fineBlockCapacity: Int = 0
    private var coarseBlockCapacity: Int = 0
    private static let blockBufferChunkSize: Int = 256

    /// Number of currently-meaningful block features (completed + partial).
    private var currentFineBlocks: Int { (populated + config.fineBlockSize - 1) / config.fineBlockSize }
    private var currentCoarseBlocks: Int { (populated + config.coarseBlockSize - 1) / config.coarseBlockSize }

    /// Public view onto the valid prefix of the pre-allocated buffer.
    public var fineBlockFeatures: MLXArray? {
        guard let buf = fineBlockBuffer, currentFineBlocks > 0 else { return nil }
        return buf[0..., ..<currentFineBlocks, 0...]
    }
    public var coarseBlockFeatures: MLXArray? {
        guard let buf = coarseBlockBuffer, currentCoarseBlocks > 0 else { return nil }
        return buf[0..., ..<currentCoarseBlocks, 0...]
    }

    public init(
        config: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        dHead: Int,
        nKVHeads: Int,
        ropeBase: Float = 10_000.0,
        layerIdx: Int
    ) {
        self.config = config
        self.dHead = dHead
        self.nKVHeads = nKVHeads
        self.ropeBase = ropeBase
        self.layerIdx = layerIdx
    }

    /// Current populated sequence length (number of valid rows in
    /// `perTokenFeatures`).
    public var seqLen: Int { populated }

    /// Update with new K rows for all heads in one call.
    ///
    /// - Parameter newKeys: `[nKVHeads, L, dHead]` post-RoPE keys.
    public func update(newKeys: MLXArray) {
        precondition(newKeys.shape.count == 3, "expected [nKVHeads, L, dHead], got \(newKeys.shape)")
        precondition(newKeys.dim(0) == nKVHeads, "head count mismatch")
        precondition(newKeys.dim(2) == dHead, "dHead mismatch")

        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(dHead: dHead, config: config)
            jlMatrixT = jlMatrix!.transposed(1, 0)
            eval(jlMatrix!, jlMatrixT!)
        }
        let WT = jlMatrixT!

        let oldSeqLen = seqLen
        let L = newKeys.dim(1)
        let newSeqLen = oldSeqLen + L

        // Project: [nKVHeads, L, dHead] @ [dHead, contentDim] = [nKVHeads, L, contentDim]
        // F-67: store features as fp16 to halve perTokenFeatures memory.
        // Score path upcasts on read; quality preserved (16-element dot
        // products fit fp16 dynamic range cleanly).
        let contentNew = matmul(newKeys, WT).asType(.float16)

        // Trig features only when lambdaPos > 0 — pure content (default
        // λ=0 ship config per F-46) ignores the trig half of the selector.
        // Skipping trig avoids ~5-6 MLX ops per update: trig features,
        // broadcast, and the concat. perTokenFeatures becomes
        // [nKVHeads, T, contentDim] instead of [nKVHeads, T, selectorDim].
        let selectorNew: MLXArray
        if config.usesTrigFeatures {
            let positions = MLXArray((Int32(oldSeqLen)..<Int32(newSeqLen))).asType(.int32)
            let trigNew = retrievalAttentionTrigFeatures(
                relativePositions: positions, base: ropeBase, config: config
            )
            let trigBroadcast = broadcast(
                trigNew.reshaped(1, L, config.trigDim),
                to: [nKVHeads, L, config.trigDim]
            )
            selectorNew = concatenated([contentNew, trigBroadcast], axis: -1)
        } else {
            selectorNew = contentNew
        }

        // Append into the pre-allocated buffer. Reallocate only when the
        // current buffer can't hold the new rows. F-51 perf win — replaces
        // the per-update concatenated([existing, selectorNew]) with an
        // in-place slice assignment when the buffer has room.
        let featureDim = selectorNew.dim(2)
        if perTokenFeatures == nil {
            let initialCap = max(
                Self.featureChunkSize,
                ((newSeqLen + Self.featureChunkSize - 1) / Self.featureChunkSize)
                    * Self.featureChunkSize
            )
            perTokenFeatures = MLXArray.zeros(
                [nKVHeads, initialCap, featureDim], dtype: selectorNew.dtype
            )
        } else if perTokenFeatures!.dim(1) < newSeqLen {
            // F-83 V1.1 → V1.4 — HYBRID grow strategy.
            //
            // Prefill (L>1, adds many tokens per call): doubling. Reduces
            // grow events to O(log N) over the whole prefill — at 128K
            // that's 7 grows instead of 128.
            //
            // Decode (L=1) or any small write: LINEAR `featureChunkSize`.
            // At 256K context, doubling from cap=256K → 512K would alloc
            // another 6 GB just to make room for 1 extra row. That's the
            // OOM trigger when the post-prefill state is already near the
            // 64 GB box ceiling (sparse-only 256K run jetsam'd here).
            //
            // Heuristic: if the increment is "small" (< current cap / 4)
            // grow linearly by featureChunkSize-multiples; otherwise
            // double. Amortized cost stays O(log N) during prefill, and
            // decode pays a fixed featureChunkSize bytes per refill.
            let currentCap = perTokenFeatures!.dim(1)
            let increment = newSeqLen - currentCap
            let newCap: Int
            if increment * 4 < currentCap {
                // small grow — pad up by featureChunkSize multiples
                let padding = ((increment + Self.featureChunkSize - 1)
                    / Self.featureChunkSize) * Self.featureChunkSize
                newCap = currentCap + padding
            } else {
                // bulk grow — keep doubling
                var c = currentCap
                while c < newSeqLen { c *= 2 }
                newCap = c
            }
            let additional = newCap - currentCap
            let pad = MLXArray.zeros(
                [nKVHeads, additional, featureDim], dtype: perTokenFeatures!.dtype
            )
            perTokenFeatures = concatenated([perTokenFeatures!, pad], axis: 1)
            eval(perTokenFeatures!)
        }
        // In-place write the new rows.
        perTokenFeatures![0..., oldSeqLen ..< newSeqLen, 0...] = selectorNew
        populated = newSeqLen

        // Block-pooled features written in-place into pre-allocated
        // buffers. F-52 perf — removes the eval() barrier that was needed
        // when block features were a growing concat'd tensor, since the
        // in-place writes don't accumulate a deep graph.
        updateBlockBufferInPlace(
            blockSize: config.fineBlockSize, oldSeqLen: oldSeqLen,
            isFine: true, featureDim: featureDim, multiToken: L > 1
        )
        if config.coarseRescueEnabled {
            updateBlockBufferInPlace(
                blockSize: config.coarseBlockSize, oldSeqLen: oldSeqLen,
                isFine: false, featureDim: featureDim, multiToken: L > 1
            )
        }
        // No explicit eval — in-place writes don't accumulate a deep
        // graph chain. The next consumer materializes dependent reads
        // naturally. (F-66 found 4.6 GB prefill overhead; adding eval
        // here didn't reduce it — the bulk of the overhead is elsewhere
        // in the forward-pass intermediate buffers, not in the index
        // update path.)
    }

    /// In-place block-pool update onto a pre-allocated buffer.
    /// For L=1 decode steps: rewrite only the last (partial) block — the
    /// only one whose token count changed.
    /// For multi-token prefill: rewrite all blocks touched by the new
    /// region. Caller guarantees `populated` is the post-update value.
    private func updateBlockBufferInPlace(
        blockSize: Int, oldSeqLen: Int, isFine: Bool,
        featureDim: Int, multiToken: Bool
    ) {
        ensureBlockBuffer(isFine: isFine, featureDim: featureDim)
        let newBlockCount = (populated + blockSize - 1) / blockSize
        let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!

        let firstAffected: Int
        if multiToken {
            firstAffected = oldSeqLen / blockSize
        } else {
            firstAffected = newBlockCount - 1  // only last block can have changed
        }

        for b in firstAffected ..< newBlockCount {
            let start = b * blockSize
            let end = min(start + blockSize, populated)
            // [nKVHeads, end-start, D]
            let slice = perTokenFeatures![0..., start ..< end, 0...]
            let mean = slice.mean(axis: 1)  // [nKVHeads, D]
            buf[0..., b, 0...] = mean
        }
    }

    /// Ensure block buffer has capacity for `currentXBlocks` blocks; grow
    /// in `blockBufferChunkSize` chunks otherwise.
    private func ensureBlockBuffer(isFine: Bool, featureDim: Int) {
        let needed = isFine ? currentFineBlocks : currentCoarseBlocks
        let cap = isFine ? fineBlockCapacity : coarseBlockCapacity
        if cap >= needed && (isFine ? fineBlockBuffer : coarseBlockBuffer) != nil {
            return
        }
        // F-83 V1.1 — doubling growth (matches the exponential strategy
        // in perTokenFeatures). At 128K context fineBlocks = 2048 and
        // coarseBlocks = 128 — both grow log(N) times instead of N/256.
        let chunkSize = Self.blockBufferChunkSize
        var newCap = max(cap, chunkSize)
        while newCap < needed { newCap *= 2 }
        let dtype = perTokenFeatures?.dtype ?? .float32
        let newBuf = MLXArray.zeros([nKVHeads, newCap, featureDim], dtype: dtype)
        // Copy old contents into new buffer at [0..<cap].
        if cap > 0, let old = isFine ? fineBlockBuffer : coarseBlockBuffer {
            newBuf[0..., ..<cap, 0...] = old[0..., ..<cap, 0...]
        }
        if isFine {
            fineBlockBuffer = newBuf
            fineBlockCapacity = newCap
        } else {
            coarseBlockBuffer = newBuf
            coarseBlockCapacity = newCap
        }
    }

    // (batchedBlockMeanPool + updateBlockTailIncremental removed in F-52;
    // updateBlockBufferInPlace handles both prefill and decode paths
    // against the pre-allocated block buffers.)

    /// Project the query for all heads in one go.
    /// - Parameter q: `[nKVHeads, dHead]`
    /// - Returns: `[nKVHeads, effectiveSelectorDim]` (`contentDim` when
    ///   trig is skipped, `selectorDim` otherwise).
    public func projectQueriesBatched(_ q: MLXArray) -> MLXArray {
        precondition(q.shape == [nKVHeads, dHead], "expected [\(nKVHeads), \(dHead)], got \(q.shape)")
        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(dHead: dHead, config: config)
            jlMatrixT = jlMatrix!.transposed(1, 0)
            eval(jlMatrix!, jlMatrixT!)
        }
        let WT = jlMatrixT!
        // [nh, dHead] @ [dHead, contentDim] = [nh, contentDim]
        let contentQ = matmul(q.asType(.float32), WT)
        guard config.usesTrigFeatures else { return contentQ }
        let trigQ = broadcast(
            retrievalAttentionTrigFeatures(
                relativePositions: MLXArray([Int32(0)]), base: ropeBase, config: config
            ).reshaped(1, config.trigDim),
            to: [nKVHeads, config.trigDim]
        )
        return concatenated([contentQ, trigQ], axis: -1)
    }

    /// Score fine blocks for all heads in one batched op.
    /// Returns `[nKVHeads, nBlocks]` scores (CPU asArray happens at call site).
    public func scoreFineBlocksBatched(projectedQ: MLXArray) -> MLXArray {
        guard let features = fineBlockFeatures else {
            return MLXArray.zeros([nKVHeads, 0], dtype: .float32)
        }
        return scoreBlocksBatched(features: features, projectedQ: projectedQ)
    }

    public func scoreCoarseBlocksBatched(projectedQ: MLXArray) -> MLXArray? {
        guard fineBlockFeatures != nil, let features = coarseBlockFeatures else {
            return nil
        }
        return scoreBlocksBatched(features: features, projectedQ: projectedQ)
    }

    /// Shared scoring kernel. `features` is `[nKVHeads, nBlocks, dim]`
    /// where dim is `effectiveSelectorDim`. When trig is skipped, both
    /// features and projectedQ are contentDim wide and we do a single
    /// elementwise-mul + sum. Otherwise we honor the lambda blend.
    private func scoreBlocksBatched(features: MLXArray, projectedQ: MLXArray) -> MLXArray {
        let lastDim = features.dim(2)
        let qExp = projectedQ.reshaped(nKVHeads, 1, lastDim)
        if !config.usesTrigFeatures {
            // λ=0 pure content. features + q are both contentDim-wide.
            return (features * qExp).sum(axis: -1)
        }
        let lambdaPos = config.lambdaPos
        let cD = config.contentDim
        if lambdaPos == 1.0 {
            let prod = features[0..., 0..., cD...] * qExp[0..., 0..., cD...]
            return prod.sum(axis: -1)
        }
        let content = (features[0..., 0..., ..<cD] * qExp[0..., 0..., ..<cD]).sum(axis: -1)
        let trig = (features[0..., 0..., cD...] * qExp[0..., 0..., cD...]).sum(axis: -1)
        return (1 - lambdaPos) * content + lambdaPos * trig
    }

    /// Top-k block starts (fine or coarse) per head. Uses MLX argPartition
    /// on GPU instead of CPU partial sort. asArray pulls only the K winning
    /// block indices per head (was nBlocks per head — typically 5-50x less
    /// data transfer per layer per step).
    ///
    /// Phase C step 1 toward fused kernel: removes the CPU partial sort
    /// hot path; sort stays on GPU.
    private func topKBlockStartsAllHeads(
        scores: MLXArray, k: Int, blockSize: Int
    ) -> [[Int]] {
        let nBlocks = scores.dim(1)
        let take = min(k, nBlocks)
        guard take > 0 else { return Array(repeating: [], count: nKVHeads) }
        // argPartition partitions ascending; the largest `take` values'
        // indices land at positions [nBlocks - take .. nBlocks).
        let pivotKth = nBlocks - take
        let partitioned: MLXArray
        if pivotKth <= 0 {
            // Every block fits; just emit block-order indices.
            let rangeArr = MLXArray(0..<Int32(nBlocks))
                .reshaped(1, nBlocks)
            partitioned = broadcast(rangeArr, to: [nKVHeads, nBlocks])
        } else {
            partitioned = argPartition(scores, kth: pivotKth, axis: -1)
        }
        let topKIdx = partitioned[0..., (nBlocks - take)...]  // [nKVH, take]
        let blockStarts = topKIdx * Int32(blockSize)
        let cpu = blockStarts.asArray(Int32.self)
        var result: [[Int]] = []
        result.reserveCapacity(nKVHeads)
        for h in 0..<nKVHeads {
            let base = h * take
            result.append((0..<take).map { Int(cpu[base + $0]) })
        }
        return result
    }

    public func topKFineBlockStartsAllHeads(projectedQ: MLXArray) -> [[Int]] {
        let scores = scoreFineBlocksBatched(projectedQ: projectedQ)
        let k = config.effectiveFineTopK(seqLen: seqLen)
        return topKBlockStartsAllHeads(scores: scores, k: k, blockSize: config.fineBlockSize)
    }

    public func topKCoarseBlockStartsAllHeads(projectedQ: MLXArray) -> [[Int]] {
        guard let scores = scoreCoarseBlocksBatched(projectedQ: projectedQ) else {
            return Array(repeating: [], count: nKVHeads)
        }
        return topKBlockStartsAllHeads(
            scores: scores, k: config.coarseTopK, blockSize: config.coarseBlockSize
        )
    }

    /// F-70 path: per-KV-head sorted gather indices for batched SDPA.
    /// Each row is one KV head's gather: static + sliding + this head's
    /// top-K fine + top-K coarse, sorted, padded to a fixed size.
    /// Padding positions hold a sentinel (T-1 reuse, mask handles it).
    ///
    /// - Returns: (gather [nKVHeads, K_padded] int32 with sentinel padding,
    ///   K_padded). Gather is SORTED per row; duplicates within a row are
    ///   marked via adjacent-diff mask at SDPA time.
    public func perKVHeadGatherGPU(
        projectedQ: MLXArray, seqLen T: Int
    ) -> (MLXArray, Int) {
        let staticInit = config.staticInit
        let slidingWindow = config.slidingWindow
        let fineBS = config.fineBlockSize
        let coarseBS = config.coarseBlockSize
        let kFineEff = min(config.effectiveFineTopK(seqLen: seqLen),
                           (fineBlockFeatures?.dim(1) ?? 0))
        let kCoarseEff = config.coarseRescueEnabled
            ? min(config.coarseTopK, (coarseBlockFeatures?.dim(1) ?? 0)) : 0
        let staticEnd = min(staticInit, T)
        let slidingStart = max(0, T - slidingWindow)
        let staticCount = staticEnd
        let slidingCount = max(0, T - slidingStart)
        // Per-head total = static + sliding + topK_fine*bs + topK_coarse*coarseBS.
        let kPadded = staticCount + slidingCount + kFineEff * fineBS + kCoarseEff * coarseBS

        // Build per-head rows in MLX. Each row is [staticRange, slidingRange,
        // fine_expanded[h], coarse_expanded[h]] all on GPU.
        let staticPositions = staticCount > 0
            ? MLXArray(Int32(0)..<Int32(staticEnd))
            : MLXArray.zeros([0], dtype: .int32)  // [staticCount]
        let slidingPositions = slidingCount > 0
            ? MLXArray(Int32(slidingStart)..<Int32(T))
            : MLXArray.zeros([0], dtype: .int32)
        // Broadcast static + sliding across nKVHeads (same rows).
        let staticPlus = concatenated([staticPositions, slidingPositions], axis: 0)
            .reshaped(1, staticCount + slidingCount)
        let staticBroadcast = broadcast(staticPlus, to: [nKVHeads, staticCount + slidingCount])

        var pieces: [MLXArray] = [staticBroadcast]

        if kFineEff > 0, let ff = fineBlockFeatures {
            let fineStarts = computeTopKBlockStarts(
                features: ff, projectedQ: projectedQ,
                k: kFineEff, blockSize: fineBS
            )  // [nKVH, kFineEff] block starts in token coords
            let offsets = MLXArray(0..<Int32(fineBS)).reshaped(1, 1, fineBS)
            let expanded = fineStarts.expandedDimensions(axis: 2) + offsets
            // [nKVH, kFineEff * fineBS]
            pieces.append(expanded.reshaped(nKVHeads, kFineEff * fineBS))
        }
        if kCoarseEff > 0, let cf = coarseBlockFeatures {
            let coarseStarts = computeTopKBlockStarts(
                features: cf, projectedQ: projectedQ,
                k: kCoarseEff, blockSize: coarseBS
            )
            let offsets = MLXArray(0..<Int32(coarseBS)).reshaped(1, 1, coarseBS)
            let expanded = coarseStarts.expandedDimensions(axis: 2) + offsets
            pieces.append(expanded.reshaped(nKVHeads, kCoarseEff * coarseBS))
        }

        let unsorted = concatenated(pieces, axis: 1)  // [nKVHeads, kPadded]
        // Clip to valid token range — guards against the last-block tail
        // overshooting T.
        let clipped = clip(unsorted, min: Int32(0), max: Int32(T - 1))
        // Sort per row → enables adjacent-diff dedupe at SDPA time.
        let sortedGather = sorted(clipped, axis: 1)
        return (sortedGather, kPadded)
    }

    /// F-59 path: return expanded per-token positions for fine + coarse
    /// top-K, ENTIRELY ON GPU (no asArray sync). Used by the
    /// mask-not-gather attention path.
    ///
    /// - Returns: A 1-D MLXArray of int32 positions in [0, seqLen). May
    ///   contain duplicates across heads and overlap with the
    ///   static/sliding regions — caller dedupes via mask scatter
    ///   (idempotent for "set to 0").
    public func expandedTopKPositionsGPU(projectedQ: MLXArray) -> MLXArray {
        var sources: [MLXArray] = []
        // Fine.
        if let ff = fineBlockFeatures {
            let kFine = min(config.effectiveFineTopK(seqLen: seqLen), ff.dim(1))
            if kFine > 0 {
                let fineStarts = computeTopKBlockStarts(
                    features: ff, projectedQ: projectedQ,
                    k: kFine, blockSize: config.fineBlockSize
                )  // [nKVH, kFine]
                // Expand each block start to blockSize positions.
                let bs = config.fineBlockSize
                let offsets = MLXArray(0..<Int32(bs)).reshaped(1, 1, bs)
                let expanded = fineStarts.expandedDimensions(axis: 2) + offsets
                sources.append(expanded.reshaped(-1))
            }
        }
        // Coarse.
        if config.coarseRescueEnabled, let cf = coarseBlockFeatures {
            let kCoarse = min(config.coarseTopK, cf.dim(1))
            if kCoarse > 0 {
                let coarseStarts = computeTopKBlockStarts(
                    features: cf, projectedQ: projectedQ,
                    k: kCoarse, blockSize: config.coarseBlockSize
                )
                let bs = config.coarseBlockSize
                let offsets = MLXArray(0..<Int32(bs)).reshaped(1, 1, bs)
                let expanded = coarseStarts.expandedDimensions(axis: 2) + offsets
                sources.append(expanded.reshaped(-1))
            }
        }
        if sources.isEmpty {
            return MLXArray.zeros([0], dtype: .int32)
        }
        return concatenated(sources, axis: 0)
    }

    /// Combined fine + coarse topK in ONE asArray sync. Both topK MLX op
    /// chains stay queued, get concat'd into a single [nKVHeads, kFine +
    /// kCoarse] tensor, and one CPU↔GPU sync pulls all the indices.
    ///
    /// F-55: fused Metal kernel replaces the score+argPartition+slice+mul
    /// op chain with a single dispatch when nBlocks ≤ 1024 (the kernel's
    /// shared-memory cap). Otherwise falls back to the MLX-ops path
    /// (topKBlockStartsMLX).
    /// F-73 GPU-tensor variant of `topKBlockStartsAllHeadsCombined`.
    /// Returns the per-KV-head top-K block starts as MLXArrays
    /// (`[nKVH, K_fine]` and `[nKVH, K_coarse]`) — no asArray sync.
    /// Feeds the fused build-mask kernel.
    public func topKBlockStartsAllHeadsCombinedGPU(
        projectedQ: MLXArray
    ) -> (fine: MLXArray, coarse: MLXArray) {
        guard let fineFeatures = fineBlockFeatures else {
            return (MLXArray.zeros([nKVHeads, 1], dtype: .int32),
                    MLXArray.zeros([nKVHeads, 1], dtype: .int32))
        }
        let fineN = fineFeatures.dim(1)
        let kFine = min(config.effectiveFineTopK(seqLen: seqLen), fineN)
        guard kFine > 0 else {
            return (MLXArray.zeros([nKVHeads, 1], dtype: .int32),
                    MLXArray.zeros([nKVHeads, 1], dtype: .int32))
        }
        let fineStarts = computeTopKBlockStarts(
            features: fineFeatures, projectedQ: projectedQ,
            k: kFine, blockSize: config.fineBlockSize
        )
        let coarse: MLXArray
        if config.coarseRescueEnabled, let coarseFeatures = coarseBlockFeatures {
            let kCoarse = min(config.coarseTopK, coarseFeatures.dim(1))
            if kCoarse > 0 {
                coarse = computeTopKBlockStarts(
                    features: coarseFeatures, projectedQ: projectedQ,
                    k: kCoarse, blockSize: config.coarseBlockSize
                )
            } else {
                coarse = MLXArray.zeros([nKVHeads, 1], dtype: .int32)
            }
        } else {
            coarse = MLXArray.zeros([nKVHeads, 1], dtype: .int32)
        }
        return (fine: fineStarts, coarse: coarse)
    }

    public func topKBlockStartsAllHeadsCombined(
        projectedQ: MLXArray
    ) -> (fine: [[Int]], coarse: [[Int]]) {
        // Fine path.
        guard let fineFeatures = fineBlockFeatures else {
            return (Array(repeating: [], count: nKVHeads), Array(repeating: [], count: nKVHeads))
        }
        let fineN = fineFeatures.dim(1)
        let kFine = min(config.effectiveFineTopK(seqLen: seqLen), fineN)
        guard kFine > 0 else {
            return (Array(repeating: [], count: nKVHeads), Array(repeating: [], count: nKVHeads))
        }
        let fineStarts = computeTopKBlockStarts(
            features: fineFeatures, projectedQ: projectedQ,
            k: kFine, blockSize: config.fineBlockSize
        )

        // Coarse path (optional).
        let hasCoarse = config.coarseRescueEnabled && coarseBlockFeatures != nil
        let combined: MLXArray
        var kCoarse = 0
        if hasCoarse {
            let coarseFeatures = coarseBlockFeatures!
            kCoarse = min(config.coarseTopK, coarseFeatures.dim(1))
            if kCoarse > 0 {
                let coarseStarts = computeTopKBlockStarts(
                    features: coarseFeatures, projectedQ: projectedQ,
                    k: kCoarse, blockSize: config.coarseBlockSize
                )
                combined = concatenated([fineStarts, coarseStarts], axis: -1)
            } else {
                combined = fineStarts
            }
        } else {
            combined = fineStarts
        }

        // ONE sync.
        let cpu = combined.asArray(Int32.self)
        let stride = kFine + kCoarse
        var fine: [[Int]] = []
        var coarse: [[Int]] = []
        fine.reserveCapacity(nKVHeads)
        coarse.reserveCapacity(nKVHeads)
        for h in 0..<nKVHeads {
            let base = h * stride
            fine.append((0..<kFine).map { Int(cpu[base + $0]) })
            if kCoarse > 0 {
                coarse.append((0..<kCoarse).map { Int(cpu[base + kFine + $0]) })
            } else {
                coarse.append([])
            }
        }
        return (fine: fine, coarse: coarse)
    }

    /// Compute the top-K block starts using the fused Metal kernel when
    /// nBlocks fits within the kernel's threadgroup cap (1024), else fall
    /// back to the score+argPartition+slice+multiply MLX op chain.
    private func computeTopKBlockStarts(
        features: MLXArray, projectedQ: MLXArray, k: Int, blockSize: Int
    ) -> MLXArray {
        let nBlocks = features.dim(1)
        // Need a power-of-2 ≥ nBlocks for the tree reduction.
        let nBlocksPow2 = Self.nextPowerOf2(nBlocks)
        if nBlocksPow2 <= 1024 && nBlocks == nBlocksPow2 {
            // Fused kernel — fits in one threadgroup.
            return retrievalAttentionScoreTopKFused(
                blockFeatures: features, projectedQ: projectedQ,
                k: k, blockSize: blockSize, nBlocksRounded: nBlocksPow2
            )
        }
        // Fallback: MLX ops (score + argPartition + slice + multiply).
        let scores = scoreBlocksBatched(features: features, projectedQ: projectedQ)
        return topKBlockStartsMLX(scores: scores, k: k, blockSize: blockSize)
    }

    private static func nextPowerOf2(_ x: Int) -> Int {
        guard x > 1 else { return 1 }
        var p = 1
        while p < x { p <<= 1 }
        return p
    }

    /// Fallback: build the `[nKVHeads, k]` topK block-starts via MLX ops.
    private func topKBlockStartsMLX(
        scores: MLXArray, k: Int, blockSize: Int
    ) -> MLXArray {
        let nBlocks = scores.dim(1)
        let pivotKth = nBlocks - k
        let partitioned: MLXArray
        if pivotKth <= 0 {
            let rangeArr = MLXArray(0..<Int32(nBlocks)).reshaped(1, nBlocks)
            partitioned = broadcast(rangeArr, to: [nKVHeads, nBlocks])
        } else {
            partitioned = argPartition(scores, kth: pivotKth, axis: -1)
        }
        let topKIdx = partitioned[0..., (nBlocks - k)...]
        return topKIdx * Int32(blockSize)
    }

    // MARK: - F-83 sparse-prefill API (M1: batched-Q union top-K)

    /// Project a chunk of queries (L > 1) for the selector.
    ///
    /// - Parameter q: `[nKVHeads, L, dHead]` post-RoPE queries.
    /// - Returns: `[nKVHeads, L, effectiveSelectorDim]`.
    ///
    /// Trig features use relativePosition=0 broadcast over L — matches
    /// the decode path's convention (the trig branch is positional bias
    /// in the *block* features; queries already had RoPE applied).
    public func projectQueriesBatchedL(_ q: MLXArray) -> MLXArray {
        precondition(q.shape.count == 3, "expected [nKVHeads, L, dHead], got \(q.shape)")
        precondition(q.dim(0) == nKVHeads && q.dim(2) == dHead,
            "expected [\(nKVHeads), L, \(dHead)], got \(q.shape)")
        let L = q.dim(1)
        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(dHead: dHead, config: config)
            jlMatrixT = jlMatrix!.transposed(1, 0)
            eval(jlMatrix!, jlMatrixT!)
        }
        let WT = jlMatrixT!
        // [nh, L, dHead] @ [dHead, contentDim] = [nh, L, contentDim]
        let contentQ = matmul(q.asType(.float32), WT)
        guard config.usesTrigFeatures else { return contentQ }
        let trig1 = retrievalAttentionTrigFeatures(
            relativePositions: MLXArray([Int32(0)]), base: ropeBase, config: config
        ).reshaped(1, 1, config.trigDim)
        let trigQ = broadcast(trig1, to: [nKVHeads, L, config.trigDim])
        return concatenated([contentQ, trigQ], axis: -1)
    }

    /// Union top-K block starts across a chunk of L queries per KV head.
    ///
    /// For each KV head, a block's score is the *max* over the L queries
    /// of `feature · projectedQ_l`. Top-K is taken on that max-pooled
    /// score. This matches the NSA GQA pattern: queries within a chunk
    /// in the same KV head all select the same blocks. Saves L× selector
    /// cost vs per-query at the cost of selecting positions that "any
    /// query wanted" rather than "this query wanted".
    ///
    /// - Parameter projectedQ: `[nKVHeads, L, effectiveSelectorDim]` —
    ///   typically from `projectQueriesBatchedL`.
    /// - Returns: `(fine: [nKVHeads, K_fine], coarse: [nKVHeads, K_coarse])`
    ///   GPU tensors of block start positions (token indices). Shape
    ///   matches `topKBlockStartsAllHeadsCombinedGPU` so the downstream
    ///   mask/bitmap builder can ingest either.
    public func topKBlockStartsUnionBatchedQGPU(
        projectedQ: MLXArray,
        fineTopKOverride: Int? = nil,
        coarseTopKOverride: Int? = nil
    ) -> (fine: MLXArray, coarse: MLXArray) {
        guard let fineFeatures = fineBlockFeatures else {
            return (MLXArray.zeros([nKVHeads, 1], dtype: .int32),
                    MLXArray.zeros([nKVHeads, 1], dtype: .int32))
        }
        let fineN = fineFeatures.dim(1)
        let kFineRequested = fineTopKOverride ?? config.effectiveFineTopK(seqLen: seqLen)
        let kFine = min(kFineRequested, fineN)
        guard kFine > 0 else {
            return (MLXArray.zeros([nKVHeads, 1], dtype: .int32),
                    MLXArray.zeros([nKVHeads, 1], dtype: .int32))
        }
        let fineStarts = unionTopKBlockStarts(
            features: fineFeatures, projectedQ: projectedQ,
            k: kFine, blockSize: config.fineBlockSize
        )
        let coarse: MLXArray
        if config.coarseRescueEnabled, let coarseFeatures = coarseBlockFeatures {
            let kCoarseRequested = coarseTopKOverride ?? config.coarseTopK
            let kCoarse = min(kCoarseRequested, coarseFeatures.dim(1))
            if kCoarse > 0 {
                coarse = unionTopKBlockStarts(
                    features: coarseFeatures, projectedQ: projectedQ,
                    k: kCoarse, blockSize: config.coarseBlockSize
                )
            } else {
                coarse = MLXArray.zeros([nKVHeads, 1], dtype: .int32)
            }
        } else {
            coarse = MLXArray.zeros([nKVHeads, 1], dtype: .int32)
        }
        return (fine: fineStarts, coarse: coarse)
    }

    /// F-83 V1.1 — cross-head + cross-L union top-K, with score-mask
    /// exclusion of static-prefix and sliding-window block ranges.
    ///
    /// Returns a single `[K]` block-start list (NOT [H, K]) — eliminates
    /// the cross-head duplicate sources that forced the V1 CPU dedupe.
    /// The score is `max over (h, l) of features[h, b, :] · projectedQ[h, l, :]`,
    /// i.e., a block scores high if ANY (head, query) pair wants it.
    ///
    /// The returned block starts are guaranteed to be outside the
    /// `[0, staticEnd)` and `[slidingStart, priorLen)` ranges, so when
    /// the caller concats static-prefix + sliding-window + fine positions
    /// into the gather list, no duplicates can arise.
    ///
    /// - Parameters:
    ///   - projectedQ: `[H, L, D_eff]`
    ///   - k: fine top-K (e.g. 16 per PRD revision 3)
    ///   - blockSize: fine block size
    ///   - staticEnd: exclude blocks fully inside `[0, staticEnd)`
    ///   - slidingStart: exclude blocks fully inside `[slidingStart, ∞)`
    /// - Returns: `[K]` int32 block-start positions, all in
    ///   `[staticAlignedEnd, slidingAlignedStart)`.
    public func crossHeadUnionTopKExcludingRangesGPU(
        projectedQ: MLXArray,
        k: Int,
        blockSize: Int,
        staticEnd: Int,
        slidingStart: Int
    ) -> MLXArray {
        let features: MLXArray
        if blockSize == config.fineBlockSize {
            guard let f = fineBlockFeatures else {
                return MLXArray.zeros([0], dtype: .int32)
            }
            features = f
        } else if blockSize == config.coarseBlockSize {
            guard let f = coarseBlockFeatures else {
                return MLXArray.zeros([0], dtype: .int32)
            }
            features = f
        } else {
            fatalError("unsupported blockSize \(blockSize)")
        }
        let nBlocks = features.dim(1)
        let staticBlocks = (staticEnd + blockSize - 1) / blockSize  // CEIL
        let slidBlock = slidingStart / blockSize                    // FLOOR
        // Available range = [staticBlocks, slidBlock). If empty (sliding
        // reaches into static or they touch), there are no fine blocks
        // to pick — return empty and let the caller cover everything
        // via static+sliding.
        let availableBlocks = Swift.max(0, slidBlock - staticBlocks)
        let take = Swift.min(k, availableBlocks)
        if take <= 0 {
            return MLXArray.zeros([0], dtype: .int32)
        }
        // Score per (H, B, L) via batched matmul, then max-pool over (L, H).
        let perHQL = matmul(features, projectedQ.transposed(0, 2, 1))
        let unionH = perHQL.max(axis: -1)        // [H, B]
        let scores = unionH.max(axis: 0)          // [B]
        // Mask scores in static/sliding ranges to -inf so top-K never picks them.
        let blockIdx = MLXArray(0..<Int32(nBlocks))
        let isStatic = blockIdx .< Int32(staticBlocks)
        let isSliding = blockIdx .>= Int32(slidBlock)
        let isCovered = isStatic .|| isSliding
        let neginf = MLXArray(-Float.infinity)
        let maskedScores = MLX.where(isCovered, neginf, scores)
        // argpartition top-K — `take` is now bounded by availableBlocks
        // so even when masked-all-but-some, we never pick a -inf-scored
        // index that would lie in the static/sliding range.
        let pivot = nBlocks - take
        let partitioned: MLXArray
        if pivot <= 0 {
            partitioned = MLXArray(0..<Int32(nBlocks))
        } else {
            partitioned = argPartition(maskedScores, kth: pivot, axis: -1)
        }
        let topKIdx = partitioned[(nBlocks - take)...]
        return (topKIdx * Int32(blockSize)).asType(.int32)
    }

    /// Internal: per-block max score across L queries → top-K block starts.
    /// Honors the same lambda blend as `scoreBlocksBatched`.
    private func unionTopKBlockStarts(
        features: MLXArray,   // [H, B, D_eff]
        projectedQ: MLXArray, // [H, L, D_eff]
        k: Int, blockSize: Int
    ) -> MLXArray {
        let nBlocks = features.dim(1)
        let take = min(k, nBlocks)
        // perQ[h, b, l] = features[h, b, :] · projectedQ[h, l, :]
        // Computed via batched matmul on the last dim. We deliberately
        // do NOT materialize an [H, L, B, D] elementwise tensor — for
        // H=8, L=1024, B=4096, D=64 that would be 8 GB.
        let unionScores: MLXArray
        if !config.usesTrigFeatures {
            // [H, B, D] @ [H, D, L] = [H, B, L]
            let perQ = matmul(features, projectedQ.transposed(0, 2, 1))
            unionScores = perQ.max(axis: -1)  // [H, B]
        } else {
            let cD = config.contentDim
            let lambdaPos = config.lambdaPos
            let contentF = features[0..., 0..., ..<cD]
            let trigF = features[0..., 0..., cD...]
            let contentQ = projectedQ[0..., 0..., ..<cD]
            let trigQ = projectedQ[0..., 0..., cD...]
            let tPart = matmul(trigF, trigQ.transposed(0, 2, 1))   // [H, B, L]
            let perQ: MLXArray
            if lambdaPos == 1.0 {
                perQ = tPart
            } else {
                let cPart = matmul(contentF, contentQ.transposed(0, 2, 1))
                perQ = (1 - lambdaPos) * cPart + lambdaPos * tPart
            }
            unionScores = perQ.max(axis: -1)
        }
        // Argpartition top-K — same shape contract as the decode-path
        // `computeTopKBlockStarts` MLX fallback so downstream code can
        // ingest either output identically.
        let pivotKth = nBlocks - take
        let partitioned: MLXArray
        if pivotKth <= 0 {
            let rangeArr = MLXArray(0..<Int32(nBlocks)).reshaped(1, nBlocks)
            partitioned = broadcast(rangeArr, to: [nKVHeads, nBlocks])
        } else {
            partitioned = argPartition(unionScores, kth: pivotKth, axis: -1)
        }
        let topKIdx = partitioned[0..., (nBlocks - take)...]
        return topKIdx * Int32(blockSize)
    }
}
