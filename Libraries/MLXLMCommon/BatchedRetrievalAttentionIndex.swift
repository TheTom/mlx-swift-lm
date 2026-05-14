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
        let W = jlMatrix!
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
            // Grow buffer in chunkSize-multiples.
            let neededTotal = ((newSeqLen + Self.featureChunkSize - 1)
                / Self.featureChunkSize) * Self.featureChunkSize
            let additional = neededTotal - perTokenFeatures!.dim(1)
            let pad = MLXArray.zeros(
                [nKVHeads, additional, featureDim], dtype: perTokenFeatures!.dtype
            )
            perTokenFeatures = concatenated([perTokenFeatures!, pad], axis: 1)
            // Materialize after grow to break graph chain.
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
        let chunkSize = Self.blockBufferChunkSize
        let newCap = max(chunkSize, ((needed + chunkSize - 1) / chunkSize) * chunkSize)
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
        let W = jlMatrix!
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
}
