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

    private(set) public var perTokenFeatures: MLXArray?
    private(set) public var fineBlockFeatures: MLXArray?
    private(set) public var coarseBlockFeatures: MLXArray?

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

    public var seqLen: Int { perTokenFeatures?.dim(1) ?? 0 }

    /// Update with new K rows for all heads in one call.
    ///
    /// - Parameter newKeys: `[nKVHeads, L, dHead]` post-RoPE keys.
    public func update(newKeys: MLXArray) {
        precondition(newKeys.shape.count == 3, "expected [nKVHeads, L, dHead], got \(newKeys.shape)")
        precondition(newKeys.dim(0) == nKVHeads, "head count mismatch")
        precondition(newKeys.dim(2) == dHead, "dHead mismatch")

        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(dHead: dHead, config: config)
        }
        let W = jlMatrix!  // [contentDim, dHead]

        let oldSeqLen = seqLen
        let L = newKeys.dim(1)
        let newSeqLen = oldSeqLen + L

        // Project: [nKVHeads, L, dHead] @ [dHead, contentDim] = [nKVHeads, L, contentDim]
        let contentNew = matmul(newKeys, W.transposed(1, 0)).asType(.float32)

        // Trig features for the new positions — same across heads.
        let positions = MLXArray((Int32(oldSeqLen)..<Int32(newSeqLen))).asType(.int32)
        let trigNew = retrievalAttentionTrigFeatures(
            relativePositions: positions, base: ropeBase, config: config
        )  // [L, trigDim]
        // Broadcast trig to [nKVHeads, L, trigDim]
        let trigBroadcast = broadcast(
            trigNew.reshaped(1, L, config.trigDim),
            to: [nKVHeads, L, config.trigDim]
        )

        // Concat content + trig along feature dim → [nKVHeads, L, selectorDim]
        let selectorNew = concatenated([contentNew, trigBroadcast], axis: -1)

        if let existing = perTokenFeatures {
            perTokenFeatures = concatenated([existing, selectorNew], axis: 1)
        } else {
            perTokenFeatures = selectorNew
        }

        // Block-pooled features. Incremental for L=1, full re-pool otherwise.
        if L == 1 && fineBlockFeatures != nil {
            updateBlockTailIncremental(blockSize: config.fineBlockSize, oldSeqLen: oldSeqLen, isFine: true)
            if config.coarseRescueEnabled && coarseBlockFeatures != nil {
                updateBlockTailIncremental(blockSize: config.coarseBlockSize, oldSeqLen: oldSeqLen, isFine: false)
            }
        } else {
            fineBlockFeatures = batchedBlockMeanPool(
                perTokenFeatures!, blockSize: config.fineBlockSize
            )
            if config.coarseRescueEnabled {
                coarseBlockFeatures = batchedBlockMeanPool(
                    perTokenFeatures!, blockSize: config.coarseBlockSize
                )
            }
        }

        // Materialize the index state to break the lazy graph chain.
        // Without this, every decode step accumulates the deferred graph of
        // prior updates, and asArray() in topK has to walk back to prefill.
        // Mirrors StandardKVCache's eval() pattern on buffer resize.
        var toEval: [MLXArray] = [perTokenFeatures!, fineBlockFeatures!]
        if let coarse = coarseBlockFeatures { toEval.append(coarse) }
        eval(toEval)
    }

    /// Mean-pool a `[nKVHeads, T, selectorDim]` array into blocks of
    /// `blockSize`, returning `[nKVHeads, nBlocks, selectorDim]`. Last
    /// block is mean over `≤ blockSize` rows.
    private func batchedBlockMeanPool(_ features: MLXArray, blockSize: Int) -> MLXArray {
        let nh = features.dim(0)
        let T = features.dim(1)
        let D = features.dim(2)
        let nBlocks = (T + blockSize - 1) / blockSize
        let paddedT = nBlocks * blockSize
        let padCount = paddedT - T
        let working: MLXArray
        if padCount == 0 {
            working = features
        } else {
            let pad = MLXArray.zeros([nh, padCount, D], dtype: features.dtype)
            working = concatenated([features, pad], axis: 1)
        }
        // [nh, nBlocks, blockSize, D] → mean over axis 2
        let pooled = working.reshaped(nh, nBlocks, blockSize, D).mean(axis: 2)
        if padCount == 0 { return pooled }
        let tail = T - (nBlocks - 1) * blockSize
        let correction = Float(blockSize) / Float(tail)
        var oneNB = Array(repeating: Float(1.0), count: nBlocks)
        oneNB[nBlocks - 1] = correction
        let mask = MLXArray(oneNB).reshaped(1, nBlocks, 1).asType(pooled.dtype)
        return pooled * mask
    }

    private func updateBlockTailIncremental(blockSize: Int, oldSeqLen: Int, isFine: Bool) {
        let pooled = isFine ? fineBlockFeatures! : coarseBlockFeatures!
        let priorBlockCount = pooled.dim(1)
        let lastBlockIdx = oldSeqLen / blockSize
        let lastBlockStart = lastBlockIdx * blockSize
        // [nh, tailLen, D]
        let tail = perTokenFeatures![0..., lastBlockStart..., 0...]
        // mean over axis 1 (token axis) → [nh, D] then reshape → [nh, 1, D]
        let tailMean = tail.mean(axis: 1).reshaped(nKVHeads, 1, pooled.dim(2))
        if priorBlockCount == lastBlockIdx + 1 {
            // Same block — replace last row along axis 1.
            if priorBlockCount == 1 {
                if isFine { fineBlockFeatures = tailMean } else { coarseBlockFeatures = tailMean }
            } else {
                let head = pooled[0..., ..<(priorBlockCount - 1), 0...]
                let merged = concatenated([head, tailMean], axis: 1)
                if isFine { fineBlockFeatures = merged } else { coarseBlockFeatures = merged }
            }
        } else if priorBlockCount == lastBlockIdx {
            let merged = concatenated([pooled, tailMean], axis: 1)
            if isFine { fineBlockFeatures = merged } else { coarseBlockFeatures = merged }
        } else {
            let full = batchedBlockMeanPool(perTokenFeatures!, blockSize: blockSize)
            if isFine { fineBlockFeatures = full } else { coarseBlockFeatures = full }
        }
    }

    /// Project the query for all heads in one go.
    /// - Parameter q: `[nKVHeads, dHead]`
    /// - Returns: `[nKVHeads, selectorDim]`
    public func projectQueriesBatched(_ q: MLXArray) -> MLXArray {
        precondition(q.shape == [nKVHeads, dHead], "expected [\(nKVHeads), \(dHead)], got \(q.shape)")
        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(dHead: dHead, config: config)
        }
        let W = jlMatrix!  // [contentDim, dHead]
        // [nh, dHead] @ [dHead, contentDim] = [nh, contentDim]
        let contentQ = matmul(q.asType(.float32), W.transposed(1, 0))
        // Trig at relative_pos=0, broadcast across heads.
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
        // features: [nh, nBlocks, selectorDim], q: [nh, selectorDim]
        // We want [nh, nBlocks] = sum over selectorDim of features * q.expand
        let qExp = projectedQ.reshaped(nKVHeads, 1, config.selectorDim)
        let lambdaPos = config.lambdaPos
        if lambdaPos == 0.0 {
            // Pure content. Sum over contentDim only.
            let cD = config.contentDim
            let prod = features[0..., 0..., ..<cD] * qExp[0..., 0..., ..<cD]
            return prod.sum(axis: -1)
        } else if lambdaPos == 1.0 {
            let cD = config.contentDim
            let prod = features[0..., 0..., cD...] * qExp[0..., 0..., cD...]
            return prod.sum(axis: -1)
        } else {
            // Mixture.
            let cD = config.contentDim
            let content = (features[0..., 0..., ..<cD] * qExp[0..., 0..., ..<cD]).sum(axis: -1)
            let trig = (features[0..., 0..., cD...] * qExp[0..., 0..., cD...]).sum(axis: -1)
            return (1 - lambdaPos) * content + lambdaPos * trig
        }
    }

    public func scoreCoarseBlocksBatched(projectedQ: MLXArray) -> MLXArray? {
        guard fineBlockFeatures != nil, let features = coarseBlockFeatures else {
            return nil
        }
        let qExp = projectedQ.reshaped(nKVHeads, 1, config.selectorDim)
        let lambdaPos = config.lambdaPos
        let cD = config.contentDim
        if lambdaPos == 0.0 {
            let prod = features[0..., 0..., ..<cD] * qExp[0..., 0..., ..<cD]
            return prod.sum(axis: -1)
        } else if lambdaPos == 1.0 {
            let prod = features[0..., 0..., cD...] * qExp[0..., 0..., cD...]
            return prod.sum(axis: -1)
        } else {
            let content = (features[0..., 0..., ..<cD] * qExp[0..., 0..., ..<cD]).sum(axis: -1)
            let trig = (features[0..., 0..., cD...] * qExp[0..., 0..., cD...]).sum(axis: -1)
            return (1 - lambdaPos) * content + lambdaPos * trig
        }
    }

    /// Top-k fine block starts per head. One asArray sync; CPU partial sort.
    public func topKFineBlockStartsAllHeads(projectedQ: MLXArray) -> [[Int]] {
        let scores = scoreFineBlocksBatched(projectedQ: projectedQ)  // [nh, nBlocks]
        let k = config.effectiveFineTopK(seqLen: seqLen)
        let arr = scores.asArray(Float.self)  // contiguous [nh*nBlocks]
        let nBlocks = scores.dim(1)
        var result: [[Int]] = []
        result.reserveCapacity(nKVHeads)
        for h in 0..<nKVHeads {
            let base = h * nBlocks
            let take = min(k, nBlocks)
            var indexed = (0..<nBlocks).map { ($0, arr[base + $0]) }
            indexed.sort { $0.1 > $1.1 }
            let starts = indexed.prefix(take).map { $0.0 * config.fineBlockSize }
            result.append(starts)
        }
        return result
    }

    public func topKCoarseBlockStartsAllHeads(projectedQ: MLXArray) -> [[Int]] {
        guard let scores = scoreCoarseBlocksBatched(projectedQ: projectedQ) else {
            return Array(repeating: [], count: nKVHeads)
        }
        let nBlocks = scores.dim(1)
        let arr = scores.asArray(Float.self)
        let k = config.coarseTopK
        var result: [[Int]] = []
        result.reserveCapacity(nKVHeads)
        for h in 0..<nKVHeads {
            let base = h * nBlocks
            let take = min(k, nBlocks)
            var indexed = (0..<nBlocks).map { ($0, arr[base + $0]) }
            indexed.sort { $0.1 > $1.1 }
            let starts = indexed.prefix(take).map { $0.0 * config.coarseBlockSize }
            result.append(starts)
        }
        return result
    }
}
