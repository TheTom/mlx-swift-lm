// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// BatchedRetrievalAttentionIndexB — selector index with an EXPLICIT
// per-request batch dimension (B). Sister to
// `BatchedRetrievalAttentionIndex` (which batches across KV heads only
// at B=1). F-85 builds this for batched sparse decode (B>1) so each
// request gets its own per-(KV-head) top-K block selection.
//
// Caveman comments: ugg need B at front. all shapes [B, H, ...] now.
// no shared selector across batch — each prompt picks its own blocks.
//
// Shape conventions:
//   fineBlockFeatures:    [B, nKVHeads, nFineBlocks, contentDim]
//   coarseBlockFeatures:  [B, nKVHeads, nCoarseBlocks, contentDim]
//   q (input to project): [B, nKVHeads, dHead]
//   projectedQ:           [B, nKVHeads, contentDim]
//   topK output:          [B, nKVHeads, K_fine]  / [B, nKVHeads, K_coarse]
//
// MEMORY DECISION (per F85_BATCHED_SPARSE_DESIGN.md risk #2):
// Drops `perTokenFeatures` storage entirely. The original
// `BatchedRetrievalAttentionIndex` kept it for re-pooling on every block
// boundary cross; here we maintain block features INCREMENTALLY using a
// running mean. Saves ~20 GB at B=8 T=128K on Qwen2.5-14B-1M.
//
// Running-mean math (per block b, contentDim c):
//   mean_after = (old_mean * old_count + sum_new_in_block) / new_count
// Cheap to update as each new K row arrives — no scan over T.

import Foundation
import MLX

public final class BatchedRetrievalAttentionIndexB {

    public let config: RetrievalAttentionConfig
    public let B: Int
    public let dHead: Int
    public let nKVHeads: Int
    public let ropeBase: Float
    public let layerIdx: Int

    /// JL projection matrix `[dHead, contentDim]`. Shared across all
    /// (B, KV-head) — JL is a model-architecture property, not per-slot.
    private var jlMatrixT: MLXArray?
    public var jlW: MLXArray? { jlMatrixT }

    /// Block-pooled features, `[B, nKVH, blockCap, contentDim]` int16/fp16.
    /// Grown geometrically. Valid prefix is `[0..<currentFineBlocks]`.
    /// Only contentDim — trig features are skipped (matches λ=0 ship config).
    private var fineBlockBuffer: MLXArray?
    private var coarseBlockBuffer: MLXArray?
    private var fineBlockCapacity: Int = 0
    private var coarseBlockCapacity: Int = 0

    /// Per (B, KV-head) running sum buffers for partial-block accumulation.
    /// Tracks the "in-progress" last block so adding new tokens just bumps
    /// the running sum + count and re-divides — no re-pool over T.
    /// `runningSums: [B, nKVH, contentDim]` fp32.
    private var fineRunningSums: MLXArray?
    private var coarseRunningSums: MLXArray?
    /// Per (B, KV-head) count of tokens in the current partial block.
    /// Plain Swift array since it's tiny + we modulo against blockSize.
    private var fineTokensInCurrentBlock: [[Int]]   // [B][nKVH]
    private var coarseTokensInCurrentBlock: [[Int]]

    /// Per-slot populated count. Decode/prefill can advance independently
    /// (e.g. continuous batching) so each slot tracks its own T.
    private var populatedPerSlot: [Int]
    /// Convenience: max-populated across slots (drives block-buffer capacity).
    public var seqLenMax: Int { populatedPerSlot.max() ?? 0 }
    /// Convenience: min populated (for rectangular T assumption check).
    public var seqLenMin: Int { populatedPerSlot.min() ?? 0 }
    /// Per-slot getter.
    public func seqLen(slot: Int) -> Int { populatedPerSlot[slot] }

    /// Block counts driven by `seqLenMax` — buffer holds enough for the
    /// "tallest" slot, shorter slots have zeros at the tail (don't hurt
    /// top-K because scores at zero-features are ~0, but we MASK them
    /// to -inf in the scoring path).
    public var currentFineBlocks: Int {
        let m = seqLenMax
        guard m > 0 else { return 0 }
        return (m + config.fineBlockSize - 1) / config.fineBlockSize
    }
    public var currentCoarseBlocks: Int {
        let m = seqLenMax
        guard m > 0 else { return 0 }
        return (m + config.coarseBlockSize - 1) / config.coarseBlockSize
    }

    /// Public views (valid prefix only).
    public var fineBlockFeatures: MLXArray? {
        guard let buf = fineBlockBuffer, currentFineBlocks > 0 else { return nil }
        return buf[0..., 0..., ..<currentFineBlocks, 0...]
    }
    public var coarseBlockFeatures: MLXArray? {
        guard let buf = coarseBlockBuffer, currentCoarseBlocks > 0 else { return nil }
        return buf[0..., 0..., ..<currentCoarseBlocks, 0...]
    }

    public init(
        config: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        B: Int,
        dHead: Int,
        nKVHeads: Int,
        ropeBase: Float = 10_000.0,
        layerIdx: Int
    ) {
        precondition(B > 0, "B must be > 0")
        self.config = config
        self.B = B
        self.dHead = dHead
        self.nKVHeads = nKVHeads
        self.ropeBase = ropeBase
        self.layerIdx = layerIdx
        self.populatedPerSlot = Array(repeating: 0, count: B)
        self.fineTokensInCurrentBlock =
            Array(repeating: Array(repeating: 0, count: nKVHeads), count: B)
        self.coarseTokensInCurrentBlock =
            Array(repeating: Array(repeating: 0, count: nKVHeads), count: B)
        // NOTE: usesTrigFeatures path NOT supported here in v1. Bench config
        // ships λ=0 so this is fine for F-85; revisit if a customer flips
        // lambdaPos > 0.
        precondition(!config.usesTrigFeatures,
            "BatchedRetrievalAttentionIndexB v1 supports λ=0 (no trig) only")
    }

    private func ensureJL() {
        guard jlMatrixT == nil else { return }
        let W = retrievalAttentionJLProjection(dHead: dHead, config: config)
        jlMatrixT = W.transposed(1, 0)
        eval(jlMatrixT!)
    }

    /// Update with a chunk of new K rows for ALL slots and ALL heads.
    ///
    /// - Parameter newKeys: `[B, nKVHeads, L, dHead]` post-RoPE keys.
    ///   All slots get the same L — assumes rectangular prefill/decode.
    ///   For continuous batching where slots have different L per step,
    ///   call this once per (slot, L) — that's a v2 path.
    public func update(newKeys: MLXArray) {
        precondition(newKeys.shape.count == 4,
            "expected [B, nKVHeads, L, dHead], got \(newKeys.shape)")
        precondition(newKeys.dim(0) == B, "B mismatch (got \(newKeys.dim(0)) expected \(B))")
        precondition(newKeys.dim(1) == nKVHeads, "nKVHeads mismatch")
        precondition(newKeys.dim(3) == dHead, "dHead mismatch")
        ensureJL()
        let L = newKeys.dim(2)
        let WT = jlMatrixT!

        // Project all (B, KV-head) tokens at once.
        //   [B, nKVH, L, dHead] @ [dHead, contentDim] = [B, nKVH, L, contentDim]
        let content = matmul(newKeys.asType(.float32), WT).asType(.float32)
        // Force materialization here — downstream blockwise pool reads
        // contiguous slabs, and lazy graph through index slicing showed up
        // as a perf cliff in the single-batch index path.
        eval(content)

        // Per slot, append L rows worth of content into running-mean block
        // features. We update fine + coarse separately because their
        // blockSize differs (default fine=64, coarse=1024).
        for slot in 0..<B {
            let oldT = populatedPerSlot[slot]
            let newT = oldT + L
            populatedPerSlot[slot] = newT
        }

        // Grow buffers BEFORE writing.
        let contentDim = content.dim(3)
        ensureBlockBuffer(isFine: true, contentDim: contentDim)
        if config.coarseRescueEnabled {
            ensureBlockBuffer(isFine: false, contentDim: contentDim)
        }

        // Per-slot per-block updates. For prefill (L large) we recompute
        // affected blocks via slab mean — cheaper than running-sum because
        // we're rewriting the whole tail. For decode (L=1) running-sum
        // wins: just bump 1 partial-block update.
        if L == 1 {
            updateDecodeStep(content: content, contentDim: contentDim)
        } else {
            updatePrefillChunk(content: content, L: L, contentDim: contentDim)
        }
    }

    /// L=1 decode update — running-mean of the partial last block per slot.
    private func updateDecodeStep(content: MLXArray, contentDim: Int) {
        // content: [B, nKVH, 1, contentDim] → squeeze L → [B, nKVH, contentDim]
        let added = content[0..., 0..., 0, 0...]
        // For each slot, fold into fine + coarse partial blocks.
        // We do this per (slot, isFine) by tracking the partial-block index
        // and a running sum. Then divide once at read time? Simpler: just
        // update mean in-place via the closed-form rolling average.
        //
        // mean_new = mean_old + (added - mean_old) / new_count
        //
        // Computed entirely with MLX ops on the [nKVH, contentDim] slice.
        updateRunningMean(isFine: true, added: added)
        if config.coarseRescueEnabled {
            updateRunningMean(isFine: false, added: added)
        }
    }

    /// Closed-form rolling mean update of the per-slot partial last block.
    private func updateRunningMean(isFine: Bool, added: MLXArray) {
        let bs = isFine ? config.fineBlockSize : config.coarseBlockSize
        let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!
        for slot in 0..<B {
            let popPost = populatedPerSlot[slot]
            // Block index of the new token = (popPost - 1) / bs.
            let blockIdx = (popPost - 1) / bs
            // Count of tokens now in this block (1..bs).
            let countInBlock = popPost - blockIdx * bs
            // added row: [nKVH, contentDim] for this slot
            let addedRow = added[slot, 0..., 0...]  // [nKVH, contentDim]
            // Read old mean of this block: buf[slot, :, blockIdx, :]
            let oldMean = buf[slot, 0..., blockIdx, 0...]
            // new_mean = old * (count - 1) / count + added / count
            // Equivalent to: new_mean = old + (added - old) / count
            let countF = Float(countInBlock)
            let newMean: MLXArray
            if countInBlock == 1 {
                newMean = addedRow  // first token in this block = mean = added
            } else {
                newMean = oldMean + (addedRow - oldMean) / countF
            }
            buf[slot, 0..., blockIdx, 0...] = newMean
        }
    }

    /// Multi-token prefill update — recompute every affected block via
    /// slab-mean. Cheaper than per-token running-mean when L is large.
    private func updatePrefillChunk(content: MLXArray, L: Int, contentDim: Int) {
        // content: [B, nKVH, L, contentDim].
        // For each slot independently because per-slot oldT may differ
        // (continuous batching). With rectangular prefill all oldT match,
        // so we can vectorize across B, but the loop stays correct either
        // way. Optimize the rectangular case first.
        let rectangular: Bool = {
            guard B > 0 else { return false }
            let firstPop = populatedPerSlot[0]
            return populatedPerSlot.allSatisfy { $0 == firstPop }
        }()

        if rectangular {
            updatePrefillRectangular(content: content, L: L, contentDim: contentDim)
        } else {
            // Fallback per-slot path (ragged-T continuous batching is v2).
            // We do not expect this path to fire from current Bridge code.
            for slot in 0..<B {
                let slotContent = content[slot ..< (slot + 1), 0..., 0..., 0...]
                updateSlotPrefillSlice(
                    slot: slot, slotContent: slotContent,
                    L: L, contentDim: contentDim)
            }
        }
    }

    /// Rectangular prefill — all slots at the same oldT. Vectorize across B.
    private func updatePrefillRectangular(content: MLXArray, L: Int, contentDim: Int) {
        // populated already advanced; recover oldT from the first slot.
        let newT = populatedPerSlot[0]
        let oldT = newT - L
        updateBlockRectangularInPlace(
            content: content, oldT: oldT, newT: newT, L: L,
            isFine: true)
        if config.coarseRescueEnabled {
            updateBlockRectangularInPlace(
                content: content, oldT: oldT, newT: newT, L: L,
                isFine: false)
        }
    }

    /// Per-block mean recompute for the rectangular case.
    /// content: [B, nKVH, L, contentDim]
    private func updateBlockRectangularInPlace(
        content: MLXArray, oldT: Int, newT: Int, L: Int, isFine: Bool
    ) {
        let bs = isFine ? config.fineBlockSize : config.coarseBlockSize
        let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!
        let firstBlock = oldT / bs
        let lastBlock = (newT - 1) / bs  // inclusive
        for blk in firstBlock ... lastBlock {
            let blockStartAbs = blk * bs
            let blockEndAbs = min(blockStartAbs + bs, newT)
            // The piece coming from existing storage is positions
            //   [blockStartAbs..oldT)   — already in buf (mean has wrong
            // count if it's a partial); we recompute by combining its
            // partial sum (mean * oldCount) with the new piece's sum.
            let leftStart = max(0, blockStartAbs)
            let leftEnd = min(oldT, blockEndAbs)
            let leftCount = max(0, leftEnd - leftStart)
            let rightStart = max(oldT, blockStartAbs)
            let rightEnd = blockEndAbs
            let rightCount = max(0, rightEnd - rightStart)
            // content L-axis index for the new piece = (rightStart - oldT)..(rightEnd - oldT)
            let cStart = rightStart - oldT
            let cEnd = rightEnd - oldT
            // newCount = leftCount + rightCount (covers this block)
            let totalCount = leftCount + rightCount
            precondition(totalCount > 0, "block must have >=1 token")
            // RIGHT piece sum: content slice [:, :, cStart..cEnd, :].sum(L axis)
            let rightSlice = content[0..., 0..., cStart ..< cEnd, 0...]
            let rightSum = rightSlice.sum(axis: 2)   // [B, nKVH, contentDim]
            if leftCount == 0 {
                let mean = rightSum / Float(totalCount)
                buf[0..., 0..., blk, 0...] = mean
            } else {
                // LEFT piece sum = oldMean * leftCount.
                let oldMean = buf[0..., 0..., blk, 0...]  // [B, nKVH, contentDim]
                let leftSum = oldMean * Float(leftCount)
                let mean = (leftSum + rightSum) / Float(totalCount)
                buf[0..., 0..., blk, 0...] = mean
            }
        }
    }

    /// Per-slot prefill slice update. Slow path for non-rectangular.
    private func updateSlotPrefillSlice(
        slot: Int, slotContent: MLXArray, L: Int, contentDim: Int
    ) {
        // slotContent: [1, nKVH, L, contentDim]
        // populated already advanced; back into oldT.
        let newT = populatedPerSlot[slot]
        let oldT = newT - L
        for isFine in [true, false] {
            if !isFine && !config.coarseRescueEnabled { continue }
            let bs = isFine ? config.fineBlockSize : config.coarseBlockSize
            let buf = isFine ? fineBlockBuffer! : coarseBlockBuffer!
            let firstBlock = oldT / bs
            let lastBlock = (newT - 1) / bs
            for blk in firstBlock ... lastBlock {
                let blockStartAbs = blk * bs
                let blockEndAbs = min(blockStartAbs + bs, newT)
                let leftStart = blockStartAbs
                let leftEnd = min(oldT, blockEndAbs)
                let leftCount = max(0, leftEnd - leftStart)
                let rightStart = max(oldT, blockStartAbs)
                let rightEnd = blockEndAbs
                let rightCount = max(0, rightEnd - rightStart)
                let cStart = rightStart - oldT
                let cEnd = rightEnd - oldT
                let totalCount = leftCount + rightCount
                let rightSlice = slotContent[0..., 0..., cStart ..< cEnd, 0...]
                let rightSum = rightSlice.sum(axis: 2)   // [1, nKVH, contentDim]
                if leftCount == 0 {
                    let mean = rightSum / Float(totalCount)
                    buf[slot ..< (slot + 1), 0..., blk, 0...] = mean
                } else {
                    let oldMean = buf[slot ..< (slot + 1), 0..., blk, 0...]
                    let leftSum = oldMean * Float(leftCount)
                    let mean = (leftSum + rightSum) / Float(totalCount)
                    buf[slot ..< (slot + 1), 0..., blk, 0...] = mean
                }
            }
        }
    }

    private func ensureBlockBuffer(isFine: Bool, contentDim: Int) {
        let needed = isFine ? currentFineBlocks : currentCoarseBlocks
        let cap = isFine ? fineBlockCapacity : coarseBlockCapacity
        let existing = isFine ? fineBlockBuffer : coarseBlockBuffer
        if cap >= needed && existing != nil {
            return
        }
        // Doubling growth — matches the F-83 V1.1 pattern. Cheap because
        // contentDim is small (16) and block counts at T=128K, fineBS=64
        // = 2048 blocks → buf = [B, nKVH, 2048, 16] fp32 = 8 MB at B=8, nKVH=8.
        let chunkSize = 256
        var newCap = max(cap, chunkSize)
        while newCap < needed { newCap *= 2 }
        let dtype: DType = .float32
        let newBuf = MLXArray.zeros([B, nKVHeads, newCap, contentDim], dtype: dtype)
        if cap > 0, let old = existing {
            newBuf[0..., 0..., ..<cap, 0...] = old[0..., 0..., ..<cap, 0...]
        }
        if isFine {
            fineBlockBuffer = newBuf
            fineBlockCapacity = newCap
        } else {
            coarseBlockBuffer = newBuf
            coarseBlockCapacity = newCap
        }
    }

    // MARK: - scoring + top-K (the hot path the F-71b kernel feeds on)

    /// Project decode queries for all (B, KV-heads) in one batched matmul.
    /// - Parameter q: `[B, nKVHeads, dHead]` decode-step query (rep-per-group).
    /// - Returns: `[B, nKVHeads, contentDim]`.
    public func projectQueriesBatched(_ q: MLXArray) -> MLXArray {
        precondition(q.shape == [B, nKVHeads, dHead],
            "expected [\(B), \(nKVHeads), \(dHead)], got \(q.shape)")
        ensureJL()
        let WT = jlMatrixT!
        // [B, nKVH, dHead] @ [dHead, contentDim] = [B, nKVH, contentDim]
        return matmul(q.asType(.float32), WT)
    }

    /// Per-(B, KV-head) top-K fine block starts.
    /// - Parameter projectedQ: `[B, nKVHeads, contentDim]`
    /// - Returns: `[B, nKVHeads, K_fine]` int32 token-position block starts.
    public func topKFineBlockStarts(projectedQ: MLXArray) -> MLXArray {
        guard let features = fineBlockFeatures else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        let k = config.effectiveFineTopK(seqLen: seqLenMax)
        return computeTopK(
            features: features, projectedQ: projectedQ,
            k: k, blockSize: config.fineBlockSize)
    }

    public func topKCoarseBlockStarts(projectedQ: MLXArray) -> MLXArray {
        guard config.coarseRescueEnabled,
              let features = coarseBlockFeatures
        else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        return computeTopK(
            features: features, projectedQ: projectedQ,
            k: config.coarseTopK, blockSize: config.coarseBlockSize)
    }

    /// Score + argPartition on a [B, nKVH, nBlocks, contentDim] buffer.
    /// Returns top-K block starts `[B, nKVH, k]` in TOKEN coords.
    private func computeTopK(
        features: MLXArray, projectedQ: MLXArray, k: Int, blockSize: Int
    ) -> MLXArray {
        let nBlocks = features.dim(2)
        let take = min(k, nBlocks)
        guard take > 0 else {
            return MLXArray.zeros([B, nKVHeads, 1], dtype: .int32)
        }
        // Score: per (B, nKVH, block, content) inner-product with projectedQ.
        //   features: [B, nKVH, nBlocks, contentDim]
        //   projQ:    [B, nKVH, contentDim] → reshape [B, nKVH, 1, contentDim]
        let qExp = projectedQ.reshaped(B, nKVHeads, 1, projectedQ.dim(2))
        let scores = (features * qExp).sum(axis: -1)  // [B, nKVH, nBlocks]
        // argPartition on axis -1 picks top-`take` at positions [N-take..N).
        let pivot = nBlocks - take
        let partitioned: MLXArray
        if pivot <= 0 {
            let rangeArr = MLXArray(0..<Int32(nBlocks))
                .reshaped(1, 1, nBlocks)
            partitioned = broadcast(rangeArr, to: [B, nKVHeads, nBlocks])
        } else {
            partitioned = argPartition(scores, kth: pivot, axis: -1)
        }
        let topKIdx = partitioned[0..., 0..., (nBlocks - take)...]
        return topKIdx * Int32(blockSize)
    }

    /// F-71b feeder — per-(B, KV-head) sorted gather list of ALL positions
    /// to attend (static prefix + sliding window + top-K fine blocks +
    /// optional top-K coarse blocks). Padded to fixed K_padded for the
    /// kernel grid; positions are clipped to `[0, T-1]` so OOB protection
    /// in the kernel doesn't fire.
    ///
    /// All slots share the same static/sliding range when rectangular.
    /// When ragged, this returns the union — the kernel applies positions
    /// per-(B, KV-head) so a slot that has fewer tokens than another still
    /// gets correct masking via the K-position clip + per-slot K accumulator.
    ///
    /// - Parameters:
    ///   - projectedQ: `[B, nKVHeads, contentDim]`
    ///   - T: rectangular cache length (caller passes K.dim(2)).
    /// - Returns: (gather [B, nKVHeads, K_padded] int32, K_padded)
    public func perKVHeadGatherBatched(
        projectedQ: MLXArray, seqLen T: Int
    ) -> (MLXArray, Int) {
        let staticInit = config.staticInit
        let slidingWindow = config.slidingWindow
        let fineBS = config.fineBlockSize
        let coarseBS = config.coarseBlockSize
        let kFineEff = min(config.effectiveFineTopK(seqLen: T),
                           fineBlockFeatures?.dim(2) ?? 0)
        let kCoarseEff = config.coarseRescueEnabled
            ? min(config.coarseTopK, coarseBlockFeatures?.dim(2) ?? 0)
            : 0
        let staticEnd = min(staticInit, T)
        let slidingStart = max(0, T - slidingWindow)
        let staticCount = staticEnd
        let slidingCount = max(0, T - slidingStart)
        let kPadded = staticCount + slidingCount
            + kFineEff * fineBS + kCoarseEff * coarseBS

        // Static + sliding ranges — same across (B, nKVH).
        let staticPositions = staticCount > 0
            ? MLXArray(Int32(0)..<Int32(staticEnd))
            : MLXArray.zeros([0], dtype: .int32)
        let slidingPositions = slidingCount > 0
            ? MLXArray(Int32(slidingStart)..<Int32(T))
            : MLXArray.zeros([0], dtype: .int32)
        let baseSS = concatenated([staticPositions, slidingPositions], axis: 0)
            .reshaped(1, 1, staticCount + slidingCount)
        let staticBroadcast = broadcast(
            baseSS, to: [B, nKVHeads, staticCount + slidingCount])

        var pieces: [MLXArray] = [staticBroadcast]
        if kFineEff > 0 {
            let fineStarts = topKFineBlockStarts(projectedQ: projectedQ)
            // expand to per-token block positions:
            //   [B, nKVH, kFineEff] + [1, 1, 1, fineBS] = [B, nKVH, kFineEff, fineBS]
            let offsets = MLXArray(0..<Int32(fineBS)).reshaped(1, 1, 1, fineBS)
            let expanded = fineStarts.expandedDimensions(axis: 3) + offsets
            pieces.append(expanded.reshaped(B, nKVHeads, kFineEff * fineBS))
        }
        if kCoarseEff > 0 {
            let coarseStarts = topKCoarseBlockStarts(projectedQ: projectedQ)
            let offsets = MLXArray(0..<Int32(coarseBS)).reshaped(1, 1, 1, coarseBS)
            let expanded = coarseStarts.expandedDimensions(axis: 3) + offsets
            pieces.append(expanded.reshaped(B, nKVHeads, kCoarseEff * coarseBS))
        }
        let unsorted = concatenated(pieces, axis: 2)  // [B, nKVH, kPadded]
        let clipped = clip(unsorted, min: Int32(0), max: Int32(T - 1))
        // `sorted` on int32 returns int64; F-71b kernel takes int32 gather
        // (kernel reads `gather[gather_base + i]` as int). Force back to
        // int32 before handing to kernel.
        let sortedGather = sorted(clipped, axis: -1).asType(.int32)
        return (sortedGather, kPadded)
    }
}
