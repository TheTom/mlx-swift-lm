// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// RetrievalAttention per-layer selector index. Incrementally maintains
// block-pooled f(K) embeddings as the KV cache populates. The "fp16
// always" precision choice (Decision 13) means this index ignores
// whatever quantization the underlying KV cache uses — selector
// quality is decoupled from KV byte savings.
//
// One instance per (layer, KV head) — see Decision 5.

import Foundation
import MLX

// MARK: - Selector index

/// Per-(layer, KV head) f(K) block-pooled index used to score blocks
/// against the projected query.
///
/// Lifecycle:
/// 1. `init` — allocate empty fp16 buffers sized for `maxSeqLen`.
/// 2. `update(newK:)` — append new keys (1+ tokens) post-RoPE. Project
///    them through the JL matrix + V3-trig basis, append to per-token
///    f(K), and re-pool the affected blocks.
/// 3. `scoreBlocks(against:)` — score every populated block against
///    a projected query.
/// 4. `topK(...)` — pick the top-k fine block starts AND top-k coarse
///    block starts (separate indexes inside).
public final class RetrievalAttentionIndex {

    public let config: RetrievalAttentionConfig

    /// Head dim of the underlying model's K.
    public let dHead: Int

    /// RoPE base — for the V3-trig features. Must match the model's
    /// `rope_theta`.
    public let ropeBase: Float

    /// Number of layers in the model — used for diagnostic-only checks.
    public let layerIdx: Int

    /// The cached projection matrix `[contentDim, dHead]`. Lazily
    /// computed at first `update` so the index can be constructed
    /// before MLX is initialized.
    private var jlMatrix: MLXArray?

    /// Maximum sequence length the index will hold. Sets the
    /// pre-allocated buffer sizes; caller picks from engine's max-prompt-
    /// len or kv cap.
    public let maxSeqLen: Int

    /// v2 pre-allocated backing buffer `[maxSeqLen, selectorDim]`,
    /// allocated lazily on first `update`. v1's per-step `concatenated()`
    /// grew the buffer linearly and held the prior MLXArray refs alive
    /// via the lazy graph — at 128K context × 64 layers × N decode steps
    /// this leaked GBs and froze the system on long-context benches.
    /// In-place slice writes on a fixed-size buffer kill that path.
    private var _perTokenBuf: MLXArray?
    private var _currentSeqLen: Int = 0

    /// Slice view into `_perTokenBuf` covering `_currentSeqLen` rows.
    /// Aliases the underlying buffer; callers must not retain across
    /// `update` calls (slice content changes when we write).
    public var perTokenFeatures: MLXArray? {
        guard let buf = _perTokenBuf, _currentSeqLen > 0 else { return nil }
        return buf[..<_currentSeqLen, 0...]
    }

    /// Block-pooled `[ceil(maxSeqLen/fineBlockSize), selectorDim]` fp32.
    private var _fineBlockBuf: MLXArray?
    private var _currentFineBlockCount: Int = 0
    public var fineBlockFeatures: MLXArray? {
        guard let buf = _fineBlockBuf, _currentFineBlockCount > 0 else { return nil }
        return buf[..<_currentFineBlockCount, 0...]
    }

    /// Same for coarse blocks.
    private var _coarseBlockBuf: MLXArray?
    private var _currentCoarseBlockCount: Int = 0
    public var coarseBlockFeatures: MLXArray? {
        guard let buf = _coarseBlockBuf, _currentCoarseBlockCount > 0 else { return nil }
        return buf[..<_currentCoarseBlockCount, 0...]
    }

    public init(
        config: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        dHead: Int,
        ropeBase: Float = 10_000.0,
        layerIdx: Int,
        maxSeqLen: Int = 131_072
    ) {
        self.config = config
        self.dHead = dHead
        self.ropeBase = ropeBase
        self.layerIdx = layerIdx
        self.maxSeqLen = maxSeqLen
    }

    /// Current cached sequence length.
    public var seqLen: Int { _currentSeqLen }

    /// Append new post-RoPE keys. Updates per-token + block features.
    ///
    /// - Parameter newK: `[L, dHead]` rows. Caller flattens any B / KV
    ///   head dim before passing (the index is per-(layer, KV-head)).
    public func update(newK: MLXArray) {
        precondition(
            newK.shape.count == 2,
            "expected [L, dHead]; got \(newK.shape)"
        )
        precondition(
            newK.dim(1) == dHead,
            "newK dim 1 (\(newK.dim(1))) != index dHead (\(dHead))"
        )

        // Lazy JL projection matrix.
        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(
                dHead: dHead, config: config
            )
        }
        let W = jlMatrix!

        // Compute selector features for the new rows.
        let oldSeqLen = _currentSeqLen
        let newSeqLen = oldSeqLen + newK.dim(0)
        precondition(newSeqLen <= maxSeqLen,
            "RetrievalAttentionIndex: newSeqLen \(newSeqLen) > maxSeqLen \(maxSeqLen)")

        // content: [L, contentDim] = newK @ Wᵀ
        let contentNew = matmul(newK, W.transposed(1, 0)).asType(.float32)

        // trig: position is "this token's index in the sequence".
        // We store ABSOLUTE positions in the index; the query-side
        // relative encoding happens in `scoreBlocks`.
        let positions = MLXArray(
            (Int32(oldSeqLen)..<Int32(newSeqLen))
        ).asType(.int32)
        let trigNew = retrievalAttentionTrigFeatures(
            relativePositions: positions, base: ropeBase, config: config
        )

        // Concat content + trig along feature dim → [L, selectorDim]
        let selectorNew = concatenated([contentNew, trigNew], axis: -1)

        // v2: in-place slice write into pre-allocated buffer instead of
        // per-step concat. Releases prior buffer refs immediately; no
        // lazy-graph chain growth. Lazy alloc on first call so MLX is
        // guaranteed initialized.
        let selectorDim = config.selectorDim
        if _perTokenBuf == nil {
            _perTokenBuf = MLXArray.zeros(
                [maxSeqLen, selectorDim], dtype: .float32)
            eval(_perTokenBuf!)
        }
        _perTokenBuf![oldSeqLen ..< newSeqLen, 0...] = selectorNew
        _currentSeqLen = newSeqLen
        // Force eval to release any lazy refs to prior selectorNew /
        // matmul intermediates — otherwise the graph keeps the chain
        // alive across decode steps and RSS climbs.
        eval(_perTokenBuf!)

        // Update fine + coarse block-pooled features. Incremental for
        // L=1 decode steps (only the tail block(s) can have changed);
        // full re-pool only for prefill / multi-token chunks.
        if newK.dim(0) == 1 && _currentFineBlockCount > 0 {
            updateBlockTailIncremental(blockSize: config.fineBlockSize, oldSeqLen: oldSeqLen, isFine: true)
            if config.coarseRescueEnabled && _currentCoarseBlockCount > 0 {
                updateBlockTailIncremental(blockSize: config.coarseBlockSize, oldSeqLen: oldSeqLen, isFine: false)
            }
        } else {
            let fullPooledFine = retrievalAttentionBlockMeanPool(
                perTokenFeatures!, blockSize: config.fineBlockSize
            )
            writePoolBuffer(
                full: fullPooledFine, blockSize: config.fineBlockSize, isFine: true)
            if config.coarseRescueEnabled {
                let fullPooledCoarse = retrievalAttentionBlockMeanPool(
                    perTokenFeatures!, blockSize: config.coarseBlockSize
                )
                writePoolBuffer(
                    full: fullPooledCoarse, blockSize: config.coarseBlockSize, isFine: false)
            }
        }
    }

    /// v2 in-place write into the pre-allocated fine/coarse block
    /// buffer. Lazy allocates on first call.
    private func writePoolBuffer(full: MLXArray, blockSize: Int, isFine: Bool) {
        let nRows = full.dim(0)
        let selectorDim = config.selectorDim
        let maxBlocks = (maxSeqLen + blockSize - 1) / blockSize
        if isFine {
            if _fineBlockBuf == nil {
                _fineBlockBuf = MLXArray.zeros(
                    [maxBlocks, selectorDim], dtype: .float32)
                eval(_fineBlockBuf!)
            }
            _fineBlockBuf![..<nRows, 0...] = full
            _currentFineBlockCount = nRows
            eval(_fineBlockBuf!)
        } else {
            if _coarseBlockBuf == nil {
                _coarseBlockBuf = MLXArray.zeros(
                    [maxBlocks, selectorDim], dtype: .float32)
                eval(_coarseBlockBuf!)
            }
            _coarseBlockBuf![..<nRows, 0...] = full
            _currentCoarseBlockCount = nRows
            eval(_coarseBlockBuf!)
        }
    }

    /// Incremental pooled-feature update for a single new token. Updates
    /// only the last block (the one containing position `oldSeqLen`) —
    /// previous blocks' means don't change when one token is appended.
    /// F-43 motivated this: full re-pool at every decode step costs 40x
    /// vs dense.
    private func updateBlockTailIncremental(
        blockSize: Int, oldSeqLen: Int, isFine: Bool
    ) {
        let priorBlockCount = isFine ? _currentFineBlockCount : _currentCoarseBlockCount
        let newSeqLen = _currentSeqLen  // == oldSeqLen + 1
        let lastBlockIdx = oldSeqLen / blockSize
        let lastBlockStart = lastBlockIdx * blockSize
        // Mean over the slice [lastBlockStart..<newSeqLen] of perTokenFeatures.
        let tail = perTokenFeatures![lastBlockStart..., 0...]
        let tailMean = tail.mean(axis: 0).reshaped(1, -1)

        // v2 in-place writes into pre-allocated _fineBlockBuf / _coarseBlockBuf.
        // Replaces the v1 `concatenated([pooled, tailMean], axis: 0)` that
        // grew a new MLXArray per decode step.
        if priorBlockCount == lastBlockIdx + 1 {
            // Same block as before — replace last row in place.
            if isFine {
                _fineBlockBuf![(priorBlockCount - 1) ..< priorBlockCount, 0...] = tailMean
                eval(_fineBlockBuf!)
            } else {
                _coarseBlockBuf![(priorBlockCount - 1) ..< priorBlockCount, 0...] = tailMean
                eval(_coarseBlockBuf!)
            }
        } else if priorBlockCount == lastBlockIdx {
            // New block started at lastBlockIdx — append in place.
            if isFine {
                _fineBlockBuf![priorBlockCount ..< (priorBlockCount + 1), 0...] = tailMean
                _currentFineBlockCount = priorBlockCount + 1
                eval(_fineBlockBuf!)
            } else {
                _coarseBlockBuf![priorBlockCount ..< (priorBlockCount + 1), 0...] = tailMean
                _currentCoarseBlockCount = priorBlockCount + 1
                eval(_coarseBlockBuf!)
            }
        } else {
            // Unexpected (skipped blocks?). Fall back to full re-pool.
            let full = retrievalAttentionBlockMeanPool(perTokenFeatures!, blockSize: blockSize)
            writePoolBuffer(full: full, blockSize: blockSize, isFine: isFine)
        }
        _ = newSeqLen
    }

    /// Build the projected query for selector scoring.
    ///
    /// The Q-side does the same JL projection (so dot product semantics
    /// survive) plus the trig features at `relative_pos = 0` (the
    /// query attends to itself at the present moment; the K-side
    /// already encodes the distance back in its trig features).
    public func projectQuery(_ q: MLXArray) -> MLXArray {
        precondition(q.shape == [dHead], "q must be [dHead], got \(q.shape)")
        if jlMatrix == nil {
            jlMatrix = retrievalAttentionJLProjection(
                dHead: dHead, config: config
            )
        }
        let W = jlMatrix!
        // matmul of W (16,128) and q (128,) is awkward in MLX; use a
        // reshape trick.
        let contentQ = matmul(W, q.reshaped(dHead, 1))
            .reshaped(config.contentDim)
            .asType(.float32)
        let trigQ = retrievalAttentionTrigFeatures(
            relativePositions: MLXArray([Int32(0)]), base: ropeBase, config: config
        ).reshaped(config.trigDim)
        return concatenated([contentQ, trigQ], axis: 0)
    }

    /// Score every fine block against the projected query.
    public func scoreFineBlocks(against projectedQ: MLXArray) -> MLXArray {
        guard let features = fineBlockFeatures else {
            return MLXArray([] as [Float])
        }
        return retrievalAttentionScoreBlocks(
            blockFeatures: features, q: projectedQ, config: config
        )
    }

    /// Score every coarse block against the projected query.
    public func scoreCoarseBlocks(against projectedQ: MLXArray) -> MLXArray? {
        guard let features = coarseBlockFeatures else { return nil }
        return retrievalAttentionScoreBlocks(
            blockFeatures: features, q: projectedQ, config: config
        )
    }

    /// Top-k fine block start positions (in token coordinates). Uses
    /// `config.effectiveFineTopK(seqLen:)` so adaptive-top-k kicks in
    /// automatically — F-41 measured this matters a LOT for multi-step
    /// generation at long context.
    public func topKFineBlockStarts(against projectedQ: MLXArray) -> [Int] {
        let scores = scoreFineBlocks(against: projectedQ)
        let k = config.effectiveFineTopK(seqLen: seqLen)
        let blockIndices = retrievalAttentionTopKBlocks(scores: scores, k: k)
        return blockIndices.map { $0 * config.fineBlockSize }
    }

    /// Top-k coarse block start positions (in token coordinates).
    public func topKCoarseBlockStarts(against projectedQ: MLXArray) -> [Int] {
        guard let scores = scoreCoarseBlocks(against: projectedQ) else {
            return []
        }
        let blockIndices = retrievalAttentionTopKBlocks(
            scores: scores, k: config.coarseTopK
        )
        return blockIndices.map { $0 * config.coarseBlockSize }
    }
}
