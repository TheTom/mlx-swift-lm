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

    /// Per-token f(K) embedding, `[T, selectorDim]` fp16. Grown via
    /// concatenation; in v1 we don't pre-allocate — the typical
    /// 1M-context worst case is 1M × 32 × 2 bytes = 64MB per (layer,
    /// KV head), well within budget. v2 can preallocate.
    private(set) public var perTokenFeatures: MLXArray?

    /// Block-pooled f(K), `[nFineBlocks, selectorDim]`. Recomputed at
    /// `update` for the affected tail block.
    private(set) public var fineBlockFeatures: MLXArray?

    /// Same for coarse blocks (1024-token blocks).
    private(set) public var coarseBlockFeatures: MLXArray?

    public init(
        config: RetrievalAttentionConfig = RetrievalAttentionConfig(),
        dHead: Int,
        ropeBase: Float = 10_000.0,
        layerIdx: Int
    ) {
        self.config = config
        self.dHead = dHead
        self.ropeBase = ropeBase
        self.layerIdx = layerIdx
    }

    /// Current cached sequence length (== rows in `perTokenFeatures`).
    public var seqLen: Int { perTokenFeatures?.dim(0) ?? 0 }

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
        let oldSeqLen = seqLen
        let newSeqLen = oldSeqLen + newK.dim(0)

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

        // Append to the running per-token buffer.
        if let existing = perTokenFeatures {
            perTokenFeatures = concatenated([existing, selectorNew], axis: 0)
        } else {
            perTokenFeatures = selectorNew
        }

        // Re-pool the fine + coarse indexes from scratch.
        // v2 optimization: only re-pool the last block (the one whose
        // tokens changed). v1 keeps it simple — full re-pool is O(T)
        // per update step and dominated by the matmul above.
        fineBlockFeatures = retrievalAttentionBlockMeanPool(
            perTokenFeatures!, blockSize: config.fineBlockSize
        )
        if config.coarseRescueEnabled {
            coarseBlockFeatures = retrievalAttentionBlockMeanPool(
                perTokenFeatures!, blockSize: config.coarseBlockSize
            )
        }
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
