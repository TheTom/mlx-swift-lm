// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// RetrievalAttention custom Metal kernels (Phase C).
//
// Goal: replace several small MLX ops in the selector hot path with a
// single fused kernel launch. F-48 measured ~14.5ms / sparse-layer overhead
// at 16K on Qwen3-0.6B-4bit, dominated by per-op MLX dispatch (~0.5ms × 25
// ops + 1 asArray sync). Fusion targets reducing op count.
//
// First step (this file): a score+top-K kernel that takes block features
// and a projected query and returns top-K block starts (token coordinates)
// in a single launch — replaces 4-5 MLX ops with 1.

import Foundation
import MLX

/// Cached one-time-constructed kernel handle. Built lazily on first call.
private final class _RAKernelCache: @unchecked Sendable {
    static let shared = _RAKernelCache()
    private let lock = NSLock()
    private var scoreTopKKernel: MLXFast.MLXFastKernel?

    func getScoreTopK() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = scoreTopKKernel { return k }
        let header = """
            // Helper to swap two (score, idx) pairs.
            inline void _ra_swap(thread float& a_score, thread int& a_idx,
                                thread float& b_score, thread int& b_idx) {
                float ts = a_score; int ti = a_idx;
                a_score = b_score; a_idx = b_idx;
                b_score = ts; b_idx = ti;
            }
            """
        let source = """
            // Template constants: N_BLOCKS, CONTENT_DIM, K, BLOCK_SIZE, N_HEADS
            // Inputs:
            //   block_features: float [N_HEADS * N_BLOCKS * CONTENT_DIM]
            //   projected_q:    float [N_HEADS * CONTENT_DIM]
            // Outputs:
            //   topk_starts: int [N_HEADS * K]
            //
            // One threadgroup per KV head. Threadgroup size = tg_size,
            // typically >= N_BLOCKS (so each thread owns one block).

            // Threadgroup memory must be declared at function scope (not inside loops).
            threadgroup float scores[N_BLOCKS];
            threadgroup int idxs[N_BLOCKS];
            threadgroup float reduce_score[N_BLOCKS];
            threadgroup int reduce_idx[N_BLOCKS];

            uint head = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint tg_size = threads_per_threadgroup.x;

            // ----- Phase 1: each thread computes scores for a stride of blocks.
            for (uint b = tid; b < N_BLOCKS; b += tg_size) {
                float s = 0.0f;
                uint feat_base = head * N_BLOCKS * CONTENT_DIM + b * CONTENT_DIM;
                uint q_base = head * CONTENT_DIM;
                for (uint d = 0; d < CONTENT_DIM; d++) {
                    s += block_features[feat_base + d] * projected_q[q_base + d];
                }
                scores[b] = s;
                idxs[b] = (int)b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // ----- Phase 2: K iterations of parallel argmax + mark-as-taken.
            for (uint k = 0; k < K; k++) {
                // Copy current scores/idxs into the reduction buffers.
                for (uint i = tid; i < N_BLOCKS; i += tg_size) {
                    reduce_score[i] = scores[i];
                    reduce_idx[i] = idxs[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Tree reduction over N_BLOCKS to find max.
                uint stride_half = N_BLOCKS / 2;
                while (stride_half > 0) {
                    if (tid < stride_half) {
                        if (reduce_score[tid + stride_half] > reduce_score[tid]) {
                            reduce_score[tid] = reduce_score[tid + stride_half];
                            reduce_idx[tid] = reduce_idx[tid + stride_half];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    stride_half = stride_half / 2;
                }

                if (tid == 0) {
                    int best = reduce_idx[0];
                    topk_starts[head * K + k] = best * BLOCK_SIZE;
                    // Knock out the winner so the next iteration finds the next-largest.
                    scores[best] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """

        let k = MLXFast.metalKernel(
            name: "ra_score_topk",
            inputNames: ["block_features", "projected_q"],
            outputNames: ["topk_starts"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        scoreTopKKernel = k
        return k
    }
}

/// Run the fused score+top-K kernel.
///
/// - Parameters:
///   - blockFeatures: `[nHeads, nBlocks, contentDim]` float32.
///   - projectedQ:    `[nHeads, contentDim]` float32.
///   - k:             top-K count.
///   - blockSize:     block stride in token coordinates.
///   - nBlocksRounded: power-of-2 ceiling of `nBlocks` (threadgroup size).
/// - Returns: `[nHeads, k]` int32 — block starts in token coordinates.
public func retrievalAttentionScoreTopKFused(
    blockFeatures: MLXArray,
    projectedQ: MLXArray,
    k: Int,
    blockSize: Int,
    nBlocksRounded: Int
) -> MLXArray {
    precondition(blockFeatures.shape.count == 3, "expected [nH, nBlocks, D]")
    precondition(projectedQ.shape.count == 2, "expected [nH, D]")
    let nHeads = blockFeatures.dim(0)
    let nBlocks = blockFeatures.dim(1)
    let contentDim = blockFeatures.dim(2)
    precondition(projectedQ.dim(0) == nHeads, "head count mismatch")
    precondition(projectedQ.dim(1) == contentDim, "content dim mismatch")
    precondition(k > 0 && k <= nBlocks, "k out of range")

    let kernel = _RAKernelCache.shared.getScoreTopK()
    // Threadgroup size: must equal nBlocks for the tree reduction. Capped
    // at 1024 (Metal max).
    let tgSize = min(1024, nBlocksRounded)
    // `grid` in MLXFast is TOTAL threads (= threadgroup count × threadgroup
    // size). One threadgroup per KV head → grid.x = nHeads × tgSize.
    let outputs = kernel(
        [blockFeatures, projectedQ],
        template: [
            ("N_BLOCKS", nBlocks),
            ("CONTENT_DIM", contentDim),
            ("K", k),
            ("BLOCK_SIZE", blockSize),
            ("N_HEADS", nHeads),
        ],
        grid: (nHeads * tgSize, 1, 1),
        threadGroup: (tgSize, 1, 1),
        outputShapes: [[nHeads, k]],
        outputDTypes: [.int32]
    )
    return outputs[0]
}
