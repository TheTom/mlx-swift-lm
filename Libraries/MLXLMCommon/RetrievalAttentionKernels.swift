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
    private var sparseSDPAKernel: MLXFast.MLXFastKernel?

    func getSparseSDPA() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = sparseSDPAKernel { return k }
        // Phase C.2 fused sparse SDPA for decode-step (L=1).
        // Replaces the mask-not-gather path with actual sparse compute:
        // only attends over `gather_indices` positions, skipping the
        // dense QK^T over T that the mask path pays for.
        //
        // Online softmax (single pass over gather positions, numerically
        // stable). One threadgroup per (B, Q head); each thread owns one
        // output dim element.
        let header = """
            // Online-softmax sparse SDPA helper kernel.

            """
        let source = """
            // Template constants: HEAD_DIM, GROUP_SIZE
            // Inputs (auto-typed by MLX from MLXArray dtype):
            //   q:       [B, NQH, 1, D]
            //   k:       [B, NKVH, T, D]
            //   v:       [B, NKVH, T, D]   (assumes Dv == D)
            //   gather:  [N_gather] int32 sorted unique positions
            //   params:  [scale, NQH_f, NKVH_f, T_f, N_gather_f] float
            // Output:
            //   out:     [B, NQH, 1, D]

            const uint d = thread_position_in_threadgroup.x;
            const uint tg_idx = threadgroup_position_in_grid.x;
            const float scale = params[0];
            const uint NQH = (uint)params[1];
            const uint NKVH = (uint)params[2];
            const uint T = (uint)params[3];
            const uint n_gather = (uint)params[4];

            const uint b = tg_idx / NQH;
            const uint qh = tg_idx % NQH;
            const uint kvh = qh / GROUP_SIZE;

            threadgroup float qk_buf[HEAD_DIM];      // QK reduction scratch
            threadgroup float wgts[2];        // [weight, correction]
            threadgroup float max_so_far;
            threadgroup float sum_exp;

            if (d == 0) {
                max_so_far = -INFINITY;
                sum_exp = 0.0;
            }

            // Load this thread's Q element
            const uint q_base = (b * NQH + qh) * HEAD_DIM;
            const float q_local = (float)q[q_base + d];
            float v_acc = 0.0;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint i = 0; i < n_gather; i++) {
                const int pos = gather[i];
                // Skip duplicates in sorted gather (adjacent-diff). Lets the
                // caller pass a sorted-but-not-uniqued list — softmax stays
                // correct because each unique pos contributes exactly once.
                if (i > 0 && pos == gather[i - 1]) {
                    continue;
                }
                const uint kv_base = (b * NKVH + (uint)kvh) * T * HEAD_DIM + (uint)pos * HEAD_DIM;

                // Q·K dot via parallel reduction across the D dim
                const float k_local = (float)k[kv_base + d];
                qk_buf[d] = q_local * k_local;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint stride_half = HEAD_DIM / 2; stride_half > 0; stride_half >>= 1) {
                    if (d < stride_half) qk_buf[d] += qk_buf[d + stride_half];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                const float score = qk_buf[0] * scale;

                // Online softmax update (thread 0 only, scalar state)
                if (d == 0) {
                    const float new_max = fmax(max_so_far, score);
                    const float correction = (max_so_far == -INFINITY)
                        ? 0.0f
                        : exp(max_so_far - new_max);
                    const float weight = exp(score - new_max);
                    sum_exp = sum_exp * correction + weight;
                    max_so_far = new_max;
                    wgts[0] = weight;
                    wgts[1] = correction;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Accumulate weighted V
                const float v_local = (float)v[kv_base + d];
                v_acc = v_acc * wgts[1] + wgts[0] * v_local;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Finalize
            const float final_sum = sum_exp;
            out[q_base + d] = v_acc / final_sum;
            """

        let k = MLXFast.metalKernel(
            name: "ra_sparse_sdpa_decode",
            inputNames: ["q", "k", "v", "gather", "params"],
            outputNames: ["out"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        sparseSDPAKernel = k
        return k
    }

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

/// Run the fused sparse SDPA kernel for decode-step (L=1).
///
/// Replaces the mask-not-gather path with actual sparse compute — only
/// attends over `gatherIndices` positions instead of running dense SDPA
/// over the full K with a mask.
///
/// - Parameters:
///   - queries: `[B, nQHeads, 1, D]` (decode step).
///   - keys: `[B, nKVHeads, T, D]` full cached K.
///   - values: `[B, nKVHeads, T, D]` full cached V (assumes Dv == D).
///   - gatherIndices: `[N_gather]` int32 — sorted unique positions to attend.
///   - scale: SDPA scale factor (typically `1/√D`).
/// - Returns: `[B, nQHeads, 1, D]` attention output.
public func retrievalAttentionFusedSparseSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    gatherIndices: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(queries.shape.count == 4, "queries must be [B, nQH, 1, D]")
    precondition(keys.shape.count == 4, "keys must be [B, nKVH, T, D]")
    precondition(values.shape.count == 4, "values must be [B, nKVH, T, D]")
    precondition(gatherIndices.shape.count == 1, "gather must be 1-D")
    let B = queries.dim(0)
    let nQH = queries.dim(1)
    precondition(queries.dim(2) == 1, "decode-step L=1 only")
    let D = queries.dim(3)
    let nKVH = keys.dim(1)
    let T = keys.dim(2)
    precondition(keys.dim(3) == D, "K head_dim must match Q")
    precondition(values.dim(3) == D, "V head_dim assumed equal to K (Dv==D)")
    precondition(nQH % nKVH == 0, "Q heads must be a multiple of KV heads")
    let groupSize = nQH / nKVH
    let nGather = gatherIndices.dim(0)
    precondition(nGather > 0, "gather must be non-empty")
    precondition([32, 64, 96, 128, 256].contains(D),
        "head_dim \(D) outside the supported template list")

    let params = MLXArray([
        scale,
        Float(nQH),
        Float(nKVH),
        Float(T),
        Float(nGather),
    ])
    // Cast K, V, Q to float32 for numerical stability inside the kernel.
    let qF = queries.asType(.float32)
    let kF = keys.asType(.float32)
    let vF = values.asType(.float32)

    let kernel = _RAKernelCache.shared.getSparseSDPA()
    // Grid: (B * nQH * D, 1, 1). Threadgroup: (D, 1, 1). One threadgroup
    // per (B, Q head); each thread owns one output dim element.
    let totalThreads = B * nQH * D
    let outputs = kernel(
        [qF, kF, vF, gatherIndices, params],
        template: [
            ("HEAD_DIM", D),
            ("GROUP_SIZE", groupSize),
        ],
        grid: (totalThreads, 1, 1),
        threadGroup: (D, 1, 1),
        outputShapes: [[B, nQH, 1, D]],
        outputDTypes: [.float32]
    )
    // Cast back to the original Q dtype for downstream compat.
    return outputs[0].asType(queries.dtype)
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
