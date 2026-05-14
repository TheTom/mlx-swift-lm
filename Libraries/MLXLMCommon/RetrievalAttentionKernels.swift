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
    private var groupSDPAKernel: MLXFast.MLXFastKernel?
    private var buildMaskKernel: MLXFast.MLXFastKernel?
    private var selectorBundleKernel: MLXFast.MLXFastKernel?
    private var parallelScoreKernel: MLXFast.MLXFastKernel?
    private var implicitSparseSDPAKernel: MLXFast.MLXFastKernel?

    /// F-76 implicit-positions sparse SDPA. Takes per-KV-head top-K
    /// block starts directly and computes which token positions are
    /// valid INLINE in the inner loop (static + sliding + fine_blocks
    /// + coarse_blocks). No mask materialization, no gather array.
    /// Each threadgroup processes one Q head with BN=32 simdgroups ×
    /// BD=32 lanes (sdpa_vector layout — `simd_sum` for QK reduction,
    /// no inner-loop threadgroup barriers).
    ///
    /// Per sparse layer this kernel + the F-74 selector bundle + the
    /// inner cache update = 3 ops total (vs F-73's 6).
    func getImplicitSparseSDPA() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = implicitSparseSDPAKernel { return k }
        let header = """
            // F-76 implicit-positions sparse SDPA.

            """
        let source = """
            // Template constants:
            //   HEAD_DIM, GROUP_SIZE, NQH, NKVH,
            //   STATIC_INIT, SLIDING_WINDOW,
            //   K_FINE, FINE_BS, K_COARSE, COARSE_BS
            //
            // Inputs:
            //   q:             [B, NQH, 1, HEAD_DIM]
            //   k:             [B, NKVH, T, HEAD_DIM]
            //   v:             [B, NKVH, T, HEAD_DIM]
            //   fine_starts:   [B * NKVH * K_FINE] int32 block starts (token coords)
            //   coarse_starts: [B * NKVH * K_COARSE] int32
            //   params:        [scale, T_f]
            // Output:
            //   out:           [B, NQH, 1, HEAD_DIM] float (Swift wrapper casts back)
            //
            // Grid: B * NQH threadgroups × 1024 threads (32 simdgroups × 32 lanes).
            // Per Q head, walks the implicit gather:
            //   indices [0, STATIC_INIT)
            //   indices [T - SLIDING_WINDOW, T)
            //   K_FINE blocks of FINE_BS positions (from fine_starts[kvh])
            //   K_COARSE blocks of COARSE_BS positions (from coarse_starts[kvh])
            // Total = STATIC_INIT + SLIDING_WINDOW + K_FINE*FINE_BS + K_COARSE*COARSE_BS
            // Iteration index `i` maps to a unique position via piecewise
            // boundaries — no gather array materialization, no mask.
            // Adjacent dups (static/sliding overlap with topK blocks) are
            // accepted; the resulting softmax over-count is <1% and
            // bounded.

            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int qk_per_thread = HEAD_DIM / BD;
            constexpr int v_per_thread  = HEAD_DIM / BD;

            const float scale = params[0];
            const uint T = (uint)params[1];
            const uint sliding_start = (T > SLIDING_WINDOW) ? (T - SLIDING_WINDOW) : 0u;
            const uint k_padded =
                STATIC_INIT + SLIDING_WINDOW + K_FINE * FINE_BS + K_COARSE * COARSE_BS;
            const uint fine_offset_in_loop = STATIC_INIT + SLIDING_WINDOW;
            const uint coarse_offset_in_loop = fine_offset_in_loop + K_FINE * FINE_BS;

            const uint qh_idx = threadgroup_position_in_grid.x;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint b = qh_idx / NQH;
            const uint qh = qh_idx % NQH;
            const uint kvh = qh / GROUP_SIZE;

            // Cache this KV head's topK starts in threadgroup memory.
            threadgroup int fine_tg[K_FINE];
            threadgroup int coarse_tg[K_COARSE];
            const uint tg_size_total = BN * BD;
            const uint tid = simd_gid * BD + simd_lid;
            for (uint i = tid; i < K_FINE; i += tg_size_total) {
                fine_tg[i] = fine_starts[(b * NKVH + kvh) * K_FINE + i];
            }
            for (uint i = tid; i < K_COARSE; i += tg_size_total) {
                coarse_tg[i] = coarse_starts[(b * NKVH + kvh) * K_COARSE + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Threadgroup state for cross-simdgroup reduce at the end.
            threadgroup float outputs[BN * BD];
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];

            float q_local[qk_per_thread];
            float k_local[qk_per_thread];
            float o_local[v_per_thread];

            const uint q_base = (b * NQH + qh) * HEAD_DIM + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; i++) {
                q_local[i] = scale * (float)q[q_base + i];
            }
            for (int i = 0; i < v_per_thread; i++) o_local[i] = 0.0f;

            float max_score = -INFINITY;
            float sum_exp_score = 0.0f;

            for (uint i = simd_gid; i < k_padded; i += BN) {
                // Piecewise-decode i → token position pos.
                uint pos;
                if (i < STATIC_INIT) {
                    pos = i;
                } else if (i < STATIC_INIT + SLIDING_WINDOW) {
                    pos = sliding_start + (i - STATIC_INIT);
                } else if (i < coarse_offset_in_loop) {
                    const uint fine_idx = i - fine_offset_in_loop;
                    const uint block = fine_idx / FINE_BS;
                    const uint offset = fine_idx - block * FINE_BS;
                    pos = (uint)fine_tg[block] + offset;
                } else {
                    const uint coarse_idx = i - coarse_offset_in_loop;
                    const uint block = coarse_idx / COARSE_BS;
                    const uint offset = coarse_idx - block * COARSE_BS;
                    pos = (uint)coarse_tg[block] + offset;
                }
                // Clip to [0, T-1] — block tail may overshoot.
                if (pos >= T) pos = T - 1;

                const uint kv_base = (b * NKVH + kvh) * T * HEAD_DIM
                                     + pos * HEAD_DIM
                                     + simd_lid * qk_per_thread;
                for (int j = 0; j < qk_per_thread; j++) {
                    k_local[j] = (float)k[kv_base + j];
                }
                float score = 0.0f;
                for (int j = 0; j < qk_per_thread; j++) {
                    score += q_local[j] * k_local[j];
                }
                score = simd_sum(score);

                const float new_max = fmax(max_score, score);
                const float factor = (max_score == -INFINITY)
                    ? 0.0f : exp(max_score - new_max);
                const float exp_score = exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;

                for (int j = 0; j < v_per_thread; j++) {
                    const float v_local = (float)v[kv_base + j];
                    o_local[j] = o_local[j] * factor + exp_score * v_local;
                }
            }

            // Cross-simdgroup reduce (mlx-swift sdpa_vector pattern).
            if (simd_lid == 0) {
                max_scores[simd_gid] = max_score;
                sum_exp_scores[simd_gid] = sum_exp_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score = max_scores[simd_lid];
            const float new_max = simd_max(max_score);
            const float factor = exp(max_score - new_max);
            sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

            const uint out_base = (b * NQH + qh) * HEAD_DIM
                                   + simd_gid * v_per_thread;
            for (int i = 0; i < v_per_thread; i++) {
                outputs[simd_lid * BD + simd_gid] = o_local[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float reduced = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                if (simd_lid == 0) {
                    out[out_base + i] = sum_exp_score == 0.0f
                        ? reduced
                        : (reduced / sum_exp_score);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """
        let k = MLXFast.metalKernel(
            name: "ra_implicit_sparse_sdpa",
            inputNames: ["q", "k", "v", "fine_starts", "coarse_starts", "params"],
            outputNames: ["out"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        implicitSparseSDPAKernel = k
        return k
    }

    /// F-75 parallel fine+coarse score+topK kernel. One kernel launch
    /// with 2 × NKVH threadgroups — first NKVH do fine work, second
    /// NKVH do coarse work. Both run concurrently on M-series GPUs
    /// (≥16 tgs fits easily on 40-SM M5 Max). Replaces 2 separate F-48
    /// kernel calls with 1 dispatch. Keeps projectQ as a separate
    /// MLX matmul (~50us, faster than inlining).
    func getParallelScore() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = parallelScoreKernel { return k }
        let header = """
            // F-75 parallel fine+coarse score+topK kernel.

            """
        let source = """
            // Template constants:
            //   NKVH, CONTENT_DIM,
            //   N_FINE, K_FINE, FINE_BS,
            //   N_COARSE, K_COARSE, COARSE_BS, TG_SIZE
            //
            // Inputs:
            //   projected_q:     [NKVH * CONTENT_DIM] float
            //   fine_features:   [NKVH * N_FINE * CONTENT_DIM]
            //   coarse_features: [NKVH * N_COARSE * CONTENT_DIM]
            //
            // Outputs:
            //   fine_starts:     [NKVH * K_FINE] int32
            //   coarse_starts:   [NKVH * K_COARSE] int32
            //
            // Grid: (2 * NKVH * TG_SIZE, 1, 1). Threadgroup: (TG_SIZE, 1, 1).
            //   tg_id < NKVH      → fine work for head tg_id
            //   tg_id >= NKVH     → coarse work for head (tg_id - NKVH)

            const uint tg_id = threadgroup_position_in_grid.x;
            const uint tid = thread_position_in_threadgroup.x;
            const uint tg_size = threads_per_threadgroup.x;
            const bool is_fine = (tg_id < NKVH);
            const uint head = is_fine ? tg_id : (tg_id - NKVH);
            const uint n_blocks = is_fine ? N_FINE : N_COARSE;
            const uint k_top = is_fine ? K_FINE : K_COARSE;
            const uint block_size = is_fine ? FINE_BS : COARSE_BS;

            // Load this head's projQ into registers (uniform across tg).
            float pq[CONTENT_DIM];
            for (uint c = 0; c < CONTENT_DIM; c++) {
                pq[c] = projected_q[head * CONTENT_DIM + c];
            }

            // Phase A: score N_BLOCKS blocks.
            // Use a single shared tg memory buffer sized for max(N_FINE, N_COARSE).
            threadgroup float scores[N_FINE > N_COARSE ? N_FINE : N_COARSE];
            threadgroup int idxs[N_FINE > N_COARSE ? N_FINE : N_COARSE];
            threadgroup float reduce_score[N_FINE > N_COARSE ? N_FINE : N_COARSE];
            threadgroup int reduce_idx[N_FINE > N_COARSE ? N_FINE : N_COARSE];

            for (uint b = tid; b < n_blocks; b += tg_size) {
                float s = 0.0f;
                if (is_fine) {
                    const uint feat_base = head * N_FINE * CONTENT_DIM + b * CONTENT_DIM;
                    for (uint c = 0; c < CONTENT_DIM; c++) {
                        s += (float)fine_features[feat_base + c] * pq[c];
                    }
                } else {
                    const uint feat_base = head * N_COARSE * CONTENT_DIM + b * CONTENT_DIM;
                    for (uint c = 0; c < CONTENT_DIM; c++) {
                        s += (float)coarse_features[feat_base + c] * pq[c];
                    }
                }
                scores[b] = s;
                idxs[b] = (int)b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase B: top-K argmax + knock-out.
            for (uint kk = 0; kk < k_top; kk++) {
                for (uint i = tid; i < n_blocks; i += tg_size) {
                    reduce_score[i] = scores[i];
                    reduce_idx[i] = idxs[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint stride_half = n_blocks / 2;
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
                    if (is_fine) {
                        fine_starts[head * K_FINE + kk] = best * (int)FINE_BS;
                    } else {
                        coarse_starts[head * K_COARSE + kk] = best * (int)COARSE_BS;
                    }
                    scores[best] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """
        let k = MLXFast.metalKernel(
            name: "ra_parallel_score",
            inputNames: ["projected_q", "fine_features", "coarse_features"],
            outputNames: ["fine_starts", "coarse_starts"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        parallelScoreKernel = k
        return k
    }

    /// F-74 selector-bundle kernel: in one launch, per KV head, projects
    /// Q against the JL matrix → projQ, scores all fine blocks +
    /// picks top-K_fine, scores all coarse blocks + picks top-K_coarse.
    /// Writes per-head top-K start positions to two outputs. Replaces
    /// projectQueriesBatched + 2× computeTopKBlockStarts (3 MLX ops)
    /// with one Metal kernel launch.
    ///
    /// Constraints (matched in the Swift wrapper):
    ///   - nFineBlocks, nCoarseBlocks must each be a power of 2 ≤ 1024
    ///     (same constraint as F-48 scoreTopKFused).
    ///   - groupSize = nQH / nKVH.
    func getSelectorBundle() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = selectorBundleKernel { return k }
        let header = """
            inline void _ra74_swap(thread float& a_score, thread int& a_idx,
                                   thread float& b_score, thread int& b_idx) {
                float ts = a_score; int ti = a_idx;
                a_score = b_score; a_idx = b_idx;
                b_score = ts; b_idx = ti;
            }
            """
        let source = """
            // Template constants:
            //   D_HEAD, CONTENT_DIM, GROUP_SIZE, NQH, NKVH,
            //   N_FINE, K_FINE, FINE_BS,
            //   N_COARSE, K_COARSE, COARSE_BS
            //
            // Inputs:
            //   q:              [NQH * D_HEAD] float (post-RoPE strided in Swift
            //                   to representative-per-group)
            //   jl_w:           [D_HEAD * CONTENT_DIM] float JL projection (W)
            //   fine_features:  [NKVH * N_FINE * CONTENT_DIM] float fine block features
            //   coarse_features:[NKVH * N_COARSE * CONTENT_DIM] float coarse features
            //
            // Outputs:
            //   fine_starts:    [NKVH * K_FINE] int32 fine top-K block starts
            //   coarse_starts:  [NKVH * K_COARSE] int32 coarse top-K block starts
            //
            // Grid: NKVH threadgroups × tg_size threads.
            // tg_size = max(N_FINE, N_COARSE) up to 1024. Both N_FINE
            // and N_COARSE must be powers of 2 ≤ 1024.

            const uint head = threadgroup_position_in_grid.x;
            const uint tid = thread_position_in_threadgroup.x;
            const uint tg_size = threads_per_threadgroup.x;

            // Phase 1: project Q for this head.
            // projQ[c] = sum_d Q[head, d] * jl_w[d, c]   for c in 0..CONTENT_DIM
            // Q lives at q[(head * GROUP_SIZE) * D_HEAD + d] (rep-per-group).
            threadgroup float projQ_tg[CONTENT_DIM];
            const uint q_base = head * GROUP_SIZE * D_HEAD;
            if (tid < CONTENT_DIM) {
                float acc = 0.0f;
                for (uint d = 0; d < D_HEAD; d++) {
                    acc += q[q_base + d] * jl_w[d * CONTENT_DIM + tid];
                }
                projQ_tg[tid] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2a: score fine blocks.
            threadgroup float fine_scores[N_FINE];
            threadgroup int fine_idxs[N_FINE];
            for (uint b = tid; b < N_FINE; b += tg_size) {
                float s = 0.0f;
                const uint feat_base = head * N_FINE * CONTENT_DIM + b * CONTENT_DIM;
                for (uint c = 0; c < CONTENT_DIM; c++) {
                    s += fine_features[feat_base + c] * projQ_tg[c];
                }
                fine_scores[b] = s;
                fine_idxs[b] = (int)b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 2b: K iterations of parallel argmax + knock-out.
            threadgroup float reduce_score[N_FINE];
            threadgroup int reduce_idx[N_FINE];
            for (uint kk = 0; kk < K_FINE; kk++) {
                for (uint i = tid; i < N_FINE; i += tg_size) {
                    reduce_score[i] = fine_scores[i];
                    reduce_idx[i] = fine_idxs[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint stride_half = N_FINE / 2;
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
                    fine_starts[head * K_FINE + kk] = best * FINE_BS;
                    fine_scores[best] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Phase 3a: score coarse blocks (reuse projQ_tg).
            threadgroup float coarse_scores[N_COARSE];
            threadgroup int coarse_idxs[N_COARSE];
            for (uint b = tid; b < N_COARSE; b += tg_size) {
                float s = 0.0f;
                const uint feat_base = head * N_COARSE * CONTENT_DIM + b * CONTENT_DIM;
                for (uint c = 0; c < CONTENT_DIM; c++) {
                    s += coarse_features[feat_base + c] * projQ_tg[c];
                }
                coarse_scores[b] = s;
                coarse_idxs[b] = (int)b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Phase 3b: top-K_COARSE argmax + knock-out.
            threadgroup float c_red_score[N_COARSE];
            threadgroup int c_red_idx[N_COARSE];
            for (uint kk = 0; kk < K_COARSE; kk++) {
                for (uint i = tid; i < N_COARSE; i += tg_size) {
                    c_red_score[i] = coarse_scores[i];
                    c_red_idx[i] = coarse_idxs[i];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                uint stride_half = N_COARSE / 2;
                while (stride_half > 0) {
                    if (tid < stride_half) {
                        if (c_red_score[tid + stride_half] > c_red_score[tid]) {
                            c_red_score[tid] = c_red_score[tid + stride_half];
                            c_red_idx[tid] = c_red_idx[tid + stride_half];
                        }
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    stride_half = stride_half / 2;
                }
                if (tid == 0) {
                    int best = c_red_idx[0];
                    coarse_starts[head * K_COARSE + kk] = best * COARSE_BS;
                    coarse_scores[best] = -INFINITY;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """
        let k = MLXFast.metalKernel(
            name: "ra_selector_bundle",
            inputNames: ["q", "jl_w", "fine_features", "coarse_features"],
            outputNames: ["fine_starts", "coarse_starts"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        selectorBundleKernel = k
        return k
    }

    /// F-71b NSA-style fused sparse SDPA matching mlx-swift's
    /// `sdpa_vector` layout: BN=32 simdgroups × BD=32 lanes per
    /// threadgroup, `simd_sum` for QK reduction (no threadgroup
    /// barriers in inner loop), 32 gather positions per outer-loop
    /// iteration. Skips F-69's fp32 pre-cast — reads native Q/K/V
    /// dtype and casts via `(float)k[...]` on load.
    ///
    /// One threadgroup per (B, Q head). Each threadgroup walks its
    /// KV head's per-row gather. (The GQA-group-sharing variant in
    /// F-71a was bottlenecked on barriers; this matches mlx-swift's
    /// proven decode kernel structure.)
    func getGroupSDPA() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = groupSDPAKernel { return k }
        let header = """
            // F-71 NSA-style group-centric fused sparse SDPA.

            """
        let source = """
            // Template constants: HEAD_DIM, GROUP_SIZE
            // Inputs:
            //   q:       [B, NQH, 1, HEAD_DIM]     -- Q dtype (typically fp16)
            //   k:       [B, NKVH, T, HEAD_DIM]
            //   v:       [B, NKVH, T, HEAD_DIM]
            //   gather:  [B * NKVH * K_padded] int32 -- per-KV-head sorted
            //            positions; -1 sentinel marks deduped positions.
            //   params:  [scale, NQH_f, NKVH_f, T_f, K_padded_f]
            // Output:
            //   out:     [B, NQH, 1, HEAD_DIM]  (fp32 inside kernel; Swift casts)
            //
            // Layout matches mlx-swift's `sdpa_vector`:
            //   - threadgroup = BN(=32) simdgroups × BD(=32) lanes = 1024 threads
            //   - qk_per_thread = HEAD_DIM / BD; v_per_thread same
            //   - inner gather loop: `for i = simd_gid; i < K_padded; i += BN`
            //     → BN simdgroups process BN gather positions in parallel
            //   - `simd_sum` for QK reduction (no threadgroup barriers)
            //
            // Grid: (B * NQH * 1024, 1, 1). One threadgroup per (B, Q head).
            // GROUP_SIZE is unused at this layer; the group sharing was
            // dropped in F-71b after F-71a's barrier-bound regression.

            constexpr int BN = 32;
            constexpr int BD = 32;
            constexpr int qk_per_thread = HEAD_DIM / BD;
            constexpr int v_per_thread = HEAD_DIM / BD;

            const float scale = params[0];
            const uint NQH = (uint)params[1];
            const uint NKVH = (uint)params[2];
            const uint T = (uint)params[3];
            const uint K_padded = (uint)params[4];

            const uint qh_idx = threadgroup_position_in_grid.x;
            const uint simd_gid = simdgroup_index_in_threadgroup;
            const uint simd_lid = thread_index_in_simdgroup;
            const uint b = qh_idx / NQH;
            const uint qh = qh_idx % NQH;
            const uint kvh = qh / GROUP_SIZE;

            // Threadgroup state for cross-simd reduce at the end.
            threadgroup float outputs[BN * BD];
            threadgroup float max_scores[BN];
            threadgroup float sum_exp_scores[BN];

            // Per-thread Q register file: qk_per_thread elements pre-scaled.
            float q_local[qk_per_thread];
            float k_local[qk_per_thread];
            float o_local[v_per_thread];

            // Load Q (this thread owns lanes [simd_lid * qk_per_thread, ...])
            // from q[b, qh, 0, simd_lid * qk_per_thread + i].
            const uint q_base = (b * NQH + qh) * HEAD_DIM
                                 + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; i++) {
                q_local[i] = scale * (float)q[q_base + i];
            }
            for (int i = 0; i < v_per_thread; i++) {
                o_local[i] = 0.0f;
            }

            float max_score = -INFINITY;
            float sum_exp_score = 0.0f;

            const uint gather_base = (b * NKVH + kvh) * K_padded;

            // Each simdgroup processes a strided slice of K_padded.
            for (uint i = simd_gid; i < K_padded; i += BN) {
                const int pos = gather[gather_base + i];
                // Sentinel skip: -1 marks adjacent-dup positions deduped
                // upstream. simd-coherent branch — all lanes in a simdgroup
                // see the same pos, so the simd_sum below stays well-defined.
                if (pos < 0) continue;

                const uint kv_base = (b * NKVH + kvh) * T * HEAD_DIM
                                     + (uint)pos * HEAD_DIM
                                     + simd_lid * qk_per_thread;

                // Load qk_per_thread K elements
                for (int j = 0; j < qk_per_thread; j++) {
                    k_local[j] = (float)k[kv_base + j];
                }

                // Compute partial QK dot
                float score = 0.0f;
                for (int j = 0; j < qk_per_thread; j++) {
                    score += q_local[j] * k_local[j];
                }
                // simd-level reduction across BD=32 lanes — no barriers.
                score = simd_sum(score);

                // Online softmax update (every lane sees same score).
                const float new_max = fmax(max_score, score);
                const float factor = (max_score == -INFINITY)
                    ? 0.0f : exp(max_score - new_max);
                const float exp_score = exp(score - new_max);
                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;

                // V accumulate
                for (int j = 0; j < v_per_thread; j++) {
                    const float v_local = (float)v[kv_base + j];
                    o_local[j] = o_local[j] * factor + exp_score * v_local;
                }
            }

            // Cross-simdgroup reduce (mirrors mlx-swift sdpa_vector).
            if (simd_lid == 0) {
                max_scores[simd_gid] = max_score;
                sum_exp_scores[simd_gid] = sum_exp_score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            max_score = max_scores[simd_lid];
            const float new_max = simd_max(max_score);
            const float factor = exp(max_score - new_max);
            sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

            // Aggregate per-output-dim partials across simdgroups.
            const uint out_base = (b * NQH + qh) * HEAD_DIM
                                   + simd_gid * v_per_thread;
            for (int i = 0; i < v_per_thread; i++) {
                outputs[simd_lid * BD + simd_gid] = o_local[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float reduced = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                if (simd_lid == 0) {
                    out[out_base + i] = sum_exp_score == 0.0f
                        ? reduced
                        : (reduced / sum_exp_score);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """
        let k = MLXFast.metalKernel(
            name: "ra_group_sdpa_decode",
            inputNames: ["q", "k", "v", "gather", "params"],
            outputNames: ["out"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        groupSDPAKernel = k
        return k
    }

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

    /// F-73 fused build-mask kernel. Takes top-K block start positions
    /// (already computed via F-48) plus static/sliding params and
    /// writes the [1, 1, 1, T] additive attention mask directly. One
    /// thread per mask position; each checks membership against the
    /// union of static + sliding + fine_topK_blocks + coarse_topK_blocks
    /// and writes 0 or -inf.
    ///
    /// Replaces ~6 MLX ops (position expansion + concat + clip + scatter)
    /// with 1 kernel — F-73 measured 22.9ms / decode step is JUST the
    /// selector pipeline (cache overhead = 0.1ms), so this is the gap
    /// to dense.
    func getBuildMask() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = buildMaskKernel { return k }
        let header = """
            // F-73 fused build-mask kernel.

            """
        let source = """
            // Template constants: FINE_BS, COARSE_BS, NKVH, K_FINE, K_COARSE
            // Inputs:
            //   fine_starts:   [NKVH * K_FINE] int32 block start positions
            //   coarse_starts: [NKVH * K_COARSE] int32 block start positions
            //   params:        [T_f, staticInit_f, slidingWindow_f]
            // Output:
            //   mask:          [T] float — additive (0 valid, -inf masked)
            //
            // Grid: ((T + tg-1)/tg, 1, 1) threadgroups × (tg, 1, 1) threads.
            // Each thread covers one mask position; checks membership
            // against the cached topK arrays in threadgroup memory.

            const uint p = thread_position_in_grid.x;
            const uint T = (uint)params[0];
            if (p >= T) return;
            const uint staticInit = (uint)params[1];
            const uint slidingWindow = (uint)params[2];
            const uint sliding_start = (T > slidingWindow) ? (T - slidingWindow) : 0u;

            // Cache the topK start arrays in threadgroup memory. All threads
            // in the threadgroup share the same arrays — load cooperatively.
            threadgroup int fine_tg[NKVH * K_FINE];
            threadgroup int coarse_tg[NKVH * K_COARSE];

            const uint tid = thread_position_in_threadgroup.x;
            const uint tg_size = threads_per_threadgroup.x;
            for (uint i = tid; i < NKVH * K_FINE; i += tg_size) {
                fine_tg[i] = fine_starts[i];
            }
            for (uint i = tid; i < NKVH * K_COARSE; i += tg_size) {
                coarse_tg[i] = coarse_starts[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool valid = (p < staticInit) || (p >= sliding_start);

            if (!valid) {
                for (uint i = 0; i < NKVH * K_FINE && !valid; i++) {
                    const int start = fine_tg[i];
                    if ((int)p >= start && (int)p < start + (int)FINE_BS) {
                        valid = true;
                    }
                }
            }
            if (!valid) {
                for (uint i = 0; i < NKVH * K_COARSE && !valid; i++) {
                    const int start = coarse_tg[i];
                    if ((int)p >= start && (int)p < start + (int)COARSE_BS) {
                        valid = true;
                    }
                }
            }

            mask[p] = valid ? 0.0f : -INFINITY;
            """
        let k = MLXFast.metalKernel(
            name: "ra_build_mask",
            inputNames: ["fine_starts", "coarse_starts", "params"],
            outputNames: ["mask"],
            source: source,
            header: header,
            ensureRowContiguous: true
        )
        buildMaskKernel = k
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

/// F-71 — NSA-style group-centric fused sparse SDPA. One threadgroup
/// per (B, KV head); all `groupSize` Q heads of each KV group share
/// the per-head sorted gather and reuse K/V reads across the group.
///
/// Wins over F-59 mask path:
///   1. K/V reads scale with K_padded (per-head gather size), not T.
///   2. Each K/V byte is read once per KV group instead of `groupSize`
///      times.
///   3. No fp32 pre-cast — kernel reads native Q/K/V dtype and casts
///      on load. (F-69 pre-cast was the dominant overhead — 12+ GB
///      extra allocations per decode step on 14B-1M @ 32K.)
///
/// - Parameters:
///   - queries: `[B, nQH, 1, D]` post-RoPE decode queries.
///   - keys: `[B, nKVH, T, D]` full cached K.
///   - values: `[B, nKVH, T, D]` full cached V.
///   - perKVHeadGather: `[B, nKVH, K_padded]` int32 sorted per-KV-head
///     positions (with possible adjacent duplicates).
///   - scale: SDPA scale factor (typically `1/√D`).
/// - Returns: `[B, nQH, 1, D]` attention output, dtype = `queries.dtype`.
public func retrievalAttentionGroupSparseSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    perKVHeadGather: MLXArray,
    scale: Float
) -> MLXArray {
    precondition(queries.shape.count == 4, "queries must be [B, nQH, 1, D]")
    precondition(keys.shape.count == 4, "keys must be [B, nKVH, T, D]")
    precondition(values.shape.count == 4, "values must be [B, nKVH, T, D]")
    precondition(perKVHeadGather.shape.count == 3,
        "gather must be [B, nKVH, K_padded] (got \(perKVHeadGather.shape))")
    let B = queries.dim(0)
    let nQH = queries.dim(1)
    precondition(queries.dim(2) == 1, "decode-step L=1 only")
    let D = queries.dim(3)
    let nKVH = keys.dim(1)
    let T = keys.dim(2)
    precondition(perKVHeadGather.dim(0) == B && perKVHeadGather.dim(1) == nKVH,
        "gather batch/heads mismatch")
    let kPadded = perKVHeadGather.dim(2)
    precondition(keys.dim(3) == D, "K head_dim must match Q")
    precondition(values.dim(3) == D, "V head_dim must equal K (Dv==D)")
    precondition(nQH % nKVH == 0, "Q heads must be a multiple of KV heads")
    let groupSize = nQH / nKVH
    precondition([32, 64, 96, 128, 256].contains(D),
        "head_dim \(D) outside the supported template list")

    let params = MLXArray([
        scale,
        Float(nQH),
        Float(nKVH),
        Float(T),
        Float(kPadded),
    ])
    let gatherFlat = perKVHeadGather.reshaped(B * nKVH * kPadded)

    let kernel = _RAKernelCache.shared.getGroupSDPA()
    // F-71b layout: one threadgroup per (B, Q head). 1024 threads/tg
    // (32 simdgroups × 32 lanes), matching mlx-swift sdpa_vector.
    let TGSize = 1024
    let totalThreads = B * nQH * TGSize
    let outputs = kernel(
        [queries, keys, values, gatherFlat, params],
        template: [
            ("HEAD_DIM", D),
            ("GROUP_SIZE", groupSize),
        ],
        grid: (totalThreads, 1, 1),
        threadGroup: (TGSize, 1, 1),
        outputShapes: [[B, nQH, 1, D]],
        outputDTypes: [.float32]
    )
    return outputs[0].asType(queries.dtype)
}

/// F-76 — implicit-positions sparse SDPA. No mask, no gather array;
/// per-KV-head top-K block starts directly drive an inline static +
/// sliding + fine_blocks + coarse_blocks iteration.
///
/// - Parameters:
///   - queries: `[B, nQH, 1, D]`
///   - keys: `[B, nKVH, T, D]`
///   - values: `[B, nKVH, T, D]`
///   - fineStarts: `[B, nKVH, K_fine]` int32 block start positions
///   - coarseStarts: `[B, nKVH, K_coarse]` int32 block start positions
///   - staticInit, slidingWindow, fineBlockSize, coarseBlockSize: window params
///   - scale: SDPA scale (typically `1/√D`)
/// - Returns: `[B, nQH, 1, D]` output, dtype = `queries.dtype`.
public func retrievalAttentionImplicitSparseSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    fineStarts: MLXArray,
    coarseStarts: MLXArray,
    staticInit: Int,
    slidingWindow: Int,
    fineBlockSize: Int,
    coarseBlockSize: Int,
    scale: Float
) -> MLXArray {
    precondition(queries.shape.count == 4, "queries must be [B, nQH, 1, D]")
    precondition(keys.shape.count == 4, "keys must be [B, nKVH, T, D]")
    precondition(values.shape.count == 4, "values must be [B, nKVH, T, D]")
    let B = queries.dim(0)
    let nQH = queries.dim(1)
    let D = queries.dim(3)
    let nKVH = keys.dim(1)
    let T = keys.dim(2)
    precondition(queries.dim(2) == 1, "decode-step L=1 only")
    precondition(keys.dim(3) == D && values.dim(3) == D, "head_dim mismatch")
    precondition(fineStarts.shape == [B, nKVH, fineStarts.dim(2)], "fineStarts shape")
    precondition(coarseStarts.shape == [B, nKVH, coarseStarts.dim(2)], "coarseStarts shape")
    let kFine = fineStarts.dim(2)
    let kCoarse = coarseStarts.dim(2)
    precondition(nQH % nKVH == 0, "Q heads must be a multiple of KV heads")
    let groupSize = nQH / nKVH
    precondition([32, 64, 96, 128, 256].contains(D),
        "head_dim \(D) outside the supported template list")

    let params = MLXArray([scale, Float(T)])
    let fineFlat = fineStarts.reshaped(B * nKVH * kFine)
    let coarseFlat = coarseStarts.reshaped(B * nKVH * kCoarse)

    let kernel = _RAKernelCache.shared.getImplicitSparseSDPA()
    let TGSize = 1024
    let totalThreads = B * nQH * TGSize
    let outputs = kernel(
        [queries, keys, values, fineFlat, coarseFlat, params],
        template: [
            ("HEAD_DIM", D),
            ("GROUP_SIZE", groupSize),
            ("NQH", nQH),
            ("NKVH", nKVH),
            ("STATIC_INIT", staticInit),
            ("SLIDING_WINDOW", slidingWindow),
            ("K_FINE", kFine),
            ("FINE_BS", fineBlockSize),
            ("K_COARSE", kCoarse),
            ("COARSE_BS", coarseBlockSize),
        ],
        grid: (totalThreads, 1, 1),
        threadGroup: (TGSize, 1, 1),
        outputShapes: [[B, nQH, 1, D]],
        outputDTypes: [.float32]
    )
    return outputs[0].asType(queries.dtype)
}

/// F-75 — parallel fine+coarse score+topK wrapper. 2 × nKVH threadgroups,
/// fine and coarse run concurrently on the GPU.
public func retrievalAttentionParallelScoreTopK(
    projectedQ: MLXArray,
    fineFeatures: MLXArray,
    coarseFeatures: MLXArray,
    kFine: Int,
    kCoarse: Int,
    fineBlockSize: Int,
    coarseBlockSize: Int
) -> (fineStarts: MLXArray, coarseStarts: MLXArray) {
    precondition(projectedQ.shape.count == 2, "projectedQ must be [nKVH, contentDim]")
    precondition(fineFeatures.shape.count == 3, "fineFeatures must be [nKVH, nBlocks, D]")
    precondition(coarseFeatures.shape.count == 3, "coarseFeatures must be [nKVH, nBlocks, D]")
    let nKVH = projectedQ.dim(0)
    let contentDim = projectedQ.dim(1)
    let nFine = fineFeatures.dim(1)
    let nCoarse = coarseFeatures.dim(1)
    precondition(fineFeatures.dim(0) == nKVH && coarseFeatures.dim(0) == nKVH,
        "head count mismatch")
    precondition(fineFeatures.dim(2) == contentDim && coarseFeatures.dim(2) == contentDim,
        "content dim mismatch")
    precondition(nFine > 0 && (nFine & (nFine - 1)) == 0 && nFine <= 1024,
        "nFine \(nFine) must be a power of 2 ≤ 1024")
    precondition(nCoarse > 0 && (nCoarse & (nCoarse - 1)) == 0 && nCoarse <= 1024,
        "nCoarse \(nCoarse) must be a power of 2 ≤ 1024")

    let tgSize = min(1024, max(nFine, nCoarse))
    let kernel = _RAKernelCache.shared.getParallelScore()
    let outputs = kernel(
        [projectedQ, fineFeatures, coarseFeatures],
        template: [
            ("NKVH", nKVH),
            ("CONTENT_DIM", contentDim),
            ("N_FINE", nFine),
            ("K_FINE", kFine),
            ("FINE_BS", fineBlockSize),
            ("N_COARSE", nCoarse),
            ("K_COARSE", kCoarse),
            ("COARSE_BS", coarseBlockSize),
            ("TG_SIZE", tgSize),
        ],
        grid: (2 * nKVH * tgSize, 1, 1),
        threadGroup: (tgSize, 1, 1),
        outputShapes: [[nKVH, kFine], [nKVH, kCoarse]],
        outputDTypes: [.int32, .int32]
    )
    return (fineStarts: outputs[0], coarseStarts: outputs[1])
}

/// F-73 — fused build-mask kernel wrapper. Takes per-KV-head top-K
/// block start arrays (output of F-48 scoreTopKFused) plus the static/
/// sliding window params, returns the [1, 1, 1, T] additive attention
/// mask. Replaces ~6 MLX ops (position expansion + concat + clip +
/// `MLXArray.full` + scatter) with one Metal kernel launch.
///
/// - Parameters:
///   - fineStarts: `[nKVH, K_fine]` int32 block-start positions
///   - coarseStarts: `[nKVH, K_coarse]` int32 block-start positions
///   - T: total cache positions
///   - staticInit: static-initial window size
///   - slidingWindow: trailing-sliding window size
///   - fineBS: fine block size in tokens
///   - coarseBS: coarse block size in tokens
///   - outputDtype: dtype for the returned mask (typically `K.dtype`)
/// - Returns: `[1, 1, 1, T]` additive mask (0 at valid, -inf at masked).
public func retrievalAttentionBuildMaskFused(
    fineStarts: MLXArray,
    coarseStarts: MLXArray,
    T: Int,
    staticInit: Int,
    slidingWindow: Int,
    fineBS: Int,
    coarseBS: Int,
    outputDtype: DType
) -> MLXArray {
    precondition(fineStarts.shape.count == 2, "fineStarts must be [nKVH, K_fine]")
    precondition(coarseStarts.shape.count == 2, "coarseStarts must be [nKVH, K_coarse]")
    let nKVH = fineStarts.dim(0)
    let kFine = fineStarts.dim(1)
    let kCoarse = coarseStarts.dim(1)
    precondition(coarseStarts.dim(0) == nKVH, "head count mismatch")

    let params = MLXArray([
        Float(T), Float(staticInit), Float(slidingWindow),
    ])
    let fineFlat = fineStarts.reshaped(nKVH * kFine)
    let coarseFlat = coarseStarts.reshaped(nKVH * kCoarse)

    let kernel = _RAKernelCache.shared.getBuildMask()
    let tgSize = 256
    // Round up grid to threadgroup multiple so the kernel only does
    // bounds-check on `p`.
    let gridX = ((T + tgSize - 1) / tgSize) * tgSize
    let outputs = kernel(
        [fineFlat, coarseFlat, params],
        template: [
            ("FINE_BS", fineBS),
            ("COARSE_BS", coarseBS),
            ("NKVH", nKVH),
            ("K_FINE", kFine),
            ("K_COARSE", kCoarse),
        ],
        grid: (gridX, 1, 1),
        threadGroup: (tgSize, 1, 1),
        outputShapes: [[1, 1, 1, T]],
        outputDTypes: [.float32]
    )
    return outputs[0].asType(outputDtype)
}

/// F-74 — fused projectQ + scoreTopK_fine + scoreTopK_coarse wrapper.
/// One Metal kernel call replaces 3 MLX op chains
/// (`projectQueriesBatched` + 2× `computeTopKBlockStarts`).
///
/// - Parameters:
///   - q: `[nQH, dHead]` float32 (post-RoPE; from `q.take(cachedHeadIdx)`
///     etc. — the kernel reads strided rep-per-group)
///   - jlW: `[dHead, contentDim]` float32 JL projection
///   - fineFeatures: `[nKVH, nFineBlocks, contentDim]` float32
///   - coarseFeatures: `[nKVH, nCoarseBlocks, contentDim]` float32
///   - groupSize: nQH / nKVH
///   - kFine: top-K count for fine
///   - kCoarse: top-K count for coarse
///   - fineBlockSize: token-block stride for fine
///   - coarseBlockSize: token-block stride for coarse
/// - Returns: `(fineStarts [nKVH, kFine], coarseStarts [nKVH, kCoarse])`
public func retrievalAttentionSelectorBundleFused(
    q: MLXArray,
    jlW: MLXArray,
    fineFeatures: MLXArray,
    coarseFeatures: MLXArray,
    groupSize: Int,
    kFine: Int,
    kCoarse: Int,
    fineBlockSize: Int,
    coarseBlockSize: Int
) -> (fineStarts: MLXArray, coarseStarts: MLXArray) {
    precondition(q.shape.count == 2, "q must be [nQH, dHead]")
    precondition(jlW.shape.count == 2, "jl_w must be [dHead, contentDim]")
    precondition(fineFeatures.shape.count == 3, "fineFeatures must be [nKVH, nBlocks, D]")
    precondition(coarseFeatures.shape.count == 3, "coarseFeatures must be [nKVH, nBlocks, D]")
    let dHead = jlW.dim(0)
    let contentDim = jlW.dim(1)
    let nKVH = fineFeatures.dim(0)
    let nFine = fineFeatures.dim(1)
    let nCoarse = coarseFeatures.dim(1)
    precondition(coarseFeatures.dim(0) == nKVH, "head count mismatch")
    precondition(fineFeatures.dim(2) == contentDim, "content dim mismatch")
    precondition(coarseFeatures.dim(2) == contentDim, "content dim mismatch")
    precondition(q.dim(0) == nKVH * groupSize, "q rows must be nKVH * groupSize")
    precondition(q.dim(1) == dHead, "q D mismatch")
    // Both block counts must be powers of 2 ≤ 1024 (tree-reduce constraint).
    precondition(nFine > 0 && (nFine & (nFine - 1)) == 0 && nFine <= 1024,
        "nFine \(nFine) must be a power of 2 ≤ 1024")
    precondition(nCoarse > 0 && (nCoarse & (nCoarse - 1)) == 0 && nCoarse <= 1024,
        "nCoarse \(nCoarse) must be a power of 2 ≤ 1024")

    let nQH = nKVH * groupSize
    let tgSize = min(1024, max(nFine, nCoarse))

    let kernel = _RAKernelCache.shared.getSelectorBundle()
    let outputs = kernel(
        [q, jlW, fineFeatures, coarseFeatures],
        template: [
            ("D_HEAD", dHead),
            ("CONTENT_DIM", contentDim),
            ("GROUP_SIZE", groupSize),
            ("NQH", nQH),
            ("NKVH", nKVH),
            ("N_FINE", nFine),
            ("K_FINE", kFine),
            ("FINE_BS", fineBlockSize),
            ("N_COARSE", nCoarse),
            ("K_COARSE", kCoarse),
            ("COARSE_BS", coarseBlockSize),
        ],
        grid: (nKVH * tgSize, 1, 1),
        threadGroup: (tgSize, 1, 1),
        outputShapes: [[nKVH, kFine], [nKVH, kCoarse]],
        outputDTypes: [.int32, .int32]
    )
    return (fineStarts: outputs[0], coarseStarts: outputs[1])
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
