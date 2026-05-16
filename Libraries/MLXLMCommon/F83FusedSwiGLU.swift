// F-83 W2: Fused dual-QGEMV (gate+up) + SwiGLU activation Metal kernel.
//
// Replaces the post-`gate_up_proj` split + compiled silu*mul tail with a
// single Metal dispatch. The qgemv math mirrors MLX's `qmv_fast_impl`
// (4-bit packed, per-group scale+bias, 2 SIMDs × 4 results_per_simd =
// 8 paired outputs per threadgroup). Each thread does BOTH the gate
// and up dot products in lockstep over the same x slice, then writes
// `silu(gate) * up` to the output. Saves the silu*mul kernel dispatch.

import Foundation
import MLX

public enum F83FusedSwiGLU {

    private nonisolated(unsafe) static var cachedKernel: MLXFast.MLXFastKernel?
    private static let lock = NSLock()

    private static func kernel() -> MLXFast.MLXFastKernel {
        lock.lock()
        defer { lock.unlock() }
        if let k = cachedKernel { return k }

        // Layout (mirrors MLX qmv_fast for 4-bit):
        //   packs_per_thread = 2, values_per_thread = 16
        //   num_simdgroups   = 2, results_per_simdgroup = 4
        //   threads_per_tg   = 64 (2 × 32)
        //   block_size       = 16 × 32 = 512 (K processed per inner iter)
        //   bytes_per_pack   = 4, pack_factor = 8 (4-bit)
        //
        // Each TG handles 8 paired columns:
        //   col_base = tg.y * 8 + simd_gid * 4
        //   gate row = col_base + r,        r in 0..4
        //   up   row = N + col_base + r
        //
        // The K loop loads 16 x values once per thread per iter, then does
        // 4 gate qdot + 4 up qdot against those x values. simd_sum at end,
        // simd_lid==0 writes silu(gate)*up for each of its 4 cols.
        let source = """
            constexpr int BITS = 4;
            constexpr int PACK_FACTOR = 32 / BITS;          // 8
            constexpr int BYTES_PER_PACK = 4;
            constexpr int PACKS_PER_THREAD = 2;
            constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;  // 16
            constexpr int NUM_SIMDGROUPS = 2;
            constexpr int RESULTS_PER_SIMDGROUP = 4;
            constexpr int SIMD_SIZE = 32;
            constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;          // 512
            constexpr int ROWS_PER_TG = NUM_SIMDGROUPS * RESULTS_PER_SIMDGROUP; // 8

            int in_vec_size = K;
            int out_vec_size = N;                              // N = intermediate (per side)
            int in_vec_size_w = in_vec_size * BYTES_PER_PACK / PACK_FACTOR;   // K/2 bytes per row
            int in_vec_size_g = in_vec_size / GROUP;           // groups per row

            uint simd_gid = simdgroup_index_in_threadgroup;    // 0..1
            uint simd_lid = thread_index_in_simdgroup;         // 0..31
            uint tg_y = threadgroup_position_in_grid.y;        // 0..N/8
            uint tg_x = threadgroup_position_in_grid.x;        // 0..M

            int col_base = (int)tg_y * ROWS_PER_TG + (int)simd_gid * RESULTS_PER_SIMDGROUP;

            if (col_base >= out_vec_size) return;

            // Per-thread x slice: VALUES_PER_THREAD floats pre-scaled per qdot trick.
            float x_thread[VALUES_PER_THREAD];

            // Pointer setup. gu_w is uint32 (mlx packing); reinterpret as uint8.
            const device uint8_t* ws_base = (const device uint8_t*)gu_w;
            const device uint8_t* gate_ws = ws_base
                + (uint)col_base * (uint)in_vec_size_w
                + (uint)simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;
            const device uint8_t* up_ws = ws_base
                + ((uint)N + (uint)col_base) * (uint)in_vec_size_w
                + (uint)simd_lid * PACKS_PER_THREAD * BYTES_PER_PACK;

            int scale_step_per_thread = GROUP / VALUES_PER_THREAD;
            // For group=64, values=16 → step=4. Each thread covers 16 elems = up to 1 group transition.
            const device T* gate_s = gu_s + (uint)col_base * (uint)in_vec_size_g + (uint)simd_lid / scale_step_per_thread;
            const device T* gate_b = gu_b + (uint)col_base * (uint)in_vec_size_g + (uint)simd_lid / scale_step_per_thread;
            const device T* up_s   = gu_s + ((uint)N + (uint)col_base) * (uint)in_vec_size_g + (uint)simd_lid / scale_step_per_thread;
            const device T* up_b   = gu_b + ((uint)N + (uint)col_base) * (uint)in_vec_size_g + (uint)simd_lid / scale_step_per_thread;
            const device T* x_ptr = x + tg_x * in_vec_size + simd_lid * VALUES_PER_THREAD;

            float gate_acc[RESULTS_PER_SIMDGROUP] = {0};
            float up_acc[RESULTS_PER_SIMDGROUP] = {0};

            for (int k = 0; k < in_vec_size; k += BLOCK_SIZE) {
                // Load 16 x values pre-scaled per quad (matches MLX 4-bit load_vector).
                float sum = 0.0f;
                for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
                    float x0 = (float)x_ptr[i];
                    float x1 = (float)x_ptr[i + 1];
                    float x2 = (float)x_ptr[i + 2];
                    float x3 = (float)x_ptr[i + 3];
                    sum += x0 + x1 + x2 + x3;
                    x_thread[i]     = x0;
                    x_thread[i + 1] = x1 / 16.0f;
                    x_thread[i + 2] = x2 / 256.0f;
                    x_thread[i + 3] = x3 / 4096.0f;
                }

                // Gate + up dot product for each of 4 rows.
                for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
                    // ---- Gate ----
                    {
                        const device uint16_t* w16 = (const device uint16_t*)(gate_ws + row * in_vec_size_w);
                        float gs = (float)gate_s[row * in_vec_size_g];
                        float gb = (float)gate_b[row * in_vec_size_g];
                        float accum = 0.0f;
                        // values_per_thread/4 = 4 quads; each w16 holds 4 4-bit weights packed.
                        for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
                            accum +=
                                (x_thread[4 * i + 0] * (float)(w16[i] & 0x000f) +
                                 x_thread[4 * i + 1] * (float)(w16[i] & 0x00f0) +
                                 x_thread[4 * i + 2] * (float)(w16[i] & 0x0f00) +
                                 x_thread[4 * i + 3] * (float)(w16[i] & 0xf000));
                        }
                        gate_acc[row] += gs * accum + sum * gb;
                    }
                    // ---- Up ----
                    {
                        const device uint16_t* w16 = (const device uint16_t*)(up_ws + row * in_vec_size_w);
                        float us = (float)up_s[row * in_vec_size_g];
                        float ub = (float)up_b[row * in_vec_size_g];
                        float accum = 0.0f;
                        for (int i = 0; i < VALUES_PER_THREAD / 4; i++) {
                            accum +=
                                (x_thread[4 * i + 0] * (float)(w16[i] & 0x000f) +
                                 x_thread[4 * i + 1] * (float)(w16[i] & 0x00f0) +
                                 x_thread[4 * i + 2] * (float)(w16[i] & 0x0f00) +
                                 x_thread[4 * i + 3] * (float)(w16[i] & 0xf000));
                        }
                        up_acc[row] += us * accum + sum * ub;
                    }
                }

                // Advance pointers.
                gate_ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
                up_ws   += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
                gate_s  += BLOCK_SIZE / GROUP;
                gate_b  += BLOCK_SIZE / GROUP;
                up_s    += BLOCK_SIZE / GROUP;
                up_b    += BLOCK_SIZE / GROUP;
                x_ptr   += BLOCK_SIZE;
            }

            // SIMD reduce + swiglu epilogue.
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; row++) {
                float g = simd_sum(gate_acc[row]);
                float u = simd_sum(up_acc[row]);
                if (simd_lid == 0) {
                    float sig = 1.0f / (1.0f + exp(-g));
                    out[tg_x * out_vec_size + col_base + row] = (T)((g * sig) * u);
                }
            }
            """

        let k = MLXFast.metalKernel(
            name: "f83_fused_qswiglu_v2",
            inputNames: ["x", "gu_w", "gu_s", "gu_b"],
            outputNames: ["out"],
            source: source
        )
        cachedKernel = k
        return k
    }

    /// Run fused dual-QGEMV + SwiGLU. Returns `[..., intermediate]`.
    public static func callAsFunction(
        x: MLXArray,
        gateUpWeight: MLXArray,
        gateUpScales: MLXArray,
        gateUpBiases: MLXArray,
        intermediate: Int,
        hiddenIn: Int,
        groupSize: Int = 64
    ) -> MLXArray {
        precondition(hiddenIn % 512 == 0, "F83FusedSwiGLU: K must be divisible by BLOCK_SIZE=512")
        precondition(hiddenIn % groupSize == 0, "F83FusedSwiGLU: K must be divisible by groupSize")
        precondition(intermediate % 8 == 0, "F83FusedSwiGLU: N must be divisible by ROWS_PER_TG=8")
        precondition(gateUpWeight.dim(0) == 2 * intermediate, "weight rows must equal 2*N")

        let originalShape = x.shape
        precondition(originalShape.last! == hiddenIn, "x last dim must equal hiddenIn")

        let m: Int
        let xFlat: MLXArray
        if originalShape.count > 2 {
            m = originalShape.dropLast().reduce(1, *)
            xFlat = x.reshaped(m, hiddenIn)
        } else if originalShape.count == 2 {
            m = originalShape[0]
            xFlat = x
        } else {
            m = 1
            xFlat = x.reshaped(1, hiddenIn)
        }

        let outDType = x.dtype
        let kernel = kernel()
        let numTGsY = intermediate / 8
        let out = kernel(
            [xFlat, gateUpWeight, gateUpScales, gateUpBiases],
            template: [
                ("T", outDType),
                ("N", intermediate),
                ("K", hiddenIn),
                ("GROUP", groupSize),
            ],
            grid: (m * 64, numTGsY, 1),
            threadGroup: (64, 1, 1),
            outputShapes: [[m, intermediate]],
            outputDTypes: [outDType],
            verbose: ProcessInfo.processInfo.environment["F83_KERNEL_VERBOSE"] == "1"
        )[0]

        if originalShape.count == 2 {
            return out
        } else if originalShape.count == 1 {
            return out.reshaped([intermediate])
        } else {
            return out.reshaped(Array(originalShape.dropLast()) + [intermediate])
        }
    }
}
