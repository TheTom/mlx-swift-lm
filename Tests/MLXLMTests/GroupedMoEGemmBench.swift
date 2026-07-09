// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// IDEAL grouped MoE Q4 GEMM micro-bench — the reference kernel for the CUDA port.
//
// THE PROBLEM (CUDA / spark stack):
//   Nemotron MoE on CUDA runs a PER-EXPERT loop: for each of 128 experts × 23
//   MoE layers it does a separate Q4-dequant kernel + a separate GEMM →
//   ~10.8k kernel launches/forward, the dequant kernels launch-latency-bound
//   (measured ~0.053 TFLOP/s, ~5% efficiency). vLLM fuses this into a GROUPED
//   kernel: gather tokens by expert → ONE grouped GEMM over all experts (Q4
//   dequant fused into the MMA prologue, no separate dequant pass, no f16
//   materialization) → scatter.
//
// On Metal, MLX's gatherQuantizedMM with sortedIndices=true already IS that
// grouped kernel (affine_gather_qmm_rhs in quantized.cpp). So this bench
// MEASURES THE IDEAL on hardware MMA, to:
//   1. quantify how much collapsing 128 launches → 1 + fusing dequant recovers,
//   2. set the achievable-MFU target the CUDA grouped Q4 GEMM must hit,
//   3. characterize the reuse wall at the prefill m (rows/expert).
//
// DIMS (Nemotron Cascade-2 30B-A3B MoE block):
//   hidden  = 2688   (input dim of gate/up, output dim of down)
//   inter   = 1856   (moe_intermediate_size — output of gate/up, input of down)
//   n_exp   = 128
//   top-k   = 6
//   S ∈ {2048, 4096, 8192} prefill tokens → routed rows = S*6
//     → rows/expert (uniform) m = 96 / 192 / 384.
//
// We isolate the dominant GEMM of the MoE — the up/gate projection
//   x[rows, hidden] @ Wq[expert, inter, hidden]^T → [rows, inter]
// and the down projection
//   h[rows, inter]  @ Wq[expert, hidden, inter]^T → [rows, hidden]
// quantized Q4 affine group_size=64. This is exactly SwitchLinear's
// gatherQuantizedMM call (see Libraries/MLXLMCommon/SwitchLayers.swift).
//
// VARIANTS compared at each m:
//   GROUPED : one gatherQuantizedMM over ALL experts, rhsIndices sorted,
//             sortedIndices=true (fused dequant, ONE dispatch). The ideal.
//   PER-EXP : loop 128 experts, each a dequantize(Wq_e)+quantizedMM on its
//             own row slice (separate dequant materialization + separate GEMM
//             per expert) — the CUDA per-expert-loop analogue.
//   DEQUANT-only: just the 128 separate dequantize() calls, to isolate how
//             much of the per-expert cost is the un-fused dequant pass.
//
// Reports TFLOP/s + dispatch (launch) count for each → recovery factor.
//
// Gated. Run:
//   RUN_GROUPED_MOE_BENCH=1 swift test --filter groupedMoEGemmBench 2>&1 \
//     | tee /tmp/grouped_moe_bench.out

import Foundation
import MLX
import Testing

@Suite("Grouped MoE Q4 GEMM bench — CUDA-port reference", .serialized)
struct GroupedMoEGemmBench {

    private static let outPath = "/tmp/grouped_moe_bench.out"

    private static func log(_ s: String) {
        Swift.print(s)
        fflush(stdout)
        if let h = FileHandle(forWritingAtPath: outPath) {
            h.seekToEndOfFile()
            h.write((s + "\n").data(using: .utf8)!)
            h.closeFile()
        }
    }

    @Test func groupedMoEGemmBench() throws {
        guard ProcessInfo.processInfo.environment["RUN_GROUPED_MOE_BENCH"] == "1"
        else { return }
        FileManager.default.createFile(atPath: Self.outPath, contents: Data(), attributes: nil)

        // --- Nemotron MoE dims ---
        let hidden = 2688
        let inter = 1856
        let nExp = 128
        let topK = 6
        let groupSize = 64
        let bits = 4
        let dtype: DType = .float16

        let iters = Int(ProcessInfo.processInfo.environment["BENCH_ITERS"] ?? "30") ?? 30
        let warmup = 5

        Self.log("=== Grouped MoE Q4 GEMM bench (Nemotron dims) ===")
        Self.log("hidden=\(hidden) inter=\(inter) nExp=\(nExp) topK=\(topK) Q\(bits) gs=\(groupSize) \(dtype)")
        Self.log("device: Metal hardware-MMA (M5 Max)")
        Self.log("")

        MLXRandom.seed(7)

        // Quantize the two expert weight banks once.
        //   Wgate/up : [nExp, inter, hidden]  (out=inter, in=hidden), transpose=true
        //   Wdown    : [nExp, hidden, inter]
        func makeBank(_ outDim: Int, _ inDim: Int)
            -> (wq: MLXArray, scales: MLXArray, biases: MLXArray?, w: MLXArray)
        {
            let scale = Float(1.0 / Double(inDim).squareRoot())
            let w = MLXRandom.uniform(low: -scale, high: scale, [nExp, outDim, inDim]).asType(dtype)
            let (wq, s, b) = MLX.quantized(w, groupSize: groupSize, bits: bits, mode: .affine)
            eval(wq, s); if let b { eval(b) }
            return (wq, s, b, w)
        }

        let up = makeBank(inter, hidden)    // x[rows,hidden] @ up^T -> [rows,inter]
        let down = makeBank(hidden, inter)  // h[rows,inter]  @ down^T -> [rows,hidden]
        eval(up.w, down.w)
        MLX.GPU.clearCache()
        Self.log(String(format: "[alloc] expert banks quantized. active=%.0f MB",
                        Double(MLX.GPU.activeMemory) / 1024 / 1024))
        Self.log("")

        // ---- timing helper. Returns median ms. ----
        func benchMs(_ name: String, _ op: () -> [MLXArray]) -> Double {
            for _ in 0..<warmup { let o = op(); eval(o) }
            var samples = [Double]()
            samples.reserveCapacity(iters)
            for _ in 0..<iters {
                let t0 = Date()
                let o = op()
                eval(o)
                samples.append(Date().timeIntervalSince(t0) * 1000.0)
            }
            samples.sort()
            let med = samples[samples.count / 2]
            let p10 = samples[max(0, samples.count / 10)]
            let pad = name.padding(toLength: 46, withPad: " ", startingAt: 0)
            Self.log(String(format: "  \(pad) median=%8.4f ms  p10=%8.4f  (n=%d)", med, p10, iters))
            return med
        }

        var roofTF = 0.0
        // ---- ROOFLINE anchor: single dense Q4 GEMM, reuse-saturated ----
        // One quantizedMM at M=8192, K=2688, N=1856 (the up-proj dims, no
        // gather, all rows share one weight = maximal weight reuse). This is
        // the achievable Q4 f16-accumulate MMA ceiling on THIS hardware; we
        // express grouped MFU relative to it (self-calibrated, no guessed peak).
        do {
            let Mr = 8192
            let xr = MLXRandom.uniform(low: -1, high: 1, [Mr, hidden]).asType(dtype)
            eval(xr)
            // reuse one expert's quantized weight as a plain (non-gathered) bank
            let wq0 = up.wq[0]; let s0 = up.scales[0]; let b0 = up.biases?[0]
            eval(wq0, s0); if let b0 { eval(b0) }
            let rms = benchMs("ROOFLINE dense Q4 GEMM [\(Mr)x\(hidden)]@[\(hidden),\(inter)]") {
                [MLX.quantizedMM(xr, wq0, scales: s0, biases: b0, transpose: true,
                                 groupSize: groupSize, bits: bits, mode: .affine)]
            }
            let rtf = (2.0 * Double(Mr) * Double(inter) * Double(hidden) / 1e9) / rms
            Self.log(String(format: "  → ROOFLINE achievable Q4 MMA = %.2f TFLOP/s (reuse-saturated)", rtf))
            Self.log("")
            roofTF = rtf
        }

        // FLOPs for one projection over `rows` total routed rows.
        //   each row: out*in MACs = 2*out*in FLOPs. Same total whether grouped
        //   or per-expert (the work is identical; only the dispatch differs).
        func gflops(rows: Int, outDim: Int, inDim: Int) -> Double {
            2.0 * Double(rows) * Double(outDim) * Double(inDim) / 1e9
        }

        struct Row {
            let S: Int, m: Int, rows: Int
            let groupedMs: Double, perExpMs: Double, dequantMs: Double
            let groupedTF: Double, perExpTF: Double
            let groupedLaunch: Int, perExpLaunch: Int
        }
        var rowsOut = [Row]()

        for S in [2048, 4096, 8192] {
            let rows = S * topK
            let m = rows / nExp  // uniform rows/expert
            Self.log("──────────────────────────────────────────────────────────────")
            Self.log("S=\(S)  routed rows=\(rows)  rows/expert m=\(m)")
            Self.log("")

            // Activations: [rows, hidden] for up, [rows, inter] for down.
            // For GROUPED: rows are pre-sorted by expert (gatherSort already
            // done upstream in SwitchGLU); each expert owns a contiguous block
            // of `m` rows. rhsIndices is the sorted per-row expert id.
            let xUp = MLXRandom.uniform(low: -1, high: 1, [rows, hidden]).asType(dtype)
            // sorted expert ids: [0]*m, [1]*m, ... matches the gatherSort output.
            var idsArr = [Int32]()
            idsArr.reserveCapacity(rows)
            for e in 0..<nExp { for _ in 0..<m { idsArr.append(Int32(e)) } }
            // pad tail if rows not divisible (here always divisible: rows=S*6, nExp=128)
            while idsArr.count < rows { idsArr.append(Int32(nExp - 1)) }
            let ids = MLXArray(idsArr)
            eval(xUp, ids)

            // gatherQuantizedMM wants x shaped [..., 1, K] with rhsIndices
            // selecting the expert per leading position. SwitchLinear expands
            // x to [rows, 1, hidden]; here we mirror that.
            let xUp3 = MLX.expandedDimensions(xUp, axis: -2)  // [rows,1,hidden]

            // -------- GROUPED up-proj (the IDEAL): one dispatch --------
            let groupedUpMs = benchMs("GROUPED up   [\(rows)x\(hidden)]@Wq[\(nExp),\(inter),\(hidden)]") {
                [MLX.gatherQuantizedMM(
                    xUp3, up.wq, scales: up.scales, biases: up.biases,
                    rhsIndices: ids, transpose: true,
                    groupSize: groupSize, bits: bits, mode: .affine,
                    sortedIndices: true)]
            }

            // pre-make down activations [rows, inter]
            let hDown = MLXRandom.uniform(low: -1, high: 1, [rows, inter]).asType(dtype)
            eval(hDown)
            let hDown3 = MLX.expandedDimensions(hDown, axis: -2)
            let groupedDownMs = benchMs("GROUPED down [\(rows)x\(inter)]@Wq[\(nExp),\(hidden),\(inter)]") {
                [MLX.gatherQuantizedMM(
                    hDown3, down.wq, scales: down.scales, biases: down.biases,
                    rhsIndices: ids, transpose: true,
                    groupSize: groupSize, bits: bits, mode: .affine,
                    sortedIndices: true)]
            }
            let groupedMs = groupedUpMs + groupedDownMs
            // dispatch count: 1 kernel each (the grouped rhs kernel).
            let groupedLaunch = 2

            // -------- PER-EXPERT loop (the CUDA analogue) --------
            // For each expert e: slice its m rows, dequantize its weight,
            // quantizedMM (no, to be faithful to the "separate dequant + GEMM"
            // CUDA path we dequantize to f16 then plain matmul). Separate
            // dispatches per expert per projection.
            // dispatch count ≈ nExp * (1 dequant + 1 matmul) per projection.
            func perExpertProj(
                xrows: MLXArray, outDim: Int, inDim: Int,
                wq: MLXArray, scales: MLXArray, biases: MLXArray?
            ) -> [MLXArray] {
                var outs = [MLXArray]()
                outs.reserveCapacity(nExp)
                for e in 0..<nExp {
                    let r0 = e * m
                    let xe = xrows[r0 ..< (r0 + m)]                 // [m, inDim]
                    let we = MLX.dequantized(                        // separate dequant pass
                        wq[e], scales: scales[e], biases: biases?[e],
                        groupSize: groupSize, bits: bits, mode: .affine)  // [outDim, inDim]
                    let ye = MLX.matmul(xe, we.swappedAxes(-1, -2))  // separate GEMM [m,outDim]
                    outs.append(ye)
                }
                return outs
            }

            let perExpUpMs = benchMs("PER-EXP up   (128× dequant+GEMM)") {
                perExpertProj(xrows: xUp, outDim: inter, inDim: hidden,
                              wq: up.wq, scales: up.scales, biases: up.biases)
            }
            let perExpDownMs = benchMs("PER-EXP down (128× dequant+GEMM)") {
                perExpertProj(xrows: hDown, outDim: hidden, inDim: inter,
                              wq: down.wq, scales: down.scales, biases: down.biases)
            }
            let perExpMs = perExpUpMs + perExpDownMs
            let perExpLaunch = nExp * 2 * 2  // (dequant+matmul) × nExp × 2 projections

            // -------- DEQUANT-only (isolate un-fused dequant cost) --------
            let dequantUpMs = benchMs("DEQUANT-only up   (128× dequantize)") {
                var outs = [MLXArray]()
                for e in 0..<nExp {
                    outs.append(MLX.dequantized(
                        up.wq[e], scales: up.scales[e], biases: up.biases?[e],
                        groupSize: groupSize, bits: bits, mode: .affine))
                }
                return outs
            }
            let dequantDownMs = benchMs("DEQUANT-only down (128× dequantize)") {
                var outs = [MLXArray]()
                for e in 0..<nExp {
                    outs.append(MLX.dequantized(
                        down.wq[e], scales: down.scales[e], biases: down.biases?[e],
                        groupSize: groupSize, bits: bits, mode: .affine))
                }
                return outs
            }
            let dequantMs = dequantUpMs + dequantDownMs

            // TFLOP/s (up+down combined work).
            let totalGF = gflops(rows: rows, outDim: inter, inDim: hidden)
                + gflops(rows: rows, outDim: hidden, inDim: inter)
            let groupedTF = totalGF / groupedMs   // GF / ms = TF/s
            let perExpTF = totalGF / perExpMs

            Self.log("")
            Self.log(String(format: "  GROUPED total  = %8.4f ms  → %6.2f TFLOP/s   (%d dispatches)",
                            groupedMs, groupedTF, groupedLaunch))
            Self.log(String(format: "  PER-EXP total  = %8.4f ms  → %6.3f TFLOP/s   (~%d dispatches)",
                            perExpMs, perExpTF, perExpLaunch))
            Self.log(String(format: "  DEQUANT-only   = %8.4f ms  (un-fused dequant pass alone)", dequantMs))
            Self.log(String(format: "  → grouped speedup vs per-expert: %.1fx", perExpMs / groupedMs))
            Self.log(String(format: "  → launch collapse: %d → %d  (%.0fx fewer dispatches)",
                            perExpLaunch, groupedLaunch, Double(perExpLaunch) / Double(groupedLaunch)))
            Self.log("")

            rowsOut.append(Row(
                S: S, m: m, rows: rows,
                groupedMs: groupedMs, perExpMs: perExpMs, dequantMs: dequantMs,
                groupedTF: groupedTF, perExpTF: perExpTF,
                groupedLaunch: groupedLaunch, perExpLaunch: perExpLaunch))
        }

        // ---- summary ----
        Self.log("══════════════════════════ SUMMARY ══════════════════════════")
        Self.log("S      m    | GROUPED TF/s  (disp) | PER-EXP TF/s (disp) | speedup | dequant ms")
        for r in rowsOut {
            Self.log(String(
                format: "%-6d %-4d | %8.2f      (%d)  | %7.3f    (%d) | %5.1fx  | %7.3f",
                r.S, r.m, r.groupedTF, r.groupedLaunch,
                r.perExpTF, r.perExpLaunch, r.perExpMs / r.groupedMs, r.dequantMs))
        }
        Self.log("")
        // MFU vs the self-calibrated achievable-Q4-MMA roofline (reuse-saturated
        // dense GEMM on this hardware). This is the fraction of the achievable
        // ceiling that the grouped MoE captures at each prefill m.
        if roofTF > 0 {
            Self.log(String(format: "MFU vs achievable Q4 roofline (%.1f TFLOP/s, reuse-saturated):", roofTF))
            for r in rowsOut {
                Self.log(String(format: "  S=%-5d m=%-4d  grouped MFU = %5.1f%%   per-exp MFU = %5.1f%%",
                                r.S, r.m, 100.0 * r.groupedTF / roofTF, 100.0 * r.perExpTF / roofTF))
            }
        }
        Self.log("=== END ===")
    }
}
