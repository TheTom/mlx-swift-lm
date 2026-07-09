// Research bench: resident-Q8 experts for the Nemotron MoE prefill GEMM.
//
// CONTEXT (the lever): the spark f16-resident-expert verdict was that
// argmax stays EXACT but e2e only +7.9% because moe_experts is now
// WEIGHT-READ BANDWIDTH-bound — its TFLOP/s halved (30->15) streaming the
// 47GB f16 expert working set per forward at low reuse (m~96). The wall is
// weight-read BW. resident-Q8 halves the footprint (~23GB) = half the
// bytes streamed = should recover much of the MoE penalty at half the RAM,
// IF argmax stays correct (Q8 amax/127 ~0.4% per-weight error).
//
// This bench evaluates resident-Q8 on Apple M5 Max hardware MMA, as a
// de-risked reference for the CUDA spark port (which sweeps resident-f16).
// Non-overlapping axis: footprint-halving (Q8) vs spark's resident-f16.
//
// Uses REAL Nemotron-Cascade-2-30B-A3B-4bit MoE experts (layer 1):
//   hidden=2688, moe_inter=1856, n_exp=128, top-6, Q4 gs=64 affine.
// The model ships Q4 — so Q4-dequant-per-fwd is BOTH the accuracy
// reference AND the per-forward-dequant baseline.
//
// Three residency forms of the per-expert MoE (fc1 -> relu^2 -> fc2):
//   1. Q4-dequant-per-fwd (baseline): keep Q4, dequantized()->f16 EACH
//      forward, then gatherMM. (what "per-forward Q4 dequant" costs)
//   2. resident-f16: dequantize ONCE at load, gatherMM f16 weights.
//   3. resident-Q8: dequantize Q4->f16 once, re-quantize to Q8 (gs=64,
//      affine) once, gatherQuantizedMM consuming Q8 weights directly.
//
// Measures per effective m in {96 (S2048,top6), 192 (S4096), 384 (S8192)}:
//   - moe_experts effective TFLOP/s for each form
//   - argmax / cosine agreement of resident-Q8 vs Q4 reference output
//   - resident RAM footprint of each form (~23GB Q8 vs 47GB f16 at full
//     model scale; reported per-layer and extrapolated x23 MoE layers)
//
// Gated. Run:
//   RUN_Q8_MOE_BENCH=1 swift test --filter residentQ8MoEBench 2>&1 \
//     | tee /tmp/q8_moe_research.out
//
// Output also appended to /tmp/q8_moe_research.out by the test.

import Foundation
import MLX
import MLXNN
import Testing

private func print(_ s: String, flush: Bool) {
    Swift.print(s)
    if flush { fflush(stdout) }
}

@Suite("Resident-Q8 MoE research")
struct ResidentQ8MoEBenchSuite {

    @Test func residentQ8MoEBench() throws {
        guard ProcessInfo.processInfo.environment["RUN_Q8_MOE_BENCH"] == "1"
        else { return }

        let outPath = "/tmp/q8_moe_research.out"
        let outURL = URL(fileURLWithPath: outPath)
        FileManager.default.createFile(atPath: outPath, contents: Data(), attributes: nil)
        let handle = try FileHandle(forWritingTo: outURL)
        func log(_ s: String) {
            print(s, flush: true)
            if let data = (s + "\n").data(using: .utf8) { handle.write(data) }
        }
        defer { try? handle.close() }

        // --- Nemotron MoE dims (confirmed from model config) ---
        let hidden = 2688
        let inter = 1856
        let nExp = 128
        let topK = 6
        let groupSize = 64

        log("=== Resident-Q8 Nemotron MoE prefill bench (Apple M5 Max, hardware MMA) ===")
        log("dims: hidden=\(hidden) inter=\(inter) n_exp=\(nExp) top-\(topK) gs=\(groupSize)")
        log("")

        // ---- Load REAL Nemotron Q4 experts (layer 1, shard 1) ----
        let modelDir = "/Users/tom/models/Nemotron-Cascade-2-30B-A3B-4bit"
        let shard1 = URL(fileURLWithPath: modelDir + "/model-00001-of-00004.safetensors")
        log("[load] reading real Q4 experts from layer 1 (shard 1) ...")
        let all = try MLX.loadArrays(url: shard1)
        let pfx = "backbone.layers.1.mixer.switch_mlp"
        func get(_ k: String) throws -> MLXArray {
            guard let a = all[k] else {
                throw NSError(domain: "bench", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "missing \(k)"])
            }
            return a
        }
        // fc1: [E, inter, hidden] (out=inter, in=hidden) packed Q4
        // fc2: [E, hidden, inter]
        let fc1wQ4 = try get("\(pfx).fc1.weight")
        let fc1sc = try get("\(pfx).fc1.scales").asType(.float16)
        let fc1bi = try get("\(pfx).fc1.biases").asType(.float16)
        let fc2wQ4 = try get("\(pfx).fc2.weight")
        let fc2sc = try get("\(pfx).fc2.scales").asType(.float16)
        let fc2bi = try get("\(pfx).fc2.biases").asType(.float16)
        eval(fc1wQ4, fc1sc, fc1bi, fc2wQ4, fc2sc, fc2bi)
        log("[load] fc1.weight(Q4 packed) shape=\(fc1wQ4.shape) dtype=\(fc1wQ4.dtype)")
        log("[load] fc1.scales shape=\(fc1sc.shape)  fc2.weight shape=\(fc2wQ4.shape)")

        // ---- Form 1 inputs: Q4 resident (dequant per forward) ----
        // Stored: packed Q4 + scales + biases.
        // ---- Form 2: f16 resident (dequant ONCE) ----
        let fc1f16 = MLX.dequantized(
            fc1wQ4, scales: fc1sc, biases: fc1bi,
            groupSize: groupSize, bits: 4, mode: .affine).asType(.float16)
        let fc2f16 = MLX.dequantized(
            fc2wQ4, scales: fc2sc, biases: fc2bi,
            groupSize: groupSize, bits: 4, mode: .affine).asType(.float16)
        eval(fc1f16, fc2f16)
        log("[form2] f16-resident fc1 shape=\(fc1f16.shape) fc2 shape=\(fc2f16.shape)")

        // ---- Form 3: Q8 resident (re-quantize f16 -> Q8 ONCE) ----
        let (fc1q8, fc1q8sc, fc1q8bi0) = MLX.quantized(
            fc1f16, groupSize: groupSize, bits: 8, mode: .affine)
        let (fc2q8, fc2q8sc, fc2q8bi0) = MLX.quantized(
            fc2f16, groupSize: groupSize, bits: 8, mode: .affine)
        let fc1q8bi = fc1q8bi0!.asType(.float16)
        let fc2q8bi = fc2q8bi0!.asType(.float16)
        let fc1q8scF = fc1q8sc.asType(.float16)
        let fc2q8scF = fc2q8sc.asType(.float16)
        eval(fc1q8, fc1q8scF, fc1q8bi, fc2q8, fc2q8scF, fc2q8bi)
        log("[form3] Q8-resident fc1 shape=\(fc1q8.shape) dtype=\(fc1q8.dtype) scales=\(fc1q8sc.shape)")
        log("")

        // ---- Resident RAM footprint (per layer, bytes of weight tensors) ----
        func nbytes(_ a: MLXArray) -> Int { a.size * a.itemSize }
        let q4Bytes = nbytes(fc1wQ4)+nbytes(fc1sc)+nbytes(fc1bi)
                    + nbytes(fc2wQ4)+nbytes(fc2sc)+nbytes(fc2bi)
        let f16Bytes = nbytes(fc1f16)+nbytes(fc2f16)
        let q8Bytes = nbytes(fc1q8)+nbytes(fc1q8scF)+nbytes(fc1q8bi)
                    + nbytes(fc2q8)+nbytes(fc2q8scF)+nbytes(fc2q8bi)
        let nMoeLayers = 23
        func gb(_ b: Int) -> Double { Double(b) / 1024.0/1024.0/1024.0 }
        log("--- resident RAM footprint (expert weights only) ---")
        log(String(format: "  per-MoE-layer:  Q4=%.3f GB  f16=%.3f GB  Q8=%.3f GB",
                   gb(q4Bytes), gb(f16Bytes), gb(q8Bytes)))
        log(String(format: "  x%d MoE layers: Q4=%.2f GB  f16=%.2f GB  Q8=%.2f GB",
                   nMoeLayers, gb(q4Bytes*nMoeLayers), gb(f16Bytes*nMoeLayers),
                   gb(q8Bytes*nMoeLayers)))
        log(String(format: "  Q8 / f16 footprint ratio = %.3f  (target ~0.5)",
                   Double(q8Bytes)/Double(f16Bytes)))
        log(String(format: "  [GPU mem after building all 3 forms] active=%.2f GB peak=%.2f GB",
                   Double(MLX.GPU.activeMemory)/1024/1024/1024,
                   Double(MLX.GPU.peakMemory)/1024/1024/1024))
        log("")

        // ---- FLOP accounting for the MoE GEMM (per forward) ----
        // rows = total routed rows across all experts (m * nExp).
        // fc1: [rows, hidden] x [inter, hidden]^T -> 2*rows*inter*hidden
        // fc2: [rows, inter] x [hidden, inter]^T -> 2*rows*hidden*inter
        // (shared-experts excluded — this isolates moe_experts.)
        func flops(_ rows: Int) -> Double {
            2.0*Double(rows)*Double(inter)*Double(hidden)   // fc1
          + 2.0*Double(rows)*Double(hidden)*Double(inter)   // fc2
        }

        // relu^2 activation (NemotronH uses relu then square)
        func relu2(_ a: MLXArray) -> MLXArray { let r = MLXNN.relu(a); return r*r }

        // ---- the three forward forms ----
        // x: [m, hidden]; indices: [m] expert ids in [0,nExp). To match the
        // production SwitchLinear contract we expand x to [m, 1, hidden]
        // (batch=m gathered rows, M=1 row each, K=hidden) and pass
        // rhsIndices=[m] flat-indexing the weight's [E,...] batch dim,
        // weightT = swap(-1,-2).
        func prep(_ x: MLXArray) -> MLXArray { MLX.expandedDimensions(x, axis: -2) }

        func fwdF16(_ x: MLXArray, _ idx: MLXArray, sorted: Bool) -> MLXArray {
            let xe = prep(x)
            let h = MLX.gatherMM(xe, fc1f16.swappedAxes(-1, -2),
                                 rhsIndices: idx, sortedIndices: sorted)
            let a = relu2(h)
            let y = MLX.gatherMM(a, fc2f16.swappedAxes(-1, -2),
                                 rhsIndices: idx, sortedIndices: sorted)
            return MLX.squeezed(y, axis: -2)
        }

        func fwdQ4deq(_ x: MLXArray, _ idx: MLXArray, sorted: Bool) -> MLXArray {
            // dequantize Q4 -> f16 EACH forward, then gatherMM
            let w1 = MLX.dequantized(fc1wQ4, scales: fc1sc, biases: fc1bi,
                        groupSize: groupSize, bits: 4, mode: .affine).asType(.float16)
            let w2 = MLX.dequantized(fc2wQ4, scales: fc2sc, biases: fc2bi,
                        groupSize: groupSize, bits: 4, mode: .affine).asType(.float16)
            let xe = prep(x)
            let h = MLX.gatherMM(xe, w1.swappedAxes(-1, -2),
                                 rhsIndices: idx, sortedIndices: sorted)
            let a = relu2(h)
            let y = MLX.gatherMM(a, w2.swappedAxes(-1, -2),
                                 rhsIndices: idx, sortedIndices: sorted)
            return MLX.squeezed(y, axis: -2)
        }

        func fwdQ8(_ x: MLXArray, _ idx: MLXArray, sorted: Bool) -> MLXArray {
            let xe = prep(x)
            let h = MLX.gatherQuantizedMM(
                xe, fc1q8, scales: fc1q8scF, biases: fc1q8bi,
                rhsIndices: idx, transpose: true,
                groupSize: groupSize, bits: 8, mode: .affine, sortedIndices: sorted)
            let a = relu2(h)
            let y = MLX.gatherQuantizedMM(
                a, fc2q8, scales: fc2q8scF, biases: fc2q8bi,
                rhsIndices: idx, transpose: true,
                groupSize: groupSize, bits: 8, mode: .affine, sortedIndices: sorted)
            return MLX.squeezed(y, axis: -2)
        }

        // Q4 forward via gatherQuantizedMM directly (the genuine resident-Q4
        // path, for an apples-to-apples quantized reference of the OUTPUT).
        func fwdQ4(_ x: MLXArray, _ idx: MLXArray, sorted: Bool) -> MLXArray {
            let xe = prep(x)
            let h = MLX.gatherQuantizedMM(
                xe, fc1wQ4, scales: fc1sc, biases: fc1bi,
                rhsIndices: idx, transpose: true,
                groupSize: groupSize, bits: 4, mode: .affine, sortedIndices: sorted)
            let a = relu2(h)
            let y = MLX.gatherQuantizedMM(
                a, fc2wQ4, scales: fc2sc, biases: fc2bi,
                rhsIndices: idx, transpose: true,
                groupSize: groupSize, bits: 4, mode: .affine, sortedIndices: sorted)
            return MLX.squeezed(y, axis: -2)
        }

        // ---- timing helper ----
        func benchMs(_ name: String, iters: Int, warmup: Int = 5,
                     _ op: () -> MLXArray) -> Double {
            for _ in 0..<warmup { eval(op()) }
            var s: [Double] = []
            for _ in 0..<iters {
                let t0 = Date(); eval(op())
                s.append(Date().timeIntervalSince(t0)*1000.0)
            }
            s.sort()
            let med = s[s.count/2]
            log("    \(name.padding(toLength: 26, withPad: " ", startingAt: 0)) median=\(String(format: "%7.3f", med)) ms")
            return med
        }

        func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
            let af = a.asType(.float32).flattened()
            let bf = b.asType(.float32).flattened()
            let dot = (af*bf).sum()
            let na = (af*af).sum().sqrt()
            let nb = (bf*bf).sum().sqrt()
            return (dot/(na*nb)).item(Float.self)
        }

        // ---- DEBUG: single-expert dense ground-truth sanity check ----
        // For expert e: ref = relu^2(x @ W1[e]^T) @ W2[e]^T using plain matmul
        // on the f16 dequantized weights. Confirms which gather path is the
        // faithful one and that low cosine is a relu^2-heavy-tail artifact,
        // not a wrong gather.
        do {
            MLXRandom.seed(1)
            let e = 5
            let xd = MLXRandom.uniform(low: -0.5, high: 0.5, [4, hidden]).asType(.float16)
            let idxd = MLXArray([Int32(e), Int32(e), Int32(e), Int32(e)])
            eval(xd, idxd)
            // dense reference for expert e
            let w1e = fc1f16[e]          // [inter, hidden]
            let w2e = fc2f16[e]          // [hidden, inter]
            let hRef = relu2(MLX.matmul(xd, w1e.swappedAxes(-1, -2)))  // [4, inter]
            let yRef = MLX.matmul(hRef, w2e.swappedAxes(-1, -2))        // [4, hidden]
            eval(yRef)
            let yF16 = fwdF16(xd, idxd, sorted: false)
            let yQ8 = fwdQ8(xd, idxd, sorted: false)
            let yQ4 = fwdQ4(xd, idxd, sorted: false)
            eval(yF16, yQ8, yQ4)
            log("--- DEBUG single-expert dense ground-truth (expert \(e), m=4) ---")
            log(String(format: "   cosine vs dense-ref: f16=%.6f  Q8=%.6f  Q4=%.6f",
                       cosine(yF16, yRef), cosine(yQ8, yRef), cosine(yQ4, yRef)))
            log(String(format: "   max|f16-ref|=%.4f  mean|ref|=%.4f",
                       MLX.abs(yF16.asType(.float32)-yRef.asType(.float32)).max().item(Float.self),
                       MLX.abs(yRef.asType(.float32)).mean().item(Float.self)))
            log("")
        }

        // routing: build [m] random expert ids, SORTED (prefill fast path).
        MLXRandom.seed(7)
        let iters = Int(ProcessInfo.processInfo.environment["BENCH_ITERS"] ?? "30") ?? 30

        let ms = [96, 192, 384, 768]
        log("--- per-m sweep ---")
        log("    m = ROWS PER EXPERT (the per-expert GEMM M dim, = reuse depth)")
        log("    m=96 ~ S2048 (2048*6/128), m=192 ~ S4096, m=384 ~ S8192.")
        log("    All \(nExp) experts active; total routed rows = m * \(nExp).")
        log("    This is the genuine prefill per-expert GEMM shape where MMA")
        log("    throughput and weight-read BW actually trade off.")
        log("")

        // raw lines for downstream
        var rawLines: [String] = []

        for m in ms {
            // total routed rows: m rows per expert, all nExp experts active.
            let totalRows = m * nExp
            // x routed input [totalRows, hidden]
            let x = MLXRandom.uniform(low: -0.5, high: 0.5, [totalRows, hidden]).asType(.float16)
            // block-contiguous expert ids: [0]*m, [1]*m, ..., [nExp-1]*m
            // (already sorted -> contiguous-per-expert fast path).
            var ids = [Int32]()
            ids.reserveCapacity(totalRows)
            for e in 0..<nExp { for _ in 0..<m { ids.append(Int32(e)) } }
            let idx = MLXArray(ids)
            eval(x, idx)

            log(">> m=\(m) rows/expert  (total routed rows=\(totalRows))")

            // correctness: compare outputs. Use sorted:false so every form
            // takes the identical (non-fast-path) gather — the sortedIndices
            // fast path is exploited differently by the f16 vs quantized
            // kernels and would otherwise inject a spurious mismatch that is
            // a kernel artifact, not a quantization-accuracy effect.
            // Reference = resident-f16 (the true dequantized math).
            let outQ4 = fwdQ4(x, idx, sorted: false)
            let outF16 = fwdF16(x, idx, sorted: false)
            let outQ8 = fwdQ8(x, idx, sorted: false)
            eval(outQ4, outF16, outQ8)
            let cosQ8_Q4 = cosine(outQ8, outQ4)
            let cosQ8_F16 = cosine(outQ8, outF16)
            let cosQ4_F16 = cosine(outQ4, outF16)

            // argmax agreement: per-row argmax over hidden output
            let amQ4 = outQ4.argMax(axis: -1)
            let amQ8 = outQ8.argMax(axis: -1)
            let amF16 = outF16.argMax(axis: -1)
            let agreeQ8Q4 = (amQ8 .== amQ4).asType(.float32).mean().item(Float.self)
            let agreeQ8F16 = (amQ8 .== amF16).asType(.float32).mean().item(Float.self)

            log(String(format: "   cosine: Q8vsQ4=%.6f  Q8vsF16=%.6f  Q4vsF16=%.6f",
                       cosQ8_Q4, cosQ8_F16, cosQ4_F16))
            log(String(format: "   per-row argmax agree: Q8==Q4 %.4f  Q8==F16 %.4f",
                       agreeQ8Q4, agreeQ8F16))

            // perf
            let msQ4deq = benchMs("Q4-dequant-per-fwd", iters: iters) {
                fwdQ4deq(x, idx, sorted: true)
            }
            let msF16 = benchMs("resident-f16", iters: iters) {
                fwdF16(x, idx, sorted: true)
            }
            let msQ8 = benchMs("resident-Q8", iters: iters) {
                fwdQ8(x, idx, sorted: true)
            }
            let msQ4 = benchMs("resident-Q4 (gatherQMM)", iters: iters) {
                fwdQ4(x, idx, sorted: true)
            }

            let fl = flops(totalRows)
            func tflops(_ msv: Double) -> Double { fl / (msv/1000.0) / 1e12 }
            log(String(format: "   TFLOP/s: Q4deq=%.2f  f16=%.2f  Q8=%.2f  Q4=%.2f",
                       tflops(msQ4deq), tflops(msF16), tflops(msQ8), tflops(msQ4)))
            log(String(format: "   Q8 vs f16 speedup = %.2fx   (>1 = Q8 recovers BW)",
                       msF16/msQ8))
            log("")

            rawLines.append(String(format:
                "m=%d q4deq_ms=%.4f f16_ms=%.4f q8_ms=%.4f q4_ms=%.4f tf_q4deq=%.2f tf_f16=%.2f tf_q8=%.2f tf_q4=%.2f cos_q8q4=%.6f cos_q8f16=%.6f argmax_q8q4=%.4f",
                m, msQ4deq, msF16, msQ8, msQ4,
                tflops(msQ4deq), tflops(msF16), tflops(msQ8), tflops(msQ4),
                cosQ8_Q4, cosQ8_F16, agreeQ8Q4))
        }

        log("=== RAW ===")
        for l in rawLines { log(l) }
        log("=== END ===")
    }
}
