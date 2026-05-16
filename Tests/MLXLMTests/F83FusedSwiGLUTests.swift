// F-83 W2 correctness test: fused dual-QGEMV + SwiGLU vs split + silu*mul.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

final class F83FusedSwiGLUTests: XCTestCase {

    func testFusedMatchesSplitAtQwen2Shape() throws {
        // Qwen2.5-14B shape: hidden=5120, intermediate=27648, group=64, 4-bit.
        let K = 1024  // shrink for test speed; still divisible by 64 and TG=64
        let N = 2048
        let group = 64
        let bits = 4

        // Build a Linear with seeded init, then quantize.
        MLXRandom.seed(42)
        let weight = MLXRandom.normal([2 * N, K], dtype: .float16) * MLXArray(Float(0.02))
        let linear = Linear(weight: weight, bias: nil)
        let q = QuantizedLinear(linear, groupSize: group, bits: bits)

        // Input.
        let x = MLXRandom.normal([1, 1, K], dtype: .float16) * MLXArray(Float(0.5))

        // Baseline: gateUp(x), split, silu(gate) * up.
        let gateUp = q(x)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let baseline = silu(parts[0]) * parts[1]

        // Fused.
        guard let biases = q.biases else {
            XCTFail("QuantizedLinear biases must be non-nil for affine mode")
            return
        }
        let fused = F83FusedSwiGLU.callAsFunction(
            x: x,
            gateUpWeight: q.weight,
            gateUpScales: q.scales,
            gateUpBiases: biases,
            intermediate: N,
            hiddenIn: K,
            groupSize: group
        )

        XCTAssertEqual(baseline.shape, fused.shape, "shape mismatch baseline=\(baseline.shape) fused=\(fused.shape)")

        // Cosine similarity over the output vectors.
        let a = baseline.asArray(Float.self)
        let b = fused.asArray(Float.self)
        XCTAssertEqual(a.count, b.count)
        var dot: Double = 0
        var na: Double = 0
        var nb: Double = 0
        for i in 0..<a.count {
            let xv = Double(a[i])
            let yv = Double(b[i])
            dot += xv * yv
            na += xv * xv
            nb += yv * yv
        }
        let cos = dot / (sqrt(na) * sqrt(nb))
        print("[F83FusedSwiGLU] cosine vs baseline: \(cos)")
        XCTAssertGreaterThan(cos, 0.995, "cosine \(cos) below 0.995 threshold")
    }

    /// Microbench at Qwen2.5-14B exact shape. Only runs if F83_BENCH=1.
    func testBenchAtQwen2Shape() throws {
        guard ProcessInfo.processInfo.environment["F83_BENCH"] == "1" else {
            throw XCTSkip("set F83_BENCH=1 to run perf microbench")
        }
        let K = 5120
        let N = 27648
        let group = 64
        let bits = 4

        MLXRandom.seed(11)
        // Keep all tensors in fp16 (matches real Qwen2.5-14B-4bit checkpoint).
        let weight = (MLXRandom.normal([2 * N, K], dtype: .float16) * MLXArray(Float(0.02))).asType(.float16)
        let linear = Linear(weight: weight, bias: nil)
        let q = QuantizedLinear(linear, groupSize: group, bits: bits)

        let x = MLXRandom.normal([1, 1, K], dtype: .float16)

        // Warmup both paths.
        for _ in 0..<5 {
            let gu = q(x)
            let parts = MLX.split(gu, parts: 2, axis: -1)
            let baseline = silu(parts[0]) * parts[1]
            baseline.eval()
            let fused = F83FusedSwiGLU.callAsFunction(
                x: x,
                gateUpWeight: q.weight,
                gateUpScales: q.scales,
                gateUpBiases: q.biases!,
                intermediate: N,
                hiddenIn: K,
                groupSize: group
            )
            fused.eval()
        }

        // Time baseline.
        let iters = 50
        var tBase: Double = 0
        for _ in 0..<iters {
            let t0 = Date()
            let gu = q(x)
            let parts = MLX.split(gu, parts: 2, axis: -1)
            let baseline = silu(parts[0]) * parts[1]
            baseline.eval()
            tBase += Date().timeIntervalSince(t0)
        }

        // Time fused.
        var tFused: Double = 0
        for _ in 0..<iters {
            let t0 = Date()
            let fused = F83FusedSwiGLU.callAsFunction(
                x: x,
                gateUpWeight: q.weight,
                gateUpScales: q.scales,
                gateUpBiases: q.biases!,
                intermediate: N,
                hiddenIn: K,
                groupSize: group
            )
            fused.eval()
            tFused += Date().timeIntervalSince(t0)
        }

        let msBase = tBase / Double(iters) * 1000.0
        let msFused = tFused / Double(iters) * 1000.0
        let speedup = (msBase - msFused) / msBase * 100.0
        print(String(format: "[F83 W2 microbench] baseline=%.3fms  fused=%.3fms  delta=%+.1f%%", msBase, msFused, speedup))
    }
}
