// Research bench: characterize mx.take(_:_:axis:) at sparse-decode shapes.
//
// Goal: decide if take(axis=2) is bandwidth-bound on the gathered subset
// (~4MB), or if it materializes a big intermediate / transposes the full
// tensor / has overhead that kills the bandwidth savings.
//
// Shapes match planned sparse-decode path:
//   keys/values: [B=1, kvHeads=8, T=131072, headDim=128] fp16
//   positions:  [k=2048] Int32
//   gathered:   [B=1, kvHeads=8, k=2048, headDim=128] fp16  (~4 MB)
//   queries:    [B=1, qHeads=32, T_q=1, headDim=128] fp16
//
// Compares:
//   1. take(K, positions, axis: 2)
//   2. Dense SDPA on full [1,8,131072,128] K/V — what we'd pay without sparse
//   3. SDPA on gathered [1,8,2048,128] K/V — post-gather attention cost
//   Verdict: does (take + small SDPA) beat dense SDPA?
//
// Patterns: contiguous (cache-friendly), strided-64 (block sample),
// shuffled (cache-hostile). Apple unified-memory cache hit rate matters.
//
// Gated. Run:
//   RUN_MX_TAKE_BENCH=1 swift test --filter mxTakeGatherBench 2>&1 \
//     | tee /tmp/mx_take_research.out
//
// Output appended to /tmp/mx_take_research.out by the test itself.

import Foundation
import MLX
import Testing

private func print(_ s: String, flush: Bool) {
    Swift.print(s)
    if flush { fflush(stdout) }
}

@Suite("MX take gather research")
struct MxTakeGatherBenchSuite {

    @Test func mxTakeGatherBench() throws {
        guard ProcessInfo.processInfo.environment["RUN_MX_TAKE_BENCH"] == "1"
        else { return }

        let outPath = "/tmp/mx_take_research.out"
        let outURL = URL(fileURLWithPath: outPath)
        // Create empty file (truncate)
        FileManager.default.createFile(atPath: outPath, contents: Data(), attributes: nil)
        let handle = try FileHandle(forWritingTo: outURL)

        func log(_ s: String) {
            print(s, flush: true)
            if let data = (s + "\n").data(using: .utf8) {
                handle.write(data)
            }
        }
        defer { try? handle.close() }

        // Shapes
        let B = 1
        let kvHeads = 8
        let qHeads = 32
        let T = 131072
        let headDim = 128
        let k = 2048
        let dtype: DType = .float16

        log("=== MX take(axis=2) sparse-decode research bench ===")
        log("device: GPU (MLX default)")
        log("shapes:")
        log("  K/V: [\(B), \(kvHeads), \(T), \(headDim)] \(dtype) (\(B*kvHeads*T*headDim*2/1024/1024) MB each)")
        log("  Q:   [\(B), \(qHeads), 1, \(headDim)] \(dtype)")
        log("  positions: [\(k)] Int32")
        log("  gathered K/V slice: ~\(B*kvHeads*k*headDim*2/1024/1024) MB each")
        log("")

        // Allocate synthetic tensors. Use uniform random (avoid normal — at
        // [1,8,131072,128] fp16 MLXRandom.normal pushes peak GPU mem to ~6.6GB
        // through intermediates; uniform stays bounded).
        // Random (non-zero) is important: SDPA over all-zeros may hit fast paths.
        log("[alloc] allocating K, V, Q (uniform random) ...")
        MLXRandom.seed(42)
        // Generate at fp32 then cast — avoids large fp32 intermediate at fp16
        // shape by using uniform (cheaper than normal's box-muller).
        let K = MLXRandom.uniform(low: -0.1, high: 0.1,
                                  [B, kvHeads, T, headDim]).asType(dtype)
        let V = MLXRandom.uniform(low: -0.1, high: 0.1,
                                  [B, kvHeads, T, headDim]).asType(dtype)
        let Q = MLXRandom.uniform(low: -0.1, high: 0.1,
                                  [B, qHeads, 1, headDim]).asType(dtype)
        // Force eval so allocation isn't part of the bench
        eval(K, V, Q)
        MLX.GPU.clearCache()
        log(String(format: "[alloc] done. peak GPU mem = %.1f MB, active = %.1f MB",
                   Double(MLX.GPU.peakMemory) / 1024.0 / 1024.0,
                   Double(MLX.GPU.activeMemory) / 1024.0 / 1024.0))
        MLX.GPU.resetPeakMemory()
        log("")

        // Helper: time `op` over `iters` iterations after `warmup` warm-up
        // iterations. Returns ms median.
        func benchMs(name: String, iters: Int, warmup: Int = 3,
                     op: () -> [MLXArray]) -> Double {
            log("[bench] starting \(name) warmup x\(warmup) ...")
            // Warmup
            for w in 0..<warmup {
                let outs = op()
                eval(outs)
                log("[bench]   warmup \(w) done")
            }
            log("[bench] timed loop x\(iters) ...")
            var samples: [Double] = []
            samples.reserveCapacity(iters)
            for i in 0..<iters {
                let t0 = Date()
                let outs = op()
                eval(outs)
                let dt = Date().timeIntervalSince(t0) * 1000.0
                samples.append(dt)
                if i < 3 || i == iters - 1 {
                    log("[bench]   iter \(i): \(String(format: "%.3f", dt)) ms")
                }
            }
            samples.sort()
            let median = samples[samples.count / 2]
            let p10 = samples[max(0, samples.count / 10)]
            let p90 = samples[min(samples.count - 1, (samples.count * 9) / 10)]
            let paddedName = name.padding(toLength: 50, withPad: " ", startingAt: 0)
            log(String(
                format: "  \(paddedName)median=%7.3f ms  p10=%7.3f  p90=%7.3f  (n=%d)",
                median, p10, p90, iters))
            return median
        }

        // Build position patterns
        func contiguousPositions(_ count: Int) -> MLXArray {
            // 0, 1, 2, ..., count-1
            MLXArray(Array(0..<Int32(count)))
        }
        func stridedPositions(_ count: Int, stride: Int, maxIdx: Int) -> MLXArray {
            // 0, stride, 2*stride, ...
            var arr = [Int32]()
            arr.reserveCapacity(count)
            var v = 0
            for _ in 0..<count {
                arr.append(Int32(v % maxIdx))
                v += stride
            }
            return MLXArray(arr)
        }
        func shuffledPositions(_ count: Int, maxIdx: Int, seed: UInt64) -> MLXArray {
            var rng = SystemRandomNumberGenerator()
            _ = rng
            // Deterministic shuffle: take 0..maxIdx, shuffle with srand48
            var src = Array(0..<Int32(maxIdx))
            srand48(Int(seed))
            for i in stride(from: src.count - 1, to: 0, by: -1) {
                let j = Int(drand48() * Double(i + 1))
                src.swapAt(i, j)
            }
            return MLXArray(Array(src.prefix(count)))
        }

        log("[debug] building positions...")
        let posContig = contiguousPositions(k)
        eval(posContig)
        log("[debug] posContig shape=\(posContig.shape) dtype=\(posContig.dtype)")
        let posStride64 = stridedPositions(k, stride: 64, maxIdx: T)
        eval(posStride64)
        let posShuf = shuffledPositions(k, maxIdx: T, seed: 123)
        eval(posShuf)
        log("[debug] positions built")

        // Smoke test: does a single take work?
        log("[debug] smoke test: K.take(posContig, axis: 2) ...")
        let smoke = K.take(posContig, axis: 2)
        eval(smoke)
        log("[debug] smoke result shape=\(smoke.shape)")

        let scale = 1.0 / Float(headDim).squareRoot()

        let iters = Int(ProcessInfo.processInfo.environment["BENCH_ITERS"] ?? "50") ?? 50

        log("--- take(K, positions, axis: 2) — varying access pattern ---")
        log("[debug] entering benchMs for contiguous ...")
        let takeContigMs = benchMs(
            name: "take K contiguous (0..k-1)", iters: iters) {
                [K.take(posContig, axis: 2)]
            }
        log("[debug] benchMs contiguous done")
        let takeStrideMs = benchMs(
            name: "take K strided-64 (every 64th over full T)", iters: iters) {
                [K.take(posStride64, axis: 2)]
            }
        let takeShufMs = benchMs(
            name: "take K shuffled (random k of T)", iters: iters) {
                [K.take(posShuf, axis: 2)]
            }
        log("")

        log("--- take K + take V (paired, what sparse decode actually does) ---")
        let takeKVContigMs = benchMs(
            name: "take K+V contiguous", iters: iters) {
                [K.take(posContig, axis: 2), V.take(posContig, axis: 2)]
            }
        let takeKVStrideMs = benchMs(
            name: "take K+V strided-64", iters: iters) {
                [K.take(posStride64, axis: 2), V.take(posStride64, axis: 2)]
            }
        let takeKVShufMs = benchMs(
            name: "take K+V shuffled", iters: iters) {
                [K.take(posShuf, axis: 2), V.take(posShuf, axis: 2)]
            }
        log("")

        log("--- SDPA cost baselines ---")
        // Dense baseline: SDPA over full [1,8,131072,128] K/V
        let denseSdpaMs = benchMs(
            name: "SDPA dense [B,32,1,128] x [B,8,131072,128]", iters: iters) {
                [MLXFast.scaledDotProductAttention(
                    queries: Q, keys: K, values: V,
                    scale: scale, mask: nil)]
            }

        // Pre-gather gathered K/V (so we measure attn cost only)
        let gK = K.take(posStride64, axis: 2)
        let gV = V.take(posStride64, axis: 2)
        eval(gK, gV)

        let smallSdpaMs = benchMs(
            name: "SDPA gathered [B,32,1,128] x [B,8,2048,128]", iters: iters) {
                [MLXFast.scaledDotProductAttention(
                    queries: Q, keys: gK, values: gV,
                    scale: scale, mask: nil)]
            }
        log("")

        log("--- Combined: take K+V + SDPA on gathered (what we'd ship) ---")
        let comboStrideMs = benchMs(
            name: "take K+V strided-64 + small SDPA", iters: iters) {
                let gk = K.take(posStride64, axis: 2)
                let gv = V.take(posStride64, axis: 2)
                let out = MLXFast.scaledDotProductAttention(
                    queries: Q, keys: gk, values: gv,
                    scale: scale, mask: nil)
                return [out]
            }
        let comboShufMs = benchMs(
            name: "take K+V shuffled + small SDPA", iters: iters) {
                let gk = K.take(posShuf, axis: 2)
                let gv = V.take(posShuf, axis: 2)
                let out = MLXFast.scaledDotProductAttention(
                    queries: Q, keys: gk, values: gv,
                    scale: scale, mask: nil)
                return [out]
            }
        log("")

        // -- Verdict block --
        log("=== VERDICT ===")
        let bestCombo = min(comboStrideMs, comboShufMs)
        let speedup = denseSdpaMs / bestCombo
        log(String(format: "dense SDPA:                  %7.3f ms", denseSdpaMs))
        log(String(format: "best combo (take+small SDPA): %7.3f ms", bestCombo))
        log(String(format: "  -> speedup vs dense:        %5.2fx",
                   speedup))
        log("")
        // Bandwidth: K is 1*8*131072*128*2 = 256MB. M5 Max ~400-500 GB/s.
        // theoretical full-read of K = 256/400 ≈ 0.64 ms.
        // Gather of 4MB = 0.01 ms theoretical.
        log("notes:")
        log("  full K read at 400GB/s = ~0.64 ms theoretical (one tensor)")
        log("  4MB gather at 400GB/s  = ~0.01 ms theoretical (one tensor)")
        log("  -> if take_K dominates dense_sdpa_K_read, gather is overhead-bound")
        log("     not bandwidth-bound, and a custom kernel would help.")
        log("  -> if take_K << dense_sdpa, axis=2 take is fine to ship.")
        log("")

        // Pattern sensitivity
        let patternRatio = takeShufMs / takeContigMs
        log(String(format: "pattern sensitivity (shuffled / contig): %.2fx", patternRatio))
        if patternRatio > 1.5 {
            log("  ! significant pattern sensitivity — cache-hostile shuffled is slow.")
            log("    sparse decode using arbitrary block positions may pay this cost.")
        } else {
            log("  pattern-insensitive — gather is memory-bound regardless of pattern.")
        }
        log("")
        log("--- raw numbers for downstream consumption ---")
        log(String(format: "take_contig_ms=%.4f", takeContigMs))
        log(String(format: "take_stride_ms=%.4f", takeStrideMs))
        log(String(format: "take_shuf_ms=%.4f", takeShufMs))
        log(String(format: "take_kv_contig_ms=%.4f", takeKVContigMs))
        log(String(format: "take_kv_stride_ms=%.4f", takeKVStrideMs))
        log(String(format: "take_kv_shuf_ms=%.4f", takeKVShufMs))
        log(String(format: "dense_sdpa_ms=%.4f", denseSdpaMs))
        log(String(format: "small_sdpa_ms=%.4f", smallSdpaMs))
        log(String(format: "combo_stride_ms=%.4f", comboStrideMs))
        log(String(format: "combo_shuf_ms=%.4f", comboShufMs))
        log("=== END ===")
    }
}
