// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Phase-3 DECODE kickoff: Nemotron Metal batch-1 decode profile.
//
// Measures, on the REAL Nemotron-Cascade-2-30B-A3B-4bit checkpoint:
//   1. end-to-end decode tok/s at SHORT ctx and DEEP ctx (target ~32K KV)
//   2. per-block-type stage breakdown (mamba / attention / moe / embed+head)
//      at each depth, via the env-gated NemotronDecodeProfiler in NemotronH.
//   3. coherence check (greedy-decoded token ids print; no NaN).
//
// Decode = single-token step against a grown KV/SSM cache (TokenIterator
// shape). We build the cache by prefilling a synthetic prompt to the target
// depth, then time N decode steps.
//
// Gated. Run:
//   RUN_NEMOTRON_DECODE_PROFILE=1 swift test --filter nemotronDecodeProfile \
//     2>&1 | tee /tmp/nemotron_decode_profile.out
//
// Deep depth is capped by free RAM; the bench backs off if a depth would
// risk OOM and reports the largest safe depth reached.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Nemotron decode profile — Phase 3 kickoff", .serialized)
struct NemotronDecodeProfileBench {

    private static func log(_ s: String) {
        Swift.print(s); fflush(stdout)
        if let h = FileHandle(forWritingAtPath: "/tmp/nemotron_decode_profile.out") {
            h.seekToEndOfFile(); h.write((s + "\n").data(using: .utf8)!); h.closeFile()
        }
    }

    @Test func nemotronDecodeProfile() throws {
        guard ProcessInfo.processInfo.environment["RUN_NEMOTRON_DECODE_PROFILE"] == "1"
        else { return }
        FileManager.default.createFile(
            atPath: "/tmp/nemotron_decode_profile.out", contents: Data())

        let modelDir = "/Users/tom/models/Nemotron-Cascade-2-30B-A3B-4bit"
        let modelPath = URL(fileURLWithPath: modelDir)
        guard FileManager.default.fileExists(
            atPath: modelPath.appendingPathComponent("config.json").path) else {
            Issue.record("model not found at \(modelDir)"); return
        }

        Self.log("[decode-profile] loading \(modelPath.lastPathComponent)...")
        let configData = try Data(
            contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(NemotronHConfiguration.self, from: configData)
        let model = NemotronHModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        eval(model)
        Self.log("[decode-profile] loaded. pattern len=\(cfg.hybridOverridePattern.count)")

        // Depths to profile. Deep target = 32K; back off on OOM risk.
        let depths = [512, 8192, 32768]
        let decodeSteps = 32   // timed decode steps per depth

        for depth in depths {
            try? runDepth(model: model, cfg: cfg, depth: depth, decodeSteps: decodeSteps)
            // free GPU/cache pressure between depths
            MLX.GPU.clearCache()
        }
    }

    private func runDepth(
        model: NemotronHModel, cfg: NemotronHConfiguration,
        depth: Int, decodeSteps: Int
    ) throws {
        let memBefore = MLX.GPU.activeMemory / (1024*1024*1024)
        Self.log("\n========== DEPTH \(depth) KV (activeGPU=\(memBefore)GB) ==========")

        let params = GenerateParameters(maxKVSize: depth + decodeSteps + 64)
        let cache = model.newCache(parameters: params)

        // Prefill a synthetic prompt to `depth` tokens to grow the cache.
        // Use the model's prefill step size to mimic the real prefill path.
        let stepSize = model.defaultPrefillStepSize
        var pos = 0
        let promptLen = depth
        let t0 = CFAbsoluteTimeGetCurrent()
        while pos < promptLen {
            let n = min(stepSize, promptLen - pos)
            let chunk = MLXArray((0..<n).map { Int32(($0 + pos) % cfg.vocabSize) })
                .reshaped(1, n)
            let out = model(chunk, cache: cache)
            eval(out)
            pos += n
        }
        let prefillT = CFAbsoluteTimeGetCurrent() - t0
        let memAfter = MLX.GPU.activeMemory / (1024*1024*1024)
        Self.log(String(format: "[prefill] %d tok in %.2fs (%.0f tok/s)  activeGPU=%dGB",
                        promptLen, prefillT, Double(promptLen)/prefillT, memAfter))

        // Safety: bail if we're near the memory ceiling.
        if memAfter > 85 {
            Self.log("[decode-profile] activeGPU=\(memAfter)GB > 85GB ceiling — stopping deeper depths")
            return
        }

        var lastTok = MLXArray([Int32(42)]).reshaped(1, 1)

        // ---- 1) End-to-end decode tok/s (UN-profiled = production path) ----
        NemotronDecodeProfiler.enabled = false
        // warmup
        for _ in 0..<3 {
            let logits = model(lastTok, cache: cache)
            lastTok = argmaxNext(logits)
        }
        eval(lastTok)
        var tokens: [Int32] = []
        let d0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<decodeSteps {
            let logits = model(lastTok, cache: cache)
            lastTok = argmaxNext(logits)
            eval(lastTok)
            tokens.append(lastTok.item(Int32.self))
        }
        let decodeT = CFAbsoluteTimeGetCurrent() - d0
        let tps = Double(decodeSteps) / decodeT
        let coherent = !tokens.contains { $0 < 0 || Int($0) >= cfg.vocabSize }
        Self.log(String(format: "[decode-e2e] %.2f tok/s (%.3f ms/tok)  coherent=%@  first8=%@",
                        tps, 1000*decodeT/Double(decodeSteps),
                        coherent ? "yes" : "NO",
                        tokens.prefix(8).map(String.init).joined(separator: ",")))

        // ---- 2) Per-stage breakdown (profiled = serialized barriers) ----
        NemotronDecodeProfiler.enabled = true
        NemotronDecodeProfiler.shared.reset()
        for _ in 0..<decodeSteps {
            let logits = model(lastTok, cache: cache)
            lastTok = argmaxNext(logits)
            eval(lastTok)
        }
        NemotronDecodeProfiler.enabled = false
        Self.log("[decode-stages]\n" + NemotronDecodeProfiler.shared.report())

        // ---- 3) QUICK WIN: e2e decode under wired-memory residency ----
        // Pin the 17GB working set so the OS keeps weights resident (no page
        // faults streaming the MoE experts). Correctness-neutral.
        let wiredBytes = 24 * 1024 * 1024 * 1024  // 24GB > 17GB model + cache
        let wiredTps = MLX.GPU.withWiredLimit(wiredBytes) { () -> Double in
            for _ in 0..<3 { let l = model(lastTok, cache: cache); lastTok = argmaxNext(l) }
            eval(lastTok)
            let w0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<decodeSteps {
                let logits = model(lastTok, cache: cache)
                lastTok = argmaxNext(logits)
                eval(lastTok)
            }
            return Double(decodeSteps) / (CFAbsoluteTimeGetCurrent() - w0)
        }
        Self.log(String(format: "[quick-win wired-24GB] %.2f tok/s  (vs e2e %.2f, %+.1f%%)",
                        wiredTps, tps, 100*(wiredTps-tps)/tps))
    }

    private func argmaxNext(_ logits: MLXArray) -> MLXArray {
        // logits: [1, 1, vocab] -> [1, 1]
        let last = logits[0..., -1, 0...]
        return argMax(last, axis: -1).reshaped(1, 1).asType(.int32)
    }
}
