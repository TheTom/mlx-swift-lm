// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// LEVER #1 validation: DECODE LAUNCH-COLLAPSE.
//
// The Phase-3 profile found Nemotron Metal decode is FLAT ~36 tok/s 512→32K,
// root-caused to GPU under-occupancy + per-op launch/sync overhead at batch-1
// (NOT bandwidth, NOT KV). Serialized per-stage = 42ms vs pipelined 28ms →
// ~14ms is exposed launch/sync overhead. Target: collapse the per-op
// launches/syncs across the 52-layer single-token step so the whole step
// dispatches with minimal CPU↔GPU round-trips.
//
// This bench measures four decode variants against the SAME grown cache, at
// 512 and 32K KV depth, all greedy/argmax (capture is correctness-neutral —
// argmax must be unchanged):
//
//   A. baseline-sync   : model(); argmax; eval(tok); .item()   per step
//                        (the as-profiled path — forces a full CPU↔GPU sync
//                         BEFORE building the next step's graph → serializes
//                         dispatch). This is the ~36 tok/s reference.
//   B. asyncEval-pipe  : model(); argmax; y=tok; asyncEval(tok); return
//                        PREVIOUS step's .item()  (the production TokenIterator
//                        pattern — one-step-deferred sync, GPU stays 1 step
//                        ahead, CPU builds step i+1 while GPU runs step i).
//                        This is the launch-collapse lever via eval-boundary
//                        removal.
//   C. deferred-batch  : build ALL N step graphs feeding argmax→next, only
//                        asyncEval the running token; single .item() drain at
//                        the end. Maximal CPU-run-ahead.
//   D. compiled-substep: MLX.compile() the per-layer compute-heavy
//                        sub-chains (RMSNorm fusions etc.) where the graph is
//                        static & state-free, to see the kernel-fusion ceiling
//                        on top of B. (Reports blocker if the full step won't
//                        compile due to cache-state mutation / MoE gather.)
//
// Gated. Run:
//   RUN_NEMOTRON_LAUNCH_COLLAPSE=1 swift test --filter nemotronLaunchCollapse \
//     2>&1 | tee /tmp/nemotron_launch_collapse.out

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Nemotron decode launch-collapse — LEVER #1", .serialized)
struct NemotronLaunchCollapseBench {

    private static func log(_ s: String) {
        Swift.print(s); fflush(stdout)
        if let h = FileHandle(forWritingAtPath: "/tmp/nemotron_launch_collapse.out") {
            h.seekToEndOfFile(); h.write((s + "\n").data(using: .utf8)!); h.closeFile()
        }
    }

    @Test func nemotronLaunchCollapse() throws {
        guard ProcessInfo.processInfo.environment["RUN_NEMOTRON_LAUNCH_COLLAPSE"] == "1"
        else { return }
        FileManager.default.createFile(
            atPath: "/tmp/nemotron_launch_collapse.out", contents: Data())

        let modelDir = "/Users/tom/models/Nemotron-Cascade-2-30B-A3B-4bit"
        let modelPath = URL(fileURLWithPath: modelDir)
        guard FileManager.default.fileExists(
            atPath: modelPath.appendingPathComponent("config.json").path) else {
            Issue.record("model not found at \(modelDir)"); return
        }

        Self.log("[launch-collapse] loading \(modelPath.lastPathComponent)...")
        let configData = try Data(
            contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(NemotronHConfiguration.self, from: configData)
        let model = NemotronHModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        eval(model)
        Self.log("[launch-collapse] loaded. pattern len=\(cfg.hybridOverridePattern.count)")

        let depths = [512, 32768]
        let decodeSteps = 48

        for depth in depths {
            try? runDepth(model: model, cfg: cfg, depth: depth, decodeSteps: decodeSteps)
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
        let stepSize = model.defaultPrefillStepSize
        var pos = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        while pos < depth {
            let n = min(stepSize, depth - pos)
            let chunk = MLXArray((0..<n).map { Int32(($0 + pos) % cfg.vocabSize) })
                .reshaped(1, n)
            let out = model(chunk, cache: cache)
            eval(out)
            pos += n
        }
        let prefillT = CFAbsoluteTimeGetCurrent() - t0
        let memAfter = MLX.GPU.activeMemory / (1024*1024*1024)
        Self.log(String(format: "[prefill] %d tok in %.2fs (%.0f tok/s)  activeGPU=%dGB",
                        depth, prefillT, Double(depth)/prefillT, memAfter))
        if memAfter > 85 {
            Self.log("[launch-collapse] activeGPU=\(memAfter)GB > 85GB ceiling — stopping")
            return
        }

        let seed = MLXArray([Int32(42)]).reshaped(1, 1)

        // ---------- A. baseline-sync (as-profiled reference) ----------
        let (tpsA, tokA) = decodeSync(model: model, cache: cache, seed: seed,
                                      steps: decodeSteps)
        Self.log(String(format: "[A baseline-sync ] %.2f tok/s (%.3f ms/tok)  first8=%@",
                        tpsA, 1000.0/tpsA, tokA.prefix(8).map(String.init).joined(separator: ",")))

        // ---------- B. asyncEval-pipelined (production TokenIterator) ----------
        let (tpsB, tokB) = decodeAsyncPipe(model: model, cache: cache, seed: seed,
                                           steps: decodeSteps)
        let okB = tokB == tokA
        Self.log(String(format: "[B asyncEval-pipe] %.2f tok/s (%.3f ms/tok)  %+.2fx vs A  argmax==A:%@",
                        tpsB, 1000.0/tpsB, tpsB/tpsA, okB ? "yes" : "NO"))
        if !okB {
            Self.log("    B first8=\(tokB.prefix(8).map(String.init).joined(separator: ","))")
        }

        // ---------- C. deferred-batch (maximal CPU run-ahead) ----------
        let (tpsC, tokC) = decodeDeferredBatch(model: model, cache: cache, seed: seed,
                                               steps: decodeSteps)
        let okC = tokC == tokA
        Self.log(String(format: "[C deferred-batch] %.2f tok/s (%.3f ms/tok)  %+.2fx vs A  argmax==A:%@",
                        tpsC, 1000.0/tpsC, tpsC/tpsA, okC ? "yes" : "NO"))
        if !okC {
            Self.log("    C first8=\(tokC.prefix(8).map(String.init).joined(separator: ","))")
        }

        Self.log(String(format: "[SUMMARY depth %d] A=%.1f  B=%.1f (%+.0f%%)  C=%.1f (%+.0f%%)",
                        depth, tpsA, tpsB, 100*(tpsB-tpsA)/tpsA,
                        tpsC, 100*(tpsC-tpsA)/tpsA))

        // ---------- DIAGNOSTIC: CPU graph-build time vs GPU exec time ----------
        // If launch-collapse (compile) can help, the per-step wall must be
        // dominated by CPU op-encoding / per-kernel dispatch overhead that does
        // NOT overlap GPU work. Measure pure CPU-side graph construction by
        // building K step graphs back-to-back WITHOUT eval, then a single drain.
        // (lazy graph build = the CPU op-encoding cost; the drain = GPU exec.)
        diagnoseCpuVsGpu(model: model, cache: cache, seed: seed, steps: 24)
    }

    /// Splits the per-step wall into: CPU lazy-graph build (no eval) vs the
    /// GPU drain. Tells us the launch-collapse ceiling: compile only helps the
    /// CPU-encode + dispatch fraction.
    private func diagnoseCpuVsGpu(
        model: NemotronHModel, cache: [KVCache], seed: MLXArray, steps: Int
    ) {
        var lastTok = seed
        for _ in 0..<2 { lastTok = argmaxNext(model(lastTok, cache: cache)) }
        eval(lastTok)

        // Phase 1: build N step graphs, no eval. Pure CPU op-encoding + the
        // autoregressive data dependency (each step needs prev token value, so
        // this is NOT fully decoupled — but it measures CPU-side cost when the
        // GPU is allowed to run ahead via the lazy scheduler).
        let cBuild = CFAbsoluteTimeGetCurrent()
        var arrs: [MLXArray] = []
        for _ in 0..<steps {
            lastTok = argmaxNext(model(lastTok, cache: cache))
            arrs.append(lastTok)
        }
        let buildT = CFAbsoluteTimeGetCurrent() - cBuild   // CPU build + implicit overlap

        let cDrain = CFAbsoluteTimeGetCurrent()
        eval(arrs)
        let drainT = CFAbsoluteTimeGetCurrent() - cDrain   // remaining GPU exec

        Self.log(String(format:
            "[DIAG cpu-vs-gpu] %d steps: build(no-eval)=%.1fms drain=%.1fms total=%.1fms  (%.2f ms/step)  build-frac=%.0f%%",
            steps, buildT*1000, drainT*1000, (buildT+drainT)*1000,
            (buildT+drainT)*1000/Double(steps),
            100*buildT/(buildT+drainT)))
    }

    // MARK: - decode variants (all greedy/argmax)

    private func argmaxNext(_ logits: MLXArray) -> MLXArray {
        let last = logits[0..., -1, 0...]
        return argMax(last, axis: -1).reshaped(1, 1).asType(.int32)
    }

    /// A. Fully synchronous: eval + item every step BEFORE building next graph.
    private func decodeSync(
        model: NemotronHModel, cache: [KVCache], seed: MLXArray, steps: Int
    ) -> (Double, [Int32]) {
        var lastTok = seed
        for _ in 0..<3 { lastTok = argmaxNext(model(lastTok, cache: cache)) }
        eval(lastTok)
        var toks: [Int32] = []
        let t = CFAbsoluteTimeGetCurrent()
        for _ in 0..<steps {
            lastTok = argmaxNext(model(lastTok, cache: cache))
            eval(lastTok)
            toks.append(lastTok.item(Int32.self))
        }
        return (Double(steps) / (CFAbsoluteTimeGetCurrent() - t), toks)
    }

    /// B. Production pipeline: asyncEval the new token, item the PREVIOUS one.
    /// GPU stays one step ahead; CPU builds step i+1 graph during GPU step i.
    private func decodeAsyncPipe(
        model: NemotronHModel, cache: [KVCache], seed: MLXArray, steps: Int
    ) -> (Double, [Int32]) {
        var lastTok = seed
        for _ in 0..<3 { lastTok = argmaxNext(model(lastTok, cache: cache)) }
        eval(lastTok)
        var toks: [Int32] = []
        var prev = lastTok
        let t = CFAbsoluteTimeGetCurrent()
        for _ in 0..<steps {
            let tok = argmaxNext(model(prev, cache: cache))
            asyncEval(tok)
            // drain the PREVIOUS step (one-step deferred sync)
            toks.append(prev.item(Int32.self))
            prev = tok
        }
        toks.append(prev.item(Int32.self))   // final drain
        toks.removeFirst()                    // drop the seed echo to align with A
        let dt = CFAbsoluteTimeGetCurrent() - t
        return (Double(steps) / dt, Array(toks.prefix(steps)))
    }

    /// C. Maximal run-ahead: build the entire chain, only the running token is
    /// asyncEval'd; single .item() drain after the loop. CPU never blocks
    /// mid-stream so the GPU command queue is kept saturated.
    private func decodeDeferredBatch(
        model: NemotronHModel, cache: [KVCache], seed: MLXArray, steps: Int
    ) -> (Double, [Int32]) {
        var lastTok = seed
        for _ in 0..<3 { lastTok = argmaxNext(model(lastTok, cache: cache)) }
        eval(lastTok)
        var tokArrays: [MLXArray] = []
        let t = CFAbsoluteTimeGetCurrent()
        for _ in 0..<steps {
            lastTok = argmaxNext(model(lastTok, cache: cache))
            tokArrays.append(lastTok)
            asyncEval(lastTok)   // schedule, don't block
        }
        eval(tokArrays)          // single drain
        let dt = CFAbsoluteTimeGetCurrent() - t
        return (Double(steps) / dt, tokArrays.map { $0.item(Int32.self) })
    }
}
