// F-83 night sprint round 2 — Swift speculative decoding bench.
//
// Uses the f83_perfBench256K_14B1M direct-loading pattern (Qwen2Model +
// loadWeights) to avoid the loadModel/factory hang. Wires up
// SpeculativeTokenIterator (port of mlx-lm's speculative_generate_step)
// directly with main=Qwen2.5-14B-1M-4bit + draft=Qwen2.5-3B-4bit.
//
// Gated: RUN_F83_SWIFT_SPEC=1 swift test --filter f83_specDecBench

import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXNN
import Testing

@Suite("F-83 Swift spec-dec")
struct F83SpecDecBenchSuite {

    @Test func f83_specDecBench() throws {
        guard ProcessInfo.processInfo.environment["RUN_F83_SWIFT_SPEC"] == "1" else {
            return
        }

        let mainPath = URL(fileURLWithPath: "\(NSHomeDirectory())/models/Qwen2.5-14B-Instruct-1M-4bit")
        // 1.5B (~10x smaller) is the sweet spot for spec-dec with this target.
        let draftName = ProcessInfo.processInfo.environment["F83_SPEC_DRAFT"]
            ?? "Qwen2.5-1.5B-Instruct-4bit"
        let draftPath = URL(fileURLWithPath: "\(NSHomeDirectory())/models/\(draftName)")
        for url in [mainPath, draftPath] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                Issue.record("model not present: \(url.path)")
                return
            }
        }

        print("[swift-spec] loading main...", flush: true)
        let mainCfg = try JSONDecoder().decode(
            Qwen2Configuration.self,
            from: Data(contentsOf: mainPath.appendingPathComponent("config.json"))
        )
        let mainModel = Qwen2Model(mainCfg)
        try loadWeights(
            modelDirectory: mainPath, model: mainModel,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[swift-spec] main loaded", flush: true)

        print("[swift-spec] loading draft...", flush: true)
        let draftCfg = try JSONDecoder().decode(
            Qwen2Configuration.self,
            from: Data(contentsOf: draftPath.appendingPathComponent("config.json"))
        )
        let draftModel = Qwen2Model(draftCfg)
        try loadWeights(
            modelDirectory: draftPath, model: draftModel,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[swift-spec] draft loaded", flush: true)

        // Build a deterministic prompt by random-int generation (matches
        // the f83 bench harness pattern). We don't use a real tokenizer
        // here — just random token ids. SpeculativeTokenIterator only
        // needs that prompt tokens are valid for both vocab sizes.
        let promptLen = Int(ProcessInfo.processInfo.environment["F83_SPEC_PROMPT_LEN"] ?? "64") ?? 64
        let maxDecode = Int(ProcessInfo.processInfo.environment["F83_SPEC_DECODE"] ?? "32") ?? 32
        // Qwen2.5 vocab = 152064; using safe upper bound = 100000.
        let vocab: Int32 = 100000
        MLXRandom.seed(0xF83C5E9D)
        // SpeculativeTokenIterator's prepare expects 1D [L] tokens (it
        // slices via axis 0). 2D [1, L] yields empty after chunk-slice
        // → reshape with -1 on empty inside model forward → fatal.
        let promptTokens2D = MLXRandom.randInt(
            low: MLXArray(Int32(0)),
            high: MLXArray(vocab),
            [1, promptLen]
        ).asType(.int32)
        eval(promptTokens2D)
        let promptTokens1D = promptTokens2D.reshaped([promptLen])
        eval(promptTokens1D)
        // Baseline uses 2D, spec uses 1D LMInput.
        let promptTokens = promptTokens2D
        let input = LMInput(tokens: promptTokens1D)
        print("[swift-spec] prompt=\(promptLen) decode=\(maxDecode)", flush: true)

        // Warm both models with one prefill so first-iter doesn't dominate.
        let warmCache: [KVCache] = mainModel.newCache(parameters: GenerateParameters())
        let warmOut = mainModel(promptTokens, cache: warmCache)
        eval(warmOut)
        let warmDraftCache: [KVCache] = draftModel.newCache(parameters: GenerateParameters())
        let warmDraftOut = draftModel(promptTokens, cache: warmDraftCache)
        eval(warmDraftOut)
        print("[swift-spec] warm done", flush: true)

        let params = GenerateParameters(maxTokens: maxDecode, temperature: 0)

        // Baseline — direct manual decode loop (matches f83 harness).
        func runBaseline() -> Double {
            let cache: [KVCache] = mainModel.newCache(parameters: params)
            // Prefill
            let prefillOut = mainModel(promptTokens, cache: cache)
            var step = argMax(prefillOut[0..., -1, 0...], axis: -1).reshaped(1, 1).asType(.int32)
            eval(step)
            let t0 = Date()
            for _ in 0..<(maxDecode - 1) {
                let out = mainModel(step, cache: cache)
                step = argMax(out[0..., -1, 0...], axis: -1).reshaped(1, 1).asType(.int32)
                eval(step)
            }
            let dt = Date().timeIntervalSince(t0)
            return dt / Double(maxDecode - 1) * 1000.0
        }

        // SpecDec via SpeculativeTokenIterator.
        func runSpec(k: Int) throws -> (msPerTok: Double, accepted: Int, proposed: Int) {
            print("[swift-spec] spec init k=\(k)", flush: true)
            var iter = try SpeculativeTokenIterator(
                input: input,
                mainModel: mainModel,
                draftModel: draftModel,
                parameters: params,
                numDraftTokens: k)
            print("[swift-spec] spec ctor done k=\(k)", flush: true)
            // Pull first token to skip prefill from timing.
            let first = iter.next()
            print("[swift-spec] spec first=\(String(describing: first)) k=\(k)", flush: true)
            let t0 = Date()
            var count = 0
            while count < (maxDecode - 1), let _ = iter.next() {
                count += 1
            }
            print("[swift-spec] spec done k=\(k) count=\(count)", flush: true)
            let dt = Date().timeIntervalSince(t0)
            return (dt / Double(max(1, count)) * 1000.0, iter.draftAcceptedCount, iter.draftProposedCount)
        }

        // Warmup baseline (discard first run).
        _ = runBaseline()
        let baseMs = runBaseline()
        print(String(format: "[swift-spec] BASELINE %.2f ms/tok (%.1f tps)",
            baseMs, 1000.0 / baseMs), flush: true)

        // NgramSpec — no draft model, uses prompt itself as draft source.
        // Works well for repetitive prompts (code, summarization).
        func runNgram(ngramSize: Int, maxDraft: Int) throws -> (msPerTok: Double, accepted: Int, proposed: Int) {
            print("[swift-spec] ngram init n=\(ngramSize) d=\(maxDraft)", flush: true)
            var ngramParams = params
            ngramParams.ngramSize = ngramSize
            ngramParams.maxNgramDraftTokens = maxDraft
            var iter = try NGramSpeculativeTokenIterator(
                input: input,
                mainModel: mainModel,
                parameters: ngramParams)
            print("[swift-spec] ngram ctor done", flush: true)
            _ = iter.next()
            print("[swift-spec] ngram first done", flush: true)
            let t0 = Date()
            var count = 0
            while count < (maxDecode - 1), let _ = iter.next() {
                count += 1
            }
            let dt = Date().timeIntervalSince(t0)
            return (dt / Double(max(1, count)) * 1000.0, iter.ngramAcceptedCount, iter.ngramProposedCount)
        }

        for (n, d) in [(2, 2), (3, 4), (3, 8)] {
            do {
                let (ms, acc, prop) = try runNgram(ngramSize: n, maxDraft: d)
                let acceptRate = prop > 0 ? Double(acc) / Double(prop) * 100 : 0
                let speedup = baseMs / ms
                print(String(format:
                    "[swift-spec] NGRAM n=%d d=%d %.2f ms/tok (%.1f tps) speedup=%.2fx accept=%.1f%% (%d/%d)",
                    n, d, ms, 1000.0 / ms, speedup, acceptRate, acc, prop), flush: true)
            } catch {
                print("[swift-spec] NGRAM n=\(n) d=\(d) ERROR: \(error)", flush: true)
            }
        }

        // Draft-model spec (with safer error handling)
        for k in [2, 3] {
            do {
                let (ms, acc, prop) = try runSpec(k: k)
                let acceptRate = prop > 0 ? Double(acc) / Double(prop) * 100 : 0
                let speedup = baseMs / ms
                print(String(format:
                    "[swift-spec] SPEC-k%d %.2f ms/tok (%.1f tps) speedup=%.2fx accept=%.1f%% (%d/%d)",
                    k, ms, 1000.0 / ms, speedup, acceptRate, acc, prop), flush: true)
            } catch {
                print("[swift-spec] SPEC-k\(k) ERROR: \(error)", flush: true)
            }
        }
    }
}

// Print with flush helper.
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
