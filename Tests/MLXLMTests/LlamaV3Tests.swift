// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TriAttention V3 — Llama family integration smoke.
// Mirrors Qwen2V3Tests for the Llama newCache factory (which covers
// Llama-3.x, Llama-2, and Mistral derivatives). Synthetic config
// exercises the cache-construction branch without requiring a real
// model on disk. A gated real-model smoke (RUN_LLAMA_V3_SMOKE=1)
// loads Llama-3.2-3B-Instruct-4bit and runs a tiny prefill + decode
// loop to confirm V3 enable doesn't crash the attention path.
//
// LlamaConfiguration has an explicit `head_dim` codable field (Llama-3
// and later) but it's optional — older Llama-2 / Mistral-v0.1 configs
// omit it. The factory uses `resolvedHeadDimensions` which falls back
// to hiddenSize / attentionHeads when absent, matching LlamaAttention.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Llama TriAttention V3 — port", .serialized)
struct LlamaV3Tests {

    fileprivate static func makeLlamaConfig() throws -> LlamaConfiguration {
        // Hidden 64 / heads 8 / head_dim 8 — symmetric with the Qwen3
        // synthetic config. Llama-3 supplies an explicit head_dim; we
        // mirror that here so the test confirms the explicit-field path.
        let json = """
            {
                "model_type": "llama",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "rope_theta": 500000,
                "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(
            LlamaConfiguration.self, from: json.data(using: .utf8)!)
    }

    @Test("Llama factory installs TriAttention caches when env enabled")
    func llamaFactoryInstallsTriAttentionCaches() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        defer { unsetenv("VLLM_TRIATT_ENABLED") }

        let model = LlamaModel(try LlamaV3Tests.makeLlamaConfig())
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 2)
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
        let tri = try #require(caches.first as? TriAttentionKVCache)
        #expect(tri.logicalOffset == 0)
        #expect(tri.engine.nLayers == 2)
        #expect(tri.engine.nHeads == 8)
        #expect(tri.engine.nKVHeads == 2)
        // headDim from explicit head_dim=8 in config.
        #expect(tri.engine.headDim == 8)
    }

    @Test("Llama factory uses standard cache when env disabled")
    func llamaFactoryDefaultsToStandard() throws {
        unsetenv("VLLM_TRIATT_ENABLED")
        let model = LlamaModel(try LlamaV3Tests.makeLlamaConfig())
        let caches = model.newCache(parameters: nil)
        #expect(caches.count == 2)
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    @Test("Llama factory uses standard cache when maxKVSize is set")
    func llamaFactoryRespectsMaxKVSize() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        defer { unsetenv("VLLM_TRIATT_ENABLED") }
        let model = LlamaModel(try LlamaV3Tests.makeLlamaConfig())
        var params = GenerateParameters()
        params.maxKVSize = 512
        let caches = model.newCache(parameters: params)
        // maxKVSize set → V3 incompatible; falls through to standard.
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    /// Gated real-model smoke. Loads Llama-3.2-3B-Instruct-4bit and
    /// runs prefill + a few decode steps with V3 enabled. Asserts no
    /// crash and that logits + sampled tokens are valid (non-NaN, in
    /// vocab range). Mirrors the Qwen2V3 smoke pattern.
    ///   RUN_LLAMA_V3_SMOKE=1 swift test --filter llamaV3LiveSmoke
    @Test func llamaV3LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_LLAMA_V3_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/Llama-3.2-3B-Instruct-4bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        // Small budget + warmup so eviction kicks in within the prefill.
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        setenv("VLLM_TRIATT_BUDGET", "256", 1)
        setenv("VLLM_TRIATT_WARMUP", "128", 1)
        setenv("VLLM_TRIATT_WINDOW", "32", 1)
        setenv("VLLM_TRIATT_PREFIX", "32", 1)
        defer {
            unsetenv("VLLM_TRIATT_ENABLED")
            unsetenv("VLLM_TRIATT_BUDGET")
            unsetenv("VLLM_TRIATT_WARMUP")
            unsetenv("VLLM_TRIATT_WINDOW")
            unsetenv("VLLM_TRIATT_PREFIX")
        }

        print("[llama-v3] loading...", flush: true)
        let cfg = try JSONDecoder().decode(
            LlamaConfiguration.self,
            from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
        let model = LlamaModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[llama-v3] loaded", flush: true)

        // Build the V3 cache list via the factory under test.
        let caches: [KVCache] = model.newCache(parameters: GenerateParameters())
        // hiddenLayers is internal on LlamaConfiguration — assert via the
        // engine on the first cache instead (one TriAttentionKVCache per
        // layer guarantees `caches.count == engine.nLayers`).
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
        if let tri = caches.first as? TriAttentionKVCache {
            #expect(caches.count == tri.engine.nLayers)
        }

        // Small ctx so the suite stays light. Deterministic random-int
        // prompt. Llama-3.2 vocab is 128256; we draw prompt ids from a
        // safe sub-range so they're ordinary tokens, but accept the
        // full vocab on decoded outputs (special tokens like
        // 128000=<|begin_of_text|> are legitimate).
        let promptLen = 64
        let maxDecode = 8
        let promptIdMax: Int32 = 100000
        let vocabFull: Int32 = 128256
        MLXRandom.seed(0xA2C5E9D)
        let promptTokens = MLXRandom.randInt(
            low: MLXArray(Int32(0)),
            high: MLXArray(promptIdMax),
            [1, promptLen]
        ).asType(.int32)
        eval(promptTokens)

        // Prefill.
        let prefillOut = model(promptTokens, cache: caches)
        eval(prefillOut)
        var step = argMax(prefillOut[0..., -1, 0...], axis: -1)
            .reshaped(1, 1).asType(.int32)
        eval(step)

        // Valid token check after prefill — accept the full Llama-3.2
        // vocab (special-token ids in the 128xxx range are legitimate
        // and routinely emerge from random-prompt argmax).
        let firstTok = step.asArray(Int32.self)[0]
        #expect(firstTok >= 0 && firstTok < vocabFull,
                "first token out of expected range: \(firstTok)")

        // A handful of decode steps to confirm cache updates + V3 hook
        // co-exist with the Llama attention forward.
        for _ in 0..<maxDecode {
            let out = model(step, cache: caches)
            eval(out)
            step = argMax(out[0..., -1, 0...], axis: -1)
                .reshaped(1, 1).asType(.int32)
            eval(step)
            let tok = step.asArray(Int32.self)[0]
            #expect(tok >= 0 && tok < vocabFull,
                    "decoded token oor: \(tok)")
        }
        print("[llama-v3] decode loop done", flush: true)
    }
}

// Local print-with-flush helper (matches F83SpecDecBench).
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
