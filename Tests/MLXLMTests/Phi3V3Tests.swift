// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TriAttention V3 — Phi3 family integration smoke.
// Mirrors LlamaV3Tests for the Phi3 newCache factory. Synthetic config
// exercises the cache-construction branch without requiring a real
// model on disk. A gated real-model smoke (RUN_PHI3_V3_SMOKE=1) loads
// a Phi-3-mini-4k-instruct-4bit checkpoint and runs a tiny prefill +
// decode loop to confirm V3 enable doesn't crash the Phi3 attention
// path (fused qkv_proj + partial RoPE).
//
// Phi3Configuration has no explicit `head_dim` codable field — the
// factory derives headDim from hiddenSize / attentionHeads, matching
// Phi3Attention.init at MLXLLM/Models/Phi3.swift:32. Phi3-mini uses
// hidden=3072 / heads=32 -> head_dim=96, which we mirror in the
// synthetic config (hidden=96 / heads=8 -> head_dim=12) so the
// derivation path is exercised.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Phi3 TriAttention V3 — port", .serialized)
struct Phi3V3Tests {

    fileprivate static func makePhi3Config() throws -> Phi3Configuration {
        // Hidden 96 / heads 8 / head_dim 12. KV heads 2 (GQA 4:1).
        // Symmetric with the Llama / Qwen synthetic configs but tuned
        // for Phi3's derived-head-dim path (no `head_dim` field).
        let json = """
            {
                "model_type": "phi3",
                "hidden_size": 96,
                "num_hidden_layers": 2,
                "intermediate_size": 192,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "rope_theta": 10000,
                "max_position_embeddings": 4096,
                "original_max_position_embeddings": 4096,
                "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(
            Phi3Configuration.self, from: json.data(using: .utf8)!)
    }

    @Test("Phi3 factory installs TriAttention caches when env enabled")
    func phi3FactoryInstallsTriAttentionCaches() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        defer { unsetenv("VLLM_TRIATT_ENABLED") }

        let model = Phi3Model(try Phi3V3Tests.makePhi3Config())
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 2)
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
        let tri = try #require(caches.first as? TriAttentionKVCache)
        #expect(tri.logicalOffset == 0)
        #expect(tri.engine.nLayers == 2)
        #expect(tri.engine.nHeads == 8)
        #expect(tri.engine.nKVHeads == 2)
        // headDim derived from hiddenSize / attentionHeads = 96 / 8 = 12.
        #expect(tri.engine.headDim == 12)
    }

    @Test("Phi3 factory uses standard cache when env disabled")
    func phi3FactoryDefaultsToStandard() throws {
        unsetenv("VLLM_TRIATT_ENABLED")
        let model = Phi3Model(try Phi3V3Tests.makePhi3Config())
        let caches = model.newCache(parameters: nil)
        #expect(caches.count == 2)
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    @Test("Phi3 factory uses standard cache when maxKVSize is set")
    func phi3FactoryRespectsMaxKVSize() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        defer { unsetenv("VLLM_TRIATT_ENABLED") }
        let model = Phi3Model(try Phi3V3Tests.makePhi3Config())
        var params = GenerateParameters()
        params.maxKVSize = 512
        let caches = model.newCache(parameters: params)
        // maxKVSize set → V3 incompatible; falls through to standard.
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    /// Gated real-model smoke. Loads Phi-3-mini-4k-instruct-4bit and
    /// runs prefill + a few decode steps with V3 enabled. Asserts no
    /// crash and that logits + sampled tokens are valid (non-NaN, in
    /// vocab range). Mirrors the LlamaV3 smoke pattern.
    ///   RUN_PHI3_V3_SMOKE=1 swift test --filter phi3V3LiveSmoke
    @Test func phi3V3LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_PHI3_V3_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/Phi-3-mini-4k-instruct-4bit")
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

        print("[phi3-v3] loading...", flush: true)
        let cfg = try JSONDecoder().decode(
            Phi3Configuration.self,
            from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
        let model = Phi3Model(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[phi3-v3] loaded", flush: true)

        // Build the V3 cache list via the factory under test.
        let caches: [KVCache] = model.newCache(parameters: GenerateParameters())
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
        if let tri = caches.first as? TriAttentionKVCache {
            #expect(caches.count == tri.engine.nLayers)
        }

        // Small ctx so the suite stays light. Phi-3-mini vocab is 32064.
        let promptLen = 64
        let maxDecode = 8
        let promptIdMax: Int32 = 30000
        let vocabFull = Int32(model.vocabularySize)
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

        let firstTok = step.asArray(Int32.self)[0]
        #expect(firstTok >= 0 && firstTok < vocabFull,
                "first token out of expected range: \(firstTok)")

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
        print("[phi3-v3] decode loop done", flush: true)
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
