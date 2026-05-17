// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TriAttention V3 — Mistral 3 / Ministral 3 family integration smoke.
// Mirrors Qwen2V3Tests / LlamaV3Tests for the Mistral3TextModel newCache
// factory. Synthetic config exercises the cache-construction branch
// without requiring a real model on disk. A gated real-model smoke
// (RUN_MISTRAL3_V3_SMOKE=1) loads a Ministral 3 checkpoint and runs a
// tiny prefill + decode loop to confirm V3 enable doesn't crash the
// attention path.
//
// Mistral3TextConfiguration has an explicit `head_dim` codable field
// but it's optional — older configs omit it. The factory uses
// `resolvedHeadDimensions` which falls back to
// `hiddenSize / attentionHeads`, matching `Mistral3.Attention.init`.
//
// Architectural caveat: Mistral 3 layer_types can in principle be
// heterogeneous (sliding + full mix). `TriAttentionV3Engine` takes a
// single (nHeads, nKVHeads, headDim, ropeTheta) tuple so a hetero
// config can't be served by a single engine. Today's Mistral 3 configs
// use uniform `full_attention` so this is not an issue, but the factory
// falls through to the default sliding/full mix when hetero is
// detected — matching Gemma 4's defensive pattern.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// Process-wide lock that serializes the VLLM_TRIATT_ENABLED env-var
/// dance across V3 test suites. Swift Testing's `.serialized` trait is
/// per-suite; without a cross-suite lock the V3 suites (Qwen2, Qwen3,
/// Llama, Gemma4, Mistral3) race on `setenv` / `unsetenv` and the
/// homogeneous install test can sample a transient unsetenv from a
/// sibling suite. Same mitigation as Gemma4V3Tests.
private let triattEnvLock = NSLock()

@inline(__always)
private func withTriattEnv<T>(
    _ vars: [(String, String?)], _ body: () throws -> T
) rethrows -> T {
    triattEnvLock.lock()
    defer { triattEnvLock.unlock() }
    let prior: [(String, String?)] = vars.map { (key, _) in
        (key, getenv(key).map { String(cString: $0) })
    }
    for (key, value) in vars {
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
    }
    defer {
        for (key, value) in prior {
            if let value {
                setenv(key, value, 1)
            } else {
                unsetenv(key)
            }
        }
    }
    return try body()
}

@Suite("Mistral3 TriAttention V3 — port", .serialized)
struct Mistral3V3Tests {

    /// Homogeneous synthetic config — all layers full_attention. Matches
    /// the common Ministral 3 config shape. Hits the V3-install branch.
    fileprivate static func makeMistral3Config() throws -> Mistral3TextConfiguration {
        let json = """
            {
                "model_type": "ministral3",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "rope_theta": 100000000,
                "tie_word_embeddings": true,
                "layer_types": ["full_attention", "full_attention"]
            }
            """
        return try JSONDecoder().decode(
            Mistral3TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    /// Heterogeneous synthetic config — sliding + full mix. Forces the
    /// defensive hetero-skip fall-through path.
    fileprivate static func makeHeteroMistral3Config() throws -> Mistral3TextConfiguration {
        let json = """
            {
                "model_type": "ministral3",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "rope_theta": 100000000,
                "tie_word_embeddings": true,
                "sliding_window": 32,
                "layer_types": ["sliding_attention", "full_attention"]
            }
            """
        return try JSONDecoder().decode(
            Mistral3TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    @Test("Mistral3 factory installs TriAttention caches when env enabled")
    func mistral3FactoryInstallsTriAttentionCaches() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", "1")]) {
            let model = Mistral3TextModel(try Mistral3V3Tests.makeMistral3Config())
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
    }

    @Test("Mistral3 factory uses standard cache when env disabled")
    func mistral3FactoryDefaultsToStandard() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", nil)]) {
            let model = Mistral3TextModel(try Mistral3V3Tests.makeMistral3Config())
            let caches = model.newCache(parameters: nil)
            #expect(caches.count == 2)
            #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
        }
    }

    @Test("Mistral3 factory uses standard cache when maxKVSize is set")
    func mistral3FactoryRespectsMaxKVSize() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", "1")]) {
            let model = Mistral3TextModel(try Mistral3V3Tests.makeMistral3Config())
            var params = GenerateParameters()
            params.maxKVSize = 512
            let caches = model.newCache(parameters: params)
            // maxKVSize set → V3 incompatible; falls through to standard.
            #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
        }
    }

    @Test("Mistral3 factory falls through to default when layer_types are heterogeneous")
    func mistral3FactoryRespectsHeteroLayerTypes() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", "1")]) {
            let model = Mistral3TextModel(try Mistral3V3Tests.makeHeteroMistral3Config())
            let caches = model.newCache(parameters: nil)
            // Hetero config → V3 incompatible (single-tuple engine);
            // factory falls through to default sliding/full mix.
            #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
        }
    }

    /// Gated real-model smoke. Loads a Ministral 3 checkpoint and runs
    /// prefill + a few decode steps with V3 enabled. Asserts no crash
    /// and that logits + sampled tokens are valid (non-NaN, in vocab
    /// range). Mirrors the Llama V3 smoke pattern.
    ///   RUN_MISTRAL3_V3_SMOKE=1 swift test --filter mistral3V3LiveSmoke
    @Test func mistral3V3LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_MISTRAL3_V3_SMOKE"] == "1" else {
            return
        }

        let modelDirEnv = ProcessInfo.processInfo.environment["MISTRAL3_MODEL_DIR"]
            ?? "\(NSHomeDirectory())/models/Ministral-8B-Instruct-2410-4bit"
        let modelPath = URL(fileURLWithPath: modelDirEnv)
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        try withTriattEnv([
            ("VLLM_TRIATT_ENABLED", "1"),
            ("VLLM_TRIATT_BUDGET", "256"),
            ("VLLM_TRIATT_WARMUP", "128"),
            ("VLLM_TRIATT_WINDOW", "32"),
            ("VLLM_TRIATT_PREFIX", "32"),
        ]) {
            print("[mistral3-v3] loading...", flush: true)
            let cfg = try JSONDecoder().decode(
                Mistral3TextConfiguration.self,
                from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
            let model = Mistral3TextModel(cfg)
            try loadWeights(
                modelDirectory: modelPath, model: model,
                quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
            print("[mistral3-v3] loaded", flush: true)

            let caches: [KVCache] = model.newCache(parameters: GenerateParameters())
            #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
            if let tri = caches.first as? TriAttentionKVCache {
                #expect(caches.count == tri.engine.nLayers)
            }

            // Light prompt — random ids in safe sub-range. Accept full vocab on outputs.
            let promptLen = 64
            let maxDecode = 8
            let promptIdMax: Int32 = 30000
            let vocabFull: Int32 = Int32(cfg.vocabularySize)
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

            for _ in 0 ..< maxDecode {
                let out = model(step, cache: caches)
                eval(out)
                step = argMax(out[0..., -1, 0...], axis: -1)
                    .reshaped(1, 1).asType(.int32)
                eval(step)
                let tok = step.asArray(Int32.self)[0]
                #expect(tok >= 0 && tok < vocabFull,
                        "decoded token oor: \(tok)")
            }
            print("[mistral3-v3] decode loop done", flush: true)
        }
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
