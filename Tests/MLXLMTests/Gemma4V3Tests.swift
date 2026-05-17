// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TriAttention V3 — Gemma 4 family integration smoke.
// Mirrors Qwen2V3Tests / TriAttentionV3Tests but exercises Gemma 4's
// sliding/global heterogeneous-attention factory branch documented in
// `Gemma4TextModel.newCache` (Libraries/MLXLLM/Models/Gemma4.swift).
//
// Architectural caveat (locked in by these tests):
// `TriAttentionV3Engine` takes a single `(nHeads, nKVHeads, headDim,
// ropeTheta)` tuple. Real Gemma 4 checkpoints (e2b / e4b / 26B-A4B)
// have sliding-attention layers and full-attention layers with DIFFERENT
// `headDim` and `kvHeads` values, so a single V3 engine cannot serve
// them simultaneously. The factory detects this and falls through to the
// default sliding-window cache mix even when `VLLM_TRIATT_ENABLED=1`
// is set. The tests assert both branches:
//   - heterogeneous synthetic config + V3 enabled  -> falls through
//   - homogeneous synthetic config + V3 enabled    -> installs V3
//   - V3 disabled (any config)                     -> default mix
//   - V3 enabled but `maxKVSize` set               -> default mix
//
// A gated live smoke (RUN_GEMMA4_V3_SMOKE=1) loads the real
// gemma-4-26b-a4b-4bit checkpoint and confirms that enabling V3 does
// NOT install TriAttentionKVCache (because the checkpoint is hetero) but
// also does not crash the prefill + decode path.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Gemma4 TriAttention V3 — port", .serialized)
struct Gemma4V3Tests {

    /// Heterogeneous synthetic config: matches the real Gemma 4 26B-A4B
    /// shape (sliding `headDim=256, kvHeads=8`, global `headDim=512,
    /// kvHeads=2`). Hits the documented V3-skip branch.
    fileprivate static func makeHeterogeneousConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "num_attention_heads": 8,
                "head_dim": 8,
                "global_head_dim": 16,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "num_key_value_heads": 4,
                "num_global_key_value_heads": 2,
                "sliding_window": 32,
                "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
                "tie_word_embeddings": true,
                "rope_theta": 10000,
                "global_rope_theta": 1000000,
                "partial_rotary_factor": 0.25,
                "hidden_size_per_layer_input": 0
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    /// Homogeneous synthetic config — `head_dim == global_head_dim` AND
    /// `num_key_value_heads == num_global_key_value_heads` AND only one
    /// layer type. Hits the V3-install branch so the test covers the
    /// factory's other arm.
    fileprivate static func makeHomogeneousConfig() throws -> Gemma4TextConfiguration {
        let json = """
            {
                "model_type": "gemma4_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "num_attention_heads": 8,
                "head_dim": 8,
                "global_head_dim": 8,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 2,
                "sliding_window": 32,
                "layer_types": ["sliding_attention", "sliding_attention"],
                "tie_word_embeddings": true,
                "rope_theta": 10000,
                "global_rope_theta": 1000000,
                "partial_rotary_factor": 0.25,
                "hidden_size_per_layer_input": 0
            }
            """
        return try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    @Test("Gemma4 factory skips V3 install on heterogeneous attention shapes")
    func gemma4FactorySkipsV3OnHeterogeneousConfig() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        setenv("VLLM_TRIATT_GEMMA4_SILENT", "1", 1)
        defer {
            unsetenv("VLLM_TRIATT_ENABLED")
            unsetenv("VLLM_TRIATT_GEMMA4_SILENT")
        }

        let model = Gemma4TextModel(try Gemma4V3Tests.makeHeterogeneousConfig())
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 4)
        // No TriAttentionKVCache anywhere — heterogeneous shape means V3
        // engine cannot be installed; factory must fall through.
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    @Test("Gemma4 factory installs V3 caches on homogeneous attention shapes")
    func gemma4FactoryInstallsV3OnHomogeneousConfig() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        setenv("VLLM_TRIATT_GEMMA4_SILENT", "1", 1)
        defer {
            unsetenv("VLLM_TRIATT_ENABLED")
            unsetenv("VLLM_TRIATT_GEMMA4_SILENT")
        }

        let model = Gemma4TextModel(try Gemma4V3Tests.makeHomogeneousConfig())
        let caches = model.newCache(parameters: nil)

        #expect(caches.count == 2)
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
        let tri = try #require(caches.first as? TriAttentionKVCache)
        #expect(tri.logicalOffset == 0)
        #expect(tri.engine.nLayers == 2)
        #expect(tri.engine.nHeads == 8)
        #expect(tri.engine.nKVHeads == 2)
        #expect(tri.engine.headDim == 8)
    }

    @Test("Gemma4 factory uses default mix when V3 env disabled")
    func gemma4FactoryDefaultsWhenV3Disabled() throws {
        unsetenv("VLLM_TRIATT_ENABLED")
        let model = Gemma4TextModel(try Gemma4V3Tests.makeHomogeneousConfig())
        let caches = model.newCache(parameters: nil)
        #expect(caches.count == 2)
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    @Test("Gemma4 factory falls through when maxKVSize is set")
    func gemma4FactoryRespectsMaxKVSize() throws {
        setenv("VLLM_TRIATT_ENABLED", "1", 1)
        setenv("VLLM_TRIATT_GEMMA4_SILENT", "1", 1)
        defer {
            unsetenv("VLLM_TRIATT_ENABLED")
            unsetenv("VLLM_TRIATT_GEMMA4_SILENT")
        }
        let model = Gemma4TextModel(try Gemma4V3Tests.makeHomogeneousConfig())
        var params = GenerateParameters()
        params.maxKVSize = 512
        let caches = model.newCache(parameters: params)
        // maxKVSize set → V3 incompatible (mirrors Qwen2/Qwen3 behavior);
        // factory falls through to the eviction-windowed default.
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
    }

    /// Gated real-model smoke. Loads gemma-4-26b-a4b-4bit and runs a tiny
    /// prefill + decode loop with V3 enabled. Asserts:
    ///   - factory does NOT install TriAttentionKVCache (heterogeneous
    ///     attention shapes — documented architectural skip).
    ///   - prefill + decode still run without crashing.
    ///   - sampled tokens stay in-vocab.
    /// Run with: RUN_GEMMA4_V3_SMOKE=1 swift test --filter gemma4V3LiveSmoke
    @Test func gemma4V3LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_GEMMA4_V3_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/gemma-4-26b-a4b-4bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        // V3 enabled — but Gemma 4 26B-A4B has heterogeneous heads so the
        // factory will log + skip V3. We still want decode to succeed.
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

        print("[gemma4-v3] loading...", flush: true)
        let cfg = try JSONDecoder().decode(
            Gemma4TextConfiguration.self,
            from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
        let model = Gemma4TextModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[gemma4-v3] loaded", flush: true)

        let caches: [KVCache] = model.newCache(parameters: GenerateParameters())
        // Real Gemma 4 26B-A4B: sliding+global heads differ → factory must
        // skip V3 install. If a future config tweak makes shapes match
        // and this assertion flips, update both factory and test together.
        #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })

        // Small ctx so the suite stays light. Deterministic random-int
        // prompt. Gemma 4 vocab is 262144; draw from a safe sub-range.
        let promptLen = 32
        let maxDecode = 4
        let promptIdMax: Int32 = 100000
        let vocabFull: Int32 = 262144
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
        print("[gemma4-v3] decode loop done", flush: true)
    }
}

// Local print-with-flush helper (matches Qwen2V3Tests / F83SpecDecBench).
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
