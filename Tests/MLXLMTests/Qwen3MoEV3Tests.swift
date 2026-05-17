// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TriAttention V3 — Qwen3 MoE family integration smoke.
// Mirrors Qwen2V3Tests / TriAttentionV3Tests but exercises the
// `Qwen3MoEModel.newCache` factory at Libraries/MLXLLM/Models/
// Qwen3MoE.swift. Qwen3MoE shares Qwen3's dense-attention shape
// (q_norm/k_norm + RoPE, single attentionHeads/kvHeads/headDim across
// all layers) so V3 install is a direct mirror of the Qwen3 dense
// factory — MoE-routed FFN is orthogonal to KV-cache layout.
//
// A gated real-model smoke (RUN_QWEN3MOE_V3_SMOKE=1) loads
// Qwen3-Coder-30B-A3B-Instruct-MLX-6bit and runs a tiny prefill +
// decode loop with V3 enabled. Uses the shared `triattEnvLock` /
// `withTriattEnv` pattern from Gemma4V3Tests to defeat the cross-
// suite env-var race.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// Process-wide lock that serializes the VLLM_TRIATT_ENABLED env-var
/// dance across V3 test suites. Matches the pattern in Gemma4V3Tests.
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

@Suite("Qwen3MoE TriAttention V3 — port", .serialized)
struct Qwen3MoEV3Tests {

    /// Synthetic Qwen3MoE config. Mirrors the real Qwen3-Coder-30B-A3B
    /// architectural shape in miniature: dense attention (head_dim=8,
    /// 8 heads, 2 KV heads) + MoE FFN (4 experts top-2). Small enough
    /// that random-weight init returns fast.
    fileprivate static func makeQwen3MoEConfig() throws -> Qwen3MoEConfiguration {
        let json = """
            {
                "model_type": "qwen3_moe",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "decoder_sparse_step": 1,
                "mlp_only_layers": [],
                "moe_intermediate_size": 64,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "rope_theta": 10000000,
                "tie_word_embeddings": true,
                "norm_topk_prob": true
            }
            """
        return try JSONDecoder().decode(
            Qwen3MoEConfiguration.self, from: json.data(using: .utf8)!)
    }

    @Test("Qwen3MoE factory installs TriAttention caches when env enabled")
    func qwen3MoEFactoryInstallsTriAttentionCaches() throws {
        // Retry loop defeats cross-suite env-var race (sibling V3 suites
        // call `unsetenv("VLLM_TRIATT_ENABLED")` between our setenv and
        // the factory's env read). Matches Gemma4V3Tests pattern.
        var caches: [KVCache] = []
        let model = Qwen3MoEModel(try Qwen3MoEV3Tests.makeQwen3MoEConfig())
        var installed = false
        for _ in 0..<8 {
            caches = withTriattEnv([
                ("VLLM_TRIATT_ENABLED", "1"),
            ]) {
                model.newCache(parameters: nil)
            }
            installed = caches.allSatisfy { $0 is TriAttentionKVCache }
            if installed { break }
        }

        #expect(caches.count == 2)
        #expect(installed)
        if let tri = caches.first as? TriAttentionKVCache {
            #expect(tri.logicalOffset == 0)
            #expect(tri.engine.nLayers == 2)
            #expect(tri.engine.nHeads == 8)
            #expect(tri.engine.nKVHeads == 2)
            #expect(tri.engine.headDim == 8)
        }
    }

    @Test("Qwen3MoE factory uses standard cache when env disabled")
    func qwen3MoEFactoryDefaultsToStandard() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", nil)]) {
            let model = Qwen3MoEModel(try Qwen3MoEV3Tests.makeQwen3MoEConfig())
            let caches = model.newCache(parameters: nil)
            #expect(caches.count == 2)
            #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
        }
    }

    @Test("Qwen3MoE factory uses standard cache when maxKVSize is set")
    func qwen3MoEFactoryRespectsMaxKVSize() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", "1")]) {
            let model = Qwen3MoEModel(try Qwen3MoEV3Tests.makeQwen3MoEConfig())
            var params = GenerateParameters()
            params.maxKVSize = 512
            let caches = model.newCache(parameters: params)
            // maxKVSize set → V3 incompatible; falls through to standard.
            #expect(caches.allSatisfy { !($0 is TriAttentionKVCache) })
        }
    }

    @Test("Qwen3MoE factory cache count matches hiddenLayers")
    func qwen3MoEFactoryCacheCount() throws {
        try withTriattEnv([("VLLM_TRIATT_ENABLED", nil)]) {
            let model = Qwen3MoEModel(try Qwen3MoEV3Tests.makeQwen3MoEConfig())
            let caches = model.newCache(parameters: nil)
            // Synthetic config has num_hidden_layers=2.
            #expect(caches.count == 2)
        }
    }

    /// Gated real-model smoke. Loads Qwen3-Coder-30B-A3B-Instruct-MLX-6bit
    /// and runs prefill + a few decode steps with V3 enabled. Asserts no
    /// crash and that sampled tokens are valid (in vocab range). Mirrors
    /// Qwen2V3Tests.qwen2V3LiveSmoke for the Qwen3-MoE family.
    ///   RUN_QWEN3MOE_V3_SMOKE=1 swift test --filter qwen3MoEV3LiveSmoke
    @Test func qwen3MoEV3LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_QWEN3MOE_V3_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit")
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

        print("[qwen3moe-v3] loading...", flush: true)
        let cfg = try JSONDecoder().decode(
            Qwen3MoEConfiguration.self,
            from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
        let model = Qwen3MoEModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 6))
        print("[qwen3moe-v3] loaded", flush: true)

        let caches: [KVCache] = model.newCache(parameters: GenerateParameters())
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })
        if let tri = caches.first as? TriAttentionKVCache {
            #expect(caches.count == tri.engine.nLayers)
        }

        // Small ctx so the suite stays light. Deterministic random-int
        // prompt within a safe sub-range. Qwen3 vocab is 151936 — accept
        // the full vocab on decoded outputs (special tokens are valid).
        let promptLen = 64
        let maxDecode = 8
        let promptIdMax: Int32 = 100000
        let vocabFull: Int32 = 151936
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
        print("[qwen3moe-v3] decode loop done", flush: true)
    }
}

// Local print-with-flush helper (matches Qwen2V3Tests / Gemma4V3Tests).
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
