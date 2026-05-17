// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// TriAttention V3 — Gemma 3 family integration smoke.
// Mirrors Qwen2V3Tests / Gemma4V3Tests but exercises Gemma 3's
// uniform-attention-shape factory branch documented in
// `Gemma3TextModel.newCache` (Libraries/MLXLLM/Models/Gemma3.swift).
//
// Architectural note (locked in by these tests):
// Gemma 3 interleaves sliding-window and global-attention layers via
// `slidingWindowPattern`, BUT — unlike Gemma 4 — uses a SINGLE
// `headDim` / `kvHeads` / `nHeads` triple across both attention types
// (see Gemma3.TextConfiguration in MLXLMCommon/Models/Gemma3.swift).
// Sliding and global layers only differ in their RoPE base frequency
// (`ropeLocalBaseFreq` vs `ropeTheta`). The V3 engine pins on the
// (nHeads, nKVHeads, headDim, ropeTheta) tuple, so one engine can
// serve all layers — we install it directly, matching the
// Qwen2 / Qwen3 / Llama pattern (no Gemma 4 style fall-through guard).
//
// The tests cover:
//   - V3 enabled + uniform synthetic config -> installs V3 caches
//   - V3 disabled (any config)              -> default sliding/global mix
//   - V3 enabled + `maxKVSize` set          -> default mix (V3 incompatible)
//
// A gated live smoke (RUN_GEMMA3_V3_SMOKE=1) loads the real
// gemma-3-4b-it-4bit checkpoint (text VLM repo) and confirms V3 install
// + a tiny prefill + decode loop runs without crashing.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

/// Process-wide lock that serializes the VLLM_TRIATT_ENABLED env-var
/// dance across V3 test suites. Swift Testing's `.serialized` trait is
/// per-suite; without a cross-suite lock the V3 suites (Qwen2, Qwen3,
/// Gemma4, Gemma3, ...) race on `setenv` / `unsetenv` and the install
/// test can sample a transient unsetenv from a sibling suite. Mirrors
/// the helper in Gemma4V3Tests.
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

@Suite("Gemma3 TriAttention V3 — port", .serialized)
struct Gemma3V3Tests {

    /// Synthetic Gemma 3 text config. Small enough to keep the test light;
    /// `headDim` / `kvHeads` / `nHeads` are uniform across layers (this is
    /// the Gemma 3 default — there is no per-layer-type override unlike
    /// Gemma 4). 4 layers with `slidingWindowPattern=2` puts one global
    /// layer at index 1 and one at index 3, exercising the interleave.
    fileprivate static func makeGemma3Config() throws -> Gemma3TextConfiguration {
        let json = """
            {
                "model_type": "gemma3_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "head_dim": 8,
                "rms_norm_eps": 0.000001,
                "vocab_size": 128,
                "num_key_value_heads": 2,
                "rope_theta": 1000000,
                "rope_local_base_freq": 10000,
                "rope_traditional": false,
                "query_pre_attn_scalar": 256,
                "sliding_window": 32,
                "sliding_window_pattern": 2,
                "max_position_embeddings": 32768
            }
            """
        return try JSONDecoder().decode(
            Gemma3TextConfiguration.self, from: json.data(using: .utf8)!)
    }

    @Test("Gemma3 factory installs V3 caches when env enabled")
    func gemma3FactoryInstallsV3() throws {
        // Cross-suite retry to defeat racy `unsetenv` from sibling V3
        // suites (Qwen2 / Qwen3 / Gemma4 — all serialize within their
        // own suite but not across). Matches Gemma4V3Tests pattern.
        let model = Gemma3TextModel(try Gemma3V3Tests.makeGemma3Config())
        var caches: [KVCache] = []
        var installed = false
        for _ in 0 ..< 8 {
            caches = withTriattEnv([("VLLM_TRIATT_ENABLED", "1")]) {
                model.newCache(parameters: nil)
            }
            installed = caches.allSatisfy { $0 is TriAttentionKVCache }
            if installed { break }
        }
        #expect(caches.count == 4)
        #expect(installed)
        if let tri = caches.first as? TriAttentionKVCache {
            #expect(tri.logicalOffset == 0)
            #expect(tri.engine.nLayers == 4)
            #expect(tri.engine.nHeads == 8)
            #expect(tri.engine.nKVHeads == 2)
            #expect(tri.engine.headDim == 8)
        }
    }

    @Test("Gemma3 factory uses default mix when V3 env disabled")
    func gemma3FactoryDefaultsWhenV3Disabled() throws {
        // Cross-suite race: sibling V3 suites (Qwen2 / Qwen3 / Gemma4 /
        // TriAttentionV3) `setenv("VLLM_TRIATT_ENABLED", "1", ...)` and
        // unset on defer — Swift Testing's `.serialized` is per-suite so
        // those calls can interleave with our `unsetenv`. Retry within
        // the lock until we observe the no-V3 outcome (or give up after
        // 8 tries). Same pattern Gemma4V3Tests uses for its install
        // assertion.
        let model = Gemma3TextModel(try Gemma3V3Tests.makeGemma3Config())
        var caches: [KVCache] = []
        var allNonV3 = false
        for _ in 0 ..< 8 {
            caches = withTriattEnv([("VLLM_TRIATT_ENABLED", nil)]) {
                model.newCache(parameters: nil)
            }
            allNonV3 = caches.allSatisfy { !($0 is TriAttentionKVCache) }
            if allNonV3 { break }
        }
        #expect(caches.count == 4)
        // Default path returns a sliding-window-aware mix; no V3.
        #expect(allNonV3)
    }

    @Test("Gemma3 factory falls through when maxKVSize is set")
    func gemma3FactoryRespectsMaxKVSize() throws {
        let model = Gemma3TextModel(try Gemma3V3Tests.makeGemma3Config())
        var params = GenerateParameters()
        params.maxKVSize = 512
        var caches: [KVCache] = []
        var allNonV3 = false
        // Retry to defeat cross-suite env-var race (see
        // gemma3FactoryDefaultsWhenV3Disabled).
        for _ in 0 ..< 8 {
            caches = withTriattEnv([("VLLM_TRIATT_ENABLED", "1")]) {
                model.newCache(parameters: params)
            }
            allNonV3 = caches.allSatisfy { !($0 is TriAttentionKVCache) }
            if allNonV3 { break }
        }
        // maxKVSize set → V3 incompatible (mirrors Qwen2/Qwen3
        // behavior); factory falls through to eviction-windowed.
        #expect(allNonV3)
    }

    /// Gated real-model smoke. Loads gemma-3-4b-it-4bit (a VLM
    /// checkpoint whose text_config is exposed via Gemma3.TextConfiguration's
    /// dual decoder) and runs a tiny prefill + decode loop with V3 enabled.
    /// Asserts:
    ///   - factory DOES install TriAttentionKVCache (uniform shape — no
    ///     Gemma 4 style skip).
    ///   - prefill + decode still run without crashing.
    ///   - sampled tokens stay in-vocab.
    /// Run with: RUN_GEMMA3_V3_SMOKE=1 swift test --filter gemma3V3LiveSmoke
    @Test func gemma3V3LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_GEMMA3_V3_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/gemma-3-4b-it-4bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

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

        print("[gemma3-v3] loading...", flush: true)
        let cfg = try JSONDecoder().decode(
            Gemma3TextConfiguration.self,
            from: Data(contentsOf: modelPath.appendingPathComponent("config.json")))
        let model = Gemma3TextModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[gemma3-v3] loaded", flush: true)

        let caches: [KVCache] = model.newCache(parameters: GenerateParameters())
        // Uniform shape → V3 must install on every layer.
        #expect(caches.allSatisfy { $0 is TriAttentionKVCache })

        // Small ctx so the suite stays light. Deterministic random-int
        // prompt. Gemma 3 vocab is 262144 / 262208; draw from a safe
        // sub-range to keep the prefill cheap.
        let promptLen = 32
        let maxDecode = 4
        let promptIdMax: Int32 = 100000
        let vocabFull: Int32 = Int32(model.vocabularySize)
        MLXRandom.seed(0xA3C3E9D)
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
        print("[gemma3-v3] decode loop done", flush: true)
    }
}

// Local print-with-flush helper (matches Gemma4V3Tests).
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
