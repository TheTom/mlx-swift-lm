// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — Qwen3.5 / Qwen3.6 hybrid smoke tests.
//
// Validates that the new Qwen35Model fullyBatchedSparseDecode path:
//   1. compiles + links against BatchedHybridSparseLLM / BatchedHybridCache
//      with `.sparseAttention(BatchedRetrievalAttentionKVCache)` slots.
//   2. produces logits of the expected shape ([B, 1, vocab]).
//   3. doesn't NaN on a synthetic random-weights Qwen35TextModel.
//   4. correctly per-layer dispatches: GDN (linear) layers use the dense
//      GDN path, attention layers route through sparseAttend.
//
// Qwen3.5 is a hybrid family (GDN/Mamba + attention), so unlike
// Qwen3MoE / Llama / Phi3 etc. this exercises the new
// `BatchedHybridSparseLLM` protocol path through `BatchedHybridCache`
// with mixed `.gdn(_)` and `.sparseAttention(_)` slots. Single port
// covers both dense Qwen3.5 and Qwen3.6 MoE — both use Qwen35TextModel.
//
// A gated live smoke (RUN_QWEN35_F85_SMOKE=1) loads ~/models/Qwen3.5-2B-4bit
// (or any Qwen3.5-* present) and runs a B=4 ctx=8K batched sparse decode
// step against synthetic random K/V to confirm end-to-end real-model
// kernel-dispatch.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Qwen3.5 hybrid F-85 batched sparse decode — port", .serialized)
struct Qwen35F85SmokeTests {

    /// Minimal hybrid config. 4 layers with `full_attention_interval=4` →
    /// indices 0,1,2 = GDN; index 3 = attention. So the test exercises
    /// per-layer-type dispatch (3 GDN + 1 sparseAttention) inside
    /// `Qwen35DecoderLayer.fullyBatchedSparseForward`.
    fileprivate static func makeSyntheticConfig() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "linear_num_value_heads": 16,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rms_norm_eps": 1e-6,
                "vocab_size": 256,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "max_position_embeddings": 512,
                "tie_word_embeddings": true,
                "attention_bias": false,
                "head_dim": 8,
                "full_attention_interval": 4
            }
            """.data(using: .utf8)!
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: json)
    }

    /// Seed the BatchedKVCache backing each attention layer with synthetic
    /// random K/V so sparseAttend has data to score against. Each slot gets
    /// a different K seed so the selector index has signal across slots.
    fileprivate static func seedAttentionSlots(
        cache: BatchedHybridCache, T: Int, dtype: DType = .float32
    ) {
        let B = cache.active
        for (li, layer) in cache.layers.enumerated() {
            guard case .sparseAttention(let raCache) = layer else { continue }
            let inner = raCache.inner
            let nKVH = inner.kvHeads
            let dHead = inner.headDim

            var ks = [MLXArray]()
            ks.reserveCapacity(B)
            for s in 0..<B {
                ks.append(MLXRandom.normal(
                    [1, nKVH, T, dHead],
                    key: MLXRandom.key(UInt64(100 + li * 10 + s))
                ).asType(dtype))
            }
            let kAll = concatenated(ks, axis: 0)
            let vAll = MLXRandom.normal(
                [B, nKVH, T, dHead],
                key: MLXRandom.key(UInt64(200 + li))
            ).asType(dtype)
            inner.keys[..<B, 0..., ..<T, 0...] = kAll.asType(inner.keys.dtype)
            inner.values[..<B, 0..., ..<T, 0...] = vAll.asType(inner.values.dtype)
            for i in 0..<B { inner.offsets[i] = T }

            raCache.updateIndex(newKeys: kAll)
        }
    }

    /// Sanity: `newBatchedHybridSparseCache` builds the right mix of
    /// `.sparseAttention(_)` + `.gdn(_)` slots and exposes the
    /// `BatchedHybridSparseLLM` surface.
    @Test
    func newBatchedHybridSparseCacheLayout() throws {
        let cfg = try Self.makeSyntheticConfig()
        let model = Qwen35TextModel(cfg)
        let raConfig = RetrievalAttentionConfig()
        let cache = model.newBatchedHybridSparseCache(
            maxBatch: 2, parameters: nil, raConfig: raConfig)

        #expect(cache.layers.count == 4)
        for (i, layer) in cache.layers.enumerated() {
            // Pattern: isLinear = (layerIdx + 1) % fullAttentionInterval != 0
            // With interval=4: indices 0,1,2 GDN, index 3 attention.
            let expectAttention = ((i + 1) % 4 == 0)
            switch layer {
            case .gdn:
                #expect(!expectAttention, "layer \(i) GDN but expected attention")
            case .sparseAttention:
                #expect(expectAttention,
                        "layer \(i) sparseAttention but expected GDN")
            case .attention:
                Issue.record(
                    "layer \(i): newBatchedHybridSparseCache returned .attention; expected .sparseAttention")
            }
        }
    }

    /// End-to-end: ONE batched sparse decode step on a synthetic random-weights
    /// Qwen35TextModel against a populated cache. Asserts shape + finite logits.
    @Test
    func qwen35FullyBatchedSparseDecodeSynthetic() throws {
        // F-85 kernel selection from the env (matches the rest of the F-85
        // smoke suite default). f73 = batched mask + MLXFast SDPA.
        setenv("VSM_SPARSE_BATCHED_KERNEL", "f73", 1)
        defer { unsetenv("VSM_SPARSE_BATCHED_KERNEL") }

        let cfg = try Self.makeSyntheticConfig()
        let model = Qwen35TextModel(cfg)

        // The Metal GDN kernel only has bf16/fp16 instantiations at the
        // Dk=128 / Dv=128 / Hk=16 / Hv=16 shape we configured. Cast all
        // params to bf16 to find the kernel (matches the existing
        // Qwen35BatchedHybridCacheTests pattern).
        eval(model)
        let bf16Params = model.parameters().mapValues { (v: MLXArray) in v.asType(.bfloat16) }
        try model.update(parameters: bf16Params, verify: [.noUnusedKeys])
        eval(model)

        // Tiny RA config (no denseFirstN/denseLastN bands so even a 4-layer
        // synthetic model has the single attention layer flagged as sparse).
        var raConfig = RetrievalAttentionConfig()
        raConfig.denseFirstN = 0
        raConfig.denseLastN = 0

        let B = 2
        let T = 256
        let cache = model.newBatchedHybridSparseCache(
            maxBatch: B, parameters: nil, raConfig: raConfig)
        for _ in 0..<B { cache.addSlot() }

        // Populate the (one) sparse attention layer's KV slots so
        // sparseAttend has signal.
        Self.seedAttentionSlots(cache: cache, T: T, dtype: .float32)

        // One decode step.
        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(inputs, caches: cache)
        eval(logits)

        let V = model.vocabularySize
        #expect(logits.shape == [B, 1, V],
            "expected logits [B=\(B), 1, V=\(V)], got \(logits.shape)")

        // Finite check.
        let flat = logits.reshaped(B * V).asArray(Float.self)
        let nonFinite = flat.filter { !$0.isFinite }.count
        #expect(nonFinite == 0,
                "fullyBatchedSparseDecode produced \(nonFinite) non-finite logits")
    }

    /// Live real-model smoke. Gated on RUN_QWEN35_F85_SMOKE=1. Loads any
    /// ~/models/Qwen3.5-* found via shell glob and runs ONE
    /// fullyBatchedSparseDecode step at B=4 against synthetic K/V.
    /// Verifies shape contract + no NaN against real weights; doesn't test
    /// generation quality.
    @Test
    func qwen35F85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_QWEN35_F85_SMOKE"] == "1" else {
            return
        }
        setenv("VSM_SPARSE_BATCHED_KERNEL", "f73", 1)
        defer { unsetenv("VSM_SPARSE_BATCHED_KERNEL") }

        let modelsDir = URL(fileURLWithPath: "\(NSHomeDirectory())/models")
        // Pick the first Qwen3.5-* checkpoint that has a config.json.
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: modelsDir, includingPropertiesForKeys: nil) else {
            Issue.record("models dir not present: \(modelsDir.path)")
            return
        }
        let candidates = entries.filter {
            $0.lastPathComponent.hasPrefix("Qwen3.5-")
                && FileManager.default.fileExists(
                    atPath: $0.appendingPathComponent("config.json").path)
        }
        guard let modelPath = candidates.first else {
            Issue.record("no Qwen3.5-* checkpoint present under ~/models")
            return
        }

        print("[qwen35-f85] loading \(modelPath.lastPathComponent)", flush: true)
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(Qwen35Configuration.self, from: configData)
        let model = Qwen35Model(cfg)
        // Most local Qwen3.5 checkpoints are 4-bit groupSize 64; ConfigI
        // / UD variants ship per-layer-quantization. The smoke test is
        // shape-only, so a 4-bit best-guess is fine — if it doesn't match
        // the checkpoint, loadWeights will throw and the test will be
        // recorded as a fail (the kernel-dispatch path was never reached).
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[qwen35-f85] loaded", flush: true)

        // Read shape fields from the JSON to size synthetic caches —
        // Qwen35TextConfiguration / Qwen35Configuration fields are internal
        // to MLXLLM and tests live in a different module.
        struct Qwen35Shape: Codable {
            let hidden_size: Int?
            let num_hidden_layers: Int?
            let num_attention_heads: Int?
            let num_key_value_heads: Int?
            let head_dim: Int?
            let vocab_size: Int?
            let full_attention_interval: Int?
        }
        let nestedShape: Qwen35Shape
        if let nested = try? JSONDecoder().decode(
            [String: Qwen35Shape].self, from: configData)["text_config"] {
            nestedShape = nested
        } else {
            nestedShape = try JSONDecoder().decode(Qwen35Shape.self, from: configData)
        }
        let hidden = nestedShape.hidden_size ?? 2048
        let nLayers = nestedShape.num_hidden_layers ?? 24
        let nAH = nestedShape.num_attention_heads ?? 32
        let nKVH = nestedShape.num_key_value_heads ?? 2
        let dHead = nestedShape.head_dim ?? (hidden / nAH)
        let vocab = nestedShape.vocab_size ?? 151936
        let interval = nestedShape.full_attention_interval ?? 4

        let B = 4
        let envT = ProcessInfo.processInfo.environment["QWEN35_F85_T"]
            .flatMap(Int.init) ?? (8 * 1024)
        let T = envT
        print("[qwen35-f85] B=\(B) T=\(T) nLayers=\(nLayers) nKVH=\(nKVH) D=\(dHead) interval=\(interval)",
              flush: true)

        var raConfig = RetrievalAttentionConfig()
        raConfig.denseFirstN = 0
        raConfig.denseLastN = 0

        let params = GenerateParameters(maxKVSize: T + 64)
        let cache = model.newBatchedHybridSparseCache(
            maxBatch: B, parameters: params, raConfig: raConfig)
        for _ in 0..<B { cache.addSlot() }

        // Seed sparse attention slots with synthetic populated K/V at T.
        // Use fp16 for the keys/values to match the 4-bit weight matmul
        // accumulator dtype.
        Self.seedAttentionSlots(cache: cache, T: T, dtype: .float16)

        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, caches: cache)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[qwen35-f85] one batched sparse decode step: \(Int((t1 - t0) * 1000))ms "
              + "(B=\(B), aggregate)", flush: true)

        #expect(logits.shape == [B, 1, vocab],
                "expected [B=\(B), 1, V=\(vocab)], got \(logits.shape)")

        let lastLogits = logits.reshaped(B, -1)
        let sample = lastLogits.asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[qwen35-f85] smoke done", flush: true)
    }
}

// Local print-with-flush helper (matches the rest of the F-85 smoke suite).
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
