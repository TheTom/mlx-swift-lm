// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — Mistral 3 / Ministral 3 smoke tests.
//
// Validates that the new Mistral3 fullyBatchedSparseDecode path:
//   1. compiles + links against BatchedRetrievalAttentionKVCache
//   2. produces logits of the expected shape ([B, 1, vocab])
//   3. doesn't NaN on a synthetic random-weights Mistral3TextModel
//
// A gated live smoke (RUN_MISTRAL3_F85_SMOKE=1) loads a real Ministral 3
// checkpoint and runs a B=4 ctx=8K batched sparse decode step against
// synthetic random K/V to confirm the end-to-end real-model path.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Mistral3 F-85 batched sparse decode — port", .serialized)
struct Mistral3F85SmokeTests {

    fileprivate static func makeSyntheticMistral3Config() throws -> Mistral3TextConfiguration {
        // Small synthetic config so the test stays light. 2 layers,
        // hiddenSize 64, 8 heads, 2 KV heads (GQA 4:1), head_dim 8.
        // All layers full_attention (no sliding) so every layer routes
        // through the sparse path under default RetrievalAttentionConfig.
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

    /// Build per-layer batched RA caches with synthetic K/V already
    /// populated. Mirrors the harness pattern in F85V2BatchedMaskTests
    /// and LlamaF85SmokeTests.
    fileprivate static func makeBatchedRACaches(
        B: Int, T: Int, nKVH: Int, dHead: Int, nLayers: Int,
        dtype: MLX.DType = .float32
    ) -> [BatchedRetrievalAttentionKVCache] {
        var cfg = RetrievalAttentionConfig()
        cfg.denseFirstN = 0
        cfg.denseLastN = 0
        var caches = [BatchedRetrievalAttentionKVCache]()
        caches.reserveCapacity(nLayers)
        for layerIdx in 0 ..< nLayers {
            let cache = BatchedKVCache(
                maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
                dtype: dtype)
            for _ in 0 ..< B { _ = cache.addRequest() }
            // Different K per slot so the selector index has signal.
            var ks = [MLXArray]()
            for s in 0 ..< B {
                ks.append(MLXRandom.normal(
                    [1, nKVH, T, dHead],
                    key: MLXRandom.key(UInt64(100 + layerIdx * 10 + s))
                ).asType(dtype))
            }
            let kAll = concatenated(ks, axis: 0)
            let vAll = MLXRandom.normal(
                [B, nKVH, T, dHead],
                key: MLXRandom.key(UInt64(200 + layerIdx))
            ).asType(dtype)
            cache.keys[..<B, 0..., ..<T, 0...] = kAll
            cache.values[..<B, 0..., ..<T, 0...] = vAll
            for i in 0 ..< B { cache.offsets[i] = T }

            let raCache = BatchedRetrievalAttentionKVCache(
                inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
                layerIdx: layerIdx, totalLayers: nLayers, raConfig: cfg)
            raCache.updateIndex(newKeys: kAll)
            caches.append(raCache)
        }
        return caches
    }

    @Test("Mistral3 fullyBatchedSparseDecode runs end-to-end on synthetic config")
    func mistral3FullyBatchedSparseDecodeSynthetic() throws {
        let cfg = try Mistral3F85SmokeTests.makeSyntheticMistral3Config()
        let model = Mistral3TextModel(cfg)

        // Build batched RA caches. Synthetic test on .float32 so we
        // don't trip a bf16 path.
        let B = 2
        let T = 256
        let nKVH = 2
        let dHead = 8
        let nLayers = 2
        let raCaches = Mistral3F85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers)

        // Run one batched sparse decode step against random input tokens.
        let tokens = (0 ..< B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)

        // Shape: [B, 1, vocab]. Synthetic config sets vocab=128.
        let V = model.vocabularySize
        #expect(logits.shape == [B, 1, V],
            "expected logits [B=\(B), 1, V=\(V)], got \(logits.shape)")

        // Finite check — random-weight Mistral3 produces finite output.
        let flat = logits.reshaped(B * V).asArray(Float.self)
        let anyNaN = flat.contains { !$0.isFinite }
        #expect(!anyNaN, "fullyBatchedSparseDecode produced NaN/Inf logits")
    }

    /// Gated real-model smoke. Loads a Ministral 3 checkpoint, builds
    /// synthetic populated batched RA caches at ctx=8K B=4, runs ONE
    /// fullyBatchedSparseDecode step. Asserts shape + finite logits.
    /// Cache K/V is randomly populated (not real prefill state) so this
    /// only validates the kernel-dispatch + shape contract, not
    /// generation quality.
    ///   RUN_MISTRAL3_F85_SMOKE=1 swift test --filter mistral3F85LiveSmoke
    @Test func mistral3F85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_MISTRAL3_F85_SMOKE"] == "1" else {
            return
        }

        let modelDirEnv = ProcessInfo.processInfo.environment["MISTRAL3_MODEL_DIR"]
            ?? "\(NSHomeDirectory())/models/Ministral-8B-Instruct-2410-4bit"
        let modelPath = URL(fileURLWithPath: modelDirEnv)
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        print("[mistral3-f85] loading...", flush: true)
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(Mistral3TextConfiguration.self, from: configData)
        let model = Mistral3TextModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[mistral3-f85] loaded", flush: true)

        // Read shape fields from the raw JSON since
        // Mistral3TextConfiguration's `headDimensions` (head_dim) is
        // optional. Mirrors the Llama F-85 smoke pattern.
        struct M3Shape: Codable {
            let hidden_size: Int
            let num_hidden_layers: Int
            let num_attention_heads: Int
            let num_key_value_heads: Int?
            let head_dim: Int?
            let vocab_size: Int
        }
        let shape = try JSONDecoder().decode(M3Shape.self, from: configData)
        let resolvedKVH = shape.num_key_value_heads ?? shape.num_attention_heads
        let resolvedHeadDim = shape.head_dim ?? (shape.hidden_size / shape.num_attention_heads)

        // B=4 ctx=8K matches the Llama F-85 smoke spec.
        let B = 4
        let T = 8 * 1024
        let nLayers = shape.num_hidden_layers
        let nKVH = resolvedKVH
        let dHead = resolvedHeadDim

        print("[mistral3-f85] B=\(B) T=\(T) nLayers=\(nLayers) nKVH=\(nKVH) D=\(dHead)",
              flush: true)

        // Synthetic populated batched RA caches. fp16 to match the
        // weight dtype expected by the matmul kernels on a 4-bit model.
        let raCaches = Mistral3F85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers,
            dtype: .float16)

        // One decode step.
        let tokens = (0 ..< B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[mistral3-f85] one batched sparse decode step: \(Int((t1 - t0) * 1000))ms "
              + "(B=\(B), aggregate)", flush: true)

        #expect(logits.shape == [B, 1, shape.vocab_size])
        #expect(model.vocabularySize == shape.vocab_size)

        // Finite check on a sample of logits.
        let lastLogits = logits.reshaped(B, -1)
        let sample = lastLogits.asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[mistral3-f85] smoke done", flush: true)
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
