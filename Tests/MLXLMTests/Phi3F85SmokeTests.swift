// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — Phi3 smoke tests.
//
// Validates that the new Phi3 fullyBatchedSparseDecode path:
//   1. compiles + links against BatchedRetrievalAttentionKVCache
//   2. produces logits of the expected shape ([B, 1, vocab])
//   3. doesn't NaN on a synthetic random-weights Phi3Model
//
// A gated live smoke (RUN_PHI3_F85_SMOKE=1) loads Phi-3-mini-4k-4bit
// and runs a B=4 ctx=8K batched sparse decode step against synthetic
// random K/V to confirm the end-to-end real-model path.
//
// Phi3 specifics: fused qkv_proj (split into Q/K/V along last axis
// after a single batched matmul) + partial-RoPE (rope applies to
// `partialRotaryFactor * headDim` channels). Both are handled inside
// `Phi3Attention.fullyBatchedSparseForward` and rely on the existing
// RoPE / SuScaledRoPE classes to honor `partialRotaryFactor`.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Phi3 F-85 batched sparse decode — port", .serialized)
struct Phi3F85SmokeTests {

    fileprivate static func makeSyntheticPhi3Config() throws -> Phi3Configuration {
        // Small synthetic config so the test stays light. 2 layers,
        // hiddenSize 96, 8 heads (head_dim 12), 2 KV heads (GQA 4:1).
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

    /// Build per-layer batched RA caches with synthetic K/V already
    /// populated. Mirrors the harness pattern in LlamaF85SmokeTests.
    fileprivate static func makeBatchedRACaches(
        B: Int, T: Int, nKVH: Int, dHead: Int, nLayers: Int,
        dtype: MLX.DType = .float32
    ) -> [BatchedRetrievalAttentionKVCache] {
        var cfg = RetrievalAttentionConfig()
        cfg.denseFirstN = 0
        cfg.denseLastN = 0
        var caches = [BatchedRetrievalAttentionKVCache]()
        caches.reserveCapacity(nLayers)
        for layerIdx in 0..<nLayers {
            let cache = BatchedKVCache(
                maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
                dtype: dtype)
            for _ in 0..<B { _ = cache.addRequest() }
            // Different K per slot so the selector index has signal.
            var ks = [MLXArray]()
            for s in 0..<B {
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
            for i in 0..<B { cache.offsets[i] = T }

            let raCache = BatchedRetrievalAttentionKVCache(
                inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
                layerIdx: layerIdx, totalLayers: nLayers, raConfig: cfg)
            raCache.updateIndex(newKeys: kAll)
            caches.append(raCache)
        }
        return caches
    }

    @Test("Phi3 fullyBatchedSparseDecode runs end-to-end on synthetic config")
    func phi3FullyBatchedSparseDecodeSynthetic() throws {
        let cfg = try Phi3F85SmokeTests.makeSyntheticPhi3Config()
        let model = Phi3Model(cfg)

        // Build batched RA caches. Synthetic test on .float32 so we
        // don't trip any bf16 codepath.
        let B = 2
        let T = 256
        let nKVH = 2
        let dHead = 12   // hidden 96 / heads 8 = 12
        let nLayers = 2
        let raCaches = Phi3F85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers)

        // Run one batched sparse decode step against random input tokens.
        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)

        // Shape: [B, 1, vocab] — vocab from Phi3Model.vocabularySize.
        // Synthetic config sets vocab=128.
        let V = model.vocabularySize
        #expect(logits.shape == [B, 1, V],
            "expected logits [B=\(B), 1, V=\(V)], got \(logits.shape)")

        // Finite check — random-weight Phi3 produces finite output.
        let flat = logits.reshaped(B * V).asArray(Float.self)
        let anyNaN = flat.contains { !$0.isFinite }
        #expect(!anyNaN, "fullyBatchedSparseDecode produced NaN/Inf logits")
    }

    /// Gated real-model smoke. Loads Phi-3-mini-4k-instruct-4bit, builds
    /// synthetic populated batched RA caches at ctx=8K B=4, runs ONE
    /// fullyBatchedSparseDecode step. Asserts shape + finite logits.
    /// Cache K/V is randomly populated (not real prefill state) so this
    /// only validates the kernel-dispatch + shape contract, not
    /// generation quality.
    ///   RUN_PHI3_F85_SMOKE=1 swift test --filter phi3F85LiveSmoke
    @Test func phi3F85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_PHI3_F85_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/Phi-3-mini-4k-instruct-4bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        print("[phi3-f85] loading...", flush: true)
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(Phi3Configuration.self, from: configData)
        let model = Phi3Model(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[phi3-f85] loaded", flush: true)

        // Read shape fields from the raw JSON since Phi3Configuration's
        // fields are internal to MLXLLM and the tests live in their own
        // module. Phi3 has no `head_dim` field — derive from
        // hidden_size / num_attention_heads, matching
        // Phi3Attention.init.
        struct Phi3Shape: Codable {
            let hidden_size: Int
            let num_hidden_layers: Int
            let num_attention_heads: Int
            let num_key_value_heads: Int
            let vocab_size: Int
        }
        let shape = try JSONDecoder().decode(Phi3Shape.self, from: configData)
        let resolvedHeadDim = shape.hidden_size / shape.num_attention_heads

        // B=4 ctx=8K matches the Llama F-85 smoke spec.
        let B = 4
        let T = 8 * 1024
        let nLayers = shape.num_hidden_layers
        let nKVH = shape.num_key_value_heads
        let dHead = resolvedHeadDim

        print("[phi3-f85] B=\(B) T=\(T) nLayers=\(nLayers) nKVH=\(nKVH) D=\(dHead)",
              flush: true)

        // Synthetic populated batched RA caches. fp16 to match the
        // weight dtype expected by the matmul kernels on a 4-bit model.
        let raCaches = Phi3F85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers,
            dtype: .float16)

        // One decode step. Use a list of Int32 tokens within the vocab.
        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[phi3-f85] one batched sparse decode step: \(Int((t1 - t0) * 1000))ms "
              + "(B=\(B), aggregate)", flush: true)

        #expect(logits.shape == [B, 1, shape.vocab_size])
        #expect(model.vocabularySize == shape.vocab_size)

        // Finite check on a sample of logits.
        let lastLogits = logits.reshaped(B, -1)
        let sample = lastLogits.asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[phi3-f85] smoke done", flush: true)
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
