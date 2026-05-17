// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — Qwen3 MoE smoke tests.
//
// Validates that the new Qwen3MoEModel fullyBatchedSparseDecode path:
//   1. compiles + links against BatchedRetrievalAttentionKVCache
//   2. produces logits of the expected shape ([B, 1, vocab])
//   3. doesn't NaN on a synthetic random-weights Qwen3MoEModel
//
// Qwen3MoE shares Qwen3's dense-attention shape (q_norm/k_norm + RoPE,
// single attentionHeads/kvHeads/headDim across all layers) so the F-85
// kernel-dispatch harness is identical to LlamaF85SmokeTests — no
// per-layer-type sizing dance like Gemma4. The MoE FFN is orthogonal:
// sparse only modifies the attention KV gather, expert routing runs
// unchanged.
//
// A gated live smoke (RUN_QWEN3MOE_F85_SMOKE=1) loads
// Qwen3-Coder-30B-A3B-Instruct-MLX-6bit and runs a B=4 ctx=8K batched
// sparse decode step against synthetic random K/V to confirm the
// end-to-end real-model path.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Qwen3MoE F-85 batched sparse decode — port", .serialized)
struct Qwen3MoEF85SmokeTests {

    fileprivate static func makeSyntheticQwen3MoEConfig() throws -> Qwen3MoEConfiguration {
        // Small synthetic config so the test stays light. 2 layers,
        // hiddenSize 64, 8 heads, 2 KV heads (GQA 4:1), head_dim 8.
        // MoE: 4 experts top-2, every layer is MoE (decoder_sparse_step=1).
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

    @Test("Qwen3MoE fullyBatchedSparseDecode runs end-to-end on synthetic config")
    func qwen3MoEFullyBatchedSparseDecodeSynthetic() throws {
        let cfg = try Qwen3MoEF85SmokeTests.makeSyntheticQwen3MoEConfig()
        let model = Qwen3MoEModel(cfg)

        // Build batched RA caches. Synthetic test on .float32 so we
        // don't trip the bf16 path. Synthetic config: 2 layers, 2 KV
        // heads, head_dim 8 — matches the config field values.
        let B = 2
        let T = 256
        let nKVH = 2
        let dHead = 8
        let nLayers = 2
        let raCaches = Qwen3MoEF85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers)

        // Run one batched sparse decode step against random input tokens.
        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)

        // Shape: [B, 1, vocab] — vocab from Qwen3MoEModel.vocabularySize.
        // Synthetic config sets vocab=128.
        let V = model.vocabularySize
        #expect(logits.shape == [B, 1, V],
            "expected logits [B=\(B), 1, V=\(V)], got \(logits.shape)")

        // Finite check — random-weight Qwen3MoE produces finite output.
        let flat = logits.reshaped(B * V).asArray(Float.self)
        let anyNaN = flat.contains { !$0.isFinite }
        #expect(!anyNaN, "fullyBatchedSparseDecode produced NaN/Inf logits")
    }

    /// Gated real-model smoke. Loads Qwen3-Coder-30B-A3B-Instruct-MLX-6bit
    /// (model_type=qwen3_moe), builds synthetic populated batched RA
    /// caches at ctx=8K B=4, runs ONE fullyBatchedSparseDecode step.
    /// Asserts shape + finite logits. Cache K/V is randomly populated
    /// (not real prefill state) so this only validates kernel-dispatch +
    /// shape contract, not generation quality.
    ///   RUN_QWEN3MOE_F85_SMOKE=1 swift test --filter qwen3MoEF85LiveSmoke
    @Test func qwen3MoEF85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_QWEN3MOE_F85_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/Qwen3-Coder-30B-A3B-Instruct-MLX-6bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        print("[qwen3moe-f85] loading...", flush: true)
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: configData)
        let model = Qwen3MoEModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 6))
        print("[qwen3moe-f85] loaded", flush: true)

        // Read shape fields from the raw JSON since Qwen3MoEConfiguration
        // fields are internal to MLXLLM and the tests live in their own
        // module. Matches the Llama/Qwen2 smoke test pattern.
        struct Qwen3MoEShape: Codable {
            let hidden_size: Int
            let num_hidden_layers: Int
            let num_attention_heads: Int
            let num_key_value_heads: Int
            let head_dim: Int
            let vocab_size: Int
        }
        let shape = try JSONDecoder().decode(Qwen3MoEShape.self, from: configData)

        // B=4 ctx=8K matches the Qwen3/Llama F-85 smoke spec.
        let B = 4
        let envT = ProcessInfo.processInfo.environment["QWEN3MOE_F85_T"]
            .flatMap(Int.init) ?? (8 * 1024)
        let T = envT
        let nLayers = shape.num_hidden_layers
        let nKVH = shape.num_key_value_heads
        let dHead = shape.head_dim

        print("[qwen3moe-f85] B=\(B) T=\(T) nLayers=\(nLayers) nKVH=\(nKVH) D=\(dHead)",
              flush: true)

        // Synthetic populated batched RA caches. fp16 to match the
        // weight dtype expected by the matmul kernels on a 6-bit model.
        let raCaches = Qwen3MoEF85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers,
            dtype: .float16)

        // One decode step.
        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[qwen3moe-f85] one batched sparse decode step: \(Int((t1 - t0) * 1000))ms "
              + "(B=\(B), aggregate)", flush: true)

        #expect(logits.shape == [B, 1, shape.vocab_size])
        #expect(model.vocabularySize == shape.vocab_size)

        // Finite check on a sample of logits.
        let lastLogits = logits.reshaped(B, -1)
        let sample = lastLogits.asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[qwen3moe-f85] smoke done", flush: true)
    }
}

// Local print-with-flush helper (matches LlamaF85SmokeTests).
private func print(_ s: String, flush: Bool) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    Swift.print(s)
    if flush {
        try? FileHandle.standardOutput.synchronize()
    }
}
