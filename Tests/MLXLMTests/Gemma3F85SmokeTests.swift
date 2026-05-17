// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — Gemma 3 smoke tests.
//
// Validates that the new Gemma3TextModel fullyBatchedSparseDecode path:
//   1. compiles + links against BatchedRetrievalAttentionKVCache
//   2. produces logits of the expected shape ([B, 1, vocab])
//   3. doesn't NaN on a synthetic random-weights Gemma3TextModel
//   4. correctly dispatches per-layer (sliding vs global use different
//      RoPE bases but share head dims on Gemma 3 — the harness builds
//      uniformly-sized RA caches and marks sliding layers dense)
//
// A gated live smoke (RUN_GEMMA3_F85_SMOKE=1) loads gemma-3-4b-it-4bit
// and runs B=4 ctx=8K batched sparse decode against synthetic populated
// caches to confirm the end-to-end real-model path.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Gemma3 F-85 batched sparse decode — port", .serialized)
struct Gemma3F85SmokeTests {

    /// Synthetic Gemma 3 text config. 4 layers with
    /// `slidingWindowPattern=2` puts global layers at indices 1 and 3,
    /// exercising the sparse-vs-dense per-layer dispatch. Uniform
    /// `headDim=8 kvHeads=2` across all layers (matches Gemma 3 — only
    /// Gemma 4 has per-layer-type dim divergence).
    fileprivate static func makeSyntheticConfig() throws -> Gemma3TextConfiguration {
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

    /// Build per-layer batched RA caches with synthetic K/V populated.
    /// On Gemma 3 every layer has the same `(nKVH, dHead)` so we use a
    /// single shape. Sliding layers (`sparseEligibleLayers` membership
    /// false) get a `RetrievalAttentionConfig` with both `denseFirstN`
    /// and `denseLastN` set to `nLayers` so `isSparseEligible` returns
    /// false — they fall through to `cache.attention` (dense) at decode
    /// time. Global layers stay sparse.
    fileprivate static func makeBatchedRACaches(
        B: Int, T: Int, nKVH: Int, dHead: Int, nLayers: Int,
        sparseEligibleLayers: Set<Int>,
        dtype: MLX.DType = .float32
    ) -> [BatchedRetrievalAttentionKVCache] {
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

            // Per-layer config: dense-band span covers everything for
            // ineligible (sliding) layers; sparse-eligible (global)
            // layers get an open config (both bands at 0).
            var layerCfg = RetrievalAttentionConfig()
            if sparseEligibleLayers.contains(layerIdx) {
                layerCfg.denseFirstN = 0
                layerCfg.denseLastN = 0
            } else {
                layerCfg.denseFirstN = nLayers
                layerCfg.denseLastN = nLayers
            }

            let raCache = BatchedRetrievalAttentionKVCache(
                inner: cache, B: B, nKVHeads: nKVH, dHead: dHead,
                layerIdx: layerIdx, totalLayers: nLayers, raConfig: layerCfg)
            // Only push K into the index for eligible layers so sliding
            // layers don't pay the index-build cost.
            if sparseEligibleLayers.contains(layerIdx) {
                raCache.updateIndex(newKeys: kAll)
            }
            caches.append(raCache)
        }
        return caches
    }

    @Test("Gemma3 fullyBatchedSparseDecode runs end-to-end on synthetic config")
    func gemma3FullyBatchedSparseDecodeSynthetic() throws {
        let cfg = try Gemma3F85SmokeTests.makeSyntheticConfig()
        let model = Gemma3TextModel(cfg)

        // Synthetic config: slidingWindowPattern=2, hiddenLayers=4 →
        // global layers at indices 1 and 3, sliding at 0 and 2.
        let nLayers = 4
        let globalLayers: Set<Int> = [1, 3]
        let nKVH = 2
        let dHead = 8

        let B = 2
        let T = 64
        let raCaches = Gemma3F85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers,
            sparseEligibleLayers: globalLayers)

        let tokens = (0 ..< B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)

        let V = model.vocabularySize
        #expect(logits.shape == [B, 1, V],
            "expected logits [B=\(B), 1, V=\(V)], got \(logits.shape)")

        let flat = logits.reshaped(B * V).asArray(Float.self)
        let anyNaN = flat.contains { !$0.isFinite }
        #expect(!anyNaN, "fullyBatchedSparseDecode produced NaN/Inf logits")
    }

    /// Gated real-model smoke. Loads gemma-3-4b-it-4bit (a VLM
    /// checkpoint whose text_config feeds Gemma3TextConfiguration via
    /// the dual decoder), builds synthetic populated per-layer batched
    /// RA caches at ctx=8K B=4, runs ONE fullyBatchedSparseDecode step.
    /// Asserts shape + finite logits. Cache K/V is randomly populated
    /// (not real prefill state) so this only validates kernel-dispatch
    /// + shape contract, not generation quality.
    ///   RUN_GEMMA3_F85_SMOKE=1 swift test --filter gemma3F85LiveSmoke
    @Test func gemma3F85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_GEMMA3_F85_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/gemma-3-4b-it-4bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        print("[gemma3-f85] loading...", flush: true)
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(
            Gemma3TextConfiguration.self, from: configData)
        let model = Gemma3TextModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[gemma3-f85] loaded", flush: true)

        // Derive nLayers + sliding pattern + vocab from the canonical
        // config via the public surface — Gemma 3 uniform shape lets us
        // reuse model.config.kvHeads / .headDim for every layer.
        let nLayers = model.config.hiddenLayers
        let pattern = model.config.slidingWindowPattern
        // Global layers are those at index `(i % pattern == pattern - 1)`.
        let globalLayers: Set<Int> = Set(
            (0 ..< nLayers).filter { $0 % pattern == pattern - 1 })
        let nKVH = model.config.kvHeads
        let dHead = model.config.headDim
        let vocabSize = model.config.vocabularySize

        // B=4 ctx=8K matches the Llama / Gemma 4 F-85 smoke spec.
        let B = 4
        let envT = ProcessInfo.processInfo.environment["GEMMA3_F85_T"]
            .flatMap(Int.init) ?? (8 * 1024)
        let T = envT

        print("[gemma3-f85] B=\(B) T=\(T) nLayers=\(nLayers) "
              + "globalLayers=\(globalLayers.count) "
              + "nKVH=\(nKVH) D=\(dHead) pattern=\(pattern)",
              flush: true)

        let raCaches = Gemma3F85SmokeTests.makeBatchedRACaches(
            B: B, T: T, nKVH: nKVH, dHead: dHead, nLayers: nLayers,
            sparseEligibleLayers: globalLayers,
            dtype: .float16)

        let tokens = (0 ..< B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[gemma3-f85] one batched sparse decode step: "
              + "\(Int((t1 - t0) * 1000))ms (B=\(B), aggregate)",
              flush: true)

        #expect(logits.shape == [B, 1, vocabSize])
        #expect(model.vocabularySize == vocabSize)

        let lastLogits = logits.reshaped(B, -1)
        let sample = lastLogits.asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[gemma3-f85] smoke done", flush: true)
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
