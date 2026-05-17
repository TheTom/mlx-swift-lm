// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — Gemma 4 smoke tests.
//
// Validates that the new Gemma4TextModel fullyBatchedSparseDecode path:
//   1. compiles + links against BatchedRetrievalAttentionKVCache
//   2. produces logits of the expected shape ([B, 1, vocab])
//   3. doesn't NaN on a synthetic random-weights Gemma4TextModel
//   4. correctly dispatches per-layer (sliding vs global use different
//      kvHeads / headDim; the harness builds per-layer-sized RA caches)
//
// A gated live smoke (RUN_GEMMA4_F85_SMOKE=1) loads gemma-4-26b-a4b-4bit
// and runs B=4 ctx=8K batched sparse decode against synthetic populated
// per-layer caches to confirm the end-to-end real-model path.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("Gemma4 F-85 batched sparse decode — port", .serialized)
struct Gemma4F85SmokeTests {

    /// Synthetic Gemma 4 config. Matches the real architectural mix
    /// (sliding `headDim=8 kvHeads=4`, global `headDim=16 kvHeads=2`)
    /// in miniature so the per-layer dispatch in
    /// `Gemma4ModelInner.fullyBatchedSparseForward` is exercised. PLE is
    /// disabled (`hiddenSizePerLayerInput=0`) to keep the test light;
    /// the sparse path doesn't touch PLE bookkeeping anyway.
    fileprivate static func makeSyntheticHeteroConfig() throws -> Gemma4TextConfiguration {
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

    /// Build per-layer batched RA caches with synthetic K/V populated.
    /// `perLayerShape[i]` = `(nKVH, headDim)` so heterogeneous layers
    /// get correctly sized caches. Sliding layers are marked sparse-
    /// ineligible via the raConfig denseFirstN/denseLastN bands so they
    /// fall through to dense `cache.attention` at decode time.
    fileprivate static func makeBatchedRACaches(
        B: Int, T: Int,
        perLayerShape: [(nKVH: Int, dHead: Int)],
        sparseEligibleLayers: Set<Int>,
        dtype: MLX.DType = .float32
    ) -> [BatchedRetrievalAttentionKVCache] {
        let nLayers = perLayerShape.count
        // Build a config that flags everything as dense; we then opt
        // specific layers IN by hand-flipping via the layerIdx itself
        // (the RetrievalAttentionConfig has a first-N/last-N band notion
        // — we exploit it by making first/last bands cover non-eligible
        // layers).
        var cfg = RetrievalAttentionConfig()
        cfg.denseFirstN = 0
        cfg.denseLastN = 0
        var caches = [BatchedRetrievalAttentionKVCache]()
        caches.reserveCapacity(nLayers)
        for layerIdx in 0..<nLayers {
            let (nKVH, dHead) = perLayerShape[layerIdx]
            let cache = BatchedKVCache(
                maxBatch: B, kvHeads: nKVH, headDim: dHead, maxSeq: T + 64,
                dtype: dtype)
            for _ in 0..<B { _ = cache.addRequest() }
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

            // Layers not in `sparseEligibleLayers` get a config where
            // their layerIdx lands inside the dense band — easiest way is
            // a per-layer config with totalLayers set so layerIdx falls
            // outside the sparse band. With `denseFirstN=0/denseLastN=0`
            // every layer is sparse by default, so we instead flip cfg
            // per-cache: layers we want dense get a `denseLastN=nLayers`
            // (everything is dense), eligible layers keep the open config.
            var layerCfg = cfg
            if !sparseEligibleLayers.contains(layerIdx) {
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

    @Test("Gemma4 fullyBatchedSparseDecode runs end-to-end on synthetic hetero config")
    func gemma4FullyBatchedSparseDecodeSynthetic() throws {
        let cfg = try Gemma4F85SmokeTests.makeSyntheticHeteroConfig()
        let model = Gemma4TextModel(cfg)

        // Use the model's batched KV dims so per-layer shapes match the
        // model expectations (mirrors what
        // `vsm_engine_init_batched(... batchedKVDims)` does at runtime).
        let dims = model.batchedKVDims()
        // Global layers are the only sparse-eligible ones; sliding stays
        // dense per the design documented on
        // `Gemma4ModelInner.fullyBatchedSparseForward`.
        // `Gemma4TextConfiguration.layerTypes` is internal, so probe the
        // sparse-eligibility per layer via the synthetic config JSON.
        let layerTypes: [String] = [
            "sliding_attention", "sliding_attention",
            "full_attention",   "sliding_attention",
        ]
        let globalLayers: Set<Int> = Set(layerTypes.enumerated()
            .filter { $0.element == "full_attention" }
            .map { $0.offset })

        let B = 2
        let T = 64
        let raCaches = Gemma4F85SmokeTests.makeBatchedRACaches(
            B: B, T: T,
            perLayerShape: dims.map { ($0.kvHeads, $0.headDim) },
            sparseEligibleLayers: globalLayers)

        let tokens = (0..<B).map { Int32($0) }
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

    /// Gated real-model smoke. Loads gemma-4-e2b-it-4bit (small, uniform
    /// 4-bit quant — the 26B-A4B variant ships per-layer mixed 4/8-bit
    /// quant which `loadWeights` does not yet thread through), builds
    /// synthetic populated per-layer batched RA caches at ctx=8K B=4,
    /// runs ONE fullyBatchedSparseDecode step. Asserts shape + finite
    /// logits. Cache K/V is randomly populated (not real prefill state)
    /// so this only validates kernel-dispatch + shape contract, not
    /// generation quality.
    ///   RUN_GEMMA4_F85_SMOKE=1 swift test --filter gemma4F85LiveSmoke
    @Test func gemma4F85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_GEMMA4_F85_SMOKE"] == "1" else {
            return
        }

        let modelPath = URL(fileURLWithPath:
            "\(NSHomeDirectory())/models/gemma-4-e2b-it-4bit")
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            Issue.record("model not present: \(modelPath.path)")
            return
        }

        print("[gemma4-f85] loading...", flush: true)
        let configData = try Data(contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(
            Gemma4TextConfiguration.self, from: configData)
        let model = Gemma4TextModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[gemma4-f85] loaded", flush: true)

        // `Gemma4TextConfiguration.layerTypes` / `vocabularySize` are
        // internal — peel them from the raw JSON. The text fields may
        // live at either the top level (text-only configs) or nested
        // under `text_config` (VLM-style configs); the 26B-A4B
        // checkpoint is text-only flat.
        struct GemmaShape: Codable {
            let layer_types: [String]
            let vocab_size: Int
        }
        struct GemmaWrapper: Codable {
            let text_config: GemmaShape?
            let layer_types: [String]?
            let vocab_size: Int?
        }
        let wrapped = try? JSONDecoder().decode(GemmaWrapper.self, from: configData)
        let shape: GemmaShape
        if let inner = wrapped?.text_config {
            shape = inner
        } else if let lt = wrapped?.layer_types, let vs = wrapped?.vocab_size {
            shape = GemmaShape(layer_types: lt, vocab_size: vs)
        } else {
            shape = try JSONDecoder().decode(GemmaShape.self, from: configData)
        }

        let dims = model.batchedKVDims()
        let globalLayers: Set<Int> = Set(shape.layer_types.enumerated()
            .filter { $0.element == "full_attention" }
            .map { $0.offset })
        let nLayers = shape.layer_types.count

        // B=4 ctx=8K matches the Qwen3 F-85 smoke spec. Smaller T helps
        // narrow down shape mismatches when the kernel-path tracing
        // doesn't directly point at the offending tensor.
        let B = 4
        let envT = ProcessInfo.processInfo.environment["GEMMA4_F85_T"]
            .flatMap(Int.init) ?? (8 * 1024)
        let T = envT

        print("[gemma4-f85] B=\(B) T=\(T) nLayers=\(nLayers) "
              + "globalLayers=\(globalLayers.count) "
              + "(sliding\(dims[0]) global\(dims.first(where: { _ in true }) ?? (0,0)))",
              flush: true)

        let raCaches = Gemma4F85SmokeTests.makeBatchedRACaches(
            B: B, T: T,
            perLayerShape: dims.map { ($0.kvHeads, $0.headDim) },
            sparseEligibleLayers: globalLayers,
            dtype: .float16)

        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, raCaches: raCaches)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[gemma4-f85] one batched sparse decode step: "
              + "\(Int((t1 - t0) * 1000))ms (B=\(B), aggregate)",
              flush: true)

        #expect(logits.shape == [B, 1, shape.vocab_size])
        #expect(model.vocabularySize == shape.vocab_size)

        let lastLogits = logits.reshaped(B, -1)
        let sample = lastLogits.asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[gemma4-f85] smoke done", flush: true)
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
