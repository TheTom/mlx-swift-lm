// Copyright © 2026 Tom Turney. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// F-85 batched sparse decode — NemotronH smoke tests.
//
// Validates that the new NemotronH `fullyBatchedSparseDecode` path:
//   1. compiles + links against `BatchedHybridSparseLLM`
//   2. produces logits of the expected shape ([B, 1, vocab])
//   3. doesn't NaN on a synthetic random-weights NemotronH hybrid model
//   4. routes attention layers through `.sparseAttention` and mamba layers
//      through `.gdn` (per-layer dispatch in NemotronHBlock)
//
// A gated live smoke (RUN_NEMOTRONH_F85_SMOKE=1) loads any locally-present
// `~/models/Nemotron*` checkpoint and runs B=4 ctx=8K batched sparse decode
// against synthetic random K/V to confirm the end-to-end real-model path.
//
// Selects the F-73 batched mask kernel via `VSM_SPARSE_BATCHED_KERNEL=f73`
// (also the default — set explicitly so the env intention is documented).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@Suite("NemotronH F-85 batched sparse decode — port", .serialized)
struct NemotronHF85SmokeTests {

    /// Force the F-73 batched mask kernel for this suite. F-73 is the
    /// default but we set it explicitly so the test reflects the intended
    /// dispatch path; alternative values (`f71b`, `f73loop`, `composegather`)
    /// are off-path here.
    fileprivate static func setKernelEnv() {
        setenv("VSM_SPARSE_BATCHED_KERNEL", "f73", 1)
    }

    /// Small synthetic hybrid pattern: 1 Mamba, 1 Attention, 1 MLP, 1 MoE.
    /// Covers every block type the per-layer dispatch needs to handle.
    fileprivate static let testPattern = "M*-E"

    /// Build a minimal NemotronH configuration. Dims kept small (hidden=64,
    /// 4 attn heads, 2 KV heads, head_dim=16) so the test stays light.
    fileprivate static func makeSyntheticNemotronHConfig() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 128,
            hiddenSize: 64,
            numHiddenLayers: testPattern.count,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 4,
            mambaHeadDim: 16,
            ssmStateSize: 16,
            convKernel: 4,
            nGroups: 2,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 4,
            numExpertsPerTok: 2,
            hybridOverridePattern: testPattern,
            layerNormEpsilon: 1e-5,
            nGroup: 2,
            topkGroup: 1
        )
    }

    /// Populate one `BatchedHybridCache` with synthetic K/V on the
    /// attention slots so the sparse path has something to select against.
    /// Mamba slot state stays at zero (addSlot wipes it).
    fileprivate static func populateSparseCache(
        _ caches: BatchedHybridCache, B: Int, T: Int
    ) {
        // Add B request slots across every layer (lockstep).
        for _ in 0..<B { _ = caches.addSlot() }

        // Fill attention slots with random K/V at offset = T so the inner
        // BatchedKVCache reports `offsets[b] == T` and `sparseAttend` sees
        // a populated keyspace.
        for (layerIdx, layer) in caches.layers.enumerated() {
            guard case let .sparseAttention(raCache) = layer else { continue }
            let inner = raCache.inner
            let nKVH = inner.keys.dim(1)
            let dHead = inner.keys.dim(3)
            // Different K per slot so the selector index has signal.
            var ks = [MLXArray]()
            for s in 0..<B {
                ks.append(MLXRandom.normal(
                    [1, nKVH, T, dHead],
                    key: MLXRandom.key(UInt64(100 + layerIdx * 10 + s))
                ).asType(inner.keys.dtype))
            }
            let kAll = concatenated(ks, axis: 0)
            let vAll = MLXRandom.normal(
                [B, nKVH, T, dHead],
                key: MLXRandom.key(UInt64(200 + layerIdx))
            ).asType(inner.values.dtype)
            inner.keys[..<B, 0..., ..<T, 0...] = kAll
            inner.values[..<B, 0..., ..<T, 0...] = vAll
            for i in 0..<B { inner.offsets[i] = T }
            // Seed the selector index with the populated K so block-feature
            // scores aren't all zero.
            raCache.updateIndex(newKeys: kAll)
        }
    }

    @Test("NemotronH fullyBatchedSparseDecode runs end-to-end on synthetic config")
    func nemotronHFullyBatchedSparseDecodeSynthetic() throws {
        Self.setKernelEnv()

        let cfg = Self.makeSyntheticNemotronHConfig()
        let model = NemotronHModel(cfg)

        // Build sparse hybrid cache. Use a dense-band config that DOES
        // include the synthetic attention layer (denseFirstN/denseLastN
        // both 0 = every attention layer is sparse-eligible).
        var raConfig = RetrievalAttentionConfig()
        raConfig.denseFirstN = 0
        raConfig.denseLastN = 0

        let B = 2
        let T = 256
        let caches = model.newBatchedHybridSparseCache(
            maxBatch: B, parameters: nil, raConfig: raConfig)

        Self.populateSparseCache(caches, B: B, T: T)

        // Verify per-layer dispatch landed on the right enum cases: the
        // single '*' layer in "M*-E" → `.sparseAttention`, the 'M' layer
        // → `.gdn`. The cache list has 2 entries (mamba + attention; mlp
        // and moe are skipped per `newBatchedHybridSparseCache`).
        #expect(caches.layers.count == 2,
            "expected 2 cache entries (1 mamba + 1 attention), got \(caches.layers.count)")
        switch caches.layers[0] {
        case .gdn:
            break  // expected
        default:
            Issue.record("cache slot 0 should be .gdn for 'M' block")
        }
        switch caches.layers[1] {
        case .sparseAttention:
            break  // expected
        default:
            Issue.record("cache slot 1 should be .sparseAttention for '*' block")
        }

        // Run one batched sparse decode step against random input tokens.
        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let logits = model.fullyBatchedSparseDecode(inputs, caches: caches)
        eval(logits)

        // Shape: [B, 1, vocab]
        let V = model.vocabularySize
        #expect(logits.shape == [B, 1, V],
            "expected logits [B=\(B), 1, V=\(V)], got \(logits.shape)")

        // Finite check — random-weight NemotronH produces finite output.
        let flat = logits.reshaped(B * V).asArray(Float.self)
        let anyNaN = flat.contains { !$0.isFinite }
        #expect(!anyNaN, "fullyBatchedSparseDecode produced NaN/Inf logits")
    }

    /// Gated real-model smoke. Detects any `~/models/Nemotron*` checkpoint
    /// via shell glob and runs ONE B=4 ctx=8K batched sparse decode step.
    /// Cache K/V randomly populated (not real prefill state) so this only
    /// validates the kernel-dispatch + shape contract, not gen quality.
    ///   RUN_NEMOTRONH_F85_SMOKE=1 swift test --filter nemotronHF85LiveSmoke
    @Test func nemotronHF85LiveSmoke() throws {
        guard ProcessInfo.processInfo.environment["RUN_NEMOTRONH_F85_SMOKE"] == "1" else {
            return
        }
        Self.setKernelEnv()

        // Shell glob for any locally-present Nemotron checkpoint.
        let modelsDir = "\(NSHomeDirectory())/models"
        let candidates: [URL]
        do {
            let contents = try FileManager.default.contentsOfDirectory(
                atPath: modelsDir)
            candidates = contents
                .filter { $0.hasPrefix("Nemotron") }
                .map { URL(fileURLWithPath: modelsDir).appendingPathComponent($0) }
                .filter { FileManager.default.fileExists(
                    atPath: $0.appendingPathComponent("config.json").path) }
        } catch {
            Issue.record("could not list \(modelsDir): \(error)")
            return
        }
        guard let modelPath = candidates.first else {
            Issue.record("no ~/models/Nemotron* checkpoint found; skipping live smoke")
            return
        }

        print("[nemotronh-f85] loading \(modelPath.lastPathComponent)...", flush: true)
        let configData = try Data(
            contentsOf: modelPath.appendingPathComponent("config.json"))
        let cfg = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: configData)
        let model = NemotronHModel(cfg)
        try loadWeights(
            modelDirectory: modelPath, model: model,
            quantization: BaseConfiguration.Quantization(groupSize: 64, bits: 4))
        print("[nemotronh-f85] loaded", flush: true)

        // B=4 ctx=8K matches the spec sheet for F-85 family ports.
        let B = 4
        let T = 8 * 1024

        var raConfig = RetrievalAttentionConfig()
        raConfig.denseFirstN = 0
        raConfig.denseLastN = 0
        // Pass maxKVSize so the inner BatchedKVCache provisions enough seq
        // budget for our synthetic populated state at offset T. Default is
        // 2048 — would broadcast-error against `[B, nKVH, T=8192, D]`.
        let params = GenerateParameters(maxKVSize: T + 64)
        let caches = model.newBatchedHybridSparseCache(
            maxBatch: B, parameters: params, raConfig: raConfig)
        Self.populateSparseCache(caches, B: B, T: T)

        print("[nemotronh-f85] B=\(B) T=\(T) cacheLayers=\(caches.layers.count) "
              + "pattern=\(cfg.hybridOverridePattern)", flush: true)

        let tokens = (0..<B).map { Int32($0) }
        let inputs = MLXArray(tokens).reshaped(B, 1)
        let t0 = CFAbsoluteTimeGetCurrent()
        let logits = model.fullyBatchedSparseDecode(inputs, caches: caches)
        eval(logits)
        let t1 = CFAbsoluteTimeGetCurrent()
        print("[nemotronh-f85] one batched sparse decode step: "
              + "\(Int((t1 - t0) * 1000))ms (B=\(B), aggregate)", flush: true)

        #expect(logits.shape == [B, 1, cfg.vocabSize])
        #expect(model.vocabularySize == cfg.vocabSize)

        let sample = logits.reshaped(B, -1).asArray(Float.self)
        let nonFinite = sample.filter { !$0.isFinite }.count
        #expect(nonFinite == 0, "got \(nonFinite) non-finite logits")
        print("[nemotronh-f85] smoke done", flush: true)
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
