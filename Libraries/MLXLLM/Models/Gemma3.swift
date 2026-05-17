//
//  Gemma3.swift
//  mlx-swift-lm
//
//  Created by Anthony DePasquale on 14.03.2025.
//  Renamed from Gemma3Text.swift on 2026-05-06 (issue #168 consolidation).
//

// Based on https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma3_text.py
//
// The text-decoder layer stack lives in MLXLMCommon/Models/Gemma3.swift as
// the public `Gemma3` namespace. This file owns only the LLM-side outer
// model wrapper.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Back-compat alias. The configuration moved into the shared
/// `Gemma3` namespace under MLXLMCommon during the issue #168
/// consolidation pass; LLMModelFactory and unit tests still construct
/// the old name.
public typealias Gemma3TextConfiguration = Gemma3.TextConfiguration

/// Public LLM-side Gemma 3 text model. Wraps the shared
/// `Gemma3.Backbone` with a Linear lm_head and the
/// LLMModel-protocol-required hooks (sanitize, newCache, prepare).
public class Gemma3TextModel: Module, LLMModel {

    @ModuleInfo public var model: Gemma3.Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma3.TextConfiguration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma3.TextConfiguration) {
        self.config = config
        self.model = Gemma3.Backbone(config)
        self._lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let optionalCache = cache?.map { $0 as KVCache? }
        let h = model(inputs, cache: optionalCache)
        return lmHead(h)
    }

    /// Batched decode: B requests with per-request per-layer caches.
    /// Pairs with `Gemma3.Backbone.batchedForward` for vllm-swift's
    /// `vsm_engine_decode_all` semi-batched path.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        let h = model.batchedForward(inputs, caches: caches)
        return lmHead(h)
    }

    /// Fully batched decode with shared per-layer `BatchedKVCache`. The
    /// backbone handles dual-mask (sliding vs global) dispatch internally.
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: [BatchedKVCache]
    ) -> MLXArray {
        let h = model.fullyBatchedForward(inputs, caches: caches)
        return lmHead(h)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processedWeights = weights

        // VLM models converted using mlx_vlm.convert have weights nested under a
        // `language_model.` prefix; strip it here so `model.layers.*` resolves.
        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Some converters pad embed_tokens / lm_head past the model's actual
        // vocab size (rounding up to a multiple). Trim to the configured size.
        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }

        // Weight tying: copy embed_tokens to lm_head when the latter is missing.
        if processedWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = processedWeights["model.embed_tokens.\(key)"] {
                    processedWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }
        return processedWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        // TriAttention V3 — KV-cache eviction policy. Mirrors the Qwen2
        // factory at MLXLLM/Models/Qwen2.swift and the Qwen3 factory at
        // MLXLLM/Models/Qwen3.swift. Engages when `VLLM_TRIATT_ENABLED`
        // is set in the environment AND the caller did not supply
        // `parameters?.maxKVSize` (which routes to the eviction-windowed
        // StandardKVCache variant instead).
        //
        // Gemma 3 architectural note (NOT a Gemma 4 style fall-through):
        // ----------------------------------------------------------------
        // Gemma 3 interleaves sliding-window and global-attention layers
        // via `slidingWindowPattern`, BUT — unlike Gemma 4 — uses a
        // SINGLE `headDim` / `kvHeads` / `nHeads` triple across both
        // attention types (see Gemma3.TextConfiguration in
        // MLXLMCommon/Models/Gemma3.swift). Sliding and global layers
        // only differ in their RoPE base frequency (`ropeLocalBaseFreq`
        // vs `ropeTheta`). The V3 engine pins on (nHeads, nKVHeads,
        // headDim, ropeTheta), so a single engine CAN serve all layers
        // — we install it directly (matches the Qwen2 / Qwen3 / Llama
        // pattern). The mixed RoPE base across sliding layers means the
        // selector's block-feature trig table is built with the global
        // theta only; sliding layers reuse those features. This matches
        // the Gemma 4 sliding-layer behaviour where V3 is opt-out at the
        // layer level via the dense band but uses one engine.
        let env = ProcessInfo.processInfo.environment
        let triEnabled = env["VLLM_TRIATT_ENABLED"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

        if triEnabled, parameters?.maxKVSize == nil {
            let engine = TriAttentionV3Engine(
                cfg: .fromEnv(),
                nLayers: config.hiddenLayers,
                nHeads: config.attentionHeads,
                nKVHeads: config.kvHeads,
                headDim: config.headDim,
                ropeTheta: config.ropeTheta
            )
            TriAttentionRescue.shared.install(on: engine)
            return (0 ..< config.hiddenLayers).map { layerIdx in
                TriAttentionKVCache(layerIdx: layerIdx, engine: engine)
            }
        }

        // Default path — sliding-window-aware per-layer cache mix.
        var caches = [KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)
            let cache = makeAttentionCache(
                parameters: parameters,
                maxSize: isGlobalLayer ? nil : slidingWindow)
            // For global layers (unbounded StandardKVCache), bump the step
            // size for long-sequence efficiency. Affine-quantized caches
            // don't honor `step` and ignore this.
            if isGlobalLayer, let standard = cache as? StandardKVCache {
                standard.step = 1024
            }
            caches.append(cache)
        }
        return caches
    }

    /// Empty-prompt guard; otherwise the iterator handles prefill itself.
    public func prepare(
        _ input: LMInput, cache: [KVCache], windowSize: Int? = nil
    ) throws -> PrepareResult {
        let promptTokens = input.text.tokens
        let promptCount = promptTokens.dim(0)

        guard promptCount > 0 else {
            print("Warning: Preparing with empty prompt tokens.")
            let emptyToken = MLXArray(Int32(0))[0 ..< 0]
            return .tokens(.init(tokens: emptyToken))
        }
        return .tokens(input.text)
    }
}

extension Gemma3TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
