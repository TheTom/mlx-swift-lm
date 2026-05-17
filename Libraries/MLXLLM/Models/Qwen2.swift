//
//  Qwen2.swift
//  LLM
//
//  Created by John Mai on 2024/3/3.
//
//  Layer stack lifted into MLXLMCommon.Qwen2 namespace during the issue
//  #168 consolidation pass (2026-05-06). This file owns only the LLM-side
//  outer model + Configuration.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen2Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false

    /// Adapter producing the shared layer-args struct consumed by
    /// `Qwen2.{Attention, MLP, DecoderLayer, ModelInner}`.
    public var layerArgs: Qwen2.LayerArgs {
        Qwen2.LayerArgs(
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeTraditional: ropeTraditional,
            ropeScaling: ropeScaling)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.ropeTraditional =
            try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        self.ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        self.tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Public LLM-side Qwen 2 model. Wraps `Qwen2.ModelInner` with an optional
/// Linear lm_head (or tied embedding).
public class Qwen2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen2.ModelInner
    let configuration: Qwen2Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen2Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen2.ModelInner(args.layerArgs, vocabularySize: args.vocabularySize)

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                args.hiddenSize, args.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, raContexts: nil)
    }

    /// Sidecar retrieval-attention overload: pass a parallel list of
    /// `RetrievalAttentionContext?` aligned to `cache` so the dispatcher
    /// (`attentionWithCacheUpdate`) routes through the sparse path
    /// without needing a wrapper KV cache. `raContexts` defaults to nil;
    /// when nil this is identical to the legacy entry point.
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?,
        raContexts: [RetrievalAttentionContext?]?
    ) -> MLXArray {
        var out = model(inputs, cache: cache, raContexts: raContexts)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Batched decode: B requests with separate per-layer caches.
    /// Pairs with `Qwen2.ModelInner.batchedForward` so vllm-swift's
    /// `vsm_engine_decode_all` can amortize weight bandwidth across
    /// concurrent requests instead of looping per-stream.
    /// inputs: [B, 1] token IDs. caches: B arrays of per-layer KVCache.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var out = model.batchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Fully batched decode with shared per-layer `BatchedKVCache`. The
    /// fastest concurrent-decode path — zero per-request loops in the
    /// hot path. vllm-swift's `vsm_engine_decode_all` routes here when
    /// `engine.batchedCaches` is populated (after `vsm_engine_init_batched`).
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: [BatchedKVCache]
    ) -> MLXArray {
        var out = model.fullyBatchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// F-85 — batched sparse decode. Pairs with `Qwen2.ModelInner.
    /// fullyBatchedSparseForward`. ONE batched forward call per token,
    /// per-layer attention routes through F-71b sparse kernel for
    /// sparse-eligible layers. vllm-swift's `vsm_engine_decode_all` calls
    /// here when sparse + B>1 sessions exist AND `VSM_SPARSE_BATCHED=1`.
    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray, raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if configuration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }
        weights = weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
        // F-83 decode sprint iter #5 — concat gate_proj + up_proj into
        // a fused gate_up_proj. Saves one matmul dispatch per layer per
        // decode step (~48 dispatches × ~80 µs = ~4 ms at 16K on M5 Max).
        weights = Qwen2.fuseGateUpWeights(weights)
        return weights
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = configuration.hiddenLayers
        let env = ProcessInfo.processInfo.environment
        let enabled = env["VLLM_TRIATT_ENABLED"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

        // TriAttention V3 — KV-cache eviction policy. Mirrors the Qwen3
        // factory at MLXLLM/Models/Qwen3.swift. V3 owns the full cache
        // list (one TriAttentionKVCache per layer) and is incompatible
        // with a caller-supplied maxKVSize (which would route to the
        // eviction-windowed StandardKVCache variant). Qwen2 has no
        // explicit `head_dim` config field — derive it from hiddenSize /
        // attentionHeads, matching `Qwen2.Attention.init` at
        // MLXLMCommon/Models/Qwen2.swift:80.
        if enabled, parameters?.maxKVSize == nil {
            let headDim = configuration.hiddenSize / configuration.attentionHeads
            let engine = TriAttentionV3Engine(
                cfg: .fromEnv(),
                nLayers: configuration.hiddenLayers,
                nHeads: configuration.attentionHeads,
                nKVHeads: configuration.kvHeads,
                headDim: headDim,
                ropeTheta: configuration.ropeTheta
            )
            TriAttentionRescue.shared.install(on: engine)
            return (0..<numLayers).map { layerIdx in
                TriAttentionKVCache(layerIdx: layerIdx, engine: engine)
            }
        }

        // Default path — route through `makeAttentionCache` so caller-
        // supplied `maxKVSize` picks the eviction-windowed variant.
        // Matches the Qwen3 factory's behavior + the
        // `KVCacheDimensionProvider` extension default in
        // MLXLMCommon/LanguageModel.swift.
        return (0..<numLayers).map { _ in
            makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
        }
    }
}

extension Qwen2Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
