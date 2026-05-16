// Copyright © 2026 Apple Inc.
//
// Shared Gemma 3 text-decoder building blocks. Both `MLXLLM/Models/Gemma3.swift`
// (the text-only Gemma3TextModel) and `MLXVLM/Models/Gemma3.swift` (the
// vision-language wrapper) consume this namespace to avoid duplication.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Gemma 3 text decoder. The outer LLM and VLM model
/// classes live in their respective targets and share this layer stack.
public enum Gemma3 {

    // MARK: - Configuration

    /// JSON config parser shared across LLM and VLM Gemma 3 variants.
    ///
    /// Handles both flat config (LLM-side `gemma3_text` repos) and the
    /// nested `text_config` wrapper used by VLM repos when a converter has
    /// merged text + vision config into a single file.
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let kvHeads: Int
        public let ropeTheta: Float
        public let ropeLocalBaseFreq: Float
        public let ropeTraditional: Bool
        public let queryPreAttnScalar: Float
        public let slidingWindow: Int
        public let slidingWindowPattern: Int
        public let maxPositionEmbeddings: Int
        public let ropeScaling: [String: StringOrNumber]?
        public let finalLogitSoftcapping: Float?

        public init(
            modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
            attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int,
            kvHeads: Int, ropeTheta: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool,
            queryPreAttnScalar: Float, slidingWindow: Int, slidingWindowPattern: Int,
            maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil,
            finalLogitSoftcapping: Float? = nil
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.headDim = headDim
            self.rmsNormEps = rmsNormEps
            self.vocabularySize = vocabularySize
            self.kvHeads = kvHeads
            self.ropeTheta = ropeTheta
            self.ropeLocalBaseFreq = ropeLocalBaseFreq
            self.ropeTraditional = ropeTraditional
            self.queryPreAttnScalar = queryPreAttnScalar
            self.slidingWindow = slidingWindow
            self.slidingWindowPattern = slidingWindowPattern
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.ropeScaling = ropeScaling
            self.finalLogitSoftcapping = finalLogitSoftcapping
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeLocalBaseFreq = "rope_local_base_freq"
            case ropeTraditional = "rope_traditional"
            case queryPreAttnScalar = "query_pre_attn_scalar"
            case slidingWindow = "sliding_window"
            case slidingWindowPattern = "sliding_window_pattern"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeScaling = "rope_scaling"
            case finalLogitSoftcapping = "final_logit_softcapping"
        }

        enum VLMCodingKeys: String, CodingKey {
            case textConfig = "text_config"
        }

        public init(from decoder: Decoder) throws {
            // VLM repos converted via mlx_lm.convert nest the text fields under
            // `text_config`; LLM repos have the fields at the top level.
            let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)
            let container =
                if nestedContainer.contains(.textConfig) {
                    try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
                } else {
                    try decoder.container(keyedBy: CodingKeys.self)
                }

            modelType = try container.decode(String.self, forKey: .modelType)
            hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
            intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
            // Some checkpoints (notably mlx_lm-converted multimodal Gemma3
            // VLM repos) omit attention head fields from config.json; fall
            // back to the canonical per-variant defaults indexed by
            // `hidden_size`.
            //   1B:  hidden=1152, heads=4,  kv=1,  head_dim=256
            //   4B:  hidden=2560, heads=8,  kv=4,  head_dim=256
            //   12B: hidden=3584, heads=16, kv=8,  head_dim=256
            //   27B: hidden=5376, heads=32, kv=16, head_dim=128
            let defHeads: Int
            let defKVHeads: Int
            let defHeadDim: Int
            if hiddenSize < 2048 {
                defHeads = 4; defKVHeads = 1; defHeadDim = 256
            } else if hiddenSize < 3072 {
                defHeads = 8; defKVHeads = 4; defHeadDim = 256
            } else if hiddenSize < 4096 {
                defHeads = 16; defKVHeads = 8; defHeadDim = 256
            } else {
                defHeads = 32; defKVHeads = 16; defHeadDim = 128
            }
            attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? defHeads
            headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? defHeadDim
            rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
            // 1B = 262144, 4B+ = 262208 — always JSON-driven.
            vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
            kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? defKVHeads
            ropeTheta =
                try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
            ropeLocalBaseFreq =
                try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
            ropeTraditional =
                try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            queryPreAttnScalar =
                try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256
            slidingWindow =
                try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
            slidingWindowPattern =
                try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 6
            maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
            ropeScaling =
                try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            finalLogitSoftcapping =
                try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
        }
    }

    // MARK: - Attention

    public class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int
        let scale: Float
        let isSliding: Bool

        @ModuleInfo(key: "q_proj") var queryProj: Linear
        @ModuleInfo(key: "k_proj") var keyProj: Linear
        @ModuleInfo(key: "v_proj") var valueProj: Linear
        @ModuleInfo(key: "o_proj") var outputProj: Linear

        @ModuleInfo(key: "q_norm") var queryNorm: Gemma.RMSNorm
        @ModuleInfo(key: "k_norm") var keyNorm: Gemma.RMSNorm

        @ModuleInfo var rope: RoPELayer

        public init(_ config: TextConfiguration, layerIdx: Int) {
            let dim = config.hiddenSize
            self.nHeads = config.attentionHeads
            self.nKVHeads = config.kvHeads
            self.headDim = config.headDim
            self.scale = pow(config.queryPreAttnScalar, -0.5)
            self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0

            self._queryProj.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._keyProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._outputProj.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            self._queryNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            self._keyNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)

            // Sliding-window layers use the local rope frequency; global layers
            // use the configured theta with optional scaling.
            if isSliding {
                self.rope = initializeRope(
                    dims: headDim, base: config.ropeLocalBaseFreq, traditional: false,
                    scalingConfig: nil, maxPositionEmbeddings: nil)
            } else {
                self.rope = initializeRope(
                    dims: headDim, base: config.ropeTheta, traditional: false,
                    scalingConfig: config.ropeScaling,
                    maxPositionEmbeddings: config.maxPositionEmbeddings)
            }
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache? = nil
        ) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            var queries = queryProj(x)
            var keys = keyProj(x)
            var values = valueProj(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            queries = queryNorm(queries)
            keys = keyNorm(keys)

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return outputProj(output)
        }

        /// Batched attention: B requests with per-request KV caches.
        /// Mirrors Qwen2.Attention.batchedForward; Gemma3 q/k norms applied
        /// post-reshape (head-wise), no bias on projections.
        public func batchedForward(
            _ x: MLXArray, caches: [KVCache?]
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            var queries = queryProj(x)
            var keys = keyProj(x)
            var values = valueProj(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            // Apply q/k norm (head-wise) on the batched tensor — same op
            // semantics as single-request path. Norm weights are broadcast.
            queries = queryNorm(queries)
            keys = keyNorm(keys)

            let firstOffset = caches[0]?.offset ?? 0
            let allSameOffset = (1 ..< B).allSatisfy {
                (caches[$0]?.offset ?? 0) == firstOffset
            }

            let qSlices: [MLXArray]
            let kSlices: [MLXArray]
            let vSlices: [MLXArray]
            if allSameOffset {
                let qRoped = rope(queries, offset: firstOffset)
                let kRoped = rope(keys, offset: firstOffset)
                qSlices = split(qRoped, parts: B, axis: 0)
                kSlices = split(kRoped, parts: B, axis: 0)
                vSlices = split(values, parts: B, axis: 0)
            } else {
                qSlices = split(queries, parts: B, axis: 0)
                kSlices = split(keys, parts: B, axis: 0)
                vSlices = split(values, parts: B, axis: 0)
            }

            var rotQ = [MLXArray]()
            var allKeys = [MLXArray]()
            var allVals = [MLXArray]()
            rotQ.reserveCapacity(B)
            allKeys.reserveCapacity(B)
            allVals.reserveCapacity(B)

            var allSameLen = true
            var firstLen = -1

            for i in 0 ..< B {
                let cache_i = caches[i]
                let qR: MLXArray
                let kR: MLXArray
                if allSameOffset {
                    qR = qSlices[i]
                    kR = kSlices[i]
                } else {
                    let offset = cache_i?.offset ?? 0
                    qR = rope(qSlices[i], offset: offset)
                    kR = rope(kSlices[i], offset: offset)
                }
                let (aK, aV) = cache_i?.update(keys: kR, values: vSlices[i])
                    ?? (kR, vSlices[i])
                rotQ.append(qR)
                allKeys.append(aK)
                allVals.append(aV)

                let sLen = aK.dim(2)
                if firstLen < 0 { firstLen = sLen }
                if sLen != firstLen { allSameLen = false }
            }

            let output: MLXArray
            if allSameLen && B > 1 {
                let bQ = concatenated(rotQ, axis: 0)
                let bK = concatenated(allKeys, axis: 0)
                let bV = concatenated(allVals, axis: 0)
                output = MLXFast.scaledDotProductAttention(
                    queries: bQ, keys: bK, values: bV,
                    scale: scale, mask: .none
                )
            } else {
                var outputs = [MLXArray]()
                outputs.reserveCapacity(B)
                for i in 0 ..< B {
                    let attn = MLXFast.scaledDotProductAttention(
                        queries: rotQ[i], keys: allKeys[i], values: allVals[i],
                        scale: scale, mask: .none
                    )
                    outputs.append(attn)
                }
                output = concatenated(outputs, axis: 0)
            }

            return outputProj(
                output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            )
        }

        /// Fully batched attention with shared `BatchedKVCache`. The caller
        /// provides the correct mask for this layer's attention type
        /// (sliding-window vs global), built once in `Backbone.fullyBatchedForward`.
        public func fullyBatchedForward(
            _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
            mask: MLXArray
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            var queries = queryProj(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            var keys = keyProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            let values = valueProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            queries = queryNorm(queries)
            keys = keyNorm(keys)

            let allSameOffset = cache.offsets[0 ..< cache.active]
                .allSatisfy { $0 == cache.offsets[0] }
            if allSameOffset {
                let offset = cache.offsets[0]
                queries = rope(queries, offset: offset)
                keys = rope(keys, offset: offset)
                cache.update(newKeys: keys, newValues: values)
            } else {
                let qSlices = split(queries, parts: B, axis: 0)
                let kSlices = split(keys, parts: B, axis: 0)
                var rotQ = [MLXArray]()
                var rotK = [MLXArray]()
                rotQ.reserveCapacity(B)
                rotK.reserveCapacity(B)
                for i in 0 ..< B {
                    let off = cache.offsets[i]
                    rotQ.append(rope(qSlices[i], offset: off))
                    rotK.append(rope(kSlices[i], offset: off))
                }
                queries = concatenated(rotQ, axis: 0)
                keys = concatenated(rotK, axis: 0)
                cache.update(newKeys: keys, newValues: values)
            }

            let output = cache.attention(queries: queries, scale: scale, mask: mask)
            return outputProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    // MARK: - MLP

    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(geluApproximate(gateProj(x)) * upProj(x))
        }
    }

    // MARK: - TransformerBlock

    public class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
        @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

        public init(_ config: TextConfiguration, layerIdx: Int) {
            self._selfAttention.wrappedValue = Attention(config, layerIdx: layerIdx)
            self.mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

            self._inputLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache? = nil
        ) -> MLXArray {
            let r = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
            let r2 = mlp(preFeedforwardLayerNorm(h))
            let out = Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
            return out
        }

        /// Batched forward: batched norms + MLP, per-request attention.
        public func batchedForward(
            _ x: MLXArray, caches: [KVCache?]
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = selfAttention.batchedForward(normed, caches: caches)
            let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
            let r2 = mlp(preFeedforwardLayerNorm(h))
            return Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
        }

        /// Fully batched forward with shared `BatchedKVCache`. Mask is
        /// pre-computed in `Backbone.fullyBatchedForward` for this layer's
        /// attention type (sliding-window vs global).
        public func fullyBatchedForward(
            _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int, mask: MLXArray
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = selfAttention.fullyBatchedForward(
                normed, cache: cache, layerIndex: layerIndex, mask: mask)
            let h = Gemma.clipResidual(x, postAttentionLayerNorm(r))
            let r2 = mlp(preFeedforwardLayerNorm(h))
            return Gemma.clipResidual(h, postFeedforwardLayerNorm(r2))
        }
    }

    // MARK: - Backbone

    /// Shared transformer backbone (embed → N transformer blocks → norm).
    /// The LLM target wraps this with a `Linear` lm_head; the VLM target wraps
    /// it with an lm_head that may be `Linear` or `QuantizedLinear` and adds
    /// optional final-logit softcapping. Both pass `inputs`; the VLM path also
    /// passes `inputEmbedding` (a vision-fused embed) when prefilling with an
    /// image attachment.
    public class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
        @ModuleInfo public var layers: [TransformerBlock]
        @ModuleInfo public var norm: Gemma.RMSNorm

        public let config: TextConfiguration

        public init(_ config: TextConfiguration) {
            self.config = config
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize)
            self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
                TransformerBlock(config, layerIdx: layerIdx)
            }
            self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ inputs: MLXArray? = nil,
            inputEmbedding: MLXArray? = nil,
            mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache?]? = nil
        ) -> MLXArray {
            let h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("Backbone requires either `inputs` or `inputEmbedding`")
            }

            // sqrt(hiddenSize) scale, computed in bf16 then cast to runtime dtype.
            let scale = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
                .asType(h.dtype)
            var stream = h * scale

            var layerCache = cache
            if layerCache == nil {
                layerCache = Array(repeating: nil as KVCache?, count: layers.count)
            }

            // Build masks once. Sliding-window layers and global layers use
            // different masks; cache[0] is sliding (if pattern>1) and
            // cache[pattern-1] is global.
            let globalMask = createAttentionMask(
                h: stream, cache: cache?[config.slidingWindowPattern - 1])
            let slidingWindowMask =
                if config.slidingWindowPattern > 1 {
                    createAttentionMask(
                        h: stream, cache: cache?[0], windowSize: config.slidingWindow)
                } else {
                    MLXFast.ScaledDotProductAttentionMaskMode.none
                }

            for (i, layer) in layers.enumerated() {
                let isGlobal =
                    (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
                let m = isGlobal ? globalMask : slidingWindowMask
                stream = layer(stream, mask: m, cache: layerCache?[i])
            }
            return norm(stream)
        }

        /// Batched forward: B requests with separate per-layer caches.
        /// caches: [[KVCache]] — outer per-request, inner per-layer.
        public func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
            var stream = embedTokens(inputs)
            // sqrt(hiddenSize) scale, computed in bf16 then cast to runtime dtype.
            let s = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
                .asType(stream.dtype)
            stream = stream * s

            for (i, layer) in layers.enumerated() {
                let layerCaches = caches.map { $0[i] as KVCache? }
                stream = layer.batchedForward(stream, caches: layerCaches)
            }
            return norm(stream)
        }

        /// Fully batched forward with shared per-layer `BatchedKVCache`.
        /// Builds two masks: one for global layers, one for sliding-window
        /// layers — and routes by `(i % slidingWindowPattern == pattern - 1)`.
        public func fullyBatchedForward(
            _ inputs: MLXArray, caches: [BatchedKVCache]
        ) -> MLXArray {
            var stream = embedTokens(inputs)
            let s = MLXArray(sqrt(Float(config.hiddenSize)), dtype: .bfloat16)
                .asType(stream.dtype)
            stream = stream * s

            // Build per-attention-type masks once. For L=1 decode with all-
            // same-offset, global mask is all-zeros up to maxPostOffset; the
            // sliding-window mask masks positions older than `slidingWindow`.
            let B = caches[0].active
            let cacheDtype = caches[0].keys.dtype
            let allSame = caches[0].offsets[0 ..< B]
                .allSatisfy { $0 == caches[0].offsets[0] }
            let maxPostOffset = (caches[0].offsets[0 ..< B].max() ?? 0) + 1

            // Global mask: all valid positions up to each request's offset.
            let globalMask: MLXArray
            if allSame {
                globalMask = MLXArray.zeros([B, 1, 1, maxPostOffset], dtype: cacheDtype)
            } else {
                let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(caches[0].offsets[0 ..< B].map { $0 + 1 })
                    .reshaped(B, 1)
                let valid = positions .< offsetsArr
                globalMask = MLX.where(
                    valid,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            }

            // Sliding-window mask: positions older than (offset - slidingWindow + 1)
            // are masked out. Only build when there is at least one sliding layer.
            let slidingMask: MLXArray
            if config.slidingWindowPattern > 1 {
                let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(caches[0].offsets[0 ..< B].map { $0 + 1 })
                    .reshaped(B, 1)
                let lowerBound = offsetsArr - Int32(config.slidingWindow)
                let inWindow = MLX.logicalAnd(
                    positions .< offsetsArr,
                    positions .>= lowerBound
                )
                slidingMask = MLX.where(
                    inWindow,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            } else {
                slidingMask = globalMask
            }

            for (i, layer) in layers.enumerated() {
                let isGlobal =
                    (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
                let m = isGlobal ? globalMask : slidingMask
                stream = layer.fullyBatchedForward(
                    stream, cache: caches[i], layerIndex: i, mask: m)
            }
            return norm(stream)
        }
    }
}
