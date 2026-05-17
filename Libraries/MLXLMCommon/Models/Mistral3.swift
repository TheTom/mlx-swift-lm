// Copyright © 2026 Apple Inc.
//
// Shared Mistral 3 / Ministral 3 text-decoder building blocks. Both
// `MLXLLM/Models/Mistral3.swift` (renamed from Mistral3Text.swift) and
// `MLXVLM/Models/Mistral3.swift` consume this namespace.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Mistral 3 / Ministral 3 text decoder. The LLM
/// target uses `attentionHeads` / `vocabularySize` / etc. naming on its
/// config; the VLM target uses `numAttentionHeads` / `vocabSize`. Rather
/// than force-unify the two configs, this namespace's layer classes
/// take a thin `LayerArgs` adapter that each target's config produces.
public enum Mistral3 {

    // MARK: - LayerArgs (adapter)

    /// Minimum field set the layer stack needs from either the LLM or VLM
    /// configuration. Both targets compute one of these from their own
    /// config struct (see the typealiases at the bottom of each target file).
    public struct LayerArgs: Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeParameters: [String: StringOrNumber]?
        public let maxPositionEmbeddings: Int?
        public let layerTypes: [String]
        public let slidingWindow: Int?

        public init(
            hiddenSize: Int, intermediateSize: Int, attentionHeads: Int, kvHeads: Int,
            headDim: Int, rmsNormEps: Float, ropeTheta: Float,
            ropeParameters: [String: StringOrNumber]?, maxPositionEmbeddings: Int?,
            layerTypes: [String], slidingWindow: Int?
        ) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.headDim = headDim
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.ropeParameters = ropeParameters
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.layerTypes = layerTypes
            self.slidingWindow = slidingWindow
        }
    }

    // MARK: - Llama4 attention scaling helper

    /// Llama 4 style position-based attention scaling. Used by Mistral 3 /
    /// Ministral 3 when `rope_parameters.llama_4_scaling_beta` is set.
    public static func llama4AttentionScale(
        start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int
    ) -> MLXArray {
        let positions = MLXArray(Int32(start) ..< Int32(stop))
        let scaling =
            1
            + beta
            * MLX.log(
                1 + MLX.floor(positions.asType(.float32) / Float(maxPositionEmbeddings)))
        return scaling[0..., .newAxis]
    }

    // MARK: - Attention

    public class Attention: Module {
        let nHeads: Int
        let nKVHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPELayer

        public init(_ args: LayerArgs) {
            let dim = args.hiddenSize
            self.nHeads = args.attentionHeads
            self.nKVHeads = args.kvHeads
            self.headDim = args.headDim
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, nHeads * headDim, bias: false)
            self._wk.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(dim, nKVHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(nHeads * headDim, dim, bias: false)

            // Prefer rope_parameters.rope_theta over the top-level value.
            let ropeTheta = args.ropeParameters?["rope_theta"]?.asFloat() ?? args.ropeTheta
            self.rope = initializeRope(
                dims: headDim, base: ropeTheta, traditional: false,
                scalingConfig: args.ropeParameters,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, attnScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            raContext: RetrievalAttentionContext? = nil
        ) -> MLXArray {
            let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
            queries = queries * attnScale

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask,
                raContext: raContext
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return wo(output)
        }

        /// Batched attention: B requests with per-request KV caches.
        /// `attnScale` is shape `[L, 1]` and broadcasts over the batch.
        public func batchedForward(
            _ x: MLXArray, attnScale: MLXArray, caches: [KVCache?]
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let firstOffset = caches[0]?.offset ?? 0
            let allSameOffset = (1 ..< B).allSatisfy {
                (caches[$0]?.offset ?? 0) == firstOffset
            }

            let qSlices: [MLXArray]
            let kSlices: [MLXArray]
            let vSlices: [MLXArray]
            if allSameOffset {
                var qRoped = rope(queries, offset: firstOffset)
                qRoped = qRoped * attnScale
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
                    var rotated = rope(qSlices[i], offset: offset)
                    rotated = rotated * attnScale
                    qR = rotated
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

            return wo(
                output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
            )
        }

        /// Fully batched attention with shared `BatchedKVCache`.
        public func fullyBatchedForward(
            _ x: MLXArray, attnScale: MLXArray,
            cache: BatchedKVCache, layerIndex: Int, mask: MLXArray
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            var queries = wq(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let allSameOffset = cache.offsets[0 ..< cache.active]
                .allSatisfy { $0 == cache.offsets[0] }
            if allSameOffset {
                let offset = cache.offsets[0]
                queries = rope(queries, offset: offset)
                queries = queries * attnScale
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
                    var rotated = rope(qSlices[i], offset: off)
                    rotated = rotated * attnScale
                    rotQ.append(rotated)
                    rotK.append(rope(kSlices[i], offset: off))
                }
                queries = concatenated(rotQ, axis: 0)
                keys = concatenated(rotK, axis: 0)
                cache.update(newKeys: keys, newValues: values)
            }

            let output = cache.attention(queries: queries, scale: scale, mask: mask)
            return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }

        /// F-85 — batched sparse forward for Mistral 3. Mirrors
        /// `Llama.Attention.fullyBatchedSparseForward` and
        /// `Qwen2.Attention.fullyBatchedSparseForward` (MLXLMCommon/Models/
        /// Qwen2.swift:365-421). Mistral 3 has no q/k RMSNorm (closer to
        /// Llama / Qwen2 than Qwen3) but threads the Llama-4 attention
        /// scale (`attnScale`) through the queries after RoPE, matching
        /// the dense `fullyBatchedForward`. Routes through
        /// `BatchedRetrievalAttentionKVCache.sparseAttend` (F-73 batched
        /// mask kernel by default) on sparse-eligible layers at L=1,
        /// else falls back to dense `cache.attention`.
        public func fullyBatchedSparseForward(
            _ x: MLXArray, attnScale: MLXArray,
            raCache: BatchedRetrievalAttentionKVCache, mask: MLXArray
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)
            let cache = raCache.inner

            // Batched Q/K/V projections (bias=false on Mistral 3).
            var queries = wq(x).reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            let allSameOffset = cache.offsets[0 ..< cache.active]
                .allSatisfy { $0 == cache.offsets[0] }
            // Capture pre-update K (post-RoPE) for the selector index.
            let preUpdateK: MLXArray
            if allSameOffset {
                let offset = cache.offsets[0]
                queries = rope(queries, offset: offset)
                queries = queries * attnScale
                keys = rope(keys, offset: offset)
                preUpdateK = keys
                cache.update(newKeys: keys, newValues: values)
            } else {
                // Ragged path — slot-by-slot RoPE. Not the v1 target but
                // here for safety.
                let qSlices = split(queries, parts: B, axis: 0)
                let kSlices = split(keys, parts: B, axis: 0)
                var rotQ = [MLXArray]()
                var rotK = [MLXArray]()
                rotQ.reserveCapacity(B)
                rotK.reserveCapacity(B)
                for i in 0 ..< B {
                    let off = cache.offsets[i]
                    var rotated = rope(qSlices[i], offset: off)
                    rotated = rotated * attnScale
                    rotQ.append(rotated)
                    rotK.append(rope(kSlices[i], offset: off))
                }
                queries = concatenated(rotQ, axis: 0)
                keys = concatenated(rotK, axis: 0)
                preUpdateK = keys
                cache.update(newKeys: keys, newValues: values)
            }

            // Selector index update (skipped for L=1 per F-72; the
            // wrapper method handles the L check).
            raCache.updateIndex(newKeys: preUpdateK)

            let output: MLXArray
            if L == 1 && raCache.isSparseEligible {
                // F-71b / F-73 batched sparse SDPA. queries: [B, nQH, 1, D].
                output = raCache.sparseAttend(
                    queries: queries, scale: scale)
            } else {
                // Dense path — prefill chunk, dense-band layer, or L>1.
                output = cache.attention(queries: queries, scale: scale, mask: mask)
            }
            return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    // MARK: - MLP

    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        public init(_ args: LayerArgs) {
            let dim = args.hiddenSize
            let hiddenDim = args.intermediateSize
            self._gate.wrappedValue = Linear(dim, hiddenDim, bias: false)
            self._down.wrappedValue = Linear(hiddenDim, dim, bias: false)
            self._up.wrappedValue = Linear(dim, hiddenDim, bias: false)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            return down(silu(gate(x)) * up(x))
        }
    }

    // MARK: - TransformerBlock

    public class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public let useSliding: Bool

        public init(_ args: LayerArgs, useSliding: Bool = false) {
            self.useSliding = useSliding
            self._attention.wrappedValue = Attention(args)
            self._mlp.wrappedValue = MLP(args)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, attnScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            raContext: RetrievalAttentionContext? = nil
        ) -> MLXArray {
            let r = attention(
                inputLayerNorm(x), attnScale: attnScale, mask: mask,
                cache: cache, raContext: raContext)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }

        /// Batched forward: batched norms + MLP, per-request attention.
        public func batchedForward(
            _ x: MLXArray, attnScale: MLXArray, caches: [KVCache?]
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = attention.batchedForward(normed, attnScale: attnScale, caches: caches)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }

        /// Fully batched forward with shared `BatchedKVCache`.
        public func fullyBatchedForward(
            _ x: MLXArray, attnScale: MLXArray,
            cache: BatchedKVCache, layerIndex: Int, mask: MLXArray
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = attention.fullyBatchedForward(
                normed, attnScale: attnScale, cache: cache,
                layerIndex: layerIndex, mask: mask)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }

        /// F-85 — batched sparse decoder layer. Same shape as
        /// `fullyBatchedForward` but threads through a
        /// `BatchedRetrievalAttentionKVCache` so sparse-eligible attention
        /// layers can route to F-71b / F-73 batched kernels. Mirrors
        /// `Llama.TransformerBlock.fullyBatchedSparseForward`.
        public func fullyBatchedSparseForward(
            _ x: MLXArray, attnScale: MLXArray,
            raCache: BatchedRetrievalAttentionKVCache, mask: MLXArray
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = attention.fullyBatchedSparseForward(
                normed, attnScale: attnScale, raCache: raCache, mask: mask)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    // MARK: - ModelInner

    /// Shared transformer backbone. Handles full-attention vs sliding-attention
    /// dispatch via `args.layerTypes`, applies Llama-4 attention scaling when
    /// `rope_parameters.llama_4_scaling_beta` is set.
    public class ModelInner: Module {
        @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
        public let layers: [TransformerBlock]
        public let norm: RMSNorm

        public let args: LayerArgs
        public let vocabularySize: Int
        public let faIndex: Int
        public let swaIndex: Int?

        public init(_ args: LayerArgs, vocabularySize: Int) {
            self.args = args
            self.vocabularySize = vocabularySize
            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: args.hiddenSize)
            self.layers = args.layerTypes.map { layerType in
                TransformerBlock(args, useSliding: layerType == "sliding_attention")
            }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self.faIndex = args.layerTypes.firstIndex(of: "full_attention") ?? 0
            self.swaIndex = args.layerTypes.firstIndex(of: "sliding_attention")
            super.init()
        }

        public func callAsFunction(
            _ inputs: MLXArray, cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil,
            raContexts: [RetrievalAttentionContext?]? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbeddings {
                h = inputEmbeddings
            } else {
                h = embedTokens(inputs)
            }

            let offset = cache?.first?.offset ?? 0

            let faMask = createAttentionMask(h: h, cache: cache?[faIndex])
            let swaMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let swaIndex = swaIndex {
                swaMask = createAttentionMask(
                    h: h, cache: cache?[swaIndex], windowSize: args.slidingWindow)
            } else {
                swaMask = .none
            }

            // Llama-4 attention scaling (or constant 1.0 when not configured).
            let attnScale: MLXArray
            if let ropeParams = args.ropeParameters,
                let beta = ropeParams["llama_4_scaling_beta"]?.asFloat(),
                let originalMaxPos = ropeParams["original_max_position_embeddings"]?.asInt()
            {
                attnScale = Self.scale(
                    h: h, offset: offset, length: h.dim(1),
                    beta: beta, maxPositionEmbeddings: originalMaxPos)
            } else {
                attnScale = MLXArray.ones([h.dim(1), 1]).asType(h.dtype)
            }

            for (i, layer) in layers.enumerated() {
                let mask = layer.useSliding ? swaMask : faMask
                h = layer(
                    h, attnScale: attnScale, mask: mask, cache: cache?[i],
                    raContext: raContexts?[i])
            }
            return norm(h)
        }

        /// Static helper so callers can compute the attention scale outside
        /// the model if needed.
        public static func scale(
            h: MLXArray, offset: Int, length: Int, beta: Float, maxPositionEmbeddings: Int
        ) -> MLXArray {
            llama4AttentionScale(
                start: offset, stop: offset + length, beta: beta,
                maxPositionEmbeddings: maxPositionEmbeddings
            ).asType(h.dtype)
        }

        /// Batched forward: B requests with separate per-layer caches.
        public func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
            var h = embedTokens(inputs)
            let attnScale = makeAttnScale(h: h, offset: caches[0][0].offset)
            for (i, layer) in layers.enumerated() {
                let layerCaches = caches.map { $0[i] as KVCache? }
                h = layer.batchedForward(h, attnScale: attnScale, caches: layerCaches)
            }
            return norm(h)
        }

        /// Fully batched forward with shared per-layer `BatchedKVCache`.
        public func fullyBatchedForward(
            _ inputs: MLXArray, caches: [BatchedKVCache]
        ) -> MLXArray {
            var h = embedTokens(inputs)

            let attnScale = makeAttnScale(h: h, offset: caches[0].offsets[0])

            let B = caches[0].active
            let cacheDtype = caches[0].keys.dtype
            let allSame = caches[0].offsets[0 ..< B]
                .allSatisfy { $0 == caches[0].offsets[0] }
            let maxPostOffset = (caches[0].offsets[0 ..< B].max() ?? 0) + 1

            // Global (full attention) mask: all valid positions up to each
            // request's offset.
            let faMask: MLXArray
            if allSame {
                faMask = MLXArray.zeros([B, 1, 1, maxPostOffset], dtype: cacheDtype)
            } else {
                let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(caches[0].offsets[0 ..< B].map { $0 + 1 })
                    .reshaped(B, 1)
                let valid = positions .< offsetsArr
                faMask = MLX.where(
                    valid,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            }

            // Sliding-window mask: positions older than (offset - window + 1)
            // are masked. Only built when at least one sliding layer present.
            let swaMask: MLXArray
            if swaIndex != nil, let window = args.slidingWindow {
                let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(caches[0].offsets[0 ..< B].map { $0 + 1 })
                    .reshaped(B, 1)
                let lowerBound = offsetsArr - Int32(window)
                let inWindow = MLX.logicalAnd(
                    positions .< offsetsArr,
                    positions .>= lowerBound
                )
                swaMask = MLX.where(
                    inWindow,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            } else {
                swaMask = faMask
            }

            for (i, layer) in layers.enumerated() {
                let mask = layer.useSliding ? swaMask : faMask
                h = layer.fullyBatchedForward(
                    h, attnScale: attnScale, cache: caches[i],
                    layerIndex: i, mask: mask)
            }
            return norm(h)
        }

        /// F-85 — batched sparse forward. Shared per-layer
        /// `BatchedRetrievalAttentionKVCache`. The mask is built once
        /// from the inner BatchedKVCache offsets (matches the dense
        /// path). Builds both a global (full-attention) and sliding-
        /// window mask, dispatched per-layer based on `layer.useSliding`.
        /// Mirrors `Llama.ModelInner.fullyBatchedSparseForward` plus the
        /// Mistral 3 sliding/full layer mix from
        /// `Mistral3.ModelInner.fullyBatchedForward`.
        public func fullyBatchedSparseForward(
            _ inputs: MLXArray, raCaches: [BatchedRetrievalAttentionKVCache]
        ) -> MLXArray {
            var h = embedTokens(inputs)

            let attnScale = makeAttnScale(h: h, offset: raCaches[0].inner.offsets[0])

            let cache0 = raCaches[0].inner
            let B = cache0.active
            let cacheDtype = cache0.keys.dtype
            let allSame = cache0.offsets[0 ..< B]
                .allSatisfy { $0 == cache0.offsets[0] }
            let maxPostOffset = (cache0.offsets[0 ..< B].max() ?? 0) + 1

            // Global (full attention) mask: all valid positions up to
            // each request's offset.
            let faMask: MLXArray
            if allSame {
                faMask = MLXArray.zeros([B, 1, 1, maxPostOffset], dtype: cacheDtype)
            } else {
                let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(cache0.offsets[0 ..< B].map { $0 + 1 })
                    .reshaped(B, 1)
                let valid = positions .< offsetsArr
                faMask = MLX.where(
                    valid,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            }

            // Sliding-window mask: positions older than
            // (offset - window + 1) are masked. Only built when at least
            // one sliding layer present.
            let swaMask: MLXArray
            if swaIndex != nil, let window = args.slidingWindow {
                let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
                let offsetsArr = MLXArray(cache0.offsets[0 ..< B].map { $0 + 1 })
                    .reshaped(B, 1)
                let lowerBound = offsetsArr - Int32(window)
                let inWindow = MLX.logicalAnd(
                    positions .< offsetsArr,
                    positions .>= lowerBound
                )
                swaMask = MLX.where(
                    inWindow,
                    MLXArray(Float(0)).asType(cacheDtype),
                    MLXArray(Float(-1e9)).asType(cacheDtype)
                ).reshaped(B, 1, 1, maxPostOffset)
            } else {
                swaMask = faMask
            }

            for (i, layer) in layers.enumerated() {
                let mask = layer.useSliding ? swaMask : faMask
                h = layer.fullyBatchedSparseForward(
                    h, attnScale: attnScale, raCache: raCaches[i], mask: mask)
            }
            return norm(h)
        }

        /// Shared helper to compute the per-layer attention scale (either
        /// the constant 1.0 fallback or Llama-4 position-based scaling).
        private func makeAttnScale(h: MLXArray, offset: Int) -> MLXArray {
            if let ropeParams = args.ropeParameters,
                let beta = ropeParams["llama_4_scaling_beta"]?.asFloat(),
                let originalMaxPos = ropeParams["original_max_position_embeddings"]?.asInt()
            {
                return Self.scale(
                    h: h, offset: offset, length: h.dim(1),
                    beta: beta, maxPositionEmbeddings: originalMaxPos)
            } else {
                return MLXArray.ones([h.dim(1), 1]).asType(h.dtype)
            }
        }
    }
}
