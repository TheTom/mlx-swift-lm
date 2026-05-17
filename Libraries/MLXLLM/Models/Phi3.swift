// Copyright © 2024 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phi3.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class Phi3Attention: Module {

    let args: Phi3Configuration
    let scale: Float

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeDim: Int

    @ModuleInfo(key: "qkv_proj") var wqkv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    public init(_ args: Phi3Configuration) {
        self.args = args

        let dim = args.hiddenSize
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads

        self.headDim = args.hiddenSize / heads
        self.ropeDim = Int(Float(headDim) * args.partialRotaryFactor)
        self.scale = pow(Float(headDim), -0.5)

        self._wqkv.wrappedValue = Linear(dim, (heads + 2 * kvHeads) * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        let ropeScale: Float

        if let ropeScaling = args.ropeScaling, ropeScaling.type == "linear",
            let factor = ropeScaling.factor
        {
            ropeScale = 1 / factor
        } else {
            ropeScale = 1
        }

        if let ropeScaling = args.ropeScaling,
            ropeScaling.type == "su" || ropeScaling.type == "longrope",
            let shortFactor = ropeScaling.shortFactor, let longFactor = ropeScaling.longFactor
        {
            self.rope =
                SuScaledRoPE(
                    dimensions: ropeDim, base: args.ropeTheta,
                    maxPositionEmbeddings: args.maxPositionEmbeddings,
                    originalMaxPositionEmbeddings: args.originalMaxPositionEmbeddings,
                    shortFactor: shortFactor,
                    longFactor: longFactor)

        } else {
            self.rope =
                RoPE(
                    dimensions: ropeDim, traditional: args.ropeTraditional, base: args.ropeTheta,
                    scale: ropeScale)
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queryPos = heads * headDim
        let qkv = split(wqkv(x), indices: [queryPos, queryPos + kvHeads * headDim], axis: -1)
        var queries = qkv[0]
        var keys = qkv[1]
        var values = qkv[2]

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
            raContext: raContext
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// Batched attention: B requests with per-request KV caches.
    /// Phi3 uses fused qkv_proj — split into Q/K/V after the single batched
    /// matmul, then proceed with per-request RoPE + cache + SDPA.
    public func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let queryPos = heads * headDim
        let qkv = split(wqkv(x), indices: [queryPos, queryPos + kvHeads * headDim], axis: -1)
        var queries = qkv[0]
        var keys = qkv[1]
        var values = qkv[2]

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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

        return wo(
            output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        )
    }

    /// Fully batched attention with shared `BatchedKVCache`.
    public func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
        mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let queryPos = heads * headDim
        let qkv = split(wqkv(x), indices: [queryPos, queryPos + kvHeads * headDim], axis: -1)
        var queries = qkv[0].reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = qkv[1].reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = qkv[2].reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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
        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    /// F-85 — batched sparse forward for Phi3. Mirrors
    /// `Llama.LlamaAttention.fullyBatchedSparseForward` but adapts to
    /// Phi3's fused `qkv_proj` (single Linear projection followed by
    /// position-based split into Q/K/V) and partial-RoPE configuration
    /// (rope applies to first `ropeDim` channels). The fused split +
    /// reshape mirrors Phi3's existing `fullyBatchedForward` exactly;
    /// after RoPE + cache.update + selector index update, sparse-
    /// eligible layers at L=1 route through
    /// `BatchedRetrievalAttentionKVCache.sparseAttend` (F-73 batched
    /// mask kernel by default). Else falls back to dense
    /// `cache.attention`.
    ///
    /// Caveman: like fullyBatchedForward but L=1 sparse layers go to
    /// F-73 mask kernel. fused qkv split same as dense path. K/V update
    /// happen via inner.update either way.
    public func fullyBatchedSparseForward(
        _ x: MLXArray, raCache: BatchedRetrievalAttentionKVCache,
        mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // Fused qkv split (Phi3-specific) — single batched matmul, then
        // position-based split into Q/K/V along the last axis. Matches
        // `fullyBatchedForward` line-for-line.
        let queryPos = heads * headDim
        let qkv = split(wqkv(x), indices: [queryPos, queryPos + kvHeads * headDim], axis: -1)
        var queries = qkv[0].reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = qkv[1].reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = qkv[2].reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let allSameOffset = cache.offsets[0 ..< cache.active]
            .allSatisfy { $0 == cache.offsets[0] }
        // Capture pre-update K (post-RoPE) for the selector index.
        let preUpdateK: MLXArray
        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            preUpdateK = keys
            cache.update(newKeys: keys, newValues: values)
        } else {
            // Ragged path — slot-by-slot RoPE. Not the v1 target but
            // here for safety. Phi3's partial RoPE (ropeDim < headDim)
            // is handled by RoPE/SuScaledRoPE internally.
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
            preUpdateK = keys
            cache.update(newKeys: keys, newValues: values)
        }

        // Selector index update (skipped for L=1 per F-72; the wrapper
        // method handles the L check).
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

class Phi3MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_up_proj") var gate_up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gate_up.wrappedValue = Linear(dimensions, 2 * hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gu = split(gate_up(x), parts: 2, axis: -1)
        return down(silu(gu[0]) * gu[1])
    }
}

class Phi3TransformerBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: Phi3Attention
    let mlp: Phi3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Phi3Configuration) {
        self._attention.wrappedValue = Phi3Attention(args)
        self.mlp = Phi3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache, raContext: raContext)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }

    /// Batched forward: batched norms + MLP, per-request attention.
    public func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = attention.batchedForward(normed, caches: caches)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }

    /// Fully batched forward with shared `BatchedKVCache`.
    public func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int, mask: MLXArray
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = attention.fullyBatchedForward(
            normed, cache: cache, layerIndex: layerIndex, mask: mask)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }

    /// F-85 — batched sparse decoder layer. Same shape as
    /// `fullyBatchedForward` but threads through a
    /// `BatchedRetrievalAttentionKVCache` so sparse-eligible attention
    /// layers can route to F-71b / F-73 batched kernels.
    public func fullyBatchedSparseForward(
        _ x: MLXArray, raCache: BatchedRetrievalAttentionKVCache,
        mask: MLXArray
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = attention.fullyBatchedSparseForward(
            normed, raCache: raCache, mask: mask)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Phi3ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Phi3TransformerBlock]
    let norm: RMSNorm
    let args: Phi3Configuration

    public init(_ args: Phi3Configuration) {
        precondition(args.vocabularySize > 0)
        self.args = args

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Phi3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, raContexts: nil)
    }

    /// Sidecar retrieval-attention overload: pass a parallel list of
    /// `RetrievalAttentionContext?` aligned to `cache` so the dispatcher
    /// (`attentionWithCacheUpdate`) routes through the sparse path
    /// without needing a wrapper KV cache. `raContexts` defaults to nil;
    /// when nil this is identical to the legacy entry point. Mirrors
    /// the Llama / Qwen2 / Qwen3 overload (F-83 sparse decode).
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?,
        raContexts: [RetrievalAttentionContext?]?
    ) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], raContext: raContexts?[i])
        }

        return norm(h)
    }

    /// Batched forward: B requests with separate per-layer caches.
    public func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let layerCaches = caches.map { $0[i] as KVCache? }
            h = layer.batchedForward(h, caches: layerCaches)
        }
        return norm(h)
    }

    /// Fully batched forward with shared per-layer `BatchedKVCache`.
    public func fullyBatchedForward(
        _ inputs: MLXArray, caches: [BatchedKVCache]
    ) -> MLXArray {
        var h = embedTokens(inputs)

        let B = caches[0].active
        let cacheDtype = caches[0].keys.dtype
        let allSame = caches[0].offsets[0 ..< B]
            .allSatisfy { $0 == caches[0].offsets[0] }
        let maxPostOffset = (caches[0].offsets[0 ..< B].max() ?? 0) + 1
        let mask: MLXArray
        if allSame {
            mask = MLXArray.zeros([B, 1, 1, maxPostOffset], dtype: cacheDtype)
        } else {
            let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
            let offsetsArr = MLXArray(caches[0].offsets[0 ..< B].map { $0 + 1 })
                .reshaped(B, 1)
            let valid = positions .< offsetsArr
            mask = MLX.where(
                valid,
                MLXArray(Float(0)).asType(cacheDtype),
                MLXArray(Float(-1e9)).asType(cacheDtype)
            ).reshaped(B, 1, 1, maxPostOffset)
        }

        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedForward(h, cache: caches[i], layerIndex: i, mask: mask)
        }
        return norm(h)
    }

    /// F-85 — batched sparse forward. Shared per-layer
    /// `BatchedRetrievalAttentionKVCache`. The mask is built once from
    /// the inner BatchedKVCache offsets (matches the dense path).
    /// Mirrors `Llama.LlamaModelInner.fullyBatchedSparseForward`.
    public func fullyBatchedSparseForward(
        _ inputs: MLXArray, raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var h = embedTokens(inputs)

        let cache0 = raCaches[0].inner
        let B = cache0.active
        let cacheDtype = cache0.keys.dtype
        let allSame = cache0.offsets[0 ..< B]
            .allSatisfy { $0 == cache0.offsets[0] }
        let maxPostOffset = (cache0.offsets[0 ..< B].max() ?? 0) + 1
        let mask: MLXArray
        if allSame {
            mask = MLXArray.zeros([B, 1, 1, maxPostOffset], dtype: cacheDtype)
        } else {
            let positions = MLXArray(0 ..< maxPostOffset).reshaped(1, maxPostOffset)
            let offsetsArr = MLXArray(cache0.offsets[0 ..< B].map { $0 + 1 })
                .reshaped(B, 1)
            let valid = positions .< offsetsArr
            mask = MLX.where(
                valid,
                MLXArray(Float(0)).asType(cacheDtype),
                MLXArray(Float(-1e9)).asType(cacheDtype)
            ).reshaped(B, 1, 1, maxPostOffset)
        }
        for (i, layer) in layers.enumerated() {
            h = layer.fullyBatchedSparseForward(
                h, raCache: raCaches[i], mask: mask)
        }
        return norm(h)
    }
}

public class Phi3Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Phi3ModelInner
    let configuration: Phi3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Phi3Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Phi3ModelInner(args)
        self.configuration = args

        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = configuration.hiddenLayers
        let env = ProcessInfo.processInfo.environment
        let enabled = env["VLLM_TRIATT_ENABLED"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

        // TriAttention V3 — KV-cache eviction policy. Mirrors the Llama
        // factory at MLXLLM/Models/Llama.swift. V3 owns the full cache
        // list (one TriAttentionKVCache per layer) and is incompatible
        // with a caller-supplied maxKVSize (which would route to the
        // eviction-windowed StandardKVCache variant). Phi3 derives
        // headDim from hiddenSize / attentionHeads (matches
        // Phi3Attention.init at line 32) — no explicit `head_dim` config
        // field on Phi3.
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
        // Matches the Llama/Qwen2/Qwen3 factory behavior + the
        // `KVCacheDimensionProvider` extension default in
        // MLXLMCommon/LanguageModel.swift.
        return (0..<numLayers).map { _ in
            makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(inputs, cache: cache, raContexts: nil)
    }

    /// Sidecar retrieval-attention overload: pass a parallel list of
    /// `RetrievalAttentionContext?` aligned to `cache` so the dispatcher
    /// (`attentionWithCacheUpdate`) routes through the sparse path
    /// without needing a wrapper KV cache. `raContexts` defaults to nil;
    /// when nil this is identical to the legacy entry point. Mirrors
    /// the LlamaModel / Qwen2Model / Qwen3Model overload (F-83 sparse
    /// decode).
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?,
        raContexts: [RetrievalAttentionContext?]?
    ) -> MLXArray {
        let out = model(inputs, cache: cache, raContexts: raContexts)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        } else if let lmHead {
            return lmHead(out)
        } else {
            fatalError(
                "Model configuration error: Neither tied embeddings nor lm_head is available")
        }
    }

    /// Batched decode: B requests with per-request per-layer caches.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        let out = model.batchedForward(inputs, caches: caches)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        } else if let lmHead {
            return lmHead(out)
        } else {
            fatalError(
                "Model configuration error: Neither tied embeddings nor lm_head is available")
        }
    }

    /// Fully batched decode with shared per-layer `BatchedKVCache`.
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: [BatchedKVCache]
    ) -> MLXArray {
        let out = model.fullyBatchedForward(inputs, caches: caches)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        } else if let lmHead {
            return lmHead(out)
        } else {
            fatalError(
                "Model configuration error: Neither tied embeddings nor lm_head is available")
        }
    }

    /// F-85 — batched sparse decode. Pairs with `Phi3ModelInner.
    /// fullyBatchedSparseForward`. ONE batched forward call per token,
    /// per-layer attention routes through the F-73 batched mask kernel
    /// (or F-71b via `VSM_SPARSE_BATCHED_KERNEL=f71b`) for sparse-
    /// eligible layers. vllm-swift's `vsm_engine_decode_all` calls here
    /// when sparse + B>1 sessions exist AND `VSM_SPARSE_BATCHED=1`.
    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray, raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        let out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if configuration.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        } else if let lmHead {
            return lmHead(out)
        } else {
            fatalError(
                "Model configuration error: Neither tied embeddings nor lm_head is available")
        }
    }
}

struct RopeScalingWithFactorArrays: Codable {
    let longFactor: [Float]?
    let shortFactor: [Float]?
    let factor: Float?
    let type: String?
    let longMScale: Float?
    let shortMScale: Float?

    enum CodingKeys: String, CodingKey {
        case type
        case factor
        case longFactor = "long_factor"
        case shortFactor = "short_factor"
        case longMScale = "long_mscale"
        case shortMScale = "short_mscale"
    }
}

public struct Phi3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: RopeScalingWithFactorArrays?
    var partialRotaryFactor: Float = 1.0
    var maxPositionEmbeddings: Int
    var originalMaxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool = false

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
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Phi3Configuration.CodingKeys> = try decoder.container(
            keyedBy: Phi3Configuration.CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: Phi3Configuration.CodingKeys.hiddenSize)
        hiddenLayers = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.hiddenLayers)
        intermediateSize = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.intermediateSize)
        attentionHeads = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.attentionHeads)
        rmsNormEps = try container.decode(
            Float.self, forKey: Phi3Configuration.CodingKeys.rmsNormEps)
        vocabularySize = try container.decode(
            Int.self, forKey: Phi3Configuration.CodingKeys.vocabularySize)
        kvHeads = try container.decode(Int.self, forKey: Phi3Configuration.CodingKeys.kvHeads)
        ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Phi3Configuration.CodingKeys.ropeTheta) ?? 10_000
        ropeTraditional =
            try container.decodeIfPresent(
                Bool.self, forKey: Phi3Configuration.CodingKeys.ropeTraditional) ?? false
        ropeScaling = try container.decodeIfPresent(
            RopeScalingWithFactorArrays.self, forKey: .ropeScaling)
        partialRotaryFactor =
            try container.decodeIfPresent(
                Float.self, forKey: .partialRotaryFactor) ?? 1.0
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        originalMaxPositionEmbeddings = try container.decode(
            Int.self, forKey: .originalMaxPositionEmbeddings)
        tieWordEmbeddings =
            try container.decodeIfPresent(
                Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

// MARK: - LoRA

extension Phi3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
