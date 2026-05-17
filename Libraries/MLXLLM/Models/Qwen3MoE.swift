//
//  Qwen3MoE.swift
//  LLM
//
//  Created by John Mai on 2025/4/30.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_moe.py

class Qwen3MoEAttention: Module {
    let args: Qwen3MoEConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    public init(_ args: Qwen3MoEConfiguration, layerIdx: Int) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeScale: Float
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // V3 calibration: pre-RoPE Q, when present. Mirrors Qwen3 ordering.
        let triCache = cache as? TriAttentionKVCache
        if B == 1, let triCache {
            let qForCalibration = queries[0].transposed(1, 0, 2).asType(.float32)
            triCache.engine.accumulateQ(qForCalibration, layerIdx: triCache.layerIdx)
        }

        // TriAttention physically compacts K/V storage. Keep RoPE position
        // tied to the original logical token stream, not the compacted
        // storage length (`cache.offset`). Mirrors Qwen3.
        let rotaryOffset = triCache?.logicalOffset ?? (cache?.offset ?? 0)
        queries = rope(queries, offset: rotaryOffset)
        keys = rope(keys, offset: rotaryOffset)

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
    /// Mirrors Qwen3Attention.batchedForward — Qwen3MoE attention is
    /// architecturally identical to Qwen3 dense attention (q/k_norm + RoPE).
    public func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
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

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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

    /// F-85 — batched sparse forward for Qwen3MoE. Mirrors
    /// `Qwen3Attention.fullyBatchedSparseForward` (Qwen3.swift:365) since
    /// Qwen3MoE shares Qwen3's attention shape: q/k RMSNorm BEFORE RoPE,
    /// then attention. The MoE FFN sub-layer is orthogonal — sparse only
    /// modifies the attention KV gather, not expert routing. Routes through
    /// `BatchedRetrievalAttentionKVCache.sparseAttend` (F-73 batched mask
    /// kernel by default) on sparse-eligible layers at L=1, else falls
    /// back to dense `cache.attention`.
    public func fullyBatchedSparseForward(
        _ x: MLXArray, raCache: BatchedRetrievalAttentionKVCache,
        mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // qNorm + kNorm applied per-head BEFORE RoPE (matches Qwen3MoE's
        // standard callAsFunction ordering). Values pass through unchanged.
        var queries = qNorm(wq(x).reshaped(B, L, args.attentionHeads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, args.kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, args.kvHeads, -1)
            .transposed(0, 2, 1, 3)

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
            // Ragged path — slot-by-slot RoPE.
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

class Qwen3MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Qwen3MoESparseMoeBlock: Module, UnaryLayer {
    let numExperts: Int
    let topK: Int
    let normTopkProb: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: Qwen3MoEConfiguration) {
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopkProb = args.normTopkProb

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize, numExperts: numExperts
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = gate(x)
        let softGates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let inds = MLX.argPartition(-gates, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        var scores = MLX.takeAlong(softGates, inds, axis: -1)

        if normTopkProb {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        return (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class Qwen3MoeDecoderLayer: Module {
    let args: Qwen3MoEConfiguration
    let layerIdx: Int

    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3MoEAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    fileprivate let mlp: UnaryLayer

    init(_ args: Qwen3MoEConfiguration, layerIdx: Int) {
        self.args = args
        self.layerIdx = layerIdx

        _selfAttn.wrappedValue = Qwen3MoEAttention(args, layerIdx: layerIdx)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        if !args.mlpOnlyLayers.contains(layerIdx),
            args.numExperts > 0, (layerIdx + 1) % args.decoderSparseStep == 0
        {
            self.mlp = Qwen3MoESparseMoeBlock(args)
        } else {
            self.mlp = Qwen3MoEMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        var r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache, raContext: raContext)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        let out = h + r
        return out
    }

    /// Batched forward: batched norms + MLP (MoE SwitchGLU is already
    /// expert-batched internally), per-request attention.
    func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = selfAttn.batchedForward(normed, caches: caches)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }

    /// Fully batched forward with shared `BatchedKVCache`.
    func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int, mask: MLXArray
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = selfAttn.fullyBatchedForward(
            normed, cache: cache, layerIndex: layerIndex, mask: mask)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }

    /// F-85 — batched sparse decoder layer. Same shape as
    /// `fullyBatchedForward` but threads through a
    /// `BatchedRetrievalAttentionKVCache` so sparse-eligible attention
    /// layers can route to F-71b / F-73 batched kernels. MoE FFN
    /// (`Qwen3MoESparseMoeBlock`) handles `[B, 1, hidden]` natively via
    /// its `SwitchGLU` so no MoE-specific path is needed.
    func fullyBatchedSparseForward(
        _ x: MLXArray, raCache: BatchedRetrievalAttentionKVCache,
        mask: MLXArray
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r = selfAttn.fullyBatchedSparseForward(
            normed, raCache: raCache, mask: mask)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Qwen3MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3MoeDecoderLayer]
    let norm: RMSNorm
    let args: Qwen3MoEConfiguration

    init(_ args: Qwen3MoEConfiguration) {
        self.args = args
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { i in
                Qwen3MoeDecoderLayer(args, layerIdx: i)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        callAsFunction(inputs, cache: cache, raContexts: nil)
    }

    /// Sidecar retrieval-attention overload: pass a parallel list of
    /// `RetrievalAttentionContext?` aligned to `cache` so the dispatcher
    /// (`attentionWithCacheUpdate`) routes through the sparse path
    /// without needing a wrapper KV cache. `raContexts` defaults to nil;
    /// when nil this is identical to the legacy entry point. Mirrors
    /// the Qwen3 overload (F-83 sparse decode).
    func callAsFunction(
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
    func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let layerCaches = caches.map { $0[i] as KVCache? }
            h = layer.batchedForward(h, caches: layerCaches)
        }
        return norm(h)
    }

    /// Fully batched forward with shared per-layer `BatchedKVCache`.
    func fullyBatchedForward(
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
    /// Mirrors `Qwen3ModelInner.fullyBatchedSparseForward`.
    func fullyBatchedSparseForward(
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

public class Qwen3MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3MoEModelInner
    let configuration: Qwen3MoEConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3MoEModelInner(args)

        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
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
    /// the Qwen3Model overload (F-83 sparse decode).
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

    /// Batched decode: B requests with per-request per-layer caches.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var out = model.batchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// Fully batched decode with shared per-layer `BatchedKVCache`.
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

    /// F-85 — batched sparse decode. Pairs with
    /// `Qwen3MoEModelInner.fullyBatchedSparseForward`. ONE batched forward
    /// call per token, per-layer attention routes through the F-73 batched
    /// mask kernel (or F-71b via `VSM_SPARSE_BATCHED_KERNEL=f71b`) for
    /// sparse-eligible layers. vllm-swift's `vsm_engine_decode_all` calls
    /// here when sparse + B>1 sessions exist AND `VSM_SPARSE_BATCHED=1`.
    /// MoE expert routing is untouched — sparse only modifies attention.
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
        var sanitizedWeights = weights

        if configuration.tieWordEmbeddings {
            sanitizedWeights["lm_head.weight"] = nil
        }

        if sanitizedWeights["model.layers.0.mlp.experts.0.up_proj.weight"] == nil {
            return sanitizedWeights
        }

        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"
            for n in ["up_proj", "down_proj", "gate_proj"] {
                if sanitizedWeights["\(prefix).mlp.experts.0.\(n).weight"] != nil {
                    let toJoin = (0 ..< configuration.numExperts).map { e in
                        sanitizedWeights.removeValue(
                            forKey: "\(prefix).mlp.experts.\(e).\(n).weight")!
                    }
                    sanitizedWeights["\(prefix).mlp.switch_mlp.\(n).weight"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitizedWeights
    }

    /// TriAttention V3 cache factory. Mirrors `Qwen3Model.newCache` —
    /// Qwen3MoE shares the same dense-attention shape (q_norm/k_norm +
    /// RoPE, single `attentionHeads`/`kvHeads`/`headDim` tuple across all
    /// layers); the MoE FFN sub-layer is orthogonal to KV-cache layout
    /// so V3 install is a direct mirror of the Qwen3 dense factory.
    /// V3 stays OFF by default and is incompatible with caller-supplied
    /// `maxKVSize` (matches sibling family behavior).
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = configuration.hiddenLayers
        let env = ProcessInfo.processInfo.environment
        let enabled = env["VLLM_TRIATT_ENABLED"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

        if enabled, parameters?.maxKVSize == nil {
            let engine = TriAttentionV3Engine(
                cfg: .fromEnv(),
                nLayers: configuration.hiddenLayers,
                nHeads: configuration.attentionHeads,
                nKVHeads: configuration.kvHeads,
                headDim: configuration.headDim,
                ropeTheta: configuration.ropeTheta
            )
            TriAttentionRescue.shared.install(on: engine)
            return (0..<numLayers).map { layerIdx in
                TriAttentionKVCache(layerIdx: layerIdx, engine: engine)
            }
        }

        return (0..<numLayers).map { _ in
            makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
        }
    }
}

public struct Qwen3MoEConfiguration: Codable, Sendable {
    var modelType: String = "qwen3_moe"
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var numExperts: Int
    var numExpertsPerToken: Int
    var decoderSparseStep: Int
    var mlpOnlyLayers: [Int]
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDim: Int
    var ropeTheta: Float = 1_000_000
    var tieWordEmbeddings: Bool = false
    var maxPositionEmbeddings: Int = 32768
    var normTopkProb: Bool = false
    var ropeScaling: [String: StringOrNumber]? = nil

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normTopkProb = "norm_topk_prob"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_moe"
        self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        self.numExperts = try container.decode(Int.self, forKey: .numExperts)
        self.numExpertsPerToken = try container.decode(Int.self, forKey: .numExpertsPerToken)
        self.decoderSparseStep = try container.decode(Int.self, forKey: .decoderSparseStep)
        self.mlpOnlyLayers = try container.decode([Int].self, forKey: .mlpOnlyLayers)
        self.moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decode(Int.self, forKey: .headDim)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        self.normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

// MARK: - LoRA

extension Qwen3MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
