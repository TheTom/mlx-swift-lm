// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/llama.py

class LlamaAttention: Module {

    let args: LlamaConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: LlamaConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.resolvedHeadDimensions
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta,
            traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

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
    /// Mirrors Qwen2.Attention.batchedForward. Llama uses standard
    /// q/k/v_proj (bias optional via args.attentionBias), no q/k norm,
    /// RoPELayer (Llama3RoPE / RoPE) that conforms to `RoPELayer`.
    public func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

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

        var queries = wq(x).reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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

    /// F-85 — batched sparse forward for Llama. Mirrors
    /// `Qwen2.Attention.fullyBatchedSparseForward` (MLXLMCommon/Models/
    /// Qwen2.swift:365-421). Llama has no q/k RMSNorm (closer to Qwen2
    /// than Qwen3) so the projection → reshape → RoPE → cache update
    /// ordering matches Qwen2 exactly. Routes through
    /// `BatchedRetrievalAttentionKVCache.sparseAttend` (F-73 batched
    /// mask kernel by default) on sparse-eligible layers at L=1, else
    /// falls back to dense `cache.attention`.
    ///
    /// Caveman: like fullyBatchedForward but L=1 sparse layers go to
    /// F-73 mask kernel. K/V update happen via inner.update either way.
    public func fullyBatchedSparseForward(
        _ x: MLXArray, raCache: BatchedRetrievalAttentionKVCache,
        mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // Batched Q/K/V projections (bias optional via args.attentionBias).
        var queries = wq(x).reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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
            // here for safety.
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

class LlamaMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ args: LlamaConfiguration) {
        self._gate.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        self._down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
        self._up.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let activation = silu(gate(x))
        return down(activation * up(x))
    }
}

class LlamaTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: LlamaAttention
    @ModuleInfo(key: "mlp") var mlp: LlamaMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: LlamaConfiguration) {
        self._attention.wrappedValue = LlamaAttention(args)
        self._mlp.wrappedValue = LlamaMLP(args)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
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

public class LlamaModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [LlamaTransformerBlock]
    let norm: RMSNorm

    init(_ args: LlamaConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers).map { _ in LlamaTransformerBlock(args) }
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
    /// the Qwen2/Qwen3 overload (F-83 sparse decode).
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?,
        raContexts: [RetrievalAttentionContext?]?
    ) -> MLXArray {
        var h = embedTokens(inputs)

        // Auto-upcast for bf16 unquantized weights:
        //
        // MLX Swift's Linear path on Apple Metal accumulates bf16 matmuls
        // with limited intermediate precision, producing sparse NaN at
        // matmul outputs for HF unquantized checkpoints (Llama-3.2-hf,
        // Mistral-7B-v0.3-hf, etc.). The NaN appears first in `down_proj`
        // of MLP layer 0 and cascades through every subsequent layer.
        //
        // Python `mlx-lm` runs the same models bf16 and gets correct output
        // — implying the Swift backend's matmul kernel differs from the
        // Python one (likely accumulation dtype). Until that's resolved at
        // the MLX kernel level, upcast the hidden stream to fp32 whenever
        // weights are bf16. ~30% slower decode, ~2× hidden-state memory,
        // but the model produces correct output.
        //
        // MLX-quantized checkpoints (4-bit/6-bit with f16 scales) are
        // unaffected — Linear's quant path uses correct fp32 accumulation.
        let needsFP32Upcast = (h.dtype == .bfloat16)
        let originalDtype = h.dtype
        if needsFP32Upcast { h = h.asType(.float32) }

        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], raContext: raContexts?[i])
        }

        var normed = norm(h)
        if needsFP32Upcast { normed = normed.asType(originalDtype) }
        return normed
    }

    /// Batched forward: B requests with separate per-layer caches.
    /// caches: [[KVCache]] — outer per-request, inner per-layer.
    public func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var h = embedTokens(inputs)
        for (i, layer) in layers.enumerated() {
            let layerCaches = caches.map { $0[i] as KVCache? }
            h = layer.batchedForward(h, caches: layerCaches)
        }
        return norm(h)
    }

    /// Fully batched forward: shared per-layer `BatchedKVCache`. The mask is
    /// built once from the (post-update) offsets and reused across all layers.
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
    /// Mirrors `Qwen2.ModelInner.fullyBatchedSparseForward`.
    public func fullyBatchedSparseForward(
        _ inputs: MLXArray, raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var h = embedTokens(inputs)

        // Auto-upcast for bf16 unquantized weights (matches dense path).
        let needsFP32Upcast = (h.dtype == .bfloat16)
        let originalDtype = h.dtype
        if needsFP32Upcast { h = h.asType(.float32) }

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
        var normed = norm(h)
        if needsFP32Upcast { normed = normed.asType(originalDtype) }
        return normed
    }
}

/// Model for Llama and Mistral model types.
public class LlamaModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: LlamaModelInner
    let configuration: LlamaConfiguration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LlamaConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = LlamaModelInner(args)
        if !args.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
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
    /// the Qwen2Model / Qwen3Model overload (F-83 sparse decode).
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?,
        raContexts: [RetrievalAttentionContext?]?
    ) -> MLXArray {
        let out = model(inputs, cache: cache, raContexts: raContexts)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// Batched decode: B requests with per-request per-layer caches.
    /// inputs: [B, 1] token IDs. caches: B arrays of per-layer KVCache.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        let out = model.batchedForward(inputs, caches: caches)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// Fully batched decode with shared per-layer `BatchedKVCache`.
    public func fullyBatchedDecode(
        _ inputs: MLXArray, caches: [BatchedKVCache]
    ) -> MLXArray {
        let out = model.fullyBatchedForward(inputs, caches: caches)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    /// F-85 — batched sparse decode. Pairs with `LlamaModelInner.
    /// fullyBatchedSparseForward`. ONE batched forward call per token,
    /// per-layer attention routes through the F-73 batched mask kernel
    /// (or F-71b via `VSM_SPARSE_BATCHED_KERNEL=f71b`) for sparse-eligible
    /// layers. vllm-swift's `vsm_engine_decode_all` calls here when
    /// sparse + B>1 sessions exist AND `VSM_SPARSE_BATCHED=1`.
    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray, raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        let out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if let lmHead {
            return lmHead(out)
        } else {
            return model.embedTokens.asLinear(out)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Remove unused precomputed rotary frequencies
        weights.filter {
            !$0.key.contains("self_attn.rotary_emb.inv_freq")
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = configuration.hiddenLayers
        let env = ProcessInfo.processInfo.environment
        let enabled = env["VLLM_TRIATT_ENABLED"].map {
            ["1", "true", "yes", "on"].contains($0.lowercased())
        } ?? false

        // TriAttention V3 — KV-cache eviction policy. Mirrors the Qwen2
        // factory at MLXLLM/Models/Qwen2.swift and the Qwen3 factory at
        // MLXLLM/Models/Qwen3.swift. V3 owns the full cache list (one
        // TriAttentionKVCache per layer) and is incompatible with a
        // caller-supplied maxKVSize (which would route to the eviction-
        // windowed StandardKVCache variant). LlamaConfiguration has a
        // `headDimensions` codable field (`head_dim` in HF config) — when
        // absent (older Llama-2 / Mistral-v0.1 checkpoints) it falls
        // back to hiddenSize / attentionHeads via `resolvedHeadDimensions`,
        // matching LlamaAttention.init.
        if enabled, parameters?.maxKVSize == nil {
            let engine = TriAttentionV3Engine(
                cfg: .fromEnv(),
                nLayers: configuration.hiddenLayers,
                nHeads: configuration.attentionHeads,
                nKVHeads: configuration.kvHeads,
                headDim: configuration.resolvedHeadDimensions,
                ropeTheta: configuration.ropeTheta
            )
            TriAttentionRescue.shared.install(on: engine)
            return (0..<numLayers).map { layerIdx in
                TriAttentionKVCache(layerIdx: layerIdx, engine: engine)
            }
        }

        // Default path — route through `makeAttentionCache` so caller-
        // supplied `maxKVSize` picks the eviction-windowed variant.
        // Matches the Qwen2/Qwen3 factory behavior + the
        // `KVCacheDimensionProvider` extension default in
        // MLXLMCommon/LanguageModel.swift.
        return (0..<numLayers).map { _ in
            makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
        }
    }

    public func messageGenerator(tokenizer: any Tokenizer) -> any MessageGenerator {
        // some models allow the system role and some do not -- this is enforced
        // by the chat template (code).
        do {
            let probe = [
                [
                    "role": "system",
                    "content": "test",
                ]
            ]
            _ = try tokenizer.applyChatTemplate(messages: probe)
            return DefaultMessageGenerator()
        } catch {
            return NoSystemMessageGenerator()
        }
    }
}

public struct LlamaConfiguration: Codable, Sendable {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool = true
    var attentionBias: Bool = false
    var mlpBias: Bool = false

    public init(
        hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int, attentionHeads: Int,
        headDimensions: Int? = nil, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        maxPositionEmbeddings: Int? = nil, ropeTheta: Float = 10_000, ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil, tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false, mlpBias: Bool = false
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
    }

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
        {
            self.ropeTraditional = ropeTraditional
        }
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }

        if let ropeScaling {
            if ropeScaling["factor"] == nil {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain 'factor'")
            }
            if let ropeType = ropeScaling["type"] ?? ropeScaling["rope_type"] {
                if case .string = ropeType {
                    let options = [
                        StringOrNumber.string("linear"), StringOrNumber.string("dynamic"),
                        StringOrNumber.string("llama3"),
                    ]
                    if !options.contains(ropeType) {
                        throw DecodingError.dataCorruptedError(
                            forKey: .ropeScaling, in: container,
                            debugDescription:
                                "rope_scaling 'type' currently only supports 'linear', 'dynamic', or 'llama3'"
                        )
                    }
                }
            } else {
                throw DecodingError.dataCorruptedError(
                    forKey: .ropeScaling, in: container,
                    debugDescription: "rope_scaling must contain either 'type' or 'rope_type'")
            }
        }
    }
}

// MARK: - LoRA

extension LlamaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
