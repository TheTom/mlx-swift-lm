//
//  Qwen3.swift
//  LLM
//
//  Created by John Mai on 2025/4/28.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py

class Qwen3Attention: Module {
    let args: Qwen3Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    // Inverse frequencies for fused RMSNorm + RoPE (MLXFast.rmsNormRoPE
    // framework kernel). Computed at init from `ropeTheta` and `headDim`.
    // Mirrors `Gemma4Attention._fusedInvFreqs` (sliding-layer branch:
    // standard RoPE with `1.0 / theta^(2i/D)` for `i in 0..<headDim/2`).
    // Underscore prefix prevents Module weight loading from looking for
    // this key in the checkpoint. Only built when the layer's RoPE is a
    // plain (non-scaled) variant — linear-scaled RoPE would need its
    // factor baked into the freqs, which is left for a follow-up.
    let _fusedInvFreqs: MLXArray?

    /// Set QWEN3_FUSED_NORM_ROPE=0 to disable for A/B testing.
    private static let useFusedNormRoPE: Bool = {
        ProcessInfo.processInfo.environment["QWEN3_FUSED_NORM_ROPE"] != "0"
    }()

    public init(_ args: Qwen3Configuration) {
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
        let isLinearScaled: Bool
        if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
            let factor = ropeScaling["factor"]
        {
            if let v = factor.asFloat() {
                ropeScale = 1 / v
                isLinearScaled = true
            } else {
                fatalError("ropeScaling.factor must be a float")
            }
        } else {
            ropeScale = 1
            isLinearScaled = false
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeTheta,
            scale: ropeScale)

        // Build inverse frequencies for the fused norm+RoPE kernel. Skip
        // when scaled RoPE is in play (the kernel does not take a scale
        // arg, and we want bit-identical fallback to the plain rope() path
        // for those rare scaled-rope checkpoints).
        if Self.useFusedNormRoPE && !isLinearScaled {
            let exponents = MLXArray(
                stride(from: Float(0), to: Float(headDim), by: 2)
            ) / Float(headDim)
            let freqs = pow(MLXArray(args.ropeTheta), exponents)
            self._fusedInvFreqs = 1.0 / freqs
        } else {
            self._fusedInvFreqs = nil
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        let triCache = cache as? TriAttentionKVCache
        if B == 1, let triCache {
            // V3 calibration consumes pre-RoPE Q shaped [tokens, heads, dim].
            // Qwen3 has queries as [B, heads, tokens, dim] at this point.
            let qForCalibration = queries[0].transposed(1, 0, 2).asType(.float32)
            triCache.engine.accumulateQ(qForCalibration, layerIdx: triCache.layerIdx)
        }

        // TriAttention physically compacts K/V storage. Keep RoPE position
        // tied to the original logical token stream, not the compacted
        // storage length (`cache.offset`).
        let rotaryOffset = triCache?.logicalOffset ?? (cache?.offset ?? 0)
        queries = rope(queries, offset: rotaryOffset)
        keys = rope(keys, offset: rotaryOffset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask,
            raContext: raContext
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }

    /// Batched attention: B requests with separate KV caches.
    /// Projections batched, per-request RoPE + attention + cache update.
    public func batchedForward(
        _ x: MLXArray, caches: [KVCache?]
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Batched projections: single matmul for all B
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        // Split into per-request slices once (avoid repeated indexing)
        let qSlices = split(queries, parts: B, axis: 0)
        let kSlices = split(keys, parts: B, axis: 0)
        let vSlices = split(values, parts: B, axis: 0)

        // Per-request: RoPE (different offsets) + cache update
        // Then batched SDPA if all caches are same length, else per-request
        var allSameLen = true
        var firstLen = -1

        var rotQ = [MLXArray]()
        var allKeys = [MLXArray]()
        var allVals = [MLXArray]()
        rotQ.reserveCapacity(B)
        allKeys.reserveCapacity(B)
        allVals.reserveCapacity(B)

        for i in 0..<B {
            let cache_i = caches[i]
            let offset = cache_i?.offset ?? 0
            let q_rot = rope(qSlices[i], offset: offset)
            let k_rot = rope(kSlices[i], offset: offset)

            let (aK, aV) = cache_i?.update(keys: k_rot, values: vSlices[i])
                ?? (k_rot, vSlices[i])

            rotQ.append(q_rot)
            allKeys.append(aK)
            allVals.append(aV)

            let sLen = aK.dim(2)
            if firstLen < 0 { firstLen = sLen }
            if sLen != firstLen { allSameLen = false }
        }

        let output: MLXArray
        if allSameLen && B > 1 {
            // Fast path: all caches same length → single batched SDPA
            // Batched SDPA: B=\(B) seq=\(firstLen)
            let bQ = concatenated(rotQ, axis: 0)      // [B, heads, 1, dim]
            let bK = concatenated(allKeys, axis: 0)    // [B, kvHeads, seq, dim]
            let bV = concatenated(allVals, axis: 0)    // [B, kvHeads, seq, dim]

            output = MLXFast.scaledDotProductAttention(
                queries: bQ, keys: bK, values: bV,
                scale: scale, mask: .none
            )
        } else {
            // Slow path: different lengths → per-request SDPA
            var outputs = [MLXArray]()
            outputs.reserveCapacity(B)
            for i in 0..<B {
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

    /// Fully batched forward with BatchedKVCache — ZERO per-request loops.
    /// Mask is pre-computed once and shared across all layers.
    public func fullyBatchedForward(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
        mask: MLXArray
    ) -> MLXArray {
        fullyBatchedForwardImpl(
            x, cache: cache, layerIndex: layerIndex,
            mask: mask, allSameOffset: nil
        )
    }

    /// Optimized variant: caller pre-computed `allSameOffset` (typically
    /// once per step in `Qwen3ModelInner.fullyBatchedForward`) so we skip
    /// the per-layer `allSatisfy` over `cache.offsets[0..<active]`. When
    /// `mask` is `nil` the caller has guaranteed all slots have identical
    /// offsets and the cache covers exactly those positions — we then
    /// dispatch SDPA with `maskMode: .none`, skipping an entire mask-tensor
    /// read per layer. At small Qwen3 (0.6B / 4B / dense) this saves both
    /// CPU op-encode (allocations + allSatisfy) and GPU bandwidth (zero-
    /// mask reads × 28-36 layers). See alpha-side perf notes in
    /// `research/retrieval_attention/F83_*`.
    public func fullyBatchedForwardFast(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
        mask: MLXArray?, allSameOffset: Bool
    ) -> MLXArray {
        fullyBatchedForwardImpl(
            x, cache: cache, layerIndex: layerIndex,
            mask: mask, allSameOffset: allSameOffset
        )
    }

    @inline(__always)
    private func fullyBatchedForwardImpl(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
        mask: MLXArray?, allSameOffset precomputed: Bool?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Batched projections — three Linear dispatches share the same `x`.
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // RoPE + cache update — use caller's pre-computed allSameOffset
        // when available, else fall back to per-layer scan.
        let allSameOffset: Bool
        if let precomputed {
            allSameOffset = precomputed
        } else {
            allSameOffset = cache.offsets[0..<cache.active]
                .allSatisfy { $0 == cache.offsets[0] }
        }

        // Fused norm+RoPE fast path. Mirrors the win agent #117 landed for
        // Gemma4 (`Gemma4Attention.fullyBatchedForward`, commit ff30de2):
        // `qNorm + reshape + transpose + rope` is 4 dispatches per Q/K.
        // `MLXFast.rmsNormRoPE` collapses that into a single kernel,
        // shaving ~6 dispatches per attention layer at B=64. Eligible when
        // all slots share the cache offset AND the layer built its
        // inverse-frequency table at init (plain RoPE, no linear scaling).
        // V still goes through plain reshape+transpose — no norm, no RoPE.
        if let invFreqs = _fusedInvFreqs, allSameOffset {
            let offset = cache.offsets[0]
            // rmsNormRoPE takes input shaped `[B, L, nHeads, headDim]`
            // (pre-transpose) and returns the same shape.
            queries = queries.reshaped(B, L, args.attentionHeads, -1)
            queries = MLXFast.rmsNormRoPE(
                queries, weight: qNorm.weight, invFreqs: invFreqs,
                eps: args.rmsNormEps, offset: offset,
                nHeads: args.attentionHeads, seqLen: L)
            queries = queries.transposed(0, 2, 1, 3)

            keys = keys.reshaped(B, L, args.kvHeads, -1)
            keys = MLXFast.rmsNormRoPE(
                keys, weight: kNorm.weight, invFreqs: invFreqs,
                eps: args.rmsNormEps, offset: offset,
                nHeads: args.kvHeads, seqLen: L)
            keys = keys.transposed(0, 2, 1, 3)

            values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
            cache.updateFast(newKeys: keys, newValues: values, allSameOffset: true)

            // No-mask fast path: caller guaranteed every cached position
            // is valid after the update above.
            let output: MLXArray
            if mask == nil {
                output = cache.attention(queries: queries, scale: scale, maskMode: .none)
            } else {
                output = cache.attention(queries: queries, scale: scale, mask: mask!)
            }
            return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }

        // Slow path (mixed offsets or fused-norm-rope disabled): keep
        // bit-identical behaviour with the unfused norm + rope sequence.
        queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            cache.updateFast(newKeys: keys, newValues: values, allSameOffset: true)
        } else {
            // Mixed offsets: per-request RoPE then cache update
            let qSlices = split(queries, parts: B, axis: 0)
            let kSlices = split(keys, parts: B, axis: 0)
            var rotQ = [MLXArray]()
            var rotK = [MLXArray]()
            for i in 0..<B {
                let off = cache.offsets[i]
                rotQ.append(rope(qSlices[i], offset: off))
                rotK.append(rope(kSlices[i], offset: off))
            }
            queries = concatenated(rotQ, axis: 0)
            keys = concatenated(rotK, axis: 0)
            cache.updateFast(newKeys: keys, newValues: values, allSameOffset: false)
        }

        // No-mask fast path: skip the zero-mask read in SDPA when caller
        // confirmed every cached position is valid (allSameOffset + cache
        // covers exactly those positions after the update above).
        let output: MLXArray
        if mask == nil {
            output = cache.attention(queries: queries, scale: scale, maskMode: .none)
        } else {
            output = cache.attention(queries: queries, scale: scale, mask: mask!)
        }

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }

    /// F-85 — batched sparse forward for Qwen3. Mirrors
    /// `Qwen2.Attention.fullyBatchedSparseForward` (MLXLMCommon/Models/
    /// Qwen2.swift:365-421) but applies Qwen3's q/k RMSNorm BEFORE RoPE
    /// — matching Qwen3's standard attention ordering. Routes through
    /// `BatchedRetrievalAttentionKVCache.sparseAttend` (F-73 batched
    /// mask kernel by default) on sparse-eligible layers at L=1, else
    /// falls back to dense `cache.attention`.
    ///
    /// Caveman: like fullyBatchedForward but qNorm+kNorm first, then
    /// RoPE, then route L=1 sparse to F-73 mask kernel.
    public func fullyBatchedSparseForward(
        _ x: MLXArray, raCache: BatchedRetrievalAttentionKVCache,
        mask: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // qNorm + kNorm applied per-head BEFORE RoPE (matches Qwen3's
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

// `Qwen3MLP` is now an alias for the shared `Qwen3.MLP` in MLXLMCommon.
// Issue #168 consolidation pass — the SwiGLU MLP is bit-identical between
// the Qwen 3 LLM and Qwen3VL.
typealias Qwen3MLP = Qwen3.MLP

class Qwen3TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen3Attention
    let mlp: Qwen3MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        _attention.wrappedValue = Qwen3Attention(args)
        self.mlp = Qwen3MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        raContext: RetrievalAttentionContext? = nil
    ) -> MLXArray {
        var r = attention(inputLayerNorm(x), mask: mask, cache: cache, raContext: raContext)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }

    /// Fully batched forward with shared BatchedKVCache — zero loops.
    public func fullyBatchedForward(_ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
                                     mask: MLXArray) -> MLXArray {
        let normed = inputLayerNorm(x)
        var r = attention.fullyBatchedForward(normed, cache: cache, layerIndex: layerIndex,
                                              mask: mask)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }

    /// Fast variant — caller passes pre-computed `allSameOffset` and an
    /// optional `mask` (nil means "no mask, all positions valid"). See
    /// `Qwen3Attention.fullyBatchedForwardFast`. Used by the small Qwen3
    /// dense models (0.6B / 4B) where per-step CPU op-encode dominates.
    public func fullyBatchedForwardFast(
        _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
        mask: MLXArray?, allSameOffset: Bool
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        var r = attention.fullyBatchedForwardFast(
            normed, cache: cache, layerIndex: layerIndex,
            mask: mask, allSameOffset: allSameOffset
        )
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }

    /// Batched forward: B requests, batched norms + MLP, per-request attention.
    public func batchedForward(_ x: MLXArray, caches: [KVCache?]) -> MLXArray {
        let normed = inputLayerNorm(x)                // [B, 1, hidden] batched
        var r = attention.batchedForward(normed, caches: caches)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))            // [B, 1, hidden] batched
        return h + r
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
        var r = attention.fullyBatchedSparseForward(
            normed, raCache: raCache, mask: mask)
        let h = x + r
        r = mlp(postAttentionLayerNorm(h))
        return h + r
    }
}

public class Qwen3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [Qwen3TransformerBlock]
    let norm: RMSNorm

    public init(_ args: Qwen3Configuration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                Qwen3TransformerBlock(args)
            }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        callAsFunction(inputs, cache: cache, raContexts: nil)
    }

    /// Sidecar retrieval-attention overload: pass a parallel list of
    /// `RetrievalAttentionContext?` aligned to `cache` so the dispatcher
    /// (`attentionWithCacheUpdate`) routes through the sparse path
    /// without needing a wrapper KV cache. `raContexts` defaults to nil;
    /// when nil this is identical to the legacy entry point. Mirrors
    /// the Qwen2 overload (F-83 sparse decode).
    public func callAsFunction(
        _ inputs: MLXArray, cache: [KVCache]?,
        raContexts: [RetrievalAttentionContext?]?
    ) -> MLXArray {
        var h = embedTokens(inputs)

        // Auto-upcast for bf16 unquantized weights to dodge MLX Swift's
        // Metal Linear bf16-accumulation bug (sparse NaN at down_proj).
        // See Llama.swift LlamaModelInner for full rationale.
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

    /// Fully batched forward: shared per-layer BatchedKVCaches.
    ///
    /// Hot-path opts (matter most at small dense Qwen3 — 0.6B / 4B —
    /// where per-step CPU op-encode dominates step time):
    ///
    /// * `allSame` is computed ONCE here and threaded into every layer's
    ///   `fullyBatchedForwardFast` so each attention layer skips its own
    ///   `cache.offsets[0..<active].allSatisfy { ... }` scan + closure
    ///   alloc. At 28-36 layers × N decode steps/sec this is real time.
    /// * When `allSame == true`, every cached position is valid post-
    ///   update — we hand each SDPA `mask: nil` so the cache attention
    ///   call uses `MLXFast.SDPA(..., mask: .none)` instead of reading a
    ///   `[B,1,1,T]` zero tensor. Saves both the mask-tensor alloc
    ///   (`MLXArray.zeros(...)`) and 28-36 zero-mask GPU bandwidth reads.
    public func fullyBatchedForward(_ inputs: MLXArray, caches: [BatchedKVCache]) -> MLXArray {
        var h = embedTokens(inputs)

        // Pull cache0 / offsets buffer once — avoids the per-layer
        // closure capture of `caches[0]` and the repeated subscript.
        let cache0 = caches[0]
        let active = cache0.active

        // When all requests have same offset (continuous decode),
        // every cached position is valid → no mask needed.
        // For mixed offsets, build a per-request lower-triangular mask.
        let first = cache0.offsets[0]
        var allSame = true
        var maxOff = first
        for i in 1..<active {
            let o = cache0.offsets[i]
            if o != first { allSame = false }
            if o > maxOff { maxOff = o }
        }
        let maxPostOffset = maxOff + 1

        if allSame {
            // No-mask fast path: hand each layer `mask: nil` so SDPA runs
            // with `maskMode: .none`. Saves one MLXArray.zeros alloc and
            // 28-36 zero-mask reads per step.
            for (i, layer) in layers.enumerated() {
                h = layer.fullyBatchedForwardFast(
                    h, cache: caches[i], layerIndex: i,
                    mask: nil, allSameOffset: true
                )
            }
        } else {
            let cacheDtype = cache0.keys.dtype
            let positions = MLXArray(0..<maxPostOffset).reshaped(1, maxPostOffset)
            let offsetsArr = MLXArray(cache0.offsets[0..<active].map { $0 + 1 })
                .reshaped(active, 1)
            let valid = positions .< offsetsArr
            let mask = MLX.where(valid,
                                 MLXArray(Float(0)).asType(cacheDtype),
                                 MLXArray(Float(-1e9)).asType(cacheDtype))
                .reshaped(active, 1, 1, maxPostOffset)

            for (i, layer) in layers.enumerated() {
                h = layer.fullyBatchedForwardFast(
                    h, cache: caches[i], layerIndex: i,
                    mask: mask, allSameOffset: false
                )
            }
        }
        return norm(h)
    }

    /// Batched forward: B requests with separate per-layer caches.
    /// caches: [[KVCache]] — outer is per-request, inner is per-layer.
    public func batchedForward(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var h = embedTokens(inputs)  // [B, 1, hidden] batched

        for (i, layer) in layers.enumerated() {
            let layerCaches = caches.map { $0[i] as KVCache? }
            h = layer.batchedForward(h, caches: layerCaches)
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

public class Qwen3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen3ModelInner
    let configuration: Qwen3Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: Qwen3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen3ModelInner(args)

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
    /// the Qwen2Model overload (F-83 sparse decode).
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

    /// Fully batched decode: zero per-request loops.
    public func fullyBatchedDecode(_ inputs: MLXArray, caches: [BatchedKVCache]) -> MLXArray {
        var out = model.fullyBatchedForward(inputs, caches: caches)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        return out
    }

    /// F-85 — batched sparse decode. Pairs with
    /// `Qwen3ModelInner.fullyBatchedSparseForward`. ONE batched forward
    /// call per token, per-layer attention routes through the F-73 batched
    /// mask kernel (or F-71b via `VSM_SPARSE_BATCHED_KERNEL=f71b`) for
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

    /// Batched decode: B requests, batched projections + MLP, per-request attention.
    /// inputs: [B, 1] token IDs. caches: B arrays of per-layer KVCache.
    /// Returns: [B, 1, vocab] logits.
    public func batchedDecode(_ inputs: MLXArray, caches: [[KVCache]]) -> MLXArray {
        var out = model.batchedForward(inputs, caches: caches)
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

        // Fuse `mlp.gate_proj` + `mlp.up_proj` into `mlp.gate_up_proj` for
        // the shared `Qwen3.MLP` (output dim doubled, rows concatenated on
        // axis 0). Mirrors `Gemma4` PR #66 — saves one Metal dispatch per
        // MLP per step. Covers `.weight`, `.scales`, and `.biases` keys
        // produced by the quantization pipeline. No-op when the checkpoint
        // already ships the fused key (re-load of a Swift-saved model).
        fuseGateUpWeights(&weights, keyFilter: ".mlp.gate_proj.", outputAxis: 0)

        return weights
    }

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

        // Eric's spec-006 cleanup: factories use `makeAttentionCache`
        // which routes to StandardKVCache (or eviction-windowed variant)
        // based on parameters + maxSize, instead of instantiating
        // KVCacheSimple/RotatingKVCache directly.
        return (0..<numLayers).map { _ in
            makeAttentionCache(parameters: parameters, maxSize: parameters?.maxKVSize)
        }
    }
}

public struct Qwen3Configuration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 1_000_000
    var headDim: Int
    var ropeScaling: [String: StringOrNumber]? = nil
    var tieWordEmbeddings = false
    var maxPositionEmbeddings: Int = 32768

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
            try decoder.container(
                keyedBy: Qwen3Configuration.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
        self.rmsNormEps = try container.decode(
            Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
            ?? 1_000_000
        self.headDim = try container.decode(
            Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    }
}

// MARK: - LoRA

extension Qwen3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
