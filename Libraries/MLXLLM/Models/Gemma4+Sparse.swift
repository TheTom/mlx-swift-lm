// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the mlx-swift-lm project
//
// Batched sparse decode hooks for Gemma 4. Gemma 4 specifics handled here:
//   - Per-layer sliding / global attention mix via `layerTypes`. Each layer
//     owns its own RoPE (sliding uses local rope theta; global uses the
//     scaled rope theta + maxPositionEmbeddings).
//   - Sliding layers and global layers have DIFFERENT head dimensions and
//     KV-head counts (global uses `globalKvHeads` / `globalHeadDim`). The
//     per-layer `BatchedRetrievalAttentionKVCache` list must match each
//     layer's actual K shape — the caller is responsible.
//   - v_norm: Gemma 4 applies a value RMSNorm with no learned weight; we
//     replicate via `MLXFast.rmsNorm(values, weight: .mlxNone, eps: ...)`.
//   - KV-shared layers (e2b: trailing `num_kv_shared_layers` layers reuse
//     the donor layer's K/V) are NOT supported on the sparse decode path
//     in this initial port — the caller must pre-condition with
//     `numKvSharedLayers == 0`. The conformance precondition surfaces this
//     so a future spec can wire shared-layer routing through the cache list.
//   - attentionKEqV (some non-sliding layers re-use K as V — no v_proj):
//     handled by reading `vProj` optionality at runtime.
//   - Embeddings are pre-scaled by sqrt(hiddenSize).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

extension Gemma4Attention {

    /// Batched sparse forward for a single decode step on a Gemma 4 attention
    /// layer (sliding OR global). RoPE is selected per layer at init, so the
    /// same call site handles both attention kinds.
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let cache = raCache.inner

        // Use the non-fused norm + RoPE path on the sparse decode site to keep
        // the math straightforward; the fused-kernel optimisation lives on the
        // existing dense path and is orthogonal to the sparse-attention swap.
        var queries = qProj(x).reshaped(B, L, nHeads, -1)
        var keys = kProj(x).reshaped(B, L, nKVHeads, -1)
        var values: MLXArray
        if attentionKEqV {
            values = keys
        } else {
            values = vProj!(x).reshaped(B, L, nKVHeads, -1)
        }
        // v_norm has no learned weight — keep parity with the dense path.
        values = MLXFast.rmsNorm(values, weight: MLXArray.mlxNone, eps: rmsNormEps)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let allSameOffset = cache.offsets[0 ..< cache.active]
            .allSatisfy { $0 == cache.offsets[0] }
        let preUpdateK: MLXArray
        if allSameOffset {
            let offset = cache.offsets[0]
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            preUpdateK = keys
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
            preUpdateK = keys
            cache.update(newKeys: keys, newValues: values)
        }

        raCache.updateIndex(newKeys: preUpdateK)

        let output: MLXArray
        if L == 1 && raCache.isSparseEligible {
            output = raCache.sparseAttend(queries: queries, scale: scale)
        } else {
            let (k, v, mask) = cache.getCachedWithMask()
            output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: k, values: v,
                scale: scale, mask: .array(mask))
        }
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

extension Gemma4TransformerBlock {

    /// Sparse forward over Gemma 4's pre/post norm + rmsNormResidual fusion.
    /// MoE / PLE / per-layer scalar are preserved exactly as in the dense
    /// `callAsFunction`. KV-shared layers are not supported on this path —
    /// caller must skip them (see ModelInner precondition).
    public func fullyBatchedSparseForward(
        _ x: MLXArray,
        raCache: BatchedRetrievalAttentionKVCache,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        let inputNorm = inputLayerNorm(x)
        let attnOut = selfAttention.fullyBatchedSparseForward(
            inputNorm, raCache: raCache)
        var h = MLXFast.rmsNormResidual(
            attnOut, residual: x,
            weight: postAttentionLayerNorm.weight,
            eps: postAttentionLayerNorm.eps)

        // FFN — shared MLP path only. The MoE expert-routing code lives in
        // the dense `callAsFunction` and uses a fileprivate top-k helper
        // (`gemma4TopK`) that is not available from this extension file.
        // Sparse decode on MoE Gemma 4 checkpoints would need the same
        // routing wired in; until then this extension preconditions on
        // non-MoE configs (synthetic smoke uses dense FFN).
        precondition(experts == nil && router == nil,
            "Gemma4 batched sparse decode does not yet support MoE blocks")
        let preFFNNorm = preFeedforwardLayerNorm(h)
        let ffnOut = sharedMLP(preFFNNorm)
        h = MLXFast.rmsNormResidual(
            ffnOut, residual: h,
            weight: postFeedforwardLayerNorm.weight,
            eps: postFeedforwardLayerNorm.eps)

        // Per-Layer Embeddings (PLE) gate + projection.
        if let gate = perLayerInputGate,
            let proj = perLayerProjection,
            let norm = postPerLayerInputNorm,
            let pli = perLayerInput
        {
            let residual = h
            var g = compiledGeluMulSparse(gate(h), pli)
            g = proj(g)
            g = norm(g)
            h = residual + g
        }

        return h * layerScalar
    }
}

/// Local copy of the compiled fused gelu*mul used by Gemma 4's PLE gate.
/// The shared dense-path one in Gemma4.swift is fileprivate, so we keep a
/// dedicated compiled closure here to avoid touching the source file.
private let compiledGeluMulSparse: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { gate, x in
        geluApproximate(gate) * x
    }

extension Gemma4ModelInner {

    /// Per-layer sparse forward. Caller is responsible for:
    ///   - Building one `BatchedRetrievalAttentionKVCache` per layer whose
    ///     `inner.keys` shape matches the layer's K (sliding vs global K-shape).
    ///   - Configuring `numKvSharedLayers == 0` (KV-shared layers are not
    ///     wired on this path yet — they would need shared-K routing through
    ///     the cache list, which is a future spec).
    public func fullyBatchedSparseForward(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        precondition(raCaches.count == layers.count,
            "raCaches count (\(raCaches.count)) must match layers count (\(layers.count))")
        precondition(config.numKvSharedLayers == 0,
            "Gemma4 batched sparse decode does not yet support KV-shared layers")

        var h = embedTokens(inputs)
        // sqrt(hiddenSize) embedding scale.
        h = h * sqrt(Float(config.hiddenSize))

        // Per-Layer Embeddings (PLE) is computed exactly as the dense path,
        // but only when the model declares it. The synthetic smoke config
        // leaves `hiddenSizePerLayerInput == 0` which skips the entire block.
        var perLayerInputs: MLXArray? = nil
        if hiddenSizePerLayerInput > 0, let embedPL = embedTokensPerLayer {
            var pli = embedPL(inputs) * embedTokensPerLayerScale
            pli = pli.reshaped(
                pli.dim(0), pli.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)
            if let proj = perLayerModelProjection {
                var plProj = proj(h) * perLayerProjectionScale
                plProj = plProj.reshaped(
                    plProj.dim(0), plProj.dim(1), config.hiddenLayers, hiddenSizePerLayerInput)
                if let norm = perLayerProjectionNorm {
                    plProj = norm(plProj)
                }
                pli = (plProj + pli) * perLayerInputScale
            }
            perLayerInputs = pli
        }

        for (i, layer) in layers.enumerated() {
            let pli = perLayerInputs.map { $0[0..., 0..., i, 0...] }
            h = layer.fullyBatchedSparseForward(
                h, raCache: raCaches[i], perLayerInput: pli)
        }
        return norm(h)
    }
}

extension Gemma4TextModel: BatchedSparseLLM {

    public func fullyBatchedSparseDecode(
        _ inputs: MLXArray,
        raCaches: [BatchedRetrievalAttentionKVCache]
    ) -> MLXArray {
        var out = model.fullyBatchedSparseForward(inputs, raCaches: raCaches)
        if config.tieWordEmbeddings {
            out = model.embedTokens.asLinear(out)
        } else {
            out = lmHead!(out)
        }
        // Final logit softcapping — only when configured (matches dense path).
        if let softcap = config.finalLogitSoftcapping, softcap > 0 {
            out = compiledLogitSoftcapSparse(MLXArray(softcap), out)
        }
        return out
    }
}

/// Local copy of the compiled logit softcap kernel (fileprivate on the dense
/// path; duplicated here to keep this extension additive).
private let compiledLogitSoftcapSparse: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { softcap, x in
        tanh(x / softcap) * softcap
    }
