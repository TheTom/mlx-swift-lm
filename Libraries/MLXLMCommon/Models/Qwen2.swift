// Copyright © 2026 Apple Inc.
//
// Shared Qwen 2 text-decoder building blocks. Consumed by:
//  - MLXLLM/Models/Qwen2.swift
//  - MLXVLM/Models/Qwen2VL.swift
//  - MLXVLM/Models/Qwen25VL.swift
//  - MLXVLM/Models/FastVLM.swift
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Qwen 2 text decoder (consumed by Qwen 2 LLM
/// + Qwen2VL / Qwen25VL / FastVLM). Configs across these consumers
/// differ in shape, so the shared layer classes take a thin
/// `Qwen2.LayerArgs` adapter.
public enum Qwen2 {

    // MARK: - LayerArgs (adapter)

    /// Minimum field set the layer stack needs from any of the consuming
    /// configurations. Each consumer's config provides a `var layerArgs`
    /// computed accessor.
    public struct LayerArgs: Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let ropeScaling: [String: StringOrNumber]?
        /// VLM consumers may rename the rope module (e.g. Qwen2VL exposes it
        /// as `rotary_emb` in module-key path). Set this to override the
        /// default `"rope"` key. Most LLM and VLM consumers use the default.
        public let ropeModuleKey: String

        public init(
            hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
            attentionHeads: Int, kvHeads: Int, rmsNormEps: Float,
            ropeTheta: Float, ropeTraditional: Bool,
            ropeScaling: [String: StringOrNumber]?,
            ropeModuleKey: String = "rope"
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.ropeTraditional = ropeTraditional
            self.ropeScaling = ropeScaling
            self.ropeModuleKey = ropeModuleKey
        }
    }

    // MARK: - Attention

    public class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        let rope: RoPE

        public init(_ args: LayerArgs) {
            let dim = args.hiddenSize
            self.heads = args.attentionHeads
            self.kvHeads = args.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._wq.wrappedValue = Linear(dim, heads * headDim, bias: true)
            self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
            self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            // Optional linear-scaling RoPE.
            let ropeScale: Float
            if let ropeScaling = args.ropeScaling,
                ropeScaling["type"] == .string("linear"),
                let factor = ropeScaling["factor"]?.asFloat()
            {
                ropeScale = 1 / factor
            } else {
                ropeScale = 1
            }
            self.rope = RoPE(
                dimensions: headDim,
                traditional: args.ropeTraditional,
                base: args.ropeTheta,
                scale: ropeScale)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            raContext: RetrievalAttentionContext? = nil
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            // F-83 night iter #5: fine-grained per-op profile inside Attention.
            // Forces eval per op; absolute inflated, relative useful.
            if Qwen2.envProfileAttnFine {
                let t0 = CFAbsoluteTimeGetCurrent()
                let q0 = wq(x); eval(q0)
                let t1 = CFAbsoluteTimeGetCurrent()
                let k0 = wk(x); eval(k0)
                let t2 = CFAbsoluteTimeGetCurrent()
                let v0 = wv(x); eval(v0)
                let t3 = CFAbsoluteTimeGetCurrent()
                let qR = q0.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3); eval(qR)
                let kR = k0.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3); eval(kR)
                let vR = v0.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3); eval(vR)
                let t4 = CFAbsoluteTimeGetCurrent()
                let qRoPE = applyRotaryPosition(rope, to: qR, cache: cache); eval(qRoPE)
                let kRoPE = applyRotaryPosition(rope, to: kR, cache: cache); eval(kRoPE)
                let t5 = CFAbsoluteTimeGetCurrent()
                let attnOut = attentionWithCacheUpdate(
                    queries: qRoPE, keys: kRoPE, values: vR,
                    cache: cache, scale: scale, mask: mask,
                    raContext: raContext
                )
                eval(attnOut)
                let t6 = CFAbsoluteTimeGetCurrent()
                let oOut = wo(attnOut.transposed(0, 2, 1, 3).reshaped(B, L, -1)); eval(oOut)
                let t7 = CFAbsoluteTimeGetCurrent()
                let toMs = { (a: CFAbsoluteTime, b: CFAbsoluteTime) in (b - a) * 1000 }
                FileHandle.standardError.write(Data(
                    "[ATTN-FINE] q=\(String(format: "%.3f", toMs(t0,t1))) k=\(String(format: "%.3f", toMs(t1,t2))) v=\(String(format: "%.3f", toMs(t2,t3))) shape=\(String(format: "%.3f", toMs(t3,t4))) rope=\(String(format: "%.3f", toMs(t4,t5))) attn=\(String(format: "%.3f", toMs(t5,t6))) o=\(String(format: "%.3f", toMs(t6,t7))) total=\(String(format: "%.3f", toMs(t0,t7)))\n"
                        .utf8))
                return oOut
            }

            var queries: MLXArray
            var keys: MLXArray
            var values: MLXArray

            // F-83 fused batched-QKV path: L=1 decode, 4-bit quantized,
            // half/bfloat. Saves 2 dispatches/layer vs 3 separate
            // quantized matmuls. The kernel only does matmul — Linear's
            // additive bias (Qwen2 q/k/v_proj all have bias=true) is
            // applied after by adding wq.bias / wk.bias / wv.bias to the
            // split outputs. Falls through to legacy 3-call path otherwise.
            if Qwen2.envFusedQKV, L == 1, B == 1,
                let qq = wq as? QuantizedLinear,
                let kk = wk as? QuantizedLinear,
                let vv = wv as? QuantizedLinear,
                qq.bits == 4, kk.bits == 4, vv.bits == 4,
                qq.groupSize == kk.groupSize, kk.groupSize == vv.groupSize,
                let qB = qq.biases, let kB = kk.biases, let vB = vv.biases,
                (x.dtype == .float16 || x.dtype == .bfloat16)
            {
                let qkv = MLXFast.batchedQKVQuantizedGEMV(
                    x,
                    wQ: qq.weight, scalesQ: qq.scales, biasesQ: qB,
                    wK: kk.weight, scalesK: kk.scales, biasesK: kB,
                    wV: vv.weight, scalesV: vv.scales, biasesV: vB,
                    groupSize: qq.groupSize)
                // qkv shape: [1, 1, n_q + n_k + n_v] for B=L=1.
                let nQ = heads * headDim
                let nK = kvHeads * headDim
                queries = qkv[0..., 0..., 0 ..< nQ]
                keys = qkv[0..., 0..., nQ ..< (nQ + nK)]
                values = qkv[0..., 0..., (nQ + nK)...]
                if let qBias = qq.bias { queries = queries + qBias }
                if let kBias = kk.bias { keys = keys + kBias }
                if let vBias = vv.bias { values = values + vBias }
            } else {
                queries = wq(x)
                keys = wk(x)
                values = wv(x)
            }

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

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
        /// Projections + output proj are batched (one matmul each); RoPE +
        /// cache update + SDPA stay per-request (each cache has its own T).
        /// Ports the Qwen3 `batchedForward` pattern for the Qwen2 family so
        /// vllm-swift's `decode_all` gets weight-bandwidth amortization
        /// across concurrent requests instead of falling through to
        /// per-request sequential stepAsync. Pairs with `Qwen2Model.batchedDecode`.
        public func batchedForward(
            _ x: MLXArray, caches: [KVCache?]
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            // Batched Q/K/V projections (Linear includes bias).
            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            // Steady-state batched decode usually has all streams at the
            // same offset (all started together, advance 1 token / step).
            // When that holds, apply RoPE ONCE to the batched [B, heads, 1, dim]
            // tensor instead of B small per-request calls — saves the worst
            // overhead in the inner loop (50µs × B × num_layers).
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
                    qR = qSlices[i]  // already RoPE-applied
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

        /// Fully batched attention with `BatchedKVCache` — zero per-request
        /// loops in the common all-same-offset case. Single batched cache
        /// update, single batched SDPA against the shared cache. Mirrors
        /// `Qwen3Attention.fullyBatchedForward` (lines 188–233 of MLXLLM
        /// Qwen3.swift) but without the Qwen3 q/k-norm — Qwen2 has none.
        /// Pairs with `Qwen2Model.fullyBatchedDecode`.
        public func fullyBatchedForward(
            _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int,
            mask: MLXArray
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            // Batched Q/K/V projections (Linear includes bias=true).
            var queries = wq(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            // RoPE + cache update.
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
    }

    // MARK: - MLP

    /// Fused swiglu — silu(gate) * up as a single compiled Metal kernel.
    /// Mirrors `mlx-lm`'s `@partial(mx.compile, shapeless=True)` swiglu in
    /// `mlx_lm/models/activations.py`. At decode time on Qwen2-14B at 16K,
    /// the unfused form (silu(gate(x)) * up(x)) costs two kernel
    /// dispatches per layer (one for silu, one for elementwise mul) — at
    /// ~80-100 µs each per layer × 48 layers, that's ~8 ms/step. Fusing
    /// matches Python's path.
    private static let compiledSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray =
        compile(shapeless: true) { gate, up in
            silu(gate) * up
        }

    /// F-83 sprint iter #12 — cached env reads. `ProcessInfo.processInfo.
    /// environment[...]` measured at ~13 µs / call (codex review). Doing
    /// it twice per layer × 48 layers = 96 lookups/step ≈ 1.3 ms of
    /// avoidable Swift-side overhead. Hoist to static lets resolved once
    /// at process load.
    private static let envFusedGateAct: Bool =
        ProcessInfo.processInfo.environment["F83_FUSED_GATE_ACT"] == "1"
    private static let envProfileDecode: Bool =
        ProcessInfo.processInfo.environment["F83_PROFILE_DECODE"] == "1"
    /// F-83 night iter #5: fine-grained per-op profile inside Attention.
    /// Sums times across all 48 layers per step; prints one summary line
    /// per decode step. Forces eval() per op so absolute numbers are
    /// inflated; relative breakdown identifies the largest dispatch.
    static let envProfileAttnFine: Bool =
        ProcessInfo.processInfo.environment["F83_PROFILE_ATTN_FINE"] == "1"
    // F-83 fused kernels gate: default OFF until verified. Enable with
    // F83_FUSED_QKV=1 and F83_FUSED_NORM_GU=1.
    static let envFusedQKV: Bool =
        ProcessInfo.processInfo.environment["F83_FUSED_QKV"] == "1"
    static let envFusedNormGU: Bool =
        ProcessInfo.processInfo.environment["F83_FUSED_NORM_GU"] == "1"
    static let envCompiledMLP: Bool =
        ProcessInfo.processInfo.environment["F83_COMPILED_MLP"] == "1"
    /// F-83 W2: fused dual-QGEMV(gate+up) + SwiGLU custom kernel.
    /// Replaces split + compiled silu*mul tail with one Metal dispatch.
    static let envFusedQSwiGLU: Bool =
        ProcessInfo.processInfo.environment["F83_FUSED_QSWIGLU"] == "1"

    public class MLP: Module, UnaryLayer {
        // Fused gate+up projection. Two separate Linears (gate_proj +
        // up_proj) both consume the same input x and produce hidden-dim
        // outputs. Concatenating their weights along the output axis at
        // sanitize-time collapses two matmul dispatches into one, then
        // we split the result before swiglu. Saves ~48 dispatches per
        // decode step on Qwen2-14B (one per layer). The fused weight is
        // produced by `Qwen2Model.sanitize` when both gate_proj and
        // up_proj are present in the checkpoint.
        @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        public init(dimensions: Int, hiddenDimensions: Int) {
            self.hiddenDim = hiddenDimensions
            self._gateUp.wrappedValue = Linear(dimensions, 2 * hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            super.init()
        }

        let hiddenDim: Int

        // F-83: wrap swiglu + down in a compiled closure (Gemma4
        // pattern). Split stays outside — `Split.output_shapes`
        // can't infer under `shapeless: true`. Win is CPU-side
        // tape-replay cost.
        private lazy var compiledTail: @Sendable (MLXArray, MLXArray) -> MLXArray = {
            compile(inputs: [self], outputs: [], shapeless: true) { [self] gate, up in
                self.down(Qwen2.compiledSwiglu(gate, up))
            }
        }()

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            // F-83 W2: fused dual-QGEMV(gate+up) + SwiGLU. Only valid for
            // 4-bit quantized gate_up_proj. Falls through on any mismatch.
            if Qwen2.envFusedQSwiGLU,
                let q = gateUp as? QuantizedLinear,
                q.bits == 4,
                let biases = q.biases
            {
                let K = q.weight.dim(1) * (32 / q.bits)
                let activated = F83FusedSwiGLU.callAsFunction(
                    x: x,
                    gateUpWeight: q.weight,
                    gateUpScales: q.scales,
                    gateUpBiases: biases,
                    intermediate: hiddenDim,
                    hiddenIn: K,
                    groupSize: q.groupSize
                )
                return down(activated)
            }
            if Qwen2.envFusedGateAct {
                let gateUpOut = gateUp(x)
                let activated = MLX.MLXFast.fusedGateActivation(
                    gateUpOut, hiddenDims: hiddenDim, activation: .silu)
                return down(activated)
            }
            if Qwen2.envCompiledMLP {
                let parts = MLX.split(gateUp(x), parts: 2, axis: -1)
                return compiledTail(parts[0], parts[1])
            }
            let parts = MLX.split(gateUp(x), parts: 2, axis: -1)
            return down(Qwen2.compiledSwiglu(parts[0], parts[1]))
        }
    }

    /// Concatenate `mlp.gate_proj.*` and `mlp.up_proj.*` weights into
    /// `mlp.gate_up_proj.*` so the two matmuls collapse into one at
    /// runtime. Handles both fp16 unquantized (single `weight` key) and
    /// quantized (`weight`/`scales`/`biases`) layouts. Idempotent — if
    /// the checkpoint already has fused keys, leaves them alone.
    public static func fuseGateUpWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights
        // Find all gate_proj keys; each implies a peer up_proj.
        let gateKeys = weights.keys
            .filter { $0.contains(".mlp.gate_proj.") }
        for gateKey in gateKeys {
            let upKey = gateKey.replacingOccurrences(of: ".gate_proj.", with: ".up_proj.")
            let fusedKey = gateKey.replacingOccurrences(of: ".gate_proj.", with: ".gate_up_proj.")
            guard let gateArr = out[gateKey], let upArr = out[upKey] else { continue }
            // Concatenate along the output (row) axis = 0. Works for
            // `weight` (packed int4 [out, in/8]), `scales` and `biases`
            // ([out, in/group_size]).
            out[fusedKey] = concatenated([gateArr, upArr], axis: 0)
            out.removeValue(forKey: gateKey)
            out.removeValue(forKey: upKey)
        }
        return out
    }

    // MARK: - DecoderLayer

    public class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        public init(_ args: LayerArgs) {
            self._attention.wrappedValue = Attention(args)
            self.mlp = MLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            self._inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            self._postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
            raContext: RetrievalAttentionContext? = nil
        ) -> MLXArray {
            // F83_PROFILE_DECODE=1 — force eval per phase to attribute GPU
            // time. Breaks lazy fusion, so absolute numbers are inflated
            // vs steady-state, but the RELATIVE breakdown shows where
            // dispatches go. Single-layer profile from layer 0 first
            // iteration is enough to isolate hot phases.
            if Qwen2.envProfileDecode {
                let t0 = CFAbsoluteTimeGetCurrent()
                let n1 = inputLayerNorm(x); eval(n1)
                let t1 = CFAbsoluteTimeGetCurrent()
                let r = attention(n1, mask: mask, cache: cache); eval(r)
                let t2 = CFAbsoluteTimeGetCurrent()
                let h = x + r; eval(h)
                let t3 = CFAbsoluteTimeGetCurrent()
                let n2 = postAttentionLayerNorm(h); eval(n2)
                let t4 = CFAbsoluteTimeGetCurrent()
                let m = mlp(n2); eval(m)
                let t5 = CFAbsoluteTimeGetCurrent()
                let out = h + m; eval(out)
                let t6 = CFAbsoluteTimeGetCurrent()
                let toMs = { (a: CFAbsoluteTime, b: CFAbsoluteTime) in (b - a) * 1000 }
                FileHandle.standardError.write(Data(
                    "[QWEN2-PROFILE] in_norm=\(String(format: "%.3f", toMs(t0,t1)))ms attn=\(String(format: "%.3f", toMs(t1,t2)))ms res1=\(String(format: "%.3f", toMs(t2,t3)))ms post_norm=\(String(format: "%.3f", toMs(t3,t4)))ms mlp=\(String(format: "%.3f", toMs(t4,t5)))ms res2=\(String(format: "%.3f", toMs(t5,t6)))ms total=\(String(format: "%.3f", toMs(t0,t6)))ms\n"
                        .utf8))
                return out
            }
            let r = attention(inputLayerNorm(x), mask: mask, cache: cache, raContext: raContext)
            let h = x + r
            // F-83 fused post_norm + gate_up_proj: L=1 decode, quantized
            // gateUp (Qwen2 MLP has bias=false). Saves 1 dispatch/layer
            // (collapses RMSNorm + quantized matmul into one kernel).
            let L = h.dim(1)
            if Qwen2.envFusedNormGU, L == 1, h.dim(0) == 1,
                let qgu = mlp.gateUp as? QuantizedLinear,
                qgu.bits == 4, qgu.bias == nil,
                let qguBiases = qgu.biases,
                (h.dtype == .float16 || h.dtype == .bfloat16)
            {
                let fused = MLXFast.rmsNormQuantizedGEMV(
                    h,
                    normWeight: postAttentionLayerNorm.weight,
                    w: qgu.weight, scales: qgu.scales, biases: qguBiases,
                    eps: postAttentionLayerNorm.eps,
                    groupSize: qgu.groupSize)
                let parts = MLX.split(fused, parts: 2, axis: -1)
                let activated = Qwen2.compiledSwiglu(parts[0], parts[1])
                return h + mlp.down(activated)
            }
            return h + mlp(postAttentionLayerNorm(h))
        }

        /// Batched forward: batched norms + MLP, per-request attention via
        /// `Attention.batchedForward`. Pairs with `ModelInner.batchedForward`
        /// for vllm-swift concurrent decode.
        public func batchedForward(
            _ x: MLXArray, caches: [KVCache?]
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = attention.batchedForward(normed, caches: caches)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }

        /// Fully batched forward with shared `BatchedKVCache` — zero loops.
        public func fullyBatchedForward(
            _ x: MLXArray, cache: BatchedKVCache, layerIndex: Int, mask: MLXArray
        ) -> MLXArray {
            let normed = inputLayerNorm(x)
            let r = attention.fullyBatchedForward(
                normed, cache: cache, layerIndex: layerIndex, mask: mask)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    // MARK: - ModelInner

    /// Shared transformer backbone. `inputEmbedding` parameter supports the
    /// VLM vision-fusion path.
    public class ModelInner: Module {
        @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
        public let layers: [DecoderLayer]
        public let norm: RMSNorm

        public init(_ args: LayerArgs, vocabularySize: Int) {
            precondition(vocabularySize > 0)
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: args.hiddenSize)
            self.layers = (0 ..< args.hiddenLayers).map { _ in DecoderLayer(args) }
            self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        public func callAsFunction(
            _ inputs: MLXArray? = nil,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            raContexts: [RetrievalAttentionContext?]? = nil
        ) -> MLXArray {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("Qwen2.ModelInner requires `inputs` or `inputEmbedding`")
            }
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i], raContext: raContexts?[i])
            }
            return norm(h)
        }

        /// Batched forward: B requests with separate per-layer caches.
        /// `caches`: outer is per-request, inner is per-layer.
        /// Each layer fans the per-request caches into `DecoderLayer.batchedForward`.
        public func batchedForward(
            _ inputs: MLXArray, caches: [[KVCache]]
        ) -> MLXArray {
            var h = embedTokens(inputs)
            for (i, layer) in layers.enumerated() {
                let layerCaches = caches.map { $0[i] as KVCache? }
                h = layer.batchedForward(h, caches: layerCaches)
            }
            return norm(h)
        }

        /// Fully batched forward: shared per-layer `BatchedKVCache`. The
        /// mask is pre-built once from the (post-update) offsets and reused
        /// across all layers. Mirrors `Qwen3ModelInner.fullyBatchedForward`.
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
    }
}
