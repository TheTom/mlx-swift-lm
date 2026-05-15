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
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: cache, scale: scale, mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)
            return wo(output)
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

        // Hidden dim is needed by the fused-gate-activation kernel — cached
        // once at init since it never changes.
        let hiddenDim: Int

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            // F-83 sprint iter #9 found that MLX.MLXFast.fusedGateActivation
            // didn't move the needle on Qwen2 16K decode (within noise vs
            // compiled-swiglu). The split+swiglu path appears to already be
            // fused by MLX's lazy graph through the compile() pattern.
            // Kept the env opt-in for future experimentation.
            if ProcessInfo.processInfo.environment["F83_FUSED_GATE_ACT"] == "1" {
                let gateUpOut = gateUp(x)
                let activated = MLX.MLXFast.fusedGateActivation(
                    gateUpOut, hiddenDims: hiddenDim, activation: .silu)
                return down(activated)
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
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            // F83_PROFILE_DECODE=1 — force eval per phase to attribute GPU
            // time. Breaks lazy fusion, so absolute numbers are inflated
            // vs steady-state, but the RELATIVE breakdown shows where
            // dispatches go. Single-layer profile from layer 0 first
            // iteration is enough to isolate hot phases.
            if ProcessInfo.processInfo.environment["F83_PROFILE_DECODE"] == "1" {
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
            let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
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
            inputEmbedding: MLXArray? = nil
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
                h = layer(h, mask: mask, cache: cache?[i])
            }
            return norm(h)
        }
    }
}
