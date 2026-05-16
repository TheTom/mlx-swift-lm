// Copyright © 2026 Apple Inc.
//
// Shared Qwen 3 building blocks. The Qwen 3 family has substantive
// architectural divergence between LLM and VLM that prevents a full
// layer-stack consolidation in this PR:
//
// - LLM `Qwen3Attention` uses standard `applyRotaryPosition(rope, ...)`
//   and ships alpha-side perf work (`batchedForward`,
//   `fullyBatchedForward` for batched inference).
// - VLM `Qwen3VL` Attention uses M-RoPE (multimodal rotary): takes a
//   `positionIds` array, computes `(cos, sin)` via a custom
//   `RotaryEmbedding` class, and applies via
//   `Qwen3VLLanguage.applyMultimodalRotary(q:k:cos:sin:)`.
//
// Forcing a single shared Attention would either gut the LLM's batched
// forward optimisations or require a closure/protocol-based dispatch
// that costs an indirection on the LLM hot path. Instead, this
// namespace lifts the small set of pieces that ARE genuinely shareable
// (configuration adapter + the SwiGLU MLP) and leaves Attention /
// DecoderLayer / ModelInner per-target.
//
// Future work (aligned with `IMPLEMENTATION-PLAN.md`):
//
// - **M-RoPE shared helper** — once a second VLM family adopts the
//   same M-RoPE pattern (Qwen3VL is currently the only one in tree
//   that needs it; Qwen 2.5 VL still uses standard rope), lift the
//   `applyMultimodalRotary` + position-id construction to a shared
//   helper and have both VLMs consume it.
// - **Issue #115 (QKV batched fusion)** — when this lands, the
//   LLM's `batchedForward` paths simplify to a single fused matmul,
//   reducing the per-target divergence enough to enable a fuller
//   Attention consolidation.
//
// Consolidation reference: issue #168.

import Foundation
import MLX
import MLXNN

/// Public namespace for the Qwen 3 text decoder. The layer-stack
/// classes are intentionally NOT included here; see file-level note.
public enum Qwen3 {

    // MARK: - LayerArgs (adapter)

    /// Minimum field set the shared MLP needs from any of the consuming
    /// configurations. Each consumer's config provides a `var layerArgs:
    /// Qwen3.LayerArgs` accessor.
    public struct LayerArgs: Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let rmsNormEps: Float

        public init(hiddenSize: Int, intermediateSize: Int, rmsNormEps: Float) {
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.rmsNormEps = rmsNormEps
        }
    }

    // MARK: - MLP

    /// SwiGLU MLP — gate_proj / up_proj / down_proj with silu(gate) * up.
    /// Bit-identical between the LLM and VLM Qwen 3 implementations.
    ///
    /// At init time we fuse `gate_proj` and `up_proj` into a single
    /// `gate_up_proj` Linear with output dim `2 * intermediate`, matching
    /// the win agent #117 landed for Gemma4 (PR #66). Weight fusion
    /// happens in the model's `sanitize()` via `fuseGateUpWeights(...)` —
    /// when the checkpoint still ships the unfused `gate_proj`/`up_proj`
    /// rows we concat them on axis 0 and rename to `gate_up_proj`. When
    /// the fused key already exists (re-load of a Swift-saved checkpoint)
    /// no work is done.
    ///
    /// Saves one Metal dispatch per layer per step vs. the prior
    /// `gate(x) + up(x)` pair. At Qwen3-0.6B (28 layers) and B=64 decode
    /// this is the next ~3% win once the fused norm+RoPE landed.
    public class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        let hiddenDims: Int

        public init(dimensions: Int, hiddenDimensions: Int) {
            self.hiddenDims = hiddenDimensions
            self._gateUp.wrappedValue = Linear(
                dimensions, 2 * hiddenDimensions, bias: false)
            self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            super.init()
        }

        /// Fused silu(gate) * up — matches mlx-lm's `@partial(mx.compile, shapeless=True)`
        /// swiglu in mlx_lm/models/activations.py. Cuts the activation +
        /// elementwise multiply from two kernel dispatches per layer to one.
        /// Per-decoder-block savings show up at small models (Qwen3-0.6B/4B)
        /// where CPU op-encode dominates step time.
        private static let compiledSwiglu: @Sendable (MLXArray, MLXArray) -> MLXArray =
            compile(shapeless: true) { gate, up in
                silu(gate) * up
            }

        /// Set QWEN3_FUSED_GATE_ACT=0 to disable the C-kernel fused
        /// split+silu*mul. Memory note `feedback_mlxfast_qgemv_qwen2_negative`
        /// shows the kernel regresses at Qwen2 hidden=5120/27648 (kernel
        /// is tuned for Gemma4 E2B's 2304). At Qwen3 small/dense (0.6B
        /// intermediate=3072, 4B intermediate=9728) the post-projection
        /// hidden-dim is closer to the kernel's sweet spot and the
        /// dispatch saving lands net-positive.
        private static let useFusedGateAct: Bool = {
            ProcessInfo.processInfo.environment["QWEN3_FUSED_GATE_ACT"] != "0"
        }()

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gateUpOut = gateUp(x)
            if Self.useFusedGateAct {
                let activated = MLXFast.fusedGateActivation(
                    gateUpOut, hiddenDims: hiddenDims, activation: .silu)
                return down(activated)
            }
            let parts = split(gateUpOut, parts: 2, axis: -1)
            return down(MLP.compiledSwiglu(parts[0], parts[1]))
        }
    }
}
