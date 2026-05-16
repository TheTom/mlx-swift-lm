# F-83 Fused Block W2 — RMSNorm + dual-QGEMV(gate+up) + SwiGLU custom kernel

**Status**: planned
**Branch**: `feature/retrieval-attention` (local-only — do NOT push)
**Target model**: Qwen2.5-14B-Instruct-1M-4bit on M5 Max 128GB
**Goal**: Single fused Metal kernel that does post-attention RMSNorm → dual 4-bit QGEMV (gate + up, sharing input) → split → SwiGLU, replacing 5 separate dispatches per decoder layer × 48 layers = 240 fewer dispatches per decode step.

---

## Why

Per agent research (ae396ccfa216b1aac, 2026-05-15):
- ZMLX measured +7.5% on Qwen3.5-9B-4bit via SwiGLU fusion in MLX (different abstraction layer).
- zinc's `dmmv_q4k_dense_gate_up_swiglu.metal` is the closest open kernel to what we need: Q4_K, NSG=2, 2 rows per SIMD, 64 threads/tg.
- bit-ml's fused gated-MLP writeup matches: 3-6% on dense single-stream decode.
- MLXFast.rmsNormQuantizedGEMV in our codebase REGRESSED at Qwen2 hidden=5120 (+7.4ms vs 24.5ms baseline) per memory `feedback_mlxfast_qgemv_qwen2_negative.md` — the kernel is tuned for Gemma's 2304. A custom kernel sized for 5120 should escape this.

Current Qwen2.swift MLP path (Libraries/MLXLMCommon/Models/Qwen2.swift:233-246):

```swift
public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let parts = MLX.split(gateUp(x), parts: 2, axis: -1)
    return down(Qwen2.compiledSwiglu(parts[0], parts[1]))
}
```

Combined with the preceding `postAttentionLayerNorm(x)` in DecoderLayer, the dispatch chain is:
1. RMSNorm
2. QuantizedLinear (gate_up_proj) — fused gate+up matmul into `[1,1,2*hidden]`
3. split → 2 arrays
4. silu(gate) * up (compiled)
5. QuantizedLinear (down_proj)

W2 collapses steps 1–4 into one kernel.

## Acceptance criteria

- Custom Metal kernel `fused_norm_gate_up_swiglu.metal` in `Source/Cmlx/mlx-generated/metal/` (or equivalent location).
- Swift wrapper `MLXFast.fusedNormGateUpSwiglu(...)` callable from `Qwen2.MLP`.
- Env-gated behind `F83_FUSED_NORM_GU_SWIGLU=1` (off by default until we ship).
- Bench harness: ≥5% decode improvement at 16K and 32K vs baseline on Qwen2.5-14B-1M-4bit.
- Quality: ≥0.999 cosine vs baseline at 32K (the prefix layer norms are exact, the matmul should be bit-identical, only float arithmetic order may shift slightly).

## Implementation plan

1. Read zinc's kernel (`dmmv_q4k_dense_gate_up_swiglu.metal`) for layout.
2. Adapt for: hidden=5120, intermediate=27648, group=64, bits=4, NSG=2.
3. Wire `MLXFast.fusedNormGateUpSwiglu(x, normWeight, gateUpW, gateUpScales, gateUpBiases, groupSize, eps, hiddenDim) -> MLXArray` returning `[B, L, hidden]` (the silu(gate)*up output, before down_proj).
4. Update `Qwen2.MLP.callAsFunction` and `DecoderLayer.callAsFunction` for the fused-norm + MLP path (need to defer postAttentionLayerNorm into the fused kernel).
5. Build, test, bench.

## Risk

- Kernel correctness at hidden=5120 is the hardest part. SIMD-group reductions for the norm need to handle 5120 features (40 SIMDs × 128 lanes).
- The MLX kernel-source API (`MLXFast.metalKernel`) limits what we can express. May need to drop into raw mlx-c Metal codegen instead.
- Quality must hit ≥0.999 cosine — float-order differences in the norm reduction can drift if not careful.

## Outcome

(TBD)
