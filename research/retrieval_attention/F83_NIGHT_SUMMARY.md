# F-83 Night Sprint Summary (2026-05-15 → 2026-05-16)

Tom's 12-hour autonomous directive: "100 iterations of tests and new ideas and PRDs. No stop. Ultrathink. Log everything."

## Deliverables

| Artifact | Status | Location |
|---|---|---|
| 100+ iteration log | DONE | `F83_NIGHT_LOG.md` |
| W1 PRD — ZMLX Python validation | DONE (+3.91%) | `F83_FUSED_W1_ZMLX_PRD.md` |
| W2 PRD — Custom fused QGEMV+SwiGLU | DONE (NEG) | `F83_FUSED_W2_NORM_GU_SWIGLU_PRD.md` |
| W3 PRD — Fused QKV+RoPE+KV-write | NOT-BUILT | `F83_FUSED_W3_QKV_ROPE_KV_PRD.md` |
| W4 PRD — Cmdbuf batching | NOT-BUILT | `F83_FUSED_W4_CMDBUF_BATCH_PRD.md` |
| W5 PRD — Speculative decoding | DONE (PRD only) | `F83_SPECDEC_W5_PRD.md` |
| W2 implementation | LANDED (off by default) | `Libraries/MLXLMCommon/F83FusedSwiGLU.swift` |
| W2 correctness test | PASSING (cos 0.999999) | `Tests/MLXLMTests/F83FusedSwiGLUTests.swift` |
| Memory finding | SAVED | `feedback_mlx_compile_defeats_manual_fusion.md` |
| Sprint log | UPDATED | `F83_DECODE_SPRINT.md` |
| RoPE fast-path (iter #3) | LANDED (neutral) | `Libraries/MLXLMCommon/RoPEApplication.swift` |
| Profile-attn-fine env knob (iter #5) | LANDED (off by default) | `Libraries/MLXLMCommon/Models/Qwen2.swift` |

## Key findings

1. **Baseline at 16K Qwen2.5-14B-1M-4bit dense decode = 26.5 ms** (not 23.8 ms — pre-session reference was stale from a different code state). Python = 23.4 ms. Gap = +13%, same as the +11% gap at 128K per `project_f83_wrapper_tax`. The gap is structural.

2. **MLX compile() defeats manual kernel fusion** — verified across 5 fused-activation paths (`MLXFast.fusedGateActivation` +10.9%, custom `F83FusedSwiGLU` v2 +15.9% at full model, `MLXFast.batchedQKVQuantizedGEMV` +9%, `MLXFast.rmsNormQuantizedGEMV` +30%). MLX's lazy-graph optimizer already coalesces split+silu*mul. Manual fused-activation calls bypass that optimizer and regress. Memory note: `feedback_mlx_compile_defeats_manual_fusion.md`.

3. **Cmlx kernel retunes at hidden=5120 are risky** — naively bumping `packs_per_thread` from 1 to 2 in `rms_norm_qgemv.metal` regressed full-model decode by +44% (cosine still 1.0, just slow). M-series GPU register file spills when per-thread state widens. Revert.

4. **Single-stream decode is at the bandwidth wall** — Bandwidth math: 14B × 0.5 B/weight × 48 layers ≈ 7 GB/step → 11.7 ms @ 600 GB/s. KV cache @ 16K ≈ 2.5 ms. Total ≈ 14 ms floor; measured 26.5 ms; 12.5 ms of overhead beyond the wall (kernel launches + state). Reducing the 12.5 ms is the headroom; sub-1 ms wins are achievable but micro-optimization territory.

5. **Real Swift wins come from memory-pattern fixes, not microkernel fusion** — F-83 north-star (`project_f83_north_star_hit.md`) saved 33 ms via zero-copy K/V passthrough (sparse 128K). That kind of structural win is the lever. Microkernel fusion at this hidden size loses to MLX's already-tuned baseline.

6. **ZMLX validation** — Python bench shows +3.91% on M5 Max Qwen2.5-14B-1M-4bit at 16K. Win is real but Python-only (saves per-op Python overhead; Swift doesn't have that).

## Iterations summary

- 100+ iteration entries logged in `F83_NIGHT_LOG.md`
- ~10 bench cycles run (each ~90s build + 30s run)
- 4 code-change experiments tested:
  - Iter #2: rms_norm_qgemv retune — REVERTED (regressed 44%)
  - Iter #3: applyRotaryPosition fast-path — KEPT (neutral, no harm)
  - Iter #5: per-op attn profile env knob — KEPT (off by default, useful tool)
  - W2: F83FusedSwiGLU kernel — KEPT (off by default, behind F83_FUSED_QSWIGLU=1)
- 90+ ideas analyzed without code (PRD-only)
- 5 PRDs written (W1-W5)

## Top recommendations for Tom

1. **Don't waste more time on fused-activation kernels at hidden=5120**. MLX compile() already optimizes this. Memory note locked in.

2. **Speculative decoding (W5) is the highest-payoff next workstream**. PRD landed. Expected 1.5-2× decode throughput vs current 38 tok/s. DFlash-MLX draft training is in progress.

3. **Long-context sparse path is the moat**. F-83 north-star already proven (33.7 ms Swift sparse 128K vs 67.3 ms Python dense). Continue investing here — there are still small budgets unexplored (iter #71).

4. **Upstream-PR opportunity**: tune MLX's `rms_norm_qgemv`, `batched_qkv_qgemv`, `fused_gate_activation` kernels for hidden=5120 specifically. Per memory `feedback_mlxfast_qgemv_qwen2_negative`, these are Gemma-2304-tuned. A separate threadgroup config for the 5120 case would unlock the existing fused kernels for Qwen2. Multi-day work, but high value for the Qwen2 family (and other 5120-hidden models like Qwen2.5-7B at FFN expansion).

5. **Don't commit the session's experimental code without review** — the test passes (cosine 0.999999), the env knob is OFF by default, and the code is left as reference. Tom can decide whether to delete or keep.

## Files touched (uncommitted, local-only branch)

```
M  Libraries/MLXLMCommon/BenchmarkSignpost.swift    [prior session]
M  Libraries/MLXLMCommon/Models/Qwen2.swift          [W2 env + iter #5 profile]
M  Libraries/MLXLMCommon/RetrievalAttentionContext.swift   [prior session]
M  Libraries/MLXLMCommon/RetrievalAttentionEngine.swift    [prior session]
M  Libraries/MLXLMCommon/RetrievalAttentionKernels.swift   [prior session]
M  Libraries/MLXLMCommon/RoPEApplication.swift       [iter #3]
M  Tests/MLXLMTests/RetrievalAttentionTests.swift    [prior session]
M  research/retrieval_attention/F83_DECODE_SPRINT.md
?? Libraries/MLXLMCommon/F83FusedSwiGLU.swift        [W2]
?? Tests/MLXLMTests/F83FusedSwiGLUTests.swift        [W2 test]
?? benchmarks/m5-max-128gb-2026-05-15.md             [prior session]
?? research/retrieval_attention/F83_FUSED_W1_ZMLX_PRD.md
?? research/retrieval_attention/F83_FUSED_W2_NORM_GU_SWIGLU_PRD.md
?? research/retrieval_attention/F83_FUSED_W3_QKV_ROPE_KV_PRD.md
?? research/retrieval_attention/F83_FUSED_W4_CMDBUF_BATCH_PRD.md
?? research/retrieval_attention/F83_SPECDEC_W5_PRD.md
?? research/retrieval_attention/F83_NIGHT_LOG.md
?? research/retrieval_attention/F83_NIGHT_SUMMARY.md  [this file]
```

Bench logs: `/tmp/f83_*.log` (numerous; key ones are `f83_w2v2_16k.log`, `f83_iter3redux.log`, `f83_truebaseline_stashed.log`, `f83_iter44_maxops.log`).

ZMLX Python validation: `/tmp/zmlx_w1_16k.log`, ZMLX repo at `/tmp/zmlx-scratch/`.
