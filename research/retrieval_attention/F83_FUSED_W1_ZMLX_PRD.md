# F-83 Fused Block W1 — ZMLX MLX Patch Validation

**Status**: in flight (started 2026-05-15)
**Branch**: `feature/retrieval-attention` (local-only — do NOT push)
**Owner**: autonomous Claude run
**Target model**: Qwen2.5-14B-Instruct-1M-4bit on M5 Max 128GB
**Goal**: Validate ZMLX's claimed +7.5% decode win on our exact model and establish a measured ceiling for Tier 1 of the fused-block sprint.

---

## Why this first

Cheapest signal. Hmbown/ZMLX (https://github.com/Hmbown/ZMLX) claims a 2-line patch over upstream `ml-explore/mlx` that fuses `swiglu_mlp` and delivers +7.5% decode on Qwen3.5-9B-4bit on M4 Max. Before we invest 1-2 weeks porting zinc's custom kernel for Qwen2.5-14B hidden=5120, we want a sanity datapoint on our exact stack:

1. Does ZMLX's fusion apply cleanly to mlx-swift's bundled `Cmlx/mlx` (we're on TheTom/mlx:vllm-swift-stable, not upstream main)?
2. Does the win generalize from M4 Max → M5 Max?
3. Does it generalize from Qwen3.5-9B → Qwen2.5-14B (different hidden dim, different head count)?

If ZMLX delivers measurable improvement, W2 (custom kernel) has a known floor. If not, we know the easy path is exhausted and W2 needs to do all the work.

## Acceptance criteria

- Apply ZMLX patches to `/Users/tom/dev/mlx-swift/Source/Cmlx/mlx`. Build clean.
- Bench Qwen2.5-14B-1M-4bit decode at three points: 16K, 32K, 128K (existing F-83 perf harness).
- Compare median tokens/sec over ≥10 runs each, paired vs current `feature/retrieval-attention` baseline.
- **Ship signal**: ≥3% decode improvement at any context length without quality regression.
- **Quality**: spot-check that top-1 logits match baseline (≥0.99 cosine on 128 random tokens) at 32K. If quality drops, abort merge.

## Implementation plan

1. Clone ZMLX into a scratch dir, identify the patch (grep for `swiglu` / `mlp` changes in commit history).
2. Apply diff to `Cmlx/mlx` (vendored submodule). Document the exact files touched in this PRD's "Outcome" section.
3. Rebuild metallib (`scripts/build-metallib.sh release`) and Cmlx target.
4. Run existing benchmark harness in release mode at 16K / 32K / 128K, 10 iters per point.
5. Record results in `benchmarks/m5-max-128gb-2026-05-15.md` under "W1-ZMLX" header.
6. Sanity quality check at 32K with `F83_PYTHON_PARITY=1`.

## Risk

- ZMLX may target upstream MLX APIs that drifted in our fork.
- M5 Max's GPU layout (vs M4 Max) may erase or invert the win.
- The fusion may collide with our F-83 retrieval-attention dispatcher (post-norm hook).

## Outcome (2026-05-15)

**Validated: +3.91% decode speedup** on M5 Max for Qwen2.5-14B-Instruct-1M-4bit at 16K context.

- Baseline (stock mlx_lm): 441.3 ms/generate() call (incl 32-tok decode + prefill).
- ZMLX-patched: 424.1 ms/generate() call → 17.3 ms saved per call ≈ 4% net.
- Per-step decode delta from EVAL-PROFILE: ~22.5 ms baseline → ~22.0 ms patched (~2-4% measured).
- Hardware path: Python `mlx_lm` 0.31.2 + `zmlx` 0.9.0 via `zmlx.patch.patch(model)`. Patched 56 modules per ZMLX log.

**Implication for Swift port (W2)**: ceiling is ~4% on this exact model (not the 7.5% claimed for Qwen3.5-9B). Worth pursuing but with realistic expectations. The kernel-level mechanic is "fuse split + silu*mul + (sometimes) dequant" — ZMLX's `_quantized_swiglu_gemv_kernel` is the reference algorithm. Bench log: `/tmp/zmlx_w1_16k.log`.

**Did NOT validate at 32K / 128K** — single 16K datapoint is enough to greenlight W2.
