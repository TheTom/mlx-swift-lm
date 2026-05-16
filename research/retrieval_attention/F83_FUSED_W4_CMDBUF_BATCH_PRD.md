# F-83 Fused Block W4 — Per-decoder-block MTLCommandBuffer batching

**Status**: planned (lowest priority — exploratory)
**Branch**: `feature/retrieval-attention` (local-only — do NOT push)
**Target model**: Qwen2.5-14B-Instruct-1M-4bit on M5 Max 128GB
**Goal**: Wrap each decoder layer in a single MTLCommandBuffer rather than letting MLX submit one per encoder stop. Eliminates per-encoder CPU sync overhead.

---

## Why

Per agent ae396:
- MetalRT (1.10–1.19× faster than mlx-lm, proprietary) and BoltzmannEntropy/metalQwen3 both reject monolithic-layer kernels in favor of "Command batching: Multiple GPU operations per command buffer."
- MLX's lazy eval already coalesces most of this, but the boundaries between encoded operations still pay driver overhead.

This is the cheapest *and* riskiest tier: cheap because no kernel writing, risky because we may be duplicating MLX's existing graph eval and gain nothing.

## Acceptance criteria

- Xcode GPU capture demonstrates command-buffer reduction per decode step (target: 1 command buffer per layer, vs current ~3-5).
- Bench: ≥1% decode improvement at 16K. If <1%, document as no-op and move on.
- No quality regression (this is purely a submission-batching change).

## Implementation plan

1. Profile current decode step in Xcode GPU capture. Count command buffers per step.
2. If MLX already submits ≤1 per layer, abandon (no headroom).
3. Otherwise: expose `MLXFast.beginBatchedDispatch()` / `endBatchedDispatch()` that the decoder layer wraps around its call.
4. Bench, decide ship.

## Risk

- MLX lazy eval may make this a no-op.
- Manual command-buffer batching can deadlock if a graph forces eval mid-batch.
- Lowest expected ROI — defer to last in the 12hr autonomous run.

## Outcome

(TBD)
