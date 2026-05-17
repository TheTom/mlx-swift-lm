# F-85 v2 — Milestone 1: F-73 mask kernel B>1 audit

Date: 2026-05-16
Branch: `feature/retrieval-attention` (local only)

## Source under audit

`Libraries/MLXLMCommon/RetrievalAttentionKernels.swift` —
- `_RAKernelCache.getBuildMask()` (kernel source) — lines 922–996
- `retrievalAttentionBuildMaskFused(...)` (Swift wrapper) — lines 1437–1480

## Current contract (B=1)

Inputs:
- `fine_starts: [NKVH * K_FINE]` — int32, flat per-head top-K block starts.
- `coarse_starts: [NKVH * K_COARSE]` — int32, flat.
- `params: [T, staticInit, slidingWindow]` — float32, length 3.

Template constants: `FINE_BS, COARSE_BS, NKVH, K_FINE, K_COARSE`.

Grid: `((T + tg-1)/tg, 1, 1)` with `(tg=256, 1, 1)` threads/threadgroup.
- Threadgroup memory: `int fine_tg[NKVH * K_FINE]`, `int coarse_tg[NKVH * K_COARSE]`.
- Cooperative load of fine_starts / coarse_starts into shared memory once
  per threadgroup, then bounds check + membership scan per thread.

Output: `mask: [1, 1, 1, T]` float32 (-inf at masked positions, 0 at valid).
Caller reshapes / casts to model dtype.

## Verdict for B>1

The kernel is NOT B-aware. Two paths to B-batch:

### Path A — kernel-level B (preferred)

Change source to:
```
// Grid: ((T + tg-1)/tg, B, 1) × (tg, 1, 1).
const uint p = thread_position_in_grid.x;
const uint b = threadgroup_position_in_grid.y;
...
// Per-slot reads:
fine_tg[i] = fine_starts[b * NKVH * K_FINE + i];
...
mask[b * T + p] = valid ? 0.0f : -INFINITY;
```

Pros:
- Single kernel launch for B masks.
- Threadgroup memory unchanged (each TG still owns one slot's top-K).
- Total work scales linearly with B (correct).

Cons:
- New kernel under a new name (don't break F-73 single-batch callers).
- Need a fresh wrapper that takes `fine_starts: [B, NKVH, K_FINE]` and
  emits `[B, 1, 1, T]`.

### Path B — Swift loop over existing kernel

```
var masks: [MLXArray] = []
for b in 0..<B {
  let m = retrievalAttentionBuildMaskFused(
    fineStarts: fineStartsAll[b], coarseStarts: coarseStartsAll[b], ...)
  masks.append(m)  // [1, 1, 1, T]
}
let stacked = concatenated(masks, axis: 0)  // [B, 1, 1, T]
```

Pros: zero kernel work.
Cons: B kernel launches + B alloc + Swift overhead per step. At B=8 that's
8 launches per layer × 48 layers ≈ 384 mask-build dispatches per token.
At ~50µs/launch worst case, 19ms of pure launch overhead — could be 1-2ms
in practice once MLX queues them.

## Decision

Implement BOTH:
- **Path A** as new public helper
  `retrievalAttentionBuildMaskFusedBatched(fineStarts: [B,nKVH,K_fine], ...)`
  with new Metal kernel name `ra_build_mask_b` (template constants
  identical, grid adds a B dim).
- **Path B** kept as Swift-side fallback for shape-mismatch / debug only.

`BatchedRetrievalAttentionKVCache.sparseAttend` will:
1. Compute per-slot `[B, nKVH, K_fine]` and `[B, nKVH, K_coarse]` block starts.
2. Call `retrievalAttentionBuildMaskFusedBatched` to get `[B, 1, 1, T]`.
3. Call `MLXFast.scaledDotProductAttention(q, k, v, scale, mask: .array(mask))`.

Env knob `VSM_SPARSE_BATCHED_KERNEL`:
- `f73` (default) — Path A batched mask + MLXFast SDPA.
- `f73loop` — Path B Swift loop fallback (for A/B sanity).
- `f71b` — original v1 fused kernel (for A/B regression check).

## Risks

- The kernel-level B path threadgroup memory bound: `NKVH * K_FINE * sizeof(int)`
  is unchanged per TG (each TG still owns one slot). Default config NKVH=8
  K_FINE=8 → 256 bytes — fine.
- `MLXFast.scaledDotProductAttention` with `mask: .array(...)` and a `[B,1,1,T]`
  mask broadcasts naturally across the nQH and L axes — already exercised
  by existing F-73 / F-74 / F-75 / F-78 paths (just at B=1).
- Quality: this milestone is plumbing-only — selector still picks
  placeholder blocks 0..K-1. M3 swaps in real selector population.

## Out of scope for M1

- Implementing the kernel.
- Bench.
