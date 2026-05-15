# F-83 V2 Design — Sparse Prefill Optimization

**Status**: Design. Drafted 2026-05-15 after V1 land.
**Parent**: `F83_SPARSE_PREFILL_PRD.md` (V1 LANDED).
**Prereq**: V1 perf bench at 256K complete.

## What V1 leaves on the table

V1 is the gather-based MLX-ops path: project Q via `projectQueriesBatchedL`,
score blocks via `topKBlockStartsUnionBatchedQGPU` (batched matmul), build
the position list with static + sliding + fine + coarse, CPU-dedupe,
gather K/V, single `MLXFast.SDPA` with `[L, P+L]` mask. Quality clears
the 0.99 bar at 32K (0.999).

The known V1 bottlenecks, in priority order:

1. **Selector matmul at long context.** `[H, B, D] @ [H, D, L]` →
   `[H, B, L]` scales as `H·B·D·L`. At T=256K, B=T/64=4096, H=8,
   D=32, L=1024: 1.07e9 ops per chunk per layer × 48 layers ×
   ~240 sparse chunks = ~12 TFLOPs of selector work. On M5 Max
   (~7 TFLOPS dense math), that's ~1.7s in the limit. Real wall-
   clock is higher due to MLX dispatch overhead per op.

2. **CPU dedupe sync.** Each chunk per layer pulls ~5-10K int32s
   to CPU. Sub-ms in wall-time but breaks the GPU pipeline.

3. **Lack of incremental selector top-K reuse across chunks.**
   F-79 decode amortizes top-K across 16 steps with 0 quality loss.
   For sparse prefill, queries within a chunk already share top-K
   via union; but BETWEEN chunks, each chunk does fresh selection.
   Adjacent chunks have neighboring query distributions and the
   same prior K — top-K should overlap heavily.

4. **No custom Metal kernel.** The PRD's bitmap path was meant to
   walk the bitmap and skip K/V loads for unset blocks. V1 still
   loads all gathered K (P positions, smaller than T but not zero).
   For very long context (1M+) a kernel that fuses gather + SDPA
   could win another 2-3x.

## V2 milestones

### V2.M1 — Fused L>1 selector kernel

Today `retrievalAttentionScoreTopKFused` takes `[H, D] q` and emits
`[H, K] starts` in one Metal kernel. Extend to `[H, L, D]` →
`[H, K] starts` (union top-K). Replaces the `matmul + max + argpart`
chain (3-4 MLX ops + 1 dispatch each) with one kernel.

Estimated win: 40-60% reduction in selector wall-time at long context.

Implementation:
- Take `qBatched: [H, L, D]` float32 tensor.
- Each threadgroup handles one KV head's union top-K.
- Iterate blocks B/threads_per_tg per pass.
- For each block, compute `max over L of dot(features[h,b,:], q[h,l,:])`.
- Online top-K with simdgroup reduction.

### V2.M2 — Cross-chunk top-K amortization

Inspired by F-79 amort=16. Cache the union top-K from chunk N, reuse
for chunks N+1 ... N+amort-1 with possible selector refresh on a
schedule (every k chunks or based on top-K-set similarity).

The "refresh trigger" can be cheap: hash the previous selection's
fine_starts, recompute every N chunks OR if hash changes (forced
recheck). Reuse otherwise.

Estimated win: 5-10x reduction in selector calls for prefill at
long context. Quality preservation needs to be measured — query
distributions drift across chunks more than across decode steps.

### V2.M3 — Bitmap-walking gather-fused SDPA kernel

The original PRD design. Walks the M2 bitmap. For each set block,
loads 64 positions of K/V. Runs the QK matmul + online softmax
streaming, never materializing the full gathered K tensor in HBM.

Estimated win: 2-3x at 256K vs V1's gather+SDPA pattern. Memory
peak drops by 3-4x (no intermediate gathered tensor).

Implementation: extends the F-69 group-sparse path from L=1 to L>1.

### V2.M4 — Per-query top-K (quality lever)

Today union top-K means queries near the start of a chunk see the
same blocks as queries near the end. If quality on RULER / NIAH
regresses, switch to per-query top-K (L parallel argpartitions).

Cost: L× selector work. Mitigation: V2.M1 fused kernel + V2.M2
amortization roughly cancel the L× cost.

### V2.M5 — TurboQuant+ compose

Combine F-83 with F-81 (V compression). Sparse prefill works on the
rawKey TQ cache. V dequant happens on the gathered subset only —
massive reduction in dequant work vs V dequanting the whole cache.

This is where the memory savings show up at 1M+ context (V at 4bit
= 4x memory drop, sparse means we only dequant K_padded ≈ T/100
positions).

## Risks for V2

- **V2.M1 kernel complexity**: union top-K across L queries with
  online reduction is non-trivial Metal code. ~3-5 days to write
  + debug.
- **V2.M2 quality**: prefill amort behavior is unproven; decode-
  side F-79 amort=16 had stable top-K because adjacent decode steps
  share the same K context. Prefill chunks have ~1024 new K added
  between chunks — top-K may shift faster.
- **V2.M3 fused kernel**: highest engineering cost, highest reward.
  Lift the sparse-SDPA + online-softmax-merge implementation from
  the existing F-69 path; extend to L>1 with bitmap iteration.

## V3 territory (not for V2)

- Multi-batch (B>1) — entire prefill path assumes B=1.
- Sliding-window cache (rotating K/V) interaction with TQ rotating
  KV. Today both work independently.
- Sinks support (MLXFast.SDPA mask + online-merge with sink logits).

## References

- V1 PRD: `F83_SPARSE_PREFILL_PRD.md`
- F-79 amort=16: `RetrievalAttentionKVCache.swift::buildAttentionMaskFusedKernel`
- F-69 group-sparse path: `RetrievalAttentionKernels.swift::retrievalAttentionGroupSparseSDPA`
- F-73 fused mask: `RetrievalAttentionKernels.swift::retrievalAttentionBuildMaskFused`
- F-81 TQ compose: `F81_PHASE_D_V2_DESIGN.md`
