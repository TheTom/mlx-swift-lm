# F-85 v3 — Compose-Gather Bench Results

Date: 2026-05-16
Branch: `feature/retrieval-attention` (local only)
Model: `Qwen2.5-14B-Instruct-1M-4bit`
Hardware: M5 Max
Config: `VSM_SPARSE_FINE_BS=64 VSM_SPARSE_FINE_TOPK=32 VSM_SPARSE_NO_ADAPTIVE=1 VSM_SPARSE_COARSE_TOPK=2`
   (K_padded ≈ static 64 + sliding 1024 + fine 32×64 + coarse 2×1024 = 5184)

## TL;DR — HONEST NEGATIVE

**Compose-gather LOSES vs dense AND vs v2 F-73 batched mask at B=8.**

- B=8 ctx=32K: compose-gather 4.5 tok/s vs dense 49.5 tok/s = **0.09× (11× SLOWER)**.
- B=8 ctx=16K: compose-gather 40.5 tok/s vs dense 69.8 tok/s = **0.58× (1.7× SLOWER)**.
- K_padded sweep (fineTopK 4/8/16/32) didn't change the picture — even the
  best cell is 0.30× dense.

The PRD bandwidth math (5184/32768 × 162ms ≈ 26ms predicted) was off by
**44×**. R3 lit was correct: per-slot gather doesn't scale at B≥8 because
each slot picks different positions, so MLX must materialize B separate
[nKVH, K_padded, D] slabs — pathological per-layer cost.

**Ship decision: v2 F-73 batched mask remains canonical.** Compose-gather
shipped as opt-in (`VSM_SPARSE_BATCHED_KERNEL=composegather`) for A/B
regression only. Next workstream: F-71c GQA-coload kernel rewrite.

Surprise positive: at B=8 ctx=16K F-73 batched **BEATS dense** 95.7 vs
69.8 tok/s (1.37×) — was missed in v2 results doc (which only reported
ctx=32K). Worth re-pitching v2 default on workloads bounded at 16K ctx.

## M2 — primary grid

Decode-only `tok/s`, prompt tokens as listed, 30 decode tokens.

### B=8 ctx=32K

| Path | tok/s | ms/step | × vs dense |
|---|---:|---:|---:|
| Dense | 49.5 | 162 | 1.00× |
| F-73 batched (v2) | 26.4 | 303 | 0.53× (1.87× slower) |
| **F-85 v3 compose-gather** | **4.5** | **2222** | **0.09× (13.7× slower) ❌** |
| F-71b v1 | 3.5 | 2286 | 0.07× |

### B=8 ctx=16K

| Path | tok/s | ms/step | × vs dense |
|---|---:|---:|---:|
| Dense | 69.8 | 115 | 1.00× |
| F-73 batched (v2) | 95.7 | 84 | 1.37× FASTER |
| **F-85 v3 compose-gather** | **40.5** | **247** | **0.58× (2.1× slower) ❌** |

## Headline: compose-gather is the WRONG primitive at B>1

Theory predicted ~50ms/step from BW math (5184/32768 × 162ms ≈ 26ms +
overheads). Actual: 2222ms — **44× worse than predicted**. Penalty
scales super-linearly with T (16K → 247ms, 32K → 2222ms = 9× for 2× ctx).

This implicates per-layer overhead that's proportional to T even though
the gather is small. Hypothesis: `takeAlong` on a 4D tensor with
[B, nKVH, K_padded, 1] index broadcasting to [B, nKVH, K_padded, D]
either:
1. Materializes a full [B, nKVH, K_padded, D] intermediate via a
   non-coalesced kernel (memory-bound on the materialization).
2. Forces an SDPA invocation that doesn't hit the MLXFast fused tile
   despite mask:.none (4D index pattern may break dispatch).

## What WORKS at B>1: F-73 batched mask

At B=8 ctx=16K F-73 batched **BEATS dense** 95.7 vs 69.8 tok/s (1.37×).
At B=8 ctx=32K F-73 loses (303 vs 162ms = 0.53×) because the mask path
still streams all T K/V — the kernel's `(B, 1, 1, T) mask × MLXFast SDPA`
contract gives compute savings but not BW savings.

## What DOESN'T work at B>1: per-(B, nKVH) takeAlong

The compose-gather path that wins at B=1 (F-84 blockGather pattern, 2.9×
faster than dense at 128K) does NOT scale to B>1 with per-slot gather.
The per-slot gather can't share the gathered slab across slots (each
slot picks different positions), so MLX must materialize B separate
gathered slabs — pathological memory cost.

## R3 lit was right

R3 lit explicitly said "compose-only is fundamentally wrong" at B≥8
long-ctx. The PRD bandwidth math was optimistic — it didn't account
for the per-slot materialization cost, which dominates over the K/V
read savings.

## What ships

**v2 F-73 batched mask remains the canonical batched-sparse path** for
B>1 decode. At 16K it BEATS dense. At 32K+ it's a worse-per-step but
the per-slot throughput (B×tok/s) is still 4-5× better than serial
sparse B=1.

Compose-gather is shipped as an opt-in (`VSM_SPARSE_BATCHED_KERNEL=
composegather`) for documentation + A/B regression checks only. NOT
the default.

## Next workstream candidates (v4)

- **F-71c GQA-coload custom kernel rewrite** — direct sparse compute,
  amortize TG-per-(B, KV head) instead of per-(B, Q head). Should
  bring F-71b's 14× slowdown down toward parity.
- **Cross-slot block-gather** — when slots happen to share top-K
  blocks (high prefix overlap in chat / multi-turn), gather once
  across slots. Workload-dependent win.
- **TurboQuant'd batched K/V** — composes with either above. Cuts
  BW pressure directly.

## M3 — K_padded sweep at B=8 ctx=32K

Cell config: dense baseline 49.5 tok/s = 162 ms/step.

| fineTopK | K_padded (approx) | tok/s | ms/step | x vs dense |
|---:|---:|---:|---:|---:|
| 32 (default) | 5184 | 4.5 | 2222 | 0.09x |
| 16 | 4160 | 14.1 | 567 | 0.29x |
| 8 | 3648 | 13.9 | 576 | 0.28x |
| 4 | 3392 | 14.9 | 537 | 0.30x |
| 32 + AMORT=48 | 5184 | 5.3 | 1887 | 0.11x |

Notably the penalty did NOT scale meaningfully with K_padded (fineTopK
4-16 gave 14-15 tok/s; fineTopK=32 was the outlier at 4.5). This rules
out "gather size" as the dominant cost. Selector amortization
(`AMORT=48`, re-use selector across all decode steps) gave 5.3 tok/s
at fineTopK=32 — basically no help, ruling out "selector cost dominates."

The fineTopK=32 cliff (3x slower than fineTopK<=16) suggests an MLX
kernel-dispatch threshold: above some gather-size watermark
`takeAlong` or `MLXFast.SDPA` flips to a slower code path. Worth a
follow-up `mxTakeGatherBench` microbench, but moot — even the best
compose-gather cell (fineTopK=4 at 0.30x dense) is far from a win.

## Tests

`Tests/MLXLMTests/F85V2BatchedMaskTests.swift`:
- `composeGatherMatchesF71b` — numerical equivalence to F-71b at fp32 ✅
- `composeGatherSanityVsDense` — guards against slot crossover ✅

## Files touched

mlx-swift-lm (`feature/retrieval-attention`):
- `Libraries/MLXLMCommon/BatchedRetrievalAttentionKVCache.swift` —
  add `sparseAttendComposeGather` private + `composegather` dispatch.
- `Tests/MLXLMTests/F85V2BatchedMaskTests.swift` — equivalence tests.

vllm-swift (`feature/retrieval-attention`):
- This results doc.
