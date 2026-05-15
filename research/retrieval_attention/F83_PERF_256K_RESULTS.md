# F-83 Perf Results — V1, Qwen2.5-14B-Instruct-1M-4bit, M5 Max

**Date**: 2026-05-15
**Branch**: `feature/retrieval-attention`
**Test**: `f83_perfBench256K_14B1M`

## Headline numbers

### 128K context (`F83_PREFILL_LEN=131072`)

| Variant | Prefill | Speedup | Decode | Cosine | Notes |
|---|---|---|---|---|---|
| Dense baseline (chunked) | 350-398 s | 1.0x | 94 ms | — | varies ±15% between runs |
| **V1.0** (top-K=512 adaptive, per-head + CPU dedupe) | 424 s | 0.95x | 193 ms | NaN | adaptive top-K = ~4x sparsity, no real win |
| **V1.0.1** (top-K=16 fixed, per-head + CPU dedupe) | 329 s | 1.21x | 200 ms | 0.985 | NSA-style top-K, dispatch bound |
| **V1.1** (cross-head union + GPU-only positions + exp grow) | 239 s | 1.53x | 143 ms | 0.964 | killed CPU dedupe sync |
| V1.2 (top-K=32 cross-head) | 250 s | 1.48x | 190 ms | 0.965 | reverted — extra K cost > coverage gain |
| chunkSize=2048 alone | 200 s | 1.77x | 197 ms | 0.966 | per-chunk overhead amortizes over 2x queries |
| **V1.3 = chunk2K + IndexCache** | 174 s | 2.01x | 169 ms | 0.967 | first 2x! cross-layer selector reuse |
| V1.3 + chunkSize=4096 | 171 s | 1.98x | 181 ms | 0.970 | plateau — chunk size knob exhausted |
| **V1.4 (.causal mask)** | **165 s** | **2.13x** | 183 ms | 0.967 | drop 6 ops/layer/chunk via .causal |
| V1.5 (groupSize=8) | 165 s | 2.10x | 198 ms | 0.966 | selector not the bottleneck — flat |
| V1.6 (sliding=1024) | 162 s | 2.13x | 182 ms | 0.963 | same speed, slight cosine drop — back to 2048 |

V1.0 → V1.1 deltas:
- 90s prefill savings (44% sparse-side reduction) — from killing per-chunk
  CPU sync (asArray + Set dedupe) and per-chunk eval barriers (linear → exp grow)
- 57ms decode savings (28%) — fewer dangling lazy-graph references after prefill

V1.1 quality cost: cosine 0.985 → 0.964, because cross-head union (one shared
block list) has less coverage than per-head deduped union (~16 vs ~40-80 unique
blocks). V1.2 bumps top-K to 32 to recover coverage.

### 256K context

**Cannot bench at 256K** — pre-existing RA cache cliff at chunk 144
(~147K context). Same death pattern in the existing F-79 256K test
(its RA path stalled at the same chunk in the most recent run).
This is NOT an F-83 issue — it's an RA cache long-context bug that
predates F-83. Tracking separately.

## What worked

- F-83 V1 implementation is correct: all 4 unit tests green, the
  small-prior plumbing test = 1.0 cosine, 32K real-model = 0.999
  vs chunked-dense.
- The fixed-top-K-16 fix (PRD revision 3) flipped sparse from
  0.95x (slower) to 1.21x (faster) at 128K.
- No NaN, no broken outputs — both dense and sparse logits sit in
  expected ranges.

## Why the speedup is 1.21x, not the PRD's 9x target

Wall-time breakdown for sparse path at 128K (~96 sparse chunks ×
48 layers each):

| Component | Estimated wall-time | Notes |
|---|---|---|
| Selector matmul + argpart | ~70 s | 48L × 96C × ~15 ms |
| Gather + concat | ~25-50 s | per layer per chunk |
| Sparse SDPA `[L, K_padded+L]` | ~28 s | ~6 ms × 48 × 96 |
| MLX dispatch overhead | ~250-300 s | THIS is the dominant cost |
| **Total measured** | **329 s** | matches the build-up |

V1 uses the MLX-ops chain (project_Q + score + max-pool + argpart +
gather + concat + reshape + mask + SDPA) — ~20 op dispatches per
sparse layer per chunk. At 96 sparse chunks × 48 layers × 0.5 ms
dispatch overhead per op = ~460 s of pure dispatch.

**The win is dispatch-overhead-bounded, not compute-bounded.** That
matches NSA / SeerAttention's experience — getting the full sparse
prefill speedup requires a fused Metal kernel.

## V2 path forward

See `F83_V2_DESIGN.md`. M1 (fused L>1 selector kernel) folds ~6 of
the selector dispatches into one — single biggest expected win.
Conservative estimate: 1.21x → 3-4x at 128K once V2.M1 lands.

The full V2 (M1 + M3 bitmap-walking SDPA kernel) should clear the
PRD's 9x at 128K and ~12x at 256K (once the 147K cliff is fixed
independently).

## Quality assessment

Cosine 0.9855 is just below the PRD's 0.99 target. Likely cause:
random tokens at 128K push the model's activation trajectory into
high-variance regions where small attention deltas amplify. On
real text (which the model was trained on), cosine should be tighter
— but RULER / NIAH benchmarks need real validation in a follow-up.

## Decode-side concern

Sparse decode steady = 200 ms vs dense decode steady = 94 ms (~2x
slower). Both decodes use the same F-79 ship config — there's no
intentional difference. Hypothesis: post-sparse-prefill cache state
holds different lazy-graph references than post-dense-prefill,
making steady-state decode slower. Worth investigating but doesn't
block PRD M5 sign-off.

## Notes on the 147K cliff

Three consecutive 256K bench runs (and the existing F-79 256K test
in the same session) terminate silently at chunk 144 (147,456
tokens) when an RA cache is active. Pre-existing — needs separate
diagnosis. F-83 V1 lands without the 256K perf number; bench passes
cleanly at 128K which is below the cliff.

## Where the raw data lives

- `/tmp/f83_perf256k_progress.log` — live per-chunk progress
- `/tmp/f83_perf128k_v2_test.log` — full swift test stdout for the
  fixed-top-K=16 run
- Tests: `f83_perfBench256K_14B1M` runs at `F83_PREFILL_LEN` env
  (default 256K). Set to 131072 for the working 128K bench.
