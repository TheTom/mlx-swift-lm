# F-83 PRD — Sparse Prefill (Chunked-Sparse Attention)

**Status**: Design / not started. Research-strengthened 2026-05-14.
**Owner**: feature/retrieval-attention (local-only).
**Created**: 2026-05-14.
**Parents**: F-79 (decode-side block-sparse, shipped), F-80 (MLX cliff fix), F-81 (RA × TQ compose scaffold).
**Reference impl**: NSA fla-org Triton — https://github.com/fla-org/native-sparse-attention (MIT). Port selector + dispatcher 1:1 to Swift; reimplement kernel via mlx-swift primitives.

## Research-derived revisions (read these BEFORE the original design)

Two research passes (prior-art + Apple Silicon kernel deep-dive) on
2026-05-14 produced these critical revisions to the original design:

### REVISION 1 — Mask-not-gather DOES NOT save FLOPs on NAX/steel
The original PRD assumed `[B, H, L, T]` -inf mask passed to existing
SDPA would give us sparse compute for free (extrapolating from
sdpa_vector's L=1 short-circuit at `sdpa_vector.h:105-107`). This is
WRONG for L>1. Steel attention (`kernels/steel/attn/`) and NAX path
(`steel_attention_nax.h:309-350`) **always run the full Q·Kᵀ matmul**
then apply mask post-matmul before softmax. A -inf mask kills the
softmax contribution but the matmul work is already paid.

**Implication**: to actually save compute we must (a) reduce kL via
real gather, OR (b) write a custom sparse kernel that walks a block
bitmap and only loads selected blocks of K/V.

### REVISION 2 — NAX is worth ~2-3x vs non-NAX steel; custom kernel ~30-50% slower than NAX per FLOP
- NAX (M5+): `bq=64, bk=32`, native MMA path
- Non-NAX steel: `bq=32, bk=16` at D=128, native simdgroup MMA
- Custom Metal kernel via `simdgroup_matrix`: ~30-50% slower per FLOP

Net effect at 96% sparsity (K_padded=8K out of T=256K):
- Raw work savings: ~24x
- Custom-kernel penalty vs NAX: ~1.5x
- **Net wall-clock speedup ≈ 16x → ~5 min at 256K (was 5000s = 83 min)**

Original PRD claimed ~20x → 250s. Realistic is 16x → ~5 min after
the kernel work. **Revised target table below.**

### REVISION 3 — Hyperparameter starting point from NSA / MoBA
- Selection block size: **64**
- Top-n selection: **16** (per query)
- Compression block size: **32** stride **16** (for the "coarse" branch)
- Sliding window: **512-2048** (default 2048 matching F-79)
- Static prefix: **128** (matching F-79)
- → K_padded per query ≈ 16·64 + 2048 + 128 = **~3200 positions**
- Out of 256K = **~99% reduction in K positions attended**

### REVISION 4 — Single sparse SDPA call, not split/merge
Original PRD had within-chunk dense + prior-chunks sparse + online
merge. NSA/SeerAttention/MInference all use **monolithic** block-sparse
including current chunk. Within-chunk is just N+1 more blocks in the
selection that happen to be local. Saves the merge cost.

**However**: the split is still preferred for **correctness simplicity**
— within-chunk causal is exact, no quality risk from "selector missed
the local context." F-79's decode path is the L=1 case of "within-chunk"
== "the new token." Keep the split for V1, revisit in V2 once we have
data on the union-monolithic alternative.

### REVISION 5 — Per-chunk union top-K per KV head, NOT per-query
- Per-query top-K: L parallel argpartitions = L=1024× selector cost
- Union top-K per chunk per KV head: 1 argpartition per chunk per KV head
- NSA's GQA-group pattern is exactly this: queries within a chunk in
  the same KV head all select the same blocks.
- Trade-off: queries late in chunk vs early in chunk may want
  different blocks. F-79's amort=16 already showed adjacent decode
  steps share top-K with 0 quality loss; chunk neighbors should too.
- Quality lever: bump from union to per-query in V2 if needed.

### REVISION 6 — Online softmax merge — exact formula
Lift from `mlx-swift/Source/Cmlx/mlx-generated/metal/sdpa_vector.h:320-394`
(the `sdpa_vector_2pass_2` kernel). Pseudo-code:

```c
// Inputs: (O_a, m_a, l_a), (O_b, m_b, l_b) per row
// Output: (O, m, l)
// All accumulators fp32; O can be fp16 at output cast.
if (m_a == -inf) { return (O_b, m_b, l_b); }
if (m_b == -inf) { return (O_a, m_a, l_a); }
float m = max(m_a, m_b);
float factor_a = exp(m_a - m);
float factor_b = exp(m_b - m);
float l = factor_a * l_a + factor_b * l_b;
if (l == 0.0f) { return (zeros, -inf, 0); }  // both empty
float O[D];
for (int i = 0; i < D; ++i) {
  O[i] = (factor_a * l_a * O_a[i] + factor_b * l_b * O_b[i]) / l;
}
return (O, m, l);
```

Already battle-tested inside mlx — copy-paste ~30 lines.

### REVISION 7 — Block bitmap, NOT materialized [L, T] mask
F-73 builds a `[1, 1, 1, T]` mask. Naive extension to `[1, 1, L, T]`
at L=1024, T=256K = 1 GB fp16. Build time scales L× per chunk ≈ 1s
per chunk per layer = unworkable.

**Right design** (from F-69 group-sparse SDPA pattern):
- Build per-query (or per-chunk-union) **block bitmap** of shape
  `[L, nBlocks]` (or `[1, nBlocks]` for union) where nBlocks = T/64.
- For union+T=256K: bitmap = 4096 bits = 512 bytes per chunk per
  layer — trivial.
- Sparse SDPA kernel walks block_idx ∈ [0, nBlocks), checks bitmap,
  skips K/V load entirely for unset blocks. F-69 already does this
  for decode; extend to L>1.

### REVISION 8 — Files to change (concrete list)

| File | Change | Loc |
|---|---|---|
| `Libraries/MLXLMCommon/AttentionUtils.swift` | Drop the `L == 1 && canGather` guard. New L>1 branch routing to `prefillSparseSDPA(queries, keys, values, blockBitmap, scale, mask)`. | ~229-320 |
| `Libraries/MLXLMCommon/RetrievalAttentionKVCache.swift` | New `unionGatherForChunk(chunkQ: MLXArray) -> (positions, K_padded)`. | ~440 |
| `Libraries/MLXLMCommon/RetrievalAttentionKernels.swift` | New `retrievalAttentionBuildBlockBitmap` (returns `[L, nBlocks]` int8) replacing the materialized mask. | ~1437 |
| `Libraries/MLXLMCommon/RetrievalAttentionKernels.swift` | New `retrievalAttentionPrefillSparseSDPA` Metal kernel — per-Q-block walk bitmap, online softmax, merge with global stats. Online merge lifted from `sdpa_vector_2pass_2`. | ~1103 |
| `Libraries/MLXLMCommon/RetrievalAttention.swift` | Add config: `sparsePrefillEnabled`, `sparsePrefillChunkSize` (default 1024), `sparsePrefillMinContext` (default 16K), `sparsePrefillFineTopN` (default 16). | ~155 |
| `Libraries/MLXLMCommon/Models/Qwen2.swift` (etc) | Thread the chunk's absolute offset into attention so selector knows the chunk's causal range. | ~120 |

### REVISION 9 — Revised target table

| Context | Dense prefill | F-83 target (realistic) | Speedup |
|---|---|---|---|
| 32K | 45s | ≤ 15s | ~3x |
| 64K | 180s | ≤ 30s | ~6x |
| 128K | 1100s | ≤ 120s | ~9x |
| 256K | 5000s | ≤ 400s (~7 min) | **~12x** |
| 512K | OOM today | ≤ 1000s | unblocks |
| 1M | OOM today | ≤ 2500s | unblocks (matches SubQ) |

The original PRD's "20x → 4 min" was over-optimistic by ~2x. Honest
target is **12x → 7 min** at 256K. Still a huge product win.

### REVISION 10 — Validation: add RULER

Original PRD said "cosine ≥ 0.99 vs dense." That's necessary but not
sufficient. Add: **RULER benchmark** at 32K, 64K, 128K. Within 2
absolute points of dense baseline (MInference's published bar).

- RULER repo: https://github.com/NVIDIA/RULER
- NIAH script: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- Per-layer cosine: each attention layer's output cosine vs dense
  (so we catch error accumulation early, not just at logit head)

### Highest-risk issues (Agent 2 ranking)

1. **Quality regression on prefill**: decode RA is robust because the
   model has full prefill context. Sparse prefill = approximate prefix
   attention, error compounds across 48 layers. Per-layer cosine
   measurement mandatory before claiming success.
2. **Selector cost at L>1**: at L=1024 each query needs its own top-K
   IF per-query path is taken. Union path collapses to 1 argpartition
   per chunk per KV head — load-bearing optimization.
3. **NAX vs custom-kernel tradeoff**: custom Metal kernel ~30-50%
   slower per FLOP. At 99% sparsity still wins big, but quality
   regressions could force lower sparsity → less win.
4. **Gather contiguous-copy cost**: `takeAlong` on 256K cache with
   K_padded=8K is 32 MB copy per layer per chunk × 256 × 48 = 400 GB
   memory traffic. Consider **gather-fused sparse SDPA kernel**
   (extension of F-71 group-sparse path) that reads from indices
   directly without intermediate copy.
5. **Stream overlap stalls**: F-78 showed M5 command queues serialize
   when both GPU-bound. Stream overlap optimistic = 10-15% wall-time
   win, not 2x.

### Things to copy-paste

- Online merge: `sdpa_vector_2pass_2` (`sdpa_vector.h:320-394`) — 30 lines
- Block-skip walk in sparse SDPA: F-69 pattern in `RetrievalAttentionKernels.swift`
- F-73 mask kernel: starting point for block bitmap builder
- F-77 parallel-bundle: starting point for batched-Q selector

### Things that are new work

- Batched-Q top-K selector (L=1024 in parallel, not L=1)
- L>1 sparse SDPA Metal kernel with bitmap-driven block skip
- Two-branch online merge in the model attention forward
- Per-layer cosine instrumentation for quality validation

### NSA porting strategy (lifted from Agent 1)

The fla-org/native-sparse-attention repo's `parallel_nsa` Python wrapper
is framework-agnostic — port these pieces 1:1 to Swift:
- Block-indices construction logic
- GQA aggregation across heads in a group
- Gate combination (compression + selection + sliding window branches)
- Sliding-window fusion logic

Reimplement the actual attention kernel via existing mlx-swift steel
SDPA + the bitmap-driven sparse path described above. **80% of the
algorithmic risk is knocked out** by following NSA's tested algorithm.

---

# Original PRD (kept for context — see revisions above for current direction)

## Motivation

F-79 amort=16 closed the **decode-side** gap vs dense — but prefill is
still O(N²). Headline measurements:

| Context | Dense prefill (14B-1M, M5 Max) |
|---|---|
| 24K | 22.3s |
| 32K | ~45s |
| 128K | ~1100s (~18 min) |
| 256K | ~5000s (~83 min) |

SubQ's headline claim is **sub-quadratic prefill** at 256K+. Open Sparse
Stack's PRD goal is "match SubQ at home"; F-79 only matches the decode
half of that claim. To complete the match we need sparse prefill.

The good news: we already have most of the machinery from F-79. The
selector index, JL projections, block features, fused mask kernel —
all reusable. The gap is generalizing from L=1 (decode) to L>1
(prefill chunk).

## Goal

Prefill latency at 256K context drops from **O(N²) to O(N · log N)**
or better, while maintaining cosine ≥ 0.99 vs dense at every decode
step after sparse prefill, and ≥ 8/8 argmax match at the ship config.

Specifically:

| Context | Dense prefill | Target sparse prefill | Speedup |
|---|---|---|---|
| 32K | 45s | ≤ 12s | ≥ 3.5x |
| 64K | 180s | ≤ 30s | ≥ 6x |
| 128K | 1100s | ≤ 90s | ≥ 12x |
| 256K | 5000s | ≤ 250s | ≥ 20x |
| 512K | OOM today | ≤ 600s | unblocks 512K |
| 1M | OOM today | ≤ 1500s | unblocks 1M (matches SubQ headline) |

## Design — Hybrid chunked-sparse prefill

**Core idea**: at each prefill chunk N, split attention into two
contributions and combine via online softmax:

1. **Within-chunk** (Q × K_chunk): dense causal SDPA. Cheap because
   chunk_size² ≪ N². At chunk_size=1024 this is 1M ops; constant
   regardless of total context. Always dense — selector index for
   chunk N hasn't been built yet when these queries are computed.

2. **Prior-chunks** (Q × K_prior): RA-style block-sparse. The
   selector index was already built incrementally from chunks
   1..N-1. For each query in chunk N, pick top-K fine + top-K coarse
   blocks from the prior `(N-1)·chunk_size` keys + static prefix +
   sliding window.

Combine via the standard online-softmax merge:
```
m_new = max(m_within, m_prior)
exp_w = exp(s_within - m_new)
exp_p = exp(s_prior - m_new)
o = (exp_w · V_within + exp_p · V_prior) / (sum_w + sum_p)
```

Pseudocode for one prefill chunk:

```
for chunk_n in chunks:
    Q = wq(x_chunk_n)
    K_new = wk(x_chunk_n) ; V_new = wv(x_chunk_n)

    # ----- within-chunk: dense causal
    O_within, m_within, l_within = sdpa_causal(Q, K_new, V_new)

    # ----- prior chunks: sparse gather
    if chunk_n > 0:
        positions = selector.pick_top_k_per_query(
            Q, K_prior_blocks, static_n=128, sliding_window=2048
        )
        K_gather = cache_K.take(positions, axis=2)
        V_gather = cache_V.take(positions, axis=2)
        mask = build_causal_mask_for_positions(positions, chunk_n)
        O_prior, m_prior, l_prior = sdpa_with_mask(Q, K_gather, V_gather, mask)
    else:
        O_prior = 0 ; m_prior = -inf ; l_prior = 0

    # ----- online merge
    O_chunk = online_softmax_merge(O_within, m_within, l_within,
                                    O_prior, m_prior, l_prior)

    # ----- update cache + selector
    cache_K.append(K_new) ; cache_V.append(V_new)
    selector.update(K_new)  # JL project + block-pool
```

## Asymptotic analysis

At chunk N processing chunk_size=L tokens with cache length T:

- Within-chunk: O(L²) — constant
- Selector lookup: O(L · log(T/64)) — for each of L queries, top-K over T/64 blocks via argpartition
- Gather + sparse SDPA: O(L · K_padded) where K_padded = bounded selector budget (~4-8K)
- Selector update: O(L · selectorDim) — JL projection

Per-chunk cost: **O(L² + L · K_padded)** — independent of T (assuming K_padded is bounded).

Total prefill: (N/L) × O(L² + L · K_padded) = **O(N · L + N · K_padded)** = **O(N)** if K_padded is bounded.

So sparse prefill is linear in N, not quadratic. 256K is ~250s if
L=1024 and per-chunk dispatch overhead is ~1s.

## Implementation milestones

**M1 (1 week) — multi-query selector lookup**:
- Extend `BatchedRetrievalAttentionIndex.topKBlockStartsAllHeadsCombined`
  to accept `[nKVHeads, L, dHead]` Q tensor (not just `[nKVHeads, dHead]`)
- Output: `[nKVHeads, L, K_padded]` block start positions
- Test: at L=1, output matches existing decode-path selector

**M2 (1 week) — L×S mask kernel**:
- Extend F-73 fused-mask kernel from `[1, 1, 1, T]` to `[1, 1, L, T]`
- Per query in the L dimension, build its own mask from per-query
  top-K positions
- Or: union across L queries → single mask, accepts some over-attention
  (cheaper, lower quality)
- Test: bit-exact vs reference dense+gather

**M3 (1 week) — within-chunk + prior-chunks split**:
- New dispatcher path in `attentionWithCacheUpdate` for L>1 prefill
  on retrievalSparse cache
- Two SDPA calls + online merge
- Validate cosine vs full-dense prefill at 8K, 16K, 32K

**M4 (1 week) — long-context validation + benchmark**:
- Quality: cosine ≥ 0.99 vs dense at 32K, 64K, 128K, 256K
- Latency: per-chunk timing, total prefill time, vs dense baseline
- Memory: peak active during prefill
- Long-context unblocks: 512K, 1M (memory-permitting)

**M5 (3 days) — within-chunk-only fallback for first chunk**:
- Chunk 0 has no prior — pure within-chunk dense
- Selector starts empty until chunk 0 finishes — easy invariant

**Total estimate: ~4-5 weeks.**

## Open questions

- **Q1**: Per-query top-K vs union top-K within a chunk?
  - Per-query: best quality (each query sees ideal subset), worst
    overhead (L parallel argpartitions, L different gather indices,
    causal mask per-row)
  - Union: simpler (single mask for the chunk), worse quality (queries
    early in chunk see positions selected for queries late in chunk —
    causally OK because positions are all prior, but selector waste)
  - **Recommend**: start with union for M1-M3, switch to per-query
    in M4 if quality demands

- **Q2**: Causal handling within sparse selection
  - Selector picks positions; mask enforces causal `position < q_pos`
  - Per-query selector + per-query mask is the safe path
  - Union selector with row-wise mask works too — each row keeps
    only positions ≤ its q_pos

- **Q3**: Sliding window vs selector budget interaction
  - Today sliding = last 2048 tokens always attended. At prefill,
    within-chunk is the sliding window for the LAST chunk's tail.
    Need to reason about per-chunk sliding semantics.

- **Q4**: Does selector quality at small T (early chunks) tank
  prefill quality?
  - At chunk 2 with T=1024, selector has ~16 blocks. Top-32 won't
    fit — fall back to dense for low-T chunks?
  - Threshold: skip sparse path when T < `sparseMinContext` (today
    16K for decode). Apply same to prefill.

- **Q5**: How does this compose with TurboQuant+ (Phase D V2)?
  - K-cache content used by selector must be raw (rawKeyMode) —
    matches F-81 V1 scaffold's constraint
  - V dequant on gather — same gather-and-dequant API as Phase D V2

## Validation plan

1. **Unit**: F-73-style mask kernel at L=16, L=128, L=512. Bit-exact
   vs reference numpy/MLX implementation.
2. **Microbench**: synthetic Q/K/V at T=32K, time sparse vs dense
   prefill on a single attention layer. Confirms kernel-level speedup.
3. **Quality**: full-model prefill at 32K with sparse path + 8 decode
   steps. Compare to full-dense baseline. Target cosine ≥ 0.99,
   argmax 8/8.
4. **Latency**: prefill time at 32K / 64K / 128K / 256K. Confirms
   N-linear scaling.
5. **Long-context unblock**: 512K, 1M prefill. Today these OOM during
   dense prefill matmuls; sparse path should make them feasible.
6. **Regression**: F-79 amort=16 decode quality unchanged after the
   prefill changes — sparse prefill produces same K/V cache as dense,
   so decode quality should be identical.

## Risks

- **Online softmax merge precision**: combining two SDPA outputs via
  online softmax is sensitive to fp16 overflow at extreme score
  differences. Use fp32 accumulators in the merge.
- **Selector quality at low T**: see Q4. Sparse path may produce
  garbage when T < min reasonable selector budget.
- **Per-query overhead**: if union doesn't hold quality, per-query
  selector adds L×selectorOps cost. May undo some of the savings.
- **Cache update timing**: must update cache K/V *and* selector
  index AFTER the sparse SDPA, not before, or selector "sees the
  future" within the chunk.
- **Memory peak during gather**: gathered K/V is `[B, nKVHeads, L * K_padded, D]`
  — for L=1024, K_padded=4096, that's 4M entries × D=128 × 2 bytes
  = 1 GB per layer. Avoid materializing — use mask path instead
  (mask-not-gather, same as F-73).

## What we are NOT doing in V1

- Pure-sparse within-chunk (within-chunk stays dense — chunk-size
  is small enough)
- Two-pass selector (selector updates AFTER chunk, not during)
- KV-quantized K for the selector (requires Phase D V2)
- Multi-batch (B>1)
- Non-Qwen2-style architectures (test on Qwen2.5-14B-1M-4bit first)

These are V2 territory.

## References

- F-79 ship doc: `Open Sparse Stack — Near-Parity with Dense.md`
- F-79 diligent log: `Open Sparse Stack — F-71 to F-78 Diligent Log.md`
- F-80 cliff fix: `F80_CLIFF_DIAGNOSIS.md`
- F-81 TQ compose: `F81_PHASE_D_V2_DESIGN.md`
- F-73 fused mask kernel: `RetrievalAttentionKernels.swift::retrievalAttentionBuildMaskFused`
- F-79 amort=16 selector: `RetrievalAttentionKVCache.swift::buildAttentionMaskFusedKernel`
- Selector index: `BatchedRetrievalAttentionIndex.swift`
- NSA paper (block-sparse during prefill, the architectural inspiration): https://arxiv.org/abs/2502.11089
- FlashAttention-2 (within-chunk dense reference): https://arxiv.org/abs/2307.08691
- SubQ benchmark (target to match): https://subq.mildlyconcerning.com/ (Tom's tracking site)
