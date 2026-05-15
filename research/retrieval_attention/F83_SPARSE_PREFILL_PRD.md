# F-83 PRD — Sparse Prefill (Chunked-Sparse Attention)

**Status**: Design / not started.
**Owner**: feature/retrieval-attention (local-only).
**Created**: 2026-05-14.
**Parents**: F-79 (decode-side block-sparse, shipped), F-80 (MLX cliff fix), F-81 (RA × TQ compose scaffold).

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
