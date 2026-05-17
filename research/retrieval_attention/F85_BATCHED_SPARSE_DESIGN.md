# F-85 — Batched Sparse Decode (B>1)

Status: Phase 1 (design) — 2026-05-16.
Branch: `feature/retrieval-attention` (local-only per Open Sparse Stack rule).
Coord-revised target kernel: **F-71b `retrievalAttentionGroupSparseSDPA`**
(NOT F-76). Coordinator note + R3/R4 audit moved the target after Phase 1.

---

## Problem

Today every retrieval-attention path hardcodes B=1 (`precondition(keys.dim(0) == 1)`
in `RetrievalAttentionKVCache.update`, `q.shape.count == 2` everywhere on the
sidecar context, slot-0 slices in `RetrievalAttentionEngine.queries[0,...]`,
`BatchedRetrievalAttentionIndex` shapes `[nKVH, T, D]` with no B dim, and
Bridge.swift falls back to serial per-request loops at B>1 — see
`Bridge.swift:1111-1136`). The aggregate effect: sparse decode at B=8 walks
sessions sequentially instead of running ONE batched forward per token.

Theoretical bandwidth math: at B=8, T=128K, K_padded ≈ 2K, batched sparse
compose-only ≈ 71ms/step vs dense 443ms = **6.3× faster**. The whole point.

## Architectural choice: per-(B, KV-head) top-K + F-71b fused kernel

Per coordinator update — F-71b is the right target, not F-76.

### Why F-71b > F-76

F-71b kernel `retrievalAttentionGroupSparseSDPA` is genuinely B-aware AND
fused-gather:
- Q shape `[B, nQH, 1, D]`
- K/V shape `[B, nKVH, T, D]`
- gather shape `[B, nKVH, K_padded]` (NATIVE — not `[1, nKVH, K_fine]` like the
  current F-76 caller hack `expandedDimensions(axis: 0)`)
- Grid `B * nQH` threadgroups × 1024 threads
- Kernel indexes `(b * NKVH + kvh) * T * HEAD_DIM + pos * HEAD_DIM` — no B=1
  assumptions
- GQA-group co-load: one threadgroup per (B, Q head), so all `groupSize` Q heads
  of a KV group reuse one block fetch from K/V (matches NSA/SeerAttention-R/DSA
  production pattern per R3 external lit)
- Per-request K stays independent — no mega-K union

F-76 (implicit-positions): also B-aware in kernel signature, but its current
caller builds `fineStarts3D = fineStarts.expandedDimensions(axis: 0)` which is
a B=1 hack. R3 audit calls compose-only "fundamentally wrong at B≥8 T≥32K"
because the F-76 wrapper materializes intermediate gathered K/V via MLX `take`
ops; F-71b fuses the gather into the attention loop and never materializes.

NOT doing: shared selector across batch (semantically wrong — different
prompts have completely different attention focus; cross-request union would
blow K_padded up to ~T at B=16+).

### Constraint: rectangular K/V

For v1 assume all B slots have the same T (prefill ctx length). Bench script
`bench_throughput.py` uses the same prompt for all batch slots, so K/V is
naturally `[B, nKVH, T, D]` rectangular. Continuous-batching ragged-T defers
to v2.

---

## Components

### Phase 2 — BatchedRetrievalAttentionIndex B-dim

Today: `perTokenFeatures: [nKVH, T, D_eff]`,
`fineBlockFeatures: [nKVH, nBlocks, D_eff]`,
`projectQueriesBatched(q: [nKVH, dHead]) -> [nKVH, D_eff]`,
top-K returns `[nKVH, K_fine]`.

After Phase 2:
- `perTokenFeatures: [B, nKVH, T, D_eff]`
- `fineBlockFeatures: [B, nKVH, nBlocks, D_eff]`
- `projectQueriesBatched(q: [B, nKVH, dHead]) -> [B, nKVH, D_eff]`
- top-K returns `[B, nKVH, K_fine]`
- `update(newKeys: [B, nKVH, L, dHead])` instead of `[nKVH, L, dHead]`

Existing `BatchedRetrievalAttentionIndex` (Phase 2 of the original index batching
across KV heads) stays mostly intact — we add a B axis at the front of every
internal MLX tensor and at every Swift loop bound.

Add Tests/MLXLMTests/BatchedRetrievalAttentionIndexBatchTests.swift:
B=2, build index from different synthetic K sequences per slot, score against
different Q per slot, verify per-slot top-K independence (slot 0 picks different
K positions than slot 1 for slot-specific Q).

### Phase 3 — BatchedRetrievalAttentionKVCache (new class)

New file `Libraries/MLXLMCommon/BatchedRetrievalAttentionKVCache.swift`.
Mirrors `RetrievalAttentionKVCache` but:
- `update(keys: [B, nKVH, 1, D], values: [B, nKVH, 1, D])` — no B==1 precondition
- Internal `inner: BatchedKVCache` (NOT StandardKVCache — we need rectangular
  `[B, nKVH, T, D]` storage shared across slots)
- `batchedIndex: BatchedRetrievalAttentionIndex` with B dim
- One sparse path: `sparseAttend(q: [B, nQH, 1, D], scale: Float) -> [B, nQH, 1, D]`
  - Project per-(B, nKVH) Q with selector → `[B, nKVH, D_eff]`
  - Build per-(B, nKVH) top-K → `[B, nKVH, K_fine]`
  - Expand to per-(B, nKVH) sorted gather list `[B, nKVH, K_padded]` (mirror
    `perKVHeadGatherGPU` but with B at the front)
  - Call `retrievalAttentionGroupSparseSDPA(queries: ..., perKVHeadGather: gather, ...)`
    directly — kernel signature already accepts `[B, nKVH, K_padded]`

Smoke test: B=2, random K/V via Index update, run sparse forward, verify output
shape `[2, nQH, 1, D]` and per-slot outputs differ when per-slot Q differs.

### Phase 4 — Bridge.swift batched-sparse path

In `vllm-swift/swift/Sources/VLLMBridge/Bridge.swift`:
- Today `vsm_engine_decode_all:1111-1136` walks `sparseSessions` sequentially.
- New: when env `VSM_SPARSE_BATCHED=1` AND B>1 sparse sessions present, build
  a SINGLE batched forward call — stack per-slot input tokens into `[B, 1]`,
  reuse the model's existing batched-decode path with a new `BatchedRetrievalAttentionKVCache`
  per layer.
- Gate behind env so existing serial fallback survives as opt-out.
- Bench output: log `batched=true/false` so it's obvious which path ran.

The model wiring is the trickier bit. Qwen2 today has no `fullyBatchedForward`
with `raContexts:` overload — that path uses `BatchedKVCache.attention` which
runs dense SDPA. Two options:

  **Option A — extend Qwen2Attention.fullyBatchedForward** to accept an
  optional per-layer `BatchedRetrievalAttentionContext?` and dispatch through
  a new `attentionWithBatchedCacheUpdate` that knows the batched sparse path.

  **Option B — new SparseSession-collection prefill path** that builds ONE
  shared `[BatchedRetrievalAttentionKVCache]` per layer (not per-request),
  prefills all B prompts into it (rectangular T), then a batched decode loop
  routes through a new `Qwen2Model.fullyBatchedSparseDecode(_:caches:)` method
  that mirrors `fullyBatchedDecode` but with the batched RA caches.

Going with **Option B** for v1 — minimum invasion of the existing
fullyBatched path; no risk of breaking dense B>1 customers. The new method
lives in MLXLMCommon/Models/Qwen2.swift.

### Phase 5 — Bench + validate

Smoke B=2 short ctx first (sanity), then real B=8 ctx=32K bench, then dense
baseline for comparison. If 128K fits in VRAM, B=4 ctx=128K too.

Numerical sanity contract:
- B=2 same prompt twice → bit-exact match between slot 0 and slot 1
- B=2 different prompts → slot 0 != slot 1 (different outputs)
- B=8 sparse vs B=1 sparse on same prompt → within numerical noise per slot
- B=8 sparse vs B=8 dense at 32K → sparse should beat dense; if not, REPORT
  HONESTLY

---

## Risks / open questions

1. F-71b regressed 1.4× vs F-73 at B=1 32K per FINDINGS. At B=8 long-ctx
   where bandwidth dominates, F-71b's actual-K-skip should win — but if it
   doesn't, the kernel either has a hidden serialization at high B (kernel
   already accepts B, but high B may oversubscribe SMs in a way that hurts
   single-warp scoreboard) or our wiring still pays a CPU-side per-slot cost.
2. Selector index memory grows linearly with B. At T=128K, B=8, fineBlockSize=64,
   contentDim=32: `[8, 8, 2048, 32]` int16 = 16 MB per layer × 40 layers
   = 640 MB just for fine block features. Coarse adds proportionally less.
   Plus `perTokenFeatures: [B, nKVH, T, D_eff]` = `[8, 8, 131072, 32]` int16
   = 512 MB per layer × 40 = 20 GB — UNUSABLE. We MUST drop the
   `perTokenFeatures` storage in the batched index; only the block-pooled
   features matter for scoring. See FINDINGS "Memory" section — already
   flagged for the single-batch index.
3. BatchedKVCache currently stores `[B, nKVH, T, D]` rectangular shared K/V.
   F-71b kernel expects exactly that — perfect fit.

## Phasing summary

| Phase | Output | Commit |
|---|---|---|
| 1 | This DESIGN.md | `docs(f85): batched sparse decode design` |
| 2 | BatchedRetrievalAttentionIndex w/ B dim + test | `feat(f85): BatchedRetrievalAttentionIndex with B dim` |
| 3 | BatchedRetrievalAttentionKVCache + F-71b call + smoke test | `feat(f85): BatchedRetrievalAttentionKVCache + F-71b batched call` |
| 4 | Bridge.swift `VSM_SPARSE_BATCHED=1` path | `feat(f85): Bridge batched-sparse decode path (opt-in)` |
| 5 | Bench results (B=2 smoke, B=8 @ 32K + dense baseline) | `bench(f85): batched sparse decode validation` |

All work LOCAL ONLY (Open Sparse Stack rule).
