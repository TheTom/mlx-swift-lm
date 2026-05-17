# F-85 Bench Results — Honest Report

Date: 2026-05-16
Branch: `feature/retrieval-attention` (local only)
Build: `swift build -c release --package-path swift` (vllm-swift), latest
`mlx-swift-lm` checkout with Phase 1-4 commits.

## TL;DR — F-71b at B=8 does NOT win

Predicted (R2 bandwidth math): B=8 T=32K sparse compose-only ≈ 71ms/step vs
dense ≈ 121ms = +1.7× speedup.

Measured: B=8 T=32K sparse via F-71b = **2260ms/step** vs dense ≈ **162ms/step**
= **14× SLOWER**, not faster. Per-slot 67× slower than serial B=1 sparse.

Smoke + plumbing all work — the F-71b kernel runs at B=8 with the per-(B, KV)
gather and emits per-slot-distinct outputs (verified at unit-test layer in
Phase 3). The path through Bridge fires (`[vsm] f85 batched=true B=8 tokens
emitted=8` confirmed). The numbers just aren't competitive.

## Measurements

Model: `Qwen2.5-14B-Instruct-1M-4bit`. Hardware: M5 Max. All numbers are
decode-only tok/s reported by `bench_throughput.py` (excludes prefill).

### B=2 ctx=4K smoke (below sparseMinContext, all dense fallback)

```
[vsm] f85 batched-sparse cache built: B=2 T=4096 nKVH=8 D=128 layers=48 maxSeq=4352
[vsm] f85 batched=true B=2 tokens emitted=2  (×7)
B=  2: e2e=2.1 decode=64.4 tok/s
```

→ smoke passes. Plumbing alive.

### B=8 ctx=32K — the headline number

| Path | tok/s (decode) | per-slot | per-step ms |
|---|---|---|---|
| Dense B=8 | 49.3 | 6.16 | 162 |
| **F-85 batched-sparse B=8 (F-71b)** | **3.5** | **0.44** | **2260** |
| Serial B=1 sparse (F-73 mask) | 29.9 | 29.9 | 33.5 |

→ F-85 batched-sparse is **14× slower than dense B=8**, **67× slower per slot
than B=1 sparse**.

### Per-step decomposition (from `--tokens 3`)

```
B=  8: decode=6.16s for 24 tokens (3 per slot)
    → 2052 ms/step batched-sparse forward
```

## Why it fails

### 1. F-71b threadgroup layout doesn't scale to B=8

The F-71b kernel uses BN=32 simdgroups × BD=32 lanes per Q head (matches
mlx-swift `sdpa_vector` layout). Grid = `B * nQH` threadgroups.

- B=1, nQH=40: 40 threadgroups × 32 simdgroups = 1280 simdgroups requested
- B=8, nQH=40: 320 threadgroups × 32 simdgroups = 10240 simdgroups requested

M5 Max has 40 SMs × ~12 simdgroups concurrent = ~480 simdgroups in flight at
once. At B=1 the kernel saturates the GPU; at B=8 we have **21× more work**
but only **3× more concurrency room** → 7× serialization on top of the work
increase. R2's bandwidth analysis assumed K/V bandwidth would dominate; in
practice compute scheduling dominates at this kernel shape.

This matches the FINDINGS warning: "F-71b regressed 1.4× vs F-73 at B=1
32K." The per-Q-head threadgroup design is the wrong granularity for
batched sparse at the F-71b kernel's current shape on Apple Silicon.

### 2. Selector index not populated

v1 of the F-85 cache build does NOT migrate per-session
`BatchedRetrievalAttentionIndex` state into the new batched index — TODO
flagged in `buildBatchedSparseCaches`. At decode the kernel sees an empty
fine-block-feature buffer → top-K picks blocks 0..K-1 by index order → gather
is basically `[0..staticInit) ∪ [T-sliding..T) ∪ [0..k*BS)` = ~10K positions
of T=32K, ~30% of T. Real top-K from the selector would pick higher-quality
~5K positions (~15% of T). This explains some of the gap but not 14×.

### 3. F-71b's per-step shape doesn't match dense's batched fast path

Dense B=8 at 32K goes through `BatchedKVCache.attention` → MLX `SDPA` on
shape `[8, 40, 1, 128] x [8, 8, 32K, 128] x [8, 8, 32K, 128]`. MLX's
`sdpa_vector_2pass` is heavily tuned for this exact shape and uses
mask=`.none` (no scatter). F-71b dispatches a custom kernel with random K
gather indices — random reads on M-series GPUs cost 2-4× vs coalesced reads.

## Conclusion

R3 external lit's claim that fused gather wins at B≥8 is true IN PRINCIPLE
(NSA/SeerAttention/FSA papers) but those papers target CUDA where
FlashAttention's sparse kernel is mature. **On Apple Silicon Metal,
mlx-swift's `sdpa_vector_2pass` with random-access mask is faster than a
hand-rolled sparse kernel.** This was also the F-71b finding at B=1
(FINDINGS line 274-278) and F-69 / F-76 finding (line 322-331).

The F-83 sprint had to amortize the SELECTOR cost (F-79) instead of fusing
the attention compute, because the kernel itself was already kernel-bound,
not bandwidth-bound. F-85 hits the same wall at B>1.

## Paths forward (NOT shipped)

1. **Run F-85 plumbing through F-73 (mask-build)** instead of F-71b. The
   F-73 mask kernel is per-position not per-Q-head, so B=8 just bumps the
   threadgroup count from `T/256` to `T/256` (same — mask is shared across
   B). Each layer's SDPA call becomes `SDPA(Q[B,nQH,1,D], K[B,nKVH,T,D],
   mask=[B,1,1,T])` which hits sdpa_vector_2pass's mask=.array fast path.
   **Predicted win at B=8 ≈ 1.5× over dense at 32K**, not 6× — but at
   least we'd be in the right direction.
2. **Custom batched sparse kernel that uses the GQA group co-load pattern
   from F-71b but with COALESCED K reads** (e.g. per-(B, KV) ordered gather
   so adjacent threads in a simdgroup read adjacent K rows). Engineering
   effort: weeks; risk of yet another null result.
3. **Pre-populate the batched selector index** from per-session state at
   migration time. Closes some of the quality gap, doesn't fix the kernel
   regression.

The honest call: F-71b is not the right kernel for batched sparse on M-series.
Path 1 (F-73 batched) is the next experiment if Tom wants to keep pushing
on F-85. The Phase 2-4 infrastructure (BatchedRetrievalAttentionIndexB +
BatchedRetrievalAttentionKVCache + Bridge wiring) is reusable for it.

## Files

- `Libraries/MLXLMCommon/BatchedRetrievalAttentionIndexB.swift` (Phase 2)
- `Libraries/MLXLMCommon/BatchedRetrievalAttentionKVCache.swift` (Phase 3)
- `Libraries/MLXLMCommon/Models/Qwen2.swift` (`fullyBatchedSparse*` triplet)
- `Libraries/MLXLLM/Models/Qwen2.swift` (`Qwen2Model.fullyBatchedSparseDecode`)
- `vllm-swift/swift/Sources/VLLMBridge/Bridge.swift`
  (`batchedSparseDecodeAll` + `buildBatchedSparseCaches`)

All commits on `feature/retrieval-attention` branch, both repos. Local only
per Open Sparse Stack rule.
