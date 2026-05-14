# F-80 — MLX 32K/49K Cliff Diagnosis (FIXED)

**Status**: ROOT CAUSE FOUND AND FIXED.
**Date**: 2026-05-14
**Branch**: `feature/retrieval-attention` (local-only)
**Fix**: TheTom/mlx@feature/f-80-allocator-zero-on-recycle:32fa8c4c
**Result**: 49K dense-vs-dense decode cosine **0.34 → 1.0** (bit-exact);
F-79 ship regression unchanged (8/8 argmax match, cosine 0.99882).

## The fix (two lines)

`mlx/backend/metal/allocator.cpp:120`:
```cpp
MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
if (buf) {
  memset(buf->contents(), 0, buf->length());
}
```

## What broke

`MetalAllocator::malloc` calls `BufferCache::reuse_from_cache` which
returns previously-freed buffers from the pool **without zeroing
contents**. Some MLX kernel reads uninitialized memory from these
recycled buffers at long-context decode. The stale data differs
between two runs depending on prior allocation patterns, producing
cross-run logit divergence at T ≥ ~32K.

`MLX.GPU.clearCache()` between two runs empties the pool, forcing
fresh allocations — that's why the cliff went away when Tom inserted
clearCache for the experiment. The proper fix is to memset on
recycle, matching that observed effect without paying the full clear
cost.

## What's NOT broken

- SDPA kernels (`sdpa_vector`, `sdpa_vector_2pass_1/2`) — tested via
  matmul-based SDPA fallback (`MLX_DETERMINISTIC_SDPA=1`); matmul also
  shows the cliff before the allocator fix, confirming SDPA was not
  the source.
- `simd_sum` / `simd_max` reductions — tested via `simd_shuffle_xor`
  butterfly replacement; no effect on cliff.
- Quantized matmul — cold-call test shows bit-determinism for `wo`,
  `down_proj`, `wq`.
- Prefill — bit-deterministic across all 48 layers (uses steel
  attention, separate code path).

## Original framing (kept for context)

## What the cliff is

Two identical model forwards (same model, same prefill tokens, same decode
tokens, same MLXRandom seed) produce divergent logits past ~32K context on
M5 Max running Qwen2.5-14B-Instruct-1M-4bit.

Measured (8 decode steps after T=49151 prefill):

```
[F-79-dense-noise-49K] dense vs dense mean_cosine=0.33977 min_cosine=0.14015
```

Qwen3-0.6B-4bit has the cliff at exactly T=32768 (cosine cliff 1.0 → 0).

## Definitive isolation results

Three diagnostic tests in `RetrievalAttentionTests.swift`:

| Test | Result | Conclusion |
|---|---|---|
| `denseCliffLayerIsolation_49K_14B1M` | cosine=1.0, maxDiff=0 across all 48 prefill layers | **Prefill is bit-deterministic** (uses steel attention, not vector SDPA) |
| `denseCliffDecodeStepIsolation_49K_14B1M` (1 step) | logit cosine=1.0, K-cache maxDiff=0 | **Single isolated decode step is bit-deterministic** |
| `denseCliffDecodeStepIsolation_49K_14B1M` (8 steps) | step 0 cosine 0.34, layer 0 K-cache maxDiff=0, layers 1-47 maxDiff 9-30 | **Multi-step decode at 49K is non-deterministic from step 0** |
| `quantMatmulDeterminism_14B1M` | wo / down_proj / wq maxDiff=0 | **Quantized matmul is bit-deterministic** (when called in isolation) |

Critical observation: the SAME test code in a "prefill + 1 decode" structure
is bit-deterministic, but in a "prefill + 8 decode" structure step 0 already
diverges. The difference is GPU-state contention from the longer run.

## Root cause

This is the **Thinking Machines batch-invariance attention non-determinism**
in its Apple Silicon flavor. The culprit is `simd_sum` / `simd_max` in
`sdpa_vector` and `sdpa_vector_2pass` kernels. Per Awni Hannun in
[ml-explore/mlx#878](https://github.com/ml-explore/mlx/issues/878):

> "It's expected that run to run won't be bitwise identical as the atomics
> (and possibly SIMD reductions) are not deterministic."

Why simd_sum is non-deterministic across launches: on Apple GPUs, the reduction
order across simdgroup slots within a threadgroup depends on which slots arrive
first at the barrier, which is a function of memory-system / dispatcher state.
Step 1 races against a clean cache; step 2 races against a cache hot from
step 1, with KV at a different length, so the simdgroup arrival order on the
cross-simd reduction permutes. Floating-point non-associativity introduces a
few ULPs of drift per call. The drift writes into the K-cache at the new
position, the next step reads slightly different K, and the error compounds.

## Failed quick fixes (so the next investigator doesn't retry)

| Attempt | Result |
|---|---|
| Patch pass-2's `simd_max` / `simd_sum` / `simd_sum` to sequential threadgroup-memory reductions | **Identical 0.34 cosine** — pass-2 reductions are NOT the source |
| Force single-pass `sdpa_vector` dispatch via `use_2pass = false` | **Identical 0.34 cosine** — single-pass kernel has the same simd_sum |
| Patch pass-1's per-key `simd_sum(score)` with `simd_shuffle_xor` butterfly | **Identical 0.34 cosine** — per-key reduction is NOT the source |
| `MLX_DISABLE_NAX=1` env | No effect (also: actual env var name is `MLX_METAL_NO_NAX`, and NAX is prefill-only anyway) |
| `MLX_ENABLE_TF32=0` env | No effect |
| `MLX_SDPA_NO_2PASS=1` env (added but functionally same as forcing single-pass) | No effect |

## Smoking gun (proves metallib swap works)

Replaced `score = simd_sum(score)` with `score = 999.0` (intentionally broken) in pass-1.
Cosine dropped from 0.34 → 0.10. Confirms patched metallib IS being loaded
at runtime — the kernel-level patches that didn't move cosine were genuinely
ineffective, not silently bypassed.

## What WOULD fix it

**Port [ProbioticFarmer/mlx-deterministic](https://github.com/ProbioticFarmer/mlx-deterministic)
into the mlx-swift fork**. Their approach: FlashAttention with fixed split-size.
Each split's reduction has a deterministic-by-construction order because the
splits are sized to the GPU's simdgroup-per-threadgroup count, not whatever the
dispatcher schedules.

Estimated scope (multi-day):

1. New kernel `sdpa_vector_deterministic` with explicit per-simdgroup
   reduction order (sequential threadgroup-memory reduction; no `simd_sum`)
2. Dispatcher gate `MLX_DETERMINISTIC=1` env var in
   `scaled_dot_product_attention.cpp:766` to route to the deterministic kernel
3. Rebuild metallib via `mlx-swift-lm/scripts/build-metallib.sh debug`
4. Validate on Qwen3-0.6B (cliff at exactly 32768) and Qwen2.5-14B-1M
   (cliff in 32K-64K band) — both should go to cosine=1.0
5. Perf regression characterization (expected ~3-5x decode slowdown when
   determinism is on; default off)

## Practical impact

- **F-79 amort=16 at 32K**: bit-clean. Ships. Cosine 0.99882, 8/8 argmax match.
- **F-79 amort=16 at 49K**: hits cliff, but cliff is in DENSE too. RA is not
  more affected than dense.
- **49K band**: lies between MLX cliffs (32K and ~64K). MLX-level
  non-determinism, not RA-specific.

## Open decision

The cliff is a real Apple Silicon Metal issue. Three paths:

- **A** — Multi-day port of ProbioticFarmer's deterministic SDPA into fork.
  Real fix. Behind opt-in env flag (no perf regression for the 99% case).
- **B** — Cap RA default `maxRetrievedContext` at 32K; document cliff in
  user-facing docs. Ships now.
- **C** — Accept as known limitation. F-79 amort=16 at 32K is already
  publishable. No additional work.

## Diagnostic tests (kept in `RetrievalAttentionTests.swift`)

- `denseCliffLayerIsolation_49K_14B1M` — prove prefill bit-determinism
- `denseCliffDecodeStepIsolation_49K_14B1M` — prove single decode is
  deterministic but multi-step diverges from step 0
- `quantMatmulDeterminism_14B1M` — prove quant matmul cold-test is deterministic

## Sources

- [ml-explore/mlx#878](https://github.com/ml-explore/mlx/issues/878) — Awni
  confirms simd reductions non-deterministic
- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [LMSYS / SGLang Deterministic Inference](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [ProbioticFarmer/mlx-deterministic](https://github.com/ProbioticFarmer/mlx-deterministic)
- [ml-explore/mlx PR #3473 — MLX_NUMERICAL_STRICT_MODE precedent](https://github.com/ml-explore/mlx/pull/3473)
- [Aditya Karnam — The Hidden Problem With MLX](https://adityakarnam.com/mlx-non-determinism-apple-silicon/)
