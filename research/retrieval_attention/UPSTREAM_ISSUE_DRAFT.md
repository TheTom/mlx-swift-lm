# Draft: Upstream MLX issue — non-deterministic SDPA above 32K context

**Status**: DRAFT. Not yet filed. Requires Tom's explicit approval before
posting to https://github.com/ml-explore/mlx/issues (standing rule: never
push without explicit per-push confirmation).

---

## Suggested title

> SDPA: same model + same input + same seed produces non-deterministic
> output past ~32K context (M5 Max, Apple Silicon)

## Body

**Hardware**: Apple M5 Max (40-core GPU, 128 GB unified)
**OS**: macOS 15.x
**MLX**: vllm-swift-stable fork branch (matches recent upstream HEAD)
**Model**: Qwen2.5-14B-Instruct-1M-4bit (GQA, nQH=40, nKVH=8, D=128)

### What I observed

Running the same model with the same input twice — deterministic random
seed → deterministic prefill tokens → deterministic force-fed decode
tokens — produces different logits at prefill ≥ ~32K. Comparing the
two runs' per-step logit vectors:

| Test | Mean logit cosine | Min |
|------|-------------------|-----|
| dense × 2 runs @ T=49151, 8 steps | 0.339 | 0.139 |
| dense × 2 runs @ T=49151, 4 steps post-warmup | 0.205 | 0.083 |
| dense × 2 runs @ T=16384 | (essentially 1.0 — stable regime) |
| dense × 2 runs @ T=32767 | 0.99960 vs amort=1 RA reference (stable) |

Two runs of identical code on identical input should be bit-identical.
They are not, by a wide margin, at T > 32K.

### Env-var workarounds attempted (none fixed it)

| Setting | Mean cosine |
|---------|-------------|
| (default) | 0.339 |
| `MLX_ENABLE_TF32=0` | 0.339 (identical) |
| `MLX_DISABLE_NAX=1` | 0.339 (identical) |

### Reproduction

Run this Swift xctest (will be attached in source):
`mlx-swift-lm/Tests/MLXLMTests/RetrievalAttentionTests.swift:denseNonDeterminism_49K_14B1M`

Steps:
1. Load Qwen2.5-14B-Instruct-1M-4bit (or any 14B with GQA + long context)
2. MLXRandom.seed(0x4910)
3. Generate prefillTokens of length 49151
4. Generate 8 forceTokens
5. Run model(prefillTokens) + 8 model(forceTokens[s]) → record logits per step (run A)
6. Run again with a freshly-allocated cache → run B
7. Compare logits per step. Cosine drops to ~0.34 mean.

### Suspected root cause

`sdpa_vector_2pass_2` combine step in
`mlx/backend/metal/kernels/sdpa_vector.h:340-394`. At K seq ≥ 1024 (or
≥ 4096 for GQA), the dispatcher
(`mlx/backend/metal/scaled_dot_product_attention.cpp:766-768`) routes
to the 2-pass kernel. The pass-2 combine does `simd_max` over
block-level partial maxes + `simd_sum(factor * sums[...])` with
`factor = fast::exp(maxs[...] - max_score)`. With K=49K we have ~64
partial blocks. If `simd_max` / `simd_sum` reduction order is not
strictly deterministic on Apple GPU, drift accumulates through
log-sum-exp and cascades through 48 transformer layers.

Per @awni in #878:
> "It's expected that run to run won't be bitwise identical as the
> atomics (and possibly SIMD reductions) are not deterministic."

This is consistent with what I'm seeing, but the magnitude (cosine 0.34
between two model forwards) seems severe — most users would expect
visually-coherent generation to remain at least logit-cosine ≥ 0.99
across runs.

### Ask

1. Is the magnitude of run-to-run drift at T > 32K expected behavior,
   or has something regressed in the 2-pass combine?
2. Could a `MLX_DETERMINISTIC=1` env flag be added to force fp32
   accumulation + ordered reductions in `sdpa_vector_2pass_2`? The
   community's prior `MLX_NUMERICAL_STRICT_MODE` proposal (PR #3473)
   was closed for 2.3× slowdown, but for long-context inference where
   throughput is already memory-bound the slowdown should be smaller.
3. Or — accept fp32 K/V cache as a workaround. Currently
   `sdpa_vector.h` already does `typedef float U` internally, so the
   accumulator IS fp32. The drift is from reduction *order*. Either
   document this loudly in the SDPA API or expose a deterministic
   variant.

I can attach the xctest as a minimal Swift repro and produce a Python
equivalent if helpful.

---

## How to file

If/when approved:

```bash
gh issue create \
  --repo ml-explore/mlx \
  --title "SDPA non-deterministic output past ~32K context (M5 Max)" \
  --body-file research/retrieval_attention/UPSTREAM_ISSUE_DRAFT.md
```

Or just paste into the GitHub web UI. Attach the xctest source from
the local-only `feature/retrieval-attention` branch.
