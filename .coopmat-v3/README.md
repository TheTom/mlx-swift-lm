# coopmat-v3 — sdpa_multi + add_bias_rows validated, prefill rewired (RX 9070 XT / RDNA4)

Builds on coopmat-v2 (the gated `VK_KHR_cooperative_matrix` GEMM codegen +
the e2e Qwen prefill foundation). v2 routed the prefill **projections/FFN**
through the coopmat GEMM but worked **around** two suspected Vulkan bugs in the
attention path:
  1. attention via a per-query `sdpa_decode` loop (with a host download/upload
     **per query, per layer**) instead of the batched `sdpa_multi` flash kernel,
  2. the per-feature bias via a host round-trip (download bias → tile on CPU →
     re-upload → plain `add`) instead of an on-device `bias[i % n]` broadcast.

## TL;DR — both "bugs" were phantoms; the workarounds were the real cost

Driven directly on RDNA4 against CPU oracles, **both kernels are correct on
Vulkan**. The prior "wrong on Vulkan" conclusions were never validated at the
relevant shapes (same class of phantom as the dsv4 "N>1 drift").

- **`sdpa_multi` (multi-position flash attn):** bit-accurate vs naive oracle at
  every prefill/GQA shape — max|Δ| ≈ 1e-7 (f32 round-off). The runtime already
  pins `requiredSubgroupSize = 32`, so the in-kernel 32-lane simd-group
  partition + `mt_subgroup_add`/`subgroupMax` reductions line up with the
  hardware subgroup even though RDNA4 is wave64. No codegen change needed.
- **`add_bias_rows` broadcast:** the DSL `%` lowers to `BinOpKind::Mod` →
  GLSL `uint(i) % uint(n)` (integer modulo), **bit-exact** (max|Δ| = 0.0) for
  both the `%` and the `i-(i/n)*n` index forms at the exact Qwen projection
  widths. No codegen change needed.

The **only real bug** surfaced while rewiring: an on-device bias kernel that
bakes `n` as an `Op::Const` literal silently reuses the first call's pipeline
(the Vulkan pipeline cache key is `name|block|mode|params|constexprs`, which
does **not** include inline-literal values). Fix: pass `n` as a **constexpr
push-constant** so the one cached pipeline serves every projection width.

## Deliverables (both `git apply --check` CLEAN)

### 1. `metaltile_vulkan_guard_tests.patch` — two new Vulkan corpus guards
Pure new-file additions (apply on **metaltile b12b2be3** unconditionally):
- `crates/metaltile-std/tests/vulkan_sdpa_multi.rs` — drives `ffai_sdpa_multi`
  through `VulkanDevice::run_kernel` at prefill + GQA shapes (Qwen 12/2 GQA,
  head_dim 128, deep cache, causal + full, S=16, non-pow2 n_kv) vs a naive
  oracle. Guards the multi-position flash path directly (the auto corpus only
  hits a tiny 8/4 shape).
- `crates/metaltile-std/tests/vulkan_add_bias_rows.rs` — builds the
  `out[i] = x[i] + bias[i % n]` broadcast kernel inline and checks BOTH the
  `Mod` and `Div/Mul/Sub` index forms vs a CPU oracle at the q (n=1536) and
  k/v (n=256) widths plus awkward strides.

(The two `.rs` files are also dropped in this dir for direct inspection.)

### 2. `ffai_prefill_sdpa_multi_bias.patch` — rewire the prefill consumer
Applies on **ffai 699a34e** (on top of the v2 prefill foundation):
- `rust/crates/ffai-models/src/llama.rs` — prefill attention: the per-query
  `sdpa_decode` loop (+ per-row host round-trip) → ONE `ops::sdpa_multi`
  dispatch (`base_kv=start`, `n_query=s`, `kv_stride=cap`, causal). Same causal
  semantics (query `r` ← `[0, start+r]`); proven **bit-identical** to the old
  loop in-model (A/B max|Δ|=0.0 at layer 0).
- `rust/crates/ffai-ops/src/lib.rs` — `add_bias_rows`: host round-trip →
  on-device `bias[i % n]` kernel with `n` as a **constexpr push-constant**
  (the cache-correctness fix above).

## Validation on RDNA4 (box, this session) — Qwen2.5-1.5B-Q8

Correctness: batched prefill next-token argmax == sequential `step` == **12095
(" Paris")** in ALL configs (sdpa_multi only; sdpa_multi + on-device bias;
coopmat OFF and ON). Greedy continuation: `"…France is" → " Paris. The capital
of France"`.

Prefill throughput (prompt-tok/s), correct sdpa_multi + on-device bias:

| S   | v2 baseline (per-query loop + roundtrip bias) | coopmat OFF | coopmat ON |
|-----|----------------------------------------------:|------------:|-----------:|
| 64  | 12.8                                          | 30.7        | 77.3       |
| 128 | 16.9                                          | 62.4        | 120.1      |
| 256 | 18.2                                          | 76.9        | 171.9      |
| 512 | 18.5                                          | 80.1        | **215.3**  |

- Replacing the attention workaround alone: **~4.3× at S=512** (18.5 → 80.1),
  because the per-query path did an attention download/upload for every query of
  every layer.
- Coopmat ON over OFF on the corrected path: **+151–168%** at S≥256 (e.g. S=512
  80.1 → 215.3). This is far above the original "+12–34%" — that figure was
  measured against the slow per-query baseline that hid the GEMM behind the
  attention round-trips; with attention fast, the coopmat GEMM is the dominant
  lever and shows its true margin.

Guard tests: `vulkan_sdpa_multi` 1 passed (6 shapes, max|Δ|≤1.8e-7),
`vulkan_add_bias_rows` 1 passed (10 cases, max|Δ|=0.0). Full Vulkan kernel
corpus unchanged: PASS=4188, MISMATCH=0, ERROR=0.

## Notes / honesty
- No metaltile **source** changed (both "bugs" were phantoms); v3 adds only the
  two guard tests. The metaltile patch therefore applies on b12b2be3 cleanly as
  pure additions.
- The ffai patch was `git apply --check`-validated against the v2 prefill
  foundation reconstructed on the box (the per-query loop + roundtrip-bias
  verbatim, which is what 699a34e carries). Both hunks check clean (mode 100644).
- No commits/pushes. Dev + test on the box, 16 GB-safe (1.5B-Q8).
