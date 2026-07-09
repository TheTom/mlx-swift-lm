# ffai coopmat e2e prefill consumer — canonical patch

**Patch:** `ffai_coopmat_e2e.patch`
**Applies clean on:** ffai `672346d` (TheTom/ffai `main` — on-device MoE + SSD-fused default-on).
Verified: `git apply --check` CLEAN against a pristine `git archive 672346d` extraction
(independent of the box working tree). 4 files, 429 insertions, atomic (the new test
file is a proper `new file` hunk).

```
git -C <ffai @672346d>/  apply  .coopmat-final/ffai_coopmat_e2e.patch
```

## What it is

The ffai **consumer** of the metaltile coopmat GEMM win. The metaltile side
(coopmat codegen, guard tests, null-stream fix, cublasGemmBatchedEx) is already
landed on canonical — tip `97a1df43` (TheTom/metaltile `feature/cuda-hip-vulkan`).
This patch wires ffai's batched Qwen prefill through that kernel.

### Files (4)

| File | Change |
|------|--------|
| `rust/crates/ffai-ops/src/lib.rs` | **+`gemm_q8_mpp`** (batched Q8 prefill GEMM → metaltile `ffai_gemm_q8_mpp`, the SimdGroup CoopTile 64×64×32 kernel; f16 staging in/out, f32 graph unchanged). **+`add_bias_rows`** (per-feature bias broadcast across token rows; `n` is a **constexpr push-constant**, not a baked `Op::Const` — the cache-key fix so one cached pipeline serves every projection width instead of silently reusing the first call's `n`). |
| `rust/crates/ffai-models/src/llama.rs` | **+`Q8Mat::gemm`** (batched projection; `FFAI_PREFILL_GEMV=1` A/B escape hatch). **+`GgufModel::prefill`** — one batched forward over all S tokens: the 7 projections + lm_head route through `gemm_q8_mpp`; attention uses **`sdpa_multi`** single-dispatch (not the per-query loop); batched RoPE + `kv_append_many`. Falls back to the per-token `step` loop when prereqs aren't met (no resident-Q8, S≤1, head_dim≠128). Decode (`gemv_q8`) untouched. |
| `rust/crates/backends/ffai-vulkan/Cargo.toml` | `[[test]] qwen25_prefill_coopmat`. |
| `rust/crates/backends/ffai-vulkan/tests/qwen25_prefill_coopmat.rs` | New test: batched-prefill argmax == sequential `step`, asserts " Paris", greedy tail, prefill tok/s @ S=64/128/256/512. |

### Coopmat gate

The GEMM is bit-transparent to the model. `MT_VK_COOPMAT=1` lights up the
Vulkan `VK_KHR_cooperative_matrix` fragment MMA path; unset → bit-exact scalar
`SoftwareLocalC` tiles. Same tokens either way, only prefill tok/s changes.

## Build + verify (RDNA4 / RX 9070 XT, Vulkan)

Clean checkouts on the box: ffai `672346d` (`C:\models\ffai-clean`), metaltile
`97a1df43` (`C:\models\mt-clean`). The metaltile `[patch]` in `rust/Cargo.toml`
is repointed to `C:/models/mt-clean` **node-locally** — that edit is deliberately
NOT in this patch (canonical keeps the git dep). Cargo.lock churn likewise excluded.

```
cargo test --release --features vulkan -p ffai-vulkan \
  --test qwen25_prefill_coopmat qwen25_1_5b_prefill_coopmat_vulkan -- --nocapture
```

Compile: clean (mt-clean 97a1df43 + ffai-clean 672346d, 2m43s cold). Correctness:
both OFF and ON predict " Paris" (argmax 12095, batched == sequential), full
continuation `"The capital of France is Paris. The capital of France"`.

### Throughput — Qwen2.5-1.5B-Q8 prefill, OFF vs ON

| S   | coopmat OFF | coopmat ON | speedup |
|-----|-------------|------------|---------|
| 64  | 29.9        | 69.1       | +131%   |
| 128 | 49.5        | 118.1      | +139%   |
| 256 | 62.2        | 165.6      | +166%   |
| 512 | 75.1        | 209.0      | +178%   |

(prompt-tok/s; OFF baseline must set `MT_VK_COOPMAT` UNSET — note a single
`cmd /c "set X=1 && cargo ..."` does NOT propagate the var due to inline
`%X%` parse-time expansion; use a `.bat` so `set` lands on its own line.)
