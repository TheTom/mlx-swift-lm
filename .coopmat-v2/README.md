# coopmat-v2 — canonical-applicable deliverables (RX 9070 XT / RDNA4)

Two clean, conflict-free patches regenerated + re-validated on the AMD box
(native Windows, `pidto`, trees under `C:\models`, AMD driver 26.x, Vulkan SDK
1.4.350, Rust nightly-2026-05-15 msvc). No commits pushed — land these.

Box state on arrival: the REAL native-Windows env was intact (NOT the bork WSL).
`C:\models` held prior metaltile/coopmat-work/ffai trees, but the `metaltile`
checkout was dirty (1078 line-ending-churned files at base 4b303e1) and the
`ffai` checkout was a divergent older line. So both were re-established from the
Mac canonical commits (metaltile 9565fdf0, ffai ceeb478) via `git archive` →
`tar` → fresh `git init` baseline, and the work re-applied/re-validated there.

═══════════════════════════════════════════════════════════════════════════
## 1. coopmat_v2.patch — metaltile gated VK_KHR_cooperative_matrix GEMM codegen
═══════════════════════════════════════════════════════════════════════════
`git apply --check` CLEAN on canonical **metaltile 9565fdf0**. (Verified on the
Mac canonical worktree AND a fresh box extraction.)

Applies to 4 source files + 1 new test (vulkan_coopmat_gemm.rs, also in this dir):
- `crates/metaltile-codegen/src/backend.rs` — `MT_VK_COOPMAT=1` flips the
  vulkan() profile MMA strategy SoftwareLocalC → VkCooperativeMatrix.
- `crates/metaltile-codegen/src/spirv/mod.rs` — coopmat extensions; fp16
  `_CTA`/`_CTB` staging tiles; register-blocked `coopmat acc[MF][NF]`;
  coopMatLoad/MulAdd/Store with A-tile reuse.
- `crates/metaltile-runtime/src/device/vulkan/ffi.rs` — cooperativeMatrix
  features struct + STRUCTURE_TYPE + extension-name const.
- `crates/metaltile-runtime/src/device/vulkan/mod.rs` — when MT_VK_COOPMAT=1,
  enables vulkanMemoryModel + chains the cooperativeMatrix feature + the
  VK_KHR_cooperative_matrix device extension. (This file's hunks were
  RECOVERED from the box's prior work — they were MISSING from the original
  .coopmat-deliverable patch, which would have left coopmat ON crashing at
  pipeline create. Now included.)

### Difference vs the original .coopmat-deliverable/coopmat_clean.patch
The original was cut against box base 4b303e1, which predates three commits now
in canonical 9565fdf0:
  - e8a1427e (npot GLSL Op::Reduce fix) — the patch's npot spirv hunk + its test
    assertion are ALREADY in canonical → DROPPED (redundant, would conflict).
  - the `vkFreeCommandBuffers` ffi declarations are ALREADY in canonical → DROPPED.
  - the `device/vulkan/mod.rs` runtime feature-enablement was MISSING from the
    original patch → ADDED here (the deliverable was incomplete without it).

### Validation on RDNA4 (box, this session)
- metaltile-codegen unit tests: **184 passed** (+1 ignored), no regression.
- coopmat gemm_q8_mpp vs f32 CPU oracle: maxRelDiff **0.0010** (MT_VK_COOPMAT=1)
  vs **0.0308** (scalar OFF) — coopmat is *more* accurate (f32 fragment accum).
- Device creates WITH the cooperativeMatrix feature, no fallback (the shader
  uses coopMatMulAdd, which would crash if the runtime hadn't enabled it —
  proves the mod.rs enablement works).
- Qwen2.5-1.5B-Q8 GGUF generates identical "Paris" token IDs OFF and ON.

═══════════════════════════════════════════════════════════════════════════
## 2. ffai_prefill_coopmat.patch — e2e Qwen prefill routed through coopmat GEMM
═══════════════════════════════════════════════════════════════════════════
`git apply --check` CLEAN on canonical **ffai ceeb478** (root has `rust/`).
Realizes the e2e prefill win the original deliverable's "What's left #1" called out.

Files:
- `rust/crates/ffai-ops/src/lib.rs` — NEW ops:
  - `gemm_q8_mpp(qs, scales, x[n_rows,k], n_rows, m, k)` — batched Q8 GEMM via
    metaltile's `ffai_gemm_q8_mpp` SimdGroup CoopTile kernel; picks up coopmat
    when MT_VK_COOPMAT=1. f32 x cast→f16 in, f16 result widened→f32 out.
  - `add_bias_rows(x[n_rows,n], bias[n], ...)` — broadcast-add the QKV bias to
    every prefill row. (Bias buffers are DEVICE-LOCAL; broadcast is done by
    copying bias to a host-visible tensor via `ffai_slice`, tiling on host,
    one `add`. A custom GLSL modulo/index broadcast kernel was tried first and
    MIS-broadcast on the Vulkan backend — see notes.)
- `rust/crates/ffai-models/src/llama.rs` — `GgufModel::prefill(tokens, start)`:
  batched multi-token forward. The seven projections + lm_head run as batched
  `[S,k]·Wᵀ` GEMMs through `gemm_q8_mpp` (the coopmat target). Attention uses
  per-query `sdpa_decode` (the proven decode kernel) over the causal prefix —
  `sdpa_multi`'s batched flash kernel is NOT yet validated on Vulkan. Decode
  `step` (gemv_q8) is untouched. S≤1 / no-Q8 / head_dim≠128 fall back to step.
  (A gated `FFAI_PREFILL_GEMV=1` debug knob routes projections through per-row
  gemv_q8 for A/B isolation.)
- `rust/crates/backends/ffai-vulkan/Cargo.toml` + NEW
  `tests/qwen25_prefill_coopmat.rs` — correctness (batched next-token argmax ==
  sequential, == "Paris") + prefill throughput at S∈{64,128,256,512}.

### NOT included (node-local): `rust/Cargo.toml` [patch] re-point
On the box the `[patch."https://github.com/TheTom/metaltile"]` paths were
re-pointed from `../../metaltile-cuda/...` to `C:/models/metaltile-canon/...`
(the coopmat-patched checkout). That edit is machine-specific — left OUT of the
patch. On your machine the existing `../../metaltile-cuda` sibling (with
coopmat_v2.patch applied) already resolves it.

### e2e prefill throughput — Qwen2.5-1.5B-Q8, Vulkan/RDNA4 (this session)
Same model, same code; only `MT_VK_COOPMAT` flips. Both produce "Paris".

| prompt S | coopmat OFF | coopmat ON | speedup |
|---------:|------------:|-----------:|--------:|
|       64 |   8.5 tok/s |  11.4 tok/s|   +34%  |
|      128 |  10.4 tok/s |  12.5 tok/s|   +20%  |
|      256 |  11.6 tok/s |  13.2 tok/s|   +14%  |
|      512 |  12.1 tok/s |  13.5 tok/s|   +12%  |

The kernel-level coopmat GEMM win is 6.2× (deliverable #1). The e2e prefill gain
is smaller and bounded by the non-GEMM ops still in the prefill path — chiefly
the per-query `sdpa_decode` loop (S dispatches + host round-trips) and the
host-tiled bias add. Routing prefill attention through a Vulkan-validated batched
flash kernel (fix `sdpa_multi` on RDNA4) is the next lever to convert more of the
6.2× GEMM win into e2e tok/s.

═══════════════════════════════════════════════════════════════════════════
## Follow-ups worth landing as separate tasks
- **sdpa_multi is wrong on Vulkan/RDNA4** for multi-position attention (its
  cross-subgroup online-softmax reduction). It's not in the Vulkan kernel
  corpus. Fixing it lets prefill attention run in ONE dispatch instead of the
  S-dispatch sdpa_decode loop used here.
- **A GLSL broadcast-index kernel mis-computes** `out[i] = x[i] + bias[i % n]` /
  `i - (i/n)*n` on the Vulkan backend (wrong bias on rows ≥ 1). The proven
  `softplus_add_rows` uses the same `i - (i/n)*n` form with a *constexpr* `n`;
  the failing version used an `Op::Const` literal. Worth root-causing — would
  let `add_bias_rows` stay fully on-device.
