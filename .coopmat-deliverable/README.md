# metaltile Vulkan coopmat GEMM codegen — RDNA4 deliverable

Gated `VK_KHR_cooperative_matrix` GEMM path in the metaltile SPIR-V/GLSL emitter.
Dev + test on the RX 9070 XT box (`/mnt/c/models/metaltile`, branch
`feature/cuda-hip-vulkan`, base `4b303e10`). No commits pushed — clean to land on PR #271.

## Gate
`MT_VK_COOPMAT=1` flips `TargetProfile::vulkan().mma` from `SoftwareLocalC`
(bit-exact scalar) to `MmaStrategy::VkCooperativeMatrix`. Unset = unchanged.
A/B-able at runtime; falls back where coopmat is unavailable.

## Files changed (see coopmat_clean.patch)
- `crates/metaltile-codegen/src/backend.rs` — env gate on the vulkan() profile.
- `crates/metaltile-codegen/src/spirv/mod.rs` — `coopmat_on()` helper; coopmat
  extensions in preamble; fp16 `_CTA`/`_CTB` staging tiles; register-blocked
  `coopmat acc[MF][NF]` declared in emit_body; CoopTileZero/Run/StoreC emit
  `coopMatLoad`/`coopMatMulAdd`/`coopMatStore` with **A-tile reuse** (matA loaded
  once per (kt,mi) K-step, reused across the NF matB column fragments) — the
  structure that hit the win, NOT the naive one-tile-per-subgroup form.
- `crates/metaltile-runtime/src/device/vulkan/ffi.rs` — `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`
  struct + STRUCTURE_TYPE + extension-name const.
- `crates/metaltile-runtime/src/device/vulkan/mod.rs` — when `MT_VK_COOPMAT=1`,
  enables `vulkanMemoryModel` + chains the cooperativeMatrix feature + enables
  the `VK_KHR_cooperative_matrix` device extension. subgroupSize stays 32
  (wave32: the 16×16 fragment spans 32 lanes; the CoopTile staging math assumes
  32-lane subgroups, so no per-kernel size change is needed and nothing else
  regresses).
- `crates/metaltile-std/tests/vulkan_coopmat_gemm.rs` — NEW. Drives the real
  `ffai_gemm_q8_mpp` 64×64×32 SimdGroup CoopTile kernel through
  `VulkanDevice::run_kernel`; validates vs an f32 CPU Q8-dequant oracle.

## Validation (on RDNA4, AMD driver 26.3.1, Vulkan 1.4.344)
- **Generated GLSL** has the target structure: `_CMAcc_gemm[2][2]` fp32
  accumulator fragments; outer `_kt`, then `_mi` loads matA once, inner `_ni`
  loads matB + coopMatMulAdd → A-reuse confirmed.
- **Correctness**: coopmat gemm_q8_mpp matches f32 oracle to maxRelDiff **0.0010**
  (vs 0.0308 for the scalar path — coopmat is *more* accurate, f32 fragment accum).
- **No regression (coopmat OFF, default)**: Qwen2.5-1.5B-Q8 GGUF generates "Paris"
  on Vulkan/RDNA4; metaltile-codegen unit tests 184+17 pass.
- **No regression (coopmat ON)**: Qwen still generates identical "Paris" token IDs;
  device creates with the cooperativeMatrix feature (no fallback).

## Throughput (standalone microbench, /mnt/c/models/coopmat-work, bit-exact)
Tuned coopmat (A-reuse + reg-block, sg=64) vs scalar, fp16→fp32:
- 2048³: coopmat **29.83 TFLOP/s** vs scalar-tiled 4.79 (**6.2×**), scalar-regblk 13.61 (2.2×). rel err 0.0.
- 4096³: coopmat **39.14 TFLOP/s** vs scalar-tiled 6.93 (**5.6×**), scalar-regblk 20.01 (2.0×). rel err 0.0.

## What's left
1. **e2e Qwen prefill speedup not yet realized**: ffai's Qwen2.5 Vulkan forward
   pass still uses the `gemv`/`gemv_q8` Reduction path for prefill, NOT
   `gemm_q8_mpp` (ffai-ops/src/lib.rs:461 — "batched/prefill cooperative matmul
   is a separate kernel, wired later"; `gemm_q8_mpp` has no caller in the model
   code). The coopmat lever is correct + fast at the kernel level but won't move
   Qwen tok/s until ffai routes prefill (n_tokens > ~16) through the MPP GEMM.
   That's an ffai-side change, separate from this metaltile codegen task.
2. **subgroupSize 64 variant**: the standalone 70-TFLOP/s peak used sg=64 +
   direct-from-global loads (2 subgroups along M, 16×NTILE strip). The codegen
   path uses the existing kernel's shared-staged 32×32 SimdGroup tiles at sg=32,
   which is correct and still ~6× over scalar but leaves headroom. A sg=64
   kernel variant + per-kernel requiredSubgroupSize=64 (the FFI struct already
   exists) is the next perf step.
3. **Generalization**: tested on m=n=k=32 (the gemm_q8_mpp tile). The emitter
   handles any m,n,k divisible by 16; other CoopTile kernels (moe_mpp*, bm64,
   gemm_q4_mpp) should work but weren't individually run this session.
