# F-85 v3.2 longctx-realistic bench grid

**2026-05-17 — Qwen2.5-14B-Instruct-1M-4bit, M5 Max**

F-73 batched sparse decode with Bridge auto `MLX_SDPA_BLOCKS=128` (v3.2 cliff fix) vs dense. 19 cells matched to longctx workload patterns (top_k=8 × 2K chunks = 16K typical, multi-turn V3 rescue 24-48K+).

## Grid

| B | ctx | dense prefill | dense decode | F-73 prefill | F-73 decode | **decode ratio** | **prefill ratio** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16K | 14.3s | 38.8 | 12.9s | 39.4 | 1.02× | 0.90× |
| 2 | 16K | 29.0s | 56.4 | 27.1s | 58.4 | 1.04× | 0.94× |
| 2 | 20K | 48.9s | 51.5 | 36.3s | 55.2 | 1.07× | 0.74× |
| 4 | 16K | 58.4s | 65.8 | 55.7s | 78.5 | **1.19×** | 0.95× |
| 4 | 24K | 123.2s | 53.5 | 99.0s | 62.3 | **1.16×** | 0.80× |
| 4 | 32K | 192.0s | 39.0 | 145.8s | 53.5 | **1.37×** | 0.76× |
| 4 | 48K | 347.1s | 29.6 | 267.7s | 43.7 | **1.48×** | 0.77× |
| 8 | 16K | 126.0s | 59.2 | 102.1s | 97.7 | **1.65×** | 0.81× |
| 8 | 20K | 166.4s | 56.0 | 138.5s | 89.0 | **1.59×** | 0.83× |
| 8 | 28K | 342.2s | 44.0 | 220.2s | 76.2 | **1.73×** | 0.64× |
| 8 | 32K | 431.1s | 35.2 | 267.3s | 45.1 | **1.28×** | 0.62× |

(B=8 F-73 numbers from v3.2 validation grid run separately)

## Mapping to longctx workloads

| longctx scenario | cell | win |
|---|---|---|
| Single-turn `longctx ask` user | B=1 ctx=16K | parity 1.02× |
| 2 MCP clients (Claude+OpenCode) | B=2 ctx=16-20K | 1.04-1.07× |
| 4 concurrent sessions | B=4 ctx=16-24K | 1.16-1.19× |
| V3 rescue multi-turn pile-up | B=4 ctx=32-48K | **1.37-1.48×** |
| 8 agent-farm concurrency | B=8 ctx=16-28K | **1.59-1.73×** |
| Heavy long-ctx batch | B=8 ctx=32K | 1.28× (post-cliff fix) |

## Headlines

- **Best decode**: B=8 ctx=28K = **1.73×** (44.0 → 76.2 tok/s)
- **Best prefill saving**: B=8 ctx=32K = **1.62× faster** (saved 164s = 38% of total prefill)
- **Long-ctx win grows with both B and ctx** until OOM cap
- Single-stream B=1: parity (sparse needs batch to amortize selector cost)

## Ship config

```bash
# In Bridge (already auto-set in vsm_engine_create):
export MLX_SDPA_BLOCKS=128  # bypass M5 Max cliff
# Per-request:
VSM_SPARSE=1 VSM_SPARSE_BATCHED=1 VSM_SPARSE_BATCHED_KERNEL=f73 VSM_SPARSE_NO_ADAPTIVE=1
```

Bridge auto-fallback to dense when `B * ctx > 224K` (still gated for safety, even though v3.2 mitigates the cliff at 32K).

## What didn't work

- **F-71b custom Metal kernel**: 14× slower than dense at B=8 ctx=32K (grid oversub on M5 Max)
- **Compose-gather**: 13.7× slower (per-slot take materializes B independent slabs)
- **Bool mask**: 25% regression (`_boolmask` MLX kernel variant worse than `_floatmask` on M5 Max)
- **PR #3023 MLX default**: blocks=256 causes the cliff (M5 Max needs blocks=128)

## Files

- Raw logs: `/tmp/f85_longctx_grid/` + `/tmp/v32_validate/`
- Bridge auto-setenv: `vllm-swift` commit `6b8e533`
- MLX backport: `mlx-swift-lm` commit `081e28b` (submodule bump for `7982330e`)
- Leak fix: `vllm-swift` commit `8a5bf88`
