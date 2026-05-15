# F-83 Perf — 256K Bench Results (V1)

**Test**: `f83_perfBench256K_14B1M`
**Model**: Qwen2.5-14B-Instruct-1M-4bit
**Hardware**: M5 Max (Tom's box)
**Date**: 2026-05-15 (overnight autonomous run)
**Branch**: `feature/retrieval-attention`

## Setup

- prefillLen = 256 × 1024 = 262,144 tokens
- chunkSize = 1024
- nDecode = 8 steps (2 warmup + 6 measured)
- Sparse config: `sparsePrefillEnabled=true`, `sparsePrefillMinContext=16384`
- Random tokens (deterministic seed 0xF8302560)

## Results

### Prefill wall-clock

| Path | Total time | Speedup vs dense |
|---|---|---|
| Dense chunked (`sparsePrefillEnabled=false`) | TBD | 1.0x |
| Sparse chunked (`sparsePrefillEnabled=true`) | TBD | TBD |

### Decode latency (post-prefill 256K, median of 6 steady-state steps)

| Path | ms/step |
|---|---|
| Dense | TBD |
| Sparse | TBD |

### Quality regression catch

- Final-token logit cosine (sparse vs dense): TBD
- Target: ≥ 0.99 (PRD M5)

## Interpretation

TBD once numbers land. PRD target was ~12x speedup at 256K (after research-revisions). V1 design uses gather-based MLX-ops (no custom Metal kernel), so realistic V1 expectation is somewhere between 5x and 15x depending on how well NAX handles the smaller K dimension on the gathered path.

## Where to find raw log

`/tmp/f83_perf256k_progress.log` (live progress, ~256 chunks × 2 paths + decode)
`/tmp/f83_perf256k_test_v2.log` (full swift test stdout)

## V2 next steps (queued)

See `F83_V2_DESIGN.md`. M1 fused L>1 selector kernel is the highest-leverage win on top of V1 numbers.
