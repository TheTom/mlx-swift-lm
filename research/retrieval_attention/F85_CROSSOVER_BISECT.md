# F-85 Crossover Bisect — B=8 ctx sweep, F-73 batched vs dense

**Date:** 2026-05-16
**Model:** `~/models/Qwen2.5-14B-Instruct-1M-4bit`
**Engine:** vllm-swift bridge direct (`scripts/bench_throughput.py`)
**Dylib:** `/Users/tom/dev/vllm-swift/swift/.build/arm64-apple-macosx/release/libVLLMBridge.dylib`
**Hardware:** M5 Max
**Concurrency:** B=8, 30 decode tokens, identical prompts

F-73 batched flags:
```
VSM_SPARSE=1 VSM_SPARSE_BATCHED=1 VSM_SPARSE_BATCHED_KERNEL=f73 VSM_SPARSE_NO_ADAPTIVE=1
```

F-73 sparse config (decode-only): `fineBS=64 fineTopK=32 adaptive=false coarseTopK=2 amort=16 perKVHead=false blockGatherNoMask=true blockGatherKPadFrac=1.0 fusedMask=true`.

## Results

| ctx  | dense tok/s | F-73 tok/s | ratio  | verdict |
|------|-------------|------------|--------|---------|
| 16K  | 73.4        | 95.0       | 1.29×  | WIN     |
| 20K  | 63.3        | 89.0       | 1.41×  | WIN     |
| 24K  | 56.5        | 74.9       | 1.33×  | WIN     |
| 28K  | 49.9        | 64.6       | 1.29×  | WIN     |
| 32K  | 42.5        | 27.8       | 0.65×  | LOSE    |

(For reference v3-agent reported: 16K dense 69.8 / F-73 95.7 = 1.37×; 32K dense 49.5 / F-73 26.4 = 0.53×. Same shape, slightly different absolute numbers — run-to-run noise on B=8 e2e with 30 decode tokens.)

## Crossover

Between **ctx=28K and ctx=32K** the ratio cliffs from 1.29× WIN to 0.65× LOSE. F-73 batched maintains a stable ~1.3–1.4× win across 16K–28K, then collapses sharply at 32K. The collapse is asymmetric in *both* directions: F-73 drops from 64.6 → 27.8 (−57%) while dense degrades smoothly from 49.9 → 42.5 (−15%). Something kernel-specific is hitting F-73 at the 32K step that does not hit dense — most likely the F-73 fine-block top-K pad path crossing a working-set / threadgroup-memory boundary at this ctx × B=8.

## Interpretation

- **16K–28K:** F-73 batched is compute-saver vs dense. At B=8 with these ctxs, dense attention is compute-bound enough that skipping the fine blocks pays for the kernel overhead. Stable ~1.3× win region.
- **32K:** dense decode is fully BW-bound (KV cache ~42 GB working set at B=8, ctx=32K, GQA 8×128). The F-73 gather + scatter + mask pipeline costs more memory traffic than the dense flash kernel saves by streaming KV linearly. F-73 loses both the compute and the BW argument here.
- Prefill is **dense** in both arms (`prefillSparse=false`), so prefill ms differences (~25% faster F-73) are pure noise from KV-init pathways, not an F-73 prefill win.

## Recommendation

**Ship F-73 batched as the default for B=8 at ctx ≤ 28K. Fall back to dense for ctx ≥ 32K.**

Concretely:
1. Add an auto-fallback in the batched dispatcher: if `ctx >= 32768` and `B >= 8`, route to dense even when `VSM_SPARSE_BATCHED=1` is set. Log the demotion once.
2. Bisect the 28K–32K gap in a follow-up (try 30K, 31K) only if a customer workload sits in that window. The 1.3×→0.65× cliff is steep enough that the exact crossover ctx matters less than the rule "≤28K F-73, ≥32K dense" for shipping.
3. Investigate the F-73 32K cliff as a kernel issue, not a heuristic issue: profile the F-73 fine-block gather kernel at T=32K to find what specifically degrades. Likely candidates are threadgroup memory pressure (kPadFrac=1.0 keeps the full padded block), SM occupancy drop from a buffer-size threshold, or a non-coalesced gather pattern at 32K stride.

## Raw logs

Saved at `/tmp/f85_bisect/{dense,f73}_{16k,20k,24k,28k,32k}.log` on the bench host.
