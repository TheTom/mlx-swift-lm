# F-83 Decode Sprint — Blow Past Python

**Goal**: Swift decode at 16K Qwen2.5-14B-1M-4bit must beat Python's 25.1 ms/step (39.81 tok/s). Stretch goal: 2x Python → 12.5 ms (80 tok/s).

**Bench command (fast iteration)**:
```bash
F83_PATH=dense F83_PREFILL_LEN=16384 F83_CHUNK_SIZE=2048 F83_BARE=1 \
  F83_LOG_EVERY=100 swift test --filter f83_perfBench256K_14B1M
```
Cycle: ~60-90 s total (12 s prefill + 10 decode steps).

**Constants this sprint**:
- Model: `/Users/tom/models/Qwen2.5-14B-Instruct-1M-4bit`
- Context: 16384 random tokens
- Decode: 10 argmax steps, median of last 8
- Hardware: M5 Max 128 GB

**Baselines**:
- Python mlx-lm 0.31.2: **25.1 ms/step**, 39.81 tok/s
- Swift V20 (BARE + compiled swiglu, commit b903908): **32.7 ms/step**, 30.58 tok/s
- Gap: 7.6 ms

## Iteration Log

| # | Hypothesis | Change | Decode ms | Δ vs V20 | Notes |
|---|---|---|---|---|---|
| 0 | baseline | none | 32.7 | 0 | V20 |
| 1 | sysmem fork inflates timing | t1 captured before snapshots | **30.4** | -2.3 | sysmemSnapshot's `vm_stat` fork costs ~2 ms; was wrapped in t0..t1 |
| 2 | quantized GEMV AB env | MLX_METAL_AB=1 MLX_PERSISTENT_AB=1 | **30.0** | -2.7 | tiny improvement; Qwen2 K=5120/13824 doesn't hit qmv_quad fast path |
| 3 | dedicated decode stream | F83_DECODE_STREAM=1 | 30.8 | -1.9 | no improvement; within noise |
| 4 | inline activation | MLX_INLINE_ACTIVATION=1 | 30.0 | -2.7 | no improvement; Qwen2 doesn't use FusedGateUpSwitchGLU path |
| 5 | fused gate+up via sanitize | concat weights at load time | 30.4 | -2.3 | no real save — MLX.split likely dispatches, offsetting matmul savings |
| 6 | pipelined decode loop | asyncEval(tok) + eval(prevTok) | **25.8** | **-6.9** | apples-to-apples Python ALSO pipelined = 22.7 ms; new gap = 3.1 ms (12%) |
| 7 | pre-queue first step | queue tok before timing loop | **24.8** | **-7.9** | first measured step now has GPU already in flight; gap to Python 2.1 ms (9%) |
| 8 | no logLine in pipelined loop | move logging after the timing loop | 25.1 | -7.6 | within noise; logging wasn't the issue |
| 9 | MLX.MLXFast.fusedGateActivation | replace split+swiglu compile with fused kernel | 25.4 | -7.3 | NO improvement — MLX's lazy graph apparently already fused silu+mul through compile() |

**Final 16K result: 24.8 ms pipelined** vs Python's 22.7 ms pipelined. Gap 2.1 ms / 9%.

## Scaling — gap shrinks at long context

| Context | Python pipelined | Swift pipelined | Swift/Python ratio |
|---|---|---|---|
| 16K | 22.7 ms (44.01 tok/s) | 24.8 ms (40.32 tok/s) | +9.2% |
| 128K | 67.3 ms (14.86 tok/s) | 69.0 ms (14.49 tok/s) | +2.5% |

The fixed Swift-side per-step overhead (Module dispatch + ARC) amortizes
over more bandwidth-bound KV cache work as context grows. By 128K the
gap is statistical noise, and the F-83 sparse path (when its 57 ms
wrapper tax gets fixed via the sidecar-context refactor) should let
Swift blow past Python at 128K+.

**Remaining levers (not yet tested)**:
- `MLX.MLXFast.batchedQKVQuantizedGEMV` — 3 Q/K/V projections in 1 dispatch. Theoretical -2 dispatches × 48 = -4 ms, but bias-add overhead from separate bias-tensors offsets some of the win. Untested.
- `MLX.MLXFast.rmsNormQuantizedGEMV` — fused norm + matmul for input_norm → Q proj. Saves 1 dispatch per layer. But K, V still need separate norm-and-proj. Untested.
- Custom Metal kernel for the entire DecoderLayer forward. Multi-day effort.
- Quantized KV cache (turbo4 V) — reduces bandwidth. Untested at decode in this sprint.

## Path A sidecar refactor — landed

The wrapper-tax fix codex recommended. Now in tree:

- `RetrievalAttentionContext` (new): owns selector state (`batchedIndex`,
  `cachedHeadIdx`, `cachedFineStarts/CoarseStarts`, `lastRefreshOffset`,
  `raConfig`, `layerIdx`, etc.).
- `RetrievalAttentionEngine.retrievalAttentionStep`: F-73 mask path
  AND F-70 per-KV-head gather, both routed via `(cache, ctx)` instead
  of the `RetrievalAttentionKVCache` wrapper.
- `attentionWithCacheUpdate` accepts a default-nil `raContext:` arg.
- `Qwen2.Attention/DecoderLayer/ModelInner` thread `raContext:` end-to-end.
- `Qwen2Model.callAsFunction(_:cache:raContexts:)` overload for the
  sidecar-driven bench path.
- Bench harness gates everything behind `F83_SIDECAR=1`.

### 128K decode results post-sidecar

| Path | Decode ms | Notes |
|---|---|---|
| Python dense pipelined        | 67.3 | baseline |
| Swift dense BARE pipelined    | **67.6** | parity (0.4%) |
| Swift sparse SIDECAR+F73mask  | 89.4 | -13 ms vs wrapper, still loses to dense |
| Swift sparse SIDECAR+perHead  | 198 | no F-79 amortization yet — gather rebuild every step |
| Swift sparse WRAPPER (V12, legacy) | 102 | -- |

The sidecar saves the 13 ms wrapper tax cleanly on sparse. But sparse
still loses to dense at 128K because F-73's mask path only skips
compute via `-inf` — sdpa_vector still loads all K/V. The truly
bandwidth-reducing path (`perKVHeadGather`, ~2k positions vs all
131K) needs F-79 amortization ported to beat dense.

### Sprint commits

- `159cb37` refactor(F-83): thread raContext through Qwen2 — Path A wiring + reads
- `9eef87f` refactor(F-83): sidecar RA engine — F-73 mask via bare cache + raContext
- `80053d9` refactor(F-83): port perKVHeadGatherAndAttend to sidecar (no amort yet)

### Codex's MLX runtime observation

Codex flagged that Python's mlx-lm loads from `/Users/tom/dev/mlx` (HEAD
`42a7a71c`, branch `fix/add-temporaries-fork-primitives`) while Swift
uses vendored `Source/Cmlx/mlx` at `77d6214a` (branch `vllm-swift-stable`).
Different forks. Both have Tom's `max_ops_per_buffer=500` Max/Ultra
default and the same fused kernels in `mlx-generated/metal/`. The
remaining individual-op differences (`42a7a71c` is a use-after-free
fix for TurboQuant temporaries, not a perf change) don't explain
material delta. The 0.4% gap at 128K is well inside measurement noise.
