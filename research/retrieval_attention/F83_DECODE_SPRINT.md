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
