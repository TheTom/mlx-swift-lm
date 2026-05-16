# F-83 Fused Block W3 — QKV-projection + RoPE + KV-cache-write fused kernel

**Status**: planned
**Branch**: `feature/retrieval-attention` (local-only — do NOT push)
**Target model**: Qwen2.5-14B-Instruct-1M-4bit on M5 Max 128GB
**Goal**: One Metal dispatch per decoder layer that does: 4-bit QGEMV for Q+K+V, reshape/transpose to head-major, RoPE on Q and K, write K and V into the cache buffer. Replaces 3 quantized matmuls + 3 reshape/transpose + 2 RoPE + 2 cache-writes ≈ 10 dispatches/layer × 48 = ~480 fewer per decode step.

---

## Why

Per agent ae396 research:
- gorroai/flash-moe shaders: "Fuses Q/K deinterleave + per-head RMSNorm + partial RoPE + KV cache append".
- trymirai/uzu has `attention_update_kv_cache.metal` + `qk_unpack.metal` covering this pattern.
- openai/gpt-oss `gptoss_f32_mf4w_moe_dense_matmul_swiglu` is the dequant+matmul+activation template for the Q/K/V leg.

Decode path is dispatch-bound at L=1 because each op is tiny (~96 µs/dispatch on M-series) but launches ~10 ops/layer × 48 layers = 480 dispatches per token. At 96 µs/dispatch this is ~46 ms of pure dispatch overhead — well above our 23 ms/step total. MLX's lazy eval coalesces some of this, but not the per-encoder-stop boundaries.

Current Qwen2.swift Attention.callAsFunction (Libraries/MLXLMCommon/Models/Qwen2.swift:152-157):

```swift
queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

queries = applyRotaryPosition(rope, to: queries, cache: cache)
keys = applyRotaryPosition(rope, to: keys, cache: cache)
// cache.update happens inside attentionWithCacheUpdate
```

W3 collapses everything above into one kernel that emits the new K/V directly into the cache buffer at the correct offset.

## Acceptance criteria

- Custom Metal kernel `fused_qkv_rope_kvwrite.metal` for: 4-bit qgemv (sized hidden=5120, heads=40, kvHeads=8, headDim=128), RoPE (base=1e6 default Qwen2, traditional=false), in-place K/V writes.
- Swift wrapper `MLXFast.fusedQKVRopeKVWrite(...)` taking the cache buffer + offset directly.
- Env-gated behind `F83_FUSED_QKV_ROPE_KV=1`.
- Bench: ≥3% decode improvement at 16K / 32K / 128K vs baseline.
- Quality: top-1 token match across 256 decode steps at 32K. (RoPE + matmul are deterministic; sole risk is RoPE freq table mismatch.)

## Implementation plan

1. Mirror uzu's kernel split: separate qkv_proj kernel + small launcher for RoPE + cache write. Or attempt fully-fused per gpt-oss.
2. Threading: K/V write into cache must respect cache layout (TheTom MLX `StandardKVCache` is `[B, kvHeads, T_alloc, headDim]`).
3. RoPE freqs as constant buffer (precomputed at init), offset = `cache.offset`.
4. Bias add: Qwen2 q/k/v_proj have biases — fold into the qgemv epilogue.
5. Build, test against tiny model first (parity vs split path), then bench Qwen2.5-14B.

## Risk

- KV cache layout mismatch is the most common failure mode. We need to confirm `cache.keys` returns the row-contiguous buffer (the F-83 north-star hit established this — see `project_f83_north_star_hit.md`).
- RoPE Qwen2 uses `theta=1e6` (1M-context variant). Hardcoding the wrong base produces silent gibberish.
- Bias add needs to happen BEFORE RoPE per Qwen2 spec (verify against Python `mlx-lm/models/qwen2.py`).

## Outcome

(TBD)
