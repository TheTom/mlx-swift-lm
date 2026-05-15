# F-83 Memory Analysis — 256K Prefill 162 GB / 97 GB Unaccounted

**Date**: 2026-05-14
**Bench**: `f83_perfBench256K_14B1M`, Qwen2.5-14B-Instruct-1M-4bit, M5 Max 64 GB
**Trigger snapshot**:
```
MEM[sparse-after-prefill]    active=162,862MB peak=163,344MB cache=60MB
MEM[sparse-after-clearCache] active=162,862MB peak=163,344MB cache=0MB
```
Decode step 0 jetsam-kills.

## TL;DR verdict

Hypothesis "asyncEval accumulates lazy graph" — **PARTIAL / WRONG mechanism**.

The lazy graph is not what's holding 97 GB. `asyncEval` walks the full graph
synchronously, calls `gpu::eval` per node, and `detach()`es each array from
its primitive before returning (Cmlx/mlx/transforms.cpp lines 169, 298–299).
So between chunks, the graph chain is broken — there is no
"unevaluated K depends on hidden" reference path.

What IS holding the memory: **Metal command-buffer completion handlers**.
`metal/eval.cpp` lines 48–58 captures `data_shared_ptr()` for inputs,
siblings, AND outputs into the completion-handler closure. Those shared_ptr
copies pin the underlying MTL::Buffer alive until the GPU finishes the
command buffer. With `asyncEval` returning before the queue drains and the
encoder bundling up to 500 ops / 100 MB per command buffer on `'s'` arch
(M5 Max, device.cpp 661–675), 128 chunks of prefill can leave many
seconds' worth of GPU work in flight, each retaining its inputs and
outputs.

`active_memory` is the live byte sum tracked in `MetalAllocator::malloc`
(allocator.cpp line 168) and decremented only in `free()` (line 191).
`free()` is only called when `array::Data`'s last `shared_ptr` is dropped.
Completion handlers hold those `shared_ptr`s. So `active_memory` ≠ "live
MLXArray" in the Swift-ARC sense — it's "live MTL::Buffer kept alive by
either an MLXArray OR a pending command-buffer completion handler."
Confirmation that this physically bills: completion handlers retain
allocator buffers, and `MTLResourceHazardTrackingModeUntracked` +
`commandBufferWithUnretainedReferences()` (allocator.cpp line 15,
device.cpp line 587) is exactly why the explicit retain in the completion
handler was added (eval.cpp lines 60–65).

`clearCache()` only releases the buffer pool — it does NOT touch
`active_memory` (allocator.cpp lines 180–183). It cannot, because those
buffers are not free to be touched while the GPU might still read them.

## What activeMemory actually counts

| Question | Answer | Evidence |
|---|---|---|
| Live MLXArrays only? | NO — also any buffer pinned by a Metal command-buffer completion handler | metal/eval.cpp:48-58, allocator.cpp:168, 191 |
| Pool buffers? | NO — those are `cacheMemory` | allocator.cpp:191-202 |
| Virtual or physical? | PHYSICAL bytes of MTL::Buffer allocations on shared unified memory | allocator.cpp:14-15 uses `MTL::ResourceStorageModeShared` |
| Views/slices? | NO — slices share buffer via `copy_shared_buffer` (common/slicing.cpp:35), so a sliced array and its parent share ONE allocator entry, counted ONCE |
| Peak or sum at moment? | SUM at this moment of currently-live MTL::Buffers (allocator.cpp:168) |
| peakMemory? | High-water mark of activeMemory (line 169) — never decremented |

So **the 162 GB is real physical bytes** owed to MTL::Buffer objects, on a
64 GB unified-memory device. Apple Silicon allows MLX to "allocate"
beyond physical RAM because of the wired-memory / memory_limit system
(`block_limit_` defaults to `1.5 × recommendedMaxWorkingSetSize`,
allocator.cpp:52); the data is in shared system memory and gets paged
out via macOS's compressor + swap until a single hot working set
exceeds jetsam's threshold.

## Sane accounting at 256K (corrected)

| Component | Size | Was | Correction source |
|---|---|---|---|
| K cache (48L × 256K × 8 × 128 × 2B) | 25.2 GB | 25.8 | exact = 256000, not 262144 |
| V cache | 25.2 GB | 25.8 | same |
| `perTokenFeatures` (48 × [8, 256K, 16] **fp16** not fp32) | **3.0 GB** | 6.1 | BatchedRetrievalAttentionIndex.swift:113 stores `.asType(.float16)`. F-67 |
| Model weights (4-bit ~14B params + overhead) | ~7 GB | 7 | as given |
| **Static sum** | **~60.4 GB** | 65 | |

So the **unaccounted band is ~102 GB**, slightly larger than Tom's 97 GB
estimate after the fp16 correction.

## Where the 102 GB actually went

Four contributors, in approximate order of magnitude. The numbers below
are upper-bound estimates per chunk × 128 chunks; in practice they
overlap and queue ageing reduces the peak count of simultaneously live
copies.

### (A) Forward-pass activations pinned by completion handlers — largest

Per chunk, the model forward pass produces, **per layer**, transient
allocations that are inputs/outputs of dispatched primitives. With
`hidden = [1, 2048, 5120]` fp16 = 20 MB:

- hidden state itself: 20 MB
- RMSNorm output: 20 MB
- Q proj `[1, 2048, 40 × 128]` fp16: 20 MB
- K proj `[1, 2048, 8 × 128]` fp16: 4 MB
- V proj `[1, 2048, 8 × 128]` fp16: 4 MB
- Q post-RoPE: 20 MB
- K post-RoPE: 4 MB
- SDPA prefill output: 20 MB
- O proj output: 20 MB
- Gate/Up/Down MLP intermediates (FFN hidden ~13824): ~166 MB combined
- Residuals: ~20 MB

Total ≈ 300 MB / layer × 48 layers ≈ **14 GB / chunk** of transient
buffers, each held alive by a completion handler retaining its
`data_shared_ptr` (metal/eval.cpp:48-58). The completion handlers don't
fire until the GPU executes that command buffer.

With `MAX_ACTIVE_TASKS = 10` (transforms.cpp:25) and `max_ops_per_buffer
= 500` / `max_mb_per_buffer = 100` on M5 Max (device.cpp:673-675), one
prefill chunk produces tens of command buffers. The scheduler only
back-pressures when `n_active_tasks > 10` AND `active_memory >
memory_limit` (transforms.cpp:264-278) — and `memory_limit` defaults to
`1.5 × max_recommended_working_set_size` ≈ 64+ GB on a 64 GB Mac.

So the back-pressure trigger is **literally where Tom's number lives.**
At 162 GB active_memory > memory_limit, the next `asyncEval` will be
gated, but anything below ~64 GB sails through. The pipeline grows to
N chunks deep before back-pressure kicks in. At chunk 128 we're past
the back-pressure floor, but the queue still has tens of chunks of
in-flight retained buffers. Estimate: 5–8 chunks × 14 GB = **70–110 GB
of forward-pass intermediates pinned by completion handlers**.

This is the dominant share of the 102 GB.

### (B) `perTokenFeatures` doubling realloc history

BatchedRetrievalAttentionIndex.swift:163–181 grows
`perTokenFeatures` by **doubling** during prefill (line 173: `while c <
newSeqLen { c *= 2 }`). At 256K, cap ascends 1024 → 2048 → ... →
262144 (8 doublings). Each grow does:
```
perTokenFeatures = concatenated([perTokenFeatures!, pad], axis: 1)
eval(perTokenFeatures!)
```

`concatenated` allocates a fresh full-size buffer (slicing.cpp:26
`out.set_data(allocator::malloc(out.nbytes()))`) and copies. The OLD
buffer is held alive briefly during the copy. The `eval` then
materializes and ARC normally drops the old reference — UNLESS a still
in-flight command buffer's completion handler captured the old buffer
(metal/eval.cpp:48-58).

This contributes 100s of MB per doubling, not GB. NOT a major driver.

### (C) SDPA prefill input copies

scaled_dot_product_attention.cpp lines 776–778 (full attention mode,
L > 8) calls `copy_unless(is_matrix_contiguous, q/k/v_pre)`. For
`prefillSparseAttend`, the keys passed are
`combinedK = concatenated([gK, chunkK], axis: 2)` and a slice of
`cachedValues`. `gK` is the gather output (fresh contiguous alloc), but
`chunkK = cachedKeys[..., priorLen..., :]` is a non-contiguous strided
view of the master K cache. `concatenated` then materializes the union
into a fresh contiguous buffer of shape `[1, 8, P+L, 128]` —
P ≈ positions count, L = 2048.

At chunk 128 with priorLen ≈ 254K and fineTopK + static + sliding
≈ 4–8K positions, that's `[1, 8, ~6K, 128]` fp16 = ~12 MB per layer,
~580 MB / chunk for K + V. With multiple chunks in flight via the queue,
total ≈ 2–4 GB. Minor.

### (D) `concatenated([gK, chunkK], axis: 2)` per layer per chunk

Same as (C) — it's the same allocation viewed from the other side. The
fresh `[1, nKVH, P+L, D]` buffer (both K and V) is the SDPA input. It
gets pinned by the SDPA completion handler. ~580 MB/chunk × queue
depth. ~2–4 GB.

### Sum

A: ~70–110 GB (dominant)
B: ~1–2 GB
C+D: ~3–6 GB

**Estimate: 75–120 GB unaccounted, centered ~95 GB.** Matches the
observed 97–102 GB.

## Verdict on Tom's specific hypotheses

| Claim | Verdict | Evidence |
|---|---|---|
| "asyncEval schedules but doesn't flush" | RIGHT in spirit, mechanism slightly off. asyncEval DOES walk the graph and detach; it dispatches all kernels for the current call. What it DOESN'T do is wait for GPU completion. | transforms.cpp:316-328 (no `.wait()`) vs eval at line 344 |
| "Lazy graph keeps hidden alive because unevaluated K depends on it" | WRONG. `eval_impl` detaches the array immediately after `gpu::eval` (transforms.cpp:298-299). The graph chain is broken by the time asyncEval returns. | array.cpp:116-129, transforms.cpp:298-299 |
| "Metal command queue piles up; ARC can't reclaim" | RIGHT — but the gatekeeper is the COMPLETION HANDLER retaining `data_shared_ptr`, not the lazy graph | metal/eval.cpp:48-58 |
| "hidden = 20 MB × 48 layers × 128 chunks = 123 GB" | Overcounts by including ALL chunks; in practice it's queue-depth × per-chunk forward, ~70–110 GB, but the ORDER OF MAGNITUDE intuition is correct | scheduler.h:67-118, transforms.cpp:25 |
| "activeMemory = virtual / accounting-only" | WRONG. Real physical MTL::Buffer bytes, shared storage mode, billed against system RAM. | allocator.cpp:14, 168, 191 |
| "take(positions) gathers creating implicit copies" | RIGHT direction, small magnitude. Each `take` allocates a fresh buffer of `[B, nKVH, len(positions), D]`, ~12 MB / layer at chunk 128. ~600 MB / chunk transient. | primitives.h:1155-1177 Gather is a UnaryPrimitive |
| "perTokenFeatures over-allocated at 12 GB via doubling" | WRONG (the OLD buffer doesn't survive the realloc — only the new one, capped at 256K = 3 GB total). Doubling history is briefly transient but quickly ARC'd. | BatchedRetrievalAttentionIndex.swift:180-181 (reassignment), array.cpp:295 (use_count) |
| "allocator holds the doubling history" | WRONG. The old buffers go through `free()` → pool (or release). They are NOT retained anywhere unless still in flight. | allocator.cpp:185-203 |

## Decode-step death — separate cause

The first decode step (L=1) after clearCache + cacheLimit=0 dies BEFORE
`model()` returns visible logits. The killer:

**`StandardKVCache.updateUnbounded` triggers a full-cache realloc on
step-boundary** (KVCache.swift:460-496):

```swift
let reset =
    if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
        true  // <-- fires on first decode step when prefill exactly fills the cap
    } else {
        self.keys == nil
    }
if reset {
    ...
    self.keys = concatenated([currentKeys, newK], axis: 2)
    self.values = concatenated([currentValues, newV], axis: 2)
    eval(self.keys!, self.values!)
}
```

At 256K, prefill leaves the cache cap at exactly 256K (256 × 1000 = 256000
with step=256). First decode step: `previous + 1 = 256001 > 256000` →
realloc fires.

`concatenated([currentKeys, newK], axis: 2)` allocates a fresh buffer of
`[1, 8, 256256, 128]` fp16 = **25.2 GB** for K alone, plus 25.2 GB for V.

If `active_memory` is already 162 GB (which it is — `clearCache` doesn't
release in-flight buffers), this 50 GB allocation request blows past
jetsam threshold. Process dies.

This is NOT triggered by the cache contents (which already exist) —
it's triggered by the cache being SIZED EXACTLY at prefill length and
the first decode step needing one more slot. The doubling-style realloc
copy is the killer, NOT the increment.

## Concrete experiments to differentiate hypotheses

1. **Pin command-buffer completion vs lazy-graph theory.** Run prefill
   with `MLX_BFS_MAX_WIDTH=1` (utils.h:149). This forces the graph
   walker to depth-first / single-width-at-a-time, which doesn't
   change command-buffer retention but DOES change graph-walk
   pressure. If the 97 GB shrinks → graph-walk side. If it doesn't →
   completion-handler retention.

2. **Lower memory_limit to force back-pressure.** Set
   `MLX.GPU.set(memoryLimit: 30_000_000_000)` before prefill. This
   activates the scheduler's `wait_for_one` loop (transforms.cpp:264-278)
   and bounds the in-flight queue. Predicted result: active_memory caps
   near 30 GB, prefill slows down N% (queue is bottleneck), no jetsam.
   This is the SURGICAL fix that doesn't require changing the chunked
   pattern.

3. **Per-chunk synchronous eval — what the user just tried.** If sync
   eval per chunk eliminates the unaccounted band, that's confirmation
   the band is the completion-handler-pinned forward intermediates. If
   it doesn't, look at (1).

4. **Per-chunk `clearCache()` like Python mlx-lm does.** Force pool
   release every chunk. With sync eval this is the canonical pattern.
   With async, this has no effect on active_memory but does free
   non-active buffers — won't help our case but is a useful control.

5. **Lower `MAX_ACTIVE_TASKS`.** Hardcoded to 10 in transforms.cpp:25
   (would need a custom build). Drop to 2. Predicted: queue depth caps
   sooner, active_memory peak halves.

6. **Pre-allocate the KV cache via `reserve`** before prefill (the
   StandardKVCache has `initialAllocSize` support for this — KVCache.swift:421-427).
   Allocate enough room for `prefillLen + nDecode + maxSlack`. This
   eliminates the decode-step-0 realloc death entirely. `reserve(256000
   + 64)` would make decode step 0 a single in-place write at offset
   256000.

## Canonical patterns we're violating

`mlx-lm` Python `generate.py` chunked prefill loop (lines 430–451):

```python
while total_prompt_tokens - prompt_processed_tokens > 1:
    n_to_process = min(prefill_step_size, remaining)
    _model_call(input_tokens=..., input_embeddings=...)
    quantize_cache_fn(prompt_cache)
    mx.eval([c.state for c in prompt_cache])   # SYNC
    prompt_processed_tokens += n_to_process
    prompt = prompt[n_to_process:]
    ...
    mx.clear_cache()                           # POOL RELEASE
```

Our `LLMModel.swift` and `Tests/.../f83_perfBench256K_14B1M` pattern:

```swift
while y.tokens.size > 1 {
    let chunkSize = min(prefillStepSize, y.tokens.size - 1)
    _ = self(input, cache: cache, state: nil)
    var cacheArrays: [MLXArray] = []
    for c in cache { cacheArrays.append(contentsOf: c.innerState()) }
    asyncEval(cacheArrays)                     // ASYNC
    y = y[chunkSize...]
}
eval(cache)                                    // single sync at end
MLX.Memory.clearCache()                        // single pool release at end
```

The Swift pattern was introduced for prefill speed (LLMModel.swift:34–40
docs the rationale). It works fine when (chunks × per-chunk forward
mem) < memory_limit. It blows up when the inequality reverses, which
is exactly what happens at 128 × 14 GB ≈ 1.8 TB notional, capped by
back-pressure at memory_limit ≈ 64–96 GB.

Python mlx-lm gets pipelining "for free" because its bindings defer eval
until a value is read, but the per-chunk `mx.eval([c.state])` then
synchronously drains. They explicitly accept ~chunk latency to bound
memory.

## GitHub-issue corroboration

- [ml-explore/mlx#3186](https://github.com/ml-explore/mlx/issues/3186)
  Kernel panic IOGPUMemory.cpp:550 on M4 Max during 173K-token prefill,
  ~26 GB model on a 36 GB box. Apple FB22091885. Direct evidence
  that long-context prefill blows physical memory accounting state on
  Apple Silicon, not just "OOM."
- [ml-explore/mlx#3216](https://github.com/ml-explore/mlx/issues/3216)
  SIGSEGV in QuantizedMatmul::eval_gpu during long generation on M2
  Ultra 128 GB. Race between Metal command-buffer coalescing and MLX's
  next dispatch. Related symptom class — Metal book-keeping fails
  under heavy queue depth.
- [Blaizzy/mlx-vlm#945](https://github.com/Blaizzy/mlx-vlm/issues/945)
  Per-chunk `mx.eval() + mx.clear_cache()` "stalls the GPU pipeline,
  waits for completion, clears the cache, and restarts, adding latency
  proportional to the number of chunks." That's the explicit
  performance counter-argument to the safe pattern. The community has
  not landed on an "async + safe" pattern.
- [ml-explore/mlx#742](https://github.com/ml-explore/mlx/issues/742)
  "GPU Memory Management?" — setting MLXArray refs to None doesn't
  reclaim memory; user asks for `torch.mps.empty_cache()` equivalent.
  This is what motivated `clear_cache`. Confirms pool retention is a
  known surprise, separately from completion-handler pinning.
- No GitHub issue specifically about "asyncEval + chunked prefill →
  active_memory blowup." That bug is in `mlx-swift-lm`'s wrapper
  pattern, not in MLX core. It would be a fair upstream issue.

## Reference: source lines underwriting every claim

| Claim | File | Line |
|---|---|---|
| asyncEval doesn't wait | Cmlx/mlx/transforms.cpp | 316-328 |
| eval does wait | Cmlx/mlx/transforms.cpp | 330-345 |
| eval_impl walks graph + detaches | Cmlx/mlx/transforms.cpp | 169, 298-299 |
| activeMemory = in-use bytes | Cmlx/mlx/backend/metal/allocator.cpp | 168 |
| free() ← shared_ptr drop | Cmlx/mlx/backend/metal/allocator.cpp | 185-202 |
| clearCache only frees pool | Cmlx/mlx/backend/metal/allocator.cpp | 180-183 |
| Completion handler captures shared_ptr | Cmlx/mlx/backend/metal/eval.cpp | 48-58 |
| Slices share buffer | Cmlx/mlx/backend/common/slicing.cpp | 20-36 |
| Concat allocates fresh buffer | Cmlx/mlx/backend/metal/slicing.cpp | 14-43 |
| Gather is Unary (always allocs) | Cmlx/mlx/primitives.h | 1155-1177 |
| SDPA copies non-contig K/V | Cmlx/mlx/backend/metal/scaled_dot_product_attention.cpp | 776-778 |
| MAX_ACTIVE_TASKS = 10 | Cmlx/mlx/transforms.cpp | 25 |
| max_ops/mb_per_buffer M5 Max | Cmlx/mlx/backend/metal/device.cpp | 661-680 |
| bfs_max_width = 20 | Cmlx/mlx/utils.h | 149-152 |
| Back-pressure trigger | Cmlx/mlx/transforms.cpp | 264-278 |
| memory_limit default | Cmlx/mlx/backend/metal/allocator.cpp | 52 |
| Snapshot doc — "activeMemory + cacheMemory = total allocated" | Source/MLX/Memory.swift | 7-30 |
| Swift asyncEval = mlx_async_eval | Source/MLX/Transforms+Eval.swift | 40-46 |
| LLMModel async chunked prefill | Libraries/MLXLLM/LLMModel.swift | 41-58 |
| StandardKVCache realloc on overflow | Libraries/MLXLMCommon/KVCache.swift | 460-496 |
| RA cache prefillSparseAttend (slice + gather + concat per chunk per layer) | Libraries/MLXLMCommon/RetrievalAttentionKVCache.swift | 1011-1018 |
| perTokenFeatures fp16 storage | Libraries/MLXLMCommon/BatchedRetrievalAttentionIndex.swift | 113 |
| perTokenFeatures doubling realloc | Libraries/MLXLMCommon/BatchedRetrievalAttentionIndex.swift | 162-181 |
| mlx-lm Python chunked prefill (sync) | mlx-lm/mlx_lm/generate.py | 430-451 |

## Recommended fix order

1. **Pre-allocate the KV cache** (`reserve` with `prefillLen + decodeBudget`
   in StandardKVCache) — kills the decode-step-0 50 GB realloc copy outright.
   No latency cost. This alone may unblock 256K decode.

2. **Drop `memoryLimit` to a safe value** (e.g. 40 GB on 64 GB Mac) before
   prefill. Activates the in-built back-pressure. Active_memory will
   plateau ~40 GB regardless of chunks. Predicted perf cost: small —
   pipeline is bounded but not serialized.

3. **Switch the long-context prefill path to sync `eval` per chunk +
   periodic `clearCache`** — Python parity. Slower (loses pipelining) but
   correct under all conditions. Add as fallback via env var
   (`F83_SYNC_EVAL=1` already exists in the test).

4. **(Optional) Lower `MAX_ACTIVE_TASKS`** via env if MLX exposes it; if
   not, a 3-line patch.

Of these, (1) is load-bearing and decoupled from prefill semantics —
do it first. (2) is the surgical fix for the prefill blowup. (3) is the
"give up the pipelining win" fallback.
