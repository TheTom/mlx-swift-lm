# F-83: SGLang Chunked Prefill Recipe (and what crosses engines)

Date: 2026-05-14
Scope: extract the actual mechanism SGLang uses for long-context prefill, compare against vLLM / llama.cpp / mlx-lm, distill the cross-engine invariant for our 256K MLX-Swift work.

Sources cloned at `/tmp/sglang` (sgl-project/sglang @ main, shallow). All citations are `python/sglang/srt/...` unless noted.

---

## TL;DR — the recipe

1. **Pre-allocate one fixed KV pool sized to `mem_fraction_static * VRAM - weights - activations`.** Never grow during prefill. Token slots come from a free-list. No per-chunk allocations.
2. **Chunk size is fixed by hardware tier, not by prompt length.** 2K / 4K / 8K / 16K. Same chunk shape every iteration → cudagraphs / kernel caches reuse.
3. **A chunk is a single EXTEND forward** where `extend_seq_lens = chunk_size` (Q) and `extend_prefix_lens = tokens_already_in_KV` (K/V from paged pool). FlashInfer ragged-Q + paged-KV wrapper.
4. **No GPU sync between chunks.** The only sync is `copy_done.synchronize()` after the device→pinned-host copy of `next_token_ids` at the END of the prefill chain — i.e. once the last chunk produces a logit. Chunk N+1 can be launched on the GPU while chunk N's CPU bookkeeping runs.
5. **Activations are bounded by `chunk_size`, not `prompt_len`.** That is the load-bearing claim. `mem_fraction_static` is literally derived as `reserved = chunk_size * 1.5 + cuda_graph_max_bs * 2` GB.

If you violate (1), (2), or (5), you are not doing chunked prefill — you are doing chunked scheduling on top of full prefill, which is what we are seeing on MLX-Swift.

---

## 1. KV cache allocation — fixed pool, computed once

`python/sglang/srt/mem_cache/memory_pool.py:897-918`

```python
def _create_buffers(self):
    ...
    self.k_buffer = [
        torch.zeros(
            (self.size + self.page_size, self.head_num, self.head_dim),
            dtype=self.store_dtype, device=self.device,
        ) for _ in range(self.layer_num)
    ]
    self.v_buffer = [ ... same shape ... ]
```

`self.size = max_total_num_tokens`, computed once at startup as:

`python/sglang/srt/model_executor/pool_configurator.py:185-187`

```python
max_total_num_tokens = available_bytes // self._cell_size
max_total_num_tokens = max_total_num_tokens // page_size * page_size
```

Where `available_bytes = mem_fraction_static * total_gpu_mem - model_weights - activations_reserve`. Allocator is a free-page bitmap (`BaseTokenToKVPoolAllocator`, `mem_cache/allocator.py:35-66`), `available_size() = len(free_pages) * page_size`.

**This is the same model as vLLM.** Pre-paged, page_size 1/16/32/64. Not grown. RadixCache is a *view* into this pool; it doesn't allocate.

## 2. Chunk size — hardware tier, not prompt

`python/sglang/srt/server_args.py:1416-1479` (full table):

| GPU mem (GB) | chunk size | cuda_graph_max_bs |
|--|--|--|
| <20 (T4, 4080) | **2048** | 8 |
| <35 (A10, 4090, 5090) | **2048** | 24 / 80 |
| <60 (A100-40, L40) | **4096** | 32 / 160 |
| <90 (H100, A100-80) | **8192** | 256 / 512 |
| <160 (H20, H200) | **8192** | 256 / 512 |
| >160 (B200, MI300) | **16384** | 512 |
| fallback | 4096 | 160 |

Quote (`server_args.py:1409-1411`):
> "The activation memory is proportional to the chunked_prefill_size. The cuda graph memory is proportional to the cuda_graph_max_bs. We use `reserved_mem = chunked_prefill_size * 1.5 + cuda_graph_max_bs * 2` to estimate the size of activations and cuda graph buffers in GB."

That heuristic is the entire long-context budget contract. On a 64 GB M5 Max, a 2K chunk reserves ~3 GB of activations; an 8K chunk reserves ~12 GB. **The chunk size IS the activation peak.**

## 3. The chunked prefill loop — no per-chunk sync

The scheduler is a **continuous event loop**, not a per-prompt loop.

`python/sglang/srt/managers/scheduler.py:1537-1561` (normal loop):

```python
def event_loop_normal(self):
    while True:
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        batch = self.get_next_batch_to_run()
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        self.last_batch = batch
```

Each iteration:
- `get_new_batch_prefill()` (`scheduler.py:2611-2700`) decides what fits in `chunked_prefill_size` tokens this step.
- A long prompt becomes a sticky `self.chunked_req` (`scheduler.py:2796-2805`); `add_chunked_req` (`schedule_policy.py:668-701`) trims `extend_input_len = min(extend_input_len, rem_chunk_tokens)` and returns the req back to the scheduler until exhausted.
- The same loop iteration launches a forward and merges with running decode batches.

**Sync model — this is the key one.** `scheduler.py:3044-3050`:

```python
batch_result.copy_done = self.device_module.Event()
if batch_result.delay_sample_func is None:
    self.future_map.store_to_map(future_indices, batch_result)
    batch_result.copy_to_cpu(...)
```

The sync only happens when the consumer needs the CPU values:

`scheduler_output_processor_mixin.py:181-211`:
```python
def process_batch_result_prefill(self, batch, result):
    if self.is_generation:
        if result.copy_done is not None:
            result.copy_done.synchronize()
        ...
        next_token_ids = next_token_ids.tolist()
```

For middle chunks of a long prompt, there are NO `next_token_ids` to copy back — the chunk is part of EXTEND, sampling only happens at the final chunk. So the GPU runs back-to-back forward calls with no host stall. The overlap loop (`event_loop_overlap`, `scheduler.py:1564-1616`) takes this further by launching batch N+1 while popping/processing batch N on the CPU.

**Contrast: mlx-lm style "per-chunk `mx.eval`" forces a host stall every chunk.** SGLang would never do this. The Metal equivalent of `copy_done.synchronize()` is `commandBuffer.waitUntilCompleted()` after a `blit` to a CPU-shared `MTLBuffer` — and you only call it once, when you actually need the token id.

## 4. Memory management during prefill — no clears

There is no "free the chunk's activations" step anywhere. Activations are transient tensors allocated by torch's caching allocator, freed when Python refs drop at end of the forward call. The KV pool itself is touched only by:
- `alloc(num_tokens)` returns indices from `free_pages` (no realloc),
- `free(indices)` returns indices back,
- writes via the attention backend (FlashInfer/FA3 `store_cache` kernels) into the pre-allocated buffer at those indices.

No buffer clearing between chunks. The `completion-handler pins buffers` problem on CUDA doesn't really exist — CUDA stream-ordered allocator + caching allocator handles it. The MLX analog is the buffer recycle pool we already wrote about in F-80 (and patched). The lesson for us: **make sure chunk-N's transient tensors are dropped (no Python refs) before chunk-N+1 starts**, otherwise MLX-Swift's lazy graph holds onto the intermediate computation graph and the activation memory grows linearly with chunk count. That is almost certainly what's eating you at 162 GB MLX-active.

## 5. RadixAttention's role here — none, for single-prompt prefill

RadixCache (`mem_cache/radix_cache.py:269+`) deduplicates prefix KV **across requests** and across turns. For a single 256K prompt with no prefix match, RadixCache contributes zero. The chunked prefill loop walks the prompt in chunk_size strides regardless.

What RadixAttention does change is what `extend_prefix_lens` is set to when a cache hit happens (`cache_unfinished_req`, `radix_cache.py:518-549`). The new chunk's Q is computed; the K/V for the matched prefix is read from the paged pool, not recomputed. This is what makes "second turn of a chat" cheap. It does NOT bound transient prefill memory — the chunk_size cap already does.

## 6. Kernels — FlashInfer for the >256K story

`python/sglang/srt/layers/attention/flashinfer_backend.py:51-52, 255-273`:

```python
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper, ...
)
self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(...)
self.prefill_wrappers_paged.append(BatchPrefillWithPagedKVCacheWrapper(...))
```

Two wrappers per layer:
- **Ragged** for new Q × new K/V within the chunk (no prior cache),
- **Paged** for new Q × all prior K/V from the page pool.

The paged wrapper is the long-context workhorse: it does a tiled FlashAttention-2/3 over paged KV without materializing the full K/V matrix in HBM. This is the same algorithm vLLM's paged attention uses. The FA3 backend (`flashattention_backend.py:31-32, 170-187`) uses `flash_attn_varlen_func` + `flash_attn_with_kvcache` from Tri Dao's flash-attn — same shape contract.

**For Apple Silicon:** the equivalent of `flash_attn_with_kvcache` for paged KV at 256K does not exist in MLX yet. Our F-83 sparse-prefill is essentially building this from sparse retrieval. The contract should match: Q tile in SRAM, K/V paged in HBM, output accumulated in SRAM, **one tile at a time, no full-rank materialization anywhere**.

## 7. Why SGLang is fastest in practice

From `lmsys.org/blog/2024-12-04-sglang-v0-4`:
> "The scheduler runs one batch ahead and prepares all the metadata required for the next batch... there is no single idle time on the GPU."

The advantages stack:
1. Python scheduler hides behind the GPU because of overlap loop (the "zero-overhead scheduler"),
2. FlashInfer paged kernels (best published prefill kernels for paged KV),
3. RadixCache prefix reuse across requests (irrelevant for cold single-prompt 256K, big for chat),
4. cudagraph reuse from fixed chunk shape,
5. Pre-allocated KV pool means no allocator stalls.

LMSYS post (2024-07): SGLang on Llama 3 8B/70B: **3.1× higher throughput vs vLLM, 2.7× vs TRT-LLM** (offline). The blog notes the gains are "system engineering" not a single kernel.

## 8. Cross-engine invariant — what ALL fast engines do

Across SGLang, vLLM, llama.cpp, mlx-lm, the constants are:

**MUST-HAVE (all fast engines)**
1. **KV cache is a pre-sized contiguous pool indexed by page id.** Never grown mid-prefill.
2. **Prefill is chunked with a fixed chunk size** chosen so chunk activations fit in a small fraction of VRAM/UMA.
3. **A chunk = one attention forward** with `Q ∈ R^{chunk × H × D}` and `K/V ∈ R^{prefix × H × D}` read paged. No prefix materialization.
4. **Sync the host only when the host needs a value** (final logits, sampling). Not per-chunk, not per-layer.
5. **Drop the chunk's transient activations before launching the next chunk** so peak memory = chunk activations + KV pool, not Σchunk activations.

**ENGINE-SPECIFIC (only some)**
- RadixCache prefix dedup (SGLang only, vLLM has APC but flat).
- cudagraph reuse of fixed chunk shape (CUDA/ROCm only).
- Zero-overhead python scheduler / one-batch-ahead overlap (SGLang).
- FlashAttention-3 kernels (CUDA H100+ only).
- Sparse attention during prefill (your F-83, not in any production engine yet).

## 9. What this means for MLX-Swift F-83 at 256K

Your 162 GB MLX-active + jetsam death at decode step 0 is almost certainly NOT a KV-pool problem (KV at 256K × 16 layers × 8 KV heads × 128 dim × 2 (K+V) × 0.5 byte (4-bit) ≈ 4 GB, even fp16 is 16 GB). It's an **activation accumulation** problem.

Hypotheses, ranked:
1. **MLX lazy graph holds chunk N's intermediate tensors when chunk N+1 starts.** Each chunk's hidden states, attention scores, MLP intermediates are kept alive by the dependency chain leading to the final eval. Per-chunk `mx.eval()` is wrong (forces stall) but per-chunk **`mx.async_eval()` + `mx.eval()` on the KV writes only** is what you want — equivalent to SGLang's "GPU keeps running, host doesn't block, but chunk N's autograd/lazy graph is collapsed before chunk N+1 builds its own".
2. **You're materializing full prefix K/V on each chunk** instead of reading paged. If sparse-prefill is doing top-k retrieval per chunk but materializing the retrieved K/V into a dense tensor in the chunk forward, that's `chunk × top_k × D` per layer, transient — survivable if you free between chunks, fatal if you don't.
3. **The "async eval per chunk" criticism is right.** The right pattern is: launch all chunk compute → eval only the KV-pool writes (small) → drop activations → start next chunk. **Not** "eval the whole chunk output (hidden states) every chunk".

Concrete next step: instrument MLX-active memory at boundary between chunks. If memory grows linearly, you have a transient-retention bug; if memory is flat, look at KV write efficiency.

---

## Citations

- SGLang chunk size table: `/tmp/sglang/python/sglang/srt/server_args.py:1416-1479`
- KV pool init: `/tmp/sglang/python/sglang/srt/mem_cache/memory_pool.py:897-918`
- Pool sizing: `/tmp/sglang/python/sglang/srt/model_executor/pool_configurator.py:185-193`
- Scheduler event loop: `/tmp/sglang/python/sglang/srt/managers/scheduler.py:1537-1616`
- Prefill batch builder: `/tmp/sglang/python/sglang/srt/managers/scheduler.py:2611-2820`
- Chunked req trimming: `/tmp/sglang/python/sglang/srt/managers/schedule_policy.py:668-701`
- Sync model: `/tmp/sglang/python/sglang/srt/managers/scheduler_output_processor_mixin.py:181-211` and `scheduler.py:3044-3050`
- FlashInfer wrappers: `/tmp/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py:51-273`
- FA3 backend: `/tmp/sglang/python/sglang/srt/layers/attention/flashattention_backend.py:31-187`
- RadixCache prefix: `/tmp/sglang/python/sglang/srt/mem_cache/radix_cache.py:269-549`
- Blog claims: https://lmsys.org/blog/2024-07-25-sglang-llama3/, https://lmsys.org/blog/2024-12-04-sglang-v0-4/
- SGLang paper: https://arxiv.org/abs/2312.07104
- Server args docs (chunked-prefill-size, mem-fraction-static): https://docs.sglang.io/advanced_features/server_arguments.html
