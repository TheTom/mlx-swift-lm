# F-83 Speedup Research — Apple Silicon MLX Sparse-Prefill Attention

**Scope:** survey of block-sparse / retrieval-attention kernels and MLX-Swift dispatch-overhead techniques that could push the current V1.1 1.53× prefill speedup higher on Qwen2.5-14B-Instruct-1M-4bit at 128K.

**Diagnosis baseline (from PRD):**
- Per-chunk sparse-attention overhead 2.5 s at chunk 128 (priorLen 127K), against 0.78 s FFN floor and ~5 ms theoretical SDPA compute. **~500× kernel-launch / glue overhead.**
- 12 ops × 48 layers × 96 sparse chunks ≈ 55K-140K op dispatches per prefill.
- No fused selector or fused L>1 sparse SDPA. Fused F-73 mask kernel exists only for L=1 (decode).

The shortest path to >2.5× is **collapsing the gather + concat + mask + SDPA tail into a single Metal launch**, plus folding the selector into a separate compile()'d unit that runs on its own stream. Sections F and "Highest-leverage next moves" are the practical landing pad — sections A–E exist to back those up.

---

## A. Published block-sparse / retrieval-attention kernels that fold selector + gather + SDPA together

### NSA — Native Sparse Attention (DeepSeek 2502.11089) and its kernel descendants

NSA's fundamental win for our problem is that the entire per-query-block pipeline — block-pool keys → block-score → top-K → gathered-K attend → sliding-window attend — happens **inside one Triton kernel** with no materialized intermediate tensors. The paper reports up to 11× decoding speedup and substantial prefill wins on 64K sequences. Algorithmically the closest match for our V1.x pipeline.

- **Paper:** https://arxiv.org/abs/2502.11089 — algorithmic description of the three-branch attention (compression, selection, sliding window) all sharing the same Q stream.
- **Reference Triton impl (fla-org):** https://github.com/fla-org/native-sparse-attention — `parallel_nsa` with kwargs `g_slc`, `g_swa`, `block_indices`, `block_counts`. Notably, since 2025-02-25 the repo ships an "online top-K selection kernel that avoids materializing the attention matrix during selection" and a "fused Triton kernel combining selected attention with sliding attention." That's the exact pattern we'd want to lift.
- **Tilde Research efficient impl:** https://github.com/tilde-research/nsa-impl — a tighter rewrite focused on kernel-throughput.
- **Lucidrains PyTorch ref:** https://github.com/lucidrains/native-sparse-attention-pytorch — readable PyTorch+FlexAttention reference; useful for porting to MLX-Swift's `MLXFast.metalKernel` since the algorithm is one routine end-to-end.
- **Flash Sparse Attention (FSA, 2508.18224, NeurIPS '25):** https://arxiv.org/abs/2508.18224 — fixes NSA's GQA-group-size assumption. They swap the loop order (outer = KV blocks, inner = query tokens) and report **up to 3.5× and on average 1.6× kernel-level latency reduction**, plus 1.36× / avg 1.11× prefill-phase speedup over NSA. Repo: https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention. Qwen2.5-14B has GQA group size 7 (28 Q heads / 4 KV heads), so the FSA loop ordering is the right one for us — porting NSA-style fused selector with FSA loop order should be a Metal port target. Even the algorithmic insight alone — invert the loops so each KV block gets reused across its many query attendees — is something to bake into a Metal kernel from day one.

**Known Metal/MLX port:** none that I can find. mlx-mfa (https://pypi.org/project/mlx-mfa/) ships sparse-prefill features and tile-skip on top of Philip Turner's metal-flash-attention (https://github.com/philipturner/metal-flash-attention), and Turner explicitly notes "the basic algorithm is designed so that customizations like block sparsity should be comparatively trivial to add." That is the foundation to extend if we want a native-Metal NSA-shaped kernel rather than porting from Triton.

### MInference 1.0 / 2.0 — vertical-slash + block-sparse + A-shape

MInference makes a different bet: pattern-classify each head offline, then dispatch to one of three specialized kernels at runtime. The block-sparse kernel mean-pools Q and K in 64-blocks, scores, and runs attention only on selected blocks — a near-identical recipe to what we do, but **fused in Triton**.

- **GitHub:** https://github.com/microsoft/MInference — exposes `block_sparse_attention(q, k, v, topk)`, `vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)`, `streaming_forward(q, k, v, init_num, local_window_num)` as standalone callable kernels. Each is a single launch.
- **Paper:** https://arxiv.org/abs/2407.02490 (NeurIPS '24 spotlight). Reports **up to 10× prefill speedup on LLaMA-3-8B-1M on A100, dropping 1M prefill from 30 min to 3 min**. SGLang + vLLM merged the kernels in 2025-04 with FA3 backend; reported speedups: 1.64× at 64K, 2.4× at 96K, 2.9× at 128K, 5.2× at 256K, 8× at 512K, 15× at 1M.
- **MMInference (ICML'25, 2504.16083):** https://arxiv.org/abs/2504.16083 — extends to VLMs with modality-aware permutation, uses FlashAttention/FlashDecoding/PIT backends.
- **Architectural lesson for us:** their A-shape head bucket essentially treats certain heads as static = static-band-only — no per-chunk top-K needed. We may already be paying selector cost on heads that would be just as accurate with a fixed dense band. Worth auditing per-head recall in the dedupe study.
- **No native Metal port.** Triton-only. PIT compiler (https://dl.acm.org/doi/10.1145/3600006.3613139) is the underlying dynamic-sparsity infra and is also CUDA-only.

### FlashInfer block-sparse SDPA wrappers

FlashInfer's `BlockSparseAttentionWrapper` and `VariableBlockSparseAttentionWrapper` are the production-grade reference for the API shape we want from a fused L>1 sparse SDPA. Each wrapper does a one-shot `plan(indptr, indices, M, N, R, C, ...)` + `run(q, k, v)` and exposes optional return-LSE so it can be composed in a two-pass merge.

- **API docs:** https://docs.flashinfer.ai/api/sparse.html
- **Repo:** https://github.com/flashinfer-ai/flashinfer — auto/FA2/FA3 backends. Vector-sparse attention achieves "90% of dense attention's throughput under identical conditions" — a useful ceiling estimate.
- **MLSys '25 paper:** https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf
- **API to copy verbatim for our Swift surface:** `(indptr, indices, R=blockRows=L, C=blockCols=128, num_qo_heads, num_kv_heads)` + optional `return_lse` → MLX-Swift signature is `func blockSparseSDPA(q, k, v, indptr, indices, blockRows, blockCols, returnLSE: Bool) -> (MLXArray, MLXArray?)`. Even if we keep MLX-Swift code on top, exposing this shape means we can swap the implementation under it.

### SeerAttention — sigmoid-gated block selection (NeurIPS '25)

SeerAttention learns a tiny gate that predicts which blocks matter, trained by self-distillation against pooled true attention. **Up to 2.43× end-to-end prefilling speedup at 128K, average 1.41× across RULER** (vs FA2). At 90% sparsity / 128K it's 7.3× over FA2 single-A100 attention-kernel only.

- **Paper (NeurIPS '25):** https://arxiv.org/abs/2410.13276 + https://openreview.net/forum?id=Nf8yfPDFTl
- **SeerAttention-R for reasoning decode:** https://arxiv.org/abs/2506.08889
- **Repo:** https://github.com/microsoft/SeerAttention
- **Relevance to us:** the gate is a learned cheap projection over pooled Q and K. We currently use a JL projection + max-pool — qualitatively the same shape. SeerAttention adds (a) trainability and (b) TopK *or* threshold mode. The threshold mode is interesting for us because it gives variable selected-block-count per query, which is exactly what `VariableBlockSparseAttentionWrapper` was designed for — meaning we could ship our selector without the fixed-K argPartition (one fewer op).

### MoBA — Mixture of Block Attention (Moonshot)

Trainable, parameterless top-K gating over blocks. The headline kernel claim: **moba_efficient is 40× faster than moba_naive** at 32K (single head, block 2048, top-K 3). The infrastructure is FlashAttention-2 based, single launch per call. The relevant lesson is that almost all of MoBA's win is "stop materializing intermediates and stop dispatching extra kernels for selector + gather."

- **Repo:** https://github.com/MoonshotAI/MoBA
- **Paper:** https://arxiv.org/abs/2502.13189
- **Caveat:** MoBA needs continued training to deploy — not drop-in. But the kernel pattern (block-pool → top-K → flash-attn on selected blocks) is identical to ours and is reference-implemented in a fused-launch style.

### Quest — query-aware sparsity, page-level metadata (ICML '24)

Page-level min/max metadata for fast page scoring, then top-K pages, then attention only on those pages. **Up to 2.23× self-attention speedup, 7.03× inference latency reduction.** The relevance: their score-page-by-min/max trick is much cheaper than our JL-feature × Q.T matmul, and is a candidate for the selector path if we hit accuracy budget.

- **Repo:** https://github.com/mit-han-lab/Quest
- **Paper:** https://arxiv.org/abs/2406.10774
- **Project page:** https://hanlab.mit.edu/projects/quest

### FlexAttention (PyTorch) — declarative `mask_mod` → fused kernel

The right architectural answer to "I want to compose sliding window + retrieval + causal without 12 ops." FlexAttention takes a `mask_mod(b, h, q_idx, kv_idx) -> bool` Python function and lowers it through `torch.compile` to a single fused FlashAttention kernel. Speed-on-Hopper hit 60-100% of FA3 in 2024, and with the new FA4 backend (March 2026) it's **1.6-3.2× faster than the Triton path on B200 and consistently faster on H200** for arbitrary masks including doc-mask + sliding-window + ALiBi.

- **Original blog:** https://pytorch.org/blog/flexattention/
- **FA4 blog (Mar 2026):** https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- **API docs:** https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html
- **Attention Gym examples:** https://github.com/pytorch-labs/attention-gym
- **Strategic implication for us:** `MLX.compile` does graph-level op fusion (see B) but does **not** lower an arbitrary `mask_mod` into a fused attention kernel. MLX-Swift has no FlexAttention equivalent today. Building one is multi-month work, but cheap-imitation route is: build one specialized fused kernel (NSA-shaped: pool-score + topK + concat-and-attend) and call it our F-83 V2.

### FlashAttention-3 / FA4 — sparse and arbitrary-mask extensions

- **FA3 paper:** https://arxiv.org/abs/2407.08608 — 1.5-2.0× over FA2 at FP16, up to 740 TFLOPS / 1.2 PFLOPS FP8. Hopper-only.
- **FA4 (under integration with FlexAttention):** https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute — CuTeDSL, Blackwell-targeted, async/warp-specialized. The relevant insight is that on hardware with fast tensor cores, softmax / exp now bottlenecks and the kernel pipelines two tiles to overlap exp with matmul. **For us, the lesson is: softmax cost is non-trivial; the 2.5 s/chunk overhead might include surprising amounts of exp time** — worth profiling separately. On M-series the SDPA path likely already does this internally, but if we hand-roll a fused kernel we need this pattern.

### Open-source MLX/Metal block-sparse repos — what exists

- **mlx-mfa:** https://pypi.org/project/mlx-mfa/ — adds per-D block configs, sparse prefill features, tile-skip / window. Strongest dense wins on M1 Max are in causal D=64/128 plus tile-skip regimes. This is the closest existing infra to extend.
- **philipturner/metal-flash-attention:** https://github.com/philipturner/metal-flash-attention — Phil Turner explicitly designed it so "customizations like block sparsity should be comparatively trivial to add." Single-headed-attention base + automatic sparsity detection in masks. Block-sparse path indirectly accelerates triangular causal masks already.
- **harvestingmoon/flash_attn_metal_cpp:** https://github.com/harvestingmoon/flash_attn_metal_cpp — metal-cpp header port. Bare-bones, useful as a structural reference.
- **No NSA, MoBA, MInference, or Quest port to Metal/MLX exists publicly as of the search date.** This is a clear open-source opportunity if we get a working port — community wedge.

---

## B. MLX-specific dispatch-overhead reduction techniques

### `MLX.compile` — what it actually fuses

`mx.compile` does **element-wise op fusion** and graph simplification. The official `gelu` example shrinks 7 unary/binary ops into a single kernel and gets **5×** speedup on M1 Max (15.5 ms → 3.1 ms on a (32, 1000, 4096) tensor). For our pipeline this directly attacks steps 3 (max-pool), 4 (score-mask), 11 (mask build) which are all element-wise chains.

- **Compile docs:** https://ml-explore.github.io/mlx/build/html/usage/compile.html
- **Compilation deep-dive:** https://deepwiki.com/ml-explore/mlx/3.4-compilation-and-graph-optimization — `compile_fuse()` recursively traverses outputs → inputs, gathering fusable primitives respecting depth and input-count limits.
- **What `compile` does NOT fuse:**
  - matmuls (op 1, op 2) — these are their own kernels.
  - `take` / gather (op 8) — indexing is not fused into surrounding ops.
  - `MLXFast.scaledDotProductAttention` (op 12) — it's already a single fused kernel; compile can't reach inside it.
  - concat (ops 7, 10, 11) — these are layout ops, not arithmetic.
- **Recompilation triggers:** shape change, dtype change, input-count change. **This is load-bearing for us:** every chunk changes `priorLen` so K_padded varies → compiled function will recompile unless we pad to fixed-bucket sizes (e.g., round K_padded up to the nearest 512 or 1024). Bucketing the gather + concat result is a free easy win.
- **`mx.compile` shrinkable, today, in the sparse path:** the entire 3-4-5 chain (max-pool → mask → argPartition) is fusable except for argPartition itself (a sort) which lives as its own kernel. The 6-7 chain (expand + concat) reduces if we keep static + sliding indices fixed-length so they don't trigger reshapes.

### `mlx.core.fast.metal_kernel` — the real lever

This is the right tool for collapsing the L>1 sparse-attention tail into one launch. The API takes a Metal source string and auto-generates the function signature from input/output shapes and `template` parameters. It's JIT-cached after the first build.

- **API docs:** https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- **Python ref:** https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
- **Real-world reference (TurboQuant fused dequantize):** https://medium.com/@antonrozanov/turboquant-on-mlx-4-6x-kv-cache-compression-with-custom-metal-kernels-9cdee3f7d2a2 — a fused Metal kernel does "the entire quantize pipeline in one dispatch." That's the pattern we need to copy.
- **Lazy-compute integration:** https://github.com/ml-explore/mlx/discussions/1977 discusses how custom kernels participate in MLX's lazy graph. Key gotcha: the kernel must declare all its outputs up-front in `output_shapes` so MLX can size them lazily.
- **Swift surface:** mlx-swift exposes the same API. Code path: `MLXFast.metalKernel(...)` returns a callable that takes `inputs`, `template`, `grid`, `threadgroup`, `output_shapes`, `output_dtypes`. **This is the same machinery already used by F-73 `retrievalAttentionBuildMaskFused` for L=1 — we just need an L>1 variant.**

### `asyncEval` vs `eval` — stream pipelining

- **mlx-swift Transforms+Eval.swift:** https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Transforms+Eval.swift
- **mlx-swift-lm Evaluate.swift:** https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Evaluate.swift
- **Cross-thread `mx.eval`:** https://github.com/ml-explore/mlx/discussions/1448 — confirms multiple Python threads can hold their own `eval` graphs in flight as long as the arrays they touch are disjoint.
- **Streams (unified memory + parallel compute):** https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html — operations placed on different streams run in parallel; MLX automatically inserts cross-stream dependencies when a downstream op consumes a different stream's output.
- **F-78 pattern (selector on a separate stream) — correct and supportable.** What's not in the docs but is implied: between `asyncEval` calls the graph is broken into a separate command-buffer batch, so the next ~30 ops can start dispatching while the previous chunk is still on the GPU. Issue: `asyncEval` itself has CPU-side scheduling cost — if a chunk has only ~10 sparse ops, you're better off batching to amortize.
- **Practical recipe:** wrap chunks-of-N-layers in a single `asyncEval`. Don't `asyncEval` per layer.

### MLX `take` (gather) cost

- **`mlx.core.take_along_axis` docs:** https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html
- **MLX vs PyTorch gather:** https://medium.com/data-science/gpt-from-scratch-with-mlx-acf2defda30e — confirms MLX lacks a direct `gather`, uses `take_along_axis` or fancy indexing for the same effect.
- **No published microbench at our shapes** (`take(priorK[B=1, H_kv=4, P=128K, D=128], positions[B=1, H_kv=4, P_top=K], axis=2)`). We should publish one — likely faster than `take`+`concat` if we instead pre-allocate the output buffer and write straight into it from a custom kernel.
- **Production insight from PyTorch land:** the FA4 blog post (Mar 2026, https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/) explicitly calls out: "Loads on the KV dimension in forward can stall the pipeline, especially for pointer-chasing patterns (e.g., document masking with per-token metadata) where aux-tensor loads are hard to overlap with compute." That's *exactly* our gather pattern. Their fix is to bake the indirection into the kernel and overlap with compute via async pipelines. On Apple Silicon the analog is to use `threadgroup` memory to stage gathered K/V tiles while the previous tile's matmul completes.

---

## C. Recent (2025+) sparse-prefill papers / reported speedups, kernel vs algorithm contribution

| Method | Context | Speedup vs FA dense (prefill) | Kernel fusion contribution? |
|---|---|---|---|
| NSA (DeepSeek) | 64K | 9× attn-only forward, 11× decode | Mostly kernel (Triton-fused selector+attend) |
| FSA (NSA improved) | 64K | +1.6× avg over NSA → ~14× over FA dense | Pure kernel re-ordering (loop swap) |
| MInference 1.0 | 128K LLaMA-3-8B-1M | 2.9× e2e prefill | Mostly kernel; pattern-classify is offline |
| MInference + FA3 (SGLang 2025) | 128K | 2.9× / 5.2× @ 256K / 8× @ 512K / 15× @ 1M | Kernel — FA3 path swap on top of MInference algorithm |
| SeerAttention | 128K | 2.43× e2e prefill; 7.3× attn-only at 90% sparse | Mostly algorithm (learned gate); kernel uses block-sparse FA2 |
| MoBA | 32K | 40× kernel (moba_efficient vs moba_naive) | Almost entirely kernel — same algorithm, fused launch |
| Quest | 32K | 2.23× attn-only, 7.03× latency e2e | ~50/50: page-min/max algorithm is cheap, kernel fuses scoring + selection |
| FlexAttention + FA4 | various | 1.6-3.2× over Triton on B200 | Pure kernel infrastructure |

**Reading:** the kernel-fusion contribution dwarfs the algorithm contribution in most reports. **Our current V1.1's 1.53× is largely algorithm-side**; the 2.5 s/chunk overhead means a 5-10× kernel-fusion win is on the table without any algorithm change. That brings us into the 7-15× e2e bracket, which is competitive with SubQ.

Sources beyond above:
- **SCBench (ICLR'25) — KV-cache lifecycle benchmark:** https://aka.ms/SCBench. Useful for cross-method fair comparison at 128K.
- **The Sparse Frontier — survey:** https://arxiv.org/pdf/2504.17768 — tradeoff space for sparse-attention in transformer LLMs.
- **UniPrefill:** https://arxiv.org/html/2605.06221 — block-wise dynamic sparsification.

---

## D. Apple Silicon Metal-specific tricks

### simdgroup matrix ops on M5 (Neural Accelerators / TensorOps / MPP)

Apple shipped GPU-resident matmul tensor cores with M5. As of Xcode 26.1 the only supported access is Metal Performance Primitives (MPP) framework + Metal Tensor APIs. MLX picks these up via the `mx.fast` path on M5+. The Apple research post reports **up to 4× speedup over M4 baseline for TTFT in LLM inference** when MLX uses Neural Accelerators.

- **Apple ML research post:** https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- **MPP programming guide (PDF):** https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
- **Investigating M5 accelerators:** https://tzakharko.github.io/apple-neural-accelerators-benchmark/
- **M5 cache + tensors analysis:** https://creativestrategies.com/research/m5-apple-silicon-its-all-about-the-cache-and-tensors/
- **Implication for our custom Metal kernel:** if we write a fused selector + SDPA in raw Metal, on M5 we should call `mpp::tensor_ops` matmul primitives from inside the kernel for the score-block matmul and the attention matmul, otherwise we leave 4× on the table.

### simdgroup_matrix (pre-M5)

- **Phil Turner's metal-benchmarks:** https://github.com/philipturner/metal-benchmarks — exhaustive microarch notes on simdgroup_matrix throughput on M1/M2/M3 Max. Critical reference for kernel writing.
- **Threadgroup memory budgets for 14B-class attention:**
  - Qwen2.5-14B: 48 layers, 28 Q heads, 4 KV heads, head_dim 128. For one attention head at L=1024, K_padded=4096, D=128: Q tile ~256 KB, K tile ~1 MB, V tile ~1 MB. Way over threadgroup limit (32 KB on most M-series).
  - Solution: tile the K/V dimension, keep Q resident in registers/threadgroup, sweep K/V tiles. Standard FA pattern — Phil Turner's MFA is the reference impl.
- **Metal Quantized Attention (Draw Things):** https://releases.drawthings.ai/p/metal-quantized-attention-pulling — Int8 attention kernel on M5 Max with row-group-wise scale Q/K and row-wise affine V. Cited in their engineering blog post: https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18 and the MFA 2.0 post: https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c. **Their kernel quantizes online inside the kernel — a real-world precedent for "do a bunch of preprocessing per tile inside the attention kernel rather than as separate launches."**

### MLX-generated "steel" attention internals

`mx.fast.scaled_dot_product_attention` lowers to MLX's "steel" attention kernels — the internal codename. Source: https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/conv.metal lives in the same family. The steel attention kernel already handles GQA, causal, and dense; the public API doesn't expose a block-sparse path, but the templates inside the C++ backend support specialization. Path to follow: copy the steel attention .metal source, add a `block_mask[NB, MB]` indirection over the KV-tile loop, expose as `MLXFast.blockSparseAttention`. This is the same shape as FlashInfer's `BlockSparseAttentionWrapper`.

### WWDC25 / MLX talks

- **Get started with MLX:** https://developer.apple.com/videos/play/wwdc2025/315/
- **Explore LLM on Apple silicon:** https://developer.apple.com/videos/play/wwdc2025/298/

---

## E. Online softmax merge — two-pass dense + sparse

Online softmax (the math that lets FlashAttention combine row-blocks) is built around storing `(m, l)` per output tile and rescaling when you merge tiles. Source: Zihao Ye's notes (https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf), FA2 paper (https://arxiv.org/pdf/2307.08691).

For F-83 the relevant question is: **can we split the K dim into two passes, dense local window + sparse retrieved tokens, and merge their outputs with a logsumexp combine?**

Yes — this is exactly the use case for `return_lse` on FlashInfer's wrappers. The math:

```
out_dense, lse_dense   = SDPA(Q, K_local, V_local, return_lse=True)         # short K dim, ~4096
out_sparse, lse_sparse = SDPA(Q, K_retrieved, V_retrieved, return_lse=True) # ~4096 after dedupe

m = max(lse_dense, lse_sparse)
w_d = exp(lse_dense  - m)
w_s = exp(lse_sparse - m)
out = (w_d * out_dense + w_s * out_sparse) / (w_d + w_s)
```

**Why this beats our current single concat-then-SDPA today:**
- Today: gather + concat builds K_padded of size (local + retrieved + sliding + static) = up to ~5K, then attends.
- Two-pass: each SDPA call has K ~2K. SDPA cost is roughly linear in K, so each pass costs ~40% of the combined. Total per-layer: 2 × 0.4 ≈ 0.8 of current, before counting the **massive saving from not doing the concat at all** (concat is one of the 12 ops).
- Plus the dense local window doesn't need any gather — it's a contiguous slice of priorK/V. So one of the two SDPA calls is gather-free.
- **MLX-Swift status:** `MLXFast.scaledDotProductAttention` does **not** today return LSE. **This is a feature request to file upstream** — see UPSTREAM_ISSUE_DRAFT.md as the existing template. Without LSE we can synthesize it on top of standard SDPA by also computing `max(Q·K^T)` separately, but that defeats the purpose.

Alternative: maintain the merge ourselves using the `softmax+log` from a `logsumexp` returned by a custom kernel. We already need a custom Metal kernel for the fused gather+SDPA — give it `return_lse: Bool` and we get this for free.

**Note: SeerAttention, MInference, and FlashInfer all support exactly this two-pass pattern via `return_lse`** and use it to compose dense sliding-window with sparse-retrieval heads. Concrete usage example in MoBA's `moba_efficient`. References:
- FA2 LSE derivation: https://princeton-nlp.github.io/flash-atttention-2/
- Online softmax: https://medium.com/data-science-collective/online-softmax-to-flash-attention-and-why-it-matters-9d676e7c50a8

---

## F. Selector-cost reduction — amortizing top-K across queries

### Production patterns

1. **Pool-then-score over multi-Q (NSA, MoBA, SeerAttention).** Selector inputs are query *blocks* not individual queries — pool Q over a window of 64 or 128, get one selector per pool. For our prefill at chunk size 128, this means **one selector pass per chunk instead of 128 per-query selectors**. F-79 decode uses `amort=16`; the prefill analog is `amort=L` (the chunk length) since all queries in the chunk share the same priorK / priorV. We may already be doing this; if not it's a free ~100× reduction in selector dispatch count.

2. **IndexCache — cross-layer index reuse (Mar 2026, Tsinghua).** This is the most directly applicable new technique I found. **Adjacent transformer layers share 70-100% of their selected blocks.** They split layers into "Full" (run own indexer) and "Shared" (reuse nearest Full layer's indices). Removes 75% of indexer computation, gives **1.82× prefill speedup, 1.48× decode speedup** on DeepSeek-V3.2 / GLM-5.

   - Paper: https://arxiv.org/abs/2603.12201
   - Repo: https://github.com/THUDM/IndexCache
   - VentureBeat overview: https://venturebeat.com/technology/indexcache-a-new-sparse-attention-optimizer-delivers-1-82x-faster-inference
   - **Direct applicability to F-83:** if we add this on top of V1.x, we run selector on (say) 1 of every 4 layers and reuse indices on the other 3. For Qwen2.5-14B with 48 layers that means **12 selector passes instead of 48 per chunk** — 4× reduction in selector cost, ~1.5-1.8× e2e prefill (matches their measured number). And it's largely orthogonal to the kernel-fusion work in A-E, so it composes.
   - Caveat: "training-free" calibration is on a small set; quality should be measured on RULER or our existing recall harness.

3. **Twilight / hierarchical top-p (Tsinghua NeurIPS '25):** http://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf — variable-K via hierarchical pruning. Useful if our fixed-K argPartition is over-budget for some queries (heads-with-static-attention-like-A-shape).

4. **StreamIndex (memory-bounded streaming top-K):** https://arxiv.org/html/2605.02568v1 — streaming top-K instead of full sort. Could replace argPartition with a single-pass kernel.

5. **MISA — Mixture of Indexer Sparse Attention:** https://arxiv.org/html/2605.07363 — gives per-head indexer mixing; relevant if our heads have different sparsity tolerance.

6. **DeepSeek Sparse Attention (DSA) lightning indexer:** https://docs.sglang.io/basic_usage/deepseek_v32.html — a learned scoring projection over compressed keys, then a sparse attention kernel reads only those. Same shape as ours, production-deployed in DeepSeek-V3.2 and GLM-5.

### What to lift first

For our V1.x branch, IndexCache (F.2) is the highest-leverage single change in this section because:
- It's algorithm-only, no kernel work.
- It composes with everything else.
- 1.5-1.8× speedup is large compared to the 0.78 s FFN floor — would directly translate.

Run a one-day experiment: log selected block_starts per layer for a 128K prompt, compute Jaccard overlap between adjacent-layer selections. If overlap matches the paper's 70-100%, ship IndexCache mode in V1.2.

---

## Highest-leverage next moves for our V1.x branch

1. **Build a single fused Metal kernel `retrievalAttentionFusedSparseSDPA(Q, priorK, priorV, blockStarts, blockCount, slidingRange, staticRange, L) -> (out, lse?)`** modeled after FlashInfer's `BlockSparseAttentionWrapper` API and NSA's three-branch algorithm. Collapses ops 6-12 into one launch. Even with naive kernel code this should drop the 2.5 s/chunk overhead by 5-10× because we're going from 96 chunks × 48 layers × ~7 dispatches = ~32K launches to ~4.6K launches. Expected e2e: 1.53× → 4-6×. Reference implementations to crib from: FSA loop order (https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention), Phil Turner's MFA tile-skip path (https://github.com/philipturner/metal-flash-attention), and the existing F-73 fused mask kernel.

2. **Land IndexCache-style cross-layer index reuse** (https://arxiv.org/abs/2603.12201). Algorithm-only. If adjacent-layer Jaccard ≥ 70% on a 128K Qwen2.5-14B prompt, run selector on 1 of every 4 layers, reuse on the other 3. Expected e2e: extra ~1.4-1.8× on top of #1. Total compounding: 5.5×-10×. **Day-of work to validate, week to ship.**

3. **Switch to two-pass dense+sparse with logsumexp merge** (section E). Requires either upstreaming `return_lse` to `MLXFast.scaledDotProductAttention` (one PR to ml-explore/mlx-swift) or returning LSE from the fused kernel in #1. Eliminates the L=1024 concat (op 10) entirely, and the local-window pass is gather-free. Expected: ~1.3-1.5× on top of #1 even before the gather-free benefit fully lands. The MLX-Swift LSE request is a clean upstream PR candidate.

4. **Bucket K_padded** so `mx.compile`-equivalent caches hit on the post-gather concat output instead of recompiling per chunk. Round K_padded up to the next multiple of 512 or 1024 (mask out the padding). This is the same trick FlashInfer / FlexAttention use to keep BlockMask reuse high. Combined with #1 this should also make the selector-side compile() reuse across all 96 chunks. Expected: 1.1-1.2× from fewer recompiles. **Half-day change.**

5. **On M5, route the fused kernel's matmuls through `mpp::tensor_ops`** so the score-block matmul (step 2) and any attention matmul inside the fused kernel hit Neural Accelerators (https://machinelearning.apple.com/research/exploring-llms-mlx-m5, https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf). Apple reports up to 4× TTFT on LLM inference vs M4 baseline when MLX uses NA. On M-series pre-M5 we fall back to `simdgroup_matrix` ops. **This is the "future-proof for M5 Max purchasers" lever** and is gated to availability checks.

---

## Appendix: full URL list

- https://arxiv.org/abs/2502.11089 (NSA paper)
- https://github.com/fla-org/native-sparse-attention (NSA Triton ref)
- https://github.com/tilde-research/nsa-impl (NSA efficient)
- https://github.com/lucidrains/native-sparse-attention-pytorch (NSA readable)
- https://arxiv.org/abs/2508.18224 (FSA paper)
- https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention (FSA repo)
- https://github.com/microsoft/MInference
- https://arxiv.org/abs/2407.02490 (MInference paper)
- https://arxiv.org/abs/2504.16083 (MMInference)
- https://docs.flashinfer.ai/api/sparse.html
- https://github.com/flashinfer-ai/flashinfer
- https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf
- https://arxiv.org/abs/2410.13276 (SeerAttention)
- https://github.com/microsoft/SeerAttention
- https://arxiv.org/abs/2506.08889 (SeerAttention-R)
- https://github.com/MoonshotAI/MoBA
- https://arxiv.org/abs/2502.13189 (MoBA paper)
- https://github.com/mit-han-lab/Quest
- https://arxiv.org/abs/2406.10774 (Quest paper)
- https://pytorch.org/blog/flexattention/
- https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/
- https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html
- https://github.com/pytorch-labs/attention-gym
- https://arxiv.org/abs/2407.08608 (FA3 paper)
- https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute (FA4 CuTe)
- https://github.com/philipturner/metal-flash-attention
- https://github.com/philipturner/metal-benchmarks
- https://pypi.org/project/mlx-mfa/
- https://github.com/harvestingmoon/flash_attn_metal_cpp
- https://ml-explore.github.io/mlx/build/html/usage/compile.html
- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.take_along_axis.html
- https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- https://deepwiki.com/ml-explore/mlx/3.4-compilation-and-graph-optimization
- https://github.com/ml-explore/mlx/discussions/1448 (cross-thread eval)
- https://github.com/ml-explore/mlx/discussions/1977 (custom kernels lazy compute)
- https://github.com/ml-explore/mlx-swift/blob/main/Source/MLX/Transforms+Eval.swift
- https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLMCommon/Evaluate.swift
- https://medium.com/@antonrozanov/turboquant-on-mlx-4-6x-kv-cache-compression-with-custom-metal-kernels-9cdee3f7d2a2
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf
- https://creativestrategies.com/research/m5-apple-silicon-its-all-about-the-cache-and-tensors/
- https://tzakharko.github.io/apple-neural-accelerators-benchmark/
- https://releases.drawthings.ai/p/metal-quantized-attention-pulling
- https://engineering.drawthings.ai/p/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
- https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
- https://arxiv.org/pdf/2307.08691 (FA2 paper)
- https://princeton-nlp.github.io/flash-atttention-2/
- https://arxiv.org/abs/2603.12201 (IndexCache paper)
- https://github.com/THUDM/IndexCache
- https://venturebeat.com/technology/indexcache-a-new-sparse-attention-optimizer-delivers-1-82x-faster-inference
- http://people.iiis.tsinghua.edu.cn/~gaomy/pubs/twilight.neurips25.pdf
- https://arxiv.org/html/2605.02568v1 (StreamIndex)
- https://arxiv.org/html/2605.07363 (MISA)
- https://arxiv.org/abs/2409.10516 (RetrievalAttention)
- https://github.com/microsoft/RetrievalAttention
- https://aka.ms/SCBench
- https://arxiv.org/pdf/2504.17768 (Sparse Frontier survey)
- https://developer.apple.com/videos/play/wwdc2025/315/
- https://developer.apple.com/videos/play/wwdc2025/298/
