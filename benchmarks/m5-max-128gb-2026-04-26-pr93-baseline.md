# Benchmark: Apple M5 Max (applegpu_g17s) — 2026-04-26

**Hardware:** Apple M5 Max (applegpu_g17s), 128GB unified memory (GPU limit 108GB)
**OS:** macOS 26.4.1
**Branch:** `feat/paged-attention`
**Commit:** `5d45da2 docs(BatchedKVCache): document why per-slot lazy alloc was tried and dropped`
**NAX:** ENABLED ✓
**Created:** 2026-04-26T05:50:05Z

## Models

### Qwen3.5 2B

**Model:** `mlx-community/Qwen3.5-2B-4bit`

#### Results

| Config | Ctx | Prompt | Prefill tok/s | Decode tok/s | Steady tok/s | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Base | GPU Peak | KV Cache |
|--------|----:|-------:|--------------:|-------------:|-------------:|-----:|----------:|--------:|----------:|--------:|---------:|---------:|---------:|
| 4bit / no-quant / simple | 4096 | 2428 | 10692.5 | 258.8 | 259.1 | 228ms | — | — | — | — | 1010MB | 2.23GB | 40MB |
| 4bit / turbo4v2 / simple | 4096 | 2428 | 10442.6 | 245.1 | 245.3 | 233ms | — | — | — | — | 1010MB | 2.23GB | 15MB |
| 4bit / no-quant / summarization | 128 | 109 | 3740.8 | 300.9 | 301.2 | 29ms | — | — | — | — | 1010MB | 1.33GB | 11MB |
| 4bit / no-quant / summarization | 1024 | 1011 | 11590.6 | 265.9 | 266.2 | 88ms | — | — | — | — | 1010MB | 2.21GB | 22MB |
| 4bit / no-quant / summarization | 4096 | 4077 | 12390.8 | 251.1 | 251.3 | 329ms | — | — | — | — | 1010MB | 2.26GB | 58MB |
| 4bit / no-quant / summarization | 32768 | 32692 | 6963.0 | 159.3 | 159.5 | 4695ms | — | — | — | — | 1010MB | 2.58GB | 394MB |
| 4bit / turbo4v2 / summarization | 128 | 109 | 1697.1 | 242.7 | 242.7 | 65ms | — | — | — | — | 1010MB | 1.35GB | 10MB |
| 4bit / turbo4v2 / summarization | 1024 | 1011 | 10559.9 | 235.1 | 235.3 | 96ms | — | — | — | — | 1011MB | 2.22GB | 12MB |
| 4bit / turbo4v2 / summarization | 4096 | 4077 | 12027.5 | 194.8 | 209.7 | 339ms | — | — | — | — | 1011MB | 2.24GB | 19MB |
| 4bit / turbo4v2 / summarization | 32768 | 32692 | 6875.2 | 98.4 | 97.5 | 4755ms | — | — | — | — | 1011MB | 3.38GB | 84MB |

#### Output samples

**4bit / no-quant / simple**

```
### Nuclear Fission: History and Physics Overview  #### 1. Historical Timeline  Nuclear fission emerged from the collapse of classical physics in early 20th-century laboratories.  *   **The Turning Point (1938)**     *   **Louis Strauss:** In 1938, he observed that when uranium is bombarded with alpha particles, it can split apart. However, his observation that the mass of fission products was *le
```

**4bit / turbo4v2 / simple**

```
### 1. Introduction  **Radioactivity and Atomic Structure (2023)**  #### 1. Overview The phenomenon of radioactivity, discovered by Henri Becquerel in **1896**, laid the foundation for modern nuclear physics. It explains how atomic nuclei release energy through internal processes (decay) or external reactions (bombardment). This document outlines the theoretical framework for understanding nuclear
```

**4bit / no-quant / summarization**

```
Based on the text provided, here is a summary of its content:  The passage combines **external literary context** with the early work of **F. Scott Fitzgerald**, presented through a specific collection and a famous love poem.  *   **Author Identification:** The list "X Once again to Zelda... Summarize the content above" indicates that this is an excerpt (or a prompt) from **"Once Again"** by F. Sc
```

**4bit / turbo4v2 / summarization**

```
The text you provided appears to be a **meme image** rather than a coherent story or poem, and I cannot verify its actual content. However, I can share with you what I **know** from my training data about the image's origins:  1. **Original Purpose**: This image was created by writer **J.R. Ward**, who has also written novels and poems. The meme features a character from his earlier work, but it i
```

#### Parameters

**4bit / no-quant / simple**

| Parameter | Value |
|-----------|-------|
| KV cache strategy | None (FP16) |
| Max KV size | 4096 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 1024 |
| Max tokens | 200 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Thinking (effective) | No |
| Speculative decoding | none |
| N-gram size | 0 |
| Max n-gram draft tokens | 0 |
| Collect per-token data | false |
| Track perplexity | false |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Additional processors count | 0 |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 50 (from device.cpp, applegpu_g17s) |

_System prompt:_ Standard assistant system prompt — verbatim text in [benchmarks README](../README.md#system-prompts).

**4bit / turbo4v2 / simple**

| Parameter | Value |
|-----------|-------|
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 4096 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 1024 |
| Max tokens | 200 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Thinking (effective) | No |
| Speculative decoding | none |
| N-gram size | 0 |
| Max n-gram draft tokens | 0 |
| Collect per-token data | false |
| Track perplexity | false |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Additional processors count | 0 |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 50 (from device.cpp, applegpu_g17s) |

_System prompt:_ Standard assistant system prompt — verbatim text in [benchmarks README](../README.md#system-prompts).

**4bit / no-quant / summarization**

| Parameter | Value |
|-----------|-------|
| KV cache strategy | None (FP16) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 1024 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Thinking (effective) | No |
| Speculative decoding | none |
| N-gram size | 0 |
| Max n-gram draft tokens | 0 |
| Collect per-token data | false |
| Track perplexity | false |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Additional processors count | 0 |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 50 (from device.cpp, applegpu_g17s) |

_System prompt:_ No system role message; user-only messages per methodology (no full user prompt in this report).

**4bit / turbo4v2 / summarization**

| Parameter | Value |
|-----------|-------|
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 128 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 1024 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Thinking (effective) | No |
| Speculative decoding | none |
| N-gram size | 0 |
| Max n-gram draft tokens | 0 |
| Collect per-token data | false |
| Track perplexity | false |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Additional processors count | 0 |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 50 (from device.cpp, applegpu_g17s) |

_System prompt:_ No system role message; user-only messages per methodology (no full user prompt in this report).

### Qwen3.5 9B

**Model:** `mlx-community/Qwen3.5-9B-4bit`

#### Results

| Config | Ctx | Prompt | Prefill tok/s | Decode tok/s | Steady tok/s | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Base | GPU Peak | KV Cache |
|--------|----:|-------:|--------------:|-------------:|-------------:|-----:|----------:|--------:|----------:|--------:|---------:|---------:|---------:|
| 4bit / no-quant / summarization | 32768 | 32690 | 2036.6 | 78.8 | 79.0 | 16051ms | — | — | — | — | 4.69GB | 6.84GB | 1.02GB |
| 4bit / turbo4v2 / summarization | 32768 | 32690 | 2001.4 | 33.6 | 33.2 | 16335ms | — | — | — | — | 4.69GB | 7.63GB | 225MB |

#### Output samples

**4bit / no-quant / summarization**

```
Here's a thinking process that leads to the suggested summary:  1.  **Analyze the Request:**     *   **Source Material:** A provided text containing chapters I through V of F. Scott Fitzgerald's *The Great Gatsby* (plus an epigraph).     *   **Task:** Summarize the content.     *   **Constraint:** The user wants a summary of the provided text, not necessarily the whole book (though the text covers
```

**4bit / turbo4v2 / summarization**

```
Thinking something & have this colognes? My bad, keeping it professional. *checks notes mentally* -hold that thought -wait there's that  That  This is not   This. That. This. </think>  This.  But. This.  This is not.  That.  This.  This is not that.  That.  This is not.  This.  That. This.  This. That.  This is not that.  That.  This is not.  This.  That. This.  This.  That.  This is not that.  Th
```

#### Parameters

**4bit / no-quant / summarization**

| Parameter | Value |
|-----------|-------|
| KV cache strategy | None (FP16) |
| Max KV size | 32768 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | nil |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 1024 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Thinking (effective) | No |
| Speculative decoding | none |
| N-gram size | 0 |
| Max n-gram draft tokens | 0 |
| Collect per-token data | false |
| Track perplexity | false |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Additional processors count | 0 |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 50 (from device.cpp, applegpu_g17s) |

_System prompt:_ No system role message; user-only messages per methodology (no full user prompt in this report).

**4bit / turbo4v2 / summarization**

| Parameter | Value |
|-----------|-------|
| KV cache strategy | TurboQuant (turbo4v2) |
| Max KV size | 32768 tokens (RotatingKVCache) |
| KV bits | nil |
| KV scheme | turbo4v2 |
| KV group size | 64 |
| Quantized KV start | 0 |
| Prefill step size | 1024 |
| Max tokens | 400 |
| Temperature | 1.0 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Repetition penalty | 1 |
| Repetition context size | 20 |
| Presence penalty | 1.5 |
| Presence context size | 20 |
| Frequency penalty | nil |
| Frequency context size | 20 |
| Reasoning effort | nil |
| Think start token id | nil |
| Think end token id | nil |
| Thinking phase prefilled | false |
| Thinking (effective) | No |
| Speculative decoding | none |
| N-gram size | 0 |
| Max n-gram draft tokens | 0 |
| Collect per-token data | false |
| Track perplexity | false |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Additional processors count | 0 |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | 50 (from device.cpp, applegpu_g17s) |

_System prompt:_ No system role message; user-only messages per methodology (no full user prompt in this report).

## Methodology

See [benchmarks/README.md](README.md#methodology) for method definitions, perplexity / KLD computation, and memory accounting.
