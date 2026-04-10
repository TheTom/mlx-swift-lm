# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-10 17:07
**Branch**: `ek/tom-eric-moe-tuning`
**Commit**: `c2361be perf: add generation stream + upstream SDPA for prefill pipelining`
**Quantization**: custom
**Model**: `mlx-community/gemma-4-e2b-it-4bit`

## Hardware

| Property | Value |
|----------|-------|
| Chip | Apple M5 Max (applegpu_g17s) |
| System RAM | 128GB |
| GPU Memory Limit | 108GB |
| macOS | 26.3.1 |

## Parameters

| Parameter | Value |
|-----------|-------|
| Temperature | 0.6 |
| Top P | 0.95 |
| Top K | 20 |
| Min P | 0.0 |
| Max Tokens | 200 |
| Thinking | No |
| Perplexity tracking (MLX_BENCH_PPL) | No |
| KL divergence (MLX_BENCH_KLD) | No |
| Batch size (MLX_BENCH_BATCH) | 1 |
| Speculative decoding | none |
| Max ops per buffer (MLX_MAX_OPS_PER_BUFFER) | default |

## Methodology

For details see [here](../README.md#methodology).

## Results

| Method | Context Limit | Prompt Tokens | KV Config | Prefill tok/s | Gen tok/s | Gen Tokens | TTFT | Think PPL | Gen PPL | Think KLD | Gen KLD | GPU Baseline | GPU Peak | KV Delta | KV Cache | Output |
|--------|---------------|---------------|-----------|---------------|-----------|------------|------|-----------|---------|-----------|---------|-------------|----------|----------|----------|--------|
| summarization | 1024 | 1008 | no-quant | 7740.3 | 194.3 | 200 | 131ms | — | — | — | — | 2.45GB | 3.22GB | 11MB | 264MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | no-quant | 7075.9 | 186.9 | 200 | 578ms | — | — | — | — | 2.45GB | 3.32GB | 27MB | 938MB | This excerpt is a collection of fragmented passages from **F |
| summarization | 8192 | 8192 | no-quant | 5486.0 | 177.9 | 200 | 1494ms | — | — | — | — | 2.45GB | 3.34GB | 53MB | 1.79GB | This is an excerpt from **Nick Carraway's narration** in **T |
