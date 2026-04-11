# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-10 18:14
**Branch**: `ek/tom-eric-moe-tuning`
**Commit**: `5cc678e perf: prefill profiling instrumentation + max_ops=50`
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
| summarization | 1024 | 1008 | no-quant | 8210.7 | 193.9 | 200 | 123ms | — | — | — | — | 2.45GB | 3.26GB | 8MB | 264MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 4096 | 4088 | no-quant | 8375.3 | 188.6 | 200 | 489ms | — | — | — | — | 2.45GB | 3.31GB | 28MB | 938MB | The provided text is an excerpt from **The Great Gatsby** by |
