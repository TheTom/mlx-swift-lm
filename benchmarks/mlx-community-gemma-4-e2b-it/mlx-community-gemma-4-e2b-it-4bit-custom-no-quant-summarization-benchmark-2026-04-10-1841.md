# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-10 18:41
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
| summarization | 1024 | 1008 | no-quant | 8806.1 | 193.4 | 200 | 115ms | — | — | — | — | 2.45GB | 3.26GB | 8MB | 264MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | no-quant | 7444.5 | 187.8 | 200 | 550ms | — | — | — | — | 2.45GB | 3.59GB | 28MB | 938MB | This excerpt appears to be a collection of interconnected pa |
| summarization | 8192 | 8192 | no-quant | 5742.5 | 181.4 | 200 | 1427ms | — | — | — | — | 2.45GB | 5.38GB | 36MB | 1.79GB | This is an excerpt from **Nick Carraway's narration** in **T |
