# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-11 05:50
**Branch**: `ek/tom-eric-moe-tuning`
**Commit**: `4526cdb perf: TASK 4 COMPLETE — native prefill wired into real prepare() path`
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
| summarization | 1024 | 1008 | no-quant | 8630.6 | 192.7 | 200 | 117ms | — | — | — | — | 2.45GB | 3.72GB | 8MB | 264MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 4096 | 4088 | no-quant | 7954.2 | 187.0 | 200 | 515ms | — | — | — | — | 2.45GB | 5.68GB | 20MB | 938MB | This excerpt from *The Great Gatsby* is a collection of dist |
| summarization | 8192 | 8192 | no-quant | 7914.0 | 178.6 | 200 | 1036ms | — | — | — | — | 2.45GB | 5.92GB | 36MB | 1.79GB | This is a fascinating excerpt from **The Great Gatsby** by F |
