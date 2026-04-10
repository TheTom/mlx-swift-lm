# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-10 15:34
**Branch**: `ek/tom-eric-moe-tuning`
**Commit**: `202dd91 committing completed bf16 gemma4 benchmark and minor note formatting`
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
| summarization | 1024 | 1008 | no-quant | 7774.7 | 194.1 | 200 | 130ms | — | — | — | — | 2.45GB | 3.22GB | 10MB | 264MB | The provided text is an excerpt from **The Great Gatsby** by |
| summarization | 4096 | 4088 | no-quant | 7143.0 | 187.0 | 200 | 573ms | — | — | — | — | 2.45GB | 3.32GB | 28MB | 938MB | This excerpt is from **The Great Gatsby** by F. Scott Fitzge |
| summarization | 8192 | 8192 | no-quant | 5489.7 | 177.9 | 200 | 1493ms | — | — | — | — | 2.45GB | 3.34GB | 52MB | 1.79GB | This is a fascinating excerpt from **The Great Gatsby** by F |
