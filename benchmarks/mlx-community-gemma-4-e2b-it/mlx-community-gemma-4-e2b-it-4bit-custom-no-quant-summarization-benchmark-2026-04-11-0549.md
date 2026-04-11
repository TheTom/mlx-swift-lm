# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-11 05:49
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
| summarization | 1024 | 1008 | no-quant | 2047.9 | 192.4 | 200 | 492ms | — | — | — | — | 2.45GB | 6.30GB | 11MB | 264MB | This excerpt from *The Great Gatsby* details the narrator's  |
| summarization | 4096 | 4088 | no-quant | 1634.8 | 186.0 | 200 | 2505ms | — | — | — | — | 2.45GB | 11.02GB | 19MB | 938MB | This provided text is a collection of excerpts from **F. Sco |
| summarization | 8192 | 8192 | no-quant | 959.8 | 169.8 | 200 | 8596ms | — | — | — | — | 2.45GB | 17.91GB | 35MB | 1.79GB | Here is a summary and analysis of the provided text excerpts |
