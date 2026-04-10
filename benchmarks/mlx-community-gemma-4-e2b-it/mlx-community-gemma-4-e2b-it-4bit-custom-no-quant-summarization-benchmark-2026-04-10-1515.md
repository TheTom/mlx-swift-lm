# Inference Benchmark - mlx-community/gemma-4-e2b-it-4bit

**Date**: 2026-04-10 15:15
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
| summarization | 128 | 110 | no-quant | 4017.1 | 191.3 | 195 | 28ms | — | — | — | — | 2.45GB | 2.62GB | 3MB | 67MB | The provided text is a fragmented excerpt, likely from a lit |
| summarization | 1024 | 1008 | no-quant | 8178.1 | 184.6 | 200 | 124ms | — | — | — | — | 2.45GB | 3.22GB | 10MB | 264MB | This excerpt from *The Great Gatsby* introduces a narrator w |
| summarization | 4096 | 4088 | no-quant | 6699.5 | 179.0 | 200 | 611ms | — | — | — | — | 2.45GB | 3.32GB | 38MB | 938MB | This excerpt appears to be from **The Great Gatsby** by F. S |
| summarization | 8192 | 8192 | no-quant | 5186.2 | 171.6 | 200 | 1580ms | — | — | — | — | 2.45GB | 3.34GB | 53MB | 1.79GB | This is a fascinating and dense excerpt from **Nick Carraway |
