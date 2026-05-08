# TriAttention V3 + longctx — M5 Validation Report

**Date:** 2026-05-07 night sprint
**Goal:** validate V3+longctx on M5 across model families, derive
data-driven default-on rule, measure compression with/without TQ+ stacking.

## Tom's strategic asks
1. Validate V3+longctx across every model family on M5
2. Various context lengths
3. Tests at increased eviction numbers (10-30%)
4. Coherence tests
5. TQ+ stacks with V3, best asym per family
6. Default-on at X% via data-driven decisions
7. Manual pre-load of RAG vs learn-on-fly
8. Correctness first, push context only after

## Status snapshot

| objective | status | gate |
|---|---|---|
| V3 hooks ported across families | **partial — Tier-1 done** | more model classes need port |
| Context-length sweep infrastructure | **shipped** | runs when GPU is free |
| Eviction-rate sweep (10-40%) | **shipped** | needs matrix execution |
| Coherence harness | **shipped** | matrix-driven |
| TQ+ stacking on M5 | **estimator only** | #187 (real stacked cache class) |
| Per-family default-on adapter | **shipped (provisional)** | matrix data refines table |
| Pre-load RAG mode | **shipped** | `compact` mode in coherence_driver |
| Correctness gate (build+tests) | **green — 14/14** | local-only; live runs need M5 GPU |

## What landed this sprint

### V3 model-family hooks (Tier-1 ported)
| family | attention hook | newCache install | notes |
|---|---|---|---|
| Qwen3 | ✅ pre-existing (Qwen3.swift) | ✅ pre-existing | reference impl |
| Qwen3.5 | ✅ added | ✅ added (pre-turbo branch) | mutually exclusive with TQ+ today |
| Qwen3MoE | ✅ added | (uses Qwen3 path) | MoE attention forward identical |
| Qwen2 | ✅ added | ✅ added | clean port |
| Llama | ✅ added | ✅ added (stored configuration) | dense Llama-3 family |
| Mistral3 | ✅ added | ✅ V3 branch added | bypasses sliding-window for V3 |
| Phi | ✅ added | (default cache) | hook-only port |
| Phi3 | ✅ added | (default cache) | hook-only port |
| Gemma3 | ✅ added | (existing newCache) | hook-only port |
| GLM4 | ✅ added | (default cache) | hook-only port |

**Models that still need hooks** (not blocking the demo, but flagged):
- Qwen3Next, Qwen35MoE — long-context models
- Gemma4, Gemma3nText
- DeepseekV3, DeepseekV4 — MoE
- NemotronH — hybrid Mamba+attention (V3 only fires on attention layers)
- LFM2/LFM2MoE — hybrid SSM
- Olmo2/Olmo3, Apertus, Cohere, Granite, Phi-MoE, BailingMoe, BaichuanM1,
  Bitnet, Exaone4, Ernie4_5, Internlm2, Jamba, MiMo, MiMoV2Flash,
  MiniCPM, MiniMax, MiniMaxM2, NanoChat, OpenELM, SmolLM3, Lille130m,
  AfMoE, FalconH1, GatedDelta, Gemma2, GLM4MOE/Lite, GPTOSS, Olmo2/3,
  OlmoE, SSM, Starcoder2

The pattern is now standardized:
- attention forward: insert `captureV3PreRopeQuery(queries: queries, B: B, cache: cache)` BEFORE `applyRotaryPosition`
- model factory: extension or override `newCache` with `makeV3CacheStack(...)` branch

### Infrastructure
- **`applyRotaryPosition` is V3-aware**: uses `triCache.logicalOffset` when cache is `TriAttentionKVCache`. RoPE positions stay tied to original token stream after V3 compaction. Backwards compatible.
- **`captureV3PreRopeQuery` helper**: one call per attention class, no-op when cache isn't V3.
- **`makeV3CacheStack`**: factory that builds engine + per-layer caches in one call, env-gated.
- **CompressionStats telemetry**: rolling rounds/before/evicted/kept, savings%, plus `stackedWithTurboQuant(bitsPerCell:)` first-order estimator for V3+TQ+ stacked savings.
- **`TriAttentionDefaults`**: per-family safe%/aggressive% rates, env-overridable. Used by callers to auto-enable V3 when `LONGCTX_ENDPOINT` is set.

### Harness (longctx repo, services/longctx-svc/harness/)
- `m5_validation_matrix.sh`: sweeps (model × ctx × eviction% × mode) cells, per-cell CSV row.
- `m5_recommend_default.py`: deterministic decision tree → per-family safe% / aggressive% / always_on.
- `m5_coherence_smoke.sh`: single-cell quick smoke against vllm-swift.
- `coherence_driver.py`: 5-family multi-hop benchmark (single_fact, multi_hop, contradiction, aggregation, temporal) with 4-outcome scoring (exact / retrieval_miss / reasoning_fail / coherent_wrong).

### Tests
- **14/14 unit tests pass**:
  - 8 prior V3 mechanic tests
  - +1 compression telemetry accumulation
  - +1 makeV3CacheStack helper builds caches
  - +1 makeV3CacheStack returns nil when V3 disabled
  - +1 Defaults family detection (8 model ids)
  - +1 Defaults decision tree (operator override + LONGCTX gate + unknown family)
  - +1 stackedWithTurboQuant first-order math

## How to run the matrix overnight

**Pre-conditions:**
```bash
# longctx-svc + vllm-swift installed (brew or pip)
brew install longctx-svc vllm-swift
# OR pip install longctx-svc, install vllm-swift via TheTom/vllm-swift bottle
```

**One-shot:**
```bash
bash /path/to/longctx/services/longctx-svc/harness/m5_validation_matrix.sh
```

**Default sweep:** 7 models × 5 ctxs × 5 rates × 3 modes = **525 cells**.
Per-cell budget ~5 min wall = ~44 hr. Cap with envs:
- `MODELS=mlx-community/Qwen3-0.6B-4bit` for 1 model (~75 cells)
- `CTX_LADDER="2048 4096 8192"` for tighter ctx ladder
- `EVICTION_RATES="0.0 0.20"` for baseline-vs-one-rate
- `MAX_TOKENS=8192` to skip large contexts

**Analyze:**
```bash
python3 m5_recommend_default.py --csv /tmp/m5_matrix/results.csv \
  --out /tmp/m5_matrix/recommendation.md
```

Recommendation table is the data-driven answer to "always-on at X%."

## Known gaps

### #187: V3 + TQ+ stacking on M5 (not done)
`TriAttentionKVCache` extends `KVCacheSimple` (FP16). Enabling V3 today
bypasses `TurboQuantKVCache`. Real stacking needs a
`TriAttentionTurboKVCache` subclass of `TurboQuantKVCache` that:
- Fires V3 hooks on `update()` (calibrate Q, score K)
- Overrides `removePositions` to dequant → slice → repack the bit-packed buffers
- Reuses the engine + telemetry path

**Workaround until it lands:** `CompressionStats.stackedWithTurboQuant(bitsPerCell:)`
gives the first-order math estimate. For Qwen3 K8V4 at 30% V3 evict:
stacked savings = 1 - (0.7 × 12/16) = **47.5%**. The matrix runner
auto-uses this estimator when reporting cells.

### #186: Hook port to non-Tier-1 models
~46 model classes still default to `KVCacheSimple` even when V3
enabled. They build clean (the hook is a no-op for non-V3 caches) but
they don't participate in V3. Each port is ~5 lines (`captureV3PreRopeQuery` + `applyRotaryPosition` already handles offset).

### Compression A/B numbers (not yet measured)
The matrix collects per-cell `compression_pct` from `[V3-compaction]`
log lines. To get the V3+TQ+ stacked number, post-process with
`stackedWithTurboQuant(bitsPerCell: parseTurboScheme("turbo8v4"))`.
Real stacked numbers need #187.

## Per-family TQ+ asym recipe (provisional, AMD-derived)

| family | recommended TQ+ codec | reason |
|---|---|---|
| Qwen2/2.5 | turbo8v4 | TurboQuant+ + AMD validated; standard MHA |
| Qwen3/3.5 | turbo8v4 or turbo8v3 | qNorm robust to V codec compression |
| Llama | turbo8v4 | position-sensitive; keep K=8 |
| Mistral | turbo8v4 | sliding window unaffected |
| Phi | turbo8v4 → fall back to turbo4 if quality drops | smaller model, less codec slack |
| Gemma | turbo0v4 (raw K, V=4) | Gemma's local/global mix; protect K |
| Nemotron-hybrid | turbo8v4 (attention layers only) | Mamba layers don't go through TQ+ |

**Replace these with matrix-derived numbers** once V3+TQ+ stacking lands.

## Decision rule (when V3 should be default-on)

```
if VLLM_TRIATT_ENABLED is set:
    obey it (operator override)
elif LONGCTX_ENDPOINT not set:
    V3 OFF (no rescue path → eviction destructive)
elif family in known-good list AND family.alwaysOn:
    V3 ON at family.safeRate (default) or family.aggressiveRate (opt-in)
else:
    V3 OFF
```

Implementation: `TriAttentionDefaults.provisional.defaultRate(for: modelId, env: env)`.

## Outstanding follow-ups (not blocking)

| task | priority | size |
|---|---|---|
| Port V3 hook to MoE families (Qwen3MoE done; Qwen35MoE, GLM4MOE, OlmoE, etc) | P1 | ~15 model classes × 5 lines |
| Port V3 hook to hybrid arch (NemotronH, LFM2, Jamba — only attention layers) | P1 | per-class layer-type filter |
| Build `TriAttentionTurboKVCache` (V3+TQ+ stacking) | P0 | 200-400 lines + tests |
| Run matrix on M5 GPU overnight | P0 | tonight (when GPU free) |
| Replace provisional defaults with matrix-derived numbers | P0 | post-matrix |
| Pre-load RAG mode test alongside learn-on-fly (compact mode in driver) | P1 | exists; needs benchmark cells |

## Build state at end of sprint

- **mlx-swift-lm**: branch `feature/triattention-v3` HEAD `2669448`, build clean, 14/14 V3 tests pass
- **longctx**: branch `main` HEAD has matrix runner + recommender; tests pass
- All commits pushed
