# F-83 W5 PRD — Speculative decoding (DFlash-MLX integration)

**Status**: planned (post-fused-block-sprint)
**Branch**: `feature/retrieval-attention` (local-only)
**Target model**: Qwen2.5-14B-Instruct-1M-4bit on M5 Max 128GB
**Goal**: Bypass the bandwidth wall on decode by parallelizing N forward passes via draft-and-verify speculative decoding. Expected: 1.5-2× throughput when draft acceptance rate ≥ 50%.

---

## Why this and why now

Per the F-83 night-sprint log (this directory's `F83_NIGHT_LOG.md`), single-stream dense decode at 16K on Qwen2.5-14B-1M-4bit hits ~26.5 ms/step on M5 Max. Bandwidth-wall math (~22 ms for weight reads at 600 GB/s) confirms there's only ~4 ms of headroom in single-stream decode. Closing the gap to Python requires either:
1. Microkernel re-tuning at the Cmlx level (multi-day; high effort)
2. Speculative decoding (multi-token verification in one forward pass; high payoff)

Speculative decoding is the bigger lever. At 8-token draft + accept rate 50%, effective throughput = 4 tokens / target step = (26.5 ms / 4) = 6.6 ms per accepted token = 150 tok/s (vs current 38 tok/s).

DFlash-MLX (Tom's `~/dev/obsidian/src/dflash_draft_trainer.py`) is the in-progress draft model. mlx-swift-lm needs the verification harness.

## Acceptance criteria

- Loads draft (Qwen2.5-1.5B-4bit or smaller) alongside target (14B) in the same MLXLMCommon process.
- Decode loop alternates: draft generates N tokens; target verifies in one batch.
- Reject-sample at logit comparison; rewind KV cache on rejection.
- Target throughput: ≥ 1.5× current dense decode tok/s at 16K.
- Quality: top-1 accuracy ≥ 99% vs single-stream argmax over 256-token generations.
- No regression in dense-only path when speculation disabled.

## Implementation plan

1. `DraftModelContainer`: like `ModelContainer` but pinned to the small model + its KV cache.
2. `SpeculativeDecodeStep`:
   - Input: current target KV cache state + last accepted token
   - Run `draft.generate(N=8)` greedy → 8 candidate tokens t_{i+1..i+8}
   - Build target input batch of shape `[1, 1+8]`: `[last_accepted, t_{i+1}, ..., t_{i+8}]`
   - Single target forward pass — extends target cache by 9 positions, produces logits at all 9
   - For each draft position j: compute target acceptance probability p = min(1, target_p_j / draft_p_j). Sample u ∼ U(0,1); accept iff u ≤ p.
   - On first rejection: rewind target KV cache offset by (8 - accepted_count); discard draft beyond rejection point; sample replacement from adjusted distribution (rejection sampling residual).
3. `cache.offset` rewind: StandardKVCache has `idx` and `offset` — rewind both.
4. CLI flag `--speculative-decode 8` to enable.

## Risk

- KV cache rewind on rejection — needs careful semantics with windowed caches.
- Draft must use same tokenizer as target. Smaller Qwen2 variants share tokenizer.
- Target forward at L=9 (vs L=1 decode) needs full prefill-style attention mask — but the mask is just causal over L=9.
- Memory: draft adds ~3 GB resident.

## Defer

- Multi-draft (different N per accept rate). Start with N=8 fixed.
- Adaptive N tuning based on observed acceptance rate.
- Tree-based speculation (multi-branch drafts). Linear chain first.

## References

- [Speculative Decoding paper](https://arxiv.org/abs/2211.17192)
- [Medusa: Multi-head decoding](https://arxiv.org/abs/2401.10774)
- ReDrafter (Apple ML), EAGLE-2, MTPLX
- Tom's `~/dev/obsidian/src/dflash_draft_trainer.py` — draft training
- `Tests/MLXLMTests/NGramSpeculativeTests.swift` — existing speculative-decode test infra (may have reusable bits)
