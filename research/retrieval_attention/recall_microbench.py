"""Synthetic recall@k microbench for the selector mixture.

This is the [F-NN] empirical answer to Codex's flagged risk
(PRD line 280): "Could be 32 dims too lossy, normalization wrong,
top-k too small, or content+position mixture wrong."

Setup:
- d_head = 128 (Llama-3 / Qwen2.5)
- seq_len varies (1K, 16K, 256K equivalents)
- block_size = 64
- N "planted needles" inserted at known positions with high cosine
  similarity to the query
- The rest are i.i.d. Gaussian noise
- We compute "dense top-k" via raw q·k (ground truth)
- Then "sparse top-k" via JL content / V3-trig / mixture over blocks
- Recall@k = |dense_top_k ∩ sparse_top_k| / k

This is SYNTHETIC. Real recall@k against a model lives in Week 1 once
we wire the diagnostic harness against mlx-swift-lm. But this gives us
the first reading of whether the selector design is in the right ZIP
code at all.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from selector import (
    CONTENT_DIM,
    SELECTOR_DIM,
    TRIG_DIM,
    block_max_norm_sentinel,
    block_mean_pool,
    jl_projection_matrix,
    score_blocks,
    top_k_block_indices,
    v3_trig_features,
)


@dataclass
class BenchConfig:
    seq_len: int
    d_head: int
    block_size: int
    n_needles: int
    needle_similarity: float  # how much the planted needle leans toward q (0..1)
    n_queries: int
    seed: int = 42


@dataclass
class Recall:
    selector: str
    recall_at_k: float  # mean over n_queries
    top_k_blocks: int


def synthesize_keys(
    *,
    cfg: BenchConfig,
    query: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate K matrix with planted needles.

    Args:
        query: [d_head] query vector (normalized).
        cfg: bench config.
        rng: numpy generator.

    Returns:
        keys: [seq_len, d_head]
        needle_positions: [n_needles] positions of the planted needles.
    """
    # All-noise base.
    keys = rng.standard_normal((cfg.seq_len, cfg.d_head)).astype(np.float32)
    # Plant n_needles tokens scattered through context.
    # Spread them: 1/4 of the way, 1/2, 3/4, ... to test position diversity.
    needle_positions = np.linspace(
        cfg.seq_len // 8, cfg.seq_len - 1 - cfg.seq_len // 8, cfg.n_needles
    ).astype(int)
    # Each needle = a vector with direction = alpha·q_unit + (1-alpha)·noise_unit,
    # scaled to the same magnitude as a typical random key so it isn't drowned
    # by magnitude artifacts. q is already unit-normalized at the call site.
    typical_key_norm = np.sqrt(cfg.d_head).astype(np.float32)  # ||N(0,1)^d|| ≈ √d
    q_unit = query / np.linalg.norm(query)
    for pos in needle_positions:
        noise = rng.standard_normal(cfg.d_head).astype(np.float32)
        noise_unit = noise / np.linalg.norm(noise)
        direction = cfg.needle_similarity * q_unit + (1.0 - cfg.needle_similarity) * noise_unit
        direction = direction / np.linalg.norm(direction)
        # Match the random-key magnitude so q·needle ≈ similarity * √d
        keys[pos] = direction * typical_key_norm
    return keys, needle_positions


def dense_top_k_blocks(
    *, query: np.ndarray, keys: np.ndarray, block_size: int, k: int
) -> np.ndarray:
    """Ground truth: rank blocks by sum-of-attention-weights of their
    constituent tokens against the query.

    Approximation of "dense top-k blocks": for each block, take the max
    token-level q·k score in that block. Then top-k by max.
    """
    raw_scores = keys @ query  # [seq_len]
    seq_len = raw_scores.shape[0]
    n_blocks = (seq_len + block_size - 1) // block_size
    block_max_scores = np.zeros(n_blocks, dtype=np.float32)
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, seq_len)
        block_max_scores[b] = raw_scores[s:e].max()
    return top_k_block_indices(block_max_scores, k)


def selector_top_k_blocks(
    *,
    query: np.ndarray,
    keys: np.ndarray,
    selector_kind: str,
    cfg: BenchConfig,
    k: int,
) -> np.ndarray:
    """Selector ranking.

    selector_kind ∈ {"content", "trig", "mixture", "content+sentinel"}.
    """
    W = jl_projection_matrix(d_head=cfg.d_head)
    # Project keys → [seq_len, CONTENT_DIM]
    content_k = (keys @ W.T).astype(np.float32)
    # Build trig features per K position: relative = (q_pos - k_pos).
    # We assume q is "the next token" → q_pos = seq_len. So relative = seq_len - k_pos.
    q_pos = keys.shape[0]
    k_positions = np.arange(keys.shape[0])
    relative_positions = q_pos - k_positions  # [seq_len], all positive
    trig_k = v3_trig_features(relative_positions)  # [seq_len, TRIG_DIM]

    # Selector key embedding: [seq_len, SELECTOR_DIM]
    selector_k = np.concatenate([content_k, trig_k], axis=-1).astype(np.float32)

    # Query embedding: project query through W and assemble trig from
    # relative_position = 0 (the query attends to itself at relative 0;
    # the position signal is encoded in K side).
    content_q = jl_projection_matrix(d_head=cfg.d_head) @ query
    # For Q's trig: relative_pos = 0 → sin(0)=0, cos(0)=1 → fixed pattern.
    # But the actual signal is in the K-side trig encoding the distance
    # back. So Q's trig acts as a "match against the closest distance"
    # in the dot product. For our synthetic test we set Q's trig to all
    # ones / all zeros depending on selector_kind.
    trig_q = v3_trig_features(np.array([0])).reshape(TRIG_DIM)

    selector_q = np.concatenate([content_q, trig_q])

    # Block-pool keys, score, top-k. Sentinel variants use max-norm-row
    # pooling; mean variants use mean-pool.
    if selector_kind.endswith("+sentinel"):
        mean_blocks = block_mean_pool(selector_k, block_size=cfg.block_size)
        sent_blocks = block_max_norm_sentinel(selector_k, block_size=cfg.block_size)
        # Take max(score(mean), score(sentinel)) per block — PRD line 208 var (a).
        base_kind = selector_kind.replace("+sentinel", "")
    elif selector_kind.endswith("_max"):
        # Pure max-norm sentinel pooling, no mean.
        sent_blocks = block_max_norm_sentinel(selector_k, block_size=cfg.block_size)
        mean_blocks = None
        base_kind = selector_kind.replace("_max", "")
    else:
        mean_blocks = block_mean_pool(selector_k, block_size=cfg.block_size)
        sent_blocks = None
        base_kind = selector_kind

    lam = {"content": 0.0, "trig": 1.0, "mixture": 0.5}[base_kind]
    if sent_blocks is not None and mean_blocks is not None:
        sm = score_blocks(selector_q, mean_blocks, lambda_pos=lam)
        ss = score_blocks(selector_q, sent_blocks, lambda_pos=lam)
        scores = np.maximum(sm, ss)
    elif sent_blocks is not None:
        scores = score_blocks(selector_q, sent_blocks, lambda_pos=lam)
    else:
        scores = score_blocks(selector_q, mean_blocks, lambda_pos=lam)

    return top_k_block_indices(scores, k)


def recall_at_k(*, gt: np.ndarray, sparse: np.ndarray) -> float:
    """|gt ∩ sparse| / |gt|."""
    return len(set(gt.tolist()) & set(sparse.tolist())) / float(len(gt))


def run_bench(cfg: BenchConfig, k: int = 32) -> dict:
    rng = np.random.default_rng(cfg.seed)
    selector_kinds = [
        "content",
        "trig",
        "mixture",
        "content_max",
        "mixture_max",
        "content+sentinel",
        "mixture+sentinel",
    ]
    results = {kind: [] for kind in selector_kinds}
    dense_blocks_planted_overlap = []

    for q_i in range(cfg.n_queries):
        # Fresh random query
        q = rng.standard_normal(cfg.d_head).astype(np.float32)
        q = q / np.linalg.norm(q)
        # Synthesize keys with needles
        keys, needle_pos = synthesize_keys(cfg=cfg, query=q, rng=rng)
        # Ground truth top-k blocks
        gt_blocks = dense_top_k_blocks(
            query=q, keys=keys, block_size=cfg.block_size, k=k
        )
        # What fraction of needle positions land in dense top-k blocks?
        needle_blocks = needle_pos // cfg.block_size
        overlap = len(set(gt_blocks.tolist()) & set(needle_blocks.tolist()))
        dense_blocks_planted_overlap.append(overlap / cfg.n_needles)
        for kind in selector_kinds:
            sparse_blocks = selector_top_k_blocks(
                query=q,
                keys=keys,
                selector_kind=kind,
                cfg=cfg,
                k=k,
            )
            results[kind].append(recall_at_k(gt=gt_blocks, sparse=sparse_blocks))

    return {
        "config": asdict(cfg),
        "top_k_blocks": k,
        "dense_top_k_block_planted_needle_recall": float(
            np.mean(dense_blocks_planted_overlap)
        ),
        "recall_at_k": {
            kind: {
                "mean": float(np.mean(results[kind])),
                "std": float(np.std(results[kind])),
                "min": float(np.min(results[kind])),
                "max": float(np.max(results[kind])),
            }
            for kind in selector_kinds
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--d-head", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-needles", type=int, default=8)
    ap.add_argument("--needle-similarity", type=float, default=0.5)
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", type=str, default="-", help="JSON output path, or '-' for stdout"
    )
    args = ap.parse_args()

    cfg = BenchConfig(
        seq_len=args.seq_len,
        d_head=args.d_head,
        block_size=args.block_size,
        n_needles=args.n_needles,
        needle_similarity=args.needle_similarity,
        n_queries=args.n_queries,
        seed=args.seed,
    )
    t0 = time.time()
    result = run_bench(cfg, k=args.top_k)
    result["wall_time_s"] = round(time.time() - t0, 3)

    if args.out == "-":
        print(json.dumps(result, indent=2))
    else:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
