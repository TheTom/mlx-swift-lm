"""Clustered-K recall@k microbench — closer-to-real attention than pure Gaussian.

Real attention K has structure that i.i.d. Gaussian K doesn't:
1. Positional clustering — adjacent tokens are semantically similar
   ("topics" span multiple tokens, not single-token spikes).
2. Variable magnitude — some tokens are "louder" (higher post-RMSNorm
   K norm) than others.
3. Q-K alignment lives along low-rank directions, not uniformly.

This bench simulates all three via a topic-mixture model:
- Sample N_TOPICS topic directions in R^{d_head}.
- For each token position, draw a topic via a smoothed Markov chain
  (so adjacent positions tend to share a topic → positional clustering).
- Each K[t] = topic_direction[t] + small per-token noise, magnitude
  scaled to vary across topics.

The needle = "boost K[needle_pos] toward q, as if a relevant token
were placed at a known location."

Hypothesis (PRD line 251): block mean-pool will work BETTER on clustered
K because the block-mean accumulates the topic signal across all 64
tokens, instead of being diluted by random orthogonal vectors. We
expect selector recall to climb significantly vs the i.i.d. Gaussian
case ([F-04]).
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
class ClusterCfg:
    seq_len: int
    d_head: int
    block_size: int = 64
    n_topics: int = 32
    topic_persistence: float = 0.95  # P(stay on same topic at next pos)
    noise_scale: float = 0.3  # per-token noise added to topic direction
    needle_similarity: float = 0.8  # q-alignment of the planted needle
    n_needles: int = 8
    n_queries: int = 30
    seed: int = 42
    needle_band: str = "spread"  # spread / recent / anchored


def synthesize_clustered_keys(cfg: ClusterCfg, rng: np.random.Generator):
    """Return (keys, topic_assignment, needle_positions, q)."""
    d = cfg.d_head
    # Random unit topic directions.
    topics = rng.standard_normal((cfg.n_topics, d)).astype(np.float32)
    topics /= np.linalg.norm(topics, axis=1, keepdims=True)
    # Vary magnitudes per topic (some topics produce louder K).
    topic_mag = (rng.uniform(0.5, 2.0, size=cfg.n_topics) * np.sqrt(d)).astype(
        np.float32
    )

    # Markov topic chain.
    assignment = np.empty(cfg.seq_len, dtype=np.int32)
    assignment[0] = rng.integers(cfg.n_topics)
    for t in range(1, cfg.seq_len):
        if rng.random() < cfg.topic_persistence:
            assignment[t] = assignment[t - 1]
        else:
            assignment[t] = rng.integers(cfg.n_topics)

    # Build K.
    keys = np.empty((cfg.seq_len, d), dtype=np.float32)
    for t in range(cfg.seq_len):
        topic = assignment[t]
        direction = topics[topic]
        noise = rng.standard_normal(d).astype(np.float32) * cfg.noise_scale
        full = direction * topic_mag[topic] + noise
        keys[t] = full

    # Query and needles.
    q = rng.standard_normal(d).astype(np.float32)
    q /= np.linalg.norm(q)
    # Default needle placement: spread across context.
    # Allow override via cfg.needle_band ∈ {"spread", "recent", "anchored"}.
    band = getattr(cfg, "needle_band", "spread")
    if band == "recent":
        # All needles in the last 10% of context — trig should help here.
        start = int(cfg.seq_len * 0.9)
        needle_positions = np.linspace(
            start, cfg.seq_len - 1, cfg.n_needles
        ).astype(int)
    elif band == "anchored":
        # All needles clustered around a single random position.
        anchor = rng.integers(cfg.seq_len // 4, 3 * cfg.seq_len // 4)
        offsets = rng.integers(-64 * cfg.n_needles, 64 * cfg.n_needles, cfg.n_needles)
        needle_positions = np.clip(anchor + offsets, 0, cfg.seq_len - 1)
    else:  # "spread"
        needle_positions = np.linspace(
            cfg.seq_len // 8,
            cfg.seq_len - 1 - cfg.seq_len // 8,
            cfg.n_needles,
        ).astype(int)
    typical_mag = np.sqrt(d).astype(np.float32)
    for pos in needle_positions:
        noise = rng.standard_normal(d).astype(np.float32)
        noise_unit = noise / np.linalg.norm(noise)
        direction = cfg.needle_similarity * q + (1 - cfg.needle_similarity) * noise_unit
        direction /= np.linalg.norm(direction)
        keys[pos] = direction * typical_mag * 2.0  # 2x boost for visibility

    return keys, assignment, needle_positions, q


def dense_top_k_blocks(*, q: np.ndarray, keys: np.ndarray, block_size: int, k: int):
    raw = keys @ q
    nb = (keys.shape[0] + block_size - 1) // block_size
    bm = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s = b * block_size
        e = min(s + block_size, keys.shape[0])
        bm[b] = raw[s:e].max()
    return top_k_block_indices(bm, k)


def selector_top_k_blocks(
    *,
    q: np.ndarray,
    keys: np.ndarray,
    cfg: ClusterCfg,
    selector_kind: str,
    k: int,
):
    W = jl_projection_matrix(d_head=cfg.d_head)
    content_k = keys @ W.T
    q_pos = keys.shape[0]
    k_positions = np.arange(keys.shape[0])
    rel = q_pos - k_positions
    trig_k = v3_trig_features(rel)
    selector_k = np.concatenate([content_k, trig_k], axis=-1).astype(np.float32)
    content_q = (W @ q).astype(np.float32)
    trig_q = v3_trig_features(np.array([0])).reshape(TRIG_DIM)
    selector_q = np.concatenate([content_q, trig_q])

    if selector_kind.endswith("+sentinel"):
        mean_b = block_mean_pool(selector_k, block_size=cfg.block_size)
        sent_b = block_max_norm_sentinel(selector_k, block_size=cfg.block_size)
        base = selector_kind.replace("+sentinel", "")
        lam = {"content": 0.0, "trig": 1.0, "mixture": 0.5}[base]
        sm = score_blocks(selector_q, mean_b, lambda_pos=lam)
        ss = score_blocks(selector_q, sent_b, lambda_pos=lam)
        scores = np.maximum(sm, ss)
    else:
        mean_b = block_mean_pool(selector_k, block_size=cfg.block_size)
        lam = {"content": 0.0, "trig": 1.0, "mixture": 0.5}[selector_kind]
        scores = score_blocks(selector_q, mean_b, lambda_pos=lam)

    return top_k_block_indices(scores, k)


def run_bench(cfg: ClusterCfg, k: int = 32) -> dict:
    rng = np.random.default_rng(cfg.seed)
    selector_kinds = [
        "content",
        "trig",
        "mixture",
        "content+sentinel",
        "mixture+sentinel",
    ]
    recalls = {kind: [] for kind in selector_kinds}
    dense_finds_needle = []

    for q_i in range(cfg.n_queries):
        keys, _assign, needle_pos, q = synthesize_clustered_keys(cfg, rng)
        dense_top = dense_top_k_blocks(
            q=q, keys=keys, block_size=cfg.block_size, k=k
        )
        # Needle recall via dense.
        needle_blocks = needle_pos // cfg.block_size
        dense_finds_needle.append(
            len(set(dense_top.tolist()) & set(needle_blocks.tolist())) / cfg.n_needles
        )
        for kind in selector_kinds:
            sparse_top = selector_top_k_blocks(
                q=q, keys=keys, cfg=cfg, selector_kind=kind, k=k
            )
            r = len(set(dense_top.tolist()) & set(sparse_top.tolist())) / k
            recalls[kind].append(r)

    return {
        "config": asdict(cfg),
        "top_k_blocks": k,
        "dense_top_k_block_planted_needle_recall": float(np.mean(dense_finds_needle)),
        "recall_at_k": {
            kind: {
                "mean": float(np.mean(recalls[kind])),
                "std": float(np.std(recalls[kind])),
            }
            for kind in selector_kinds
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--d-head", type=int, default=128)
    ap.add_argument("--n-topics", type=int, default=32)
    ap.add_argument("--topic-persistence", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--n-queries", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--needle-band", choices=["spread", "recent", "anchored"], default="spread"
    )
    ap.add_argument("--out", default="-")
    args = ap.parse_args()
    cfg = ClusterCfg(
        seq_len=args.seq_len,
        d_head=args.d_head,
        n_topics=args.n_topics,
        topic_persistence=args.topic_persistence,
        n_queries=args.n_queries,
        seed=args.seed,
        needle_band=args.needle_band,
    )
    t0 = time.time()
    result = run_bench(cfg, k=args.top_k)
    result["wall_time_s"] = round(time.time() - t0, 3)
    out = json.dumps(result, indent=2)
    if args.out == "-":
        print(out)
    else:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
