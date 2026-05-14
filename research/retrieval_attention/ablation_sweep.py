"""PRD Week 2 ablation sweep on clustered-K synthetic test.

Runs the A/Bs the PRD calls for (1, 3, 5, 7) on synthetic structured K
so Tom has empirical priors before Week 1 real-K test.

A/B 1: lambda ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (content vs trig blend)
A/B 3: recency_alpha ∈ {0, 0.05, 0.1, 0.25, 0.5} (ALiBi-style bias)
A/B 5: fine_block_size ∈ {32, 64, 128} (token granularity)
A/B 7: sentinel on/off (mean-pool vs mean+sentinel max)

For each setup: clustered K at seq_len=16384, persistence=0.95,
n_topics=32, 8 needles spread across context, 30 queries.

Output: results/ablation_sweep.json with the full grid.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from clustered_recall import ClusterCfg, synthesize_clustered_keys
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


def dense_top_k_blocks(*, q, keys, block_size, k):
    raw = keys @ q
    nb = (keys.shape[0] + block_size - 1) // block_size
    bm = np.zeros(nb, dtype=np.float32)
    for b in range(nb):
        s = b * block_size
        e = min(s + block_size, keys.shape[0])
        bm[b] = raw[s:e].max()
    return top_k_block_indices(bm, k)


def selector_top_k(
    *,
    q,
    keys,
    cfg: ClusterCfg,
    lambda_pos: float,
    recency_alpha: float,
    block_size: int,
    use_sentinel: bool,
    k: int,
) -> np.ndarray:
    W = jl_projection_matrix(d_head=cfg.d_head)
    content_k = keys @ W.T
    q_pos = keys.shape[0]
    rel = q_pos - np.arange(keys.shape[0])
    trig_k = v3_trig_features(rel)
    selector_k = np.concatenate([content_k, trig_k], axis=-1).astype(np.float32)
    content_q = (W @ q).astype(np.float32)
    trig_q = v3_trig_features(np.array([0])).reshape(TRIG_DIM)
    selector_q = np.concatenate([content_q, trig_q])

    mean_b = block_mean_pool(selector_k, block_size=block_size)
    if use_sentinel:
        sent_b = block_max_norm_sentinel(selector_k, block_size=block_size)
        sm = score_blocks(
            selector_q, mean_b, lambda_pos=lambda_pos, recency_alpha=recency_alpha
        )
        ss = score_blocks(
            selector_q, sent_b, lambda_pos=lambda_pos, recency_alpha=recency_alpha
        )
        scores = np.maximum(sm, ss)
    else:
        scores = score_blocks(
            selector_q, mean_b, lambda_pos=lambda_pos, recency_alpha=recency_alpha
        )
    return top_k_block_indices(scores, k)


def measure_recall(
    cfg: ClusterCfg,
    lambda_pos: float,
    recency_alpha: float,
    block_size: int,
    use_sentinel: bool,
    k: int,
    n_queries: int,
):
    rng = np.random.default_rng(cfg.seed)
    recalls = []
    for _ in range(n_queries):
        keys, _assign, needle_pos, q = synthesize_clustered_keys(cfg, rng)
        # Adjust block_size in dense oracle for apples-to-apples — both
        # oracle and selector use the same block_size.
        dense = dense_top_k_blocks(
            q=q, keys=keys, block_size=block_size, k=k
        )
        sparse = selector_top_k(
            q=q,
            keys=keys,
            cfg=cfg,
            lambda_pos=lambda_pos,
            recency_alpha=recency_alpha,
            block_size=block_size,
            use_sentinel=use_sentinel,
            k=k,
        )
        r = len(set(dense.tolist()) & set(sparse.tolist())) / k
        recalls.append(r)
    return float(np.mean(recalls)), float(np.std(recalls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=16384)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--n-queries", type=int, default=20)
    ap.add_argument("--needle-band", choices=["spread", "recent", "anchored"],
                    default="spread")
    ap.add_argument("--out", default="results/ablation_sweep.json")
    args = ap.parse_args()

    base_cfg = ClusterCfg(
        seq_len=args.seq_len,
        d_head=128,
        topic_persistence=0.95,
        n_topics=32,
        n_queries=args.n_queries,
        seed=42,
        needle_band=args.needle_band,
    )

    grid = []
    t0 = time.time()

    # A/B 1: lambda sweep, recency=0, block_size=64, sentinel off & on
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for sent in [False, True]:
            mean, std = measure_recall(
                base_cfg, lambda_pos=lam, recency_alpha=0.0,
                block_size=64, use_sentinel=sent, k=args.top_k,
                n_queries=args.n_queries,
            )
            grid.append({
                "ab": "A/B-1 lambda",
                "lambda": lam,
                "recency_alpha": 0.0,
                "block_size": 64,
                "sentinel": sent,
                "recall_mean": mean,
                "recall_std": std,
            })

    # A/B 3: recency alpha sweep, lambda=0.0, sentinel on (winning prior)
    for alpha in [0.0, 0.05, 0.1, 0.25, 0.5]:
        mean, std = measure_recall(
            base_cfg, lambda_pos=0.0, recency_alpha=alpha,
            block_size=64, use_sentinel=True, k=args.top_k,
            n_queries=args.n_queries,
        )
        grid.append({
            "ab": "A/B-3 recency",
            "lambda": 0.0,
            "recency_alpha": alpha,
            "block_size": 64,
            "sentinel": True,
            "recall_mean": mean,
            "recall_std": std,
        })

    # A/B 5: block_size sweep, sentinel on
    for bs in [32, 64, 128]:
        mean, std = measure_recall(
            base_cfg, lambda_pos=0.0, recency_alpha=0.0,
            block_size=bs, use_sentinel=True, k=args.top_k,
            n_queries=args.n_queries,
        )
        grid.append({
            "ab": "A/B-5 block_size",
            "lambda": 0.0,
            "recency_alpha": 0.0,
            "block_size": bs,
            "sentinel": True,
            "recall_mean": mean,
            "recall_std": std,
        })

    wall_time = round(time.time() - t0, 2)

    out_dict = {
        "config": {
            "seq_len": args.seq_len,
            "n_topics": 32,
            "topic_persistence": 0.95,
            "needle_band": args.needle_band,
            "n_queries": args.n_queries,
            "top_k": args.top_k,
        },
        "wall_time_s": wall_time,
        "grid": grid,
    }
    with open(args.out, "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"wrote {args.out}")

    # Pretty-print table
    print()
    print(f"{'A/B':18s} {'λ':>4s} {'α':>5s} {'block':>6s} {'sent':>5s} {'recall':>10s}")
    print("-" * 56)
    for row in grid:
        print(
            f"{row['ab']:18s} "
            f"{row['lambda']:4.2f} {row['recency_alpha']:5.2f} "
            f"{row['block_size']:6d} {str(row['sentinel'])[:5]:>5s} "
            f"{row['recall_mean']*100:6.1f}%±{row['recall_std']*100:4.1f}"
        )


if __name__ == "__main__":
    main()
