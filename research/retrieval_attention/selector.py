"""V3-trig positional basis + JL random-projection content basis.

Per Open Sparse Stack PRD v5 lines 156-167:

- Selector = `[content_proj | trig_proj]` concatenated, total dim 32
  - `content_proj`: 16-dim JL random projection (deterministic seed,
    same matrix for Q and K to preserve dot-product semantics)
  - `trig_proj`: 16-dim V3-trig RoPE-derived basis (8 freq × sin+cos).
    Positional signal from RELATIVE position
    `(query_pos - block_center_pos)`, not absolute K position
- Mixture weight `lambda` blends content vs position:
  `score = (1-lambda) * content_score + lambda * trig_score + recency_bias`

This is the offline calibration / sanity prototype. Final Swift port
will reuse the trig coefficients but execute in MLX.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------- Constants from PRD

CONTENT_DIM = 16  # JL random projection output dim
TRIG_DIM = 16  # V3-trig basis output dim (8 freq × 2 phases)
SELECTOR_DIM = CONTENT_DIM + TRIG_DIM  # 32 — Decision 7

# Deterministic seed for the JL projection (PRD line 159: "fixed
# deterministic seed, identical for prefill, decode, and across chunks").
JL_SEED = 0x4F50_4E53  # "OPNS" (open sparse), 1330926915


# ----------------------------------------------------- JL random projection


def jl_projection_matrix(d_head: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Build a deterministic JL projection from d_head → CONTENT_DIM.

    Following the standard JL construction: entries i.i.d. N(0, 1/CONTENT_DIM).
    The 1/CONTENT_DIM scaling preserves expected norms (E[||Wx||²] = ||x||²).

    Same matrix used for Q and K so that
        (W·q) · (W·k) = q · (Wᵀ·W) · k ≈ q · k  (in expectation)
    which is the JL lemma's pairwise-distance preservation.

    Args:
        d_head: head dim of the underlying model (Q and K dim).
        rng: optional seeded numpy Generator; if None, uses JL_SEED.

    Returns:
        Array of shape [CONTENT_DIM, d_head], dtype fp32.
    """
    if rng is None:
        rng = np.random.default_rng(JL_SEED)
    W = rng.normal(loc=0.0, scale=1.0 / np.sqrt(CONTENT_DIM), size=(CONTENT_DIM, d_head))
    return W.astype(np.float32)


def jl_project(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Project x [..., d_head] → [..., CONTENT_DIM] via W."""
    return x @ W.T


# ----------------------------------------------------- V3-trig basis


def v3_trig_features(relative_positions: np.ndarray, base: float = 10_000.0) -> np.ndarray:
    """Compute 16-dim V3-trig features from relative positions.

    Mirrors RoPE's frequency structure:
        freq_i = 1 / base ** (2*i / TRIG_DIM)  for i in [0..TRIG_DIM/2)
        feature_2i   = sin(pos * freq_i)
        feature_2i+1 = cos(pos * freq_i)

    Args:
        relative_positions: array of shape [...]. Values may be negative
            (q_pos − k_pos can be negative for q_pos < k_pos — RA only
            attends to positions ≤ q_pos so this won't happen in practice,
            but we don't assume positivity here).
        base: RoPE theta. PRD line 234 default 10_000; Qwen2 uses 1_000_000
            (see Qwen2Configuration.ropeTheta) — the trig basis must use
            the SAME theta as the model's RoPE for the positional manifold
            to align (PRD line 260 — "consume rope_scaling factors").

    Returns:
        Array of shape [..., TRIG_DIM], dtype fp32.
    """
    # half-dim frequencies: 8 of them for TRIG_DIM=16.
    half = TRIG_DIM // 2
    freqs = base ** (-np.arange(half, dtype=np.float32) * 2.0 / TRIG_DIM)  # [8]
    # broadcast: relative_positions [...] × freqs [8] → [..., 8]
    angles = relative_positions[..., None].astype(np.float32) * freqs  # [..., 8]
    sin_part = np.sin(angles)
    cos_part = np.cos(angles)
    # Interleave so feature_2i = sin, feature_2i+1 = cos.
    out = np.empty(relative_positions.shape + (TRIG_DIM,), dtype=np.float32)
    out[..., 0::2] = sin_part
    out[..., 1::2] = cos_part
    return out


# ----------------------------------------------------- Block centroid + sentinel


def block_mean_pool(features: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Mean-pool a `[seq_len, ...]` feature array into blocks.

    Args:
        features: shape [seq_len, ...] (typically [seq_len, SELECTOR_DIM]).
        block_size: PRD default 64.

    Returns:
        Shape [n_blocks, ...] where n_blocks = ceil(seq_len / block_size).
        Last block is mean over <= block_size tokens.
    """
    seq_len = features.shape[0]
    n_blocks = (seq_len + block_size - 1) // block_size
    pooled = np.zeros((n_blocks,) + features.shape[1:], dtype=features.dtype)
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, seq_len)
        pooled[b] = features[s:e].mean(axis=0)
    return pooled


def block_max_norm_sentinel(features: np.ndarray, block_size: int = 64) -> np.ndarray:
    """Pick the highest-L2-norm row per block (PRD line 207 sentinel).

    For each block, return the row whose ||features[t]||₂ is largest.
    Outlier preservation that hedges against mean-pool smoothing away
    a needle (PRD A/B 7).
    """
    seq_len = features.shape[0]
    n_blocks = (seq_len + block_size - 1) // block_size
    sentinels = np.zeros((n_blocks,) + features.shape[1:], dtype=features.dtype)
    for b in range(n_blocks):
        s = b * block_size
        e = min(s + block_size, seq_len)
        norms = np.linalg.norm(features[s:e], axis=-1)
        sentinels[b] = features[s + int(np.argmax(norms))]
    return sentinels


# ----------------------------------------------------- Scoring


def score_blocks(
    q_features: np.ndarray,
    block_features: np.ndarray,
    lambda_pos: float = 0.5,
    recency_alpha: float = 0.0,
) -> np.ndarray:
    """Score blocks for one query.

    Args:
        q_features: [SELECTOR_DIM]. The query projected via
            [content_proj | trig_proj] same as keys.
        block_features: [n_blocks, SELECTOR_DIM]. Block-pooled.
        lambda_pos: 0=pure content, 1=pure position, 0.5=balanced.
        recency_alpha: ALiBi-style recency bias coefficient (PRD line 166).

    Returns:
        Per-block scalar scores, shape [n_blocks].
    """
    # Split into content + trig halves
    q_content, q_trig = q_features[:CONTENT_DIM], q_features[CONTENT_DIM:]
    b_content, b_trig = (
        block_features[:, :CONTENT_DIM],
        block_features[:, CONTENT_DIM:],
    )
    content_score = b_content @ q_content
    trig_score = b_trig @ q_trig
    score = (1.0 - lambda_pos) * content_score + lambda_pos * trig_score
    if recency_alpha != 0.0:
        n_blocks = block_features.shape[0]
        block_idx = np.arange(n_blocks)
        # PRD line 166: `score += -alpha * log(distance_in_blocks + 1)`.
        # Distance from the assumed most-recent block.
        distance = (n_blocks - 1) - block_idx
        score = score - recency_alpha * np.log(distance + 1.0)
    return score


def top_k_block_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scoring blocks (descending)."""
    if k >= scores.shape[0]:
        return np.argsort(-scores)
    # Use argpartition for O(n) top-k, then sort the top-k partition.
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]
