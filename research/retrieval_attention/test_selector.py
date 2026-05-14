"""Tests for JL random projection + V3-trig basis + block pooling +
scoring.

The big ones (Codex's flagged risk: "32 dims too lossy, normalization
wrong, top-k too small") are checked by [F-NN] empirical recall@k
microbench, not unit tests. These tests pin the mechanical correctness
that a regression would silently break:

- JL projection is deterministic across calls (same seed → same matrix)
- JL preserves dot products in expectation (within ε at large n)
- V3-trig is invertible enough to discriminate positions
- block_mean_pool / block_max_norm_sentinel handle edge cases
- top_k_block_indices returns in descending score order
"""

from __future__ import annotations

import numpy as np
import pytest

from selector import (
    CONTENT_DIM,
    SELECTOR_DIM,
    TRIG_DIM,
    block_max_norm_sentinel,
    block_mean_pool,
    jl_project,
    jl_projection_matrix,
    score_blocks,
    top_k_block_indices,
    v3_trig_features,
)


# ----------------------------------------------------- JL projection


def test_jl_projection_is_deterministic_across_calls():
    W1 = jl_projection_matrix(d_head=128)
    W2 = jl_projection_matrix(d_head=128)
    np.testing.assert_array_equal(W1, W2)


def test_jl_projection_shape():
    W = jl_projection_matrix(d_head=128)
    assert W.shape == (CONTENT_DIM, 128)
    assert W.dtype == np.float32


def test_jl_preserves_dot_products_in_expectation():
    """JL lemma: for any pair (q, k), E[<Wq, Wk>] = <q, k>.

    At dim=16 the variance is high, so we check the empirical mean over
    many independent pairs falls within tolerance of the true mean.
    """
    rng = np.random.default_rng(7)
    n = 200
    d_head = 128
    q = rng.standard_normal((n, d_head)).astype(np.float32)
    k = rng.standard_normal((n, d_head)).astype(np.float32)
    W = jl_projection_matrix(d_head=d_head)
    proj_q = jl_project(q, W)
    proj_k = jl_project(k, W)

    true_dots = (q * k).sum(axis=1)
    proj_dots = (proj_q * proj_k).sum(axis=1)

    # On average, JL preserves dots. Tolerance is generous because
    # CONTENT_DIM=16 is low — the std error scales like ||q||·||k|| / sqrt(d).
    assert abs(true_dots.mean() - proj_dots.mean()) < 1.0, (
        f"JL mean drift too high: true={true_dots.mean():.3f}, "
        f"proj={proj_dots.mean():.3f}"
    )


# ----------------------------------------------------- V3-trig basis


def test_trig_features_shape():
    positions = np.arange(100)
    feats = v3_trig_features(positions)
    assert feats.shape == (100, TRIG_DIM)
    assert feats.dtype == np.float32


def test_trig_features_at_zero_are_alternating_sin_cos():
    """sin(0)=0, cos(0)=1 → at position 0, features are [0,1,0,1,...]."""
    feats = v3_trig_features(np.array([0]))[0]
    np.testing.assert_allclose(feats[0::2], 0.0, atol=1e-6)
    np.testing.assert_allclose(feats[1::2], 1.0, atol=1e-6)


def test_trig_features_unit_norm_per_freq_pair():
    """For each freq pair, sin² + cos² = 1 always."""
    positions = np.array([0, 1, 100, 10_000, -5, -1_000])
    feats = v3_trig_features(positions)  # [6, 16]
    for f in range(TRIG_DIM // 2):
        sin_c = feats[:, 2 * f]
        cos_c = feats[:, 2 * f + 1]
        np.testing.assert_allclose(sin_c**2 + cos_c**2, 1.0, atol=1e-5)


def test_trig_features_distinguish_different_relative_positions():
    """Different relative positions produce different trig vectors —
    this is what makes the positional component useful."""
    f1 = v3_trig_features(np.array([10]))
    f2 = v3_trig_features(np.array([100]))
    # They should not be approximately equal.
    assert not np.allclose(f1, f2, atol=1e-3)


def test_trig_features_respect_rope_theta():
    """Same relative position with different base → different features.
    (Validates we can plug in Qwen2's 1_000_000 base.)"""
    pos = np.array([1000])
    f_10k = v3_trig_features(pos, base=10_000.0)
    f_1m = v3_trig_features(pos, base=1_000_000.0)
    assert not np.allclose(f_10k, f_1m, atol=1e-3)


# ----------------------------------------------------- Block pooling


def test_block_mean_pool_divides_evenly():
    features = np.arange(256, dtype=np.float32).reshape(256, 1)
    pooled = block_mean_pool(features, block_size=64)
    assert pooled.shape == (4, 1)
    # Block 0 = mean(0..63) = 31.5
    np.testing.assert_allclose(pooled[0, 0], 31.5)
    np.testing.assert_allclose(pooled[3, 0], 31.5 + 3 * 64)


def test_block_mean_pool_handles_remainder():
    """seq_len=200, block_size=64 → 4 blocks, last has 8 tokens."""
    features = np.arange(200, dtype=np.float32).reshape(200, 1)
    pooled = block_mean_pool(features, block_size=64)
    assert pooled.shape == (4, 1)
    # Last block = mean(192..199) = (192+199)/2 = 195.5
    np.testing.assert_allclose(pooled[3, 0], 195.5)


def test_block_max_norm_sentinel_picks_largest_row():
    features = np.array(
        [
            [1.0, 0.0],
            [10.0, 0.0],  # max norm in block 0
            [3.0, 4.0],  # norm = 5
        ],
        dtype=np.float32,
    )
    sentinels = block_max_norm_sentinel(features, block_size=2)
    # Block 0 = rows 0,1 → row 1 has the larger norm
    np.testing.assert_array_equal(sentinels[0], [10.0, 0.0])
    # Block 1 = row 2 alone
    np.testing.assert_array_equal(sentinels[1], [3.0, 4.0])


# ----------------------------------------------------- Scoring


def test_score_blocks_pure_content():
    q = np.zeros(SELECTOR_DIM, dtype=np.float32)
    q[:CONTENT_DIM] = 1.0  # all-ones content vector
    blocks = np.zeros((3, SELECTOR_DIM), dtype=np.float32)
    blocks[0, :CONTENT_DIM] = 1.0  # block 0 matches content
    blocks[1, CONTENT_DIM:] = 1.0  # block 1 has only trig signal
    blocks[2, :CONTENT_DIM] = -1.0  # block 2 anti-matches content

    scores = score_blocks(q, blocks, lambda_pos=0.0)  # pure content
    # block 0 highest, block 2 lowest, block 1 = 0 (q has no trig signal)
    assert scores[0] > scores[1] > scores[2]


def test_score_blocks_pure_position():
    q = np.zeros(SELECTOR_DIM, dtype=np.float32)
    q[CONTENT_DIM:] = 1.0  # all-ones trig vector
    blocks = np.zeros((3, SELECTOR_DIM), dtype=np.float32)
    blocks[0, :CONTENT_DIM] = 1.0  # block 0 content (ignored at lambda=1)
    blocks[1, CONTENT_DIM:] = 1.0  # block 1 matches trig
    blocks[2, CONTENT_DIM:] = -1.0

    scores = score_blocks(q, blocks, lambda_pos=1.0)  # pure position
    assert scores[1] > scores[0] > scores[2]
    np.testing.assert_allclose(scores[0], 0.0)  # no trig match


def test_score_blocks_recency_bias_lowers_old_blocks():
    """alpha > 0 should reduce the score of older blocks."""
    q = np.ones(SELECTOR_DIM, dtype=np.float32)
    blocks = np.zeros((5, SELECTOR_DIM), dtype=np.float32)
    no_bias = score_blocks(q, blocks, recency_alpha=0.0)
    with_bias = score_blocks(q, blocks, recency_alpha=1.0)
    # First block is furthest from recency → most penalty
    assert with_bias[0] < no_bias[0]
    # Last block has distance=0 → no penalty
    np.testing.assert_allclose(with_bias[-1], no_bias[-1])


# ----------------------------------------------------- top-k


def test_top_k_returns_descending_score_order():
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])
    top3 = top_k_block_indices(scores, k=3)
    assert list(top3) == [1, 4, 2]  # 0.9, 0.7, 0.5


def test_top_k_k_exceeds_n_returns_full_sort():
    scores = np.array([0.1, 0.9, 0.5])
    top10 = top_k_block_indices(scores, k=10)
    assert list(top10) == [1, 2, 0]


def test_top_k_k_equals_n():
    scores = np.array([0.1, 0.9, 0.5])
    top3 = top_k_block_indices(scores, k=3)
    assert list(top3) == [1, 2, 0]
