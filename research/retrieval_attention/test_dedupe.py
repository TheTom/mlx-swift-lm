"""Unit tests for the gather-indices dedupe step (PRD v5).

Validates:
- No duplicate positions in result.
- Sorted ascending.
- All positions < seq_len.
- Pre-dedupe budget = 6272 (PRD line 127).
- Overlapping regions (static + sliding at small context, fine inside
  sliding, coarse inside fine) collapse correctly.
- Static-window-only mode (Week 1 Day 1 baseline) works.
"""

from __future__ import annotations

import pytest

from dedupe import build_gather_indices, expected_budget


# ----------------------------------------------------- Pre-dedupe budget


def test_prd_budget_is_6272():
    """PRD line 127: 128 + 2048 + 2048 + 2048 = 6272."""
    assert expected_budget() == 6272


# ----------------------------------------------------- Invariants


def test_result_is_sorted_ascending():
    idx = build_gather_indices(
        seq_len=100_000,
        fine_block_starts=[50_000, 30_000, 10_000],  # out of order
        coarse_block_starts=[70_000, 20_000],  # out of order
    )
    assert idx == sorted(idx)


def test_result_has_no_duplicates():
    idx = build_gather_indices(
        seq_len=100_000,
        # Fine block at 90_000 overlaps with sliding window
        # (sliding starts at 100_000 - 2048 = 97_952)
        fine_block_starts=[95_000, 97_900, 98_000],
        coarse_block_starts=[97_000],  # spans 97_000..98_023 → overlaps fine + sliding
    )
    assert len(idx) == len(set(idx))


def test_all_positions_under_seq_len():
    idx = build_gather_indices(
        seq_len=200,
        # Fine block at 150 would naively span 150..213, but seq_len=200.
        fine_block_starts=[150],
        coarse_block_starts=[100],
    )
    assert all(0 <= p < 200 for p in idx)


# ----------------------------------------------------- Static-window-only mode


def test_static_window_only_no_blocks():
    """Week 1 Day 1 baseline: 128 init + 2048 sliding, drop everything else."""
    idx = build_gather_indices(
        seq_len=10_000,
        fine_block_starts=[],
        coarse_block_starts=[],
    )
    assert idx[:128] == list(range(128))  # static head
    # Sliding starts at 10000 - 2048 = 7952
    assert 7952 in idx
    assert 9999 in idx
    # Gap in the middle.
    assert 4000 not in idx


def test_static_and_sliding_touch_at_short_context():
    """At seq_len ≤ static + sliding, regions touch / overlap.

    seq_len=2000 → static [0..127] ∪ sliding [-2048..1999] → which is
    actually [0..1999] since sliding floors at 0. Result = [0..1999].
    """
    idx = build_gather_indices(
        seq_len=2000, fine_block_starts=[], coarse_block_starts=[]
    )
    assert idx == list(range(2000))


# ----------------------------------------------------- Overlap collapse


def test_fine_overlapping_sliding_dedupes():
    """At seq_len=10K, sliding is [7952..9999]. A fine block at 9000
    overlaps. Token 9000 must appear once."""
    idx = build_gather_indices(
        seq_len=10_000,
        fine_block_starts=[9000],
        coarse_block_starts=[],
    )
    assert idx.count(9000) == 1
    # Length: 128 (static) + 2048 (sliding 7952..9999) = 2176.
    # Fine block 9000..9063 is fully inside sliding → no growth.
    assert len(idx) == 2176


def test_coarse_overlapping_fine_dedupes():
    """Fine block at 5000 and coarse block at 4500 overlap on [5000..5063]."""
    idx = build_gather_indices(
        seq_len=20_000,
        fine_block_starts=[5000],
        coarse_block_starts=[4500],  # spans 4500..5523
    )
    expected_unique = (
        128  # static
        + 2048  # sliding [17952..19999]
        + 1024  # coarse 4500..5523 (entirely outside static + sliding)
        # fine 5000..5063 is fully inside coarse → no additional unique tokens
    )
    assert len(idx) == expected_unique


def test_two_fine_blocks_adjacent_dedupe():
    """Fine block at 5000 and fine block at 5050 overlap on [5050..5063]."""
    idx = build_gather_indices(
        seq_len=20_000,
        fine_block_starts=[5000, 5050],
        coarse_block_starts=[],
    )
    # Static 128 + sliding 2048 = 2176 in their regions.
    # Fine union: [5000..5113] = 114 tokens (5000..5063 + 5050..5113 → 5000..5113)
    assert len(idx) == 2176 + 114


# ----------------------------------------------------- Out-of-range guards


def test_negative_fine_start_raises():
    with pytest.raises(ValueError):
        build_gather_indices(
            seq_len=10_000,
            fine_block_starts=[-1],
        )


def test_fine_start_past_seq_len_raises():
    with pytest.raises(ValueError):
        build_gather_indices(
            seq_len=10_000,
            fine_block_starts=[10_000],
        )


# ----------------------------------------------------- PRD example


def test_prd_example_full_budget_at_1m():
    """1M context, full PRD config: 32 fine blocks + 2 coarse blocks,
    none overlapping. Result should be exactly 6272 unique tokens."""
    seq_len = 1_000_000
    sliding_start = seq_len - 2048  # 997_952
    # Place 32 fine blocks at well-spaced positions, none inside sliding.
    fine_starts = [10_000 + i * 20_000 for i in range(32)]
    assert max(fine_starts) + 64 < sliding_start  # sanity: no sliding overlap
    # 2 coarse blocks, none overlapping fine or sliding.
    coarse_starts = [800_000, 900_000]
    # Ensure they don't overlap each other or fine blocks
    for fs in fine_starts:
        for cs in coarse_starts:
            assert fs + 64 <= cs or cs + 1024 <= fs

    idx = build_gather_indices(
        seq_len=seq_len,
        fine_block_starts=fine_starts,
        coarse_block_starts=coarse_starts,
    )
    assert len(idx) == 128 + 2048 + 32 * 64 + 2 * 1024
    assert len(idx) == 6272
