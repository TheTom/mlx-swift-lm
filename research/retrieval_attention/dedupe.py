"""Dedupe gather indices from the four RetrievalAttention regions.

Per Open Sparse Stack PRD v5:

- Static initial: first 128 tokens
- Sliding window: last 2048 tokens
- Fine retrieved: 32 blocks of 64 tokens
- Coarse rescue: 2 blocks of 1024 tokens

These can overlap at small contexts (static + sliding touch) and at any
context (fine top-k may pick blocks inside sliding; coarse may overlap
fine). Without dedupe, a token attended twice gets its softmax weight
effectively doubled — wrong attention semantics.

This is the pure-Python reference. The Swift port lives at
`Libraries/MLXLMCommon/RetrievalAttention.swift` once we write it.
"""

from __future__ import annotations


def build_gather_indices(
    *,
    seq_len: int,
    static_init: int = 128,
    sliding_window: int = 2048,
    fine_block_size: int = 64,
    fine_block_starts: list[int] | None = None,
    coarse_block_size: int = 1024,
    coarse_block_starts: list[int] | None = None,
) -> list[int]:
    """Return a sorted, deduplicated list of token positions to gather.

    Args:
        seq_len: current cache length (number of populated KV positions).
        static_init: first N tokens always attended.
        sliding_window: last N tokens always attended.
        fine_block_size: size of each fine retrieved block (default 64).
        fine_block_starts: token positions where each fine block begins.
            Pass `[]` for static-window-only mode (Week 1 Day 1).
        coarse_block_size: size of each coarse rescue block (default 1024).
        coarse_block_starts: token positions for each coarse block.

    Returns:
        Sorted list of unique token positions, all within [0, seq_len).
        Guarantees: result is sorted ascending, contains no duplicates,
        every position is < seq_len.
    """
    fine_block_starts = fine_block_starts or []
    coarse_block_starts = coarse_block_starts or []

    indices: set[int] = set()

    # Static initial: positions 0 .. min(static_init, seq_len) - 1.
    indices.update(range(min(static_init, seq_len)))

    # Sliding window: last `sliding_window` tokens.
    sliding_start = max(0, seq_len - sliding_window)
    indices.update(range(sliding_start, seq_len))

    # Fine retrieved blocks.
    for block_start in fine_block_starts:
        if block_start < 0 or block_start >= seq_len:
            raise ValueError(
                f"fine block start {block_start} out of range [0, {seq_len})"
            )
        block_end = min(block_start + fine_block_size, seq_len)
        indices.update(range(block_start, block_end))

    # Coarse rescue blocks.
    for block_start in coarse_block_starts:
        if block_start < 0 or block_start >= seq_len:
            raise ValueError(
                f"coarse block start {block_start} out of range [0, {seq_len})"
            )
        block_end = min(block_start + coarse_block_size, seq_len)
        indices.update(range(block_start, block_end))

    return sorted(indices)


def expected_budget(
    *,
    static_init: int = 128,
    sliding_window: int = 2048,
    n_fine_blocks: int = 32,
    fine_block_size: int = 64,
    n_coarse_blocks: int = 2,
    coarse_block_size: int = 1024,
) -> int:
    """The PRD's quoted pre-dedupe budget: 128 + 2048 + 32*64 + 2*1024 = 6272."""
    return (
        static_init
        + sliding_window
        + n_fine_blocks * fine_block_size
        + n_coarse_blocks * coarse_block_size
    )
