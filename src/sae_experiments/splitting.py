"""Matched-Pair Randomization splitting for balanced data partitioning."""

from __future__ import annotations

import random
from math import gcd
from functools import reduce
from typing import Dict, List


def matched_pair_split(
    data: List[Dict],
    ratios: List[int],
    key: str = "p_correct",
    seed: int = 42,
) -> List[List[Dict]]:
    """Split data into groups with matched distributions of a continuous variable.

    Uses Matched-Pair Randomization (standard in clinical trials): sorts data
    by ``key``, forms blocks whose size equals the sum of the normalised ratios,
    then randomly assigns elements within each block according to the ratios.

    This guarantees that every output set has a near-identical distribution of
    ``key`` values, without requiring arbitrary bin counts.

    Args:
        data: List of dicts, each containing a field named ``key``.
        ratios: Integer ratios for each output set (e.g. ``[50, 25, 25]``
            or equivalently ``[2, 1, 1]``).  At least two values required.
        key: Dict key whose value is used for sorting.
        seed: Random seed for reproducibility.

    Returns:
        A list of lists, one per element in ``ratios``.

    Examples::

        train, val, test = matched_pair_split(data, [50, 25, 25])
        grid_eval, holdout = matched_pair_split(data, [50, 50])
        a, b, c, d = matched_pair_split(data, [1, 1, 1, 1])
    """
    if len(ratios) < 2:
        raise ValueError("Need at least 2 ratios")
    if any(r <= 0 for r in ratios):
        raise ValueError("All ratios must be positive integers")
    if len(data) == 0:
        return [[] for _ in ratios]

    # Normalise ratios by GCD → minimal block size
    divisor = reduce(gcd, ratios)
    normed = [r // divisor for r in ratios]
    block_size = sum(normed)

    # Build assignment pattern: e.g. normed=[2,1,1] → [0,0,1,2]
    assignment = []
    for set_idx, count in enumerate(normed):
        assignment.extend([set_idx] * count)

    # Sort by key
    sorted_data = sorted(data, key=lambda x: x[key])

    rng = random.Random(seed)
    n_sets = len(ratios)
    sets: List[List[Dict]] = [[] for _ in range(n_sets)]

    # Process full blocks
    n_full_blocks = len(sorted_data) // block_size
    for b in range(n_full_blocks):
        block = sorted_data[b * block_size : (b + 1) * block_size]
        shuffled_assignment = assignment.copy()
        rng.shuffle(shuffled_assignment)
        for item, set_idx in zip(block, shuffled_assignment):
            sets[set_idx].append(item)

    # Handle remaining elements (< block_size)
    remainder = sorted_data[n_full_blocks * block_size :]
    if remainder:
        # Distribute remainder proportionally by shuffling and round-robin
        # weighted by the original ratios
        rng.shuffle(remainder)
        # Build a weighted target list for the remainder
        total_ratio = sum(ratios)
        for i, item in enumerate(remainder):
            # Pick set with largest deficit (ratio-weighted)
            deficits = [
                ratios[s] / total_ratio - len(sets[s]) / max(len(data), 1)
                for s in range(n_sets)
            ]
            target = max(range(n_sets), key=lambda s: deficits[s])
            sets[target].append(item)

    return sets
