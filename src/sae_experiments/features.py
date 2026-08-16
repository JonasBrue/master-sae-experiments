"""Feature tracking for freq_diff analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch


class FeatureTracker:
    """Tracks SAE feature activations by success/failure for freq_diff computation."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.success_acts: List[Dict[int, float]] = []
        self.failure_acts: List[Dict[int, float]] = []

    def update(
        self,
        active_indices: torch.Tensor,
        active_values: torch.Tensor,
        correct: bool,
    ) -> None:
        act_dict = {idx.item(): val.item() for idx, val in zip(active_indices, active_values)}
        if correct:
            self.success_acts.append(act_dict)
        else:
            self.failure_acts.append(act_dict)

    def analyze(self, min_occurrences: int = 5) -> Dict[int, Dict]:
        n_s, n_f = len(self.success_acts), len(self.failure_acts)
        if n_s == 0 or n_f == 0:
            return {}

        s_count: Dict[int, int] = defaultdict(int)
        f_count: Dict[int, int] = defaultdict(int)

        for act in self.success_acts:
            for idx in act:
                s_count[idx] += 1
        for act in self.failure_acts:
            for idx in act:
                f_count[idx] += 1

        all_features = set(s_count.keys()) | set(f_count.keys())
        scores = {}

        for idx in all_features:
            total = s_count[idx] + f_count[idx]
            if total < min_occurrences:
                continue

            s_rate = s_count[idx] / n_s
            f_rate = f_count[idx] / n_f

            scores[idx] = {
                "freq_diff": s_rate - f_rate,
                "s_count": s_count[idx],
                "f_count": f_count[idx],
                "s_rate": s_rate,
                "f_rate": f_rate,
            }

        return scores

    def get_top_features(
        self,
        n_positive: int,
        n_negative: int,
        min_occurrences: int,
    ) -> Tuple[List, List]:
        """Return top-N positive and negative features by freq_diff."""
        analysis = self.analyze(min_occurrences)
        if not analysis:
            return [], []

        pos = sorted(
            [(idx, data) for idx, data in analysis.items() if data["freq_diff"] > 0],
            key=lambda x: x[1]["freq_diff"],
            reverse=True,
        )[:n_positive]

        neg = sorted(
            [(idx, data) for idx, data in analysis.items() if data["freq_diff"] < 0],
            key=lambda x: x[1]["freq_diff"],
        )[:n_negative]

        return pos, neg

    def reset(self) -> None:
        self.success_acts = []
        self.failure_acts = []
