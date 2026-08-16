"""Intervention vector construction."""

from __future__ import annotations

import torch
import torch.nn as nn
import pandas as pd


def build_intervention_vector(
    pos_features: pd.DataFrame,
    neg_features: pd.DataFrame,
    top_k_pos: int,
    top_k_neg: int,
    strength: float,
    ranking_criterion: str,
    weighting_mode: str,
    *,
    d_hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an intervention vector in feature space.

    Args:
        pos_features: DataFrame of positive causal results (must have 'feature_id', 'delta', 'freq_diff').
        neg_features: DataFrame of negative causal results.
        top_k_pos: Number of top positive features to include.
        top_k_neg: Number of top negative features to include.
        strength: Intervention strength (lambda).
        ranking_criterion: 'causal_delta' or 'freq_diff'.
        weighting_mode: 'uniform' or 'proportional'.
        d_hidden: SAE hidden dimension.
        device: Torch device.
        dtype: Torch dtype.
    """
    vec = torch.zeros(d_hidden, device=device, dtype=dtype)

    if ranking_criterion == "causal_delta":
        score_col = "delta"
    elif ranking_criterion == "freq_diff":
        score_col = "freq_diff"
    else:
        raise ValueError(f"Unknown ranking_criterion: {ranking_criterion}")

    # Positive features (strengthen)
    if len(pos_features) > 0 and top_k_pos > 0:
        pos_to_use = pos_features[pos_features[score_col] > 0]

        if len(pos_to_use) > 0:
            k = min(top_k_pos, len(pos_to_use))
            pos_sorted = pos_to_use.nlargest(k, score_col)

            if weighting_mode == "uniform":
                for _, row in pos_sorted.iterrows():
                    vec[int(row["feature_id"])] = strength
            elif weighting_mode == "proportional":
                scores = pos_sorted[score_col].values
                max_score = scores.max()
                if max_score > 0:
                    for _, row in pos_sorted.iterrows():
                        normalized_score = row[score_col] / max_score
                        vec[int(row["feature_id"])] = strength * max(normalized_score, 0.1)
                else:
                    for _, row in pos_sorted.iterrows():
                        vec[int(row["feature_id"])] = strength

    # Negative features (dampen)
    if len(neg_features) > 0 and top_k_neg > 0:
        if ranking_criterion == "causal_delta":
            neg_to_use = neg_features[neg_features["delta"] > 0]
            if len(neg_to_use) > 0:
                k = min(top_k_neg, len(neg_to_use))
                neg_sorted = neg_to_use.nlargest(k, "delta")
            else:
                neg_sorted = pd.DataFrame()
        else:
            neg_to_use = neg_features[neg_features["freq_diff"] < 0]
            if len(neg_to_use) > 0:
                k = min(top_k_neg, len(neg_to_use))
                neg_sorted = neg_to_use.nsmallest(k, "freq_diff")
            else:
                neg_sorted = pd.DataFrame()

        if len(neg_sorted) > 0:
            if weighting_mode == "uniform":
                for _, row in neg_sorted.iterrows():
                    vec[int(row["feature_id"])] = -strength
            elif weighting_mode == "proportional":
                if ranking_criterion == "causal_delta":
                    scores = neg_sorted["delta"].values
                    max_score = scores.max() if len(scores) > 0 and scores.max() > 0 else 1.0
                    for _, row in neg_sorted.iterrows():
                        normalized_score = row["delta"] / max_score if max_score > 0 else 1.0
                        vec[int(row["feature_id"])] = -strength * max(normalized_score, 0.1)
                else:
                    scores = neg_sorted["freq_diff"].abs().values
                    max_score = scores.max() if len(scores) > 0 else 1.0
                    for _, row in neg_sorted.iterrows():
                        normalized_score = abs(row["freq_diff"]) / max_score if max_score > 0 else 1.0
                        vec[int(row["feature_id"])] = -strength * max(normalized_score, 0.1)

    return vec


def compute_approximated_hidden_vector(
    intervention_vec: torch.Tensor, sae: nn.Module
) -> torch.Tensor:
    """Precompute the hidden-space displacement of the intervention (SAE-free approximation).

    The SAE decoder is affine, ``decode(x) = x @ W_dec + b_dec``, so ``decode(s)`` on its own
    carries the decoder bias ``b_dec``. The exact intervention (``InterventionHook``) cancels
    that bias through the reconstruction-residual restoration,
    ``decode(f + s) + (h - decode(f)) = h + W_dec @ (relu(f + s) - f)``, so its displacement is
    ``W_dec @ s`` (bias-free) when the clamp is inactive. To make this approximation match the
    exact operator, the bias must be removed: ``decode(s) - decode(0) = W_dec @ s``.

    Omitting the ``- decode(0)`` term injects ``b_dec`` into every steered hidden state, which is
    negligible where ``b_dec`` is small but collapses accuracy where it is large (observed on
    some BatchTopK/JumpReLU layers, e.g. EXP-330: exact +0.8% vs biased approx -31.8%).
    """
    with torch.no_grad():
        zeros = torch.zeros_like(intervention_vec)
        return (
            sae.decode(intervention_vec.unsqueeze(0)) - sae.decode(zeros.unsqueeze(0))
        ).squeeze(0)
