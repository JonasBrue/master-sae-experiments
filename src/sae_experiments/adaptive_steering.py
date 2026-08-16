"""Difficulty-adaptive steering intervention.

This module implements adaptive intervention strength based on estimated
question difficulty. The core insight: steering vectors only help on
questions of certain difficulty levels. Applying them everywhere causes
harm on easy questions.

Key concept:
- p_correct < 0.15: Too hard, intervention has no effect (model has no basis)
- p_correct ∈ [0.15, 0.35]: Sweet spot, full intervention strength
- p_correct ∈ [0.35, 0.50]: Transition zone, reduced strength
- p_correct > 0.50: Easy enough, intervention causes harm → skip

Reference: methodology_review.md, strategic_plan.md
"""

from __future__ import annotations

from typing import Callable, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn

from .hooks import _gather_last_tokens, _scatter_last_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive strength functions
# ═══════════════════════════════════════════════════════════════════════════════


def step_strength_fn(
    p_correct: float,
    p_min: float = 0.15,
    p_sweet: float = 0.35,
    p_max: float = 0.50,
    transition_strength: float = 0.3,
) -> float:
    """Step-function adaptive strength.
    
    Args:
        p_correct: Estimated probability of correct answer (model confidence).
        p_min: Lower bound for intervention (below this: too hard, no effect).
        p_sweet: Upper bound of sweet spot (full strength between p_min and p_sweet).
        p_max: Upper bound for intervention (above this: too easy, causes harm).
        
    Returns:
        Intervention strength multiplier (0.0 to 1.0).
        
    Example:
        >>> step_strength_fn(0.10)  # Too hard
        0.0
        >>> step_strength_fn(0.25)  # Sweet spot
        1.0
        >>> step_strength_fn(0.40)  # Transition
        0.3
        >>> step_strength_fn(0.70)  # Too easy
        0.0
    """
    if p_correct < p_min:
        return 0.0  # Too hard, intervention ineffective
    elif p_correct < p_sweet:
        return 1.0  # Sweet spot, full strength
    elif p_correct < p_max:
        return transition_strength  # Transition zone, reduced strength
    else:
        return 0.0  # Too easy, intervention causes harm


def linear_strength_fn(
    p_correct: float,
    p_min: float = 0.15,
    p_max: float = 0.50,
) -> float:
    """Linear interpolation adaptive strength.
    
    Strength ramps up from 0 at p_min to 1 at the midpoint, then down to 0 at p_max.
    
    Args:
        p_correct: Estimated probability of correct answer.
        p_min: Lower bound (strength = 0).
        p_max: Upper bound (strength = 0).
        
    Returns:
        Intervention strength multiplier.
    """
    if p_correct < p_min or p_correct > p_max:
        return 0.0
    
    midpoint = (p_min + p_max) / 2
    if p_correct < midpoint:
        # Ramping up
        return (p_correct - p_min) / (midpoint - p_min)
    else:
        # Ramping down
        return 1.0 - (p_correct - midpoint) / (p_max - midpoint)


def gaussian_strength_fn(
    p_correct: float,
    mu: float = 0.25,
    sigma: float = 0.15,
) -> float:
    """Gaussian-weighted adaptive strength.
    
    Strength follows a Gaussian curve centered at mu.
    
    Args:
        p_correct: Estimated probability of correct answer.
        mu: Center of the Gaussian (optimal difficulty).
        sigma: Standard deviation (controls width of intervention zone).
        
    Returns:
        Intervention strength multiplier.
    """
    return float(np.exp(-0.5 * ((p_correct - mu) / sigma) ** 2))


def constant_strength_fn(p_correct: float, strength: float = 1.0) -> float:
    """Constant strength (baseline, non-adaptive).
    
    This is the original behavior: always apply the same strength.
    Useful for comparison and backward compatibility.
    
    Args:
        p_correct: Ignored (for API consistency).
        strength: The constant strength to return.
        
    Returns:
        Always returns `strength`.
    """
    return strength


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive intervention hook
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptiveInterventionHook:
    """Exact intervention with adaptive strength based on question difficulty.
    
    Unlike InterventionHook which applies a fixed strength, this hook
    dynamically adjusts intervention strength based on p_correct.
    
    The p_correct value must be provided before each forward pass via
    the `set_p_correct()` method or by setting `self.p_correct` directly.
    
    Example:
        >>> hook = AdaptiveInterventionHook(sae, vec, strength_fn=step_strength_fn)
        >>> hook.set_p_correct(0.25)  # Sweet spot
        >>> hook.enabled = True
        >>> with hook_mgr.intervention(hook):
        ...     outputs = model(input_ids=...)
    """
    
    def __init__(
        self,
        sae_model: nn.Module,
        intervention_vec: torch.Tensor,
        strength_fn: Callable[[float], float] = step_strength_fn,
        base_strength: float = 1.0,
    ):
        """Initialize adaptive intervention hook.
        
        Args:
            sae_model: The SAE model for encoding/decoding.
            intervention_vec: The intervention vector in feature space.
            strength_fn: Function that maps p_correct -> strength multiplier.
            base_strength: Base strength value (multiplied by strength_fn result).
        """
        self.sae = sae_model
        self.vec = intervention_vec
        self.strength_fn = strength_fn
        self.base_strength = base_strength
        self.enabled = False
        self.p_correct: Optional[float] = None
    
    def set_p_correct(self, p_correct: float) -> None:
        """Set the estimated difficulty for the next forward pass."""
        self.p_correct = p_correct
    
    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        # Determine strength based on difficulty
        if self.p_correct is not None:
            multiplier = self.strength_fn(self.p_correct)
            effective_vec = self.vec * multiplier * self.base_strength
        else:
            # Fallback to constant strength if p_correct not set
            effective_vec = self.vec * self.base_strength
        
        hidden = output[0] if isinstance(output, tuple) else output
        seq_lengths = getattr(module, '_batch_seq_lengths', None)

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths).to(self.sae.dtype)
        else:
            last_hidden = hidden[:, -1:, :].to(self.sae.dtype)

        features = self.sae.encode(last_hidden)
        reconstructed = self.sae.decode(features)
        error = last_hidden - reconstructed

        features = features + effective_vec.unsqueeze(0).unsqueeze(0)
        features = torch.clamp(features, min=0)

        modified_last = (self.sae.decode(features) + error).to(hidden.dtype)

        if seq_lengths is not None:
            modified = _scatter_last_tokens(hidden, modified_last, seq_lengths)
        else:
            context = hidden[:, :-1, :]
            modified = torch.cat([context, modified_last], dim=1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


class AdaptiveApproximatedInterventionHook:
    """Approximated intervention with adaptive strength (no SAE at runtime).
    
    Same as AdaptiveInterventionHook but uses precomputed hidden vector
    instead of SAE encode/decode at runtime.
    """
    
    def __init__(
        self,
        precomputed_hidden_vec: torch.Tensor,
        strength_fn: Callable[[float], float] = step_strength_fn,
        base_strength: float = 1.0,
    ):
        """Initialize adaptive approximated intervention hook.
        
        Args:
            precomputed_hidden_vec: Precomputed intervention vector in hidden space.
            strength_fn: Function that maps p_correct -> strength multiplier.
            base_strength: Base strength value (multiplied by strength_fn result).
        """
        self.vec = precomputed_hidden_vec
        self.strength_fn = strength_fn
        self.base_strength = base_strength
        self.enabled = False
        self.p_correct: Optional[float] = None
    
    def set_p_correct(self, p_correct: float) -> None:
        """Set the estimated difficulty for the next forward pass."""
        self.p_correct = p_correct
    
    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        
        # Determine strength based on difficulty
        if self.p_correct is not None:
            multiplier = self.strength_fn(self.p_correct)
            effective_vec = self.vec * multiplier * self.base_strength
        else:
            effective_vec = self.vec * self.base_strength
        
        hidden = output[0] if isinstance(output, tuple) else output
        seq_lengths = getattr(module, '_batch_seq_lengths', None)

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths)
        else:
            last_hidden = hidden[:, -1:, :]

        modified_last = last_hidden + effective_vec.unsqueeze(0).unsqueeze(0).to(hidden.dtype)

        if seq_lengths is not None:
            modified = _scatter_last_tokens(hidden, modified_last, seq_lengths)
        else:
            context = hidden[:, :-1, :]
            modified = torch.cat([context, modified_last], dim=1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


# ═══════════════════════════════════════════════════════════════════════════════
# Factory functions
# ═══════════════════════════════════════════════════════════════════════════════


# Type alias for strength mode
StrengthMode = Literal["step", "linear", "gaussian", "constant"]


def get_strength_fn(
    mode: StrengthMode,
    **kwargs,
) -> Callable[[float], float]:
    """Get a strength function by name.
    
    Args:
        mode: One of "step", "linear", "gaussian", "constant".
        **kwargs: Additional arguments passed to the strength function.
        
    Returns:
        A callable that maps p_correct -> strength multiplier.
        
    Raises:
        ValueError: If mode is unknown.
    """
    if mode == "step":
        return lambda p: step_strength_fn(
            p,
            p_min=kwargs.get("p_min", 0.15),
            p_sweet=kwargs.get("p_sweet", 0.35),
            p_max=kwargs.get("p_max", 0.50),
            transition_strength=kwargs.get("transition_strength", 0.3),
        )
    elif mode == "linear":
        return lambda p: linear_strength_fn(
            p,
            p_min=kwargs.get("p_min", 0.15),
            p_max=kwargs.get("p_max", 0.50),
        )
    elif mode == "gaussian":
        return lambda p: gaussian_strength_fn(
            p,
            mu=kwargs.get("mu", 0.25),
            sigma=kwargs.get("sigma", 0.15),
        )
    elif mode == "constant":
        return lambda p: constant_strength_fn(p, strength=kwargs.get("strength", 1.0))
    else:
        raise ValueError(
            f"Unknown strength mode: {mode}. "
            f"Choose from: step, linear, gaussian, constant"
        )


def create_adaptive_hook(
    sae_model: nn.Module,
    intervention_vec: torch.Tensor,
    mode: StrengthMode = "step",
    base_strength: float = 1.0,
    **kwargs,
) -> AdaptiveInterventionHook:
    """Create an adaptive intervention hook with the specified strength mode.
    
    Args:
        sae_model: The SAE model for encoding/decoding.
        intervention_vec: The intervention vector in feature space.
        mode: Strength function mode ("step", "linear", "gaussian", "constant").
        base_strength: Base strength value.
        **kwargs: Additional arguments for the strength function.
        
    Returns:
        An AdaptiveInterventionHook instance.
    """
    strength_fn = get_strength_fn(mode, **kwargs)
    return AdaptiveInterventionHook(sae_model, intervention_vec, strength_fn, base_strength)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions for analysis
# ═══════════════════════════════════════════════════════════════════════════════


def compute_strength_distribution(
    p_correct_values: list[float],
    mode: StrengthMode = "step",
    **kwargs,
) -> Dict[str, float]:
    """Compute statistics on how strength is distributed across difficulties.
    
    Useful for analyzing the expected impact of adaptive steering on a dataset.
    
    Args:
        p_correct_values: List of p_correct values from screening.
        mode: Strength function mode.
        **kwargs: Additional arguments for the strength function.
        
    Returns:
        Dict with statistics: mean_strength, std_strength, fractions at each level.
    """
    strength_fn = get_strength_fn(mode, **kwargs)
    strengths = [strength_fn(p) for p in p_correct_values]
    
    # Count by strength level
    n_total = len(strengths)
    n_full = sum(1 for s in strengths if s >= 0.95)
    n_reduced = sum(1 for s in strengths if 0 < s < 0.95)
    n_zero = sum(1 for s in strengths if s == 0)
    
    return {
        "n_total": n_total,
        "mean_strength": float(np.mean(strengths)),
        "std_strength": float(np.std(strengths)),
        "n_full_strength": n_full,
        "n_reduced_strength": n_reduced,
        "n_zero_strength": n_zero,
        "frac_full_strength": n_full / n_total if n_total > 0 else 0,
        "frac_reduced_strength": n_reduced / n_total if n_total > 0 else 0,
        "frac_zero_strength": n_zero / n_total if n_total > 0 else 0,
    }


def suggest_adaptive_params(p_correct_values: list[float]) -> Dict[str, float]:
    """Suggest optimal adaptive parameters based on difficulty distribution.
    
    Analyzes the distribution of p_correct values and suggests parameters
    that would maximize the benefit of adaptive steering.
    
    Args:
        p_correct_values: List of p_correct values from screening.
        
    Returns:
        Dict with suggested parameters.
    """
    arr = np.array(p_correct_values)
    
    # Find the range where most "sweet spot" questions lie
    # Sweet spot is typically around 0.15-0.35
    sweet_mask = (arr >= 0.10) & (arr <= 0.50)
    if sweet_mask.sum() > 10:
        sweet_values = arr[sweet_mask]
        suggested_mu = float(sweet_values.mean())
        suggested_sigma = float(sweet_values.std())
    else:
        suggested_mu = 0.25
        suggested_sigma = 0.15
    
    # Find natural thresholds
    p10 = float(np.percentile(arr, 10))
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    
    return {
        "suggested_p_min": max(0.10, p10),
        "suggested_p_max": min(0.60, p90),
        "suggested_mu": suggested_mu,
        "suggested_sigma": max(0.08, min(0.20, suggested_sigma)),
        "median_p_correct": p50,
        "mean_p_correct": float(arr.mean()),
    }
