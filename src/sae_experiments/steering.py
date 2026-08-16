"""Difficulty-adaptive steering with hybrid estimation and safety mechanisms.

This module implements the enhanced DifficultyAdaptiveSteering class that combines:
1. Hybrid Difficulty Estimator (40% Confidence + 30% Feature Diversity + 30% Critic Value)
2. Adaptive Strength based on difficulty thresholds
3. Confidence Gating (skip steering on high-confidence easy questions)
4. Over-Steering Prevention via repetition detection

Reference: difficulty_estimation_research.md, analyst_notes_2026-02-24.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn

from .hooks import _gather_last_logits, _gather_last_tokens, _scatter_last_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes for configuration
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DifficultyThresholds:
    """Thresholds for adaptive strength mapping."""
    
    very_hard: float = 0.20   # p < 0.20: strength 1.5
    hard: float = 0.35        # p < 0.35: strength 1.0
    medium: float = 0.50      # p < 0.50: strength 0.3
    # p >= 0.50: strength 0.0
    

@dataclass
class StrengthValues:
    """Strength values for each difficulty level."""
    
    very_hard: float = 1.5   # Maximum steering
    hard: float = 1.0        # Normal steering
    medium: float = 0.3      # Light steering
    easy: float = 0.0        # No steering


@dataclass
class HybridWeights:
    """Weights for hybrid difficulty estimator."""
    
    confidence: float = 0.4       # Weight for confidence component
    feature_diversity: float = 0.3  # Weight for feature diversity
    critic_value: float = 0.3     # Weight for critic value


@dataclass
class ConfidenceGatingConfig:
    """Configuration for confidence gating."""
    
    enabled: bool = True
    confidence_threshold: float = 0.7   # Above this confidence
    difficulty_threshold: float = 0.3   # Below this difficulty
    # When both conditions met: skip steering


@dataclass
class RepetitionDetectionConfig:
    """Configuration for repetition detection (over-steering prevention)."""
    
    enabled: bool = True
    n_gram_size: int = 3
    max_repetition_ratio: float = 0.3
    decay_factor: float = 0.5
    max_retries: int = 3


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid Difficulty Estimator
# ═══════════════════════════════════════════════════════════════════════════════


class HybridDifficultyEstimator:
    """Estimates question difficulty using multiple signals.
    
    Combines:
    - Model confidence (softmax probability of predicted answer)
    - Feature diversity (number of active SAE features)
    - Critic value (optional, from trained critic network)
    
    The intuition:
    - Low confidence = high difficulty (model is uncertain)
    - High feature diversity = high difficulty (complex task)
    - Low critic value = high difficulty (steering unlikely to help)
    """
    
    def __init__(
        self,
        weights: Optional[HybridWeights] = None,
        feature_diversity_scale: float = 100.0,  # Scale factor for diversity
    ):
        self.weights = weights or HybridWeights()
        self.feature_diversity_scale = feature_diversity_scale
        
    def estimate_confidence(
        self,
        logits: torch.Tensor,
        answer_indices: Optional[torch.Tensor] = None,
    ) -> float:
        """Estimate confidence from model logits.
        
        Args:
            logits: Logits tensor [batch, seq, vocab] or [batch, vocab]
            answer_indices: Optional indices of answer choices (A, B, C, D)
            
        Returns:
            Confidence score (probability of most likely answer).
        """
        # Get last token logits
        if logits.dim() == 3:
            sl = getattr(self, '_batch_seq_lengths', None)
            logits = _gather_last_logits(logits, sl) if sl is not None else logits[:, -1, :]
        
        with torch.no_grad():
            if answer_indices is not None:
                # Multiple choice: get probability of answer choices
                choice_logits = logits[:, answer_indices]  # [batch, n_choices]
                probs = torch.softmax(choice_logits, dim=-1)
                confidence = probs.max().item()
            else:
                # General case: use full vocabulary
                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max().item()
        
        return confidence
    
    def estimate_feature_diversity(
        self,
        sae: nn.Module,
        hidden_state: torch.Tensor,
        normalize: bool = True,
    ) -> float:
        """Estimate feature diversity from SAE activation pattern.
        
        From CRL Paper (2602.10437): Complex tasks activate more features.
        GSM8K (complex): ~6.7 active features
        HarmBench (focused): ~2.9 active features
        
        Args:
            sae: Sparse Autoencoder model
            hidden_state: Hidden state tensor [batch, seq, hidden] or [batch, hidden]
            normalize: Whether to normalize by total features
            
        Returns:
            Feature diversity score (fraction of active features if normalized).
        """
        # Get last token hidden state
        if hidden_state.dim() == 3:
            sl = getattr(self, '_batch_seq_lengths', None)
            hidden_state = _gather_last_tokens(hidden_state, sl).squeeze(1) if sl is not None else hidden_state[:, -1, :]

        with torch.no_grad():
            features = sae.encode(hidden_state.to(sae.dtype))
            n_active = (features > 0).sum().item()
            
            if normalize:
                n_total = features.shape[-1]
                diversity = n_active / n_total
            else:
                diversity = float(n_active)
        
        return diversity
    
    def estimate_critic_value(
        self,
        critic_network: Optional[nn.Module],
        hidden_state: torch.Tensor,
    ) -> float:
        """Estimate critic value (if critic network available).
        
        From CRL Paper: Critic networks learn a value function for 
        steering effectiveness. Higher values = steering more likely to help.
        
        Note: For single-token tasks (MMLU-style), critics have limited effectiveness.
        
        Args:
            critic_network: Optional trained critic network
            hidden_state: Hidden state tensor
            
        Returns:
            Critic value (0.5 = neutral if no critic available).
        """
        if critic_network is None:
            return 0.5  # Neutral default
        
        # Get last token hidden state
        if hidden_state.dim() == 3:
            sl = getattr(self, '_batch_seq_lengths', None)
            hidden_state = _gather_last_tokens(hidden_state, sl).squeeze(1) if sl is not None else hidden_state[:, -1, :]

        with torch.no_grad():
            value = critic_network(hidden_state).item()
            value = float(torch.sigmoid(torch.tensor(value)))  # Normalize to [0, 1]
        
        return value
    
    def estimate_difficulty(
        self,
        confidence: float,
        feature_diversity: float,
        critic_value: float = 0.5,
    ) -> float:
        """Compute hybrid difficulty score.
        
        Difficulty = (1 - confidence) * w_c + diversity_scaled * w_d + (1 - critic) * w_v
        
        High difficulty = model struggles, complex task, steering unlikely to help.
        Low difficulty = model confident, simple task, steering helps.
        
        Args:
            confidence: Model confidence (0-1)
            feature_diversity: Feature diversity score (0-1 if normalized)
            critic_value: Critic value (0-1, 0.5 = neutral)
            
        Returns:
            Difficulty score (0-1).
        """
        w = self.weights
        
        # Scale feature diversity (raw values are tiny, e.g. 6.7/65000 ≈ 0.0001)
        scaled_diversity = min(feature_diversity * self.feature_diversity_scale, 1.0)
        
        # Invert confidence: low confidence = high difficulty component
        confidence_component = 1.0 - confidence
        
        # Invert critic: low critic value = high difficulty component
        critic_component = 1.0 - critic_value
        
        # Weighted combination
        difficulty = (
            confidence_component * w.confidence +
            scaled_diversity * w.feature_diversity +
            critic_component * w.critic_value
        )
        
        return min(max(difficulty, 0.0), 1.0)  # Clamp to [0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive Strength Calculator
# ═══════════════════════════════════════════════════════════════════════════════


class AdaptiveStrengthCalculator:
    """Calculates intervention strength based on difficulty (p_correct).
    
    Threshold mapping:
    - p < 0.20 (very_hard): strength 1.5 (maximum steering)
    - p < 0.35 (hard): strength 1.0 (normal steering)
    - p < 0.50 (medium): strength 0.3 (light steering)
    - p >= 0.50 (easy): strength 0.0 (no steering)
    
    Supports smooth transitions (linear interpolation) or hard thresholds.
    """
    
    def __init__(
        self,
        thresholds: Optional[DifficultyThresholds] = None,
        strengths: Optional[StrengthValues] = None,
        smooth_transition: bool = True,
    ):
        self.thresholds = thresholds or DifficultyThresholds()
        self.strengths = strengths or StrengthValues()
        self.smooth_transition = smooth_transition
    
    def compute_strength(self, p_correct: float) -> float:
        """Compute intervention strength for given difficulty.
        
        Args:
            p_correct: Estimated probability of correct answer (model confidence).
                       Lower = harder question = more steering needed.
            
        Returns:
            Intervention strength multiplier.
        """
        t = self.thresholds
        s = self.strengths
        
        if self.smooth_transition:
            return self._smooth_strength(p_correct, t, s)
        else:
            return self._step_strength(p_correct, t, s)
    
    def _step_strength(
        self,
        p_correct: float,
        t: DifficultyThresholds,
        s: StrengthValues,
    ) -> float:
        """Step-function strength (hard thresholds)."""
        if p_correct < t.very_hard:
            return s.very_hard
        elif p_correct < t.hard:
            return s.hard
        elif p_correct < t.medium:
            return s.medium
        else:
            return s.easy
    
    def _smooth_strength(
        self,
        p_correct: float,
        t: DifficultyThresholds,
        s: StrengthValues,
    ) -> float:
        """Smooth strength with linear interpolation between thresholds."""
        if p_correct < t.very_hard:
            # Very hard: max strength
            return s.very_hard
        elif p_correct < t.hard:
            # Interpolate between very_hard and hard
            ratio = (p_correct - t.very_hard) / (t.hard - t.very_hard)
            return s.very_hard + ratio * (s.hard - s.very_hard)
        elif p_correct < t.medium:
            # Interpolate between hard and medium
            ratio = (p_correct - t.hard) / (t.medium - t.hard)
            return s.hard + ratio * (s.medium - s.hard)
        else:
            # Easy: no steering (or interpolate to easy if easy > 0)
            # For the standard case where easy = 0.0, we return 0 directly
            if s.easy == 0.0:
                return 0.0
            # If easy strength is non-zero, interpolate
            ratio = min((p_correct - t.medium) / (1.0 - t.medium), 1.0)
            return s.medium + ratio * (s.easy - s.medium)


# ═══════════════════════════════════════════════════════════════════════════════
# Confidence Gating
# ═══════════════════════════════════════════════════════════════════════════════


class ConfidenceGate:
    """Determines whether steering should be applied based on confidence.
    
    Gating rule: Skip steering when:
    - Confidence > threshold (model is very confident)
    - AND difficulty < threshold (question is easy for model)
    
    Rationale: Steering on high-confidence easy questions can cause harm
    (over-correction) without any benefit.
    """
    
    def __init__(self, config: Optional[ConfidenceGatingConfig] = None):
        self.config = config or ConfidenceGatingConfig()
    
    def should_steer(self, confidence: float, difficulty: float) -> bool:
        """Check if steering should be applied.
        
        Args:
            confidence: Model confidence (p_correct)
            difficulty: Estimated difficulty (inverse of confidence typically)
            
        Returns:
            True if steering should be applied, False to skip.
        """
        if not self.config.enabled:
            return True
        
        # Skip steering on high-confidence easy questions
        if (confidence > self.config.confidence_threshold and 
            difficulty < self.config.difficulty_threshold):
            return False
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Repetition Detection (Over-Steering Prevention)
# ═══════════════════════════════════════════════════════════════════════════════


class RepetitionDetector:
    """Detects over-steering via n-gram repetition analysis.
    
    From SAE-Steering Paper (2601.03595): Excessive steering causes 
    repetitive outputs. This detector catches such cases and enables
    progressive strength reduction.
    """
    
    def __init__(self, config: Optional[RepetitionDetectionConfig] = None):
        self.config = config or RepetitionDetectionConfig()
    
    def detect_repetition(self, text: str) -> float:
        """Detect n-gram repetition ratio in generated text.
        
        Args:
            text: Generated text to analyze.
            
        Returns:
            Repetition ratio (0 = no repetition, 1 = full repetition).
        """
        if not text or not text.strip():
            return 0.0
        
        tokens = text.split()
        n = self.config.n_gram_size
        
        if len(tokens) < n:
            return 0.0
        
        # Extract n-grams
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        if len(ngrams) == 0:
            return 0.0
        
        # Calculate repetition ratio
        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - (len(unique_ngrams) / len(ngrams))
        
        return repetition_ratio
    
    def is_over_steering(self, text: str) -> bool:
        """Check if text shows signs of over-steering.
        
        Args:
            text: Generated text to check.
            
        Returns:
            True if over-steering detected (high repetition).
        """
        if not self.config.enabled:
            return False
        
        repetition = self.detect_repetition(text)
        return repetition > self.config.max_repetition_ratio
    
    def compute_adjusted_strength(
        self,
        initial_strength: float,
        generated_text: str,
    ) -> Tuple[float, bool]:
        """Compute adjusted strength if over-steering detected.
        
        Args:
            initial_strength: Initial steering strength.
            generated_text: Generated text to check.
            
        Returns:
            Tuple of (adjusted_strength, should_regenerate).
        """
        if not self.is_over_steering(generated_text):
            return initial_strength, False
        
        # Reduce strength
        adjusted = initial_strength * self.config.decay_factor
        return adjusted, True


# ═══════════════════════════════════════════════════════════════════════════════
# Main DifficultyAdaptiveSteering Class
# ═══════════════════════════════════════════════════════════════════════════════


class DifficultyAdaptiveSteering:
    """Main class for difficulty-adaptive steering intervention.
    
    Combines all components:
    1. Hybrid Difficulty Estimator
    2. Adaptive Strength Calculator
    3. Confidence Gating
    4. Repetition Detection
    
    Usage:
        >>> steering = DifficultyAdaptiveSteering(sae, intervention_vec)
        >>> strength = steering.compute_intervention_strength(
        ...     model, input_ids, hidden_state
        ... )
        >>> if strength > 0:
        ...     # Apply steering with computed strength
        ...     modified = steering.apply_steering(hidden_state, strength)
    """
    
    def __init__(
        self,
        sae: nn.Module,
        intervention_vec: torch.Tensor,
        # Component configurations
        difficulty_weights: Optional[HybridWeights] = None,
        thresholds: Optional[DifficultyThresholds] = None,
        strengths: Optional[StrengthValues] = None,
        gating_config: Optional[ConfidenceGatingConfig] = None,
        repetition_config: Optional[RepetitionDetectionConfig] = None,
        # Behavior flags
        smooth_transition: bool = True,
        use_feature_diversity: bool = True,
        use_critic: bool = False,
        critic_network: Optional[nn.Module] = None,
    ):
        """Initialize difficulty-adaptive steering.
        
        Args:
            sae: Sparse Autoencoder model.
            intervention_vec: Intervention vector in feature space.
            difficulty_weights: Weights for hybrid difficulty estimator.
            thresholds: Difficulty thresholds for strength mapping.
            strengths: Strength values for each difficulty level.
            gating_config: Configuration for confidence gating.
            repetition_config: Configuration for repetition detection.
            smooth_transition: Use smooth interpolation vs hard thresholds.
            use_feature_diversity: Include feature diversity in difficulty estimation.
            use_critic: Include critic value in difficulty estimation.
            critic_network: Optional trained critic network.
        """
        self.sae = sae
        self.intervention_vec = intervention_vec
        self.use_feature_diversity = use_feature_diversity
        self.use_critic = use_critic
        self.critic_network = critic_network
        
        # Initialize components
        self.difficulty_estimator = HybridDifficultyEstimator(
            weights=difficulty_weights or HybridWeights()
        )
        
        self.strength_calculator = AdaptiveStrengthCalculator(
            thresholds=thresholds or DifficultyThresholds(),
            strengths=strengths or StrengthValues(),
            smooth_transition=smooth_transition,
        )
        
        self.confidence_gate = ConfidenceGate(gating_config)
        self.repetition_detector = RepetitionDetector(repetition_config)
        
        # State for tracking
        self.last_difficulty: Optional[float] = None
        self.last_confidence: Optional[float] = None
        self.last_strength: Optional[float] = None
        self.last_steered: Optional[bool] = None
    
    def estimate_difficulty(
        self,
        logits: torch.Tensor,
        hidden_state: torch.Tensor,
        answer_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """Estimate difficulty using hybrid approach.
        
        Args:
            logits: Model output logits.
            hidden_state: Hidden state at intervention layer.
            answer_indices: Optional indices for multiple-choice answers.
            
        Returns:
            Tuple of (difficulty, confidence, feature_diversity).
        """
        # Get confidence
        confidence = self.difficulty_estimator.estimate_confidence(
            logits, answer_indices
        )
        
        # Get feature diversity (optional)
        if self.use_feature_diversity:
            feature_diversity = self.difficulty_estimator.estimate_feature_diversity(
                self.sae, hidden_state
            )
        else:
            feature_diversity = 0.5  # Neutral
        
        # Get critic value (optional)
        if self.use_critic and self.critic_network is not None:
            critic_value = self.difficulty_estimator.estimate_critic_value(
                self.critic_network, hidden_state
            )
        else:
            critic_value = 0.5  # Neutral
        
        # Compute hybrid difficulty
        difficulty = self.difficulty_estimator.estimate_difficulty(
            confidence, feature_diversity, critic_value
        )
        
        return difficulty, confidence, feature_diversity
    
    def compute_intervention_strength(
        self,
        logits: torch.Tensor,
        hidden_state: torch.Tensor,
        answer_indices: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute optimal intervention strength.
        
        This is the main entry point for determining how much steering
        to apply based on estimated question difficulty.
        
        Args:
            logits: Model output logits.
            hidden_state: Hidden state at intervention layer.
            answer_indices: Optional indices for multiple-choice answers.
            
        Returns:
            Intervention strength (0 = no steering, >0 = apply steering).
        """
        # Estimate difficulty
        difficulty, confidence, _ = self.estimate_difficulty(
            logits, hidden_state, answer_indices
        )
        
        # Store for analysis
        self.last_difficulty = difficulty
        self.last_confidence = confidence
        
        # Check confidence gate
        if not self.confidence_gate.should_steer(confidence, difficulty):
            self.last_strength = 0.0
            self.last_steered = False
            return 0.0
        
        # Compute adaptive strength based on p_correct (confidence)
        strength = self.strength_calculator.compute_strength(confidence)
        
        # Store for analysis
        self.last_strength = strength
        self.last_steered = strength > 0
        
        return strength
    
    def apply_steering(
        self,
        hidden_states: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """Apply steering vector to hidden states.
        
        Args:
            hidden_states: Hidden states tensor [batch, seq, hidden].
            strength: Intervention strength multiplier.
            
        Returns:
            Modified hidden states.
        """
        if strength == 0.0:
            return hidden_states
        
        # Get device and dtype
        device = hidden_states.device
        dtype = hidden_states.dtype
        sae_dtype = self.sae.dtype if hasattr(self.sae, 'dtype') else torch.float32
        
        # Prepare intervention vector
        effective_vec = (self.intervention_vec * strength).to(device=device, dtype=sae_dtype)
        
        # Apply intervention
        # For SAE-based intervention: encode -> add -> decode
        # For approximated intervention: direct addition
        
        seq_lengths = getattr(self, '_batch_seq_lengths', None)

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden_states, seq_lengths).to(sae_dtype)
        else:
            last_hidden = hidden_states[:, -1:, :].to(sae_dtype)

        # SAE encode
        features = self.sae.encode(last_hidden)
        reconstructed = self.sae.decode(features)
        error = last_hidden - reconstructed

        # Apply intervention in feature space
        features = features + effective_vec.unsqueeze(0).unsqueeze(0)
        features = torch.clamp(features, min=0)  # ReLU

        # SAE decode
        modified_last = (self.sae.decode(features) + error).to(dtype)

        if seq_lengths is not None:
            return _scatter_last_tokens(hidden_states, modified_last, seq_lengths)
        context = hidden_states[:, :-1, :]
        return torch.cat([context, modified_last], dim=1)
    
    def progressive_generate(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        generate_fn: Callable,
        max_new_tokens: int = 10,
    ) -> Tuple[str, float, Dict]:
        """Generate with progressive strength adjustment if repetition detected.
        
        This method handles the full steering + generation pipeline with
        automatic over-steering prevention.
        
        Args:
            model: Language model.
            input_ids: Input token IDs.
            generate_fn: Function to generate text (model.generate or similar).
            max_new_tokens: Maximum tokens to generate.
            
        Returns:
            Tuple of (generated_text, final_strength, metadata).
        """
        # First, get logits and hidden state for difficulty estimation
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits
            hidden_state = outputs.hidden_states[-1]  # Last layer
        
        # Compute initial strength
        strength = self.compute_intervention_strength(logits, hidden_state)
        
        if strength == 0:
            # No steering needed
            text = generate_fn(input_ids, max_new_tokens=max_new_tokens)
            return text, 0.0, {"steered": False, "reason": "gated"}
        
        # Generate with steering
        current_strength = strength
        for attempt in range(self.repetition_detector.config.max_retries):
            # Generate with current strength
            # Note: This requires model-specific generation with hooks
            # Simplified version here
            text = generate_fn(
                input_ids, 
                max_new_tokens=max_new_tokens,
                steering_strength=current_strength,
            )
            
            # Check for over-steering
            adjusted, should_retry = self.repetition_detector.compute_adjusted_strength(
                current_strength, text
            )
            
            if not should_retry:
                # Good generation
                return text, current_strength, {
                    "steered": True,
                    "strength": current_strength,
                    "attempts": attempt + 1,
                }
            
            # Reduce strength and retry
            current_strength = adjusted
        
        # Fallback: return last generation
        return text, current_strength, {
            "steered": True,
            "strength": current_strength,
            "attempts": self.repetition_detector.config.max_retries,
            "fallback": True,
        }
    
    def get_diagnostic_info(self) -> Dict:
        """Get diagnostic information about last intervention.
        
        Returns:
            Dict with last computed values for analysis.
        """
        return {
            "difficulty": self.last_difficulty,
            "confidence": self.last_confidence,
            "strength": self.last_strength,
            "steered": self.last_steered,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Factory Functions
# ═══════════════════════════════════════════════════════════════════════════════


def create_difficulty_adaptive_steering(
    sae: nn.Module,
    intervention_vec: torch.Tensor,
    mode: Literal["conservative", "aggressive", "balanced"] = "balanced",
) -> DifficultyAdaptiveSteering:
    """Create DifficultyAdaptiveSteering with preset configuration.
    
    Args:
        sae: Sparse Autoencoder model.
        intervention_vec: Intervention vector in feature space.
        mode: Preset mode for configuration.
            - "conservative": Lower strengths, stricter gating
            - "aggressive": Higher strengths, looser gating
            - "balanced": Default values from research
            
    Returns:
        Configured DifficultyAdaptiveSteering instance.
    """
    if mode == "conservative":
        thresholds = DifficultyThresholds(
            very_hard=0.15,
            hard=0.30,
            medium=0.45,
        )
        strengths = StrengthValues(
            very_hard=1.0,
            hard=0.7,
            medium=0.2,
            easy=0.0,
        )
        gating = ConfidenceGatingConfig(
            enabled=True,
            confidence_threshold=0.65,
            difficulty_threshold=0.35,
        )
        smooth = True
        
    elif mode == "aggressive":
        thresholds = DifficultyThresholds(
            very_hard=0.25,
            hard=0.40,
            medium=0.55,
        )
        strengths = StrengthValues(
            very_hard=2.0,
            hard=1.5,
            medium=0.5,
            easy=0.0,
        )
        gating = ConfidenceGatingConfig(
            enabled=True,
            confidence_threshold=0.80,
            difficulty_threshold=0.25,
        )
        smooth = True
        
    else:  # balanced
        thresholds = DifficultyThresholds()  # Defaults: 0.20, 0.35, 0.50
        strengths = StrengthValues()  # Defaults: 1.5, 1.0, 0.3, 0.0
        gating = ConfidenceGatingConfig()  # Defaults: 0.7, 0.3
        smooth = True
    
    return DifficultyAdaptiveSteering(
        sae=sae,
        intervention_vec=intervention_vec,
        thresholds=thresholds,
        strengths=strengths,
        gating_config=gating,
        smooth_transition=smooth,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with existing hooks
# ═══════════════════════════════════════════════════════════════════════════════


class DifficultyAdaptiveInterventionHook:
    """Hook for difficulty-adaptive intervention during forward pass.
    
    This hook integrates DifficultyAdaptiveSteering into the model's
    forward pass, automatically computing and applying the appropriate
    steering strength.
    
    Usage:
        >>> steering = DifficultyAdaptiveSteering(sae, vec)
        >>> hook = DifficultyAdaptiveInterventionHook(steering)
        >>> 
        >>> # Before forward pass, set logits for difficulty estimation
        >>> hook.set_logits(model(input_ids).logits)
        >>> hook.enabled = True
        >>> 
        >>> with hook_mgr.intervention(hook):
        ...     outputs = model(input_ids)
    """
    
    def __init__(self, steering: DifficultyAdaptiveSteering):
        """Initialize the hook.
        
        Args:
            steering: DifficultyAdaptiveSteering instance.
        """
        self.steering = steering
        self.enabled = False
        self._logits: Optional[torch.Tensor] = None
        self._answer_indices: Optional[torch.Tensor] = None
    
    def set_logits(self, logits: torch.Tensor) -> None:
        """Set logits for difficulty estimation."""
        self._logits = logits
    
    def set_answer_indices(self, indices: torch.Tensor) -> None:
        """Set answer indices for multiple-choice confidence."""
        self._answer_indices = indices
    
    def __call__(self, module, input, output):
        """Apply intervention during forward pass."""
        if not self.enabled:
            return output
        
        hidden = output[0] if isinstance(output, tuple) else output
        
        # Compute strength if logits available
        strength = 0.0
        if self._logits is not None:
            strength = self.steering.compute_intervention_strength(
                self._logits, hidden, self._answer_indices
            )
        
        if strength == 0.0:
            return output
        
        # Apply steering
        modified = self.steering.apply_steering(hidden, strength)
        
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified
