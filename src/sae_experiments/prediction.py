"""Forward pass, logit extraction, and prediction functions."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adaptive_steering import (
    AdaptiveApproximatedInterventionHook,
    AdaptiveInterventionHook,
    get_strength_fn,
)
from .benchmarks import extract_answer, prepare_input
from .config import AdaptiveConfig, BenchmarkConfig
from .hooks import (
    ApproximatedInterventionHook,
    HookManager,
    InterventionHook,
)


def get_logit_prediction(
    sample: Dict,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    extract_features: bool = True,
) -> Dict:
    """Run a forward pass and extract logit-based prediction."""
    input_ids, attention_mask = prepare_input(sample, config, tokenizer, device)

    hook_mgr.cache.clear()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]
    answer_logits = logits[:, answer_token_id_list]
    probs = torch.softmax(answer_logits, dim=-1)

    pred_idx = torch.argmax(answer_logits, dim=-1).item()
    pred_letter = config.choices[pred_idx]

    correct_letter, correct_idx = extract_answer(sample, config)
    is_correct = pred_idx == correct_idx

    result = {
        "prediction": pred_letter,
        "correct": is_correct,
        "expected": correct_letter,
        "prob_correct": probs[0, correct_idx].item(),
        "max_confidence": probs[0].max().item(),
    }

    if extract_features:
        last_hidden = hook_mgr.cache.get_last_token()
        if last_hidden is not None:
            features = sae.encode(last_hidden.to(sae.dtype))
            active_mask = features[0, 0, :] > 0
            active_indices = torch.where(active_mask)[0]
            active_values = features[0, 0, active_mask]
        else:
            active_indices = torch.tensor([], device=device)
            active_values = torch.tensor([], device=device)

        result["active_indices"] = active_indices
        result["active_values"] = active_values

    return result


def get_prediction_with_intervention(
    sample: Dict,
    intervention_vec: torch.Tensor,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
) -> Dict:
    """Forward pass with exact SAE intervention."""
    int_hook = InterventionHook(sae, intervention_vec)
    int_hook.enabled = True

    input_ids, attention_mask = prepare_input(sample, config, tokenizer, device)

    with hook_mgr.intervention(int_hook):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]
    answer_logits = logits[:, answer_token_id_list]
    probs = torch.softmax(answer_logits, dim=-1)

    pred_idx = torch.argmax(answer_logits, dim=-1).item()
    pred_letter = config.choices[pred_idx]

    correct_letter, correct_idx = extract_answer(sample, config)
    is_correct = pred_idx == correct_idx

    return {
        "prediction": pred_letter,
        "correct": is_correct,
        "expected": correct_letter,
        "prob_correct": probs[0, correct_idx].item(),
        "max_confidence": probs[0].max().item(),
    }


def get_prediction_with_approximated_intervention(
    sample: Dict,
    precomputed_hidden_vec: torch.Tensor,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    answer_token_id_list: List[int],
    device: torch.device,
    sae=None,
) -> Dict:
    """Forward pass with approximated intervention (no SAE at runtime)."""
    int_hook = ApproximatedInterventionHook(precomputed_hidden_vec)
    int_hook.enabled = True

    input_ids, attention_mask = prepare_input(sample, config, tokenizer, device)

    with hook_mgr.intervention(int_hook):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]
    answer_logits = logits[:, answer_token_id_list]
    probs = torch.softmax(answer_logits, dim=-1)

    pred_idx = torch.argmax(answer_logits, dim=-1).item()
    pred_letter = config.choices[pred_idx]

    correct_letter, correct_idx = extract_answer(sample, config)
    is_correct = pred_idx == correct_idx

    return {
        "prediction": pred_letter,
        "correct": is_correct,
        "expected": correct_letter,
        "prob_correct": probs[0, correct_idx].item(),
        "max_confidence": probs[0].max().item(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Adaptive intervention variants
# ═══════════════════════════════════════════════════════════════════════════════


def get_prediction_with_adaptive_intervention(
    sample: Dict,
    intervention_vec: torch.Tensor,
    p_correct: float,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    adaptive_config: Optional[AdaptiveConfig] = None,
) -> Dict:
    """Forward pass with difficulty-adaptive exact SAE intervention.
    
    The intervention strength is dynamically adjusted based on p_correct,
    which estimates the difficulty of the question.
    
    Args:
        sample: The question sample.
        intervention_vec: The intervention vector in feature space.
        p_correct: Estimated probability of correct answer (from screening).
        adaptive_config: Configuration for adaptive steering. If None, uses defaults.
        Other args: Same as get_prediction_with_intervention.
        
    Returns:
        Dict with prediction, correct, expected, prob_correct.
    """
    # Use default adaptive config if not provided
    if adaptive_config is None:
        adaptive_config = AdaptiveConfig()
    
    # Create adaptive hook
    strength_fn = get_strength_fn(
        adaptive_config.mode,
        p_min=adaptive_config.p_min,
        p_sweet=adaptive_config.p_sweet,
        p_max=adaptive_config.p_max,
        mu=adaptive_config.mu,
        sigma=adaptive_config.sigma,
        transition_strength=adaptive_config.transition_strength,
    )

    int_hook = AdaptiveInterventionHook(sae, intervention_vec, strength_fn)
    int_hook.set_p_correct(p_correct)
    int_hook.enabled = True

    input_ids, attention_mask = prepare_input(sample, config, tokenizer, device)

    with hook_mgr.intervention(int_hook):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]
    answer_logits = logits[:, answer_token_id_list]
    probs = torch.softmax(answer_logits, dim=-1)

    pred_idx = torch.argmax(answer_logits, dim=-1).item()
    pred_letter = config.choices[pred_idx]

    correct_letter, correct_idx = extract_answer(sample, config)
    is_correct = pred_idx == correct_idx

    return {
        "prediction": pred_letter,
        "correct": is_correct,
        "expected": correct_letter,
        "prob_correct": probs[0, correct_idx].item(),
    }


def get_prediction_with_adaptive_approximated_intervention(
    sample: Dict,
    precomputed_hidden_vec: torch.Tensor,
    p_correct: float,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    answer_token_id_list: List[int],
    device: torch.device,
    sae=None,
    adaptive_config: Optional[AdaptiveConfig] = None,
) -> Dict:
    """Forward pass with difficulty-adaptive approximated intervention (no SAE at runtime).
    
    Same as get_prediction_with_adaptive_intervention but uses precomputed
    hidden vector instead of SAE encode/decode.
    
    Args:
        sample: The question sample.
        precomputed_hidden_vec: Precomputed intervention vector in hidden space.
        p_correct: Estimated probability of correct answer (from screening).
        adaptive_config: Configuration for adaptive steering. If None, uses defaults.
        Other args: Same as get_prediction_with_approximated_intervention.
        
    Returns:
        Dict with prediction, correct, expected, prob_correct.
    """
    # Use default adaptive config if not provided
    if adaptive_config is None:
        adaptive_config = AdaptiveConfig()
    
    # Create adaptive hook
    strength_fn = get_strength_fn(
        adaptive_config.mode,
        p_min=adaptive_config.p_min,
        p_sweet=adaptive_config.p_sweet,
        p_max=adaptive_config.p_max,
        mu=adaptive_config.mu,
        sigma=adaptive_config.sigma,
        transition_strength=adaptive_config.transition_strength,
    )

    int_hook = AdaptiveApproximatedInterventionHook(precomputed_hidden_vec, strength_fn)
    int_hook.set_p_correct(p_correct)
    int_hook.enabled = True

    input_ids, attention_mask = prepare_input(sample, config, tokenizer, device)

    with hook_mgr.intervention(int_hook):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[:, -1, :]
    answer_logits = logits[:, answer_token_id_list]
    probs = torch.softmax(answer_logits, dim=-1)

    pred_idx = torch.argmax(answer_logits, dim=-1).item()
    pred_letter = config.choices[pred_idx]

    correct_letter, correct_idx = extract_answer(sample, config)
    is_correct = pred_idx == correct_idx

    return {
        "prediction": pred_letter,
        "correct": is_correct,
        "expected": correct_letter,
        "prob_correct": probs[0, correct_idx].item(),
    }
