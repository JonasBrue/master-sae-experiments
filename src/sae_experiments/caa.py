"""Contrastive Activation Addition (CAA) baseline.

Computes a steering vector as the mean difference of hidden states
between correctly and incorrectly answered questions, directly in
the residual stream (no SAE decomposition).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Union

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BenchmarkConfig
from .fast_inference import (
    batched_intervention_predictions,
    make_approx_intervention_fn,
    prepare_batch_inputs,
)
from .hooks import HookManager
from .screening import screening_to_sample


def compute_caa_vector(
    train_data: List[Dict],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute a CAA steering vector from training data.

    For each question in train_data, captures the last-token hidden state
    from the hooked layer. Splits by correct/incorrect and returns:

        caa_vec = mean(hidden_correct) - mean(hidden_incorrect)

    Args:
        train_data: Screening entries with 'correct' labels and '_sample'.
        model: The language model.
        tokenizer: The tokenizer.
        config: Benchmark config for prompt formatting.
        hook_mgr: Hook manager with cache hook registered on the target layer.
        answer_token_id_list: Token IDs for answer choices.
        device: Torch device.
        batch_size: Batch size for forward passes.

    Returns:
        Tensor of shape [d_model] — the CAA steering vector.
    """
    print("=" * 80)
    print("CAA: COMPUTING CONTRASTIVE ACTIVATION VECTOR")
    print("=" * 80)

    samples = [screening_to_sample(e) for e in train_data]
    labels = [e["correct"] for e in train_data]

    n_correct = sum(labels)
    n_incorrect = len(labels) - n_correct
    print(f"  Training data: {len(train_data)} questions ({n_correct} correct, {n_incorrect} incorrect)")

    # Accumulate hidden states using running sums to save memory
    sum_correct = None
    sum_incorrect = None
    count_correct = 0
    count_incorrect = 0

    for start in tqdm(range(0, len(samples), batch_size), desc="CAA Hidden States"):
        end = min(start + batch_size, len(samples))
        batch_samples = samples[start:end]
        batch_labels = labels[start:end]

        input_ids, attention_mask, seq_lengths = prepare_batch_inputs(
            batch_samples, config, tokenizer, device,
        )

        hook_mgr.layer_module._batch_seq_lengths = seq_lengths
        hook_mgr.cache.clear()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden = hook_mgr.cache.get_last_token(seq_lengths)  # [B, 1, d_model]

        for i, is_correct in enumerate(batch_labels):
            h = last_hidden[i, 0, :].float()  # [d_model], upcast to float32 for stable mean
            if is_correct:
                if sum_correct is None:
                    sum_correct = torch.zeros_like(h)
                sum_correct += h
                count_correct += 1
            else:
                if sum_incorrect is None:
                    sum_incorrect = torch.zeros_like(h)
                sum_incorrect += h
                count_incorrect += 1

    if count_correct == 0 or count_incorrect == 0:
        raise ValueError(
            f"Cannot compute CAA vector: {count_correct} correct, {count_incorrect} incorrect. "
            "Need at least one of each."
        )

    mean_correct = sum_correct / count_correct
    mean_incorrect = sum_incorrect / count_incorrect
    caa_vec = mean_correct - mean_incorrect

    print(f"  CAA vector computed:")
    print(f"    Correct mean norm:   {(sum_correct / count_correct).norm().item():.4f}")
    print(f"    Incorrect mean norm: {(sum_incorrect / count_incorrect).norm().item():.4f}")
    print(f"    CAA vector norm:     {caa_vec.norm().item():.4f}")
    print(f"    CAA vector dtype:    {caa_vec.dtype}")
    print("=" * 80)

    return caa_vec


def run_caa_strength_sweep(
    caa_vec: torch.Tensor,
    strengths: List[float],
    eval_data: List[Dict],
    baseline_eval_results: List[Dict],
    baseline_eval_acc: float,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Sweep over strength values for the CAA vector.

    For each strength, scales the CAA vector and evaluates on the given
    eval set using the approximated intervention path.

    Returns:
        DataFrame with columns: strength, accuracy, delta, n_improved, n_worsened.
    """
    print("=" * 80)
    print("CAA: STRENGTH SWEEP")
    print(f"  Strengths: {strengths}")
    print(f"  Eval set: {len(eval_data)} questions")
    print("=" * 80)

    eval_samples = [screening_to_sample(e) for e in eval_data]
    results = []

    for strength in tqdm(strengths, desc="CAA Strength Sweep"):
        scaled_vec = caa_vec * strength
        intervention_fn = make_approx_intervention_fn(scaled_vec)

        int_results = batched_intervention_predictions(
            eval_samples, intervention_fn,
            batch_size=batch_size,
            model=model, tokenizer=tokenizer, config=config,
            hook_mgr=hook_mgr, sae=sae,
            answer_token_id_list=answer_token_id_list,
            device=device,
            desc=f"Strength {strength}",
        )

        int_correct = sum(r["correct"] for r in int_results)
        n_improved = sum(
            1 for i, r in enumerate(int_results)
            if not baseline_eval_results[i]["correct"] and r["correct"]
        )
        n_worsened = sum(
            1 for i, r in enumerate(int_results)
            if baseline_eval_results[i]["correct"] and not r["correct"]
        )

        acc = int_correct / len(eval_data)
        delta = acc - baseline_eval_acc

        results.append({
            "strength": strength,
            "accuracy": acc,
            "delta": delta,
            "n_improved": n_improved,
            "n_worsened": n_worsened,
            "net_change": n_improved - n_worsened,
        })

        print(f"  strength={strength:.2f}: acc={acc*100:.1f}%, delta={delta*100:+.1f}% "
              f"({n_improved}+ {n_worsened}-)")

    print("=" * 80)
    return pd.DataFrame(results)
