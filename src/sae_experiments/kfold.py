"""K-fold partitioned evaluation of the best configuration."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest, t as t_dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import BenchmarkConfig, CausalConfig
from .fast_inference import batched_logit_predictions, batched_intervention_predictions
from .hooks import HookManager
from .prediction import get_logit_prediction, get_prediction_with_intervention
from .screening import screening_to_sample


def _create_kfold_splits(
    data: List, n_folds: int, seed: int
) -> List[Tuple[List[int], List[int]]]:
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    fold_size = len(indices) // n_folds
    folds = []

    for k in range(n_folds):
        start_idx = k * fold_size
        end_idx = len(indices) if k == n_folds - 1 else start_idx + fold_size
        test_indices = indices[start_idx:end_idx]
        train_indices = indices[:start_idx] + indices[end_idx:]
        folds.append((train_indices, test_indices))

    return folds


def run_kfold_validation(
    holdout_data: List[Dict],
    intervention: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    *,
    causal_config: CausalConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 1,
) -> Dict:
    """Run k-fold partitioned evaluation on holdout data and return aggregated results.

    Uses only holdout_data (never seen by feature discovery, causal screening,
    or grid search) for an uncontaminated variance estimate.
    """
    print("=" * 80)
    print("K-FOLD VALIDATION")
    print("=" * 80)

    kfold_data = list(holdout_data)
    n_folds = causal_config.n_folds
    print(f"  Data: {len(kfold_data)} questions, K={n_folds} folds (batch_size={batch_size})")

    folds = _create_kfold_splits(kfold_data, n_folds, causal_config.seed)
    kfold_results = []

    # Callable interventions (e.g. CAA hidden-space vectors) require batched inference
    if callable(intervention) and batch_size <= 1:
        batch_size = 32

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        fold_test_data = [kfold_data[i] for i in test_idx]
        fold_test_samples = [screening_to_sample(e) for e in fold_test_data]

        if batch_size > 1:
            # Batched baseline
            baseline_results = batched_logit_predictions(
                fold_test_samples,
                batch_size=batch_size,
                model=model, tokenizer=tokenizer, config=config,
                hook_mgr=hook_mgr, sae=sae,
                answer_token_id_list=answer_token_id_list, device=device,
                extract_features=False,
                desc=f"Fold {fold_id+1} Baseline",
            )
            fold_baseline_results = [r["correct"] for r in baseline_results]

            # Batched intervention
            int_results = batched_intervention_predictions(
                fold_test_samples,
                intervention,
                batch_size=batch_size,
                model=model, tokenizer=tokenizer, config=config,
                hook_mgr=hook_mgr, sae=sae,
                answer_token_id_list=answer_token_id_list, device=device,
                desc=f"Fold {fold_id+1} Intervention",
            )

            fold_int_correct = sum(r["correct"] for r in int_results)
            fold_improved = sum(
                1 for i, r in enumerate(int_results)
                if not fold_baseline_results[i] and r["correct"]
            )
            fold_worsened = sum(
                1 for i, r in enumerate(int_results)
                if fold_baseline_results[i] and not r["correct"]
            )
        else:
            # Single-sample baseline
            fold_baseline_results = []
            for screening_entry in fold_test_data:
                sample = screening_to_sample(screening_entry)
                result = get_logit_prediction(
                    sample,
                    model=model, tokenizer=tokenizer, config=config,
                    hook_mgr=hook_mgr, sae=sae,
                    answer_token_id_list=answer_token_id_list,
                    device=device, extract_features=False,
                )
                fold_baseline_results.append(result["correct"])

            # Single-sample intervention
            fold_int_correct = 0
            fold_improved = 0
            fold_worsened = 0

            for i, screening_entry in enumerate(fold_test_data):
                sample = screening_to_sample(screening_entry)
                result = get_prediction_with_intervention(
                    sample, intervention,
                    model=model, tokenizer=tokenizer, config=config,
                    hook_mgr=hook_mgr, sae=sae,
                    answer_token_id_list=answer_token_id_list, device=device,
                )
                fold_int_correct += result["correct"]

                if not fold_baseline_results[i] and result["correct"]:
                    fold_improved += 1
                elif fold_baseline_results[i] and not result["correct"]:
                    fold_worsened += 1

        fold_baseline_acc = np.mean(fold_baseline_results)
        fold_int_acc = fold_int_correct / len(fold_test_data)
        fold_delta = fold_int_acc - fold_baseline_acc

        print(f"  Fold {fold_id + 1}: Baseline={fold_baseline_acc*100:.1f}%, "
              f"Int={fold_int_acc*100:.1f}%, Delta={fold_delta*100:+.1f}% "
              f"({fold_improved}+ {fold_worsened}-)")

        kfold_results.append({
            "fold_id": fold_id,
            "n_questions": len(fold_test_data),
            "baseline_acc": fold_baseline_acc,
            "intervention_acc": fold_int_acc,
            "delta": fold_delta,
            "n_improved": fold_improved,
            "n_worsened": fold_worsened,
        })

    df_kfold = pd.DataFrame(kfold_results)

    total_improved = int(df_kfold["n_improved"].sum())
    total_worsened = int(df_kfold["n_worsened"].sum())
    mean_delta = float(df_kfold["delta"].mean())
    std_delta = float(df_kfold["delta"].std())
    t_crit = t_dist.ppf(0.975, df=n_folds - 1)
    ci_95 = t_crit * std_delta / np.sqrt(n_folds)

    kfold_p = None
    if total_improved + total_worsened > 0:
        kfold_p = binomtest(
            total_improved, total_improved + total_worsened, 0.5, alternative="greater"
        ).pvalue

    n_positive_folds = int((df_kfold["delta"] > 0).sum())

    print(f"\n  Mean delta: {mean_delta*100:+.2f}% +/- {std_delta*100:.2f}%")
    print(f"  95% CI: [{(mean_delta-ci_95)*100:+.2f}%, {(mean_delta+ci_95)*100:+.2f}%]")
    print(f"  Positive folds: {n_positive_folds}/{n_folds}")
    if kfold_p is not None:
        sig = "SIGNIFICANT" if kfold_p < 0.05 else "not significant"
        print(f"  Binomial p-value (one-sided): {kfold_p:.4f} ({sig})")
    print("=" * 80)

    return {
        "n_folds": n_folds,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "ci_95_lower": mean_delta - ci_95,
        "ci_95_upper": mean_delta + ci_95,
        "total_improved": total_improved,
        "total_worsened": total_worsened,
        "binomial_p": float(kfold_p) if kfold_p is not None else None,
        "n_positive_folds": n_positive_folds,
        "fold_details": df_kfold.to_dict("records"),
    }
