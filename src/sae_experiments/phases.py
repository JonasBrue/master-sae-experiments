"""Phase 1-4 orchestration: screening, freq_diff, causal screening, grid search, full reporting."""

from __future__ import annotations

import itertools
import logging
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adaptive_steering import get_strength_fn
from .benchmarks import extract_answer
from .config import AdaptiveConfig, BenchmarkConfig, CausalConfig, GridConfig, ModelConfig
from .fast_inference import (
    batched_logit_predictions,
    batched_intervention_predictions,
    make_approx_intervention_fn,
    make_adaptive_exact_intervention_fn,
    make_adaptive_approx_intervention_fn,
)
from .features import FeatureTracker
from .hooks import HookManager
from .intervention import build_intervention_vector, compute_approximated_hidden_vector
from .prediction import (
    get_logit_prediction,
    get_prediction_with_approximated_intervention,
    get_prediction_with_intervention,
    get_prediction_with_adaptive_intervention,
    get_prediction_with_adaptive_approximated_intervention,
)
from .screening import run_logit_screening, screening_to_sample
from .splitting import matched_pair_split


def _predict_kwargs(
    model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device
) -> Dict:
    """Build the common keyword args for prediction functions."""
    return dict(
        model=model,
        tokenizer=tokenizer,
        config=config,
        hook_mgr=hook_mgr,
        sae=sae,
        answer_token_id_list=answer_token_id_list,
        device=device,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────


def run_screening_and_split(
    all_questions: List[Dict],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    model_config: ModelConfig,
    causal_config: CausalConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    output_dir: str,
    force_rerun: bool = False,
    batch_size: int = 1,
) -> Dict:
    """Run logit screening, filter, and create train/val/test splits.

    Uses Matched-Pair Randomization to guarantee balanced p_correct
    distributions across all splits.

    Returns a dict with keys:
        screening_data, filtered_data, train_data, val_data,
        test_data_filtered, test_data_full,
        grid_eval_data, holdout_data, holdout_filtered
    """
    screening_data, answer_diagnostics = run_logit_screening(
        all_questions,
        model=model,
        tokenizer=tokenizer,
        config=config,
        hook_mgr=hook_mgr,
        sae=sae,
        answer_token_id_list=answer_token_id_list,
        device=device,
        output_dir=output_dir,
        model_short_name=model_config.short_name,
        force_rerun=force_rerun,
        batch_size=batch_size,
    )

    # Filter by P(correct) range
    min_p = causal_config.min_p_correct
    max_p = causal_config.max_p_correct
    filtered_data = [
        s for s in screening_data if min_p <= s["p_correct"] <= max_p
    ]

    total_acc = np.mean([s["correct"] for s in screening_data])
    filtered_acc = np.mean([s["correct"] for s in filtered_data]) if filtered_data else 0

    print(f"\nData overview: {config.name}")
    print(f"  Total questions:    {len(screening_data)}")
    print(f"  Total accuracy:     {total_acc*100:.1f}%")
    print(f"  P(correct) range:   [{min_p:.2f}, {max_p:.2f}]")
    print(f"  Filtered questions: {len(filtered_data)}")
    print(f"  Filtered accuracy:  {filtered_acc*100:.1f}%")

    if len(filtered_data) < 100:
        print(f"\n  Warning: only {len(filtered_data)} questions after filter!")

    # Train/val/test split using Matched-Pair Randomization
    # Ratios derived from config fractions (default 50/25/25)
    train_pct = int(causal_config.train_fraction * 100)
    val_pct = int(causal_config.val_fraction * 100)
    test_pct = 100 - train_pct - val_pct
    train_data, val_data, test_data_filtered = matched_pair_split(
        filtered_data, [train_pct, val_pct, test_pct],
        key="p_correct", seed=causal_config.seed,
    )

    # Full test set: all questions not used in train/val
    used_indices = set(s["q_idx"] for s in train_data) | set(s["q_idx"] for s in val_data)
    test_data_full = [s for s in screening_data if s["q_idx"] not in used_indices]

    # Split test_full into grid_eval (for Phase 3 selection) and holdout (for Phase 4 reporting)
    # using Matched-Pair Randomization to guarantee identical p_correct distributions
    grid_eval_data, holdout_data = matched_pair_split(
        test_data_full, [50, 50],
        key="p_correct", seed=causal_config.seed,
    )

    # Derive holdout_filtered: subset of holdout that falls within the p_correct filter range
    filtered_q_idx_set = {s["q_idx"] for s in filtered_data}
    holdout_filtered = [s for s in holdout_data if s["q_idx"] in filtered_q_idx_set]

    print(f"\n  Train (freq_diff):  {len(train_data)}")
    print(f"  Val (causal test):  {len(val_data)}")
    print(f"  Test (filtered):    {len(test_data_filtered)}")
    print(f"  Test (full):        {len(test_data_full)}")
    print(f"  Grid eval:          {len(grid_eval_data)}")
    print(f"  Holdout:            {len(holdout_data)}")
    print(f"  Holdout (filtered): {len(holdout_filtered)}")

    return {
        "screening_data": screening_data,
        "filtered_data": filtered_data,
        "train_data": train_data,
        "val_data": val_data,
        "test_data_filtered": test_data_filtered,
        "test_data_full": test_data_full,
        "grid_eval_data": grid_eval_data,
        "holdout_data": holdout_data,
        "holdout_filtered": holdout_filtered,
        "answer_diagnostics": answer_diagnostics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: freq_diff calculation
# ─────────────────────────────────────────────────────────────────────────────


def run_phase1_freq_diff(
    train_data: List[Dict],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    causal_config: CausalConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 1,
) -> Tuple[FeatureTracker, List, List]:
    """Phase 1: compute freq_diff on training data and return candidate features."""
    print("=" * 80)
    print("PHASE 1: freq_diff COMPUTATION")
    print("=" * 80)

    tracker = FeatureTracker(sae.d_hidden)

    print(f"  Analyzing {len(train_data)} training questions (batch_size={batch_size})...")

    if batch_size > 1:
        samples = [screening_to_sample(e) for e in train_data]
        results = batched_logit_predictions(
            samples,
            batch_size=batch_size,
            model=model,
            tokenizer=tokenizer,
            config=config,
            hook_mgr=hook_mgr,
            sae=sae,
            answer_token_id_list=answer_token_id_list,
            device=device,
            extract_features=True,
            desc="Feature Tracking",
        )
        for result in results:
            tracker.update(result["active_indices"], result["active_values"], result["correct"])
    else:
        kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)
        for screening_entry in tqdm(train_data, desc="Feature Tracking"):
            sample = screening_to_sample(screening_entry)
            result = get_logit_prediction(sample, **kw, extract_features=True)
            tracker.update(result["active_indices"], result["active_values"], result["correct"])

    n_success = len(tracker.success_acts)
    n_failure = len(tracker.failure_acts)
    train_acc = n_success / (n_success + n_failure) if (n_success + n_failure) > 0 else 0

    print(f"  Successes: {n_success}, Failures: {n_failure}, Accuracy: {train_acc*100:.1f}%")

    dynamic_min_occurrences = max(3, int(len(train_data) * 0.01))
    print(f"  Dynamic min_occurrences: {dynamic_min_occurrences}")

    pos_candidates, neg_candidates = tracker.get_top_features(
        n_positive=causal_config.n_candidate_features_pos,
        n_negative=causal_config.n_candidate_features_neg,
        min_occurrences=dynamic_min_occurrences,
    )

    print(f"  Positive candidates: {len(pos_candidates)}")
    print(f"  Negative candidates: {len(neg_candidates)}")

    for i, (idx, data) in enumerate(pos_candidates[:10]):
        print(f"    {i+1:2d}. Feature {idx:5d}: freq_diff = {data['freq_diff']*100:+.1f}%")

    return tracker, pos_candidates, neg_candidates


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Causal feature screening
# ─────────────────────────────────────────────────────────────────────────────


def run_phase2_causal_screening(
    pos_candidates: List,
    neg_candidates: List,
    val_data: List[Dict],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    causal_config: CausalConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Phase 2: test each candidate feature individually on the validation set.

    Returns (df_pos, df_neg, baseline_val_acc). baseline_val_acc is reused by
    Phase 2.5 (additivity check) to avoid recomputing the val baseline.
    """
    print("=" * 80)
    print("PHASE 2: CAUSAL FEATURE SCREENING")
    print("=" * 80)

    kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)

    # Baseline on val set
    print(f"  Computing baseline on {len(val_data)} val questions (batch_size={batch_size})...")
    if batch_size > 1:
        val_samples = [screening_to_sample(e) for e in val_data]
        baseline_results_list = batched_logit_predictions(
            val_samples,
            batch_size=batch_size,
            model=model,
            tokenizer=tokenizer,
            config=config,
            hook_mgr=hook_mgr,
            sae=sae,
            answer_token_id_list=answer_token_id_list,
            device=device,
            extract_features=False,
            desc="Baseline",
        )
        baseline_val_results = [r["correct"] for r in baseline_results_list]
    else:
        baseline_val_results = []
        for screening_entry in tqdm(val_data, desc="Baseline"):
            sample = screening_to_sample(screening_entry)
            result = get_logit_prediction(sample, **kw, extract_features=False)
            baseline_val_results.append(result["correct"])

    baseline_val_acc = np.mean(baseline_val_results)
    print(f"  Baseline accuracy: {baseline_val_acc*100:.1f}%")

    val_samples = [screening_to_sample(e) for e in val_data] if batch_size > 1 else None

    strengths = causal_config.causal_test_strength
    if strengths is None or len(strengths) == 0:
        raise ValueError(
            "causal_test_strength is empty. ExperimentConfig validator should populate it "
            "from grid.strength — check that you constructed CausalConfig via ExperimentConfig."
        )

    print(f"\n  Testing at strengths: {strengths}")

    pos_results_all: List[Dict] = []
    neg_results_all: List[Dict] = []

    for s in strengths:
        print(f"\n  --- test_strength = {s} ---")
        print(f"  Testing {len(pos_candidates)} positive candidates...")
        pos_results = _test_candidates(
            pos_candidates, val_data, baseline_val_results, baseline_val_acc,
            strength=s,
            direction=1.0, sae=sae, device=device, kw=kw,
            batch_size=batch_size, val_samples=val_samples,
            answer_token_id_list=answer_token_id_list, config=config,
        )
        for r in pos_results:
            r["test_strength"] = s
        pos_results_all.extend(pos_results)

        print(f"  Testing {len(neg_candidates)} negative candidates...")
        neg_results = _test_candidates(
            neg_candidates, val_data, baseline_val_results, baseline_val_acc,
            strength=s,
            direction=-1.0, sae=sae, device=device, kw=kw,
            batch_size=batch_size, val_samples=val_samples,
            answer_token_id_list=answer_token_id_list, config=config,
        )
        for r in neg_results:
            r["test_strength"] = s
        neg_results_all.extend(neg_results)

    df_pos = pd.DataFrame(pos_results_all)
    df_neg = pd.DataFrame(neg_results_all)

    print("\n  Per-strength causal counts:")
    for s in strengths:
        sub_pos = df_pos[df_pos["test_strength"] == s] if len(df_pos) > 0 else df_pos
        sub_neg = df_neg[df_neg["test_strength"] == s] if len(df_neg) > 0 else df_neg
        n_pos = int(sub_pos["is_causal"].sum()) if len(sub_pos) > 0 else 0
        n_neg = int(sub_neg["is_causal"].sum()) if len(sub_neg) > 0 else 0
        print(f"    s={s}: causal positive {n_pos}/{len(sub_pos)}, causal negative {n_neg}/{len(sub_neg)}")
    print("=" * 80)

    return df_pos, df_neg, baseline_val_acc


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2.5: Pairwise combination stability
# ─────────────────────────────────────────────────────────────────────────────


_PAIR_TYPES = ("pos_pos", "neg_neg", "pos_neg")


def _split_pair_budget(n_pairs: int) -> Dict[str, int]:
    """Split the per-strength pair budget across pos_pos / neg_neg / pos_neg.

    Even split with any remainder going to pos_pos (the most-common Phase-3
    configuration usually has top_k_pos > top_k_neg).
    """
    base = n_pairs // 3
    remainder = n_pairs - 3 * base
    return {
        "pos_pos": base + remainder,
        "neg_neg": base,
        "pos_neg": base,
    }


def _sample_pairs(
    pool_a: List[int], pool_b: List[int], n: int, rng: random.Random, same_pool: bool,
) -> List[Tuple[int, int]]:
    """Sample up to n unordered pairs without replacement.

    same_pool=True: draw unordered pairs (i, j) with i != j from a single pool.
    same_pool=False: draw cross pairs (a, b) from two disjoint roles.
    """
    if same_pool:
        if len(pool_a) < 2:
            return []
        candidates = [
            (pool_a[i], pool_a[j])
            for i in range(len(pool_a))
            for j in range(i + 1, len(pool_a))
        ]
    else:
        if len(pool_a) == 0 or len(pool_b) == 0:
            return []
        candidates = [(a, b) for a in pool_a for b in pool_b]
    rng.shuffle(candidates)
    return candidates[: min(n, len(candidates))]


def run_pair_stability_check(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    val_data: List[Dict],
    baseline_val_acc: float,
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
) -> pd.DataFrame:
    """Phase 2.5: pairwise combination stability check.

    For each tested strength, sample feature pairs of three types
    (pos_pos, neg_neg, pos_neg) — using the same sign convention Phase 3 will
    apply (positive features at +strength, negative features at -strength) —
    and evaluate the combined intervention vector on the validation set.

    The diagnostic answers: "when we combine two individually-validated causal
    features, does the combined effect remain positive and at least as good as
    the best of the two alone?" This is weaker than additivity (delta_sum) but
    closer to what matters for Phase 3 / Phase 4 selection.

    Returns a DataFrame with one row per sampled pair, columns include:
      test_strength, pair_type, feature_i, feature_j, sign_i, sign_j,
      delta_i, delta_j, delta_combined,
      delta_best_single, residual_vs_best, remains_positive, retains_best,
      delta_sum_marginals, residual, ratio_combined_over_sum.
    """
    if (not causal_config.combination_stability_check
            or causal_config.combination_stability_n_pairs == 0):
        return pd.DataFrame()

    if (df_pos is None or df_pos.empty or "test_strength" not in df_pos.columns) and \
       (df_neg is None or df_neg.empty or "test_strength" not in df_neg.columns):
        return pd.DataFrame()

    print("=" * 80)
    print("PHASE 2.5: PAIRWISE COMBINATION STABILITY")
    print("=" * 80)

    val_samples = [screening_to_sample(e) for e in val_data] if batch_size > 1 else None
    kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)

    budget = _split_pair_budget(causal_config.combination_stability_n_pairs)
    print(f"  Per-strength pair budget split: {budget}")

    strengths_pos = sorted(df_pos["test_strength"].unique().tolist()) if df_pos is not None and not df_pos.empty and "test_strength" in df_pos.columns else []
    strengths_neg = sorted(df_neg["test_strength"].unique().tolist()) if df_neg is not None and not df_neg.empty and "test_strength" in df_neg.columns else []
    strengths = sorted(set(strengths_pos) | set(strengths_neg))

    results: List[Dict] = []

    for s in strengths:
        # Per-strength RNG so reordering causal_test_strength doesn't change which
        # pairs are sampled at a given strength. Use a deterministic string seed
        # (random.Random rejects tuple seeds since Python 3.11).
        rng = random.Random(f"{causal_config.seed}-{round(float(s), 6):.6f}")

        sub_pos = (
            df_pos[(df_pos["test_strength"] == s) & df_pos["is_causal"]]
            if df_pos is not None and not df_pos.empty else pd.DataFrame()
        )
        sub_neg = (
            df_neg[(df_neg["test_strength"] == s) & df_neg["is_causal"]]
            if df_neg is not None and not df_neg.empty else pd.DataFrame()
        )

        pos_ids = [int(x) for x in sub_pos["feature_id"].tolist()] if len(sub_pos) > 0 else []
        neg_ids = [int(x) for x in sub_neg["feature_id"].tolist()] if len(sub_neg) > 0 else []
        pos_delta = {int(r["feature_id"]): float(r["delta"]) for _, r in sub_pos.iterrows()} if len(sub_pos) > 0 else {}
        neg_delta = {int(r["feature_id"]): float(r["delta"]) for _, r in sub_neg.iterrows()} if len(sub_neg) > 0 else {}

        # Sample pairs of each type. Pool is the full is_causal set per direction
        # (no top-k bias toward the strongest marginal features, which would
        # over-emphasise saturation effects).
        pairs_by_type = {
            "pos_pos": [
                (a, b, +1, +1) for (a, b) in _sample_pairs(pos_ids, [], budget["pos_pos"], rng, same_pool=True)
            ],
            "neg_neg": [
                (a, b, -1, -1) for (a, b) in _sample_pairs(neg_ids, [], budget["neg_neg"], rng, same_pool=True)
            ],
            "pos_neg": [
                (a, b, +1, -1) for (a, b) in _sample_pairs(pos_ids, neg_ids, budget["pos_neg"], rng, same_pool=False)
            ],
        }

        n_pairs_this_strength = sum(len(v) for v in pairs_by_type.values())
        print(
            f"\n  s={s}: pos_causal={len(pos_ids)} neg_causal={len(neg_ids)} "
            f"-> sampled pos_pos={len(pairs_by_type['pos_pos'])}, "
            f"neg_neg={len(pairs_by_type['neg_neg'])}, pos_neg={len(pairs_by_type['pos_neg'])} "
            f"(total {n_pairs_this_strength})"
        )

        for pair_type, pair_list in pairs_by_type.items():
            for pair_idx, (f_i, f_j, sign_i, sign_j) in enumerate(pair_list):
                d_i = (pos_delta if sign_i > 0 else neg_delta)[f_i]
                d_j = (pos_delta if sign_j > 0 else neg_delta)[f_j]

                vec = torch.zeros(sae.d_hidden, device=device, dtype=sae.dtype)
                vec[f_i] = sign_i * s
                vec[f_j] = sign_j * s

                if batch_size > 1 and val_samples is not None:
                    int_results = batched_intervention_predictions(
                        val_samples, vec,
                        batch_size=batch_size, **kw,
                        desc=f"s={s} {pair_type} {pair_idx+1}/{len(pair_list)}",
                    )
                    int_correct = sum(r["correct"] for r in int_results)
                else:
                    int_correct = 0
                    for screening_entry in val_data:
                        sample = screening_to_sample(screening_entry)
                        r = get_prediction_with_intervention(sample, vec, **kw)
                        int_correct += r["correct"]

                int_acc = int_correct / len(val_data)
                delta_combined = int_acc - baseline_val_acc

                # Stability metrics (primary)
                delta_best_single = max(d_i, d_j)
                residual_vs_best = delta_combined - delta_best_single
                remains_positive = bool(delta_combined > 0)
                retains_best = bool(delta_combined >= delta_best_single - 1e-9)

                # Additivity metrics (secondary diagnostic, retained for reference)
                delta_sum = d_i + d_j
                residual_vs_sum = delta_combined - delta_sum
                ratio_vs_sum = delta_combined / delta_sum if abs(delta_sum) > 1e-9 else float("nan")

                results.append({
                    "test_strength": s,
                    "pair_type": pair_type,
                    "feature_i": f_i,
                    "feature_j": f_j,
                    "sign_i": sign_i,
                    "sign_j": sign_j,
                    "delta_i": d_i,
                    "delta_j": d_j,
                    "delta_combined": delta_combined,
                    "delta_best_single": delta_best_single,
                    "residual_vs_best": residual_vs_best,
                    "remains_positive": remains_positive,
                    "retains_best": retains_best,
                    "delta_sum_marginals": delta_sum,
                    "residual": residual_vs_sum,
                    "ratio_combined_over_sum": ratio_vs_sum,
                })

    df_stab = pd.DataFrame(results)
    if not df_stab.empty:
        print("\n  Stability summary per (strength, pair_type):")
        for (s, pt), grp in df_stab.groupby(["test_strength", "pair_type"]):
            print(
                f"    s={s} {pt}: n={len(grp)}, "
                f"remains_positive={grp['remains_positive'].mean()*100:.0f}%, "
                f"retains_best={grp['retains_best'].mean()*100:.0f}%, "
                f"mean residual_vs_best={grp['residual_vs_best'].mean():+.4f}, "
                f"mean delta_combined={grp['delta_combined'].mean():+.4f}"
            )
    print("=" * 80)
    return df_stab


def _test_candidates(
    candidates, val_data, baseline_results, baseline_acc,
    strength, direction, sae, device, kw,
    batch_size=1, val_samples=None,
    answer_token_id_list=None, config=None,
) -> List[Dict]:
    results = []
    desc = "Positive Features" if direction > 0 else "Negative Features"

    if batch_size > 1 and val_samples is not None:
        # Batched candidate testing
        for idx, data in tqdm(candidates, desc=desc):
            intervention_vec = torch.zeros(sae.d_hidden, device=device, dtype=sae.dtype)
            intervention_vec[idx] = direction * strength

            int_results = batched_intervention_predictions(
                val_samples, intervention_vec,
                batch_size=batch_size, **kw,
                desc=f"Feature {idx}",
            )

            int_correct = sum(r["correct"] for r in int_results)
            n_improved = sum(
                1 for i, r in enumerate(int_results)
                if not baseline_results[i] and r["correct"]
            )
            n_worsened = sum(
                1 for i, r in enumerate(int_results)
                if baseline_results[i] and not r["correct"]
            )

            int_acc = int_correct / len(val_data)
            delta = int_acc - baseline_acc

            results.append({
                "feature_id": idx,
                "freq_diff": data["freq_diff"],
                "baseline_acc": baseline_acc,
                "intervention_acc": int_acc,
                "delta": delta,
                "n_improved": n_improved,
                "n_worsened": n_worsened,
                "net_change": n_improved - n_worsened,
                "is_causal": delta > 0,
            })
    else:
        for idx, data in tqdm(candidates, desc=desc):
            intervention_vec = torch.zeros(sae.d_hidden, device=device, dtype=sae.dtype)
            intervention_vec[idx] = direction * strength

            int_correct = 0
            n_improved = 0
            n_worsened = 0

            for i, screening_entry in enumerate(val_data):
                sample = screening_to_sample(screening_entry)
                int_result = get_prediction_with_intervention(sample, intervention_vec, **kw)
                int_correct += int_result["correct"]

                if not baseline_results[i] and int_result["correct"]:
                    n_improved += 1
                elif baseline_results[i] and not int_result["correct"]:
                    n_worsened += 1

            int_acc = int_correct / len(val_data)
            delta = int_acc - baseline_acc

            results.append({
                "feature_id": idx,
                "freq_diff": data["freq_diff"],
                "baseline_acc": baseline_acc,
                "intervention_acc": int_acc,
                "delta": delta,
                "n_improved": n_improved,
                "n_worsened": n_worsened,
                "net_change": n_improved - n_worsened,
                "is_causal": delta > 0,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Grid search
# ─────────────────────────────────────────────────────────────────────────────


def _select_features_for_ranking(
    df: pd.DataFrame, ranking: str, strength: float
) -> pd.DataFrame:
    """Pick the feature rows that should be ranked for this (ranking, strength) cell.

    For causal_delta: restrict to rows whose test_strength matches the applied
    strength (so the delta-score used for ranking was measured at that strength),
    then keep only is_causal-survivors. Strengths are normalised at config
    construction (rounded to 6 decimals), so exact float-equality matching is
    safe here.

    For freq_diff: strength is irrelevant — dedupe to one row per feature_id so
    nlargest doesn't pick the same feature multiple times when df has multiple
    test_strength entries per feature.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if ranking == "causal_delta":
        if "test_strength" in df.columns and df["test_strength"].nunique() > 1:
            sub = df[df["test_strength"] == round(float(strength), 6)]
        else:
            # Single-strength mode (or no column): the only available Phase-2
            # ranking is reused for every grid.strength cell.
            sub = df
        if "is_causal" in sub.columns:
            return sub[sub["is_causal"]].copy()
        return sub.copy()
    if ranking == "freq_diff":
        return df.drop_duplicates(subset="feature_id").copy()
    raise ValueError(
        f"Unknown ranking_criterion: {ranking!r}. "
        f"Expected 'causal_delta' or 'freq_diff'."
    )


def run_phase3_grid_search(
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    grid_eval_data: List[Dict],
    baseline_grid_eval_results: List[Dict],
    baseline_grid_eval_acc: float,
    *,
    grid_config: GridConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 1,
) -> Optional[pd.DataFrame]:
    """Phase 3: grid search evaluated on grid_eval set (disjoint from holdout)."""
    print("=" * 80)
    print("PHASE 3: GRID SEARCH")
    print("=" * 80)

    combinations = list(itertools.product(
        grid_config.top_k_positive,
        grid_config.top_k_negative,
        grid_config.strength,
        grid_config.ranking_criterion,
        grid_config.weighting_mode,
    ))

    print(f"  Total combinations: {len(combinations)}")
    print(f"  Grid eval set: {len(grid_eval_data)} questions")

    # Graceful handling: Wenn keine Feature-Kandidaten gefunden wurden
    if df_pos.empty and df_neg.empty:
        warnings.warn("No feature candidates found in Phase 1/2. Skipping grid search.")
        logger.warning("No feature candidates found in Phase 1/2. Skipping grid search.")
        return None

    baseline_dict = {r["q_idx"]: r["correct"] for r in baseline_grid_eval_results}

    eval_samples = [screening_to_sample(e) for e in grid_eval_data] if batch_size > 1 else None
    kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)
    grid_results = []
    start_time = datetime.now()

    for run_idx, (top_k_pos, top_k_neg, strength, ranking, weighting) in enumerate(
        tqdm(combinations, desc="Grid Search")
    ):
        pos_features_to_use = _select_features_for_ranking(df_pos, ranking, strength)
        neg_features_to_use = _select_features_for_ranking(df_neg, ranking, strength)

        vec = build_intervention_vector(
            pos_features=pos_features_to_use,
            neg_features=neg_features_to_use,
            top_k_pos=top_k_pos,
            top_k_neg=top_k_neg,
            strength=strength,
            ranking_criterion=ranking,
            weighting_mode=weighting,
            d_hidden=sae.d_hidden,
            device=device,
            dtype=sae.dtype,
        )

        n_features_used = (vec != 0).sum().item()

        # Evaluate on grid_eval set
        if batch_size > 1 and eval_samples is not None:
            int_results = batched_intervention_predictions(
                eval_samples, vec,
                batch_size=batch_size, **kw,
                desc=f"Grid {run_idx+1}/{len(combinations)}",
            )
        else:
            int_results = []
            for screening_entry in grid_eval_data:
                sample = screening_to_sample(screening_entry)
                result = get_prediction_with_intervention(sample, vec, **kw)
                int_results.append(result)

        # Compute metrics on grid_eval set
        correct = 0
        improved = 0
        worsened = 0

        for entry, r in zip(grid_eval_data, int_results):
            q_idx = entry["q_idx"]
            baseline_correct = baseline_dict[q_idx]
            is_correct = r["correct"]

            correct += is_correct
            if not baseline_correct and is_correct:
                improved += 1
            elif baseline_correct and not is_correct:
                worsened += 1

        acc = correct / len(grid_eval_data)
        delta = acc - baseline_grid_eval_acc

        grid_results.append({
            "top_k_pos": top_k_pos,
            "top_k_neg": top_k_neg,
            "strength": strength,
            "ranking_criterion": ranking,
            "weighting_mode": weighting,
            "n_features_used": n_features_used,
            "accuracy_grid_eval": acc,
            "delta_grid_eval": delta,
            "n_improved": improved,
            "n_worsened": worsened,
            "net_change": improved - worsened,
        })

        if (run_idx + 1) % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            print(f"  [{run_idx+1}/{len(combinations)}] Elapsed: {elapsed:.1f} min")

    print("Grid search complete")
    print("=" * 80)
    return pd.DataFrame(grid_results)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Full reporting
# ─────────────────────────────────────────────────────────────────────────────


def run_phase4_full_reporting(
    best_config: Dict,
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    holdout_filtered: List[Dict],
    holdout_data: List[Dict],
    baseline_holdout_filtered_results: List[Dict],
    baseline_holdout_filtered_acc: float,
    baseline_holdout_results: List[Dict],
    baseline_holdout_acc: float,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    adaptive_config: Optional[AdaptiveConfig] = None,
    batch_size: int = 1,
) -> Dict:
    """Phase 4: full reporting on holdout data (never seen by grid search).

    Evaluates the best config on holdout_filtered and holdout_data (the full
    holdout set including out-of-range questions). If adaptive_config is
    provided and enabled, also runs adaptive variants.

    Returns dict with:
        best_intervention_vec, approximated_hidden_vec,
        full_reporting_results, difficulty_analysis, adaptive_results (if applicable)
    """
    print("=" * 80)
    print("PHASE 4: FULL REPORTING")
    if adaptive_config and adaptive_config.enabled:
        print("  [Adaptive steering ENABLED]")
    print("=" * 80)

    BEST_K_POS = best_config["top_k_pos"]
    BEST_K_NEG = best_config["top_k_neg"]
    BEST_STRENGTH = best_config["strength"]
    BEST_RANKING = best_config["ranking_criterion"]
    BEST_WEIGHTING = best_config["weighting_mode"]

    # Feature selection: matches Phase-3 cell semantics (causal_delta uses
    # rows whose test_strength == BEST_STRENGTH; freq_diff dedupes per feature)
    pos_features_final = _select_features_for_ranking(df_pos, BEST_RANKING, BEST_STRENGTH)
    neg_features_final = _select_features_for_ranking(df_neg, BEST_RANKING, BEST_STRENGTH)

    # Build exact intervention vector
    best_intervention_vec = build_intervention_vector(
        pos_features=pos_features_final,
        neg_features=neg_features_final,
        top_k_pos=BEST_K_POS,
        top_k_neg=BEST_K_NEG,
        strength=BEST_STRENGTH,
        ranking_criterion=BEST_RANKING,
        weighting_mode=BEST_WEIGHTING,
        d_hidden=sae.d_hidden,
        device=device,
        dtype=sae.dtype,
    )

    n_features_in_vec = (best_intervention_vec != 0).sum().item()
    print(f"  Exact intervention vector: {n_features_in_vec} features")

    # Build approximated hidden vector
    approximated_hidden_vec = compute_approximated_hidden_vector(best_intervention_vec, sae)
    print(f"  Approximated hidden vector norm: {approximated_hidden_vec.norm().item():.4f}")

    kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)
    baseline_holdout_dict = {r["q_idx"]: r["correct"] for r in baseline_holdout_results}

    full_reporting_results = {}

    bkw = dict(
        batch_size=batch_size, model=model, tokenizer=tokenizer, config=config,
        hook_mgr=hook_mgr, sae=sae, answer_token_id_list=answer_token_id_list, device=device,
    )

    if batch_size > 1:
        filtered_samples = [screening_to_sample(e) for e in holdout_filtered]
        full_samples = [screening_to_sample(e) for e in holdout_data]
        approx_fn = make_approx_intervention_fn(approximated_hidden_vec)

        # Report 1: Filtered + Exact
        r1 = _evaluate_report_batched(
            "Filtered + Exact", filtered_samples, best_intervention_vec,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_exact"] = r1

        # Report 2: Full + Exact
        r2 = _evaluate_report_full_batched(
            "Full + Exact", full_samples, best_intervention_vec,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_exact"] = r2

        # Report 3: Filtered + Approximated
        r3 = _evaluate_report_batched(
            "Filtered + Approximated", filtered_samples, approx_fn,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_approx"] = r3

        # Report 4: Full + Approximated
        r4 = _evaluate_report_full_batched(
            "Full + Approximated", full_samples, approx_fn,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_approx"] = r4

        difficulty_analysis = _run_difficulty_analysis_batched(
            holdout_data, baseline_holdout_dict, full_samples, best_intervention_vec, **bkw,
        )
    else:
        # Report 1: Filtered + Exact
        r1 = _evaluate_report(
            "Filtered + Exact", holdout_filtered, baseline_holdout_filtered_results,
            baseline_holdout_filtered_acc, best_intervention_vec, "exact", kw,
            approximated_hidden_vec=None,
        )
        full_reporting_results["filtered_exact"] = r1

        # Report 2: Full + Exact
        r2 = _evaluate_report_full(
            "Full + Exact", holdout_data, baseline_holdout_dict,
            baseline_holdout_acc, best_intervention_vec, "exact", kw,
            approximated_hidden_vec=None,
        )
        full_reporting_results["full_exact"] = r2

        # Report 3: Filtered + Approximated
        r3 = _evaluate_report(
            "Filtered + Approximated", holdout_filtered, baseline_holdout_filtered_results,
            baseline_holdout_filtered_acc, best_intervention_vec, "approx", kw,
            approximated_hidden_vec=approximated_hidden_vec,
        )
        full_reporting_results["filtered_approx"] = r3

        # Report 4: Full + Approximated
        r4 = _evaluate_report_full(
            "Full + Approximated", holdout_data, baseline_holdout_dict,
            baseline_holdout_acc, best_intervention_vec, "approx", kw,
            approximated_hidden_vec=approximated_hidden_vec,
        )
        full_reporting_results["full_approx"] = r4

        # Difficulty analysis
        difficulty_analysis = _run_difficulty_analysis(
            holdout_data, baseline_holdout_dict, best_intervention_vec, kw,
        )

    # Print summary table
    print(f"\n{'=' * 80}")
    print("FULL REPORTING SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Report':<30} {'N':>6} {'Baseline':>10} {'Interv.':>10} {'Delta':>8} {'p-value':>10}")
    print(f"{'─' * 80}")

    for key, r in full_reporting_results.items():
        p_str = f"{r['binomial_p']:.4f}" if r["binomial_p"] is not None else "N/A"
        print(f"{r['name']:<30} {r['n_questions']:>6} {r['baseline_acc']*100:>9.1f}% "
              f"{r['intervention_acc']*100:>9.1f}% {r['delta']*100:>+7.1f}% {p_str:>10}")

    return {
        "best_intervention_vec": best_intervention_vec,
        "approximated_hidden_vec": approximated_hidden_vec,
        "n_features_in_vec": n_features_in_vec,
        "full_reporting_results": full_reporting_results,
        "difficulty_analysis": difficulty_analysis,
    }


def _evaluate_report(
    name, test_data, baseline_results, baseline_acc,
    intervention_vec, mode, kw, approximated_hidden_vec,
) -> Dict:
    """Evaluate a single report on indexed test data."""
    print(f"\n  {name}...")
    correct, improved, worsened = 0, 0, 0

    for i, screening_entry in enumerate(tqdm(test_data, desc=name)):
        sample = screening_to_sample(screening_entry)
        if mode == "exact":
            result = get_prediction_with_intervention(sample, intervention_vec, **kw)
        else:
            result = get_prediction_with_approximated_intervention(
                sample, approximated_hidden_vec, **kw,
            )
        correct += result["correct"]

        if not baseline_results[i]["correct"] and result["correct"]:
            improved += 1
        elif baseline_results[i]["correct"] and not result["correct"]:
            worsened += 1

    acc = correct / len(test_data)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(test_data),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _evaluate_report_full(
    name, test_data, baseline_dict, baseline_acc,
    intervention_vec, mode, kw, approximated_hidden_vec,
) -> Dict:
    """Evaluate a single report on full test data (using baseline dict lookup)."""
    print(f"\n  {name}...")
    correct, improved, worsened = 0, 0, 0

    for screening_entry in tqdm(test_data, desc=name):
        sample = screening_to_sample(screening_entry)
        if mode == "exact":
            result = get_prediction_with_intervention(sample, intervention_vec, **kw)
        else:
            result = get_prediction_with_approximated_intervention(
                sample, approximated_hidden_vec, **kw,
            )
        correct += result["correct"]

        baseline_correct = baseline_dict[screening_entry["q_idx"]]
        if not baseline_correct and result["correct"]:
            improved += 1
        elif baseline_correct and not result["correct"]:
            worsened += 1

    acc = correct / len(test_data)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(test_data),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _evaluate_report_batched(
    name, samples, intervention, baseline_results, baseline_acc, *,
    batch_size, model, tokenizer, config, hook_mgr, sae,
    answer_token_id_list, device,
) -> Dict:
    """Evaluate a report using batched intervention (indexed baseline_results)."""
    print(f"\n  {name}...")
    int_results = batched_intervention_predictions(
        samples, intervention,
        batch_size=batch_size, model=model, tokenizer=tokenizer, config=config,
        hook_mgr=hook_mgr, sae=sae, answer_token_id_list=answer_token_id_list,
        device=device, desc=name,
    )

    correct = sum(r["correct"] for r in int_results)
    improved = sum(
        1 for i, r in enumerate(int_results)
        if not baseline_results[i]["correct"] and r["correct"]
    )
    worsened = sum(
        1 for i, r in enumerate(int_results)
        if baseline_results[i]["correct"] and not r["correct"]
    )

    acc = correct / len(int_results)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(int_results),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _evaluate_report_full_batched(
    name, samples, intervention, test_data, baseline_dict, baseline_acc, *,
    batch_size, model, tokenizer, config, hook_mgr, sae,
    answer_token_id_list, device,
) -> Dict:
    """Evaluate a report using batched intervention (dict-based baseline lookup)."""
    print(f"\n  {name}...")
    int_results = batched_intervention_predictions(
        samples, intervention,
        batch_size=batch_size, model=model, tokenizer=tokenizer, config=config,
        hook_mgr=hook_mgr, sae=sae, answer_token_id_list=answer_token_id_list,
        device=device, desc=name,
    )

    correct = sum(r["correct"] for r in int_results)
    improved = 0
    worsened = 0
    for screening_entry, r in zip(test_data, int_results):
        baseline_correct = baseline_dict[screening_entry["q_idx"]]
        if not baseline_correct and r["correct"]:
            improved += 1
        elif baseline_correct and not r["correct"]:
            worsened += 1

    acc = correct / len(int_results)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(int_results),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _run_difficulty_analysis_batched(
    test_data_full, baseline_full_dict, samples, intervention, *,
    batch_size, model, tokenizer, config, hook_mgr, sae,
    answer_token_id_list, device,
) -> Dict:
    """Analyze intervention effect across difficulty ranges using batched inference."""
    print("\n  Running difficulty analysis...")

    int_results = batched_intervention_predictions(
        samples, intervention,
        batch_size=batch_size, model=model, tokenizer=tokenizer, config=config,
        hook_mgr=hook_mgr, sae=sae, answer_token_id_list=answer_token_id_list,
        device=device, desc="Difficulty Analysis",
    )

    detailed_results = []
    for screening_entry, r in zip(test_data_full, int_results):
        baseline_correct = baseline_full_dict[screening_entry["q_idx"]]
        detailed_results.append({
            "q_idx": screening_entry["q_idx"],
            "p_correct": screening_entry["p_correct"],
            "baseline_correct": baseline_correct,
            "intervention_correct": r["correct"],
            "improved": not baseline_correct and r["correct"],
            "worsened": baseline_correct and not r["correct"],
        })

    df = pd.DataFrame(detailed_results)

    difficulty_ranges = [
        ("Very hard", 0.00, 0.15),
        ("Hard", 0.15, 0.35),
        ("Medium", 0.35, 0.65),
        ("Easy", 0.65, 0.85),
        ("Very easy", 0.85, 1.01),
    ]

    analysis = []
    for range_name, low, high in difficulty_ranges:
        subset = df[(df["p_correct"] >= low) & (df["p_correct"] < high)]
        if len(subset) == 0:
            continue
        analysis.append({
            "range_name": range_name,
            "range_low": low,
            "range_high": high,
            "n_questions": len(subset),
            "baseline_acc": float(subset["baseline_correct"].mean()),
            "intervention_acc": float(subset["intervention_correct"].mean()),
            "delta": float(subset["intervention_correct"].mean() - subset["baseline_correct"].mean()),
            "n_improved": int(subset["improved"].sum()),
            "n_worsened": int(subset["worsened"].sum()),
        })

    totals = {
        "n_questions": len(df),
        "baseline_acc": float(df["baseline_correct"].mean()),
        "intervention_acc": float(df["intervention_correct"].mean()),
        "delta": float(df["intervention_correct"].mean() - df["baseline_correct"].mean()),
        "n_improved": int(df["improved"].sum()),
        "n_worsened": int(df["worsened"].sum()),
    }

    return {"difficulty_ranges": analysis, "totals": totals}


def compute_baselines(
    eval_data: List[Dict],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    batch_size: int = 1,
    label: str = "eval",
    filtered_subset: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], float, Optional[List[Dict]], Optional[float]]:
    """Compute baseline predictions on eval_data; optionally derive filtered subset.

    Args:
        eval_data: Questions to evaluate.
        filtered_subset: If given, derive filtered results from eval results
            (must be a subset of eval_data by q_idx).
        label: Label for progress bar and log output.

    Returns:
        (eval_results, eval_acc, filtered_results, filtered_acc)
        filtered_results/filtered_acc are None if filtered_subset is None.
    """
    print(f"Computing baselines for {label} (batch_size={batch_size})...")

    if batch_size > 1:
        samples = [screening_to_sample(e) for e in eval_data]
        raw_results = batched_logit_predictions(
            samples,
            batch_size=batch_size,
            model=model, tokenizer=tokenizer, config=config,
            hook_mgr=hook_mgr, sae=sae,
            answer_token_id_list=answer_token_id_list, device=device,
            extract_features=False, desc=f"Baseline ({label})",
        )
        eval_results = [
            {"q_idx": e["q_idx"], "correct": r["correct"], "prob_correct": r["prob_correct"]}
            for e, r in zip(eval_data, raw_results)
        ]
    else:
        kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)
        eval_results = []
        for screening_entry in tqdm(eval_data, desc=f"Baseline ({label})"):
            sample = screening_to_sample(screening_entry)
            result = get_logit_prediction(sample, **kw, extract_features=False)
            eval_results.append({
                "q_idx": screening_entry["q_idx"],
                "correct": result["correct"],
                "prob_correct": result["prob_correct"],
            })

    eval_acc = np.mean([r["correct"] for r in eval_results])
    print(f"  {label}: {eval_acc*100:.1f}% ({len(eval_data)} questions)")

    # Optionally derive filtered baseline
    filtered_results = None
    filtered_acc = None
    if filtered_subset is not None:
        results_by_qidx = {r["q_idx"]: r for r in eval_results}
        filtered_results = [results_by_qidx[e["q_idx"]] for e in filtered_subset]
        filtered_acc = np.mean([r["correct"] for r in filtered_results])
        print(f"  {label} (filtered): {filtered_acc*100:.1f}% ({len(filtered_subset)} questions)")

    return eval_results, eval_acc, filtered_results, filtered_acc


def _run_difficulty_analysis(
    test_data_full, baseline_full_dict, best_intervention_vec, kw,
) -> Dict:
    """Analyze intervention effect across difficulty ranges."""
    print("\n  Running difficulty analysis...")

    detailed_results = []
    for screening_entry in tqdm(test_data_full, desc="Difficulty Analysis"):
        sample = screening_to_sample(screening_entry)
        result = get_prediction_with_intervention(sample, best_intervention_vec, **kw)

        baseline_correct = baseline_full_dict[screening_entry["q_idx"]]
        detailed_results.append({
            "q_idx": screening_entry["q_idx"],
            "p_correct": screening_entry["p_correct"],
            "baseline_correct": baseline_correct,
            "intervention_correct": result["correct"],
            "improved": not baseline_correct and result["correct"],
            "worsened": baseline_correct and not result["correct"],
        })

    df = pd.DataFrame(detailed_results)

    difficulty_ranges = [
        ("Very hard", 0.00, 0.15),
        ("Hard", 0.15, 0.35),
        ("Medium", 0.35, 0.65),
        ("Easy", 0.65, 0.85),
        ("Very easy", 0.85, 1.01),
    ]

    analysis = []
    for name, low, high in difficulty_ranges:
        subset = df[(df["p_correct"] >= low) & (df["p_correct"] < high)]
        if len(subset) == 0:
            continue

        analysis.append({
            "range_name": name,
            "range_low": low,
            "range_high": high,
            "n_questions": len(subset),
            "baseline_acc": float(subset["baseline_correct"].mean()),
            "intervention_acc": float(subset["intervention_correct"].mean()),
            "delta": float(subset["intervention_correct"].mean() - subset["baseline_correct"].mean()),
            "n_improved": int(subset["improved"].sum()),
            "n_worsened": int(subset["worsened"].sum()),
        })

    totals = {
        "n_questions": len(df),
        "baseline_acc": float(df["baseline_correct"].mean()),
        "intervention_acc": float(df["intervention_correct"].mean()),
        "delta": float(df["intervention_correct"].mean() - df["baseline_correct"].mean()),
        "n_improved": int(df["improved"].sum()),
        "n_worsened": int(df["worsened"].sum()),
    }

    return {"difficulty_ranges": analysis, "totals": totals}


# ════════════════════════════════════════════════════════════════════════════════
# Adaptive intervention evaluation
# ════════════════════════════════════════════════════════════════════════════════


def run_phase4_adaptive_reporting(
    best_config: Dict,
    df_pos: pd.DataFrame,
    df_neg: pd.DataFrame,
    holdout_filtered: List[Dict],
    holdout_data: List[Dict],
    baseline_holdout_filtered_results: List[Dict],
    baseline_holdout_filtered_acc: float,
    baseline_holdout_results: List[Dict],
    baseline_holdout_acc: float,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    adaptive_config: AdaptiveConfig,
    batch_size: int = 1,
) -> Dict:
    """Phase 4 with adaptive steering: evaluates both standard and adaptive interventions.

    This function runs the same 4 evaluation scenarios as run_phase4_full_reporting,
    but with adaptive strength that adjusts per-question based on p_correct.

    Returns dict with:
        best_intervention_vec, approximated_hidden_vec,
        full_reporting_results, adaptive_results, difficulty_analysis
    """
    print("=" * 80)
    print("PHASE 4: FULL REPORTING (with Adaptive Steering)")
    print(f"  Mode: {adaptive_config.mode}")
    print(f"  p_min={adaptive_config.p_min}, p_sweet={adaptive_config.p_sweet}, p_max={adaptive_config.p_max}")
    print("=" * 80)

    BEST_K_POS = best_config["top_k_pos"]
    BEST_K_NEG = best_config["top_k_neg"]
    BEST_STRENGTH = best_config["strength"]
    BEST_RANKING = best_config["ranking_criterion"]
    BEST_WEIGHTING = best_config["weighting_mode"]

    # Feature selection: matches Phase-3 cell semantics (causal_delta uses
    # rows whose test_strength == BEST_STRENGTH; freq_diff dedupes per feature)
    pos_features_final = _select_features_for_ranking(df_pos, BEST_RANKING, BEST_STRENGTH)
    neg_features_final = _select_features_for_ranking(df_neg, BEST_RANKING, BEST_STRENGTH)

    # Build exact intervention vector
    best_intervention_vec = build_intervention_vector(
        pos_features=pos_features_final,
        neg_features=neg_features_final,
        top_k_pos=BEST_K_POS,
        top_k_neg=BEST_K_NEG,
        strength=BEST_STRENGTH,
        ranking_criterion=BEST_RANKING,
        weighting_mode=BEST_WEIGHTING,
        d_hidden=sae.d_hidden,
        device=device,
        dtype=sae.dtype,
    )

    n_features_in_vec = (best_intervention_vec != 0).sum().item()
    print(f"  Exact intervention vector: {n_features_in_vec} features")

    # Build approximated hidden vector
    approximated_hidden_vec = compute_approximated_hidden_vector(best_intervention_vec, sae)
    print(f"  Approximated hidden vector norm: {approximated_hidden_vec.norm().item():.4f}")

    kw = _predict_kwargs(model, tokenizer, config, hook_mgr, sae, answer_token_id_list, device)
    baseline_holdout_dict = {r["q_idx"]: r["correct"] for r in baseline_holdout_results}

    full_reporting_results = {}

    bkw = dict(
        batch_size=batch_size, model=model, tokenizer=tokenizer, config=config,
        hook_mgr=hook_mgr, sae=sae, answer_token_id_list=answer_token_id_list, device=device,
    )

    if batch_size > 1:
        filtered_samples = [screening_to_sample(e) for e in holdout_filtered]
        full_samples = [screening_to_sample(e) for e in holdout_data]

        strength_fn = get_strength_fn(
            adaptive_config.mode,
            p_min=adaptive_config.p_min, p_sweet=adaptive_config.p_sweet,
            p_max=adaptive_config.p_max, mu=adaptive_config.mu, sigma=adaptive_config.sigma,
            transition_strength=adaptive_config.transition_strength,
        )

        metric_key = adaptive_config.difficulty_metric  # "p_correct" or "max_confidence"
        filtered_p_correct = [e[metric_key] for e in holdout_filtered]
        full_p_correct = [e[metric_key] for e in holdout_data]

        # Static reports
        r1 = _evaluate_report_batched(
            "Filtered + Exact (static)", filtered_samples, best_intervention_vec,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_exact_static"] = r1

        r2 = _evaluate_report_full_batched(
            "Full + Exact (static)", full_samples, best_intervention_vec,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_exact_static"] = r2

        # Adaptive exact reports
        adaptive_exact_fn_filtered = make_adaptive_exact_intervention_fn(
            sae, best_intervention_vec, strength_fn, filtered_p_correct,
        )
        r3 = _evaluate_report_batched(
            "Filtered + Exact (adaptive)", filtered_samples, adaptive_exact_fn_filtered,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_exact_adaptive"] = r3

        adaptive_exact_fn_full = make_adaptive_exact_intervention_fn(
            sae, best_intervention_vec, strength_fn, full_p_correct,
        )
        r4 = _evaluate_report_full_batched(
            "Full + Exact (adaptive)", full_samples, adaptive_exact_fn_full,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_exact_adaptive"] = r4

        # Adaptive approximated reports
        adaptive_approx_fn_filtered = make_adaptive_approx_intervention_fn(
            approximated_hidden_vec, strength_fn, filtered_p_correct,
        )
        r5 = _evaluate_report_batched(
            "Filtered + Approximated (adaptive)", filtered_samples, adaptive_approx_fn_filtered,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_approx_adaptive"] = r5

        adaptive_approx_fn_full = make_adaptive_approx_intervention_fn(
            approximated_hidden_vec, strength_fn, full_p_correct,
        )
        r6 = _evaluate_report_full_batched(
            "Full + Approximated (adaptive)", full_samples, adaptive_approx_fn_full,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_approx_adaptive"] = r6

        difficulty_analysis = _run_difficulty_analysis_batched(
            holdout_data, baseline_holdout_dict, full_samples, adaptive_exact_fn_full, **bkw,
        )
    else:
        # Standard (non-adaptive) reports
        r1 = _evaluate_report(
            "Filtered + Exact (static)", holdout_filtered, baseline_holdout_filtered_results,
            baseline_holdout_filtered_acc, best_intervention_vec, "exact", kw,
            approximated_hidden_vec=None,
        )
        full_reporting_results["filtered_exact_static"] = r1

        r2 = _evaluate_report_full(
            "Full + Exact (static)", holdout_data, baseline_holdout_dict,
            baseline_holdout_acc, best_intervention_vec, "exact", kw,
            approximated_hidden_vec=None,
        )
        full_reporting_results["full_exact_static"] = r2

        # Adaptive reports
        r3 = _evaluate_report_adaptive(
            "Filtered + Exact (adaptive)", holdout_filtered, baseline_holdout_filtered_results,
            baseline_holdout_filtered_acc, best_intervention_vec, kw, adaptive_config,
        )
        full_reporting_results["filtered_exact_adaptive"] = r3

        r4 = _evaluate_report_full_adaptive(
            "Full + Exact (adaptive)", holdout_data, baseline_holdout_dict,
            baseline_holdout_acc, best_intervention_vec, kw, adaptive_config,
        )
        full_reporting_results["full_exact_adaptive"] = r4

        # Adaptive approximated
        r5 = _evaluate_report_adaptive_approx(
            "Filtered + Approximated (adaptive)", holdout_filtered, baseline_holdout_filtered_results,
            baseline_holdout_filtered_acc, approximated_hidden_vec, kw, adaptive_config,
        )
        full_reporting_results["filtered_approx_adaptive"] = r5

        r6 = _evaluate_report_full_adaptive_approx(
            "Full + Approximated (adaptive)", holdout_data, baseline_holdout_dict,
            baseline_holdout_acc, approximated_hidden_vec, kw, adaptive_config,
        )
        full_reporting_results["full_approx_adaptive"] = r6

        # Difficulty analysis with adaptive intervention
        difficulty_analysis = _run_difficulty_analysis_adaptive(
            holdout_data, baseline_holdout_dict, best_intervention_vec, kw, adaptive_config,
        )

    # Print summary table
    print(f"\n{'=' * 80}")
    print("FULL REPORTING SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Report':<35} {'N':>6} {'Baseline':>10} {'Interv.':>10} {'Delta':>8} {'p-value':>10}")
    print(f"{'─' * 80}")

    for key, r in full_reporting_results.items():
        p_str = f"{r['binomial_p']:.4f}" if r["binomial_p"] is not None else "N/A"
        print(f"{r['name']:<35} {r['n_questions']:>6} {r['baseline_acc']*100:>9.1f}% "
              f"{r['intervention_acc']*100:>9.1f}% {r['delta']*100:>+7.1f}% {p_str:>10}")

    return {
        "best_intervention_vec": best_intervention_vec,
        "approximated_hidden_vec": approximated_hidden_vec,
        "n_features_in_vec": n_features_in_vec,
        "full_reporting_results": full_reporting_results,
        "difficulty_analysis": difficulty_analysis,
    }


def _evaluate_report_adaptive(
    name, test_data, baseline_results, baseline_acc,
    intervention_vec, kw, adaptive_config,
) -> Dict:
    """Evaluate adaptive intervention on indexed test data."""
    print(f"\n  {name}...")
    correct, improved, worsened = 0, 0, 0
    metric_key = adaptive_config.difficulty_metric

    for i, screening_entry in enumerate(tqdm(test_data, desc=name)):
        sample = screening_to_sample(screening_entry)
        p_correct = screening_entry[metric_key]
        result = get_prediction_with_adaptive_intervention(
            sample, intervention_vec, p_correct,
            adaptive_config=adaptive_config, **kw
        )
        correct += result["correct"]

        if not baseline_results[i]["correct"] and result["correct"]:
            improved += 1
        elif baseline_results[i]["correct"] and not result["correct"]:
            worsened += 1

    acc = correct / len(test_data)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(test_data),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _evaluate_report_full_adaptive(
    name, test_data, baseline_dict, baseline_acc,
    intervention_vec, kw, adaptive_config,
) -> Dict:
    """Evaluate adaptive intervention on full test data."""
    print(f"\n  {name}...")
    correct, improved, worsened = 0, 0, 0
    metric_key = adaptive_config.difficulty_metric

    for screening_entry in tqdm(test_data, desc=name):
        sample = screening_to_sample(screening_entry)
        p_correct = screening_entry[metric_key]
        result = get_prediction_with_adaptive_intervention(
            sample, intervention_vec, p_correct,
            adaptive_config=adaptive_config, **kw
        )
        correct += result["correct"]

        baseline_correct = baseline_dict[screening_entry["q_idx"]]
        if not baseline_correct and result["correct"]:
            improved += 1
        elif baseline_correct and not result["correct"]:
            worsened += 1

    acc = correct / len(test_data)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(test_data),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _evaluate_report_adaptive_approx(
    name, test_data, baseline_results, baseline_acc,
    approximated_hidden_vec, kw, adaptive_config,
) -> Dict:
    """Evaluate adaptive approximated intervention on indexed test data."""
    print(f"\n  {name}...")
    correct, improved, worsened = 0, 0, 0
    metric_key = adaptive_config.difficulty_metric

    for i, screening_entry in enumerate(tqdm(test_data, desc=name)):
        sample = screening_to_sample(screening_entry)
        p_correct = screening_entry[metric_key]
        result = get_prediction_with_adaptive_approximated_intervention(
            sample, approximated_hidden_vec, p_correct,
            adaptive_config=adaptive_config, **kw
        )
        correct += result["correct"]

        if not baseline_results[i]["correct"] and result["correct"]:
            improved += 1
        elif baseline_results[i]["correct"] and not result["correct"]:
            worsened += 1

    acc = correct / len(test_data)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(test_data),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _evaluate_report_full_adaptive_approx(
    name, test_data, baseline_dict, baseline_acc,
    approximated_hidden_vec, kw, adaptive_config,
) -> Dict:
    """Evaluate adaptive approximated intervention on full test data."""
    print(f"\n  {name}...")
    correct, improved, worsened = 0, 0, 0
    metric_key = adaptive_config.difficulty_metric

    for screening_entry in tqdm(test_data, desc=name):
        sample = screening_to_sample(screening_entry)
        p_correct = screening_entry[metric_key]
        result = get_prediction_with_adaptive_approximated_intervention(
            sample, approximated_hidden_vec, p_correct,
            adaptive_config=adaptive_config, **kw
        )
        correct += result["correct"]

        baseline_correct = baseline_dict[screening_entry["q_idx"]]
        if not baseline_correct and result["correct"]:
            improved += 1
        elif baseline_correct and not result["correct"]:
            worsened += 1

    acc = correct / len(test_data)
    delta = acc - baseline_acc
    p_val = (
        binomtest(improved, improved + worsened, 0.5, alternative="greater").pvalue
        if improved + worsened > 0
        else 1.0
    )

    print(f"    Baseline: {baseline_acc*100:.1f}%, Intervention: {acc*100:.1f}%, Delta: {delta*100:+.1f}%")

    return {
        "name": name,
        "n_questions": len(test_data),
        "baseline_acc": baseline_acc,
        "intervention_acc": acc,
        "delta": delta,
        "n_improved": improved,
        "n_worsened": worsened,
        "binomial_p": p_val,
    }


def _run_difficulty_analysis_adaptive(
    test_data_full, baseline_full_dict, best_intervention_vec, kw, adaptive_config,
) -> Dict:
    """Analyze adaptive intervention effect across difficulty ranges."""
    print("\n  Running difficulty analysis (adaptive)...")
    metric_key = adaptive_config.difficulty_metric

    detailed_results = []
    for screening_entry in tqdm(test_data_full, desc="Difficulty Analysis (adaptive)"):
        sample = screening_to_sample(screening_entry)
        p_correct = screening_entry[metric_key]
        result = get_prediction_with_adaptive_intervention(
            sample, best_intervention_vec, p_correct,
            adaptive_config=adaptive_config, **kw
        )

        baseline_correct = baseline_full_dict[screening_entry["q_idx"]]
        detailed_results.append({
            "q_idx": screening_entry["q_idx"],
            "p_correct": screening_entry["p_correct"],
            "baseline_correct": baseline_correct,
            "intervention_correct": result["correct"],
            "improved": not baseline_correct and result["correct"],
            "worsened": baseline_correct and not result["correct"],
        })

    df = pd.DataFrame(detailed_results)

    difficulty_ranges = [
        ("Very hard", 0.00, 0.15),
        ("Hard", 0.15, 0.35),
        ("Medium", 0.35, 0.65),
        ("Easy", 0.65, 0.85),
        ("Very easy", 0.85, 1.01),
    ]

    analysis = []
    for name, low, high in difficulty_ranges:
        subset = df[(df["p_correct"] >= low) & (df["p_correct"] < high)]
        if len(subset) == 0:
            continue

        analysis.append({
            "range_name": name,
            "range_low": low,
            "range_high": high,
            "n_questions": len(subset),
            "baseline_acc": float(subset["baseline_correct"].mean()),
            "intervention_acc": float(subset["intervention_correct"].mean()),
            "delta": float(subset["intervention_correct"].mean() - subset["baseline_correct"].mean()),
            "n_improved": int(subset["improved"].sum()),
            "n_worsened": int(subset["worsened"].sum()),
        })

    totals = {
        "n_questions": len(df),
        "baseline_acc": float(df["baseline_correct"].mean()),
        "intervention_acc": float(df["intervention_correct"].mean()),
        "delta": float(df["intervention_correct"].mean() - df["baseline_correct"].mean()),
        "n_improved": int(df["improved"].sum()),
        "n_worsened": int(df["worsened"].sum()),
    }

    return {"difficulty_ranges": analysis, "totals": totals}


# ════════════════════════════════════════════════════════════════════════════════
# Phase 4 with precomputed hidden-space vector (for CAA and similar baselines)
# ════════════════════════════════════════════════════════════════════════════════


def run_phase4_reporting_with_vector(
    hidden_vec: torch.Tensor,
    best_strength: float,
    holdout_filtered: List[Dict],
    holdout_data: List[Dict],
    baseline_holdout_filtered_results: List[Dict],
    baseline_holdout_filtered_acc: float,
    baseline_holdout_results: List[Dict],
    baseline_holdout_acc: float,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    adaptive_config: Optional[AdaptiveConfig] = None,
    batch_size: int = 1,
    label: str = "CAA",
) -> Dict:
    """Phase 4 reporting with a precomputed hidden-space steering vector.

    Unlike run_phase4_full_reporting which builds the vector from SAE features,
    this function takes a ready-to-use hidden-space vector (e.g. from CAA).

    Runs static evaluation. If adaptive_config is provided and enabled,
    also runs adaptive variants.

    Returns dict with:
        best_intervention_vec, approximated_hidden_vec,
        n_features_in_vec (always 0 for hidden-space vectors),
        full_reporting_results, difficulty_analysis
    """
    print("=" * 80)
    print(f"PHASE 4: FULL REPORTING ({label})")
    if adaptive_config and adaptive_config.enabled:
        print(f"  [Adaptive steering ENABLED — mode: {adaptive_config.mode}]")
    print(f"  Vector norm: {hidden_vec.norm().item():.4f}")
    print(f"  Strength: {best_strength}")
    print("=" * 80)

    scaled_vec = hidden_vec * best_strength
    baseline_holdout_dict = {r["q_idx"]: r["correct"] for r in baseline_holdout_results}

    full_reporting_results = {}

    bkw = dict(
        batch_size=batch_size, model=model, tokenizer=tokenizer, config=config,
        hook_mgr=hook_mgr, sae=sae, answer_token_id_list=answer_token_id_list, device=device,
    )

    filtered_samples = [screening_to_sample(e) for e in holdout_filtered]
    full_samples = [screening_to_sample(e) for e in holdout_data]
    approx_fn = make_approx_intervention_fn(scaled_vec)

    if adaptive_config and adaptive_config.enabled:
        # ── Static reports ──
        r1 = _evaluate_report_batched(
            "Filtered (static)", filtered_samples, approx_fn,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_exact_static"] = r1

        r2 = _evaluate_report_full_batched(
            "Full (static)", full_samples, approx_fn,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_exact_static"] = r2

        # ── Adaptive reports ──
        strength_fn = get_strength_fn(
            adaptive_config.mode,
            p_min=adaptive_config.p_min, p_sweet=adaptive_config.p_sweet,
            p_max=adaptive_config.p_max, mu=adaptive_config.mu, sigma=adaptive_config.sigma,
            transition_strength=adaptive_config.transition_strength,
        )

        metric_key = adaptive_config.difficulty_metric
        filtered_p_correct = [e[metric_key] for e in holdout_filtered]
        full_p_correct = [e[metric_key] for e in holdout_data]

        adaptive_fn_filtered = make_adaptive_approx_intervention_fn(
            scaled_vec, strength_fn, filtered_p_correct,
        )
        r3 = _evaluate_report_batched(
            "Filtered (adaptive)", filtered_samples, adaptive_fn_filtered,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_approx_adaptive"] = r3

        adaptive_fn_full = make_adaptive_approx_intervention_fn(
            scaled_vec, strength_fn, full_p_correct,
        )
        r4 = _evaluate_report_full_batched(
            "Full (adaptive)", full_samples, adaptive_fn_full,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_approx_adaptive"] = r4

        # Difficulty analysis with adaptive
        adaptive_fn_diff = make_adaptive_approx_intervention_fn(
            scaled_vec, strength_fn, full_p_correct,
        )
        difficulty_analysis = _run_difficulty_analysis_batched(
            holdout_data, baseline_holdout_dict, full_samples, adaptive_fn_diff, **bkw,
        )
    else:
        # ── Static-only reports ──
        r1 = _evaluate_report_batched(
            "Filtered", filtered_samples, approx_fn,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc, **bkw,
        )
        full_reporting_results["filtered_exact"] = r1

        r2 = _evaluate_report_full_batched(
            "Full", full_samples, approx_fn,
            holdout_data, baseline_holdout_dict, baseline_holdout_acc, **bkw,
        )
        full_reporting_results["full_exact"] = r2

        # Also add approx keys (same as exact for hidden-space vectors)
        full_reporting_results["filtered_approx"] = r1
        full_reporting_results["full_approx"] = r2

        difficulty_analysis = _run_difficulty_analysis_batched(
            holdout_data, baseline_holdout_dict, full_samples, approx_fn, **bkw,
        )

    # Print summary table
    print(f"\n{'=' * 80}")
    print(f"FULL REPORTING SUMMARY ({label})")
    print(f"{'=' * 80}")
    print(f"{'Report':<35} {'N':>6} {'Baseline':>10} {'Interv.':>10} {'Delta':>8} {'p-value':>10}")
    print(f"{'─' * 80}")

    for key, r in full_reporting_results.items():
        p_str = f"{r['binomial_p']:.4f}" if r["binomial_p"] is not None else "N/A"
        print(f"{r['name']:<35} {r['n_questions']:>6} {r['baseline_acc']*100:>9.1f}% "
              f"{r['intervention_acc']*100:>9.1f}% {r['delta']*100:>+7.1f}% {p_str:>10}")

    return {
        "best_intervention_vec": scaled_vec,
        "approximated_hidden_vec": scaled_vec,
        "n_features_in_vec": 0,
        "full_reporting_results": full_reporting_results,
        "difficulty_analysis": difficulty_analysis,
    }
