"""JSON summary building, saving, and final summary printing."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ExperimentConfig

# Reconstruction guarantee: any grid cell with top_k_pos/neg <= this value can be
# rebuilt from the saved phase1_top_features / phase2_top_features_per_strength
# sections. A grid runner using a larger top_k silently loses reconstructability —
# the experiment still runs and the JSON is still valid, no runtime error fires.
N_TOP_FEATURES_TRACKED = 25


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to Python built-ins."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _build_causal_screening_section(
    df_pos: Optional[pd.DataFrame],
    df_neg: Optional[pd.DataFrame],
    n_causal_pos: int,
    n_causal_neg: int,
) -> Dict[str, Any]:
    """Per-strength causal-screening breakdown.

    df_pos/df_neg now contain one row per (feature, test_strength) pair, so the
    headline counts go to unique-feature counts while a per_strength list keeps
    the strength-aware stats.
    """
    def _per_feature_count(df: Optional[pd.DataFrame]) -> int:
        if df is None or df.empty or "feature_id" not in df.columns:
            return 0
        return int(df["feature_id"].nunique())

    section: Dict[str, Any] = {
        "n_unique_tested_positive": _per_feature_count(df_pos),
        "n_unique_tested_negative": _per_feature_count(df_neg),
        "n_causal_positive_unique_features": int(n_causal_pos),
        "n_causal_negative_unique_features": int(n_causal_neg),
        "per_strength": [],
    }

    strengths: List[float] = []
    if df_pos is not None and not df_pos.empty and "test_strength" in df_pos.columns:
        strengths = sorted(df_pos["test_strength"].unique().tolist())
    elif df_neg is not None and not df_neg.empty and "test_strength" in df_neg.columns:
        strengths = sorted(df_neg["test_strength"].unique().tolist())

    def _strength_subset(df: Optional[pd.DataFrame], s: float) -> pd.DataFrame:
        if df is None or df.empty or "test_strength" not in df.columns:
            return pd.DataFrame()
        return df[df["test_strength"] == s]

    for s in strengths:
        pos_sub = _strength_subset(df_pos, s)
        neg_sub = _strength_subset(df_neg, s)
        n_pos_causal = int(pos_sub["is_causal"].sum()) if len(pos_sub) > 0 else 0
        n_neg_causal = int(neg_sub["is_causal"].sum()) if len(neg_sub) > 0 else 0
        section["per_strength"].append({
            "test_strength": float(s),
            "n_tested_positive": int(len(pos_sub)),
            "n_causal_positive": n_pos_causal,
            "pct_causal_positive": n_pos_causal / len(pos_sub) if len(pos_sub) > 0 else 0.0,
            "n_tested_negative": int(len(neg_sub)),
            "n_causal_negative": n_neg_causal,
            "pct_causal_negative": n_neg_causal / len(neg_sub) if len(neg_sub) > 0 else 0.0,
            "avg_delta_positive": float(pos_sub["delta"].mean()) if len(pos_sub) > 0 else 0.0,
            "avg_delta_negative": float(neg_sub["delta"].mean()) if len(neg_sub) > 0 else 0.0,
        })
    return section


def _build_pair_stability_section(df_stab: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Per-(strength, pair_type) pairwise-combination-stability summary plus per-pair records."""
    section: Dict[str, Any] = {
        "n_pairs_total": 0,
        "per_strength_pair_type": [],
        "samples": [],
    }
    if df_stab is None or df_stab.empty:
        return section

    section["n_pairs_total"] = int(len(df_stab))

    def _safe_median(series: pd.Series) -> float:
        cleaned = series.replace([np.inf, -np.inf], np.nan).dropna()
        return float(cleaned.median()) if len(cleaned) > 0 else float("nan")

    grouped = df_stab.groupby(["test_strength", "pair_type"], sort=True)
    for (s, pair_type), grp in grouped:
        section["per_strength_pair_type"].append({
            "test_strength": float(s),
            "pair_type": str(pair_type),
            "n_pairs": int(len(grp)),
            # Primary stability metrics
            "pct_remains_positive": float(grp["remains_positive"].mean()),
            "pct_retains_best": float(grp["retains_best"].mean()),
            "mean_residual_vs_best": float(grp["residual_vs_best"].mean()),
            "std_residual_vs_best": float(grp["residual_vs_best"].std()) if len(grp) > 1 else 0.0,
            "mean_delta_combined": float(grp["delta_combined"].mean()),
            "mean_delta_best_single": float(grp["delta_best_single"].mean()),
            # Secondary additivity reference
            "mean_residual_vs_sum": float(grp["residual"].mean()),
            "median_ratio_combined_over_sum": _safe_median(grp["ratio_combined_over_sum"]),
            "mean_delta_sum_marginals": float(grp["delta_sum_marginals"].mean()),
        })

    for _, row in df_stab.iterrows():
        ratio = row["ratio_combined_over_sum"]
        section["samples"].append({
            "test_strength": float(row["test_strength"]),
            "pair_type": str(row["pair_type"]),
            "feature_i": int(row["feature_i"]),
            "feature_j": int(row["feature_j"]),
            "sign_i": int(row["sign_i"]),
            "sign_j": int(row["sign_j"]),
            "delta_i": float(row["delta_i"]),
            "delta_j": float(row["delta_j"]),
            "delta_combined": float(row["delta_combined"]),
            "delta_best_single": float(row["delta_best_single"]),
            "residual_vs_best": float(row["residual_vs_best"]),
            "remains_positive": bool(row["remains_positive"]),
            "retains_best": bool(row["retains_best"]),
            "delta_sum_marginals": float(row["delta_sum_marginals"]),
            "residual_vs_sum": float(row["residual"]),
            "ratio_combined_over_sum": (
                float(ratio) if pd.notna(ratio) and not np.isinf(ratio) else None
            ),
        })
    return section


def _build_phase1_top_features_section(
    pos_candidates: Optional[List],
    neg_candidates: Optional[List],
) -> Dict[str, Any]:
    """Top-N Phase-1 features by freq_diff for each direction.

    Inputs are the (feature_id, data_dict) tuples returned by
    FeatureTracker.get_top_features — already sorted (desc for pos, asc for neg).
    None is treated as an empty list.
    """
    def _serialise(candidates: Optional[List]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        return [
            {
                "feature_id": int(idx),
                "freq_diff": float(data["freq_diff"]),
                "s_count": int(data["s_count"]),
                "f_count": int(data["f_count"]),
            }
            for idx, data in candidates[:N_TOP_FEATURES_TRACKED]
        ]

    return {
        "n_kept": N_TOP_FEATURES_TRACKED,  # configured cap; actual count may be lower
        "positive": _serialise(pos_candidates),
        "negative": _serialise(neg_candidates),
    }


def _build_phase2_top_features_per_strength_section(
    df_pos: Optional[pd.DataFrame],
    df_neg: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Top-N Phase-2 features by delta within each test_strength bucket.

    Sorts both positive and negative rows by delta descending: Phase 2 already
    encodes intervention direction in delta (positive features tested at
    +strength, negative features at -strength), so the highest-delta rows are
    the ones Phase 3 would pick for `ranking_criterion='causal_delta'`.

    None or empty inputs produce an empty list. Bucketing uses the union of
    test_strengths found in df_pos and df_neg, ordered ascending.
    """
    def _has_strength_col(df: Optional[pd.DataFrame]) -> bool:
        return df is not None and not df.empty and "test_strength" in df.columns

    strengths_pos = (
        set(df_pos["test_strength"].unique().tolist()) if _has_strength_col(df_pos) else set()
    )
    strengths_neg = (
        set(df_neg["test_strength"].unique().tolist()) if _has_strength_col(df_neg) else set()
    )
    strengths = sorted(strengths_pos | strengths_neg)
    if not strengths:
        return []

    def _bucket(df: Optional[pd.DataFrame], s: float) -> List[Dict[str, Any]]:
        if not _has_strength_col(df):
            return []
        sub = df[df["test_strength"] == s]
        if sub.empty:
            return []
        top = sub.nlargest(N_TOP_FEATURES_TRACKED, "delta")
        records: List[Dict[str, Any]] = []
        for _, row in top.iterrows():
            records.append({
                "feature_id": int(row["feature_id"]),
                "freq_diff": float(row["freq_diff"]),
                "delta": float(row["delta"]),
                "is_causal": bool(row["is_causal"]),
                "n_improved": int(row["n_improved"]),
                "n_worsened": int(row["n_worsened"]),
            })
        return records

    return [
        {
            "test_strength": float(s),
            "positive": _bucket(df_pos, s),
            "negative": _bucket(df_neg, s),
        }
        for s in strengths
    ]


def build_summary(
    *,
    experiment_config: ExperimentConfig,
    sae_d_hidden: int,
    sae_top_k: Optional[int],
    all_questions_count: int,
    filtered_data_count: int,
    train_data_count: int,
    val_data_count: int,
    test_filtered_count: int,
    test_full_count: int,
    grid_eval_count: int = 0,
    holdout_count: int = 0,
    holdout_filtered_count: int = 0,
    baseline_filtered_acc: float = 0.0,
    baseline_full_acc: float = 0.0,
    dynamic_min_occurrences: int = 0,
    phase1_pos_candidates: Optional[List] = None,
    phase1_neg_candidates: Optional[List] = None,
    df_pos: pd.DataFrame = None,
    df_neg: pd.DataFrame = None,
    n_causal_pos: int = 0,
    n_causal_neg: int = 0,
    df_pair_stability: Optional[pd.DataFrame] = None,
    grid_config_dict: Dict = None,
    n_combinations: int = 0,
    best_config: Dict = None,
    n_features_in_vec: int = 0,
    full_reporting_results: Dict = None,
    difficulty_analysis: Dict = None,
    kfold_results: Dict = None,
    df_grid: pd.DataFrame = None,
    generalisation_results: Optional[Dict] = None,
    answer_diagnostics: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a comprehensive JSON-serializable summary dict."""
    cfg = experiment_config
    sae_cfg = cfg.sae
    bench_cfg = cfg.benchmark
    causal_cfg = cfg.causal

    # Approximation comparison — handle both standard and adaptive key naming
    def _get_report(key):
        """Look up report by key, falling back to _static or _adaptive suffix."""
        if key in full_reporting_results:
            return full_reporting_results[key]
        for suffix in ("_static", "_adaptive"):
            if key + suffix in full_reporting_results:
                return full_reporting_results[key + suffix]
        return {"delta": 0.0}  # safe fallback

    exact_delta_filtered = _get_report("filtered_exact")["delta"]
    approx_delta_filtered = _get_report("filtered_approx")["delta"]
    exact_delta_full = _get_report("full_exact")["delta"]
    approx_delta_full = _get_report("full_approx")["delta"]

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "experiment_version": "23_full_reporting",

        "model": {
            "model_id": cfg.model.model_id,
            "short_name": cfg.model.short_name,
            "d_model": cfg.model.d_model,
            "n_layers": cfg.model.n_layers,
        },

        "sae": {
            "name": sae_cfg.name,
            "architecture": sae_cfg.architecture,
            "repo": sae_cfg.repo,
            "layer": sae_cfg.layer,
            "d_hidden": sae_d_hidden,
            "top_k": sae_top_k,
        },

        "benchmark": {
            "name": bench_cfg.name,
            "dataset": bench_cfg.dataset_name,
            "config": bench_cfg.dataset_config,
            "choices": bench_cfg.choices,
        },

        "data_stats": {
            "total_questions": all_questions_count,
            "filtered_questions": filtered_data_count,
            "p_correct_range": [causal_cfg.min_p_correct, causal_cfg.max_p_correct],
            "train_questions": train_data_count,
            "val_questions": val_data_count,
            "test_filtered_questions": test_filtered_count,
            "test_full_questions": test_full_count,
            "grid_eval_questions": grid_eval_count,
            "holdout_questions": holdout_count,
            "holdout_filtered_questions": holdout_filtered_count,
            "baseline_filtered_accuracy": baseline_filtered_acc,
            "baseline_full_accuracy": baseline_full_acc,
            "dynamic_min_occurrences": dynamic_min_occurrences,
        },

        "causal_config": {
            "causal_test_strength": list(causal_cfg.causal_test_strength)
            if causal_cfg.causal_test_strength is not None else [],
            "n_candidate_features_pos": causal_cfg.n_candidate_features_pos,
            "n_candidate_features_neg": causal_cfg.n_candidate_features_neg,
            "combination_stability_check": causal_cfg.combination_stability_check,
            "combination_stability_n_pairs": causal_cfg.combination_stability_n_pairs,
        },

        "causal_screening": _build_causal_screening_section(
            df_pos, df_neg, n_causal_pos, n_causal_neg,
        ),

        "phase1_top_features": _build_phase1_top_features_section(
            phase1_pos_candidates, phase1_neg_candidates,
        ),

        "phase2_top_features_per_strength":
            _build_phase2_top_features_per_strength_section(df_pos, df_neg),

        "pair_stability_check": _build_pair_stability_section(df_pair_stability),

        "grid_config": grid_config_dict,
        "n_combinations": n_combinations,

        "best_config": {
            "top_k_pos": best_config["top_k_pos"],
            "top_k_neg": best_config["top_k_neg"],
            "strength": float(best_config["strength"]),
            "ranking_criterion": best_config["ranking_criterion"],
            "weighting_mode": best_config["weighting_mode"],
            "n_features_used": n_features_in_vec,
        },

        "full_reporting": {
            key: {
                "name": r["name"],
                "n_questions": int(r["n_questions"]),
                "baseline_acc": float(r["baseline_acc"]),
                "intervention_acc": float(r["intervention_acc"]),
                "delta": float(r["delta"]),
                "n_improved": int(r["n_improved"]),
                "n_worsened": int(r["n_worsened"]),
                "binomial_p": float(r["binomial_p"]) if r["binomial_p"] is not None else None,
                "significant": bool(r["binomial_p"] and r["binomial_p"] < 0.05),
            }
            for key, r in full_reporting_results.items()
        },

        "approximation_comparison": {
            "filtered": {
                "exact_delta": float(exact_delta_filtered),
                "approx_delta": float(approx_delta_filtered),
                "difference": float(approx_delta_filtered - exact_delta_filtered),
            },
            "full": {
                "exact_delta": float(exact_delta_full),
                "approx_delta": float(approx_delta_full),
                "difference": float(approx_delta_full - exact_delta_full),
            },
        },

        "difficulty_analysis": difficulty_analysis,
        "kfold_results": kfold_results,
        "all_grid_results": df_grid.to_dict("records"),
    }

    # Ranking comparison (based on full benchmark delta)
    ranking_comparison = {}
    delta_col = next(
        (c for c in ("delta_grid_eval", "delta_full", "delta") if c in df_grid.columns),
        "delta",
    )
    for ranking in df_grid["ranking_criterion"].unique():
        subset = df_grid[df_grid["ranking_criterion"] == ranking]
        if len(subset) > 0:
            best_in_subset = subset.loc[subset[delta_col].idxmax()]
            ranking_comparison[ranking] = {
                "best_delta_full": float(best_in_subset[delta_col]),
                "avg_delta_full": float(subset[delta_col].mean()),
                "positive_configs_full": int((subset[delta_col] > 0).sum()),
                "total_configs": len(subset),
            }
    summary["ranking_comparison"] = ranking_comparison

    if generalisation_results is not None:
        summary["generalisation"] = generalisation_results

    if answer_diagnostics is not None:
        summary["answer_diagnostics"] = answer_diagnostics

    return summary


def build_caa_summary(
    *,
    experiment_config: ExperimentConfig,
    sae_layer: int,
    all_questions_count: int,
    filtered_data_count: int,
    train_data_count: int,
    val_data_count: int,
    test_filtered_count: int,
    test_full_count: int,
    baseline_filtered_acc: float,
    baseline_full_acc: float,
    caa_vector_norm: float,
    best_strength: float,
    strength_sweep_results: pd.DataFrame,
    full_reporting_results: Dict,
    difficulty_analysis: Dict,
    kfold_results: Dict,
    generalisation_results: Optional[Dict] = None,
    answer_diagnostics: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build a comprehensive JSON-serializable summary for a CAA experiment."""
    cfg = experiment_config
    bench_cfg = cfg.benchmark
    causal_cfg = cfg.causal

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "experiment_version": "caa_baseline",
        "method": "caa",

        "model": {
            "model_id": cfg.model.model_id,
            "short_name": cfg.model.short_name,
            "d_model": cfg.model.d_model,
            "n_layers": cfg.model.n_layers,
        },

        "intervention_layer": sae_layer,

        "benchmark": {
            "name": bench_cfg.name,
            "dataset": bench_cfg.dataset_name,
            "config": bench_cfg.dataset_config,
            "choices": bench_cfg.choices,
        },

        "data_stats": {
            "total_questions": all_questions_count,
            "filtered_questions": filtered_data_count,
            "p_correct_range": [causal_cfg.min_p_correct, causal_cfg.max_p_correct],
            "train_questions": train_data_count,
            "val_questions": val_data_count,
            "test_filtered_questions": test_filtered_count,
            "test_full_questions": test_full_count,
            "baseline_filtered_accuracy": baseline_filtered_acc,
            "baseline_full_accuracy": baseline_full_acc,
        },

        "caa": {
            "vector_norm": caa_vector_norm,
            "best_strength": best_strength,
        },

        "strength_sweep": strength_sweep_results.to_dict("records"),

        "full_reporting": {
            key: {
                "name": r["name"],
                "n_questions": int(r["n_questions"]),
                "baseline_acc": float(r["baseline_acc"]),
                "intervention_acc": float(r["intervention_acc"]),
                "delta": float(r["delta"]),
                "n_improved": int(r["n_improved"]),
                "n_worsened": int(r["n_worsened"]),
                "binomial_p": float(r["binomial_p"]) if r["binomial_p"] is not None else None,
                "significant": bool(r["binomial_p"] and r["binomial_p"] < 0.05),
            }
            for key, r in full_reporting_results.items()
        },

        "difficulty_analysis": difficulty_analysis,
        "kfold_results": kfold_results,
    }

    # Adaptive config info
    if causal_cfg.adaptive.enabled:
        summary["adaptive_config"] = {
            "enabled": True,
            "mode": causal_cfg.adaptive.mode,
            "difficulty_metric": causal_cfg.adaptive.difficulty_metric,
            "p_min": causal_cfg.adaptive.p_min,
            "p_sweet": causal_cfg.adaptive.p_sweet,
            "p_max": causal_cfg.adaptive.p_max,
            "transition_strength": causal_cfg.adaptive.transition_strength,
        }

    if generalisation_results is not None:
        summary["generalisation"] = generalisation_results

    if answer_diagnostics is not None:
        summary["answer_diagnostics"] = answer_diagnostics

    return summary


def save_caa_summary(summary: Dict, output_dir: str, benchmark_config, layer: int) -> str:
    """Save CAA summary to a JSON file and return the path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"caa_{benchmark_config.cache_name}_L{layer}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpySafeEncoder)

    print(f"CAA summary saved: {filepath}")
    return filepath


def save_summary(summary: Dict, output_dir: str, sae_config, benchmark_config) -> str:
    """Save summary to a JSON file and return the path."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sae_suffix = "batchtopk" if sae_config.architecture == "batchtopk" else "relu"
    layer_suffix = f"_L{sae_config.layer}"
    k_suffix = f"_K{sae_config.top_k}" if sae_config.top_k else ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{benchmark_config.cache_name}_{sae_suffix}{layer_suffix}{k_suffix}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpySafeEncoder)

    print(f"Summary saved: {filepath}")
    return filepath


def print_final_summary(
    experiment_config: ExperimentConfig,
    full_reporting_results: Dict,
    kfold_results: Dict,
    n_features_in_vec: int,
    best_config: Dict,
    summary_file: str,
    plot_file: Optional[str] = None,
) -> None:
    """Print the comprehensive final summary."""
    cfg = experiment_config
    sae_cfg = cfg.sae
    bench_cfg = cfg.benchmark

    def _get_report(key):
        if key in full_reporting_results:
            return full_reporting_results[key]
        for suffix in ("_static", "_adaptive"):
            if key + suffix in full_reporting_results:
                return full_reporting_results[key + suffix]
        return {"delta": 0.0, "binomial_p": None}

    fr_filtered = _get_report("filtered_exact")
    fr_full = _get_report("full_exact")

    exact_delta_filtered = fr_filtered["delta"]
    approx_delta_filtered = _get_report("filtered_approx")["delta"]
    exact_delta_full = fr_full["delta"]
    approx_delta_full = _get_report("full_approx")["delta"]

    kfold_p = kfold_results.get("binomial_p")
    kfold_sig = kfold_p is not None and kfold_p < 0.05
    n_positive_folds = kfold_results["n_positive_folds"]
    n_folds = kfold_results["n_folds"]
    mean_delta = kfold_results["mean_delta"]
    std_delta = kfold_results["std_delta"]

    print("=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    print(f"\n  SAE: {sae_cfg.name} ({sae_cfg.architecture}, layer {sae_cfg.layer})")
    print(f"  Benchmark: {bench_cfg.name}")
    print(f"  Model: {cfg.model.short_name}")

    print(f"\n  Best config: K+={best_config['top_k_pos']}, K-={best_config['top_k_neg']}, "
          f"strength={best_config['strength']}, ranking={best_config['ranking_criterion']}, "
          f"weighting={best_config['weighting_mode']}")
    print(f"  Features used: {n_features_in_vec}")

    # Full reporting table
    print(f"\n{'─' * 80}")
    print(f"{'Report':<30} {'N':>6} {'Baseline':>10} {'Interv.':>10} {'Delta':>8} {'p-value':>10}")
    print(f"{'─' * 80}")

    for key, r in full_reporting_results.items():
        sig = "*" if r["binomial_p"] is not None and r["binomial_p"] < 0.05 else ""
        p_str = f"{r['binomial_p']:.4f}" if r["binomial_p"] is not None else "N/A"
        print(f"{r['name']:<30} {r['n_questions']:>6} {r['baseline_acc']*100:>9.1f}% "
              f"{r['intervention_acc']*100:>9.1f}% {r['delta']*100:>+7.1f}% {p_str:>10}{sig}")

    # K-fold
    print(f"\n  K-Fold: mean delta = {mean_delta*100:+.2f}% +/- {std_delta*100:.2f}%")
    print(f"  Positive folds: {n_positive_folds}/{n_folds}")
    if kfold_p is not None:
        print(f"  Binomial p-value (one-sided): {kfold_p:.4f}")

    # Verdict
    positive_indicators = 0
    if fr_filtered["binomial_p"] is not None and fr_filtered["binomial_p"] < 0.05:
        positive_indicators += 1
    if fr_full["delta"] > 0:
        positive_indicators += 1
    if kfold_sig:
        positive_indicators += 1
    if n_positive_folds > n_folds // 2:
        positive_indicators += 1

    if positive_indicators >= 3:
        verdict = "SUCCESS: Robust improvement demonstrated"
    elif positive_indicators >= 2:
        verdict = "PARTIAL: Positive tendency, not fully robust"
    else:
        verdict = "NO CLEAR ADVANTAGE on this benchmark"

    print(f"\n  Verdict ({positive_indicators}/4 positive indicators): {verdict}")

    print(f"\n  Files saved:")
    print(f"    {summary_file}")
    if plot_file:
        print(f"    {plot_file}")
    print("=" * 80)
