"""Tests for Phase 1 / Phase 2 top-feature tracking in build_summary."""

import json

import numpy as np
import pandas as pd

from sae_experiments.config import (
    BenchmarkConfig,
    CausalConfig,
    ExperimentConfig,
    FieldMapping,
    GridConfig,
    ModelConfig,
    SAEConfig,
)
from sae_experiments.reporting import (
    N_TOP_FEATURES_TRACKED,
    _NumpySafeEncoder,
    _build_phase1_top_features_section,
    _build_phase2_top_features_per_strength_section,
    build_summary,
)


def _make_candidate(idx: int, freq_diff: float, s_count: int, f_count: int) -> tuple:
    """Match the shape returned by FeatureTracker.get_top_features:
    (feature_id, {freq_diff, s_count, f_count, s_rate, f_rate})."""
    return (
        idx,
        {
            "freq_diff": freq_diff,
            "s_count": s_count,
            "f_count": f_count,
            "s_rate": s_count / 100.0,
            "f_rate": f_count / 100.0,
        },
    )


def test_phase1_constant_is_25():
    assert N_TOP_FEATURES_TRACKED == 25


def test_phase1_truncates_to_25_and_preserves_order():
    pos = [_make_candidate(i, 0.30 - 0.01 * i, 50, 10) for i in range(30)]
    neg = [_make_candidate(100 + i, -0.20 + 0.01 * i, 5, 50) for i in range(20)]

    section = _build_phase1_top_features_section(pos, neg)

    assert section["n_kept"] == 25
    assert len(section["positive"]) == 25
    assert len(section["negative"]) == 20
    assert section["positive"][0]["feature_id"] == 0
    assert section["positive"][0]["freq_diff"] == 0.30
    assert section["positive"][0]["s_count"] == 50
    assert section["positive"][0]["f_count"] == 10
    assert section["negative"][0]["feature_id"] == 100
    assert section["negative"][0]["freq_diff"] == -0.20


def test_phase1_handles_empty_inputs():
    section = _build_phase1_top_features_section([], [])
    assert section == {"n_kept": 25, "positive": [], "negative": []}


def test_phase1_handles_none_inputs():
    section = _build_phase1_top_features_section(None, None)
    assert section == {"n_kept": 25, "positive": [], "negative": []}


def test_phase1_handles_fewer_than_25():
    pos = [_make_candidate(i, 0.10 - 0.01 * i, 30, 5) for i in range(7)]
    section = _build_phase1_top_features_section(pos, [])
    assert len(section["positive"]) == 7
    assert section["negative"] == []


def test_phase1_emits_only_documented_fields():
    pos = [_make_candidate(1, 0.18, 42, 7)]
    section = _build_phase1_top_features_section(pos, [])
    assert set(section["positive"][0].keys()) == {
        "feature_id", "freq_diff", "s_count", "f_count",
    }


def test_phase1_section_is_json_serialisable():
    pos = [_make_candidate(1, 0.18, np.int64(42), np.int64(7))]
    section = _build_phase1_top_features_section(pos, [])
    json.dumps(section, cls=_NumpySafeEncoder)


def _make_phase2_df(features_per_strength: dict) -> pd.DataFrame:
    """Build a Phase-2 dataframe from {strength: [(feature_id, freq_diff, delta), ...]}."""
    rows = []
    for strength, feats in features_per_strength.items():
        for feature_id, freq_diff, delta in feats:
            rows.append({
                "feature_id": feature_id,
                "freq_diff": freq_diff,
                "baseline_acc": 0.5,
                "intervention_acc": 0.5 + delta,
                "delta": delta,
                "n_improved": int(max(delta, 0) * 100),
                "n_worsened": int(max(-delta, 0) * 100),
                "net_change": int(delta * 100),
                "is_causal": delta > 0,
                "test_strength": strength,
            })
    return pd.DataFrame(rows)


def test_phase2_buckets_per_strength_and_sorts_by_delta():
    df_pos = _make_phase2_df({
        0.5: [(i, 0.18 - 0.001 * i, 0.05 - 0.001 * i) for i in range(30)],
        1.0: [(i, 0.18 - 0.001 * i, 0.04 - 0.001 * i) for i in range(30)],
    })
    df_neg = _make_phase2_df({
        0.5: [(100 + i, -0.10 + 0.001 * i, 0.03 - 0.001 * i) for i in range(20)],
        1.0: [(100 + i, -0.10 + 0.001 * i, 0.02 - 0.001 * i) for i in range(20)],
    })

    section = _build_phase2_top_features_per_strength_section(df_pos, df_neg)

    assert isinstance(section, list)
    assert len(section) == 2
    assert [b["test_strength"] for b in section] == [0.5, 1.0]

    bucket = section[0]
    assert len(bucket["positive"]) == 25
    assert len(bucket["negative"]) == 20
    deltas_pos = [r["delta"] for r in bucket["positive"]]
    assert deltas_pos == sorted(deltas_pos, reverse=True)
    deltas_neg = [r["delta"] for r in bucket["negative"]]
    assert deltas_neg == sorted(deltas_neg, reverse=True)


def test_phase2_handles_empty_dfs():
    section = _build_phase2_top_features_per_strength_section(
        pd.DataFrame(), pd.DataFrame()
    )
    assert section == []


def test_phase2_handles_none_inputs():
    section = _build_phase2_top_features_per_strength_section(None, None)
    assert section == []


def test_phase2_handles_one_direction_empty():
    df_pos = _make_phase2_df({0.5: [(1, 0.18, 0.05)]})
    section = _build_phase2_top_features_per_strength_section(df_pos, pd.DataFrame())
    assert len(section) == 1
    assert section[0]["test_strength"] == 0.5
    assert len(section[0]["positive"]) == 1
    assert section[0]["negative"] == []


def test_phase2_emits_only_documented_fields():
    df_pos = _make_phase2_df({0.5: [(1, 0.18, 0.05)]})
    section = _build_phase2_top_features_per_strength_section(df_pos, pd.DataFrame())
    row = section[0]["positive"][0]
    assert set(row.keys()) == {
        "feature_id", "freq_diff", "delta", "is_causal",
        "n_improved", "n_worsened",
    }


def test_phase2_section_is_json_serialisable():
    df_pos = _make_phase2_df({0.5: [(1, 0.18, 0.05)]})
    df_neg = _make_phase2_df({0.5: [(100, -0.10, 0.03)]})
    section = _build_phase2_top_features_per_strength_section(df_pos, df_neg)
    json.dumps(section, cls=_NumpySafeEncoder)


def test_phase2_strengths_union_when_dfs_differ():
    """Defensive: if df_pos and df_neg have different strengths (cannot happen in
    practice — Phase 2 iterates the same list for both directions — but the
    builder takes the union to match _build_causal_screening_section's
    behaviour)."""
    df_pos = _make_phase2_df({0.5: [(1, 0.18, 0.05)]})
    df_neg = _make_phase2_df({1.0: [(100, -0.10, 0.03)]})
    section = _build_phase2_top_features_per_strength_section(df_pos, df_neg)
    assert [b["test_strength"] for b in section] == [0.5, 1.0]
    assert section[0]["positive"] != []
    assert section[0]["negative"] == []
    assert section[1]["positive"] == []
    assert section[1]["negative"] != []


def test_phase2_top_by_delta_implies_is_causal_when_data_is_clean():
    """Spec §1 claim: top-by-delta automatically satisfies is_causal=True
    at the head when delta > 0 for the top rows. This is the invariant
    Phase 3's causal_delta ranking relies on for reconstruction."""
    df_pos = _make_phase2_df({
        0.5: [(i, 0.18 - 0.001 * i, 0.05 - 0.001 * i) for i in range(30)],
    })
    section = _build_phase2_top_features_per_strength_section(df_pos, None)
    assert all(r["is_causal"] for r in section[0]["positive"])


def _minimal_experiment_config() -> ExperimentConfig:
    model = ModelConfig(
        model_id="hf/whatever", short_name="dummy",
        d_model=4, n_layers=2, dtype="float32",
    )
    sae = SAEConfig(
        name="dummy_sae", architecture="relu", repo="hf/dummy",
        filename="dummy.pt", layer=0, available_layers=[0],
        d_model=4, d_hidden=8, hook_point="resid_post", model="dummy",
    )
    benchmark = BenchmarkConfig(
        name="dummy_bench", dataset_name="dummy_dataset",
        choices=["A", "B"], answer_key_type="letter",
        field_mapping=FieldMapping(
            question="q", choices_format="list", choices_field="choices", answer="a",
        ),
        prompt_instruction="pick one", cache_name="dummy",
    )
    return ExperimentConfig(
        model=model, sae=sae, benchmark=benchmark,
        grid=GridConfig(strength=[0.5, 1.0]),
        causal=CausalConfig(causal_test_strength=[0.5, 1.0]),
    )


def _minimal_build_summary_kwargs(cfg: ExperimentConfig) -> dict:
    """Just enough to make build_summary return without crashing."""
    df_grid = pd.DataFrame([
        {
            "top_k_pos": 5, "top_k_neg": 0, "strength": 0.5,
            "ranking_criterion": "freq_diff", "weighting_mode": "uniform",
            "n_features_used": 5, "accuracy_grid_eval": 0.6,
            "delta_grid_eval": 0.05, "n_improved": 5, "n_worsened": 0,
            "net_change": 5,
        },
    ])
    full_reporting_results = {
        "filtered_exact": {"name": "Filtered Exact", "n_questions": 10,
            "baseline_acc": 0.5, "intervention_acc": 0.6, "delta": 0.1,
            "n_improved": 1, "n_worsened": 0, "binomial_p": 0.03},
        "filtered_approx": {"name": "Filtered Approx", "n_questions": 10,
            "baseline_acc": 0.5, "intervention_acc": 0.55, "delta": 0.05,
            "n_improved": 1, "n_worsened": 0, "binomial_p": 0.1},
        "full_exact": {"name": "Full Exact", "n_questions": 20,
            "baseline_acc": 0.4, "intervention_acc": 0.5, "delta": 0.1,
            "n_improved": 2, "n_worsened": 0, "binomial_p": 0.02},
        "full_approx": {"name": "Full Approx", "n_questions": 20,
            "baseline_acc": 0.4, "intervention_acc": 0.45, "delta": 0.05,
            "n_improved": 1, "n_worsened": 0, "binomial_p": 0.1},
    }
    return dict(
        experiment_config=cfg,
        sae_d_hidden=8, sae_top_k=None,
        all_questions_count=100, filtered_data_count=80,
        train_data_count=40, val_data_count=20,
        test_filtered_count=20, test_full_count=20,
        grid_eval_count=10, holdout_count=10, holdout_filtered_count=10,
        baseline_filtered_acc=0.5, baseline_full_acc=0.4,
        dynamic_min_occurrences=3,
        # tests override df_pos/df_neg when populated Phase-2 data is needed
        df_pos=pd.DataFrame(), df_neg=pd.DataFrame(),
        n_causal_pos=0, n_causal_neg=0,
        df_pair_stability=None,
        grid_config_dict=cfg.grid.model_dump(),
        n_combinations=1,
        best_config={
            "top_k_pos": 5, "top_k_neg": 0, "strength": 0.5,
            "ranking_criterion": "freq_diff", "weighting_mode": "uniform",
        },
        n_features_in_vec=5,
        full_reporting_results=full_reporting_results,
        difficulty_analysis={},
        kfold_results={"n_folds": 5, "mean_delta": 0.05, "std_delta": 0.02,
            "ci_95_lower": 0.0, "ci_95_upper": 0.1,
            "total_improved": 10, "total_worsened": 2,
            "binomial_p": 0.01, "n_positive_folds": 4,
            "fold_details": []},
        df_grid=df_grid,
    )


def test_build_summary_includes_phase1_and_phase2_sections():
    cfg = _minimal_experiment_config()
    pos = [_make_candidate(i, 0.30 - 0.01 * i, 50, 10) for i in range(30)]
    neg = [_make_candidate(100 + i, -0.20 + 0.01 * i, 5, 50) for i in range(20)]
    df_pos = _make_phase2_df({
        0.5: [(i, 0.30 - 0.01 * i, 0.05 - 0.001 * i) for i in range(30)],
        1.0: [(i, 0.30 - 0.01 * i, 0.04 - 0.001 * i) for i in range(30)],
    })
    df_neg = _make_phase2_df({
        0.5: [(100 + i, -0.20 + 0.01 * i, 0.03 - 0.001 * i) for i in range(20)],
        1.0: [(100 + i, -0.20 + 0.01 * i, 0.02 - 0.001 * i) for i in range(20)],
    })

    kwargs = _minimal_build_summary_kwargs(cfg)
    kwargs.update(
        df_pos=df_pos, df_neg=df_neg,
        phase1_pos_candidates=pos, phase1_neg_candidates=neg,
    )
    summary = build_summary(**kwargs)

    assert "phase1_top_features" in summary
    assert summary["phase1_top_features"]["n_kept"] == 25
    assert len(summary["phase1_top_features"]["positive"]) == 25
    assert len(summary["phase1_top_features"]["negative"]) == 20

    assert "phase2_top_features_per_strength" in summary
    assert [b["test_strength"] for b in summary["phase2_top_features_per_strength"]] == [0.5, 1.0]

    json.dumps(summary, cls=_NumpySafeEncoder)


def test_build_summary_phase1_defaults_to_empty_when_not_passed():
    cfg = _minimal_experiment_config()
    kwargs = _minimal_build_summary_kwargs(cfg)
    summary = build_summary(**kwargs)

    assert summary["phase1_top_features"] == {
        "n_kept": 25, "positive": [], "negative": [],
    }
    assert summary["phase2_top_features_per_strength"] == []
