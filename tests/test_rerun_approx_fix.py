"""Tests for scripts/rerun_approx_fix.py — the post-fix result repair script.

The script reconstructs the data split and the intervention vector from files already
on disk, recomputes only the approximation-dependent results, and rewrites them into
the existing summary JSON. The tests below cover, in order:

  * the pure helpers (override parsing, pre/post-fix detection, file discovery)
  * the central claim — that a vector rebuilt from the serialised feature tables is
    *identical* to the one the pipeline builds from the full Phase-1/Phase-2 objects,
    across all 90 grid cells, both weighting modes and single- vs multi-strength Phase 2
  * tie handling, which is where that claim is fragile: the Phase-2 sections are stored
    in ``nlargest(delta)`` order and pandas does not keep tied rows in input order, so
    the frames have to be sorted back into candidate order before anything is ranked
  * the offline guards in ``verify_offline`` — the two failure modes reconstruction
    genuinely has (too few causal survivors inside the 25-row cap, and a delta tie
    straddling that cap) plus the split checks
  * ``run_generalisation_shared``, checked against the ``_evaluate_single_benchmark``
    it replaces, on identical inputs
  * ``rerun_cell`` end to end against a synthetic experiment directory: which keys it
    rewrites, which it must leave untouched, and that a failed checksum writes nothing

A final integration test replays the reconstruction against the real committed
experiment directories; it skips when those are not present.
"""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from sae_experiments.generalisation import _evaluate_single_benchmark
from sae_experiments.intervention import build_intervention_vector
from sae_experiments.phases import _select_features_for_ranking
from sae_experiments.reporting import (
    _build_phase1_top_features_section,
    _build_phase2_top_features_per_strength_section,
)

# The script lives under scripts/, not in the package. Load it by path so the tests
# don't depend on scripts/ being importable.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "rerun_approx_fix.py"
_spec = importlib.util.spec_from_file_location("rerun_approx_fix", _SCRIPT_PATH)
rerun = importlib.util.module_from_spec(_spec)
sys.modules["rerun_approx_fix"] = rerun
_spec.loader.exec_module(rerun)

REPO_ROOT = Path(__file__).parent.parent

# The grid every final-matrix cell was run with (90 combinations).
GRID = {
    "top_k_positive": [3, 5, 7, 10, 15],
    "top_k_negative": [0, 3, 5],
    "strength": [0.5, 1.0, 1.5],
    "ranking_criterion": ["causal_delta", "freq_diff"],
    "weighting_mode": ["uniform"],
}
D_HIDDEN = 4096
CPU = torch.device("cpu")


# ── fixtures: Phase-1/Phase-2 tables in their real shapes ─────────


def make_tables(n_pos=30, n_neg=14, strength=1.0, causal_every=1):
    """Phase-1 candidates plus the Phase-2 DataFrames derived from them.

    Mirrors the real relationship: ``FeatureTracker.get_top_features`` emits candidates
    ordered by freq_diff (descending for positive, ascending for negative), and Phase 2
    appends exactly one row per candidate in that same order. Deltas repeat on purpose
    so tie-breaking is exercised, including across the 25-row serialisation cap.

    ``causal_every`` thins out the causal survivors: 1 keeps every positive delta > 0
    causal, 4 marks only every fourth one causal.
    """
    delta_cycle = [0.030, 0.020, 0.020, 0.010, 0.010, 0.010, 0.005, 0.002, 0.000, -0.005]

    pos_candidates, pos_rows = [], []
    for i in range(n_pos):
        freq = round(0.30 - 0.01 * i, 6)
        delta = delta_cycle[i % len(delta_cycle)]
        pos_candidates.append(
            (1000 + i, {"freq_diff": freq, "s_count": 60 - i, "f_count": 10 + i})
        )
        pos_rows.append({
            "feature_id": 1000 + i,
            "freq_diff": freq,
            "baseline_acc": 0.50,
            "intervention_acc": 0.50 + delta,
            "delta": delta,
            "n_improved": max(int(delta * 400), 0),
            "n_worsened": 1,
            "net_change": 0,
            "is_causal": bool(delta > 0 and i % causal_every == 0),
            "test_strength": strength,
        })

    neg_candidates, neg_rows = [], []
    for i in range(n_neg):
        freq = round(-0.25 + 0.01 * i, 6)
        delta = delta_cycle[(i + 3) % len(delta_cycle)]
        neg_candidates.append(
            (2000 + i, {"freq_diff": freq, "s_count": 5 + i, "f_count": 60 - i})
        )
        neg_rows.append({
            "feature_id": 2000 + i,
            "freq_diff": freq,
            "baseline_acc": 0.50,
            "intervention_acc": 0.50 + delta,
            "delta": delta,
            "n_improved": max(int(delta * 400), 0),
            "n_worsened": 1,
            "net_change": 0,
            "is_causal": bool(delta > 0 and i % causal_every == 0),
            "test_strength": strength,
        })

    return pd.DataFrame(pos_rows), pd.DataFrame(neg_rows), pos_candidates, neg_candidates


def serialise_tables(df_pos, df_neg, pos_candidates, neg_candidates):
    """The two summary sections the script reconstructs from, built by the real code."""
    return {
        "phase1_top_features": _build_phase1_top_features_section(
            pos_candidates, neg_candidates,
        ),
        "phase2_top_features_per_strength":
            _build_phase2_top_features_per_strength_section(df_pos, df_neg),
    }


def grid_rows():
    for kp, kn, s, rank, weight in itertools.product(
        GRID["top_k_positive"], GRID["top_k_negative"], GRID["strength"],
        GRID["ranking_criterion"], GRID["weighting_mode"],
    ):
        yield {
            "top_k_pos": kp, "top_k_neg": kn, "strength": s,
            "ranking_criterion": rank, "weighting_mode": weight,
        }


def reference_vector(df_pos, df_neg, row):
    """What the pipeline itself builds for this grid cell."""
    return build_intervention_vector(
        pos_features=_select_features_for_ranking(
            df_pos, row["ranking_criterion"], row["strength"]),
        neg_features=_select_features_for_ranking(
            df_neg, row["ranking_criterion"], row["strength"]),
        top_k_pos=row["top_k_pos"], top_k_neg=row["top_k_neg"],
        strength=row["strength"], ranking_criterion=row["ranking_criterion"],
        weighting_mode=row["weighting_mode"],
        d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32,
    )


# ── pure helpers ──────────────────────────────────────────────────


def test_parse_overrides_evaluates_literals_and_keeps_strings():
    parsed = rerun.parse_overrides([
        "causal.min_p_correct=0.15",
        "causal.causal_test_strength=[1.0]",
        "grid.weighting_mode=uniform",
    ])
    assert parsed["causal.min_p_correct"] == 0.15
    assert parsed["causal.causal_test_strength"] == [1.0]
    assert parsed["grid.weighting_mode"] == "uniform"


def test_parse_overrides_handles_values_containing_equals():
    parsed = rerun.parse_overrides(["benchmark.prefix=a=b"])
    assert parsed["benchmark.prefix"] == "a=b"


def test_is_pre_fix_splits_on_the_fix_commit_timestamp():
    """Real timestamps: EXP-330 ran before fed574a, EXP-336 right after it."""
    assert rerun.is_pre_fix({"timestamp": "2026-05-30T14:28:58.412531"})
    assert not rerun.is_pre_fix({"timestamp": "2026-07-23T22:31:05.000000"})
    assert not rerun.is_pre_fix({"timestamp": rerun.FIX_TIMESTAMP})


def test_is_pre_fix_treats_a_missing_timestamp_as_pre_fix():
    assert rerun.is_pre_fix({})


def test_summary_path_ignores_the_screening_cache(tmp_path):
    (tmp_path / "logit_screening_arc_challenge_gemma2_2b.json").write_text("[]")
    summary = tmp_path / "arc_challenge_batchtopk_L1_K50_20260530_142858.json"
    summary.write_text("{}")
    assert rerun.summary_path(tmp_path) == summary


def test_summary_path_returns_none_when_absent_or_ambiguous(tmp_path):
    assert rerun.summary_path(tmp_path) is None
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    assert rerun.summary_path(tmp_path) is None


def test_screening_cache_path_requires_exactly_one_cache(tmp_path):
    assert rerun.screening_cache_path(tmp_path) is None
    one = tmp_path / "logit_screening_arc_challenge_gemma2_2b.json"
    one.write_text("[]")
    assert rerun.screening_cache_path(tmp_path) == one
    (tmp_path / "logit_screening_commonsenseqa_gemma2_2b.json").write_text("[]")
    assert rerun.screening_cache_path(tmp_path) is None


# ── vector reconstruction: the central equivalence claim ──────────


def test_rebuilt_vector_is_identical_for_every_grid_cell():
    """Reconstruction from the serialised tables must equal what the pipeline built.

    Single-strength Phase 2 (one bucket at 1.0) reused across grid strengths
    0.5/1.0/1.5 — exactly the configuration every final-matrix cell ran with.
    """
    df_pos, df_neg, pos_c, neg_c = make_tables()
    frames = rerun.feature_frames(serialise_tables(df_pos, df_neg, pos_c, neg_c))

    checked = 0
    for row in grid_rows():
        got = rerun.build_vector(frames, row, d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32)
        assert torch.equal(got, reference_vector(df_pos, df_neg, row)), (
            f"vector mismatch for {row}"
        )
        checked += 1
    assert checked == 90


def test_rebuilt_vector_is_identical_under_proportional_weighting():
    """Proportional weighting reads the score values, not just the ranking."""
    df_pos, df_neg, pos_c, neg_c = make_tables()
    frames = rerun.feature_frames(serialise_tables(df_pos, df_neg, pos_c, neg_c))

    for row in grid_rows():
        row = {**row, "weighting_mode": "proportional"}
        got = rerun.build_vector(frames, row, d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32)
        assert torch.equal(got, reference_vector(df_pos, df_neg, row)), (
            f"vector mismatch for {row}"
        )


def test_rebuilt_vector_is_identical_with_multi_strength_phase2():
    """The test-strength side study ran Phase 2 at three strengths; buckets are keyed."""
    frames_pos, frames_neg = [], []
    for s in (0.5, 1.0, 1.5):
        p, n, pc, nc = make_tables(strength=s)
        frames_pos.append(p)
        frames_neg.append(n)
    df_pos = pd.concat(frames_pos, ignore_index=True)
    df_neg = pd.concat(frames_neg, ignore_index=True)
    _, _, pos_c, neg_c = make_tables()
    frames = rerun.feature_frames(serialise_tables(df_pos, df_neg, pos_c, neg_c))

    for row in grid_rows():
        got = rerun.build_vector(frames, row, d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32)
        assert torch.equal(got, reference_vector(df_pos, df_neg, row)), (
            f"vector mismatch for {row}"
        )


def test_select_features_rejects_an_unknown_ranking_criterion():
    df_pos, df_neg, pos_c, neg_c = make_tables()
    frames = rerun.feature_frames(serialise_tables(df_pos, df_neg, pos_c, neg_c))
    with pytest.raises(ValueError, match="Unknown ranking_criterion"):
        rerun.select_features(frames, "nonsense", 1.0)


# ── offline guards ────────────────────────────────────────────────


def make_screening(n=60):
    """Screening entries with a p_correct spread that straddles the 0.15–0.45 filter."""
    return [
        {
            "q_idx": i,
            "p_correct": round((i % 20) / 20.0, 6),
            "max_confidence": 0.9,
            "prediction": "A",
            "correct": bool(i % 3),
            "expected": "A",
            "_sample": {"question": f"q{i}", "answer": "A"},
        }
        for i in range(n)
    ]


class Causal:
    """The causal-config fields rebuild_split reads."""

    min_p_correct = 0.15
    max_p_correct = 0.45
    train_fraction = 0.50
    val_fraction = 0.25
    seed = 42


def make_summary(screening, df_pos, df_neg, pos_c, neg_c, best=None, timestamp="2026-05-30T14:28:58"):
    """A summary JSON consistent with the given screening data and feature tables.

    data_stats is derived through the script's own rebuild_split. That is fine here —
    these tests target write-back behaviour and guard behaviour; the fidelity of
    rebuild_split itself is pinned by test_real_experiment_dirs_reconstruct_exactly.
    """
    split = rerun.rebuild_split(screening, Causal)
    holdout, filtered = split["holdout"], split["holdout_filtered"]
    best = best or {
        "top_k_pos": 3, "top_k_neg": 0, "strength": 1.0,
        "ranking_criterion": "causal_delta", "weighting_mode": "uniform",
    }
    frames = rerun.feature_frames(serialise_tables(df_pos, df_neg, pos_c, neg_c))
    n_best = int((rerun.build_vector(
        frames, best, d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32) != 0).sum().item())

    all_grid = []
    for row in grid_rows():
        vec = rerun.build_vector(frames, row, d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32)
        all_grid.append({
            **row,
            "n_features_used": int((vec != 0).sum().item()),
            "accuracy_grid_eval": 0.5, "delta_grid_eval": 0.01,
            "n_improved": 3, "n_worsened": 2, "net_change": 1,
        })

    def acc(entries):
        return sum(e["correct"] for e in entries) / len(entries) if entries else 0.0

    summary = {
        "timestamp": timestamp,
        "experiment_version": "23_full_reporting",
        "model": {"model_id": "google/gemma-2-2b", "short_name": "gemma2_2b",
                  "d_model": 2304, "n_layers": 26},
        "sae": {"name": "fake", "architecture": "batchtopk", "repo": "x/y",
                "layer": 1, "d_hidden": D_HIDDEN, "top_k": 50},
        "benchmark": {"name": "ARC-Challenge", "dataset": "allenai/ai2_arc",
                      "config": "ARC-Challenge", "choices": ["A", "B", "C", "D"]},
        "data_stats": {
            "total_questions": len(screening),
            "filtered_questions": len(split["filtered"]),
            "p_correct_range": [Causal.min_p_correct, Causal.max_p_correct],
            "train_questions": len(split["train"]),
            "val_questions": len(split["val"]),
            "test_filtered_questions": 0,
            "test_full_questions": len(split["test_full"]),
            "grid_eval_questions": len(split["grid_eval"]),
            "holdout_questions": len(holdout),
            "holdout_filtered_questions": len(filtered),
            "baseline_filtered_accuracy": acc(filtered),
            "baseline_full_accuracy": acc(holdout),
            "dynamic_min_occurrences": 3,
        },
        "causal_screening": {
            "n_unique_tested_positive": int(df_pos["feature_id"].nunique()) if len(df_pos) else 0,
            "n_unique_tested_negative": int(df_neg["feature_id"].nunique()) if len(df_neg) else 0,
            "n_causal_positive_unique_features": int(df_pos["is_causal"].sum()) if len(df_pos) else 0,
            "n_causal_negative_unique_features": int(df_neg["is_causal"].sum()) if len(df_neg) else 0,
            "per_strength": [],
        },
        "best_config": {**best, "n_features_used": n_best},
        "full_reporting": {
            "filtered_exact": _report("Filtered + Exact", len(filtered), acc(filtered), 0.02, 5, 2),
            "full_exact": _report("Full + Exact", len(holdout), acc(holdout), 0.01, 7, 4),
            "filtered_approx": _report("Filtered + Approximated", len(filtered),
                                       acc(filtered), -0.30, 2, 40),
            "full_approx": _report("Full + Approximated", len(holdout),
                                   acc(holdout), -0.31, 3, 60),
        },
        "approximation_comparison": {
            "filtered": {"exact_delta": 0.02, "approx_delta": -0.30, "difference": -0.32},
            "full": {"exact_delta": 0.01, "approx_delta": -0.31, "difference": -0.32},
        },
        "difficulty_analysis": {"difficulty_ranges": [], "totals": {}},
        "kfold_results": {"n_folds": 5, "mean_delta": 0.01},
        "all_grid_results": all_grid,
        "ranking_comparison": {},
        "generalisation": {
            "benchmarks": [{
                "benchmark": "HellaSwag", "dataset": "hellaswag", "n_questions": 10,
                "n_choices": 4, "baseline_acc": 0.4, "intervention_acc": 0.1,
                "delta": -0.3, "n_improved": 0, "n_worsened": 3,
                "binomial_p": 1.0, "significant": False,
            }],
            "summary": {"n_benchmarks": 1, "n_positive": 0,
                        "n_significant": 0, "avg_delta": -0.3},
        },
    }
    summary.update(serialise_tables(df_pos, df_neg, pos_c, neg_c))
    return summary


def _report(name, n, baseline_acc, delta, improved, worsened):
    return {
        "name": name, "n_questions": n, "baseline_acc": baseline_acc,
        "intervention_acc": baseline_acc + delta, "delta": delta,
        "n_improved": improved, "n_worsened": worsened,
        "binomial_p": 0.5, "significant": False,
    }


def test_verify_offline_accepts_a_consistent_summary():
    screening = make_screening()
    summary = make_summary(screening, *make_tables())
    ok, split, problems, _warn = rerun.verify_offline(summary, screening, Causal)
    assert ok, problems
    assert len(split["holdout"]) == summary["data_stats"]["holdout_questions"]


def test_verify_offline_rejects_a_wrong_holdout_count():
    screening = make_screening()
    summary = make_summary(screening, *make_tables())
    summary["data_stats"]["holdout_questions"] += 1
    ok, _split, problems, _warn = rerun.verify_offline(summary, screening, Causal)
    assert not ok
    assert any("holdout_questions" in p for p in problems)


def test_verify_offline_rejects_a_single_flipped_baseline_answer():
    """The accuracy check is what proves question-for-question identity of the split."""
    screening = make_screening()
    summary = make_summary(screening, *make_tables())
    holdout_ids = {e["q_idx"] for e in rerun.rebuild_split(screening, Causal)["holdout"]}
    for entry in screening:
        if entry["q_idx"] in holdout_ids:
            entry["correct"] = not entry["correct"]
            break

    ok, _split, problems, _warn = rerun.verify_offline(summary, screening, Causal)
    assert not ok
    assert any("baseline_full_accuracy" in p for p in problems)


def test_verify_offline_rejects_a_grid_vector_mismatch():
    screening = make_screening()
    summary = make_summary(screening, *make_tables())
    summary["all_grid_results"][0]["n_features_used"] += 1
    ok, _split, problems, _warn = rerun.verify_offline(summary, screening, Causal)
    assert not ok
    assert any("grid vectors disagree" in p for p in problems)


def test_verify_offline_catches_too_few_causal_survivors_in_the_cap():
    """Reconstruction's one real failure mode, and proof the guard sees it.

    Only every fourth positive feature is causal here, so the top-25-by-delta slice
    that gets serialised cannot supply the 15 causal features a top_k_pos=15 cell
    selected from the full table. The rebuilt vector is short, and n_features_used
    disagrees — which is exactly what stops the script before it writes anything.
    """
    df_pos, df_neg, pos_c, neg_c = make_tables(n_pos=80, n_neg=40, causal_every=4)
    frames = rerun.feature_frames(serialise_tables(df_pos, df_neg, pos_c, neg_c))
    row = {"top_k_pos": 15, "top_k_neg": 5, "strength": 1.0,
           "ranking_criterion": "causal_delta", "weighting_mode": "uniform"}

    got = rerun.build_vector(frames, row, d_hidden=D_HIDDEN, device=CPU, dtype=torch.float32)
    ref = reference_vector(df_pos, df_neg, row)
    assert int((got != 0).sum()) < int((ref != 0).sum()), (
        "expected the serialisation cap to starve this cell of causal features"
    )

    screening = make_screening()
    summary = make_summary(screening, df_pos, df_neg, pos_c, neg_c)
    summary["all_grid_results"] = [{**row, "n_features_used": int((ref != 0).sum().item())}]
    ok, _split, problems, _warn = rerun.verify_offline(summary, screening, Causal)
    assert not ok
    assert any("grid vectors disagree" in p for p in problems)


# ── tie handling across the serialisation cap ─────────────────────


def _tied_tables(n_pos=40, delta=0.010):
    """Every positive feature causal with the *same* delta, so the cap cuts a tie."""
    candidates, rows = [], []
    for i in range(n_pos):
        freq = round(0.40 - 0.001 * i, 6)
        candidates.append((1000 + i, {"freq_diff": freq, "s_count": 60, "f_count": 10}))
        rows.append({
            "feature_id": 1000 + i, "freq_diff": freq, "baseline_acc": 0.50,
            "intervention_acc": 0.50 + delta, "delta": delta, "n_improved": 4,
            "n_worsened": 1, "net_change": 3, "is_causal": True, "test_strength": 1.0,
        })
    return pd.DataFrame(rows), pd.DataFrame(), candidates, []


def test_feature_frames_restores_the_candidate_order_of_phase2_buckets():
    """nlargest scrambles tied deltas on the way out; the frames must undo that.

    Positive rows come back in descending freq_diff and negative rows in ascending
    freq_diff — the order FeatureTracker emitted them in, which is the order whose row
    positions Phase 3 used to break delta ties.
    """
    df_pos, df_neg, pos_c, neg_c = make_tables()
    serialised = serialise_tables(df_pos, df_neg, pos_c, neg_c)
    raw_pos = [r["feature_id"] for r in serialised["phase2_top_features_per_strength"][0]["positive"]]

    _fp, _fn, buckets = rerun.feature_frames(serialised)
    bucket_pos, bucket_neg = buckets[1.0]

    assert list(bucket_pos["freq_diff"]) == sorted(bucket_pos["freq_diff"], reverse=True)
    assert list(bucket_neg["freq_diff"]) == sorted(bucket_neg["freq_diff"])
    assert list(bucket_pos["feature_id"]) != raw_pos, (
        "fixture should exercise the reordering, otherwise this test proves nothing"
    )
    # Candidate order is feature_id order in this fixture, so it is directly checkable.
    assert list(bucket_pos["feature_id"]) == sorted(bucket_pos["feature_id"])


def test_no_ambiguity_warning_when_ties_stay_clear_of_the_cap():
    screening = make_screening()
    df_pos, df_neg, pos_c, neg_c = make_tables()
    summary = make_summary(screening, df_pos, df_neg, pos_c, neg_c)
    ok, _split, problems, warnings = rerun.verify_offline(summary, screening, Causal)
    assert ok, problems
    assert warnings == []


def test_ambiguity_is_reported_when_a_delta_tie_straddles_the_cap():
    """Size-correct but not member-provable: flagged for the exact checksum to settle."""
    screening = make_screening()
    df_pos, df_neg, pos_c, neg_c = _tied_tables()
    summary = make_summary(screening, df_pos, df_neg, pos_c, neg_c)

    ok, _split, problems, warnings = rerun.verify_offline(summary, screening, Causal)
    assert ok, problems              # counts all still line up
    assert any("positive causal_delta" in w for w in warnings), warnings
    assert any("tie at the 25-row cap" in w for w in warnings), warnings


def test_ambiguity_is_not_reported_when_the_best_config_ranks_by_freq_diff():
    """The check is scoped to the best config — the only vector the script builds.

    Five real cells (EXP-303/307/309/328/358) have causal_delta ties at the cap in grid
    cells they never selected; flagging those would be noise.
    """
    screening = make_screening()
    df_pos, df_neg, pos_c, neg_c = _tied_tables()
    best = {"top_k_pos": 3, "top_k_neg": 0, "strength": 1.0,
            "ranking_criterion": "freq_diff", "weighting_mode": "uniform"}
    summary = make_summary(screening, df_pos, df_neg, pos_c, neg_c, best=best)

    _ok, _split, _problems, warnings = rerun.verify_offline(summary, screening, Causal)
    assert warnings == []


# ── split reconstruction properties ───────────────────────────────


def test_rebuild_split_is_deterministic_and_partitions_the_data():
    screening = make_screening()
    a = rerun.rebuild_split(screening, Causal)
    b = rerun.rebuild_split(screening, Causal)

    for key in a:
        assert [e["q_idx"] for e in a[key]] == [e["q_idx"] for e in b[key]], key

    train = {e["q_idx"] for e in a["train"]}
    val = {e["q_idx"] for e in a["val"]}
    grid_eval = {e["q_idx"] for e in a["grid_eval"]}
    holdout = {e["q_idx"] for e in a["holdout"]}
    filtered = {e["q_idx"] for e in a["filtered"]}

    assert not (train & val)
    assert not (grid_eval & holdout)
    assert grid_eval | holdout == {e["q_idx"] for e in a["test_full"]}
    assert not ((grid_eval | holdout) & (train | val)), "holdout must never touch train/val"
    assert {e["q_idx"] for e in a["holdout_filtered"]} == holdout & filtered
    assert all(
        Causal.min_p_correct <= e["p_correct"] <= Causal.max_p_correct for e in a["filtered"]
    )


# ── shared-baseline generalisation ────────────────────────────────


class FakeSAE:
    """Affine decoder with a large bias — the thing the fix removes."""

    d_hidden = D_HIDDEN
    dtype = torch.float32

    def __init__(self, d_model=8):
        torch.manual_seed(0)
        self.W = torch.randn(D_HIDDEN, d_model)
        self.b = torch.full((d_model,), 12.5)

    def decode(self, x):
        return x @ self.W + self.b


def _fake_predictions(correct_flags):
    return [{"correct": bool(c), "prob_correct": 0.5} for c in correct_flags]


@pytest.fixture
def gen_stubs(monkeypatch, tmp_path):
    """Deterministic benchmark + prediction stubs shared by the generalisation tests."""
    questions = [{"question": f"q{i}"} for i in range(20)]
    baseline = [i % 2 == 0 for i in range(20)]          # 10 correct
    intervened = [i % 3 == 0 for i in range(20)]        # 7 correct

    class Bench:
        name = "HellaSwag"
        dataset_name = "hellaswag"
        cache_name = "hellaswag"
        choices = ["A", "B", "C", "D"]

    calls = {"baseline": 0, "intervention": 0}

    def fake_load_benchmark(cfg):
        return questions

    def fake_baseline(qs, **kwargs):
        calls["baseline"] += 1
        return _fake_predictions(baseline)

    def fake_intervention(qs, fn, **kwargs):
        calls["intervention"] += 1
        return _fake_predictions(intervened)

    def fake_answer_ids(choices, tokenizer):
        return {c: i for i, c in enumerate(choices)}

    for mod in (rerun, sys.modules["sae_experiments.generalisation"]):
        monkeypatch.setattr(mod, "load_benchmark", fake_load_benchmark, raising=False)
        monkeypatch.setattr(mod, "batched_logit_predictions", fake_baseline, raising=False)
        monkeypatch.setattr(mod, "batched_intervention_predictions", fake_intervention,
                            raising=False)
        monkeypatch.setattr(mod, "get_answer_token_ids", fake_answer_ids, raising=False)
    monkeypatch.setattr(rerun, "_load_benchmark_config", lambda path: Bench())
    monkeypatch.setattr(rerun, "GEN_BASELINE_DIR", tmp_path / "gen_baselines")

    return {"bench": Bench(), "questions": questions, "calls": calls}


def test_shared_generalisation_matches_the_function_it_replaces(gen_stubs):
    """Same inputs, same numbers as _evaluate_single_benchmark — every field."""
    vec = torch.zeros(8)
    common = dict(model=object(), tokenizer=object(), hook_mgr=object(),
                  sae=FakeSAE(), device=CPU, batch_size=1)

    reference = _evaluate_single_benchmark(
        gen_stubs["bench"], lambda *a, **k: None, **common,
    )
    got = rerun.run_generalisation_shared(
        vec, ["configs/benchmarks/hellaswag.yaml"],
        model_short_name="gemma2_2b", **common,
    )

    assert got["summary"]["n_benchmarks"] == 1
    assert got["benchmarks"][0] == reference


def test_shared_generalisation_summary_aggregates_like_run_generalisation(gen_stubs):
    got = rerun.run_generalisation_shared(
        torch.zeros(8), ["a.yaml", "b.yaml"], model_short_name="gemma2_2b",
        model=object(), tokenizer=object(), hook_mgr=object(), sae=FakeSAE(),
        device=CPU, batch_size=1,
    )
    deltas = [b["delta"] for b in got["benchmarks"]]
    assert got["summary"] == {
        "n_benchmarks": 2,
        "n_positive": sum(1 for d in deltas if d > 0),
        "n_significant": sum(1 for b in got["benchmarks"] if b["significant"]),
        "avg_delta": sum(deltas) / len(deltas),
    }


def test_generalisation_baseline_is_computed_once_and_then_reused(gen_stubs):
    """The baseline depends on neither SAE nor layer, so cells of a model share it."""
    kwargs = dict(model_short_name="gemma2_2b", model=object(), tokenizer=object(),
                  hook_mgr=object(), sae=FakeSAE(), device=CPU, batch_size=1)
    first = rerun.run_generalisation_shared(torch.zeros(8), ["a.yaml"], **kwargs)
    assert gen_stubs["calls"]["baseline"] == 1

    second = rerun.run_generalisation_shared(torch.zeros(8), ["a.yaml"], **kwargs)
    assert gen_stubs["calls"]["baseline"] == 1, "baseline pass should have been cached"
    assert gen_stubs["calls"]["intervention"] == 2, "intervention must run per cell"
    assert first["benchmarks"][0]["baseline_acc"] == second["benchmarks"][0]["baseline_acc"]


def test_generalisation_baseline_cache_is_rejected_when_the_size_changed(gen_stubs, tmp_path):
    cache_dir = tmp_path / "gen_baselines"
    cache_dir.mkdir(parents=True)
    (cache_dir / "gemma2_2b__hellaswag.json").write_text(json.dumps({"n": 3, "correct": [1, 0, 1]}))

    rerun.run_generalisation_shared(
        torch.zeros(8), ["a.yaml"], model_short_name="gemma2_2b", model=object(),
        tokenizer=object(), hook_mgr=object(), sae=FakeSAE(), device=CPU, batch_size=1,
    )
    assert gen_stubs["calls"]["baseline"] == 1, "stale cache must trigger a recompute"


# ── as_summary_report ─────────────────────────────────────────────


def test_as_summary_report_matches_the_build_summary_shape():
    report = {
        "name": "Full + Approximated", "n_questions": 738, "baseline_acc": 0.58,
        "intervention_acc": 0.60, "delta": 0.02, "n_improved": 10,
        "n_worsened": 4, "binomial_p": 0.0123,
    }
    out = rerun.as_summary_report(report)
    assert set(out) == {
        "name", "n_questions", "baseline_acc", "intervention_acc", "delta",
        "n_improved", "n_worsened", "binomial_p", "significant",
    }
    assert out["significant"] is True
    assert rerun.as_summary_report({**report, "binomial_p": 0.9})["significant"] is False
    assert rerun.as_summary_report({**report, "binomial_p": None})["binomial_p"] is None


# ── rerun_cell end to end ─────────────────────────────────────────


def make_args(**overrides):
    base = dict(dry_run=False, checksum_only=False, force=False, skip_generalisation=False,
                verbatim_generalisation=False, allow_exact_drift=False,
                no_provenance=False, reuse_model=False)
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def cell(tmp_path, monkeypatch):
    """A synthetic EXP-999 directory plus every GPU-side dependency stubbed out."""
    from run_experiments_final import Experiment

    monkeypatch.setattr(rerun, "ROOT", tmp_path)
    exp_dir = tmp_path / "results" / "EXP-999"
    exp_dir.mkdir(parents=True)

    screening = make_screening()
    df_pos, df_neg, pos_c, neg_c = make_tables()
    summary = make_summary(screening, df_pos, df_neg, pos_c, neg_c)

    summary_file = exp_dir / "arc_challenge_batchtopk_L1_K50_20260530_142858.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    (exp_dir / "logit_screening_arc_challenge_gemma2_2b.json").write_text(json.dumps(screening))

    # Absolute config paths survive `ROOT / spec.sae` while ROOT points at tmp_path.
    spec = Experiment(
        name="EXP-999: fake cell",
        sae=str(REPO_ROOT / "configs" / "saes" / "chanind_gemma2_2b_tied_topk_k50.yaml"),
        layer=1,
        benchmark=str(REPO_ROOT / "configs" / "benchmarks" / "arc_challenge.yaml"),
        overrides=["causal.min_p_correct=0.15", "causal.max_p_correct=0.45",
                   "causal.causal_test_strength=[1.0]"],
        batch_size=1,
        generalisation=["configs/benchmarks/hellaswag.yaml"],
    )

    sae = FakeSAE()
    seen = {}

    class FakeHookManager:
        def __init__(self, layer_module):
            self.removed = False

        def register_cache_hook(self):
            pass

        def remove_cache_hook(self):
            self.removed = True

    def fake_eval_full(name, data, baseline_dict, baseline_acc, vec, mode, kw, approx_vec):
        seen[mode + "_full"] = {"vec": vec.clone(),
                                "approx": None if approx_vec is None else approx_vec.clone()}
        stored = summary["full_reporting"]["full_exact"]
        if mode == "exact":  # reproduce the recorded exact report by default
            return {"name": name, "n_questions": len(data),
                    "baseline_acc": baseline_acc,
                    "intervention_acc": stored["intervention_acc"],
                    "delta": stored["delta"], "n_improved": stored["n_improved"],
                    "n_worsened": stored["n_worsened"], "binomial_p": 0.5}
        return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                "intervention_acc": baseline_acc + 0.012, "delta": 0.012,
                "n_improved": 9, "n_worsened": 3, "binomial_p": 0.04}

    def fake_eval(name, data, baseline_results, baseline_acc, vec, mode, kw, approx_vec):
        seen[mode + "_filtered"] = {"approx": None if approx_vec is None else approx_vec.clone()}
        return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                "intervention_acc": baseline_acc + 0.021, "delta": 0.021,
                "n_improved": 4, "n_worsened": 1, "binomial_p": 0.2}

    def fake_generalisation(approx_vec, paths, **kwargs):
        seen["gen_vec"] = approx_vec.clone()
        return {
            "benchmarks": [{
                "benchmark": "HellaSwag", "dataset": "hellaswag", "n_questions": 10,
                "n_choices": 4, "baseline_acc": 0.4, "intervention_acc": 0.42,
                "delta": 0.02, "n_improved": 3, "n_worsened": 1,
                "binomial_p": 0.3, "significant": False,
            }],
            "summary": {"n_benchmarks": 1, "n_positive": 1,
                        "n_significant": 0, "avg_delta": 0.02},
        }

    monkeypatch.setattr(rerun, "load_model_and_tokenizer",
                        lambda cfg: (object(), object(), CPU))
    monkeypatch.setattr(rerun, "load_sae", lambda cfg, device: sae)
    monkeypatch.setattr(rerun, "get_layer_module",
                        lambda m, layer, hook, expected_n_layers=None: object())
    monkeypatch.setattr(rerun, "HookManager", FakeHookManager)
    monkeypatch.setattr(rerun, "get_answer_token_ids",
                        lambda choices, tok: {c: i for i, c in enumerate(choices)})
    monkeypatch.setattr(rerun, "_evaluate_report_full", fake_eval_full)
    monkeypatch.setattr(rerun, "_evaluate_report", fake_eval)
    monkeypatch.setattr(rerun, "run_generalisation_shared", fake_generalisation)
    monkeypatch.setattr(rerun, "generate_all_plots", lambda **kw: "plot.png")

    return {"spec": spec, "dir": exp_dir, "file": summary_file, "sae": sae,
            "summary": summary, "seen": seen, "screening": screening}


def test_rerun_cell_repairs_only_the_affected_keys(cell):
    before = json.loads(cell["file"].read_text())
    status = rerun.rerun_cell(cell["spec"], make_args(), {})
    assert status == "repaired"
    after = json.loads(cell["file"].read_text())

    changed = {k for k in after if k not in before or after[k] != before[k]}
    assert changed == {"full_reporting", "approximation_comparison",
                       "generalisation", "approximation_fix"}

    # Within full_reporting only the two approximated entries moved.
    assert after["full_reporting"]["filtered_exact"] == before["full_reporting"]["filtered_exact"]
    assert after["full_reporting"]["full_exact"] == before["full_reporting"]["full_exact"]
    assert after["full_reporting"]["full_approx"]["delta"] == pytest.approx(0.012)
    assert after["full_reporting"]["filtered_approx"]["delta"] == pytest.approx(0.021)
    assert after["full_reporting"]["full_approx"]["significant"] is True


def test_rerun_cell_recomputes_the_approximation_comparison_from_both_halves(cell):
    rerun.rerun_cell(cell["spec"], make_args(), {})
    after = json.loads(cell["file"].read_text())

    exact_full = after["full_reporting"]["full_exact"]["delta"]
    exact_filt = after["full_reporting"]["filtered_exact"]["delta"]
    comparison = after["approximation_comparison"]
    assert comparison["full"]["exact_delta"] == pytest.approx(exact_full)
    assert comparison["full"]["approx_delta"] == pytest.approx(0.012)
    assert comparison["full"]["difference"] == pytest.approx(0.012 - exact_full)
    assert comparison["filtered"]["difference"] == pytest.approx(0.021 - exact_filt)


def test_rerun_cell_steers_with_the_bias_free_vector(cell):
    """The whole point of the fix: the hidden vector must be s @ W_dec, no b_dec."""
    rerun.rerun_cell(cell["spec"], make_args(), {})
    sae = cell["sae"]
    feature_vec = cell["seen"]["approx_full"]["vec"]
    expected = feature_vec @ sae.W                      # bias-free displacement

    for key in ("approx_full", "approx_filtered"):
        assert torch.allclose(cell["seen"][key]["approx"], expected, atol=1e-5)
    assert torch.allclose(cell["seen"]["gen_vec"], expected, atol=1e-5)
    # The buggy version would have added b_dec on top.
    assert not torch.allclose(cell["seen"]["approx_full"]["approx"], expected + sae.b, atol=1e-5)


def test_bias_free_check_passes_on_the_current_implementation():
    """fed574a's decode(s) - decode(0) displaces a zero intervention by exactly zero."""
    assert rerun.bias_free_displacement_error(FakeSAE(), CPU) == 0.0


def test_bias_free_check_catches_the_pre_fix_implementation(monkeypatch):
    """The pre-fix decode(s) returns b_dec for a zero intervention."""
    monkeypatch.setattr(
        rerun, "compute_approximated_hidden_vector",
        lambda vec, sae: sae.decode(vec.unsqueeze(0)).squeeze(0),
    )
    assert rerun.bias_free_displacement_error(FakeSAE(), CPU) == pytest.approx(12.5)


def test_rerun_cell_refuses_to_run_from_a_pre_fix_checkout(cell, monkeypatch):
    """A stale clone must not rewrite results with the same bug and label them fixed.

    fix_commit is a hardcoded constant and proves nothing on its own; this guard is what
    makes the claim true.
    """
    monkeypatch.setattr(
        rerun, "compute_approximated_hidden_vector",
        lambda vec, sae: sae.decode(vec.unsqueeze(0)).squeeze(0),
    )
    before = cell["file"].read_bytes()

    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "failed"
    assert cell["file"].read_bytes() == before
    assert "approx_full" not in cell["seen"], "must refuse before any approximated pass"
    assert "exact_full" in cell["seen"], "the exact checksum still runs first"


def test_provenance_records_the_verified_bias_free_check(cell):
    rerun.rerun_cell(cell["spec"], make_args(), {})
    prov = json.loads(cell["file"].read_text())["approximation_fix"]
    assert prov["approximation_verified_bias_free"] is True


def test_rerun_cell_passes_none_as_the_approx_vector_to_the_exact_checksum(cell):
    rerun.rerun_cell(cell["spec"], make_args(), {})
    assert cell["seen"]["exact_full"]["approx"] is None


def test_rerun_cell_writes_nothing_when_the_exact_checksum_fails(cell, monkeypatch):
    def drifting(name, data, baseline_dict, baseline_acc, vec, mode, kw, approx_vec):
        if mode == "exact":
            return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                    "intervention_acc": 0.99, "delta": 0.4,
                    "n_improved": 999, "n_worsened": 0, "binomial_p": 0.01}
        raise AssertionError("approximated evaluation must not run after a failed checksum")

    monkeypatch.setattr(rerun, "_evaluate_report_full", drifting)
    before = cell["file"].read_text()

    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "failed"
    assert cell["file"].read_text() == before, "a failed checksum must leave the file untouched"


def test_rerun_cell_can_be_forced_past_a_drifting_checksum(cell, monkeypatch):
    def drifting(name, data, baseline_dict, baseline_acc, vec, mode, kw, approx_vec):
        if mode == "exact":
            return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                    "intervention_acc": 0.99, "delta": 0.4,
                    "n_improved": 999, "n_worsened": 0, "binomial_p": 0.01}
        return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                "intervention_acc": baseline_acc + 0.012, "delta": 0.012,
                "n_improved": 9, "n_worsened": 3, "binomial_p": 0.04}

    monkeypatch.setattr(rerun, "_evaluate_report_full", drifting)
    assert rerun.rerun_cell(cell["spec"], make_args(allow_exact_drift=True), {}) == "repaired"
    after = json.loads(cell["file"].read_text())
    assert after["approximation_fix"]["exact_checksum"] == "drifted"


def test_a_drifting_checksum_records_how_big_the_drift_was(cell, monkeypatch):
    """``"drifted"`` on its own is not auditable — the size decides whether it matters.

    For the andyrdt_llama_k64 cells the drift is 2-8 questions against measured effects
    of 4-31, so the magnitude is what separates "negligible" from "the noise exceeds the
    signal". Without it in the JSON that number survives only in an uncommitted console
    log.
    """
    stored = cell["summary"]["full_reporting"]["full_exact"]
    n = stored["n_questions"]
    drifted_acc = stored["intervention_acc"] + 2 / n  # exactly two questions

    def drifting(name, data, baseline_dict, baseline_acc, vec, mode, kw, approx_vec):
        if mode == "exact":
            return {"name": name, "n_questions": n, "baseline_acc": baseline_acc,
                    "intervention_acc": drifted_acc, "delta": 0.4,
                    "n_improved": stored["n_improved"] + 2,
                    "n_worsened": stored["n_worsened"], "binomial_p": 0.01}
        return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                "intervention_acc": baseline_acc + 0.012, "delta": 0.012,
                "n_improved": 9, "n_worsened": 3, "binomial_p": 0.04}

    monkeypatch.setattr(rerun, "_evaluate_report_full", drifting)
    rerun.rerun_cell(cell["spec"], make_args(allow_exact_drift=True), {})

    d = json.loads(cell["file"].read_text())["approximation_fix"]["exact_checksum_drift"]
    assert d["drift_questions"] == 2
    assert d["n_questions"] == n
    assert d["stored"]["intervention_acc"] == stored["intervention_acc"]
    assert d["stored"]["n_improved"] == stored["n_improved"]
    assert d["recomputed"]["intervention_acc"] == pytest.approx(drifted_acc)
    assert d["recomputed"]["n_improved"] == stored["n_improved"] + 2


def test_a_reproducing_checksum_records_no_drift(cell):
    """An explicit null says "checked, zero", which a missing key does not."""
    rerun.rerun_cell(cell["spec"], make_args(), {})

    prov = json.loads(cell["file"].read_text())["approximation_fix"]
    assert prov["exact_checksum"] == "reproduced"
    assert prov["exact_checksum_drift"] is None


def test_rerun_cell_records_the_previous_values_as_provenance(cell):
    before = json.loads(cell["file"].read_text())
    rerun.rerun_cell(cell["spec"], make_args(), {})
    after = json.loads(cell["file"].read_text())

    prov = after["approximation_fix"]
    assert prov["fix_commit"] == rerun.FIX_COMMIT
    assert prov["exact_checksum"] == "reproduced"
    assert prov["previous"]["full_approx"] == before["full_reporting"]["full_approx"]
    assert prov["previous"]["approximation_comparison"] == before["approximation_comparison"]
    assert "generalisation" in prov["recomputed"]


def test_rerun_cell_omits_provenance_on_request(cell):
    rerun.rerun_cell(cell["spec"], make_args(no_provenance=True), {})
    assert "approximation_fix" not in json.loads(cell["file"].read_text())


def test_rerun_cell_skips_generalisation_on_request(cell):
    rerun.rerun_cell(cell["spec"], make_args(skip_generalisation=True), {})
    after = json.loads(cell["file"].read_text())
    assert "gen_vec" not in cell["seen"]
    assert after["generalisation"] == cell["summary"]["generalisation"], (
        "generalisation must be left as-is when skipped"
    )

    prov = after["approximation_fix"]
    assert "generalisation" not in prov["recomputed"]
    assert "generalisation_summary" not in prov["previous"], (
        "recording a previous value for a key that was never replaced misreads as a repair"
    )


def test_previous_covers_generalisation_when_it_was_recomputed(cell):
    rerun.rerun_cell(cell["spec"], make_args(), {})
    prov = json.loads(cell["file"].read_text())["approximation_fix"]
    assert "generalisation" in prov["recomputed"]
    assert set(prov["previous"]) == {
        "filtered_approx", "full_approx", "approximation_comparison", "generalisation_summary",
    }
    assert set(prov["previous"]) == set(
        k.split(".")[-1].replace("generalisation", "generalisation_summary")
        for k in prov["recomputed"]
    ), "previous must mirror recomputed exactly"


def test_rerun_cell_aborts_when_a_generalisation_baseline_disagrees(cell, monkeypatch):
    def shifted_baseline(approx_vec, paths, **kwargs):
        return {
            "benchmarks": [{
                "benchmark": "HellaSwag", "dataset": "hellaswag", "n_questions": 10,
                "n_choices": 4, "baseline_acc": 0.9,  # recorded run had 0.4
                "intervention_acc": 0.92, "delta": 0.02, "n_improved": 3,
                "n_worsened": 1, "binomial_p": 0.3, "significant": False,
            }],
            "summary": {"n_benchmarks": 1, "n_positive": 1,
                        "n_significant": 0, "avg_delta": 0.02},
        }

    monkeypatch.setattr(rerun, "run_generalisation_shared", shifted_baseline)
    before = cell["file"].read_text()
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "failed"
    assert cell["file"].read_text() == before


def test_rerun_cell_skips_a_post_fix_summary(cell):
    summary = json.loads(cell["file"].read_text())
    summary["timestamp"] = "2026-07-25T09:10:27"
    cell["file"].write_text(json.dumps(summary, indent=2))
    before = cell["file"].read_text()

    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "skipped"
    assert cell["file"].read_text() == before
    assert rerun.rerun_cell(cell["spec"], make_args(force=True), {}) == "repaired"


def test_checksum_only_stops_after_the_exact_report(cell):
    """Hardware-parity probe: touches the GPU, compares, writes nothing."""
    before = cell["file"].read_bytes()
    assert rerun.rerun_cell(cell["spec"], make_args(checksum_only=True), {}) == "checksum-ok"
    assert cell["file"].read_bytes() == before
    assert "exact_full" in cell["seen"], "the exact report must actually have run"
    assert "approx_full" not in cell["seen"], "no approximated evaluation in checksum mode"
    assert "gen_vec" not in cell["seen"], "no generalisation in checksum mode"


def test_checksum_only_still_fails_on_drift(cell, monkeypatch):
    def drifting(name, data, baseline_dict, baseline_acc, vec, mode, kw, approx_vec):
        return {"name": name, "n_questions": len(data), "baseline_acc": baseline_acc,
                "intervention_acc": 0.99, "delta": 0.4, "n_improved": 999,
                "n_worsened": 0, "binomial_p": 0.01}

    monkeypatch.setattr(rerun, "_evaluate_report_full", drifting)
    assert rerun.rerun_cell(cell["spec"], make_args(checksum_only=True), {}) == "failed"


def test_provenance_records_the_hardware_that_produced_the_numbers(cell):
    """The original logs only say "Device: cuda"; the repair pins torch/CUDA/device."""
    rerun.rerun_cell(cell["spec"], make_args(), {})
    env = json.loads(cell["file"].read_text())["approximation_fix"]["environment"]
    assert set(env) == {"torch", "cuda", "device"}
    assert env["torch"]


def test_rerun_cell_dry_run_verifies_without_writing(cell):
    before = cell["file"].read_text()
    assert rerun.rerun_cell(cell["spec"], make_args(dry_run=True), {}) == "verified"
    assert cell["file"].read_text() == before
    assert cell["seen"] == {}, "dry run must not touch the model or the SAE"


def test_rerun_cell_aborts_before_loading_a_model_when_reconstruction_fails(cell, monkeypatch):
    summary = json.loads(cell["file"].read_text())
    summary["data_stats"]["holdout_questions"] += 3
    cell["file"].write_text(json.dumps(summary, indent=2))

    def boom(cfg):
        raise AssertionError("must not load a model after a failed reconstruction")

    monkeypatch.setattr(rerun, "load_model_and_tokenizer", boom)
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "failed"


def test_rerun_cell_refuses_anything_that_is_not_batch_size_1(cell):
    """BS32 is not reproducible and must never be written into a result file.

    ExperimentConfig defaults batch_size to 32, so a cell missing its explicit
    batch_size=1 has to fail loudly rather than run batched.
    """
    cell["spec"].batch_size = 32
    before = cell["file"].read_bytes()
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "failed"
    assert cell["file"].read_bytes() == before
    assert cell["seen"] == {}, "must refuse before any evaluation runs"


def test_rerun_cell_refuses_when_batch_size_falls_back_to_the_config_default(cell):
    cell["spec"].batch_size = None          # no override -> ExperimentConfig default of 32
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "failed"


def test_every_final_matrix_cell_pins_batch_size_1():
    """The runner itself must never hand this script a non-BS1 cell."""
    from run_experiments_final import EXPERIMENTS

    offenders = [e.name for e in EXPERIMENTS if e.batch_size != 1]
    assert offenders == [], f"cells without batch_size=1: {offenders[:5]}"


def test_a_fully_repaired_cell_is_skipped_so_a_batch_can_resume(cell):
    """A 24 h batch must be resumable: re-invoking must not redo finished cells."""
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "repaired"
    after_first = cell["file"].read_bytes()
    cell["seen"].clear()

    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "skipped"
    assert cell["file"].read_bytes() == after_first
    assert cell["seen"] == {}, "a skipped cell must not load a model or evaluate"

    assert rerun.rerun_cell(cell["spec"], make_args(force=True), {}) == "repaired"


def test_a_partially_repaired_cell_is_picked_up_again_by_a_full_run(cell):
    """--skip-generalisation leaves the cell incomplete; the next full run finishes it."""
    assert rerun.rerun_cell(cell["spec"], make_args(skip_generalisation=True), {}) == "repaired"
    assert rerun.rerun_cell(cell["spec"], make_args(skip_generalisation=True), {}) == "skipped"

    # A run that would also recompute generalisation still has work to do.
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "repaired"
    prov = json.loads(cell["file"].read_text())["approximation_fix"]
    assert "generalisation" in prov["recomputed"]


def test_second_repair_keeps_the_original_pre_fix_values(cell):
    """previous must point at the dd646c9 numbers, not at the first repair's output."""
    original = json.loads(cell["file"].read_text())["full_reporting"]["full_approx"]

    rerun.rerun_cell(cell["spec"], make_args(skip_generalisation=True), {})
    rerun.rerun_cell(cell["spec"], make_args(), {})

    prov = json.loads(cell["file"].read_text())["approximation_fix"]
    assert prov["previous"]["full_approx"] == original, (
        "the second pass overwrote previous with the first repair's numbers"
    )


def test_rerun_cell_skips_a_cell_that_never_produced_a_summary(cell):
    cell["file"].unlink()
    assert rerun.rerun_cell(cell["spec"], make_args(), {}) == "skipped"


def test_write_back_is_byte_stable_for_untouched_content(cell):
    """The JSON round-trip must not reformat the file, so the diff stays minimal."""
    rerun.rerun_cell(cell["spec"], make_args(no_provenance=True), {})
    raw = cell["file"].read_bytes()
    assert raw == json.dumps(json.loads(raw), indent=2).encode("utf-8")


def test_write_back_keeps_unix_line_endings(cell):
    """Written on Windows, Path.write_text would emit CRLF and rewrite every line.

    The results are committed with LF, so a translated newline would turn a handful of
    changed values into a whole-file diff. read_text() normalises newlines away, hence
    the byte-level assertion.
    """
    rerun.rerun_cell(cell["spec"], make_args(), {})
    assert b"\r\n" not in cell["file"].read_bytes()


def test_rerun_cell_leaves_the_screening_cache_and_log_untouched(cell):
    """Only the summary JSON and the figure may move; inputs are read-only."""
    exp_dir = cell["dir"]
    (exp_dir / "20260530_102005_EXP-999.log").write_bytes(b"run log\nsecond line\n")
    cache_file = exp_dir / "logit_screening_arc_challenge_gemma2_2b.json"
    before = {p.name: p.read_bytes() for p in exp_dir.iterdir()}

    rerun.rerun_cell(cell["spec"], make_args(), {})

    after = {p.name: p.read_bytes() for p in exp_dir.iterdir()}
    assert set(after) == set(before), "no files may be added to or removed from the folder"
    assert after[cache_file.name] == before[cache_file.name]
    assert after["20260530_102005_EXP-999.log"] == before["20260530_102005_EXP-999.log"]
    assert after[cell["file"].name] != before[cell["file"].name]


# ── model lifetime across cells ───────────────────────────────────


def _counting_loader(monkeypatch):
    """Replace the fixture's model loader with one that records every load."""
    loaded = []

    def fake_load(cfg):
        loaded.append(cfg.model_id)
        return object(), object(), CPU

    monkeypatch.setattr(rerun, "load_model_and_tokenizer", fake_load)
    return loaded


def test_a_second_cell_on_the_same_model_reloads_it_by_default(cell, monkeypatch):
    """Carrying one loaded model across cells is what broke the EXP-322 batch.

    EXP-322 asserts on question 7 when it inherits the gemma-2-2b that EXP-321 had just
    driven through ~21,500 forward passes, and reproduces the recorded run exactly when
    it loads its own. Until that state is understood, a fresh model per cell is the
    default; the reuse it gives up is worth ~80 s against a 30-60 min cell.
    """
    loaded = _counting_loader(monkeypatch)
    slot = {}

    rerun.rerun_cell(cell["spec"], make_args(force=True), slot)
    rerun.rerun_cell(cell["spec"], make_args(force=True), slot)

    assert len(loaded) == 2, "the second cell must not inherit the first cell's model"


def test_reuse_model_opts_back_into_sharing_the_loaded_model(cell, monkeypatch):
    """The old behaviour stays reachable, so the saving can be taken deliberately."""
    loaded = _counting_loader(monkeypatch)
    slot = {}

    rerun.rerun_cell(cell["spec"], make_args(force=True, reuse_model=True), slot)
    rerun.rerun_cell(cell["spec"], make_args(force=True, reuse_model=True), slot)

    assert len(loaded) == 1


def test_reloading_releases_the_previous_model_first(cell, monkeypatch):
    """Two copies of an 8B model on one card is an OOM, so the release must come first."""
    order = []
    monkeypatch.setattr(rerun, "release_model", lambda slot: order.append("release"))

    def fake_load(cfg):
        order.append("load")
        return object(), object(), CPU

    monkeypatch.setattr(rerun, "load_model_and_tokenizer", fake_load)
    slot = {}

    rerun.rerun_cell(cell["spec"], make_args(force=True), slot)
    rerun.rerun_cell(cell["spec"], make_args(force=True), slot)

    assert order == ["release", "load", "release", "load"]


# ── main(): batch-level error handling ────────────────────────────


class _FakeSpec:
    """Stand-in for an Experiment; main() only ever reads ``.name`` off it."""

    def __init__(self, name):
        self.name = name


def _cuda_assert_error():
    """The exception torch raises once a device-side assert has killed the context.

    Mirrors what EXP-322 produced: after this fires, every later CUDA call in the same
    process raises the same thing, whatever it was actually asked to do.
    """
    cls = getattr(torch, "AcceleratorError", RuntimeError)
    return cls(
        "CUDA error: device-side assert triggered\n"
        "CUDA kernel errors might be asynchronously reported at some other API call, "
        "so the stacktrace below might be incorrect."
    )


def _run_main(monkeypatch, capsys, cells, side_effect, release=None):
    """Drive main() over stand-in cells. Returns (attempted, exit_code, stdout)."""
    attempted = []

    def fake_rerun_cell(spec, args, model_slot):
        attempted.append(spec.name)
        return side_effect(spec)

    monkeypatch.setattr(sys, "argv", ["rerun_approx_fix.py", "--all"])
    monkeypatch.setattr(rerun, "select_cells", lambda args: cells)
    monkeypatch.setattr(rerun, "rerun_cell", fake_rerun_cell)
    monkeypatch.setattr(rerun, "release_model", release or (lambda slot: None))

    code = 0
    try:
        rerun.main()
    except SystemExit as exc:
        code = exc.code
    return attempted, code, capsys.readouterr().out


def test_a_cuda_error_aborts_the_batch_instead_of_being_charged_to_every_later_cell(
    monkeypatch, capsys
):
    """A dead CUDA context is not recoverable in-process, so continuing is pointless.

    The real run turned one genuine failure into 25 reported ones: each later cell
    re-read its config, verified reconstruction offline, loaded an SAE and then hit the
    same poisoned context. That buries the one error that mattered.
    """
    cells = [_FakeSpec(f"EXP-{n}") for n in (313, 322, 323, 324)]

    def side_effect(spec):
        if spec.name == "EXP-322":
            raise _cuda_assert_error()
        return "repaired"

    attempted, code, out = _run_main(monkeypatch, capsys, cells, side_effect)

    assert attempted == ["EXP-313", "EXP-322"], (
        "the batch must stop at the first CUDA error, not attempt the remaining cells"
    )
    assert code == 1
    assert "2 cell(s) not attempted" in out


def test_an_ordinary_cell_failure_still_advances_to_the_next_cell(monkeypatch, capsys):
    """Only CUDA errors are fatal; a bad cell must not cost the whole batch.

    This pins the other side of the discrimination the abort introduces — without it
    the fix would trade 25 false failures for 25 skipped repairs.
    """
    cells = [_FakeSpec(f"EXP-{n}") for n in (313, 322, 323, 324)]

    def side_effect(spec):
        if spec.name == "EXP-322":
            raise ValueError("this cell's summary is malformed")
        return "repaired"

    attempted, code, out = _run_main(monkeypatch, capsys, cells, side_effect)

    assert attempted == ["EXP-313", "EXP-322", "EXP-323", "EXP-324"]
    assert code == 1
    assert "3 repaired" in out and "1 failed" in out


def test_a_failing_cell_prints_the_traceback(monkeypatch, capsys):
    """``ERROR: AcceleratorError: CUDA error`` on its own names no line of code.

    Device-side asserts are reported asynchronously, so the message points nowhere and
    the stack is the only thing that narrows the search.
    """
    cells = [_FakeSpec("EXP-322")]

    def side_effect(spec):
        raise ValueError("boom")

    _attempted, _code, out = _run_main(monkeypatch, capsys, cells, side_effect)

    assert "Traceback (most recent call last)" in out
    assert "ValueError: boom" in out


def test_the_summary_still_prints_when_releasing_the_model_raises(monkeypatch, capsys):
    """``release_model`` runs in a finally: raising there discards the whole run report.

    That is exactly what happened — ``torch.cuda.empty_cache()`` hit the dead context,
    so the batch exited on an unhandled traceback and never said what it had achieved.
    """
    cells = [_FakeSpec("EXP-313")]

    def exploding_release(_slot):
        raise _cuda_assert_error()

    _attempted, code, out = _run_main(
        monkeypatch, capsys, cells, lambda spec: "repaired", release=exploding_release,
    )

    assert "Done in" in out
    assert "1 repaired" in out
    assert code == 0


# ── integration: the real committed experiment directories ────────


def _real_pre_fix_dirs():
    if not (REPO_ROOT / "results").is_dir():
        return []
    out = []
    for exp_dir in sorted((REPO_ROOT / "results").glob("EXP-*")):
        summary_file = rerun.summary_path(exp_dir)
        cache_file = rerun.screening_cache_path(exp_dir)
        if summary_file is None or cache_file is None:
            continue
        try:
            summary = json.loads(summary_file.read_text())
        except json.JSONDecodeError:
            continue
        if rerun.is_pre_fix(summary) and "all_grid_results" in summary:
            out.append((exp_dir.name, summary_file, cache_file))
    return out


@pytest.mark.parametrize("name,summary_file,cache_file", _real_pre_fix_dirs())
def test_real_experiment_dirs_reconstruct_exactly(name, summary_file, cache_file):
    """Split and vectors rebuilt from disk must match what the original run recorded.

    This is the test that pins fidelity: the baseline accuracies agreeing to 1e-9
    means the rebuilt holdout set is the same questions, not merely the same size.
    """
    summary = json.loads(summary_file.read_text())
    screening = json.loads(cache_file.read_text())

    class RealCausal:
        min_p_correct, max_p_correct = summary["data_stats"]["p_correct_range"]
        train_fraction, val_fraction, seed = 0.50, 0.25, 42

    ok, _split, problems, _warn = rerun.verify_offline(summary, screening, RealCausal)
    assert ok, f"{name}: " + "; ".join(problems)


def test_the_integration_set_is_not_silently_empty():
    """Guard against the parametrised test above vacuously passing on a stripped tree."""
    if not (REPO_ROOT / "results").is_dir():
        pytest.skip("results/ not present in this checkout")
    assert _real_pre_fix_dirs(), "expected committed pre-fix experiment directories"
