#!/usr/bin/env python3
"""Repair pre-fix experiment results in place after the approximation fix (fed574a).

``fed574a`` changed ``compute_approximated_hidden_vector`` from ``decode(s)`` to
``decode(s) - decode(0) = s @ W_dec``, removing the decoder bias from the SAE-free
approximation. Only two things in the pipeline consume that vector:

  * Phase 4 reports 3 and 4 (``filtered_approx``, ``full_approx``)
  * Phase 5 cross-benchmark generalisation (approximation-only)

Everything else — screening, the split, Phase 1, Phase 2, pair stability, the Phase 3
grid search, the best-config selection, the exact Phase 4 reports, the difficulty
analysis and k-fold — runs through the exact ``InterventionHook`` and is unaffected.
This script recomputes exactly the affected parts and rewrites them into the existing
summary JSON, so the file ends up as if the cell had been run with the fix in place.

Nothing under ``src/`` is modified: the evaluation itself calls the very same helpers
the pipeline calls (``_evaluate_report``, ``_evaluate_report_full``,
``compute_approximated_hidden_vector``), which is what makes the output faithful.

What is reconstructed rather than recomputed, and why it is safe:

  * The data split comes from the committed per-experiment screening cache plus the
    seeded ``matched_pair_split``. Verified offline against ``data_stats``: counts and
    both baseline accuracies must match to 1e-9 before any GPU work starts.
  * The intervention vector is rebuilt from the summary's own feature tables
    (``phase1_top_features`` for freq_diff ranking, ``phase2_top_features_per_strength``
    for causal_delta). Verified by rebuilding the vector for all grid cells and
    comparing every ``n_features_used`` against the recorded value.
  * As a final guard the exact holdout report is re-evaluated on the GPU and must
    reproduce the stored ``full_exact`` numbers exactly. If it does not, the cell is
    skipped and nothing is written — that single check covers split identity, vector
    identity and GPU/BS1 determinism at once.

Usage::

    uv run python scripts/rerun_approx_fix.py --list
    uv run python scripts/rerun_approx_fix.py --all --dry-run       # offline checks only
    uv run python scripts/rerun_approx_fix.py --all --checksum-only # GPU parity, no writes
    uv run python scripts/rerun_approx_fix.py --only 330
    uv run python scripts/rerun_approx_fix.py --all 2>&1 | tee rerun_approx_fix.log
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_experiments_final import EXPERIMENTS, Experiment, _extract_exp_id  # noqa: E402
from sae_experiments.benchmarks import get_answer_token_ids, load_benchmark  # noqa: E402
from sae_experiments.config import load_experiment_config  # noqa: E402
from sae_experiments.fast_inference import (  # noqa: E402
    batched_intervention_predictions,
    batched_logit_predictions,
    make_approx_intervention_fn,
)
from sae_experiments.generalisation import _load_benchmark_config, run_generalisation  # noqa: E402
from sae_experiments.hooks import HookManager  # noqa: E402
from sae_experiments.intervention import (  # noqa: E402
    build_intervention_vector,
    compute_approximated_hidden_vector,
)
from sae_experiments.models import get_layer_module, load_model_and_tokenizer  # noqa: E402
from sae_experiments.phases import (  # noqa: E402
    _evaluate_report,
    _evaluate_report_full,
    _predict_kwargs,
)
from sae_experiments.plotting import generate_all_plots  # noqa: E402
from sae_experiments.reporting import _NumpySafeEncoder  # noqa: E402
from sae_experiments.sae import load_sae  # noqa: E402
from sae_experiments.splitting import matched_pair_split  # noqa: E402

# fix(intervention): remove decoder bias from SAE-free approximation
FIX_COMMIT = "fed574a9879e8e793bf81f0cd018e472be6dd77d"
FIX_TIMESTAMP = "2026-07-23T20:41:26"

TOL = 1e-9
GEN_BASELINE_DIR = ROOT / "cache" / "gen_baselines"

EXACT_KEYS = ("filtered_exact", "full_exact")
APPROX_KEYS = ("filtered_approx", "full_approx")


# ── result-file helpers ───────────────────────────────────────────


def summary_path(exp_dir: Path) -> Optional[Path]:
    """The experiment's summary JSON, or None when the cell never completed."""
    cands = [p for p in exp_dir.glob("*.json") if not p.name.startswith("logit_screening_")]
    if len(cands) != 1:
        return None
    return cands[0]


def screening_cache_path(exp_dir: Path) -> Optional[Path]:
    cands = sorted(exp_dir.glob("logit_screening_*.json"))
    return cands[0] if len(cands) == 1 else None


def is_pre_fix(summary: Dict[str, Any]) -> bool:
    return str(summary.get("timestamp", "")) < FIX_TIMESTAMP


def already_repaired(summary: Dict[str, Any], spec: Experiment, args) -> bool:
    """True when a previous run already recomputed everything this one would.

    The summary's own ``timestamp`` never changes, so without this a re-invocation
    after an interrupted batch would redo finished cells. Comparing against what *this*
    invocation would recompute also means a cell repaired with --skip-generalisation is
    correctly picked up again by a full run.
    """
    recorded = set((summary.get("approximation_fix") or {}).get("recomputed") or [])
    needed = {
        "full_reporting.filtered_approx",
        "full_reporting.full_approx",
        "approximation_comparison",
    }
    if spec.generalisation and not args.skip_generalisation:
        needed.add("generalisation")
    return needed <= recorded


def parse_overrides(items: List[str]) -> Dict[str, Any]:
    """Mirror the ``--set KEY=VALUE`` parsing in ``sae_experiments.__main__``."""
    import ast

    out: Dict[str, Any] = {}
    for item in items:
        key, value = item.split("=", 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        out[key] = value
    return out


# ── reconstruction ────────────────────────────────────────────────


def rebuild_split(screening_data: List[Dict], causal_cfg) -> Dict[str, List[Dict]]:
    """Reproduce ``run_screening_and_split`` from the cached screening entries."""
    min_p, max_p = causal_cfg.min_p_correct, causal_cfg.max_p_correct
    filtered = [s for s in screening_data if min_p <= s["p_correct"] <= max_p]

    train_pct = int(causal_cfg.train_fraction * 100)
    val_pct = int(causal_cfg.val_fraction * 100)
    test_pct = 100 - train_pct - val_pct
    train, val, _test_filtered = matched_pair_split(
        filtered, [train_pct, val_pct, test_pct], key="p_correct", seed=causal_cfg.seed,
    )

    used = {s["q_idx"] for s in train} | {s["q_idx"] for s in val}
    test_full = [s for s in screening_data if s["q_idx"] not in used]
    grid_eval, holdout = matched_pair_split(
        test_full, [50, 50], key="p_correct", seed=causal_cfg.seed,
    )

    filtered_idx = {s["q_idx"] for s in filtered}
    holdout_filtered = [s for s in holdout if s["q_idx"] in filtered_idx]

    return {
        "filtered": filtered, "train": train, "val": val, "test_full": test_full,
        "grid_eval": grid_eval, "holdout": holdout, "holdout_filtered": holdout_filtered,
    }


def _restore_candidate_order(df: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    """Undo the reordering the Phase-2 serialisation applied.

    ``_build_phase2_top_features_per_strength_section`` stores each bucket in
    ``nlargest(25, "delta")`` order, and pandas' nlargest does not preserve the input
    order across equal values. Phase 3 ranked the rows in Phase-1 candidate order
    instead — freq_diff descending for positive features, ascending for negative ones —
    and its own ``nlargest`` broke delta ties by that position. Restoring the candidate
    order before any ranking makes the rebuilt selection agree with the original one.
    """
    if df.empty or "freq_diff" not in df.columns:
        return df
    return df.sort_values("freq_diff", ascending=ascending, kind="stable").reset_index(drop=True)


def feature_frames(summary: Dict[str, Any]):
    """Rebuild the feature tables Phase 3 ranks over, per ranking criterion."""
    p1 = summary.get("phase1_top_features") or {}
    # Phase-1 sections are already serialised in candidate order.
    freq_pos = pd.DataFrame(p1.get("positive") or [])
    freq_neg = pd.DataFrame(p1.get("negative") or [])

    buckets: Dict[float, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for entry in summary.get("phase2_top_features_per_strength") or []:
        buckets[round(float(entry["test_strength"]), 6)] = (
            _restore_candidate_order(pd.DataFrame(entry.get("positive") or []), ascending=False),
            _restore_candidate_order(pd.DataFrame(entry.get("negative") or []), ascending=True),
        )
    return freq_pos, freq_neg, buckets


def select_features(frames, ranking: str, strength: float):
    """Mirror ``_select_features_for_ranking`` on the reconstructed tables."""
    freq_pos, freq_neg, buckets = frames

    if ranking == "freq_diff":
        return freq_pos, freq_neg
    if ranking != "causal_delta":
        raise ValueError(f"Unknown ranking_criterion: {ranking!r}")

    key = round(float(strength), 6)
    if len(buckets) == 1:
        # Single-strength Phase 2: the one ranking is reused for every grid strength.
        pos, neg = next(iter(buckets.values()))
    elif key in buckets:
        pos, neg = buckets[key]
    else:
        raise KeyError(f"No Phase-2 bucket for test_strength={key}")

    if len(pos):
        pos = pos[pos["is_causal"]].copy()
    if len(neg):
        neg = neg[neg["is_causal"]].copy()
    return pos, neg


def build_vector(
    frames, cfg_row: Dict[str, Any], *, d_hidden: int, device, dtype,
) -> torch.Tensor:
    pos, neg = select_features(frames, cfg_row["ranking_criterion"], cfg_row["strength"])
    return build_intervention_vector(
        pos_features=pos,
        neg_features=neg,
        top_k_pos=int(cfg_row["top_k_pos"]),
        top_k_neg=int(cfg_row["top_k_neg"]),
        strength=float(cfg_row["strength"]),
        ranking_criterion=cfg_row["ranking_criterion"],
        weighting_mode=cfg_row["weighting_mode"],
        d_hidden=d_hidden,
        device=device,
        dtype=dtype,
    )


# ── offline verification ──────────────────────────────────────────


def ranking_ambiguities(summary: Dict[str, Any], frames) -> List[str]:
    """Whether the best config's causal_delta ranking is *provably* recoverable.

    A Phase-2 bucket is capped at 25 rows. If Phase 2 tested more candidates than that,
    and the k-th selected delta ties with the smallest delta that survived the cap, the
    original run may have broken the tie in favour of a feature that was never
    serialised. The rebuilt vector then has the right size — so ``n_features_used``
    still matches — but possibly different members. Only the exact checksum settles it,
    which is why this is reported as a warning rather than a hard failure.

    Scoped to the best config: it is the only vector this script ever builds. Other grid
    cells are still integrity-checked by count in ``verify_offline``, but their ranking
    never reaches the GPU.
    """
    best = summary.get("best_config") or {}
    _freq_pos, _freq_neg, buckets = frames
    if best.get("ranking_criterion") != "causal_delta" or not buckets:
        return []

    key = round(float(best["strength"]), 6)
    bucket = next(iter(buckets.values())) if len(buckets) == 1 else buckets.get(key)
    if bucket is None:
        return []

    causal_screening = summary.get("causal_screening") or {}
    found = []
    for stored, k, n_tested, label in (
        (bucket[0], int(best["top_k_pos"]),
         int(causal_screening.get("n_unique_tested_positive", 0)), "positive"),
        (bucket[1], int(best["top_k_neg"]),
         int(causal_screening.get("n_unique_tested_negative", 0)), "negative"),
    ):
        if k <= 0 or stored.empty or len(stored) >= n_tested:
            continue  # nothing was cut off, so the stored rows are the whole table
        causal = stored[stored["is_causal"]]
        if len(causal) < k:
            continue  # shows up as an n_features_used mismatch instead
        if causal.nlargest(k, "delta")["delta"].min() <= stored["delta"].min():
            found.append(
                f"{label} causal_delta top_k={k}: delta tie at the 25-row cap "
                f"({n_tested} features tested) — selection not provably identical"
            )
    return found


def verify_offline(
    summary: Dict[str, Any], screening_data: List[Dict], causal_cfg,
) -> Tuple[bool, Dict[str, List[Dict]], List[str], List[str]]:
    """Check that split and vector reconstruction match what the run recorded.

    Returns ``(ok, split, problems, warnings)``. Problems block the cell; warnings flag
    reconstructions that are size-correct but not provably member-identical, and are
    left for the exact checksum to resolve.
    """
    problems: List[str] = []
    split = rebuild_split(screening_data, causal_cfg)
    ds = summary["data_stats"]

    for key, field in (
        ("filtered", "filtered_questions"), ("train", "train_questions"),
        ("val", "val_questions"), ("grid_eval", "grid_eval_questions"),
        ("holdout", "holdout_questions"), ("holdout_filtered", "holdout_filtered_questions"),
    ):
        if len(split[key]) != ds[field]:
            problems.append(f"{field}: rebuilt {len(split[key])} != recorded {ds[field]}")

    for key, field in (
        ("holdout", "baseline_full_accuracy"),
        ("holdout_filtered", "baseline_filtered_accuracy"),
    ):
        entries = split[key]
        acc = sum(e["correct"] for e in entries) / len(entries) if entries else 0.0
        if abs(acc - ds[field]) > TOL:
            problems.append(f"{field}: rebuilt {acc!r} != recorded {ds[field]!r}")

    frames = feature_frames(summary)
    d_hidden = summary["sae"]["d_hidden"]
    mismatched = 0
    for row in summary.get("all_grid_results") or []:
        vec = build_vector(
            frames, row, d_hidden=d_hidden, device=torch.device("cpu"), dtype=torch.float32,
        )
        if int((vec != 0).sum().item()) != int(row["n_features_used"]):
            mismatched += 1
    if mismatched:
        problems.append(
            f"{mismatched}/{len(summary['all_grid_results'])} grid vectors disagree on n_features_used"
        )

    best = summary["best_config"]
    bvec = build_vector(
        frames, best, d_hidden=d_hidden, device=torch.device("cpu"), dtype=torch.float32,
    )
    if int((bvec != 0).sum().item()) != int(best["n_features_used"]):
        problems.append(
            f"best-config vector: rebuilt {int((bvec != 0).sum().item())} features "
            f"!= recorded {best['n_features_used']}"
        )

    return not problems, split, problems, ranking_ambiguities(summary, frames)


# ── generalisation with shared baselines ──────────────────────────


def _gen_baseline_correct(
    bench_config, questions, *, model_short_name, model, tokenizer, hook_mgr, sae,
    device, batch_size,
) -> List[bool]:
    """Per-question baseline correctness, cached per (model, benchmark).

    The generalisation baseline uses no intervention, so it depends only on model and
    benchmark — not on the SAE, the layer or the steering vector. Caching it lets every
    cell of the same model share one pass instead of repeating it.
    """
    GEN_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = GEN_BASELINE_DIR / f"{model_short_name}__{bench_config.cache_name}.json"

    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("n") == len(questions):
            print(f"    baseline from cache: {path.name}")
            return [bool(c) for c in cached["correct"]]
        print(f"    baseline cache size mismatch, recomputing: {path.name}")

    answer_token_ids = get_answer_token_ids(bench_config.choices, tokenizer)
    results = batched_logit_predictions(
        questions,
        batch_size=batch_size,
        model=model, tokenizer=tokenizer, config=bench_config,
        hook_mgr=hook_mgr, sae=sae,
        answer_token_id_list=[answer_token_ids[c] for c in bench_config.choices],
        device=device, extract_features=False,
        desc=f"Generalisation Baseline: {bench_config.name}",
    )
    correct = [bool(r["correct"]) for r in results]
    path.write_text(
        json.dumps({"n": len(questions), "correct": [int(c) for c in correct]}),
        encoding="utf-8", newline="\n",
    )
    return correct


def run_generalisation_shared(
    approx_vec: torch.Tensor, benchmark_paths: List[str], *, model_short_name,
    model, tokenizer, hook_mgr, sae, device, batch_size,
) -> Dict[str, Any]:
    """``run_generalisation`` with the baseline pass served from a shared cache.

    Produces the same structure and the same numbers; only the baseline compute is
    shared across cells. The caller checks each ``baseline_acc`` against the value the
    pre-fix run recorded, which is unaffected by the approximation bug.
    """
    print("=" * 80)
    print("PHASE 5: CROSS-BENCHMARK GENERALISATION TEST (shared baselines)")
    print(f"  Intervention vector norm: {approx_vec.norm().item():.4f}")
    print("=" * 80)

    intervention_fn = make_approx_intervention_fn(approx_vec)
    results: List[Dict[str, Any]] = []

    for path in benchmark_paths:
        bench_config = _load_benchmark_config(str(path))
        print(f"\n  Evaluating: {bench_config.name} ({path})")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        questions = load_benchmark(bench_config)
        answer_token_ids = get_answer_token_ids(bench_config.choices, tokenizer)
        answer_token_id_list = [answer_token_ids[c] for c in bench_config.choices]

        baseline_correct = _gen_baseline_correct(
            bench_config, questions, model_short_name=model_short_name, model=model,
            tokenizer=tokenizer, hook_mgr=hook_mgr, sae=sae, device=device,
            batch_size=batch_size,
        )

        int_results = batched_intervention_predictions(
            questions, intervention_fn,
            batch_size=batch_size,
            model=model, tokenizer=tokenizer, config=bench_config,
            hook_mgr=hook_mgr, sae=sae,
            answer_token_id_list=answer_token_id_list, device=device,
            desc=f"Generalisation Intervention: {bench_config.name}",
        )

        baseline_acc = sum(baseline_correct) / len(questions)
        int_acc = sum(r["correct"] for r in int_results) / len(questions)
        n_improved = sum(
            1 for i, r in enumerate(int_results) if not baseline_correct[i] and r["correct"]
        )
        n_worsened = sum(
            1 for i, r in enumerate(int_results) if baseline_correct[i] and not r["correct"]
        )
        binomial_p = None
        if n_improved + n_worsened > 0:
            binomial_p = binomtest(
                n_improved, n_improved + n_worsened, 0.5, alternative="greater",
            ).pvalue

        results.append({
            "benchmark": bench_config.name,
            "dataset": bench_config.dataset_name,
            "n_questions": len(questions),
            "n_choices": len(bench_config.choices),
            "baseline_acc": float(baseline_acc),
            "intervention_acc": float(int_acc),
            "delta": float(int_acc - baseline_acc),
            "n_improved": int(n_improved),
            "n_worsened": int(n_worsened),
            "binomial_p": float(binomial_p) if binomial_p is not None else None,
            "significant": bool(binomial_p is not None and binomial_p < 0.05),
        })
        r = results[-1]
        print(f"    N={r['n_questions']}, Baseline={r['baseline_acc']*100:.1f}%, "
              f"Intervention={r['intervention_acc']*100:.1f}%, Delta={r['delta']*100:+.1f}%")

    avg_delta = sum(r["delta"] for r in results) / len(results) if results else 0.0
    return {
        "benchmarks": results,
        "summary": {
            "n_benchmarks": len(results),
            "n_positive": sum(1 for r in results if r["delta"] > 0),
            "n_significant": sum(1 for r in results if r["significant"]),
            "avg_delta": float(avg_delta),
        },
    }


# ── per-cell re-run ───────────────────────────────────────────────


def bias_free_displacement_error(sae, device) -> float:
    """How far the loaded approximation is from being bias-free. 0.0 means fixed.

    ``fix_commit`` only records which change this repair is *for*; it cannot tell whether
    the checkout actually contains it. This can. The fixed
    ``compute_approximated_hidden_vector`` returns ``decode(s) - decode(0)``, so a zero
    intervention must displace the hidden state by exactly zero. The pre-fix version
    returns ``decode(0) = b_dec`` instead. Where ``b_dec`` is zero the two implementations
    agree anyway, so this check is exactly as sensitive as the bug is consequential.
    """
    zero = torch.zeros(sae.d_hidden, device=device, dtype=sae.dtype)
    with torch.no_grad():
        return float(compute_approximated_hidden_vector(zero, sae).abs().max().item())


def as_summary_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """Serialise a report the way ``build_summary`` does."""
    return {
        "name": report["name"],
        "n_questions": int(report["n_questions"]),
        "baseline_acc": float(report["baseline_acc"]),
        "intervention_acc": float(report["intervention_acc"]),
        "delta": float(report["delta"]),
        "n_improved": int(report["n_improved"]),
        "n_worsened": int(report["n_worsened"]),
        "binomial_p": float(report["binomial_p"]) if report["binomial_p"] is not None else None,
        "significant": bool(report["binomial_p"] and report["binomial_p"] < 0.05),
    }


def rerun_cell(spec: Experiment, args, model_slot: Dict[str, Any]) -> str:
    """Recompute the approximation-dependent results for one cell. Returns a status."""
    exp_id = _extract_exp_id(spec.name)
    exp_dir = ROOT / "results" / exp_id
    print("\n" + "=" * 80)
    print(spec.name)
    print("=" * 80)

    spath = summary_path(exp_dir)
    cpath = screening_cache_path(exp_dir)
    if spath is None or cpath is None:
        print("  no completed summary / screening cache — skipping")
        return "skipped"

    summary = json.loads(spath.read_text(encoding="utf-8"))
    if not is_pre_fix(summary) and not args.force:
        print(f"  timestamp {summary.get('timestamp')} is post-fix — skipping (use --force)")
        return "skipped"

    if already_repaired(summary, spec, args) and not args.force:
        done = (summary["approximation_fix"] or {}).get("rerun_at")
        print(f"  already repaired at {done} — skipping (use --force to redo)")
        return "skipped"

    missing = [k for k in EXACT_KEYS + APPROX_KEYS if k not in summary.get("full_reporting", {})]
    if missing:
        print(f"  unexpected full_reporting keys (missing {missing}) — skipping")
        return "skipped"

    overrides = parse_overrides(spec.overrides)
    overrides["output_dir"] = str(exp_dir.relative_to(ROOT)).replace("\\", "/")
    overrides["cache_dir"] = overrides["output_dir"]
    if spec.batch_size is not None:
        overrides["batch_size"] = spec.batch_size

    config = load_experiment_config(
        sae_path=ROOT / spec.sae,
        layer=spec.layer,
        benchmark_path=ROOT / spec.benchmark,
        overrides=overrides,
    )

    # BS1 only. ExperimentConfig defaults batch_size to 32, so a cell that lost its
    # explicit batch_size=1 would silently run generalisation batched and produce
    # results that do not belong in the thesis. Refuse loudly instead.
    if config.batch_size != 1:
        print(f"  effective batch_size is {config.batch_size}, not 1 — refusing "
              f"(BS32 results are not reproducible)")
        return "failed"

    screening_data = json.loads(cpath.read_text(encoding="utf-8"))
    ok, split, problems, warnings = verify_offline(summary, screening_data, config.causal)
    if not ok:
        print("  RECONSTRUCTION MISMATCH — nothing written:")
        for p in problems:
            print(f"    - {p}")
        return "failed"
    print(f"  reconstruction verified (holdout {len(split['holdout'])}, "
          f"filtered {len(split['holdout_filtered'])}, "
          f"all {len(summary['all_grid_results'])} grid vectors)")
    for w in warnings:
        print(f"  NOTE: {w}")

    if args.dry_run:
        return "verified"

    # ── model / SAE ────────────────────────────────────────────────
    # Carrying one loaded model across cells is what broke the EXP-322 batch: it hit a
    # device-side assert on question 7 of a holdout it reproduces exactly when it loads
    # its own copy, after inheriting the gemma-2-2b that the previous cell had driven
    # through ~21,500 forward passes. Until that state is understood every cell gets a
    # fresh model; the reuse is worth ~80 s against a cell that runs for 30-60 min.
    if args.reuse_model and model_slot.get("model_id") == config.model.model_id:
        print(f"  reusing loaded model: {config.model.model_id}")
        model, tokenizer, device = model_slot["model"], model_slot["tokenizer"], model_slot["device"]
    else:
        release_model(model_slot)
        model, tokenizer, device = load_model_and_tokenizer(config.model)
        model_slot.update(
            model_id=config.model.model_id, model=model, tokenizer=tokenizer, device=device,
        )

    sae = load_sae(config.sae, device)
    if model_slot.get("hook_mgr") is not None:
        model_slot["hook_mgr"].remove_cache_hook()
    hook_mgr = HookManager(get_layer_module(
        model, config.sae.layer, config.sae.hook_point,
        expected_n_layers=config.model.n_layers,
    ))
    hook_mgr.register_cache_hook()
    model_slot["hook_mgr"] = hook_mgr

    answer_token_ids = get_answer_token_ids(config.benchmark.choices, tokenizer)
    answer_token_id_list = [answer_token_ids[c] for c in config.benchmark.choices]
    kw = _predict_kwargs(
        model, tokenizer, config.benchmark, hook_mgr, sae, answer_token_id_list, device,
    )

    # Baselines come from the screening cache — verified equal to the recorded ones.
    holdout, holdout_filtered = split["holdout"], split["holdout_filtered"]
    baseline_results = [
        {"q_idx": e["q_idx"], "correct": e["correct"], "prob_correct": e["p_correct"]}
        for e in holdout
    ]
    baseline_dict = {r["q_idx"]: r["correct"] for r in baseline_results}
    baseline_acc = float(np.mean([r["correct"] for r in baseline_results]))
    baseline_filtered_results = [
        {"q_idx": e["q_idx"], "correct": e["correct"], "prob_correct": e["p_correct"]}
        for e in holdout_filtered
    ]
    baseline_filtered_acc = float(np.mean([r["correct"] for r in baseline_filtered_results]))

    frames = feature_frames(summary)
    best_config = summary["best_config"]
    vec = build_vector(
        frames, best_config, d_hidden=sae.d_hidden, device=device, dtype=sae.dtype,
    )
    print(f"  Exact intervention vector: {int((vec != 0).sum().item())} features")

    # ── checksum: the exact report must reproduce the stored numbers ───
    stored_exact = summary["full_reporting"]["full_exact"]
    check = _evaluate_report_full(
        "Full + Exact (checksum)", holdout, baseline_dict, baseline_acc, vec, "exact", kw, None,
    )
    drift = [
        f"{k}: recomputed {check[k]!r} != stored {stored_exact[k]!r}"
        for k in ("n_improved", "n_worsened")
        if int(check[k]) != int(stored_exact[k])
    ]
    if abs(check["intervention_acc"] - stored_exact["intervention_acc"]) > TOL:
        drift.append(
            f"intervention_acc: recomputed {check['intervention_acc']!r} "
            f"!= stored {stored_exact['intervention_acc']!r}"
        )
    # Record the size, not just the fact. "drifted" on its own is not auditable, and
    # whole questions is the unit that makes it judgeable: a cell whose exact report
    # moves by 8 questions cannot carry a claim about an effect worth 4. Reproducing
    # cells store an explicit null, which says "checked, zero" where a missing key
    # would only say "unknown".
    n_holdout = int(stored_exact["n_questions"])
    drift_acc = float(check["intervention_acc"]) - float(stored_exact["intervention_acc"])
    drift_questions = round(drift_acc * n_holdout)
    drift_record = {
        "n_questions": n_holdout,
        "stored": {
            k: stored_exact[k] for k in ("intervention_acc", "n_improved", "n_worsened")
        },
        "recomputed": {
            "intervention_acc": float(check["intervention_acc"]),
            "n_improved": int(check["n_improved"]),
            "n_worsened": int(check["n_worsened"]),
        },
        "drift_acc": drift_acc,
        "drift_questions": drift_questions,
    } if drift else None

    if drift and not args.allow_exact_drift:
        print(f"  EXACT CHECKSUM FAILED ({drift_questions:+d} question(s) of {n_holdout}) "
              f"— nothing written:")
        for d in drift:
            print(f"    - {d}")
        return "failed"
    if drift:
        print(f"  WARNING: exact checksum drifted by {drift_questions:+d} question(s) of "
              f"{n_holdout}, continuing because --allow-exact-drift:")
        for d in drift:
            print(f"    - {d}")
    else:
        print("  exact checksum OK — split, vector and inference reproduce the recorded run")

    if args.checksum_only:
        return "checksum-ok"

    # ── the actual repair ──────────────────────────────────────────
    # Prove the running checkout contains fed574a instead of taking FIX_COMMIT's word
    # for it — otherwise a stale clone would rewrite results with the same bug and
    # label them fixed.
    bias_error = bias_free_displacement_error(sae, device)
    if bias_error != 0.0:
        print(f"  APPROXIMATION IS STILL BIASED (zero intervention displaces "
              f"{bias_error:.4f}) — this checkout predates {FIX_COMMIT[:7]}. "
              f"Nothing written.")
        return "failed"
    print("  approximation verified bias-free (zero intervention displaces exactly 0)")

    approx_vec = compute_approximated_hidden_vector(vec, sae)
    print(f"  Approximated hidden vector norm (fixed): {approx_vec.norm().item():.4f}")

    r_filtered = _evaluate_report(
        "Filtered + Approximated", holdout_filtered, baseline_filtered_results,
        baseline_filtered_acc, vec, "approx", kw, approx_vec,
    )
    r_full = _evaluate_report_full(
        "Full + Approximated", holdout, baseline_dict, baseline_acc,
        vec, "approx", kw, approx_vec,
    )

    generalisation = None
    if spec.generalisation and not args.skip_generalisation:
        gen_kwargs = dict(
            model=model, tokenizer=tokenizer, hook_mgr=hook_mgr, sae=sae,
            device=device, batch_size=config.batch_size,
        )
        if args.verbatim_generalisation:
            generalisation = run_generalisation(
                approx_vec, [str(ROOT / p) for p in spec.generalisation], **gen_kwargs,
            )
        else:
            generalisation = run_generalisation_shared(
                approx_vec, [str(ROOT / p) for p in spec.generalisation],
                model_short_name=config.model.short_name, **gen_kwargs,
            )

        stored_gen = {b["benchmark"]: b for b in (summary.get("generalisation") or {}).get("benchmarks", [])}
        gen_drift = [
            f"{b['benchmark']}: baseline {b['baseline_acc']!r} != stored "
            f"{stored_gen[b['benchmark']]['baseline_acc']!r}"
            for b in generalisation["benchmarks"]
            if b["benchmark"] in stored_gen
            and abs(b["baseline_acc"] - stored_gen[b["benchmark"]]["baseline_acc"]) > TOL
        ]
        if gen_drift and not args.allow_exact_drift:
            print("  GENERALISATION BASELINE MISMATCH — nothing written:")
            for d in gen_drift:
                print(f"    - {d}")
            return "failed"

    # ── write back ─────────────────────────────────────────────────
    # Only record what this run actually replaces — a "previous" entry for a key that
    # was left alone (e.g. generalisation under --skip-generalisation) would misread as
    # a repair that happened.
    prior = (summary.get("approximation_fix") or {}).get("previous") or {}

    def _original(key: str, current: Any) -> Any:
        """On a second repair pass keep the genuinely pre-fix value.

        A cell repaired once with --skip-generalisation and then again in full would
        otherwise record the first repair's output as "previous", losing the original.
        """
        return prior[key] if key in prior else current

    previous = {
        "filtered_approx": _original(
            "filtered_approx", summary["full_reporting"]["filtered_approx"]),
        "full_approx": _original("full_approx", summary["full_reporting"]["full_approx"]),
        "approximation_comparison": _original(
            "approximation_comparison", summary["approximation_comparison"]),
    }
    if generalisation is not None:
        previous["generalisation_summary"] = _original(
            "generalisation_summary", (summary.get("generalisation") or {}).get("summary"))

    summary["full_reporting"]["filtered_approx"] = as_summary_report(r_filtered)
    summary["full_reporting"]["full_approx"] = as_summary_report(r_full)
    summary["approximation_comparison"] = {
        "filtered": {
            "exact_delta": float(summary["full_reporting"]["filtered_exact"]["delta"]),
            "approx_delta": float(r_filtered["delta"]),
            "difference": float(
                r_filtered["delta"] - summary["full_reporting"]["filtered_exact"]["delta"]
            ),
        },
        "full": {
            "exact_delta": float(summary["full_reporting"]["full_exact"]["delta"]),
            "approx_delta": float(r_full["delta"]),
            "difference": float(
                r_full["delta"] - summary["full_reporting"]["full_exact"]["delta"]
            ),
        },
    }
    if generalisation is not None:
        summary["generalisation"] = generalisation

    if not args.no_provenance:
        summary["approximation_fix"] = {
            "rerun_at": datetime.now().isoformat(timespec="seconds"),
            "fix_commit": FIX_COMMIT,
            "script": "scripts/rerun_approx_fix.py",
            "recomputed": [
                "full_reporting.filtered_approx", "full_reporting.full_approx",
                "approximation_comparison",
            ] + (["generalisation"] if generalisation is not None else []),
            "exact_checksum": "drifted" if drift else "reproduced",
            "exact_checksum_drift": drift_record,
            "approximation_verified_bias_free": True,
            "reconstruction_warnings": warnings,
            # The original run logs record only "Device: cuda", so pin the hardware and
            # stack that produced these numbers.
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
                ),
            },
            "previous": previous,
        }

    # newline="\n" so a Windows run cannot rewrite every line as CRLF and turn a
    # handful of changed values into a whole-file diff.
    spath.write_text(
        json.dumps(summary, indent=2, cls=_NumpySafeEncoder), encoding="utf-8", newline="\n",
    )
    print(f"  updated {spath.relative_to(ROOT)}")
    print(f"    filtered_approx: {previous['filtered_approx']['delta']*100:+.2f}% "
          f"-> {r_filtered['delta']*100:+.2f}%")
    print(f"    full_approx:     {previous['full_approx']['delta']*100:+.2f}% "
          f"-> {r_full['delta']*100:+.2f}%")
    if generalisation is not None and previous["generalisation_summary"]:
        print(f"    gen avg_delta:   {previous['generalisation_summary']['avg_delta']*100:+.2f}% "
              f"-> {generalisation['summary']['avg_delta']*100:+.2f}%")

    # ── figure ─────────────────────────────────────────────────────
    try:
        runtime_reports = {
            "filtered_exact": dict(summary["full_reporting"]["filtered_exact"]),
            "full_exact": dict(summary["full_reporting"]["full_exact"]),
            "filtered_approx": r_filtered,
            "full_approx": r_full,
        }
        plot_file = generate_all_plots(
            full_reporting_results=runtime_reports,
            df_grid=pd.DataFrame(summary["all_grid_results"]),
            best_config=best_config,
            screening_data=screening_data,
            filtered_data=split["filtered"],
            causal_config=config.causal,
            experiment_config=config,
            output_dir=str(exp_dir),
        )
        print(f"  regenerated {plot_file}")
    except Exception as exc:  # plotting must never cost us the JSON
        print(f"  WARNING: plot regeneration failed: {exc}")

    del sae, vec, approx_vec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "repaired"


def is_cuda_fatal(exc: BaseException) -> bool:
    """Whether this error leaves the process's CUDA context unusable.

    A device-side assert poisons the context for the lifetime of the process: every
    later CUDA call raises the same error no matter what it was asked to do. Cells
    attempted afterwards cannot succeed, so the batch has to stop rather than charge one
    failure to each of them. Out-of-memory is deliberately excluded — the context
    survives it, and ``empty_cache()`` plus a smaller next cell may well proceed.
    """
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None and isinstance(exc, oom):
        return False
    if type(exc).__name__ in {"AcceleratorError", "CudaError"}:
        return True
    text = str(exc)
    return "CUDA error" in text or "device-side assert" in text


def release_model(model_slot: Dict[str, Any]) -> None:
    if model_slot.get("hook_mgr") is not None:
        model_slot["hook_mgr"].remove_cache_hook()
    model_slot.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── CLI ───────────────────────────────────────────────────────────


def select_cells(args) -> List[Experiment]:
    by_id = {_extract_exp_id(e.name): e for e in EXPERIMENTS}

    if args.only:
        wanted = [f"EXP-{int(x.strip())}" for x in args.only.split(",")]
        missing = [w for w in wanted if w not in by_id]
        if missing:
            sys.exit(f"Unknown experiment(s) in --only: {', '.join(missing)}")
        return [by_id[w] for w in wanted]

    cells = []
    for exp_id, spec in by_id.items():
        exp_dir = ROOT / "results" / exp_id
        spath = summary_path(exp_dir)
        if spath is None:
            continue
        try:
            summary = json.loads(spath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if is_pre_fix(summary) or args.force:
            cells.append(spec)
    # Group by model so the loaded model and the shared baselines get reused.
    return sorted(cells, key=lambda e: (e.sae, e.layer, e.name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute approximation-dependent results after fix fed574a.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="every completed pre-fix cell")
    group.add_argument("--only", metavar="N[,N...]", help="specific experiment numbers")
    group.add_argument("--list", action="store_true", help="list the cells that would run")
    parser.add_argument("--dry-run", action="store_true",
                        help="offline reconstruction checks only, no GPU, no writes")
    parser.add_argument("--checksum-only", action="store_true",
                        help="run the exact-report checksum on the GPU and stop; writes "
                             "nothing. Use this to confirm the current box reproduces the "
                             "recorded runs before committing to the full repair.")
    parser.add_argument("--skip-generalisation", action="store_true",
                        help="repair the holdout approx reports only")
    parser.add_argument("--verbatim-generalisation", action="store_true",
                        help="call run_generalisation() from src instead of the shared-baseline path")
    parser.add_argument("--reuse-model", action="store_true",
                        help="keep one loaded model across consecutive cells that share it. "
                             "Saves ~80s per cell but is how the EXP-322 device-side assert "
                             "was triggered; off by default until that is understood.")
    parser.add_argument("--force", action="store_true",
                        help="also process cells whose timestamp is already post-fix")
    parser.add_argument("--allow-exact-drift", action="store_true",
                        help="write results even if the exact checksum does not reproduce")
    parser.add_argument("--no-provenance", action="store_true",
                        help="omit the approximation_fix provenance block from the JSON")
    args = parser.parse_args()

    cells = select_cells(args)
    if args.list:
        print(f"{len(cells)} cell(s):")
        for spec in cells:
            print(f"  {spec.name}")
        return
    if not cells:
        print("Nothing to do.")
        return

    print(f"{len(cells)} cell(s) to process"
          f"{' (dry run)' if args.dry_run else ''}")

    counts: Dict[str, int] = {}
    model_slot: Dict[str, Any] = {}
    started = datetime.now()
    aborted_after: Optional[str] = None
    not_attempted = 0
    try:
        for i, spec in enumerate(cells):
            fatal = False
            try:
                status = rerun_cell(spec, args, model_slot)
            except Exception as exc:
                # Always keep the stack: a device-side assert is reported
                # asynchronously, so its message names no line of code and the
                # traceback is the only thing that narrows the search.
                traceback.print_exc(file=sys.stdout)
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                status = "failed"
                fatal = is_cuda_fatal(exc)
            counts[status] = counts.get(status, 0) + 1
            if fatal:
                aborted_after, not_attempted = spec.name, len(cells) - i - 1
                print(f"\n  CUDA context is unusable after this error — every later cell "
                      f"could only re-report it. Stopping with {not_attempted} cell(s) "
                      f"not attempted.")
                break
    finally:
        # The context that just died is the same one empty_cache() talks to, so this
        # can raise too. Letting it would discard the whole run report.
        try:
            release_model(model_slot)
        except Exception as exc:
            print(f"  WARNING: releasing the model failed: {type(exc).__name__}: {exc}")

    elapsed = datetime.now() - started
    print("\n" + "=" * 80)
    print(f"Done in {elapsed}. " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())))
    if not_attempted:
        print(f"ABORTED on a CUDA error at {aborted_after} — {not_attempted} cell(s) not "
              f"attempted. Diagnose with CUDA_LAUNCH_BLOCKING=1 on that cell alone, then "
              f"re-run the batch; cells already repaired are skipped automatically.")
    if counts.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
