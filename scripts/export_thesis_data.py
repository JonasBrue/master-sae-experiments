#!/usr/bin/env python
"""Export the completed final-matrix cells into the thesis data bundle.

Read-only with respect to ``results/``. Writes ``thesis/data/``:

    cells.csv        one flat row per completed cell, read by Typst's csv()
    cells.jsonl      the same cells with the nested blocks kept intact
    canonical.json   the reference experiment's result JSON, verbatim
    constants.json   pipeline settings, with the count of cells agreeing
    screening.csv    one row per distinct (model, benchmark) screening
    RESULTS.md       human-readable digest: coverage, axes, robustness

Cells are never selected on their outcome. The population is every cell whose
status in ``final_matrix_summary.jsonl`` is ``done``; cells that never ran are
absent from the bundle entirely, per outline decision 10.6.

    uv run python scripts/export_thesis_data.py
    uv run python scripts/export_thesis_data.py --out-dir /tmp/data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from math import comb, erfc, sqrt

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
SUMMARY = RESULTS / "final_matrix_summary.jsonl"
DEFAULT_OUT = REPO.parent / "thesis" / "data"

CANONICAL = "EXP-355"
ALPHA = 0.05

# A cell is flagged when one p_correct value covers more than this share of the
# benchmark. There the difficulty screen returns what is effectively a constant score,
# so Phase 0's band selects everything and its matched-block split has no ordering to
# sort by: the stage has no input to work on rather than a noisy one. Observed values
# are 0.996 and 0.991 (Pythia-70M) against a next-highest of 0.557, so any threshold in
# (0.56, 0.99) yields the same partition. NOT a fixed rule — screening.csv records
# modal_p_share, n_distinct_p, top_letter and top_letter_share, so the flag can be
# recomputed at any threshold without re-exporting.
DEGENERATE_MODAL_SHARE = 0.90

# Reporting keys carry an optional _static / _adaptive suffix.
READINGS = ("filtered_exact", "full_exact", "filtered_approx", "full_approx")


# --------------------------------------------------------------------------- io

def _report(full_reporting: dict, key: str) -> dict:
    if key in full_reporting:
        return full_reporting[key]
    for suffix in ("_static", "_adaptive"):
        if key + suffix in full_reporting:
            return full_reporting[key + suffix]
    return {}


_DISCOVERY_RE = re.compile(rb"Successes:\s*(\d+),\s*Failures:\s*(\d+)")


def _discovery_counts(exp: str) -> tuple[int | None, int | None]:
    """Class balance of the discovery subset, which Phase 1 prints but never stores.

    It is the concrete instance of the argument in methodology 3.3 that raw counts are
    normalised because the correct and incorrect groups differ in size. Reading it back
    out of the log follows the pattern summarize_final_matrix.py already uses for
    durations, so this adds no new dependency.
    """
    logs = sorted((RESULTS / exp).glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in logs:
        try:
            m = _DISCOVERY_RE.search(p.read_bytes())
        except OSError:
            continue
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def _screening_digest(exp: str, p_min: float, p_max: float) -> dict:
    """Deciles of p_correct over the whole benchmark, plus its band decomposition.

    Screening is SAE-independent, so one file per distinct screening is enough. The
    deciles are what let the chapter say that band share tracks where a model's
    probability mass sits rather than how accurate it is.
    """
    files = list((RESULTS / exp).glob("logit_screening_*.json"))
    if not files:
        return {}
    rows = json.loads(files[0].read_text(encoding="utf-8"))
    n = len(rows)
    if not n:
        return {}
    ps = sorted(r["p_correct"] for r in rows)
    out = {f"d{q*10}": round(ps[min(n - 1, n * q // 10)], 6) for q in range(1, 10)}
    out["bench_below_pmin_share"] = round(sum(1 for p in ps if p < p_min) / n, 4)
    out["bench_inband_share"] = round(sum(1 for p in ps if p_min <= p <= p_max) / n, 4)
    out["bench_above_pmax_share"] = round(sum(1 for p in ps if p > p_max) / n, 4)

    # Resolution diagnostics. These are the raw measurements behind the `degenerate`
    # flag; the flag is derived from modal_p_share against DEGENERATE_MODAL_SHARE, so a
    # later decision to move or drop that threshold needs no re-export.
    counts = Counter(round(p, 6) for p in ps)
    letters = Counter(r["prediction"] for r in rows)
    out["n_distinct_p"] = len(counts)
    out["modal_p_share"] = round(counts.most_common(1)[0][1] / n, 4)
    top_letter, top_n = letters.most_common(1)[0]
    out["top_letter"] = top_letter
    out["top_letter_share"] = round(top_n / n, 4)
    return out


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# Inclusion rule, applied before any statistic is computed and independently of
# outcome: a model enters the study only if it was run on both primary benchmarks.
# These four carry one benchmark only, so their five completed cells are excluded.
# All four are non-significant on the primary endpoint, so the rule cuts neither
# for nor against the thesis; it lives here rather than in the thesis because the
# bundle is what defines the reported study.
#
# google/gemma-2b-it is deliberately NOT excluded: its ARC-Challenge counterpart
# EXP-504 is queued, after which it satisfies the rule. Drop this note and add the
# model above only if that run is abandoned.
EXCLUDED_MODELS = frozenset({
    "openai-community/gpt2",
    "google/gemma-3-4b-pt",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-2-9b",
})


def load_cells() -> list[dict]:
    rows = [json.loads(l) for l in SUMMARY.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [
        r for r in rows
        if r["status"] == "done" and r["model_id"] not in EXCLUDED_MODELS
    ]


# ------------------------------------------------------------------ statistics

def binom_ge(k: int, n: int, p: float = 0.5) -> float:
    """One-sided P(X >= k) for X ~ Bin(n, p)."""
    if n <= 0:
        return 1.0
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def sign_test(k: int, n: int) -> float:
    """Two-sided sign test on k of n untied comparisons."""
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, lo + 1)) / 2 ** n)


def fisher_one_sided(a: int, n1: int, b: int, n2: int) -> float:
    """P(>= a successes in group 1) under the hypergeometric null."""
    k, n = a + b, n1 + n2
    if not k or n1 + n2 == 0:
        return 1.0
    return sum(comb(n1, i) * comb(n2, k - i)
               for i in range(a, min(n1, k) + 1) if 0 <= k - i <= n2) / comb(n, k)


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation with midranks for ties (matches scipy.stats.spearmanr)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mid = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mid
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def _release_family(repo: str) -> tuple[str, str]:
    """Provider plus release name with the site suffix stripped.

    Site pairs must be matched on this, not on sae_config: keying cells on
    (model, layer, benchmark, site) silently overwrites when a model has two releases at
    one layer, which once produced a Seonglae resid SAE paired against a Gemma Scope mlp
    SAE at gemma2_2b L19.
    """
    provider, _, name = repo.partition("/")
    name = name.lower()
    for suffix in ("-pt-res", "-pt-mlp", "-res", "-mlp", "_res", "_mlp"):
        name = name.replace(suffix, "")
    return provider.lower(), name


def _release_stem(config: str) -> str:
    """Collapse per-layer config stems of one SAE release into a single family.

    Gemma Scope 1 on Gemma-2 2B ships one config file per layer
    (gemma_scope_2b_pt_res_w16k_l12, ..._l19); every other release spans its layers
    within one stem. Layer contrasts must group on this, or the release loses its
    cross-layer contrasts. Mirrors thesis/figures/depth-contrasts.typ.
    """
    for prefix in ("gemma_scope_2b_pt_res_w16k", "gemma_scope_2b_pt_mlp_w16k"):
        if config.startswith(prefix):
            return prefix
    return config


def mannwhitney_one_sided(xs: list[float], ys: list[float]) -> float:
    """One-sided Mann-Whitney U test, P(xs stochastically greater than ys).

    Normal approximation with midranks, tie correction, and continuity correction,
    matching scipy.stats.mannwhitneyu's asymptotic method.
    """
    n1, n2 = len(xs), len(ys)
    if not n1 or not n2:
        return 1.0
    pooled = sorted([(v, True) for v in xs] + [(v, False) for v in ys])
    vals = [v for v, _ in pooled]
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j < len(vals) and vals[j] == vals[i]:
            j += 1
        for t in range(i, j):
            ranks[t] = (i + j + 1) / 2
        i = j
    r1 = sum(r for r, (_, is_x) in zip(ranks, pooled) if is_x)
    u = r1 - n1 * (n1 + 1) / 2
    nn = n1 + n2
    ties = sum(t ** 3 - t for t in Counter(vals).values())
    var = n1 * n2 / 12 * ((nn + 1) - ties / (nn * (nn - 1)))
    if var <= 0:
        return 1.0
    z = (u - n1 * n2 / 2 - 0.5) / sqrt(var)
    return 0.5 * erfc(z / sqrt(2))


def benjamini_hochberg(pvals: list[float], q: float) -> int:
    """Number of discoveries under BH control at level q."""
    ordered = sorted(pvals)
    n = len(ordered)
    cut = 0
    for i, p in enumerate(ordered, start=1):
        if p <= i / n * q:
            cut = i
    return cut


# ------------------------------------------------------------------ extraction

def extract(cell: dict) -> dict:
    """Flatten one cell's result JSON into the fields Chapter 4 consumes."""
    j = json.loads((RESULTS / cell["exp"] / cell["result_json"]).read_text(encoding="utf-8"))
    ds, cs, kf = j["data_stats"], j["causal_screening"], j["kfold_results"]
    bc = j["best_config"]
    n_succ, n_fail = _discovery_counts(cell["exp"])

    rec: dict = {
        # identity and axes
        "exp": cell["exp"],
        "model": cell["model"],
        "model_id": cell["model_id"],
        "params": cell["params"],
        "d_model": j["model"]["d_model"],
        "n_layers": j["model"]["n_layers"],
        "sae_config": cell["sae_config"],
        "sae_name": cell["sae_name"],
        "sae_repo": j["sae"]["repo"],
        # NOT the architecture the release declares. This is the activation path our
        # loader was configured to use, and the two disagree on several releases:
        # chanind/...batch-topk-matryoshka... declares "jumprelu", chanind/sae-gemma-2-2b-tied-topk
        # declares "topk" where we configure "batchtopk", and the EleutherAI suites declare a bare
        # k with no batch flag. Our BatchTopKSparseAutoEncoder.encode is per-example topk over the
        # feature axis, which at batch size 1 coincides with BatchTopK anyway. Do NOT aggregate
        # this into a family split for the thesis; Appendix C carries the per-release architecture,
        # checked by hand against each model card.
        "loader_activation": cell["arch"],
        "width": cell["width"],
        "expansion": round(cell["width"] / j["model"]["d_model"], 2) if j["model"]["d_model"] else None,
        "top_k": cell["top_k"],
        "site": cell["site"],
        "layer": cell["layer"],
        "layer_frac": cell["layer_frac"],
        "benchmark": cell["benchmark"],
        "n_choices": len(j["benchmark"]["choices"]),
        "chance": round(1.0 / len(j["benchmark"]["choices"]), 4),
        # provenance
        "duration_s": cell["duration_s"],
        "timestamp": cell["timestamp"],
        "approx_repaired": "approximation_fix" in j,
        # data sizes
        "n_total": ds["total_questions"],
        "n_filtered": ds["filtered_questions"],
        "n_discovery": ds["train_questions"],
        "n_validation": ds["val_questions"],
        "n_reserve": ds["test_filtered_questions"],
        "n_selection": ds["grid_eval_questions"],
        "n_holdout": ds["holdout_questions"],
        "n_holdout_filtered": ds["holdout_filtered_questions"],
        "band_share": round(ds["filtered_questions"] / ds["total_questions"], 4) if ds["total_questions"] else None,
        "p_min": ds["p_correct_range"][0],
        "p_max": ds["p_correct_range"][1],
        "base_acc_full": ds["baseline_full_accuracy"],
        "base_acc_filtered": ds["baseline_filtered_accuracy"],
        "below_chance": ds["baseline_full_accuracy"] < 1.0 / len(j["benchmark"]["choices"]),
        "min_occurrences": ds["dynamic_min_occurrences"],
        # phase 1 and 2. phase2_tested IS the Phase-1 output: run_phase1 returns candidate
        # lists already capped at n_candidate_features_pos/neg and filtered by
        # min_occurrences, and Phase 2 tests exactly those.
        "n_discovery_success": n_succ,
        "n_discovery_failure": n_fail,
        "discovery_accuracy": (round(n_succ / (n_succ + n_fail), 6)
                               if n_succ is not None and n_fail is not None and (n_succ + n_fail)
                               else None),
        "phase2_tested": cs["n_unique_tested_positive"] + cs["n_unique_tested_negative"],
        "phase2_tested_pos": cs["n_unique_tested_positive"],
        "phase2_tested_neg": cs["n_unique_tested_negative"],
        "phase2_survived": cs["n_causal_positive_unique_features"] + cs["n_causal_negative_unique_features"],
        "phase2_survived_pos": cs["n_causal_positive_unique_features"],
        "phase2_survived_neg": cs["n_causal_negative_unique_features"],
        # phase 3
        "best_top_k_pos": bc["top_k_pos"],
        "best_top_k_neg": bc["top_k_neg"],
        "best_strength": bc["strength"],
        "best_ranking": bc["ranking_criterion"],
        "best_n_features": bc["n_features_used"],
        "n_grid_combinations": j["n_combinations"],
    }

    rec["phase2_rejection_rate"] = (
        round(1 - rec["phase2_survived"] / rec["phase2_tested"], 4) if rec["phase2_tested"] else None
    )

    # phase 4, all four readings with their discordant counts
    for key, short in zip(READINGS, ("flt_exact", "full_exact", "flt_approx", "full_approx")):
        r = _report(j["full_reporting"], key)
        rec[f"{short}_n"] = r.get("n_questions")
        rec[f"{short}_base"] = r.get("baseline_acc")
        rec[f"{short}_acc"] = r.get("intervention_acc")
        rec[f"{short}_delta"] = r.get("delta")
        rec[f"{short}_improved"] = r.get("n_improved")
        rec[f"{short}_worsened"] = r.get("n_worsened")
        rec[f"{short}_p"] = r.get("binomial_p")
        rec[f"{short}_sig"] = r.get("significant")

    ac = j.get("approximation_comparison", {})
    rec["approx_gap_full"] = ac.get("full", {}).get("difference")
    rec["approx_gap_filtered"] = ac.get("filtered", {}).get("difference")

    rec.update({
        "kfold_n": kf["n_folds"],
        "kfold_mean": kf["mean_delta"],
        "kfold_std": kf["std_delta"],
        "kfold_ci_lo": kf["ci_95_lower"],
        "kfold_ci_hi": kf["ci_95_upper"],
        "kfold_pos_folds": kf["n_positive_folds"],
        "kfold_ci_excludes_zero": kf["ci_95_lower"] > 0 or kf["ci_95_upper"] < 0,
    })

    # selection-set delta of the winning configuration, and the grid's shape
    grid = j["all_grid_results"]
    win = next(
        (g for g in grid
         if all(g[k] == bc[k] for k in ("top_k_pos", "top_k_neg", "strength", "ranking_criterion"))),
        None,
    )
    rec["selection_delta"] = win["delta_grid_eval"] if win else None
    rec["selection_to_holdout"] = (
        rec["full_exact_delta"] - rec["selection_delta"]
        if win and rec["full_exact_delta"] is not None else None
    )
    deltas = sorted(g["delta_grid_eval"] for g in grid)
    rec["grid_delta_min"] = deltas[0]
    rec["grid_delta_max"] = deltas[-1]
    rec["grid_delta_median"] = statistics.median(deltas)
    rec["grid_positive"] = sum(1 for d in deltas if d > 0)

    rc = j.get("ranking_comparison", {})
    for crit in ("causal_delta", "freq_diff"):
        c = rc.get(crit, {})
        rec[f"rank_{crit}_best"] = c.get("best_delta_full")
        rec[f"rank_{crit}_avg"] = c.get("avg_delta_full")
        rec[f"rank_{crit}_positive"] = c.get("positive_configs_full")

    # the structurally one-sided band: baseline is 0 there, so worsened is forced to 0
    bands = j["difficulty_analysis"]["difficulty_ranges"]
    zero = next((b for b in bands if b["baseline_acc"] == 0.0), None)
    zi, zw = (zero["n_improved"], zero["n_worsened"]) if zero else (0, 0)
    rec["zerobase_n"] = zero["n_questions"] if zero else 0
    rec["zerobase_share"] = (
        round(zero["n_questions"] / rec["full_exact_n"], 4) if zero and rec["full_exact_n"] else 0.0
    )
    rec["zerobase_improved"] = zi
    # Three-part decomposition of the holdout: below p_min (baseline 0 by construction),
    # inside the band, above p_max. Asserted to sum, so a drift in any input is caught.
    _below, _inband = rec["zerobase_n"], rec["n_holdout_filtered"]
    rec["holdout_above_n"] = rec["full_exact_n"] - _below - _inband
    if rec["full_exact_n"]:
        rec["holdout_inband_share"] = round(_inband / rec["full_exact_n"], 4)
        rec["holdout_above_share"] = round(rec["holdout_above_n"] / rec["full_exact_n"], 4)
    else:
        rec["holdout_inband_share"] = rec["holdout_above_share"] = None
    assert rec["holdout_above_n"] >= 0, f"{cell['exp']}: holdout decomposition is negative"
    ri = (rec["full_exact_improved"] or 0) - zi
    rw = (rec["full_exact_worsened"] or 0) - zw
    rec["restricted_improved"] = ri
    rec["restricted_worsened"] = rw
    rec["restricted_p"] = binom_ge(ri, ri + rw) if (ri + rw) else 1.0
    rec["restricted_survives"] = bool(ri > rw and rec["restricted_p"] < ALPHA)
    rec["restricted_reverses"] = bool(ri < rw)

    # transfer to the three benchmarks disjoint from selection
    gen = j["generalisation"]
    rec["transfer_n_positive"] = gen["summary"]["n_positive"]
    rec["transfer_n_significant"] = gen["summary"]["n_significant"]
    rec["transfer_avg_delta"] = gen["summary"].get("avg_delta")
    for b in gen["benchmarks"]:
        slug = {"ARC-Challenge": "arc", "CommonsenseQA": "csqa",
                "HellaSwag": "hswag", "OpenBookQA": "obqa"}[b["benchmark"]]
        rec[f"tr_{slug}_delta"] = b["delta"]
        rec[f"tr_{slug}_p"] = b["binomial_p"]
        rec[f"tr_{slug}_sig"] = b["significant"]
        rec[f"tr_{slug}_improved"] = b["n_improved"]
        rec[f"tr_{slug}_worsened"] = b["n_worsened"]

    # outcome class, for the abstention analysis of section 4.4
    if rec["phase2_survived"] == 0:
        rec["outcome"] = "no_features"
    elif (rec["full_exact_improved"] or 0) == 0 and (rec["full_exact_worsened"] or 0) == 0:
        rec["outcome"] = "no_change"
    elif rec["full_exact_sig"]:
        rec["outcome"] = "significant"
    else:
        rec["outcome"] = "acted_ns"

    # The full 90-row grid stays out of the bundle: its summary is already in the
    # flat record, and only the reference experiment plots the surface, which
    # canonical.json carries verbatim.
    nested = {
        "folds": kf["fold_details"],
        "bands": bands,
        "transfer": gen["benchmarks"],
    }
    return {"flat": rec, "nested": nested}


# ---------------------------------------------------------------------- digest

def pct(x, nd=2):
    return "-" if x is None else f"{x * 100:+.{nd}f}"


def write_digest(path: pathlib.Path, cells: list[dict], head: str | None) -> None:
    n = len(cells)
    ps = [c["full_exact_p"] for c in cells]
    k = sum(1 for p in ps if p < ALPHA)
    L = []
    add = L.append

    add(f"# Chapter 4 results digest — {n} completed cells\n")
    add(f"Generated from `results/` at git HEAD `{head or 'unknown'}`. "
        f"Population is every cell with status `done`; no cell is selected on its outcome "
        f"(outline 10.6). Regenerate with `scripts/export_thesis_data.py`.\n")

    add("## 1. Coverage\n")
    add(f"- {len(set(c['model'] for c in cells))} models, "
        f"{len(set(c['sae_config'] for c in cells))} SAE releases")
    add(f"- benchmarks: {dict(Counter(c['benchmark'] for c in cells))}")
    add(f"- sites: {dict(Counter(c['site'] for c in cells))}")
    # Reported as loader activation paths, not as SAE families. See the note at the
    # `loader_activation` field for why the two are not interchangeable.
    add(f"- loader activation paths: {dict(Counter(c['loader_activation'] for c in cells))}")
    # Tie-break by model name: several models share a params label, and a
    # key on _pnum alone leaves their order to set iteration, which varies per run.
    order = sorted({(c["params"], c["model"]) for c in cells}, key=lambda t: (_pnum(t[0]), t[1]))
    add(f"- scale range: {order[0][0]} ({order[0][1]}) to {order[-1][0]} ({order[-1][1]})\n")

    add("## 2. Primary endpoint (full holdout, exact intervention)\n")
    ds = sorted(c["full_exact_delta"] for c in cells)
    add(f"- significant at alpha={ALPHA}: **{k} of {n}**, against {n * ALPHA:.1f} expected under a global null")
    add(f"- P(>= {k} | global null) = {binom_ge(k, n, ALPHA):.3g}")
    for q in (0.05, 0.10):
        add(f"- Benjamini-Hochberg at q={q:.2f}: **{benjamini_hochberg(ps, q)}** discoveries")
    add(f"- Bonferroni at {ALPHA}/{n}: {sum(1 for p in ps if p < ALPHA / n)} survive")
    add(f"- effect distribution: min {pct(ds[0])}pp, median {pct(statistics.median(ds))}pp, "
        f"max {pct(ds[-1])}pp")
    add(f"- sign: {sum(1 for d in ds if d > 0)} positive, {sum(1 for d in ds if d == 0)} zero, "
        f"{sum(1 for d in ds if d < 0)} negative\n")

    add("### Significant cells\n")
    add("| exp | model | L | site | bench | full Δ | p | +/− | restricted +/− | restricted verdict | transfer +/3 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted((c for c in cells if c["full_exact_sig"]), key=lambda c: -c["full_exact_delta"]):
        verdict = ("reverses" if c["restricted_reverses"]
                   else "survives" if c["restricted_survives"] else "loses power")
        add(f"| {c['exp']} | {c['model']} | {c['layer']} | {c['site']} | {c['benchmark']} | "
            f"{pct(c['full_exact_delta'])}pp | {c['full_exact_p']:.2g} | "
            f"+{c['full_exact_improved']}/−{c['full_exact_worsened']} | "
            f"+{c['restricted_improved']}/−{c['restricted_worsened']} | {verdict} | "
            f"{c['transfer_n_positive']}/3 |")
    add("")

    add("## 3. Outcome classes (section 4.4)\n")
    oc = Counter(c["outcome"] for c in cells)
    add(f"- `significant` {oc['significant']} · `acted_ns` {oc['acted_ns']} · "
        f"`no_change` {oc['no_change']} · `no_features` {oc['no_features']}")
    inert = sum(1 for c in cells
                if (c["full_exact_improved"] or 0) == 0 and (c["full_exact_worsened"] or 0) == 0)
    add(f"- cells where Phase 2 kept nothing: **{oc['no_features']}** — the causal gate declined to act")
    add(f"- cells whose intervention changed no answer at all: **{inert}** "
        f"(all of them also in the class above, which is why `no_change` is empty: "
        f"the classifier attributes an inert cell to the earlier mechanism that explains it)")
    rr = [c["phase2_rejection_rate"] for c in cells if c["phase2_rejection_rate"] is not None]
    add(f"- Phase-2 rejection rate: median {statistics.median(rr):.1%}, "
        f"min {min(rr):.1%}, max {max(rr):.1%}\n")
    for c in sorted((c for c in cells if c["outcome"] in ("no_features", "no_change")),
                    key=lambda c: (c["model"], c["benchmark"])):
        add(f"  - {c['exp']} {c['model']} {c['benchmark']} {c['site']} L{c['layer']} — "
            f"{c['outcome']}, tested {c['phase2_tested']}, survived {c['phase2_survived']}, "
            f"Δ {pct(c['full_exact_delta'])}pp")
    add("")

    add("## 4. Measurement validity (section 4.5)\n")
    zs = sorted(c["zerobase_share"] for c in cells)
    add(f"- share of the full holdout with a 0% baseline: median {statistics.median(zs):.1%}, "
        f"max {max(zs):.1%}")
    add(f"- cells with a below-chance baseline: {sum(1 for c in cells if c['below_chance'])}")
    drawn = [c for c in cells if c["zerobase_improved"] > 0]
    add(f"- cells drawing any improvement from that band: {len(drawn)}")
    for c in sorted(drawn, key=lambda c: -c["zerobase_improved"]):
        add(f"  - {c['exp']} {c['model']} {c['benchmark']}: {c['zerobase_improved']} of "
            f"{c['full_exact_improved']} improvements, Δ {pct(c['full_exact_delta'])}pp")
    add("")

    add("## 5. Pipeline components (section 4.4)\n")
    add(f"- ranking criterion chosen: {dict(Counter(c['best_ranking'] for c in cells))}")
    add(f"- strength chosen: {dict(Counter(c['best_strength'] for c in cells))}")
    add(f"- k+ chosen: {dict(Counter(c['best_top_k_pos'] for c in cells))}")
    add(f"- k− chosen: {dict(Counter(c['best_top_k_neg'] for c in cells))}")
    gaps = sorted(abs(c["approx_gap_full"]) for c in cells if c["approx_gap_full"] is not None)
    add(f"- |exact − approximated| on the full holdout: median {gaps[len(gaps)//2]*100:.3f}pp, "
        f"max {gaps[-1]*100:.3f}pp; within 0.1pp in "
        f"{sum(1 for g in gaps if g < 0.001)}/{len(gaps)} cells")
    sh = [c["selection_to_holdout"] for c in cells if c["selection_to_holdout"] is not None]
    add(f"- holdout minus selection Δ: median {statistics.median(sh)*100:+.3f}pp; "
        f"holdout exceeds selection in {sum(1 for x in sh if x > 0)}/{len(sh)} cells\n")

    add("## 6. Transfer (RQ4)\n")
    add(f"- {n} cells x 3 benchmarks = {n*3} transfer measurements")
    add(f"- cells positive on all three: {sum(1 for c in cells if c['transfer_n_positive'] == 3)}")
    add(f"- cells significant on at least one: "
        f"{sum(1 for c in cells if c['transfer_n_significant'] > 0)}\n")

    add("## 7. Axis populations\n")
    # Same pairing rule as the section-8 RQ3 test and thesis/figures/site-pairs.typ:
    # within each (model, layer, benchmark) the best release-family-matched
    # resid-mlp combination, never a raw (model, layer, benchmark, site) lookup,
    # which silently overwrites when two releases share a layer.
    site_groups = defaultdict(lambda: defaultdict(list))
    for c in cells:
        site_groups[(c["model"], c["layer"], c["benchmark"])][c["site"]].append(c)
    matched = []
    for (m, l, b), sites in site_groups.items():
        if "resid" not in sites or "mlp" not in sites:
            continue
        fam, r, mm = max(((_release_family(r["sae_repo"]) == _release_family(mm["sae_repo"]), r, mm)
                          for r in sites["resid"] for mm in sites["mlp"]), key=lambda t: t[0])
        matched.append((m, l, b, fam, r, mm))
    add(f"### RQ3 — {len(matched)} site pairs across "
        f"{len(set(m for m, *_ in matched))} models\n")
    add("| model | L | bench | resid Δ | mlp Δ | releases matched |")
    add("|---|---|---|---|---|---|")
    for m, l, b, fam, r, mm in sorted(matched, key=lambda t: (t[0], t[1], t[2])):
        add(f"| {m} | {l} | {b} | {pct(r['full_exact_delta'])}pp | "
            f"{pct(mm['full_exact_delta'])}pp | {'yes' if fam else 'no'} |")
    add("")

    byms = defaultdict(list)
    for c in cells:
        byms[(c["model"], c["site"], c["benchmark"], _release_stem(c["sae_config"]))].append(c)
    contrasts = {k: v for k, v in byms.items() if len({c["layer"] for c in v}) >= 2}
    add(f"### RQ2 — {len(contrasts)} controlled layer contrasts "
        f"(model, site, benchmark, and SAE release all fixed)\n")
    for (m, s, b, sae), v in sorted(contrasts.items()):
        seq = "  ".join(f"L{c['layer']}:{pct(c['full_exact_delta'])}pp"
                        for c in sorted(v, key=lambda c: c["layer"]))
        add(f"- {m} · {s} · {b} · `{sae}` — {seq}")
    add("")

    add("### RQ1 — provider-controlled scale ladders\n")
    # "gemma_scope" catches Gemma Scope 2 on Gemma-3 270M/1B and Gemma Scope 1 on
    # Gemma-2 2B, all width 16,384 — the constant-width suite outline 10.9 names.
    for prefix, label in (("gemma_scope", "Gemma Scope, width 16,384 (Gemma-3 270M/1B + Gemma-2 2B)"),
                          ("eleutherai_pythia", "EleutherAI / Pythia")):
        sel = sorted((c for c in cells if c["sae_config"].startswith(prefix)),
                     key=lambda c: (_pnum(c["params"]), c["layer"], c["site"], c["benchmark"]))
        if not sel:
            continue
        add(f"**{label}** ({len(sel)} cells)\n")
        add("| params | model | L | l/N | site | bench | full Δ | p | base | survival |")
        add("|---|---|---|---|---|---|---|---|---|---|")
        for c in sel:
            surv = (f"{c['phase2_survived'] / c['phase2_tested']:.0%}"
                    if c["phase2_tested"] else "-")
            add(f"| {c['params']} | {c['model']} | {c['layer']} | {c['layer_frac']} | {c['site']} | "
                f"{c['benchmark']} | {pct(c['full_exact_delta'])}pp | {c['full_exact_p']:.3f} | "
                f"{c['base_acc_full']*100:.1f}% | {surv} |")
        add("")

    add("## 8. Research-question tests\n")
    add(f"Every axis test is reported twice: over all {n} cells, and over the "
        f"{sum(1 for c in cells if not c['degenerate'])} cells whose difficulty screen returns a "
        f"varying score. The flagged cells are those where one `p_correct` value covers more than "
        f"{DEGENERATE_MODAL_SHARE:.0%} of the benchmark, so Phase 0 has no ordering to sort by and "
        f"no subset to select. The exclusion rule is *not* fixed — `screening.csv` carries "
        f"`modal_p_share`, so any threshold can be applied without re-exporting.\n")

    for label, pool in (("all cells", cells),
                        ("varying-score cells only", [c for c in cells if not c["degenerate"]])):
        m = len(pool)
        add(f"### {label} (n = {m})\n")

        ps_ = [c["full_exact_p"] for c in pool]
        k_ = sum(1 for p in ps_ if p < ALPHA)
        add(f"- **CQ** — {k_} significant at alpha={ALPHA} against {m*ALPHA:.1f} expected under a "
            f"global null, P(>= {k_}) = {binom_ge(k_, m, ALPHA):.3g}; "
            f"BH q=0.05 -> {benjamini_hochberg(ps_, 0.05)}, q=0.10 -> {benjamini_hochberg(ps_, 0.10)}")

        # The strata the chapter reports (4.3.1): 410M and below, 1-2B, 8B.
        strata = [(0, 0.5, "<=410M"), (0.5, 3, "1-2B"), (3, 1e9, "8B")]
        acc, surv = [], []
        for lo, hi, name in strata:
            sel = [c for c in pool if lo <= _pnum(c["params"]) < hi]
            if not sel:
                continue
            sig = sum(1 for c in sel if c["full_exact_sig"])
            sv = [c["phase2_survived"] / c["phase2_tested"] for c in sel if c["phase2_tested"]]
            acc.append(f"{name} {sig}/{len(sel)}")
            surv.append(f"{name} {statistics.median(sv):.1%}" if sv else f"{name} -")
        add(f"- **RQ1a** accuracy by scale — significant per stratum: {', '.join(acc)}")
        add(f"- **RQ1b** Phase-2 survival by scale — median: {', '.join(surv)}")
        withs = [c for c in pool if c["phase2_tested"]]
        sc = [(_pnum(c["params"]), c["phase2_survived"] / c["phase2_tested"]) for c in withs]
        if len(sc) > 2:
            add(f"  - Spearman(survival, scale) = {spearman([a for a, _ in sc], [b for _, b in sc]):+.3f}; "
                f"Spearman(survival, validation size) = "
                f"{spearman([c['n_validation'] for c in withs], [b for _, b in sc]):+.3f}"
                f" — the confound runs the opposite way; "
                f"Spearman(survival, dictionary width) = "
                f"{spearman([c['width'] for c in withs], [b for _, b in sc]):+.3f}")
            bym = defaultdict(list)
            for c in withs:
                bym[(_pnum(c["params"]), c["model"])].append(c["phase2_survived"] / c["phase2_tested"])
            pm = sorted((p, statistics.median(v)) for (p, _), v in bym.items())
            add(f"  - per-model median survival vs scale (n = {len(pm)}): Spearman = "
                f"{spearman([p for p, _ in pm], [s for _, s in pm]):+.3f}")
            bins = ((0, 300), (300, 1000), (1000, 3000))
            parts = []
            for lo, hi in bins:
                grp = [c for c in withs if lo <= c["n_validation"] < hi]
                small = [c["phase2_survived"] / c["phase2_tested"] for c in grp if _pnum(c["params"]) < 1.5]
                large = [c["phase2_survived"] / c["phase2_tested"] for c in grp if _pnum(c["params"]) >= 1.5]
                if small and large:
                    parts.append(f"[{lo},{hi}): {statistics.median(large):.1%} (n={len(large)}) vs "
                                 f"{statistics.median(small):.1%} (n={len(small)})")
            add(f"  - survival within validation-size bins, models >= 1.5B vs < 1.5B, medians: "
                f"{'; '.join(parts)}")

        by = defaultdict(list)
        for c in pool:
            by[(c["model"], c["site"], c["benchmark"], _release_stem(c["sae_config"]))].append(c)
        sh = dp = ti = 0
        gaps_pp = []
        for v in by.values():
            if len({c["layer"] for c in v}) < 2:
                continue
            v = sorted(v, key=lambda c: c["layer"])
            da, db = v[0]["full_exact_delta"], v[-1]["full_exact_delta"]
            gaps_pp.append(abs(da - db) * 100)
            ti, sh, dp = (ti + 1, sh, dp) if da == db else (ti, sh + (da > db), dp + (da < db))
        add(f"- **RQ2** layer depth — {sh + dp + ti} controlled contrasts: {sh} shallower larger, "
            f"{dp} deeper larger, {ti} tied; two-sided sign test p = {sign_test(sh, sh + dp):.3f}; "
            f"median |gap| {statistics.median(gaps_pp):.2f}pp, "
            f"{sum(1 for g in gaps_pp if g <= 0.5)}/{len(gaps_pp)} within half a point")

        idx = {(c["model"], c["layer"], c["benchmark"], c["site"]): c for c in pool}
        groups = defaultdict(lambda: defaultdict(list))
        for c in pool:
            groups[(c["model"], c["layer"], c["benchmark"])][c["site"]].append(c)
        rb = mb = tie = 0
        npairs = 0
        for sites in groups.values():
            if "resid" not in sites or "mlp" not in sites:
                continue
            best = max(((_release_family(r["sae_repo"]) == _release_family(m2["sae_repo"]), r, m2)
                        for r in sites["resid"] for m2 in sites["mlp"]), key=lambda t: t[0])
            _, r, m2 = best
            npairs += 1
            dr, dm = r["full_exact_delta"], m2["full_exact_delta"]
            tie, rb, mb = (tie + 1, rb, mb) if dr == dm else (tie, rb + (dr > dm), mb + (dr < dm))
        add(f"- **RQ3** site — {npairs} release-matched pairs: {rb} resid larger, {mb} mlp larger, "
            f"{tie} tied; two-sided sign test p = {sign_test(rb, rb + mb):.3f}")

        sig_pool = [c for c in pool if c["full_exact_sig"]]
        ns_pool = [c for c in pool if not c["full_exact_sig"]]
        # The association the chapter reports (4.3.4): carrying at least one
        # significant transfer, among significant vs non-significant cells.
        a = sum(1 for c in sig_pool if c["transfer_n_significant"] >= 1)
        b = sum(1 for c in ns_pool if c["transfer_n_significant"] >= 1)
        add(f"- **RQ4** transfer — {m*3} measurements; a significant transfer under "
            f"{a}/{len(sig_pool)} significant cells against {b}/{len(ns_pool)} non-significant; "
            f"one-sided Fisher p = {fisher_one_sided(a, len(sig_pool), b, len(ns_pool)):.4g}")
        add(f"  - transfer measurements reaching significance: "
            f"{sum(c['transfer_n_significant'] for c in sig_pool)}/{3*len(sig_pool)} under significant "
            f"cells, {sum(c['transfer_n_significant'] for c in ns_pool)}/{3*len(ns_pool)} under "
            f"non-significant cells; median per-cell mean transfer delta "
            f"{pct(statistics.median([c['transfer_avg_delta'] for c in sig_pool]))}pp vs "
            f"{pct(statistics.median([c['transfer_avg_delta'] for c in ns_pool]))}pp")

        ga = [c for c in pool if c["best_ranking"] == "causal_delta"]
        gb = [c for c in pool if c["best_ranking"] == "freq_diff"]
        ka = sum(1 for c in ga if c["full_exact_sig"])
        kb = sum(1 for c in gb if c["full_exact_sig"])
        add(f"- **RQ5** ranking criterion — causal_delta {ka}/{len(ga)} significant "
            f"(mean {statistics.mean([c['full_exact_delta'] for c in ga])*100:+.3f}pp), "
            f"freq_diff {kb}/{len(gb)} (mean {statistics.mean([c['full_exact_delta'] for c in gb])*100:+.3f}pp); "
            f"one-sided Fisher p = {fisher_one_sided(ka, len(ga), kb, len(gb)):.4g}. "
            f"Associational: the grid chose the criterion.")
        opt_a = [c["selection_to_holdout"] for c in ga if c["selection_to_holdout"] is not None]
        opt_b = [c["selection_to_holdout"] for c in gb if c["selection_to_holdout"] is not None]
        add(f"  - holdout Δ median: causal_delta "
            f"{statistics.median([c['full_exact_delta'] for c in ga])*100:+.2f}pp vs freq_diff "
            f"{statistics.median([c['full_exact_delta'] for c in gb])*100:+.2f}pp; "
            f"selection-to-holdout optimism median {statistics.median(opt_a)*100:+.2f}pp vs "
            f"{statistics.median(opt_b)*100:+.2f}pp, one-sided Mann-Whitney "
            f"p = {mannwhitney_one_sided(opt_a, opt_b):.2g} (causal side loses less)")
        add("")

    add("## 9. Baselines and the difficulty band\n")
    add("| model | bench | baseline (full) | chance | in band | filtered holdout |")
    add("|---|---|---|---|---|---|")
    seen = set()
    for c in sorted(cells, key=lambda c: -c["base_acc_full"]):
        key = (c["model"], c["benchmark"])
        if key in seen:
            continue
        seen.add(key)
        add(f"| {c['model']} | {c['benchmark']} | {c['base_acc_full']*100:.1f}% | "
            f"{c['chance']*100:.0f}% | {c['n_filtered']}/{c['n_total']} "
            f"({c['band_share']*100:.0f}%) | {c['n_holdout_filtered']} |")
    add("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _decode(v):
    """Counter keys are hashable, so lists were stored JSON-encoded. Undo that."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


"""Column documentation. Exact names first, then patterns for the repetitive families.
Every emitted column must resolve to a description or the export fails, so a new field
cannot silently ship undocumented."""
COLDOC = {
    "exp": "experiment id, EXP-N",
    "model": "internal short name; NEVER use in thesis text, use model_id",
    "model_id": "HuggingFace identifier — the name the thesis uses",
    "params": "parameter count as a label (70M … 9B)",
    "d_model": "model width", "n_layers": "layer count",
    "sae_config": "SAE yaml stem", "sae_name": "SAE display name",
    "sae_repo": "HuggingFace repo of the SAE; no revision pin exists",
    "loader_activation": "activation path our loader was configured with (batchtopk | jump_relu | relu). NOT the architecture the release declares, which several disagree with. Appendix C carries the per-release architecture; do not aggregate this into a family split",
    "width": "d_sae", "expansion": "width / d_model", "top_k": "k, where the family defines one",
    "site": "resid | mlp", "layer": "intervention layer", "layer_frac": "layer / (n_layers - 1)",
    "benchmark": "CSQA | ARC (the primary benchmark of this cell)",
    "n_choices": "answer options", "chance": "1 / n_choices",
    "duration_s": "wall clock", "timestamp": "result file timestamp",
    "approx_repaired": "result was repaired in place by rerun_approx_fix.py",
    "n_total": "benchmark size after split combination",
    "n_filtered": "questions inside the difficulty band",
    "n_discovery": "Phase 1 subset", "n_validation": "Phase 2 subset",
    "n_reserve": "in-band remainder held out of feature work",
    "n_selection": "Phase 3 grid-evaluation subset", "n_holdout": "Phase 4 holdout",
    "n_holdout_filtered": "in-band part of the holdout",
    "band_share": "n_filtered / n_total", "p_min": "band lower bound", "p_max": "band upper bound",
    "base_acc_full": "baseline on the WHOLE HOLDOUT (not the whole benchmark)",
    "base_acc_filtered": "baseline on the in-band holdout",
    "below_chance": "base_acc_full < chance",
    "min_occurrences": "n_min admission threshold, max(3, floor(0.01 * n_discovery))",
    "n_discovery_success": "discovery questions answered correctly (parsed from the log)",
    "n_discovery_failure": "discovery questions answered incorrectly",
    "discovery_accuracy": "success / (success + failure) on the discovery subset",
    "phase2_tested": "candidates delivered by Phase 1 and screened by Phase 2 (<= 100)",
    "phase2_tested_pos": "positive-side candidates", "phase2_tested_neg": "negative-side candidates",
    "phase2_survived": "candidates with causal_delta > 0",
    "phase2_survived_pos": "positive-side survivors", "phase2_survived_neg": "negative-side survivors",
    "phase2_rejection_rate": "1 - survived / tested",
    "best_top_k_pos": "k+ of the winning configuration", "best_top_k_neg": "k- of the winner",
    "best_strength": "alpha of the winner", "best_ranking": "causal_delta | freq_diff",
    "best_n_features": "features actually used = min(k+, available+) + min(k-, available-)",
    "n_grid_combinations": "grid size, 90 throughout",
    "approx_gap_full": "approximated minus exact delta, full holdout",
    "approx_gap_filtered": "same, in-band holdout",
    "kfold_n": "folds", "kfold_mean": "mean per-fold delta", "kfold_std": "sd of per-fold deltas",
    "kfold_ci_lo": "Student-t 95% lower", "kfold_ci_hi": "Student-t 95% upper",
    "kfold_pos_folds": "folds with a positive delta",
    "kfold_ci_excludes_zero": "CI clear of zero",
    "selection_delta": "the winner's delta on the selection subset",
    "selection_to_holdout": "full_exact_delta - selection_delta (optimism)",
    "grid_delta_min": "smallest selection delta in the grid", "grid_delta_max": "largest",
    "grid_delta_median": "median", "grid_positive": "grid cells with a positive selection delta",
    "zerobase_n": "holdout questions below p_min, where baseline accuracy is 0 by construction",
    "zerobase_share": "zerobase_n / holdout", "zerobase_improved": "improvements from that band",
    "restricted_improved": "improvements with the zero-baseline band removed",
    "restricted_worsened": "worsenings with that band removed",
    "restricted_p": "one-sided binomial p on the restricted counts",
    "restricted_survives": "restricted result still significant and positive",
    "restricted_reverses": "restricted result flips sign",
    "holdout_above_n": "holdout questions above p_max",
    "holdout_inband_share": "in-band share of the holdout",
    "holdout_above_share": "above-p_max share of the holdout",
    "transfer_n_positive": "transfer benchmarks with a positive delta (of 3)",
    "transfer_n_significant": "transfer benchmarks reaching significance",
    "transfer_avg_delta": "mean delta over the three transfer benchmarks",
    "outcome": "significant | acted_ns | no_change | no_features",
    "screening_id": "partition fingerprint; two cells share a partition iff this matches",
    "screening_n_cells": "cells sharing that screening",
    "n_distinct_p": "distinct p_correct values in the screening",
    "modal_p_share": "share of the benchmark at the most common p_correct",
    "top_letter": "most frequently predicted answer letter",
    "top_letter_share": "share of questions receiving it",
    "degenerate": f"modal_p_share > {DEGENERATE_MODAL_SHARE}; the difficulty screen returns a "
                  "constant score, so Phase 0 has no ordering and no subset. Threshold is a "
                  "parameter, not a finding — recompute from modal_p_share to change it.",
    "representative": "the screening the band figure draws for a (model, benchmark) pair: "
                      "the reference experiment's own where the pair contains it, otherwise "
                      "the screening shared by most of the pair's cells",
    "n_cells": "cells using this screening",
    "bench_below_pmin_share": "share of the whole benchmark below p_min",
    "bench_inband_share": "share inside the band",
    "bench_above_pmax_share": "share above p_max",
}
_READING = {"n": "questions", "base": "baseline accuracy", "acc": "intervention accuracy",
            "delta": "intervention minus baseline", "improved": "wrong -> right (discordant)",
            "worsened": "right -> wrong (discordant)", "p": "one-sided binomial p (McNemar)",
            "sig": "p < 0.05"}
_SCOPE = {"flt_exact": "in-band holdout, exact intervention",
          "full_exact": "full holdout, exact intervention — THE PRIMARY ENDPOINT",
          "flt_approx": "in-band holdout, approximated intervention",
          "full_approx": "full holdout, approximated intervention"}
_BENCH = {"arc": "ARC-Challenge", "csqa": "CommonsenseQA", "hswag": "HellaSwag", "obqa": "OpenBookQA"}


def _describe(col: str) -> str | None:
    if col in COLDOC:
        return COLDOC[col]
    for scope, sd in _SCOPE.items():
        if col.startswith(scope + "_"):
            key = col[len(scope) + 1:]
            if key in _READING:
                return f"{_READING[key]} — {sd}"
    if col.startswith("tr_"):
        _, bench, key = col.split("_", 2)
        if bench in _BENCH and key in _READING:
            return f"{_READING[key]} — transfer to {_BENCH[bench]}"
    if col.startswith("rank_"):
        for crit in ("causal_delta", "freq_diff"):
            if col.startswith(f"rank_{crit}_"):
                what = {"best": "best", "avg": "mean", "positive": "count of positive"}
                return f"{what[col.rsplit('_', 1)[1]]} selection delta over the 45 {crit} grid cells"
    if re.fullmatch(r"d[1-9]0", col):
        return f"{col[1:]}th percentile of p_correct over the whole benchmark"
    return None


def write_readme(path: pathlib.Path, cells: list[dict], cell_fields: list[str],
                 scr_fields: list[str], head: str | None) -> None:
    # cell_fields is the UNION across rows, which is what cells.csv actually writes:
    # transfer columns differ per cell (a CSQA cell carries tr_arc_*, an ARC cell
    # tr_csqa_*), so documenting cells[0] alone silently skips a whole family.
    missing = [c for c in cell_fields + scr_fields if _describe(c) is None]
    if missing:
        raise SystemExit(f"undocumented columns, add them to COLDOC: {missing}")
    L = ["# thesis/data — what is in here\n",
         f"Generated by `scripts/export_thesis_data.py` from `results/` at git HEAD "
         f"`{head or 'unknown'}`. Do not hand-edit; re-run the script.\n",
         "| file | rows | what |", "|---|---|---|",
         f"| `cells.csv` | {len(cells)} | one row per completed cell, flat; read by Typst `csv()` |",
         f"| `cells.jsonl` | {len(cells)} | same cells plus nested `folds`, `bands`, `transfer` |",
         "| `screening.csv` | 36 | one row per distinct screening, not per pair |",
         "| `constants.json` | — | pipeline settings with the count of cells agreeing |",
         "| `canonical.json` | — | the reference experiment's result JSON, verbatim |",
         "| `RESULTS.md` | — | human-readable digest; § 8 carries the research-question tests |",
         "",
         "**Population.** Every cell with status `done`. No cell is selected on its outcome "
         "(outline § 10.6). Cells that never ran are absent by design.\n",
         "## cells.csv\n", "| column | meaning |", "|---|---|"]
    for c in cell_fields:
        L.append(f"| `{c}` | {_describe(c)} |")
    L += ["", "## screening.csv\n", "| column | meaning |", "|---|---|"]
    for c in scr_fields:
        L.append(f"| `{c}` | {_describe(c)} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _pnum(p: str | None) -> float:
    if not p:
        return 9e9
    v = float(p[:-1])
    return v / 1000 if p.endswith("M") else v


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--canonical", default=CANONICAL)
    args = ap.parse_args()

    if not SUMMARY.exists():
        print(f"missing {SUMMARY}; run scripts/summarize_final_matrix.py first", file=sys.stderr)
        return 2

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    cells = load_cells()
    if not cells:
        print("no completed cells found", file=sys.stderr)
        return 2

    extracted = [extract(c) for c in cells]
    flat = [e["flat"] for e in extracted]

    # Screening fingerprint. Screening is re-run per cell, and three (model, benchmark)
    # pairs were screened more than once with materially different results, which makes
    # their partitions non-identical. Fingerprinting keeps that machine-detectable so a
    # later comparison can check whether two cells share a partition instead of relying
    # on somebody remembering.
    prints: dict[tuple, str] = {}
    for r in flat:
        key = (r["model"], r["benchmark"], r["n_filtered"], round(r["base_acc_full"], 6))
        if key not in prints:
            h = hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:8]
            prints[key] = f"scr-{h}"
        r["screening_id"] = prints[key]
    shared = Counter(r["screening_id"] for r in flat)
    for r in flat:
        r["screening_n_cells"] = shared[r["screening_id"]]

    # One screening digest per distinct screening, not per cell: screening is
    # SAE-independent, so 36 reads cover 102 cells.
    digests: dict[str, dict] = {}
    for r in flat:
        if r["screening_id"] not in digests:
            digests[r["screening_id"]] = _screening_digest(r["exp"], r["p_min"], r["p_max"])
    for r in flat:
        d = digests[r["screening_id"]]
        for k in ("n_distinct_p", "modal_p_share", "top_letter", "top_letter_share"):
            r[k] = d.get(k)
        r["degenerate"] = (None if d.get("modal_p_share") is None
                           else d["modal_p_share"] > DEGENERATE_MODAL_SHARE)

    flat.sort(key=lambda r: (_pnum(r["params"]), r["model"], r["layer"], r["site"], r["benchmark"]))

    # cells.csv — the spine Typst reads. Header is the union across rows, because the
    # transfer columns depend on which benchmark a cell is primary for.
    cell_fields = list(flat[0].keys())
    for r in flat:
        for k in r:
            if k not in cell_fields:
                cell_fields.append(k)
    with (out / "cells.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cell_fields, restval="")
        w.writeheader()
        for r in flat:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    # cells.jsonl — the same cells with the nested blocks kept
    by_exp = {e["flat"]["exp"]: e for e in extracted}
    with (out / "cells.jsonl").open("w", encoding="utf-8") as fh:
        for r in flat:
            merged = dict(r)
            merged.update(by_exp[r["exp"]]["nested"])
            fh.write(json.dumps(merged, ensure_ascii=False) + "\n")

    # canonical.json — the reference experiment, verbatim
    canon = next((c for c in cells if c["exp"] == args.canonical), None)
    if canon is None:
        print(f"canonical cell {args.canonical} is not among the completed cells", file=sys.stderr)
        return 2
    shutil.copyfile(RESULTS / canon["exp"] / canon["result_json"], out / "canonical.json")

    # constants.json — settings, with the count of cells that agree
    probe = defaultdict(Counter)
    for c in cells:
        j = json.loads((RESULTS / c["exp"] / c["result_json"]).read_text(encoding="utf-8"))
        probe["difficulty_band"][json.dumps(j["data_stats"]["p_correct_range"])] += 1
        probe["causal_test_strength"][json.dumps(j["causal_config"]["causal_test_strength"])] += 1
        probe["candidates_pos"][j["causal_config"]["n_candidate_features_pos"]] += 1
        probe["candidates_neg"][j["causal_config"]["n_candidate_features_neg"]] += 1
        for key in ("top_k_positive", "top_k_negative", "strength",
                    "ranking_criterion", "weighting_mode"):
            probe["grid_" + key][json.dumps(j["grid_config"][key])] += 1
        probe["n_grid_combinations"][j["n_combinations"]] += 1
        probe["kfold_n"][j["kfold_results"]["n_folds"]] += 1
        probe["experiment_version"][j["experiment_version"]] += 1

    constants = {
        "n_cells": len(cells),
        "canonical_cell": args.canonical,
        "seed": 42,
        "batch_size": 1,
        "split_fractions": {"discovery": 0.50, "validation": 0.25, "reserve": 0.25,
                            "then_selection": 0.50, "then_holdout": 0.50},
        "uniform": {k: {"value": _decode(v), "cells_agreeing": n, "uniform": n == len(cells)}
                    for k, c in probe.items() for v, n in [c.most_common(1)[0]]},
        "degeneracy_threshold": DEGENERATE_MODAL_SHARE,
        "degeneracy_note": "A cell is flagged when one p_correct value covers more than this "
                           "share of the benchmark. Catches Pythia-70M only (8 cells); every "
                           "threshold from 0.60 to 0.99 gives the same partition. A parameter, "
                           "not a finding — recompute from screening.csv's modal_p_share.",
        "note": "seed, batch size and split fractions come from src/sae_experiments/config.py; "
                "every other entry is read back out of the result JSONs and counted.",
    }
    (out / "constants.json").write_text(
        json.dumps(constants, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # screening.csv — one row per DISTINCT screening, not per pair, so the three
    # repeated pairs keep both of theirs. `representative` marks the screening the band
    # figure draws for a pair: the reference experiment's own where the pair contains it,
    # so that the walk-through and the figure quote the same partition, and otherwise the
    # screening shared by most of the pair's cells.
    by_id: dict[str, dict] = {}
    for r in flat:
        by_id.setdefault(r["screening_id"], r)
    canonical_sid = next((r["screening_id"] for r in flat if r["exp"] == args.canonical), None)
    best_per_pair: dict[tuple[str, str], str] = {}
    for sid, r in by_id.items():
        pair = (r["model"], r["benchmark"])
        cur = best_per_pair.get(pair)
        if cur is None or sid == canonical_sid or (
            cur != canonical_sid
            and r["screening_n_cells"] > by_id[cur]["screening_n_cells"]
        ):
            best_per_pair[pair] = sid

    fields = ["screening_id", "representative", "n_cells", "model", "model_id", "benchmark",
              "n_choices", "chance", "n_total", "n_filtered", "band_share",
              "base_acc_full", "base_acc_filtered", "below_chance",
              "n_holdout", "n_holdout_filtered",
              "bench_below_pmin_share", "bench_inband_share", "bench_above_pmax_share",
              "n_distinct_p", "modal_p_share", "degenerate", "top_letter", "top_letter_share"]
    fields += [f"d{q*10}" for q in range(1, 10)]
    with (out / "screening.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for sid, r in sorted(by_id.items(), key=lambda kv: (_pnum(kv[1]["params"]),
                                                            kv[1]["model"], kv[1]["benchmark"])):
            row = {k: r.get(k, "") for k in fields}
            row["screening_id"] = sid
            row["representative"] = best_per_pair[(r["model"], r["benchmark"])] == sid
            row["n_cells"] = r["screening_n_cells"]
            row.update(digests[sid])
            row["degenerate"] = r["degenerate"]
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    write_digest(out / "RESULTS.md", flat, _git_head())
    write_readme(out / "README.md", flat, cell_fields, fields, _git_head())

    print(f"wrote {len(flat)} cells to {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<16} {f.stat().st_size:>9,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
