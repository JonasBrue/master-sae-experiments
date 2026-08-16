#!/usr/bin/env python3
"""Summarise every non-attention cell of the final matrix into one JSONL file
plus one aligned Markdown table.

Cell-driven: the 220 cells of run_experiments_final.py are the spine, so a cell
that has not run yet still gets a row with its full comparison axes. Attention
cells (hook_point: "attention") are excluded unless --include-attention.

Reads only; never writes inside results/EXP-*. Run it after `git pull` — the
header records the git HEAD and the newest result timestamp, so staleness shows.

Usage:
    uv run python scripts/summarize_final_matrix.py
    uv run python scripts/summarize_final_matrix.py --print
    uv run python scripts/summarize_final_matrix.py --out-dir /tmp
    uv run python scripts/summarize_final_matrix.py --include-attention
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

N_EXPECTED_TOTAL = 220
N_EXPECTED_ATTENTION = 30

_EXP_ID_RE = re.compile(r"EXP-\d+")
_PARAM_RE = re.compile(r"_(\d+)([mb])(?:_|$)")
# Names the regex cannot read: gpt2_small has no size token, and DeepSeek's
# "15b" means 1.5B.
_PARAM_OVERRIDES = {"gpt2_small": ("124M", 0.124), "deepseek_r1_qwen_15b": ("1.5B", 1.5)}
_SITE_BY_HOOK = {"resid_post": "resid", "mlp": "mlp", "attention": "attn"}
_BENCH_SHORT = {
    "commonsenseqa": "CSQA",
    "arc_challenge": "ARC",
    "hellaswag": "HellaSwag",
    "openbookqa": "OBQA",
}


class PopulationError(RuntimeError):
    """The matrix no longer has 220 cells / 30 attention cells."""


@dataclass
class Cell:
    idx: int          # position in EXPERIMENTS; `--only idx` runs this cell
    exp: str          # "EXP-300"
    sae: str          # "configs/saes/....yaml"
    layer: int
    benchmark: str    # "configs/benchmarks/....yaml"


# ── cells and axes ────────────────────────────────────────────────────────────


def load_cells(repo_root: Path = REPO_ROOT) -> list[Cell]:
    """Load the runner's EXPERIMENTS list as Cell records, 1-based idx."""
    path = repo_root / "run_experiments_final.py"
    spec = importlib.util.spec_from_file_location("run_experiments_final", path)
    module = importlib.util.module_from_spec(spec)
    # Registering the module is required: @dataclass resolves __module__ via sys.modules.
    sys.modules["run_experiments_final"] = module
    spec.loader.exec_module(module)
    return [
        Cell(idx=i, exp=_EXP_ID_RE.search(e.name).group(0), sae=e.sae,
             layer=e.layer, benchmark=e.benchmark)
        for i, e in enumerate(module.EXPERIMENTS, 1)
    ]


def parse_params(short_name: str) -> tuple[str | None, float]:
    """("pythia_160m_deduped") -> ("160M", 0.16). Sort key is billions of params."""
    if short_name in _PARAM_OVERRIDES:
        return _PARAM_OVERRIDES[short_name]
    m = _PARAM_RE.search(short_name)
    if not m:
        return None, float("inf")
    value, unit = int(m.group(1)), m.group(2)
    display = f"{value}{unit.upper()}"
    return display, value / 1000 if unit == "m" else float(value)


def axis_meta(cell: Cell, repo_root: Path = REPO_ROOT) -> dict:
    """Comparison axes for a cell, read from its SAE and model YAML.

    Never raises: an unreadable config yields None axes plus meta_error, so one
    broken config cannot hide the other 189 rows.
    """
    meta = {k: None for k in (
        "sae_config", "sae_name", "model", "model_id", "params", "arch", "width",
        "top_k", "site", "hook_point", "n_layers", "layer_frac", "benchmark",
        "benchmark_config", "meta_error",
    )}
    meta["sae_config"] = Path(cell.sae).stem
    meta["benchmark_config"] = Path(cell.benchmark).stem
    meta["benchmark"] = _BENCH_SHORT.get(meta["benchmark_config"], meta["benchmark_config"])
    meta["_params_sort"] = float("inf")
    try:
        sae_path = repo_root / cell.sae
        sae = yaml.safe_load(sae_path.read_text(encoding="utf-8"))
        meta["sae_name"] = sae.get("name")
        meta["arch"] = sae.get("architecture")
        meta["width"] = sae.get("d_hidden")
        meta["top_k"] = int(sae["top_k"]) if sae.get("top_k") else None
        meta["hook_point"] = sae.get("hook_point")
        meta["site"] = _SITE_BY_HOOK.get(sae.get("hook_point"))
        model_path = (sae_path.parent / sae["model"]).resolve()
        model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        meta["model"] = model.get("short_name")
        meta["model_id"] = model.get("model_id")
        meta["n_layers"] = model.get("n_layers")
        meta["params"], meta["_params_sort"] = parse_params(meta["model"] or "")
        if meta["n_layers"] and meta["n_layers"] > 1:
            meta["layer_frac"] = round(cell.layer / (meta["n_layers"] - 1), 3)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        meta["meta_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return meta


def split_cells(cells: list[Cell], repo_root: Path = REPO_ROOT):
    """Split into (reported, attention) as [(cell, meta), ...] pairs."""
    kept, attention = [], []
    for cell in cells:
        meta = axis_meta(cell, repo_root)
        (attention if meta["hook_point"] == "attention" else kept).append((cell, meta))
    return kept, attention


# ── status from the filesystem ────────────────────────────────────────────────

_JSON_STAMP_RE = re.compile(r"_(\d{8})_(\d{6})\.json$")
_FOOTER_RESULT_RE = re.compile(r"^Result:\s*(SUCCESS|FAILED)\s*$", re.M)
_FOOTER_ERROR_RE = re.compile(r"^Error:\s*(.+)$", re.M)
_FOOTER_DURATION_RE = re.compile(
    r"^Duration:\s*(?:(\d+)\s+days?,\s*)?(\d+):(\d\d):(\d\d)", re.M
)
_LOG_TAIL_BYTES = 8192  # the footer is the last few lines; logs run to megabytes


def _result_jsons(exp_dir: Path) -> list[Path]:
    return sorted(
        p for p in exp_dir.glob("*.json") if not p.name.startswith("logit_screening_")
    )


def _json_sort_key(path: Path) -> tuple:
    m = _JSON_STAMP_RE.search(path.name)
    # Stamped names sort by stamp; unstamped ones fall back to mtime and lose ties.
    return (1, m.group(1) + m.group(2)) if m else (0, f"{path.stat().st_mtime:020.0f}")


def _json_timestamp(path: Path) -> str | None:
    m = _JSON_STAMP_RE.search(path.name)
    if not m:
        return None
    d, t = m.group(1), m.group(2)
    return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"


def _read_log_tail(exp_dir: Path) -> str:
    logs = sorted(exp_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return ""
    try:
        with open(logs[-1], "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - _LOG_TAIL_BYTES))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def resolve_status(exp: str, results_dir: Path) -> dict:
    """Classify a cell from the filesystem. See spec §5 — first match wins."""
    info = {"status": "pending", "dir": None, "result_json": None, "timestamp": None,
            "duration_s": None, "n_result_files": 0, "error": None}
    exp_dir = results_dir / exp
    if exp_dir.is_dir():
        info["dir"] = f"results/{exp}"
        tail = _read_log_tail(exp_dir)
        m = _FOOTER_DURATION_RE.search(tail)
        if m:
            days = int(m.group(1) or 0)
            info["duration_s"] = (days * 86400 + int(m.group(2)) * 3600
                                  + int(m.group(3)) * 60 + int(m.group(4)))
        jsons = _result_jsons(exp_dir)
        info["n_result_files"] = len(jsons)
        if jsons:
            newest = max(jsons, key=_json_sort_key)
            info["status"] = "done"
            info["result_json"] = newest
            info["timestamp"] = _json_timestamp(newest)
            return info
        footer = _FOOTER_RESULT_RE.search(tail)
        if footer and footer.group(1) == "FAILED":
            info["status"] = "failed"
            err = _FOOTER_ERROR_RE.search(tail)
            info["error"] = err.group(1).strip()[:200] if err else None
        elif footer:
            info["status"] = "done_no_json"
        else:
            info["status"] = "running"
        return info

    for suffix, status in (("-ABORT", "aborted"), ("-FAILED", "failed")):
        sibling = results_dir / f"{exp}{suffix}"
        if sibling.is_dir():
            info["status"] = status
            info["dir"] = f"results/{sibling.name}"
            return info
    archived = sorted(results_dir.glob(f"{exp}_archived_*"))
    if archived:
        info["status"] = "archived_only"
        info["dir"] = f"results/{archived[-1].name}"
    return info


# ── metrics from a result JSON ────────────────────────────────────────────────

METRIC_KEYS = [
    "n_total", "n_filtered", "n_holdout", "n_holdout_filtered",
    "base_acc_full", "base_acc_filtered",
    # Primary: the full holdout. Kept ahead of the filtered variants so the JSONL
    # reads in the same priority order as the table.
    "full_exact_delta", "full_exact_p", "full_exact_sig", "full_exact_n",
    "full_approx_delta", "full_approx_p", "full_approx_sig",
    # Secondary: the filtered holdout.
    "flt_exact_delta", "flt_exact_p", "flt_exact_sig", "flt_exact_n",
    "flt_approx_delta", "flt_approx_p", "flt_approx_sig",
    "approx_gap_full", "approx_gap_filtered",
    "kfold_mean", "kfold_std", "kfold_pos_folds", "kfold_p",
    "gen_avg_delta", "gen_n_positive", "gen_n_significant", "gen_n_benchmarks",
    "best_top_k_pos", "best_top_k_neg", "best_strength", "best_ranking",
    "best_weighting", "best_n_features",
]

# Benchmark display names as the result JSON writes them, per benchmark config stem.
_BENCH_JSON_NAME = {
    "commonsenseqa": "CommonsenseQA",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "openbookqa": "OpenBookQA",
}


def _report(full_reporting: dict, key: str) -> dict:
    """full_reporting keys carry an optional _static/_adaptive suffix."""
    if key in full_reporting:
        return full_reporting[key]
    for suffix in ("_static", "_adaptive"):
        if key + suffix in full_reporting:
            return full_reporting[key + suffix]
    return {}


def metrics_from_result(data: dict) -> dict:
    """Flatten one result JSON into METRIC_KEYS. Missing pieces stay None."""
    out = {k: None for k in METRIC_KEYS}
    if not data:
        return out
    stats = data.get("data_stats", {})
    out["n_total"] = stats.get("total_questions")
    out["n_filtered"] = stats.get("filtered_questions")
    out["n_holdout"] = stats.get("holdout_questions")
    out["n_holdout_filtered"] = stats.get("holdout_filtered_questions")
    out["base_acc_full"] = stats.get("baseline_full_accuracy")
    out["base_acc_filtered"] = stats.get("baseline_filtered_accuracy")

    fr = data.get("full_reporting", {})
    for prefix, key in (("full_exact", "full_exact"), ("full_approx", "full_approx"),
                        ("flt_exact", "filtered_exact"), ("flt_approx", "filtered_approx")):
        r = _report(fr, key)
        out[f"{prefix}_delta"] = r.get("delta")
        out[f"{prefix}_p"] = r.get("binomial_p")
        out[f"{prefix}_sig"] = r.get("significant")
        if f"{prefix}_n" in out:
            out[f"{prefix}_n"] = r.get("n_questions")

    approx = data.get("approximation_comparison", {})
    out["approx_gap_filtered"] = approx.get("filtered", {}).get("difference")
    out["approx_gap_full"] = approx.get("full", {}).get("difference")

    kf = data.get("kfold_results", {})
    out["kfold_mean"] = kf.get("mean_delta")
    out["kfold_std"] = kf.get("std_delta")
    out["kfold_p"] = kf.get("binomial_p")
    if kf.get("n_folds"):
        out["kfold_pos_folds"] = f"{kf.get('n_positive_folds', 0)}/{kf['n_folds']}"

    gen = data.get("generalisation", {}).get("summary", {})
    out["gen_n_benchmarks"] = gen.get("n_benchmarks")
    out["gen_n_positive"] = gen.get("n_positive")
    out["gen_n_significant"] = gen.get("n_significant")
    if gen.get("n_benchmarks"):
        out["gen_avg_delta"] = gen.get("avg_delta")

    best = data.get("best_config", {})
    out["best_top_k_pos"] = best.get("top_k_pos")
    out["best_top_k_neg"] = best.get("top_k_neg")
    out["best_strength"] = best.get("strength")
    out["best_ranking"] = best.get("ranking_criterion")
    out["best_weighting"] = best.get("weighting_mode")
    out["best_n_features"] = best.get("n_features_used")
    return out


def detect_mismatch(data: dict, cell: Cell, meta: dict) -> bool:
    """True when the result JSON describes a different configuration than the cell.

    Such a folder holds output from another run and its numbers must not be used.
    """
    layer = data.get("sae", {}).get("layer", data.get("intervention_layer"))
    if layer is not None and int(layer) != cell.layer:
        return True
    expected = _BENCH_JSON_NAME.get(meta["benchmark_config"])
    found = data.get("benchmark", {}).get("name")
    return bool(expected and found and expected != found)


# ── rows ─────────────────────────────────────────────────────────────────────

AXIS_KEYS = [
    "exp", "idx", "sae_config", "sae_name", "model", "model_id", "params", "arch",
    "width", "top_k", "site", "hook_point", "layer", "n_layers", "layer_frac",
    "benchmark", "benchmark_config",
]
STATUS_KEYS = [
    "status", "dir", "result_json", "timestamp", "duration_s", "n_result_files",
    "mismatch", "error", "meta_error",
]
ROW_KEYS = AXIS_KEYS + STATUS_KEYS + METRIC_KEYS

_SITE_ORDER = {"resid": 0, "mlp": 1, "attn": 2}
_BENCH_ORDER = {"CSQA": 0, "ARC": 1}
_STATUS_ABBREV = {
    "done": "done", "running": "run", "failed": "fail", "aborted": "abrt",
    "pending": "pend", "archived_only": "arcv", "done_no_json": "?json",
    "done_bad_json": "?bad",
}


def build_rows(repo_root: Path = REPO_ROOT, results_dir: Path | None = None,
               include_attention: bool = False) -> list[dict]:
    """One row per reported cell, EXP-id ascending, uniform ROW_KEYS."""
    results_dir = Path(results_dir) if results_dir else repo_root / "results"
    cells = load_cells(repo_root)
    kept, attention = split_cells(cells, repo_root)
    expected = N_EXPECTED_TOTAL if include_attention else N_EXPECTED_TOTAL - N_EXPECTED_ATTENTION
    if len(cells) != N_EXPECTED_TOTAL or len(attention) != N_EXPECTED_ATTENTION:
        raise PopulationError(
            f"expected {N_EXPECTED_TOTAL} cells with {N_EXPECTED_ATTENTION} attention, "
            f"found {len(cells)} with {len(attention)}: "
            f"{sorted(c.exp for c, _ in attention)}"
        )
    pairs = kept + attention if include_attention else kept

    rows = []
    for cell, meta in pairs:
        row = {k: None for k in ROW_KEYS}
        row["exp"], row["idx"], row["layer"] = cell.exp, cell.idx, cell.layer
        for k in AXIS_KEYS:
            if k in meta:
                row[k] = meta[k]
        info = resolve_status(cell.exp, results_dir)
        row["status"] = info["status"]
        row["dir"] = info["dir"]
        row["timestamp"] = info["timestamp"]
        row["duration_s"] = info["duration_s"]
        row["n_result_files"] = info["n_result_files"]
        row["error"] = info["error"]
        row["meta_error"] = meta["meta_error"]
        if info["result_json"] is not None:
            row["result_json"] = info["result_json"].name
            try:
                data = json.loads(info["result_json"].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                row["status"] = "done_bad_json"
                row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            else:
                row.update(metrics_from_result(data))
                row["mismatch"] = detect_mismatch(data, cell, meta)
                row["timestamp"] = row["timestamp"] or data.get("timestamp")
        row["_params_sort"] = meta.get("_params_sort", float("inf"))
        rows.append(row)

    rows.sort(key=lambda r: r["exp"])
    assert len(rows) == expected, f"{len(rows)} rows, expected {expected}"
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps({k: row[k] for k in ROW_KEYS}, ensure_ascii=False) + "\n")


# ── table ────────────────────────────────────────────────────────────────────


def _pct(v) -> str:
    return "-" if v is None else f"{v * 100:+.2f}%"


def _p3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _sig(v) -> str:
    return "-" if v is None else ("*" if v else "")


def _num(v) -> str:
    return "-" if v is None else str(v)


_COLUMNS = [
    ("exp", lambda r: r["exp"].replace("EXP-", ""), ">"),
    ("idx", lambda r: _num(r["idx"]), ">"),
    ("model", lambda r: _num(r["model"]), "<"),
    ("prm", lambda r: _num(r["params"]), ">"),
    ("L", lambda r: _num(r["layer"]), ">"),
    ("l/N", lambda r: "-" if r["layer_frac"] is None else f"{r['layer_frac']:.2f}", ">"),
    ("site", lambda r: _num(r["site"]), "<"),
    ("arch", lambda r: _num(r["arch"]), "<"),
    ("wid", lambda r: _num(r["width"]), ">"),
    ("K", lambda r: _num(r["top_k"]), ">"),
    ("bench", lambda r: _num(r["benchmark"]), "<"),
    ("status", lambda r: _STATUS_ABBREV.get(r["status"], r["status"])
                         + ("!" if r["mismatch"] else ""), "<"),
    # Primary result: the full holdout. Its N leads, and the unqualified `p`/`s`
    # columns belong to it — nothing else in the table carries a bare p-value.
    ("nHold", lambda r: _num(r["n_holdout"]), ">"),
    ("fullEx", lambda r: _pct(r["full_exact_delta"]), ">"),
    ("p", lambda r: _p3(r["full_exact_p"]), ">"),
    ("s", lambda r: _sig(r["full_exact_sig"]), "<"),
    ("fullAp", lambda r: _pct(r["full_approx_delta"]), ">"),
    # Secondary: the filtered holdout. Its N is often small enough that one question
    # moves the delta by whole percentage points, so it is prefixed throughout.
    ("nFlt", lambda r: _num(r["n_filtered"]), ">"),
    ("nFltH", lambda r: _num(r["n_holdout_filtered"]), ">"),
    ("fltEx", lambda r: _pct(r["flt_exact_delta"]), ">"),
    ("fltP", lambda r: _p3(r["flt_exact_p"]), ">"),
    ("fltS", lambda r: _sig(r["flt_exact_sig"]), "<"),
    ("fltAp", lambda r: _pct(r["flt_approx_delta"]), ">"),
    ("kfold", lambda r: _pct(r["kfold_mean"]), ">"),
    ("+f", lambda r: _num(r["kfold_pos_folds"]), ">"),
    ("gen", lambda r: _pct(r["gen_avg_delta"]), ">"),
    # Wall-clock the cell took, from the runner's log footer. Last column so the
    # cost of a row reads next to what the row bought.
    ("hrs", lambda r: "-" if r["duration_s"] is None else f"{r['duration_s'] / 3600:.1f}", ">"),
]


def total_hours(rows: list[dict]) -> float:
    """Summed wall-clock over the cells whose duration is known, in hours."""
    return round(sum(r["duration_s"] for r in rows if r["duration_s"]) / 3600, 1)


_TABLE_ROW_RE = re.compile(r"^\s*(\d+)\s")
_NOTE_RE = re.compile(r"\s(\(.*\S)\s*$")


def extract_notes(markdown: str) -> dict[str, str]:
    """Hand-written per-row annotations read back out of a previous table.

    The whole file is regenerated on every run, so anything typed into it by hand
    would otherwise be lost. A note is recognised by its leading parenthesis: no
    generated column ever contains one, which makes the split unambiguous and lets
    the annotations stay in the file they annotate instead of a sidecar that has to
    be kept in step. Unparseable lines are skipped rather than guessed at.
    """
    notes: dict[str, str] = {}
    for line in markdown.splitlines():
        row = _TABLE_ROW_RE.match(line)
        note = _NOTE_RE.search(line) if row else None
        if note:
            notes[f"EXP-{row.group(1)}"] = note.group(1)
    return notes


def render_table(rows: list[dict], notes: dict[str, str] | None = None) -> str:
    """Aligned fixed-width table; blank line between model blocks.

    ``notes`` are appended verbatim after the last column and take no part in the
    width calculation, so a long annotation cannot push the numeric columns apart.
    """
    notes = notes or {}
    ordered = sorted(rows, key=lambda r: (
        r.get("_params_sort", float("inf")), r["model"] or "", r["layer"],
        _SITE_ORDER.get(r["site"], 9), _BENCH_ORDER.get(r["benchmark"], 9), r["exp"],
    ))
    cells = [[fmt(r) for _h, fmt, _a in _COLUMNS] for r in ordered]
    widths = [max([len(h)] + [c[i] and len(c[i]) or 0 for c in cells])
              for i, (h, _f, _a) in enumerate(_COLUMNS)]
    lines = ["  ".join(f"{h:{a}{w}}" for (h, _f, a), w in zip(_COLUMNS, widths)).rstrip()]
    prev_model = None
    for row, formatted in zip(ordered, cells):
        if prev_model is not None and row["model"] != prev_model:
            lines.append("")
        prev_model = row["model"]
        line = "  ".join(f"{v:{a}{w}}" for v, (_h, _f, a), w
                         in zip(formatted, _COLUMNS, widths)).rstrip()
        note = notes.get(row["exp"])
        lines.append(f"{line} {note}" if note else line)
    return "\n".join(lines)


def git_head(repo_root: Path) -> tuple[str | None, str | None]:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=10)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root,
                                capture_output=True, text=True, timeout=10)
        if sha.returncode or branch.returncode:
            return None, None
        return sha.stdout.strip(), branch.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, None


def render_markdown(rows: list[dict], repo_root: Path, generated: str | None = None,
                    notes: dict[str, str] | None = None) -> str:
    n_attn = sum(1 for r in rows if r["site"] == "attn")
    title = (f"# Final matrix — {len(rows)} cells ({len(rows) - n_attn} non-attention"
             f" + {n_attn} attention)" if n_attn else
             f"# Final matrix — {len(rows)} non-attention cells")
    sha, branch = git_head(repo_root)
    stamps = [r["timestamp"] for r in rows if r["timestamp"]]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    order = ["done", "running", "failed", "aborted", "pending", "archived_only",
             "done_no_json", "done_bad_json"]
    count_line = " · ".join(f"{s} {counts[s]}" for s in order if s in counts)
    n_timed = sum(1 for r in rows if r["duration_s"])
    count_line += (f" · {total_hours(rows)} h wall-clock over {n_timed} timed cell(s)"
                   if n_timed else "")
    generated = generated or datetime.now().strftime("%Y-%m-%d %H:%M")
    head = f"git HEAD {sha} ({branch})" if sha else "git HEAD unknown"
    newest = max(stamps) if stamps else "none"
    n_mismatch = sum(1 for r in rows if r["mismatch"])
    parts = [
        title, "",
        f"generated {generated} · {head} · newest result {newest}", "",
        count_line, "",
        "```", render_table(rows, notes), "```", "",
        "**The primary result is the full holdout.** `nHold` = full-holdout questions · "
        "`fullEx`/`p`/`s` = full-exact delta, binomial p, significance · "
        "`fullAp` = full-approximated delta · "
        "Secondary, on the much smaller filtered holdout: `nFlt` = filtered questions · "
        "`nFltH` = filtered-holdout questions · "
        "`fltEx`/`fltP`/`fltS` = filtered-exact delta, binomial p, significance · "
        "`fltAp` = filtered-approximated delta · "
        "`kfold`/`+f` = k-fold mean delta and positive folds · "
        "`gen` = mean delta over the three generalisation benchmarks · "
        "`hrs` = wall-clock the cell took, from its log footer · "
        "`!` after a status = result JSON disagrees with the cell definition.",
        "",
        "Anything in parentheses after the last column is a hand-written note. Edit those "
        "in place — this script reads them back out of the previous table and re-emits "
        "them, so regenerating never loses them. Every other character is generated.",
        "",
        f"Machine-readable form: `final_matrix_summary.jsonl` ({len(rows)} lines, "
        f"one JSON object per cell, EXP-id order).",
    ]
    if n_mismatch:
        parts += ["", f"**{n_mismatch} cell(s) flagged `!` — their numbers must not be used.**"]
    return "\n".join(parts) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None,
                    help="where to write the two files (default: <repo>/results)")
    ap.add_argument("--results-dir", default=None,
                    help="experiment folders to read (default: <repo>/results)")
    ap.add_argument("--include-attention", action="store_true",
                    help="also report the 30 attention cells")
    ap.add_argument("--print", action="store_true", dest="do_print",
                    help="also print the table to stdout")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "results"
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results"
    try:
        rows = build_rows(REPO_ROOT, results_dir, include_attention=args.include_attention)
    except PopulationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    jsonl_path = out_dir / "final_matrix_summary.jsonl"
    md_path = out_dir / "final_matrix_summary.md"
    write_jsonl(rows, jsonl_path)
    # Read the hand-written notes back before the file is overwritten. Done here
    # rather than inside render_markdown so that rendering stays a pure function of
    # its arguments and --out-dir cannot silently pick up another run's notes.
    notes = {}
    if md_path.exists():
        try:
            notes = extract_notes(md_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"WARNING: could not read existing notes: {exc}", file=sys.stderr)
    md = render_markdown(rows, REPO_ROOT, notes=notes)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8", newline="\n")

    if args.do_print:
        print(render_table(rows))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    # Console output stays ASCII: the Windows console is cp1252 and raises
    # UnicodeEncodeError on the arrows and middots used inside the Markdown file.
    print(f"{len(rows)} cells -> {jsonl_path}")
    print(f"{len(rows)} cells -> {md_path}")
    print("  " + " | ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    broken = [r["exp"] for r in rows if r["meta_error"]]
    if broken:
        print(f"WARNING: unreadable config for {len(broken)} cell(s): "
              f"{', '.join(broken[:10])}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
