"""Dump the dashboard overview table to stdout for quick inspection.

Usage:
    uv run python dump_overview.py              # full table
    uv run python dump_overview.py --csv        # CSV to stdout
    uv run python dump_overview.py --exp EXP-042  # single experiment detail
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _extract_row(d: dict) -> dict:
    if d.get("_failed"):
        return {
            "exp_id": d.get("_exp_id", ""),
            "failed": True,
            "model": "", "sae_name": "", "architecture": "",
            "layer": 0, "top_k": None, "benchmark": "", "n_questions": 0,
            "full_exact_delta": None, "filtered_exact_delta": None,
            "full_approx_delta": None, "filtered_approx_delta": None,
            "kfold_mean_delta": None, "kfold_pos_folds": "",
            "gen_avg_delta": None,
        }

    model = d.get("model", {})
    sae = d.get("sae", {})
    bench = d.get("benchmark", {})
    stats = d.get("data_stats", {})
    kfold = d.get("kfold_results", {})
    gen = d.get("generalisation", {})

    fr = d.get("full_reporting", {})

    def _get_report(key):
        if key in fr:
            return fr[key]
        for suffix in ("_static", "_adaptive"):
            if key + suffix in fr:
                return fr[key + suffix]
        return {}

    full_exact = _get_report("full_exact")
    filtered_exact = _get_report("filtered_exact")
    full_approx = _get_report("full_approx")
    filtered_approx = _get_report("filtered_approx")

    gen_summary = gen.get("summary", {}) if isinstance(gen, dict) else {}

    pos_folds = kfold.get("n_positive_folds", 0)
    n_folds = kfold.get("n_folds", 0)

    return {
        "exp_id": d.get("_exp_id", ""),
        "failed": False,
        "model": model.get("short_name", ""),
        "sae_name": sae.get("name", ""),
        "architecture": sae.get("architecture", ""),
        "layer": sae.get("layer", d.get("intervention_layer", 0)),
        "top_k": int(sae["top_k"]) if sae.get("top_k") else None,
        "benchmark": bench.get("name", ""),
        "n_questions": stats.get("total_questions", 0),
        "full_exact_delta": full_exact.get("delta"),
        "filtered_exact_delta": filtered_exact.get("delta"),
        "full_approx_delta": full_approx.get("delta"),
        "filtered_approx_delta": filtered_approx.get("delta"),
        "kfold_mean_delta": kfold.get("mean_delta"),
        "kfold_pos_folds": f"{pos_folds}/{n_folds}" if n_folds else "",
        "gen_avg_delta": gen_summary.get("avg_delta") if gen_summary.get("n_benchmarks", 0) > 0 else None,
    }


def load_all() -> list[dict]:
    records = []
    for exp_dir in sorted(RESULTS_DIR.glob("EXP-*")):
        if not exp_dir.is_dir():
            continue
        exp_id = exp_dir.name
        json_files = [
            jf for jf in exp_dir.glob("*.json")
            if not jf.name.startswith("logit_screening_")
        ]
        if json_files:
            for jf in json_files:
                data = _load_json(jf)
                if data:
                    data["_exp_id"] = exp_id
                    data["_source"] = str(jf)
                    records.append(data)
        else:
            log_files = list(exp_dir.glob("*.log"))
            records.append({
                "_exp_id": exp_id,
                "_source": str(log_files[0]) if log_files else str(exp_dir),
                "_failed": True,
            })
    return [_extract_row(d) for d in records]


def fmt_delta(val) -> str:
    if val is None:
        return "   -   "
    return f"{val * 100:+6.2f}%"


def print_table(rows: list[dict]) -> None:
    # Header
    print(f"{'ID':<10} {'Model':<12} {'SAE':<28} {'L':>2} {'K':>4} {'Bench':<18} "
          f"{'N':>4} {'Full+Ex':>8} {'Flt+Ex':>8} {'Full+Ap':>8} {'Flt+Ap':>8} "
          f"{'K-Fold':>8} {'Folds':>5} {'GenAvg':>8}")
    print("-" * 155)

    for r in rows:
        if r["failed"]:
            print(f"{r['exp_id']:<10} {'FAILED':<12}")
            continue
        k = str(r["top_k"]) if r["top_k"] is not None else "-"
        print(f"{r['exp_id']:<10} {r['model']:<12} {r['sae_name']:<28} {r['layer']:>2} {k:>4} "
              f"{r['benchmark']:<18} {r['n_questions']:>4} "
              f"{fmt_delta(r['full_exact_delta']):>8} {fmt_delta(r['filtered_exact_delta']):>8} "
              f"{fmt_delta(r['full_approx_delta']):>8} {fmt_delta(r['filtered_approx_delta']):>8} "
              f"{fmt_delta(r['kfold_mean_delta']):>8} {r['kfold_pos_folds']:>5} "
              f"{fmt_delta(r['gen_avg_delta']):>8}")


def print_detail(exp_id: str) -> None:
    exp_dir = RESULTS_DIR / exp_id
    if not exp_dir.is_dir():
        print(f"Not found: {exp_id}")
        return

    json_files = [jf for jf in exp_dir.glob("*.json") if not jf.name.startswith("logit_screening_")]
    if not json_files:
        print(f"{exp_id}: no JSON (failed experiment)")
        log_files = list(exp_dir.glob("*.log"))
        if log_files:
            print(f"Log: {log_files[0]}")
        return

    for jf in json_files:
        data = _load_json(jf)
        if not data:
            continue

        print(f"=== {exp_id} ({jf.name}) ===\n")

        # Config
        best = data.get("best_config", {})
        if best:
            print("Best config:")
            for k, v in best.items():
                print(f"  {k}: {v}")
            print()

        # Full reporting
        fr = data.get("full_reporting", {})
        if fr:
            print("Full reporting:")
            for key, r in fr.items():
                delta = r.get("delta")
                p = r.get("binomial_p")
                sig = r.get("significant", False)
                n = r.get("n_questions", 0)
                bl = r.get("baseline_acc", 0)
                iv = r.get("intervention_acc", 0)
                p_str = f"p={p:.4f}" if p is not None else "p=  N/A"
                print(f"  {r.get('name', key):<30} n={n:>4}  "
                      f"base={bl*100:5.1f}%  int={iv*100:5.1f}%  "
                      f"delta={fmt_delta(delta)}  {p_str}  {'SIG' if sig else '   '}")
            print()

        # K-fold
        kfold = data.get("kfold_results", {})
        if kfold:
            print(f"K-fold: mean={fmt_delta(kfold.get('mean_delta'))}  "
                  f"std={fmt_delta(kfold.get('std_delta'))}  "
                  f"+folds={kfold.get('n_positive_folds', 0)}/{kfold.get('n_folds', 0)}")
            print()

        # Generalisation
        gen = data.get("generalisation", {})
        if gen and "benchmarks" in gen:
            print("Generalisation:")
            for b in gen["benchmarks"]:
                print(f"  {b['benchmark']:<25} delta={fmt_delta(b['delta'])}  "
                      f"p={b.get('binomial_p', 0):.4f}  "
                      f"{'SIG' if b.get('significant') else '   '}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Dump SAE experiment overview")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")
    parser.add_argument("--exp", type=str, help="Show detail for specific experiment (e.g. EXP-042)")
    args = parser.parse_args()

    if args.exp:
        print_detail(args.exp)
        return

    rows = load_all()

    if args.csv:
        import csv
        import sys
        writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    else:
        print_table(rows)
        print(f"\n{len(rows)} experiments total, "
              f"{sum(1 for r in rows if not r['failed'])} successful, "
              f"{sum(1 for r in rows if r['failed'])} failed")


if __name__ == "__main__":
    main()
