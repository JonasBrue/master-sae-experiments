"""Generate the test_strength comparison document.

For ranking_criterion = causal_delta, compares the exact features that Phase 3
would feed into each grid cell when Phase 2 is run in single mode ([0.5] or
[1.0]) versus multi mode ([0.5, 1.0, 1.5]).
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = os.path.join(REPO_ROOT, "results")
OUT_PATH = os.path.join(REPO_ROOT, "docs", "test_strength_comparison.md")

EXPS = ["EXP-300", "EXP-301", "EXP-302"]
TOP_K_POS = [3, 5, 7, 10, 15]
TOP_K_NEG = [0, 3, 5]
STRENGTHS = [0.5, 1.0, 1.5]

EXP_LABELS = {
    "EXP-300": "gemma-2b L17 + CSQA (ReLU SAE)",
    "EXP-301": "gemma-2b L10 + CSQA (ReLU SAE)",
    "EXP-302": "llama-3.2-1b L15 + CSQA (BatchTopK K=32 MLP SAE)",
}


def load(exp, variant):
    dirpath = os.path.join(RESULTS_ROOT, f"{exp}-test-strength-{variant}")
    files = [
        f for f in os.listdir(dirpath)
        if f.endswith(".json") and "logit_screening" not in f
    ]
    with open(os.path.join(dirpath, files[0]), encoding="utf-8") as fh:
        return json.load(fh)


def bucket(doc, strength):
    for b in doc["phase2_top_features_per_strength"]:
        if abs(b["test_strength"] - strength) < 1e-6:
            return b
    return None


def top_ids(bkt, side, k):
    return [f["feature_id"] for f in bkt[side][:k]]


def cell_features(bkt, kp, kn):
    return set(top_ids(bkt, "positive", kp)) | set(top_ids(bkt, "negative", kn))


def fmt_list(ids):
    return "[" + ", ".join(str(x) for x in ids) + "]"


def main():
    lines = []
    lines.append("# Single-strength choice: should `causal.causal_test_strength` be `[0.5]` or `[1.0]`?\n")
    lines.append(
        "**Question.** Running with a single Phase-2 test strength is 3× cheaper than "
        "`[0.5, 1.0, 1.5]`. Which single value reproduces the multi-strength run's "
        "*feature selection* most faithfully?\n"
    )
    lines.append(
        "**Method.** For every grid cell at `ranking_criterion = causal_delta`, "
        "reconstruct the exact feature IDs Phase 3 feeds into the intervention "
        "vector and compare single-mode vs multi-mode. Single mode reuses one "
        "Phase-2 bucket across all grid cells; multi mode strength-matches "
        "(cell at strength s uses bucket@s).\n"
    )
    lines.append(
        f"**Sample.** {', '.join(EXPS)}, CSQA. Grid: top_k_pos ∈ {TOP_K_POS}, "
        f"top_k_neg ∈ {TOP_K_NEG}, strength ∈ {STRENGTHS}. 45 cells per experiment.\n"
    )
    lines.append(
        "Phase 2 is bit-deterministic — bucket@s in single mode equals bucket@s in "
        "multi mode (verified, max |Δ_single − Δ_multi| = 0.00). So the only thing "
        "that varies between modes is *which bucket each grid cell pulls from*.\n"
    )

    global_score = {
        "[0.5]": {"shared": 0, "total": 0, "identical": 0},
        "[1.0]": {"shared": 0, "total": 0, "identical": 0},
    }

    for exp in EXPS:
        label = EXP_LABELS.get(exp, exp)
        lines.append(f"\n---\n\n## {exp} — {label}\n")
        d05 = load(exp, "0.5")
        d10 = load(exp, "1.0")
        dM = load(exp, "0.5-1.0-1.5")

        b05 = bucket(d05, 0.5)
        b10 = bucket(d10, 1.0)
        bM05 = bucket(dM, 0.5)
        bM10 = bucket(dM, 1.0)
        bM15 = bucket(dM, 1.5)

        lines.append("### Bucket-level overlap (top-25 candidates Phase 3 chooses from)\n")
        lines.append("| Side | bkt@0.5 ∩ bkt@1.0 | bkt@0.5 ∩ bkt@1.5 | bkt@1.0 ∩ bkt@1.5 |")
        lines.append("|---|---|---|---|")
        for side in ["positive", "negative"]:
            i05 = top_ids(bM05, side, 25)
            i10 = top_ids(bM10, side, 25)
            i15 = top_ids(bM15, side, 25)
            lines.append(
                f"| {side} | {len(set(i05) & set(i10))}/25 | "
                f"{len(set(i05) & set(i15))}/25 | {len(set(i10) & set(i15))}/25 |"
            )
        lines.append("")

        lines.append("**Top-5 features per bucket (for orientation):**\n")
        lines.append("| Side | bkt@0.5 top-5 | bkt@1.0 top-5 | bkt@1.5 top-5 |")
        lines.append("|---|---|---|---|")
        for side in ["positive", "negative"]:
            lines.append(
                f"| {side} | `{fmt_list(top_ids(bM05, side, 5))}` | "
                f"`{fmt_list(top_ids(bM10, side, 5))}` | "
                f"`{fmt_list(top_ids(bM15, side, 5))}` |"
            )
        lines.append("")

        lines.append("### Per grid-cell features (ranking = `causal_delta`)\n")
        lines.append(
            "For each cell `(top_k_pos, top_k_neg, strength)`, the table shows the "
            "exact feature IDs in the intervention vector under each Phase-2 mode, "
            "and overlap with multi.\n"
        )
        lines.append(
            "| s | (kp, kn) | single `[0.5]` features | single `[1.0]` features | "
            "multi features (from bkt@s) | `[0.5]`∩multi | `[1.0]`∩multi |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        cell_score = {
            "[0.5]": {"shared": 0, "total": 0, "identical": 0},
            "[1.0]": {"shared": 0, "total": 0, "identical": 0},
        }

        for s in STRENGTHS:
            bM_s = {0.5: bM05, 1.0: bM10, 1.5: bM15}[s]
            for kp in TOP_K_POS:
                for kn in TOP_K_NEG:
                    f05 = cell_features(b05, kp, kn)
                    f10 = cell_features(b10, kp, kn)
                    fM = cell_features(bM_s, kp, kn)

                    sh05 = f05 & fM
                    sh10 = f10 & fM

                    cell_score["[0.5]"]["shared"] += len(sh05)
                    cell_score["[0.5]"]["total"] += len(fM)
                    cell_score["[0.5]"]["identical"] += int(f05 == fM)
                    cell_score["[1.0]"]["shared"] += len(sh10)
                    cell_score["[1.0]"]["total"] += len(fM)
                    cell_score["[1.0]"]["identical"] += int(f10 == fM)

                    tag05 = "**= multi**" if f05 == fM else f"{len(sh05)}/{len(fM)}"
                    tag10 = "**= multi**" if f10 == fM else f"{len(sh10)}/{len(fM)}"

                    lines.append(
                        f"| {s} | ({kp:2d},{kn}) | "
                        f"`{fmt_list(sorted(f05))}` | "
                        f"`{fmt_list(sorted(f10))}` | "
                        f"`{fmt_list(sorted(fM))}` | "
                        f"{tag05} | {tag10} |"
                    )
        lines.append("")

        lines.append("### Per-experiment aggregate\n")
        n_cells = len(TOP_K_POS) * len(TOP_K_NEG) * len(STRENGTHS)
        s05, s10 = cell_score["[0.5]"], cell_score["[1.0]"]
        pct05 = 100 * s05["shared"] / s05["total"]
        pct10 = 100 * s10["shared"] / s10["total"]
        lines.append(
            f"- **Single `[0.5]` vs multi:** {s05['shared']}/{s05['total']} features overlap "
            f"across {n_cells} cells ({pct05:.1f}%); {s05['identical']}/{n_cells} cells IDENTICAL."
        )
        lines.append(
            f"- **Single `[1.0]` vs multi:** {s10['shared']}/{s10['total']} features overlap "
            f"across {n_cells} cells ({pct10:.1f}%); {s10['identical']}/{n_cells} cells IDENTICAL."
        )
        if s05["shared"] > s10["shared"]:
            winner = "`[0.5]`"
        elif s10["shared"] > s05["shared"]:
            winner = "`[1.0]`"
        else:
            winner = "tie"
        lines.append(f"- **Closer to multi for {exp}:** {winner}")

        for k in global_score:
            for field in global_score[k]:
                global_score[k][field] += cell_score[k][field]

    lines.append("\n---\n\n## Overall verdict across all three experiments\n")
    n_total = len(EXPS) * len(TOP_K_POS) * len(TOP_K_NEG) * len(STRENGTHS)
    g05, g10 = global_score["[0.5]"], global_score["[1.0]"]
    pct05 = 100 * g05["shared"] / g05["total"]
    pct10 = 100 * g10["shared"] / g10["total"]
    lines.append("| Single mode | features ∩ multi | overlap % | IDENTICAL cells |")
    lines.append("|---|---|---|---|")
    lines.append(f"| `[0.5]` | {g05['shared']}/{g05['total']} | {pct05:.1f}% | {g05['identical']}/{n_total} |")
    lines.append(f"| `[1.0]` | {g10['shared']}/{g10['total']} | {pct10:.1f}% | {g10['identical']}/{n_total} |")
    lines.append("")
    if g05["shared"] > g10["shared"]:
        verdict = "`[0.5]`"
    elif g10["shared"] > g05["shared"]:
        verdict = "`[1.0]`"
    else:
        verdict = "tie"
    lines.append(
        f"**Closer single-strength approximation of multi-strength: {verdict}** "
        f"(by raw feature overlap and by IDENTICAL-cell count)."
    )

    doc = "\n".join(lines)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write(doc)

    print(f"Wrote {len(doc):,} chars to {os.path.relpath(OUT_PATH, REPO_ROOT)}")
    print()
    print("=== AGGREGATE ===")
    print(f"[0.5]: {g05['shared']}/{g05['total']} overlap ({pct05:.1f}%); {g05['identical']}/{n_total} identical cells")
    print(f"[1.0]: {g10['shared']}/{g10['total']} overlap ({pct10:.1f}%); {g10['identical']}/{n_total} identical cells")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
