"""9-panel visualization for experiment results."""

from __future__ import annotations

import matplotlib
matplotlib.use("agg")

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import CausalConfig, ExperimentConfig


def generate_all_plots(
    *,
    full_reporting_results: Dict,
    df_grid: pd.DataFrame,
    best_config: Dict,
    screening_data: List[Dict],
    filtered_data: List[Dict],
    causal_config: CausalConfig,
    experiment_config: ExperimentConfig,
    output_dir: str,
) -> Optional[str]:
    """Generate the 9-panel summary figure and return the saved file path."""
    sae_cfg = experiment_config.sae
    bench_cfg = experiment_config.benchmark

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(
        f"Full Reporting: {bench_cfg.name} | SAE: {sae_cfg.name}",
        fontsize=14, fontweight="bold",
    )

    # Plot 1: Full Reporting Delta comparison
    ax1 = axes[0, 0]
    report_names = [r["name"] for r in full_reporting_results.values()]
    deltas = [r["delta"] * 100 for r in full_reporting_results.values()]
    colors = ["green" if d > 0 else "red" for d in deltas]

    bars = ax1.bar(range(len(report_names)), deltas, color=colors, alpha=0.7, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xticks(range(len(report_names)))
    ax1.set_xticklabels(report_names, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Delta (%)")
    ax1.set_title("Full Reporting: Delta Comparison")
    ax1.grid(True, alpha=0.3, axis="y")

    for bar, delta in zip(bars, deltas):
        ypos = bar.get_height() + 0.3 if delta >= 0 else bar.get_height() - 0.8
        ax1.text(bar.get_x() + bar.get_width() / 2, ypos, f"{delta:+.1f}%",
                 ha="center", va="bottom" if delta >= 0 else "top", fontsize=9)

    # Plot 2: Exact vs Approximated
    ax2 = axes[0, 1]
    def _get_report(key):
        if key in full_reporting_results:
            return full_reporting_results[key]
        for suffix in ("_static", "_adaptive"):
            if key + suffix in full_reporting_results:
                return full_reporting_results[key + suffix]
        return {"delta": 0.0}

    exact_delta_filtered = _get_report("filtered_exact")["delta"] * 100
    approx_delta_filtered = _get_report("filtered_approx")["delta"] * 100
    exact_delta_full = _get_report("full_exact")["delta"] * 100
    approx_delta_full = _get_report("full_approx")["delta"] * 100

    methods = ["Filtered", "Full"]
    exact_deltas = [exact_delta_filtered, exact_delta_full]
    approx_deltas = [approx_delta_filtered, approx_delta_full]

    x = np.arange(len(methods))
    width = 0.35

    ax2.bar(x - width / 2, exact_deltas, width, label="Exact (with SAE)", color="blue", alpha=0.7)
    ax2.bar(x + width / 2, approx_deltas, width, label="Approximated (no SAE)", color="orange", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel("Delta (%)")
    ax2.set_title("Exact vs Approximated")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Improvements vs Degradations
    ax3 = axes[0, 2]
    short_names = [r["name"] for r in full_reporting_results.values()]
    improved = [r["n_improved"] for r in full_reporting_results.values()]
    worsened = [-r["n_worsened"] for r in full_reporting_results.values()]

    x = np.arange(len(short_names))
    ax3.bar(x, improved, label="Improved", color="green", alpha=0.7)
    ax3.bar(x, worsened, label="Worsened", color="red", alpha=0.7)
    ax3.axhline(0, color="black", linewidth=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax3.set_ylabel("Number of questions")
    ax3.set_title("Improvements vs Degradations")
    ax3.legend()

    # Detect whether df_grid has SAE grid columns (absent for CAA sweep)
    _is_sae_grid = "ranking_criterion" in df_grid.columns

    # Plot 4: Heatmap causal_delta + uniform (SAE only)
    ax4 = axes[1, 0]
    if _is_sae_grid:
        subset = df_grid[
            (df_grid["strength"] == best_config["strength"])
            & (df_grid["ranking_criterion"] == "causal_delta")
            & (df_grid["weighting_mode"] == "uniform")
        ]
        if len(subset) > 0:
            pivot = subset.pivot_table(values=next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in subset.columns), "delta"), index="top_k_neg", columns="top_k_pos", aggfunc="mean") * 100
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax4,
                        cbar_kws={"label": "Delta (%)"})
            ax4.set_title(f'Causal + Uniform (strength={best_config["strength"]})')
            ax4.set_xlabel("Top-K Positive")
            ax4.set_ylabel("Top-K Negative")
        else:
            ax4.text(0.5, 0.5, "No data", ha="center", va="center")
            ax4.set_title("Causal + Uniform")
    else:
        ax4.text(0.5, 0.5, "N/A (CAA)", ha="center", va="center", fontsize=12, color="gray")
        ax4.set_title("Causal + Uniform (SAE only)")

    # Plot 5: Heatmap freq_diff + uniform (SAE only)
    ax5 = axes[1, 1]
    if _is_sae_grid:
        subset = df_grid[
            (df_grid["strength"] == best_config["strength"])
            & (df_grid["ranking_criterion"] == "freq_diff")
            & (df_grid["weighting_mode"] == "uniform")
        ]
        if len(subset) > 0:
            pivot = subset.pivot_table(values=next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in subset.columns), "delta"), index="top_k_neg", columns="top_k_pos", aggfunc="mean") * 100
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax5,
                        cbar_kws={"label": "Delta (%)"})
            ax5.set_title(f'freq_diff + Uniform (strength={best_config["strength"]})')
            ax5.set_xlabel("Top-K Positive")
            ax5.set_ylabel("Top-K Negative")
        else:
            ax5.text(0.5, 0.5, "No data", ha="center", va="center")
            ax5.set_title("freq_diff + Uniform")
    else:
        ax5.text(0.5, 0.5, "N/A (CAA)", ha="center", va="center", fontsize=12, color="gray")
        ax5.set_title("freq_diff + Uniform (SAE only)")

    # Plot 6: Boxplot by ranking criterion (SAE) or Delta vs Strength (CAA)
    ax6 = axes[1, 2]
    if _is_sae_grid:
        _dcol = next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in df_grid.columns), "delta")
        df_grid.boxplot(column=_dcol, by="ranking_criterion", ax=ax6)
        ax6.set_title("Delta by Ranking Criterion")
        ax6.set_xlabel("Ranking Criterion")
        ax6.set_ylabel("Delta")
        ax6.axhline(0, color="red", linestyle="--", alpha=0.5)
        # boxplot(by=...) sets a pandas suptitle — restore our own
        fig.suptitle(
            f"Full Reporting: {bench_cfg.name} | SAE: {sae_cfg.name}",
            fontsize=14, fontweight="bold",
        )
    else:
        _dcol = next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in df_grid.columns), "delta")
        ax6.bar(range(len(df_grid)), df_grid[_dcol] * 100, color="steelblue", alpha=0.7)
        ax6.set_xticks(range(len(df_grid)))
        ax6.set_xticklabels([f"{s:.1f}" for s in df_grid["strength"]], rotation=45)
        ax6.set_xlabel("Strength")
        ax6.set_ylabel("Delta (%)")
        ax6.set_title("CAA Strength Sweep")
        ax6.axhline(0, color="red", linestyle="--", alpha=0.5)

    # Plot 7: Delta vs Strength
    ax7 = axes[2, 0]
    if _is_sae_grid:
        for ranking in df_grid["ranking_criterion"].unique():
            for weighting in df_grid["weighting_mode"].unique():
                sub = df_grid[
                    (df_grid["ranking_criterion"] == ranking)
                    & (df_grid["weighting_mode"] == weighting)
                    & (df_grid["top_k_neg"] == 0)
                ]
                if len(sub) > 0:
                    _dcol = next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in sub.columns), "delta")
                    grouped = sub.groupby("strength")[_dcol].mean() * 100
                    label = f"{ranking[:6]}+{weighting[:4]}"
                    ax7.plot(grouped.index, grouped.values, "o-", label=label, alpha=0.7)
        ax7.legend(fontsize=8)
    else:
        _dcol = next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in df_grid.columns), "delta")
        ax7.plot(df_grid["strength"], df_grid[_dcol] * 100, "o-", color="steelblue")
    ax7.axhline(0, color="black", linewidth=1, linestyle="--")
    ax7.set_xlabel("Strength")
    ax7.set_ylabel("Avg Delta (%)")
    ax7.set_title("Delta vs Strength")
    ax7.grid(True, alpha=0.3)

    # Plot 8: P(correct) distribution
    ax8 = axes[2, 1]
    ax8.hist(
        [s["p_correct"] for s in screening_data], bins=50, alpha=0.5,
        label=f"All (N={len(screening_data)})", color="blue",
    )
    ax8.hist(
        [s["p_correct"] for s in filtered_data], bins=50, alpha=0.5,
        label=f"Filtered (N={len(filtered_data)})", color="orange",
    )
    ax8.axvline(causal_config.min_p_correct, color="red", linewidth=2, linestyle="--")
    ax8.axvline(
        causal_config.max_p_correct, color="red", linewidth=2, linestyle="--",
        label=f"Filter: [{causal_config.min_p_correct}, {causal_config.max_p_correct}]",
    )
    ax8.set_xlabel("P(correct)")
    ax8.set_ylabel("Count")
    ax8.set_title("P(correct) Distribution")
    ax8.legend(fontsize=8)
    ax8.grid(True, alpha=0.3)

    # Plot 9: Grid search / strength sweep delta distribution
    ax9 = axes[2, 2]
    _dcol = next((c for c in ("delta_grid_eval", "delta_full", "delta") if c in df_grid.columns), "delta")
    if _is_sae_grid:
        for ranking in df_grid["ranking_criterion"].unique():
            sub = df_grid[df_grid["ranking_criterion"] == ranking]
            ax9.hist(sub[_dcol] * 100, bins=20, alpha=0.5, label=ranking)
    else:
        ax9.hist(df_grid[_dcol] * 100, bins=max(5, len(df_grid) // 2), alpha=0.7, color="steelblue")

    ax9.axvline(0, color="black", linewidth=2, linestyle="-")
    ax9.axvline(
        df_grid[_dcol].mean() * 100, color="blue", linewidth=2, linestyle="--",
        label=f"Avg: {df_grid[_dcol].mean()*100:+.1f}%",
    )
    ax9.set_xlabel("Delta (%)")
    ax9.set_ylabel("Number of configurations")
    ax9.set_title("Sweep Delta Distribution")
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sae_suffix = "batchtopk" if sae_cfg.architecture == "batchtopk" else "relu"
    layer_suffix = f"_L{sae_cfg.layer}"
    k_suffix = f"_K{sae_cfg.top_k}" if sae_cfg.top_k else ""
    plot_file = os.path.join(
        output_dir,
        f"full_reporting_{bench_cfg.cache_name}_{sae_suffix}{layer_suffix}{k_suffix}.png",
    )
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved: {plot_file}")
    return plot_file
