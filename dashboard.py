"""
Streamlit dashboard for SAE experiment results.

Launch:
    uv run --group dashboard streamlit run dashboard.py
"""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS_DIR = Path("results")


# ── Data loading ──────────────────────────────────────────────────


@st.cache_data
def load_all_experiments() -> pd.DataFrame:
    """Scan results/EXP-*/ for experiment JSONs, including failed (log-only) runs."""
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
            # Failed experiment — log only
            log_files = list(exp_dir.glob("*.log"))
            records.append({
                "_exp_id": exp_id,
                "_source": str(log_files[0]) if log_files else str(exp_dir),
                "_failed": True,
            })

    if not records:
        return pd.DataFrame()

    rows = [_extract_row(d) for d in records]
    df = pd.DataFrame(rows)
    df["top_k"] = df["top_k"].astype("Int64")

    return df.sort_values("exp_id", ascending=False).reset_index(drop=True)


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _extract_row(d: dict) -> dict:
    """Flatten a result JSON into a single row for the overview table."""
    if d.get("_failed"):
        return {
            "exp_id": d.get("_exp_id", ""),
            "failed": True,
            "model": "", "model_id": "", "sae_name": "", "architecture": "",
            "layer": 0, "top_k": None, "benchmark": "", "n_questions": 0,
            "full_exact_delta": None, "full_exact_p": None, "full_exact_sig": False,
            "filtered_exact_delta": None, "filtered_exact_p": None,
            "filtered_exact_sig": False,
            "full_approx_delta": None, "filtered_approx_delta": None,
            "kfold_mean_delta": None, "kfold_std_delta": None, "kfold_p": None,
            "kfold_positive_folds": 0, "kfold_n_folds": 0,
            "gen_avg_delta": None, "gen_n_benchmarks": 0,
            "timestamp": "", "source_file": d.get("_source", ""),
        }

    model = d.get("model", {})
    sae = d.get("sae", {})
    bench = d.get("benchmark", {})
    stats = d.get("data_stats", {})
    best = d.get("best_config", {})
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

    return {
        "exp_id": d.get("_exp_id", ""),
        "failed": False,
        "model": model.get("short_name", ""),
        "model_id": model.get("model_id", ""),
        "sae_name": sae.get("name", ""),
        "architecture": sae.get("architecture", ""),
        "layer": sae.get("layer", d.get("intervention_layer", 0)),
        "top_k": int(sae["top_k"]) if sae.get("top_k") else None,
        "benchmark": bench.get("name", ""),
        "n_questions": stats.get("total_questions", 0),
        # Deltas
        "full_exact_delta": full_exact.get("delta", 0),
        "full_exact_p": full_exact.get("binomial_p"),
        "full_exact_sig": full_exact.get("significant", False),
        "filtered_exact_delta": filtered_exact.get("delta", 0),
        "filtered_exact_p": filtered_exact.get("binomial_p"),
        "filtered_exact_sig": filtered_exact.get("significant", False),
        "full_approx_delta": full_approx.get("delta", 0),
        "filtered_approx_delta": filtered_approx.get("delta", 0),
        # K-fold
        "kfold_mean_delta": kfold.get("mean_delta", 0),
        "kfold_std_delta": kfold.get("std_delta", 0),
        "kfold_p": kfold.get("binomial_p"),
        "kfold_positive_folds": kfold.get("n_positive_folds", 0),
        "kfold_n_folds": kfold.get("n_folds", 0),
        # Generalisation
        "gen_avg_delta": gen_summary.get("avg_delta", 0),
        "gen_n_benchmarks": gen_summary.get("n_benchmarks", 0),
        # Meta
        "timestamp": d.get("timestamp", ""),
        "source_file": d.get("_source", ""),
    }


@st.cache_data
def load_experiment_detail(source_file: str) -> dict:
    """Load full JSON for detail view."""
    return _load_json(Path(source_file)) or {}


# ── Formatting helpers ────────────────────────────────────────────

# Column display names for the overview table
_COL_LABELS = {
    "exp_id": "ID",
    "model": "Model",
    "sae_name": "SAE",
    "benchmark": "Benchmark",
    "layer": "L",
    "top_k": "K",
    "n_questions": "N",
    "full_exact_delta": "Full+Exact",
    "filtered_exact_delta": "Filt+Exact",
    "full_approx_delta": "Full+Approx",
    "filtered_approx_delta": "Filt+Approx",
    "kfold_mean_delta": "K-Fold",
    "kfold_positive_folds": "+Folds",
    "gen_avg_delta": "Gen Avg",
}


def _fmt_delta(val) -> str:
    if val is None or pd.isna(val):
        return "-"
    return f"{val * 100:+.2f}%"


def _fmt_pct(val) -> str:
    if val is None or pd.isna(val):
        return ""
    return f"{val * 100:.1f}%"


def _color_delta(val):
    if val is None or pd.isna(val):
        return ""
    if val > 0.005:
        return "color: green; font-weight: bold"
    elif val < -0.005:
        return "color: red; font-weight: bold"
    return "color: gray"


# ── Pages ─────────────────────────────────────────────────────────


def page_overview(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No experiments found. Check results/ directory.")
        return

    # Sidebar filters
    with st.sidebar:
        st.subheader("Filters")

        models = ["All"] + sorted(df["model"].unique().tolist())
        sel_model = st.selectbox("Model", models)

        benchmarks = ["All"] + sorted(df["benchmark"].unique().tolist())
        sel_bench = st.selectbox("Benchmark", benchmarks)

        saes = ["All"] + sorted(df["sae_name"].unique().tolist())
        sel_sae = st.selectbox("SAE", saes)

        layer_min, layer_max = int(df["layer"].min()), int(df["layer"].max())
        if layer_min < layer_max:
            sel_layer = st.slider("Layer range", layer_min, layer_max, (layer_min, layer_max))
        else:
            st.text(f"Layer: {layer_min}")
            sel_layer = (layer_min, layer_max)

        sig_filter = st.selectbox("Significance", ["All", "Significant only", "Non-significant only"])

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if sel_model != "All":
        mask &= df["model"] == sel_model
    if sel_bench != "All":
        mask &= df["benchmark"] == sel_bench
    if sel_sae != "All":
        mask &= df["sae_name"] == sel_sae
    mask &= df["layer"].between(sel_layer[0], sel_layer[1])
    if sig_filter == "Significant only":
        mask &= df["full_exact_sig"] == True
    elif sig_filter == "Non-significant only":
        mask &= df["full_exact_sig"] == False

    filtered = df[mask].copy()
    st.caption(f"{len(filtered)} of {len(df)} experiments")

    # Display columns
    display_cols = [
        "exp_id", "sae_name", "layer", "top_k", "benchmark", "n_questions",
        "full_exact_delta", "full_approx_delta",
        "filtered_exact_delta", "filtered_approx_delta",
        "kfold_mean_delta", "kfold_positive_folds",
        "gen_avg_delta",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    display_df = filtered[display_cols].rename(columns=_COL_LABELS)
    display_df = display_df.set_index("ID")

    # Replace NaN/0 with "-" for failed experiments, format deltas
    failed_mask = filtered["failed"].values
    delta_labels = [_COL_LABELS.get(c, c) for c in display_cols if "delta" in c]

    styled = display_df.style

    # Row background: green for positive delta, red for negative, gray for failed
    def _row_bg(row):
        idx_pos = display_df.index.get_loc(row.name)
        if failed_mask[idx_pos]:
            return ["background-color: rgba(128,128,128,0.15); color: #999"] * len(row)
        delta = filtered.iloc[idx_pos]["full_exact_delta"]
        if delta is not None and not pd.isna(delta):
            if delta > 0.005:
                return ["background-color: rgba(0,128,0,0.12)"] * len(row)
            elif delta < -0.005:
                return ["background-color: rgba(255,0,0,0.10)"] * len(row)
        return [""] * len(row)

    styled = styled.apply(_row_bg, axis=1)

    for col in delta_labels:
        if col in display_df.columns:
            styled = styled.map(_color_delta, subset=[col])
            styled = styled.format(_fmt_delta, subset=[col])

    # Show "-" for failed rows
    def _fmt_failed(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "-"
        return val

    non_delta_cols = [c for c in display_df.columns if c not in delta_labels]
    for col in non_delta_cols:
        styled = styled.format(_fmt_failed, subset=[col])

    event = st.dataframe(
        styled,
        width="stretch",
        height=600,
        selection_mode="single-row",
        on_select="rerun",
        key="overview_table",
    )

    # Navigate: failed → Log, success → Detail
    selected_rows = event.selection.rows
    if selected_rows:
        display_idx = selected_rows[0]
        orig_idx = filtered.index[display_idx]
        global_pos = df.index.get_loc(orig_idx)
        if filtered.loc[orig_idx, "failed"]:
            st.session_state["_go_log"] = int(global_pos)
        else:
            st.session_state["_go_detail"] = int(global_pos)
        st.rerun()


def _detail_label(r) -> str:
    if r.get("failed"):
        return f"{r['exp_id']} — FAILED"
    parts = [r["exp_id"], " — ", r["model"], " / ", r["sae_name"], " / L", str(r["layer"])]
    if pd.notna(r["top_k"]):
        parts += [" / K", str(int(r["top_k"]))]
    parts += [" / ", r["benchmark"], f" (n={r['n_questions']})"]
    return "".join(parts)


def page_detail(df: pd.DataFrame) -> None:
    success_df = df[~df["failed"]].reset_index(drop=True)
    if success_df.empty:
        st.warning("No successful experiments loaded.")
        return

    options = success_df.apply(_detail_label, axis=1).tolist()

    # Map global df index to success_df index
    default_idx = st.session_state.pop("detail_selection", None)
    if default_idx is not None:
        exp_id = df.iloc[default_idx]["exp_id"]
        match = success_df.index[success_df["exp_id"] == exp_id].tolist()
        if match:
            st.session_state["detail_select"] = options[match[0]]

    selected = st.selectbox("Select experiment", options, key="detail_select")
    if not selected:
        return

    idx = options.index(selected)
    row = success_df.iloc[idx]
    # Store global df position for log navigation
    global_idx = df.index[df["exp_id"] == row["exp_id"]].tolist()
    detail_global = global_idx[0] if global_idx else idx
    st.session_state["_current_detail_idx"] = detail_global

    col_left, col_right = st.columns([1, 1])
    with col_left:
        if st.button("< Back to Overview"):
            st.session_state["_go_overview"] = True
            st.rerun()
    with col_right:
        if st.button("View Log >", key="nav_log"):
            st.session_state["_go_log"] = detail_global
            st.rerun()

    data = load_experiment_detail(row["source_file"])

    if not data:
        st.error(f"Could not load: {row['source_file']}")
        return

    # Summary metrics (same order as Overview table)
    def _colored_metric(container, label, val):
        text = _fmt_delta(val)
        if val is not None and not pd.isna(val):
            if val > 0.005:
                color = "green"
            elif val < -0.005:
                color = "red"
            else:
                color = "gray"
        else:
            color = "gray"
        container.markdown(
            f"**{label}**<br><span style='font-size:1.4em;font-weight:bold;"
            f"color:{color}'>{text}</span>",
            unsafe_allow_html=True,
        )

    cols = st.columns(6)
    _colored_metric(cols[0], "Full+Exact", row["full_exact_delta"])
    _colored_metric(cols[1], "Full+Approx", row["full_approx_delta"])
    _colored_metric(cols[2], "Filt+Exact", row["filtered_exact_delta"])
    _colored_metric(cols[3], "Filt+Approx", row["filtered_approx_delta"])
    _colored_metric(cols[4], "K-Fold", row["kfold_mean_delta"])
    gen_val = row["gen_avg_delta"] if row["gen_n_benchmarks"] > 0 else None
    _colored_metric(cols[5], "Gen Avg", gen_val)

    # Full reporting table (sorted: full_exact, full_approx, filtered_exact, filtered_approx)
    st.subheader("Full Reporting")
    fr = data.get("full_reporting", {})
    if fr:
        sort_order = ["full_exact", "full_approx", "filtered_exact", "filtered_approx"]
        sorted_keys = sorted(
            fr.keys(),
            key=lambda k: next(
                (i for i, prefix in enumerate(sort_order) if k.startswith(prefix)),
                len(sort_order),
            ),
        )
        fr_rows = []
        for key in sorted_keys:
            r = fr[key]
            delta = r.get("delta")
            if delta is not None and not pd.isna(delta):
                if delta > 0.005:
                    color = "green"
                elif delta < -0.005:
                    color = "red"
                else:
                    color = "gray"
            else:
                color = "gray"
            fr_rows.append({
                "Report": r.get("name", key),
                "N": r.get("n_questions", 0),
                "Baseline": _fmt_pct(r.get("baseline_acc")),
                "Intervention": _fmt_pct(r.get("intervention_acc")),
                "Delta": _fmt_delta(delta),
                "p-value": f"{r['binomial_p']:.4f}" if r.get("binomial_p") is not None else "N/A",
                "Sig.": "Yes" if r.get("significant") else "No",
                "_delta_raw": delta,
            })
        fr_df = pd.DataFrame(fr_rows)
        styled_fr = fr_df.drop(columns=["_delta_raw"]).style.apply(
            lambda row_s: [
                f"color: {'green' if fr_rows[row_s.name]['_delta_raw'] is not None and fr_rows[row_s.name]['_delta_raw'] > 0.005 else 'red' if fr_rows[row_s.name]['_delta_raw'] is not None and fr_rows[row_s.name]['_delta_raw'] < -0.005 else 'gray'}"
                if col == "Delta" else ""
                for col in row_s.index
            ],
            axis=1,
        )
        st.dataframe(styled_fr, width="stretch", hide_index=True)

    # Best config
    st.subheader("Best Config")
    best = data.get("best_config", data.get("caa", {}))
    if best:
        st.json(best)

    # Difficulty analysis
    st.subheader("Difficulty Analysis")
    da = data.get("difficulty_analysis", {})
    if da and "difficulty_ranges" in da:
        ranges = da["difficulty_ranges"]
        def _range_label(r):
            low = r.get("range_low")
            high = r.get("range_high")
            if low is not None and high is not None:
                return f"{r['range_name']}<br><sub>p∈[{low:.2f}, {high:.2f})</sub>"
            return r["range_name"]

        x_labels = [_range_label(r) for r in ranges]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_labels,
            y=[r["baseline_acc"] * 100 for r in ranges],
            name="Baseline",
            marker_color="steelblue",
        ))
        fig.add_trace(go.Bar(
            x=x_labels,
            y=[r["intervention_acc"] * 100 for r in ranges],
            name="Intervention",
            marker_color="coral",
        ))
        fig.update_layout(barmode="group", yaxis_title="Accuracy (%)", height=350)
        st.plotly_chart(fig, width="stretch")

    # K-fold details
    st.subheader("K-Fold Validation")
    kfold = data.get("kfold_results", {})
    if kfold:
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Delta", _fmt_delta(kfold.get("mean_delta")))
        col2.metric("Std Delta", _fmt_delta(kfold.get("std_delta")))
        col3.metric("+Folds", f"{kfold.get('n_positive_folds', 0)}/{kfold.get('n_folds', 0)}")

        folds = kfold.get("fold_details", [])
        if folds:
            fig = go.Figure(go.Bar(
                x=[f"Fold {i+1}" for i in range(len(folds))],
                y=[f["delta"] * 100 for f in folds],
                marker_color=["green" if f["delta"] > 0 else "red" for f in folds],
            ))
            fig.update_layout(yaxis_title="Delta (%)", height=300)
            st.plotly_chart(fig, width="stretch")

    # Generalisation
    gen = data.get("generalisation", {})
    if gen and "benchmarks" in gen:
        st.subheader("Generalisation")
        gen_rows = []
        for b in gen["benchmarks"]:
            gen_rows.append({
                "Benchmark": b["benchmark"],
                "N": b["n_questions"],
                "Baseline": _fmt_pct(b["baseline_acc"]),
                "Intervention": _fmt_pct(b["intervention_acc"]),
                "Delta": _fmt_delta(b["delta"]),
                "p-value": f"{b['binomial_p']:.4f}" if b.get("binomial_p") is not None else "N/A",
                "Sig.": "Yes" if b.get("significant") else "No",
            })
        st.dataframe(pd.DataFrame(gen_rows), width="stretch", hide_index=True)

    # Plot image
    exp_dir = Path(row["source_file"]).parent
    png_files = list(exp_dir.glob("*.png"))
    if png_files:
        st.subheader("Plot")
        for png in png_files:
            st.image(str(png), caption=png.name)

    st.caption(f"Source: `{row['source_file']}`")


def page_log(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No experiments loaded.")
        return

    options = df.apply(_detail_label, axis=1).tolist()

    default_idx = st.session_state.pop("log_selection", None)
    if default_idx is not None:
        default_idx = min(default_idx, len(options) - 1)
        st.session_state["log_select"] = options[default_idx]

    selected = st.selectbox("Select experiment", options, key="log_select")
    if not selected:
        return

    idx = options.index(selected)
    st.session_state["_current_log_idx"] = idx
    row = df.iloc[idx]
    is_failed = row.get("failed", False)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        if st.button("< Back to Overview"):
            st.session_state["_go_overview"] = True
            st.rerun()
    with col_right:
        if st.button("View Detail >", key="nav_detail", disabled=is_failed):
            st.session_state["_go_detail"] = idx
            st.rerun()

    exp_dir = Path(row["source_file"]).parent
    log_files = sorted(exp_dir.glob("*.log"), key=lambda p: p.name, reverse=True)
    if not log_files:
        st.warning("No log file found for this experiment.")
        return

    if len(log_files) > 1:
        selected_log = st.selectbox(
            "Log file", log_files, format_func=lambda p: p.name, key="log_file_select",
        )
    else:
        selected_log = log_files[0]

    log_text = selected_log.read_text(encoding="utf-8", errors="surrogateescape")

    search = st.text_input("Search log", placeholder="Filter lines (case-insensitive)...")
    if search:
        lines = [l for l in log_text.splitlines() if search.lower() in l.lower()]
        st.caption(f"{len(lines)} matching lines")
        display_text = "\n".join(lines)
    else:
        display_text = log_text

    escaped = html_mod.escape(display_text)
    st.html(
        f'<div style="max-height:600px;overflow:auto;display:flex;'
        f'flex-direction:column-reverse;'
        f'background:#0e1117;color:#fafafa;border-radius:4px;'
        f'font-family:monospace;font-size:13px;line-height:1.4">'
        f'<pre style="margin:0;padding:1em;white-space:pre">{escaped}</pre>'
        f"</div>"
    )


def page_charts(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No experiments loaded.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Layer Sweep", "Model Comparison", "K-Value Comparison",
        "Hook Point Comparison", "Generalisation Matrix",
    ])

    with tab1:
        _chart_layer_sweep(df)

    with tab2:
        _chart_model_comparison(df)

    with tab3:
        _chart_k_value(df)

    with tab4:
        _chart_hook_point(df)

    with tab5:
        _chart_generalisation_matrix(df)


def _chart_layer_sweep(df: pd.DataFrame) -> None:
    st.subheader("Delta vs Layer")

    groups = df.groupby(["model", "benchmark"]).filter(lambda g: g["layer"].nunique() > 1)
    if groups.empty:
        st.info("No multi-layer sweeps found.")
        return

    groups = groups.copy()
    groups["group"] = groups["model"] + " / " + groups["benchmark"]

    fig = px.line(
        groups, x="layer", y="full_exact_delta",
        color="group", markers=True,
        labels={"full_exact_delta": "Full+Exact Delta", "layer": "Layer"},
    )
    fig.update_layout(yaxis_tickformat=".1%", height=450)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")


def _chart_model_comparison(df: pd.DataFrame) -> None:
    st.subheader("Delta by Model")

    sel_bench = st.selectbox("Benchmark", sorted(df["benchmark"].unique()), key="model_cmp_bench")
    subset = df[df["benchmark"] == sel_bench]

    if subset.empty:
        st.info("No data for this benchmark.")
        return

    fig = px.bar(
        subset, x="model", y="full_exact_delta",
        text="exp_id",
        labels={"full_exact_delta": "Full+Exact Delta"},
    )
    fig.update_layout(yaxis_tickformat=".1%", height=400)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")


def _chart_k_value(df: pd.DataFrame) -> None:
    st.subheader("Delta vs TopK")
    btk = df[(df["architecture"] == "batchtopk") & (df["top_k"] != "")]
    if btk.empty:
        st.info("No BatchTopK experiments with varying K values.")
        return

    groups = btk.groupby(["model", "benchmark", "layer"]).filter(
        lambda g: g["top_k"].nunique() > 1
    )
    if groups.empty:
        st.info("No multi-K sweeps found.")
        return

    groups = groups.copy()
    groups["group"] = groups["model"] + " / " + groups["benchmark"] + " / L" + groups["layer"].astype(str)

    fig = px.line(
        groups, x="top_k", y="full_exact_delta",
        color="group", markers=True,
        labels={"full_exact_delta": "Full+Exact Delta", "top_k": "TopK"},
    )
    fig.update_layout(yaxis_tickformat=".1%", height=400)
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, width="stretch")


def _chart_hook_point(df: pd.DataFrame) -> None:
    st.subheader("MLP vs Resid Post")

    hook_df = df.copy()
    hook_df["hook_hint"] = hook_df["sae_name"].str.lower().apply(
        lambda x: "mlp" if "mlp" in x else ("resid" if "resid" in x else "other")
    )

    paired = hook_df[hook_df["hook_hint"].isin(["mlp", "resid"])]
    if paired.empty:
        st.info("No MLP vs Resid experiments found.")
        return

    pivot = paired.pivot_table(
        index=["model", "benchmark", "layer"],
        columns="hook_hint",
        values="full_exact_delta",
    ).dropna()

    if pivot.empty or "mlp" not in pivot.columns or "resid" not in pivot.columns:
        st.info("No paired MLP/Resid comparisons found.")
        return

    pivot = pivot.reset_index()
    pivot["label"] = pivot["model"] + " / " + pivot["benchmark"] + " / L" + pivot["layer"].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pivot["resid"] * 100, y=pivot["mlp"] * 100,
        mode="markers+text", text=pivot["label"],
        textposition="top center",
        marker=dict(size=10),
    ))
    max_val = max(abs(pivot["resid"].max()), abs(pivot["mlp"].max()), 0.01) * 100 * 1.2
    fig.add_shape(type="line", x0=-max_val, y0=-max_val, x1=max_val, y1=max_val,
                  line=dict(dash="dash", color="gray"))
    fig.update_layout(
        xaxis_title="Resid Post Delta (%)", yaxis_title="MLP Delta (%)",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


def _chart_generalisation_matrix(df: pd.DataFrame) -> None:
    st.subheader("Generalisation Heatmap")

    gen_rows = []
    for _, row in df.iterrows():
        data = load_experiment_detail(row["source_file"])
        gen = data.get("generalisation", {})
        if not gen or "benchmarks" not in gen:
            continue
        for b in gen["benchmarks"]:
            gen_rows.append({
                "trained_on": row["benchmark"],
                "tested_on": b["benchmark"],
                "delta": b["delta"],
                "model": row["model"],
                "exp_id": row["exp_id"],
            })

    if not gen_rows:
        st.info("No generalisation data found.")
        return

    gen_df = pd.DataFrame(gen_rows)

    sel_model = st.selectbox(
        "Model", sorted(gen_df["model"].unique()), key="gen_model"
    )
    gen_df = gen_df[gen_df["model"] == sel_model]

    pivot = gen_df.pivot_table(
        index="trained_on", columns="tested_on", values="delta", aggfunc="mean"
    )

    fig = px.imshow(
        pivot.values * 100,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        labels=dict(color="Delta (%)"),
        text_auto=".2f",
    )
    fig.update_layout(
        xaxis_title="Tested On", yaxis_title="Trained On", height=450,
    )
    st.plotly_chart(fig, width="stretch")


# ── Main ──────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="SAE Experiments Dashboard", layout="wide")

    df = load_all_experiments()

    # Handle programmatic navigation
    if "_go_overview" in st.session_state:
        st.session_state.pop("_go_overview")
        st.session_state["page_radio"] = "Overview"
        st.session_state.pop("overview_table", None)

    if "_go_detail" in st.session_state:
        st.session_state["detail_selection"] = st.session_state.pop("_go_detail")
        st.session_state["page_radio"] = "Detail"
        st.session_state.pop("overview_table", None)

    if "_go_log" in st.session_state:
        st.session_state["log_selection"] = st.session_state.pop("_go_log")
        st.session_state["page_radio"] = "Log"

    pages = ["Overview", "Detail", "Log", "Charts"]
    page = st.sidebar.radio("Page", pages, key="page_radio")

    st.header("SAE Experiments Dashboard")

    if page == "Overview":
        page_overview(df)
    elif page == "Detail":
        page_detail(df)
    elif page == "Log":
        page_log(df)
    elif page == "Charts":
        page_charts(df)


if __name__ == "__main__":
    main()
