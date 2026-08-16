"""Tests for scripts/summarize_final_matrix.py — the final-matrix summary generator.

The script is cell-driven: it walks the 220 cells of run_experiments_final.py, drops
the 30 attention cells, and reports the remaining 190 whether they have run or not.
The tests below cover, in order:

  * cell loading and the axis metadata read from the SAE/model YAMLs
  * the attention exclusion rule (hook_point, never filename) against the ID list
    that temp.txt excludes
  * status classification from the filesystem, including the log-footer cases and
    the -ABORT / -FAILED / _archived_ siblings
  * metric extraction from a result JSON, incl. the _static/_adaptive key suffixes
  * row assembly, the JSONL/Markdown writers and the CLI

A final pair of tests runs the script against the real committed results/ tree.
"""

import importlib.util
import json
import sys
from pathlib import Path

# The script lives under scripts/, not in the package. Load it by path so the tests
# don't depend on scripts/ being importable.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "summarize_final_matrix.py"
_spec = importlib.util.spec_from_file_location("summarize_final_matrix", _SCRIPT_PATH)
summarize = importlib.util.module_from_spec(_spec)
sys.modules["summarize_final_matrix"] = summarize
_spec.loader.exec_module(summarize)

REPO_ROOT = Path(__file__).parent.parent

# The 30 attention cells, copied from temp.txt's exclusion list.
ATTENTION_IDS = {
    "EXP-365", "EXP-366", "EXP-367", "EXP-368", "EXP-391", "EXP-392", "EXP-393",
    "EXP-394", "EXP-403", "EXP-404", "EXP-405", "EXP-406", "EXP-426", "EXP-427",
    "EXP-428", "EXP-429", "EXP-442", "EXP-443", "EXP-444", "EXP-445", "EXP-458",
    "EXP-459", "EXP-460", "EXP-461", "EXP-470", "EXP-471", "EXP-472", "EXP-473",
    "EXP-486", "EXP-487",
}


# ── Task 1: cells, axes, exclusion ────────────────────────────────────────────


def test_load_cells_returns_220_cells_with_stable_only_indices():
    cells = summarize.load_cells(REPO_ROOT)
    assert len(cells) == 220
    assert cells[0].idx == 1 and cells[0].exp == "EXP-300"
    # --only N addresses EXP-(299+N); the runner never renumbers.
    for c in cells:
        assert int(c.exp.split("-")[1]) == 299 + c.idx


def test_split_cells_excludes_exactly_the_temp_txt_attention_ids():
    cells = summarize.load_cells(REPO_ROOT)
    kept, attention = summarize.split_cells(cells, REPO_ROOT)
    assert {c.exp for c, _ in attention} == ATTENTION_IDS
    assert len(kept) == 190
    assert all(m["site"] in ("resid", "mlp") for _, m in kept)
    assert all(m["site"] == "attn" for _, m in attention)


def test_axis_meta_reads_sae_and_model_yaml():
    cells = {c.exp: c for c in summarize.load_cells(REPO_ROOT)}
    meta = summarize.axis_meta(cells["EXP-375"], REPO_ROOT)
    assert meta["model"] == "pythia_160m_deduped"
    assert meta["model_id"] == "EleutherAI/pythia-160m-deduped"
    assert meta["params"] == "160M"
    assert meta["arch"] == "batchtopk"
    assert meta["width"] == 65536
    assert meta["top_k"] == 32
    assert meta["site"] == "resid"
    assert meta["n_layers"] == 12
    assert meta["layer_frac"] == 1.0            # L11 of 12 layers
    assert meta["benchmark"] == "CSQA"
    assert meta["benchmark_config"] == "commonsenseqa"
    assert meta["meta_error"] is None


def test_axis_meta_reports_missing_yaml_instead_of_raising():
    bad = summarize.Cell(idx=1, exp="EXP-999", sae="configs/saes/nope.yaml",
                         layer=3, benchmark="configs/benchmarks/commonsenseqa.yaml")
    meta = summarize.axis_meta(bad, REPO_ROOT)
    assert meta["meta_error"] is not None
    assert meta["site"] is None


def test_parse_params_handles_the_awkward_model_names():
    assert summarize.parse_params("pythia_70m_deduped")[0] == "70M"
    assert summarize.parse_params("gemma2_2b_it")[0] == "2B"
    assert summarize.parse_params("gemma3_12b_it")[0] == "12B"
    assert summarize.parse_params("gemma3_270m")[0] == "270M"
    assert summarize.parse_params("gpt2_small")[0] == "124M"            # override map
    assert summarize.parse_params("deepseek_r1_qwen_15b")[0] == "1.5B"  # override map
    assert summarize.parse_params("pythia_70m")[1] < summarize.parse_params("gemma_2b")[1]


# ── Task 2: status resolution ─────────────────────────────────────────────────


def _write_log(d: Path, name: str, body: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


FOOTER_OK = """some output
======================================================================
Finished: 2026-07-26T15:31:13
Duration: 4:15:44
Result: SUCCESS
"""
FOOTER_FAIL = """some output
======================================================================
Finished: 2026-07-26T15:31:13
Duration: 0:01:12
Result: FAILED
Error: CUDA out of memory
"""


def test_status_done_picks_newest_result_json_and_ignores_screening(tmp_path):
    d = tmp_path / "EXP-400"
    _write_log(d, "20260726_100000_EXP-400.log", FOOTER_OK)
    (d / "logit_screening_commonsenseqa_x.json").write_text("{}", encoding="utf-8")
    (d / "commonsenseqa_relu_L5_20260101_010101.json").write_text("{}", encoding="utf-8")
    (d / "commonsenseqa_relu_L5_20260726_153113.json").write_text("{}", encoding="utf-8")
    info = summarize.resolve_status("EXP-400", tmp_path)
    assert info["status"] == "done"
    assert info["result_json"].name == "commonsenseqa_relu_L5_20260726_153113.json"
    assert info["n_result_files"] == 2
    assert info["timestamp"] == "2026-07-26 15:31:13"
    assert info["duration_s"] == 4 * 3600 + 15 * 60 + 44
    assert info["dir"] == "results/EXP-400"


def test_status_running_when_log_has_no_footer(tmp_path):
    _write_log(tmp_path / "EXP-401", "20260728_090000_EXP-401.log", "loading model...\n")
    assert summarize.resolve_status("EXP-401", tmp_path)["status"] == "running"


def test_status_failed_reads_error_from_footer(tmp_path):
    _write_log(tmp_path / "EXP-402", "20260728_090000_EXP-402.log", FOOTER_FAIL)
    info = summarize.resolve_status("EXP-402", tmp_path)
    assert info["status"] == "failed"
    assert info["error"] == "CUDA out of memory"
    assert info["duration_s"] == 72


def test_status_done_no_json_flags_success_footer_without_result(tmp_path):
    _write_log(tmp_path / "EXP-407", "20260728_090000_EXP-407.log", FOOTER_OK)
    assert summarize.resolve_status("EXP-407", tmp_path)["status"] == "done_no_json"


def test_status_abort_sibling_wins_over_failed_sibling(tmp_path):
    (tmp_path / "EXP-488-ABORT").mkdir(parents=True)
    (tmp_path / "EXP-488-FAILED").mkdir(parents=True)
    info = summarize.resolve_status("EXP-488", tmp_path)
    assert info["status"] == "aborted"
    assert info["dir"] == "results/EXP-488-ABORT"


def test_status_failed_and_archived_siblings(tmp_path):
    (tmp_path / "EXP-486-FAILED").mkdir(parents=True)
    (tmp_path / "EXP-489_archived_20260726_101010").mkdir(parents=True)
    assert summarize.resolve_status("EXP-486", tmp_path)["status"] == "failed"
    assert summarize.resolve_status("EXP-489", tmp_path)["status"] == "archived_only"


def test_status_pending_and_variant_dirs_ignored(tmp_path):
    (tmp_path / "EXP-300-test-strength-0.5").mkdir(parents=True)
    info = summarize.resolve_status("EXP-300", tmp_path)
    assert info["status"] == "pending"
    assert info["dir"] is None


# ── Task 3: metrics and mismatch ──────────────────────────────────────────────


RESULT_FIXTURE = {
    "timestamp": "2026-07-26T15:31:13",
    "sae": {"layer": 11, "architecture": "batchtopk", "top_k": 32},
    "benchmark": {"name": "CommonsenseQA"},
    "data_stats": {
        "total_questions": 10962, "filtered_questions": 6052,
        "holdout_questions": 3211, "holdout_filtered_questions": 757,
        "baseline_filtered_accuracy": 0.2615, "baseline_full_accuracy": 0.1759,
    },
    # Suffixed keys: the reporting mode decides whether the plain or the
    # _static/_adaptive name is written.
    "full_reporting": {
        "filtered_exact_static": {"n_questions": 757, "delta": -0.00924,
                                  "binomial_p": 0.998, "significant": False},
        "full_exact_adaptive": {"n_questions": 3211, "delta": 0.0123,
                                "binomial_p": 0.031, "significant": True},
        "filtered_approx": {"delta": -0.00264, "binomial_p": 0.9375, "significant": False},
        "full_approx": {"delta": -0.00124, "binomial_p": 0.769, "significant": False},
    },
    "approximation_comparison": {"filtered": {"difference": 0.0066},
                                 "full": {"difference": 0.00062}},
    "kfold_results": {"mean_delta": -0.00186, "std_delta": 0.00617,
                      "n_positive_folds": 1, "n_folds": 5, "binomial_p": 0.8295},
    "generalisation": {"summary": {"n_benchmarks": 3, "n_positive": 2,
                                   "n_significant": 0, "avg_delta": 0.00032}},
    "best_config": {"top_k_pos": 10, "top_k_neg": 5, "strength": 1.0,
                    "ranking_criterion": "causal_delta", "weighting_mode": "uniform",
                    "n_features_used": 15},
}


def test_metrics_from_result_handles_static_and_adaptive_suffixes():
    m = summarize.metrics_from_result(RESULT_FIXTURE)
    assert m["flt_exact_delta"] == -0.00924
    assert m["flt_exact_p"] == 0.998
    assert m["flt_exact_sig"] is False
    assert m["flt_exact_n"] == 757
    assert m["full_exact_delta"] == 0.0123
    assert m["full_exact_sig"] is True
    assert m["flt_approx_delta"] == -0.00264
    assert m["approx_gap_filtered"] == 0.0066
    assert m["kfold_pos_folds"] == "1/5"
    assert m["gen_avg_delta"] == 0.00032
    assert m["best_strength"] == 1.0
    assert m["n_holdout_filtered"] == 757
    assert m["base_acc_filtered"] == 0.2615


def test_metrics_from_result_returns_all_keys_none_for_empty_dict():
    m = summarize.metrics_from_result({})
    assert set(m) == set(summarize.METRIC_KEYS)
    assert all(v is None for v in m.values())


def test_gen_avg_delta_is_none_when_no_generalisation_benchmarks():
    data = json.loads(json.dumps(RESULT_FIXTURE))
    data["generalisation"]["summary"]["n_benchmarks"] = 0
    assert summarize.metrics_from_result(data)["gen_avg_delta"] is None


def test_detect_mismatch_on_layer_and_benchmark():
    cell = summarize.Cell(idx=76, exp="EXP-375",
                          sae="configs/saes/eleutherai_pythia_160m_deduped_32k_resid.yaml",
                          layer=11, benchmark="configs/benchmarks/commonsenseqa.yaml")
    meta = summarize.axis_meta(cell, REPO_ROOT)
    assert summarize.detect_mismatch(RESULT_FIXTURE, cell, meta) is False

    wrong_layer = json.loads(json.dumps(RESULT_FIXTURE))
    wrong_layer["sae"]["layer"] = 5
    assert summarize.detect_mismatch(wrong_layer, cell, meta) is True

    wrong_bench = json.loads(json.dumps(RESULT_FIXTURE))
    wrong_bench["benchmark"]["name"] = "ARC-Challenge"
    assert summarize.detect_mismatch(wrong_bench, cell, meta) is True


# ── Task 4: rows, writers, CLI ────────────────────────────────────────────────


def test_build_rows_covers_190_cells_in_exp_order_with_uniform_keys(tmp_path):
    rows = summarize.build_rows(REPO_ROOT, tmp_path)   # empty results dir
    assert len(rows) == 190
    assert [r["exp"] for r in rows] == sorted(r["exp"] for r in rows)
    assert all(all(k in r for k in summarize.ROW_KEYS) for r in rows)
    assert all(r["status"] == "pending" for r in rows)
    assert all(r["flt_exact_delta"] is None for r in rows)
    assert all(r["site"] in ("resid", "mlp") for r in rows)


def test_build_rows_include_attention_adds_the_30_attention_cells(tmp_path):
    rows = summarize.build_rows(REPO_ROOT, tmp_path, include_attention=True)
    assert len(rows) == 220
    assert sum(1 for r in rows if r["site"] == "attn") == 30


def test_render_table_row_count_matches_and_formats_percent(tmp_path):
    rows = summarize.build_rows(REPO_ROOT, tmp_path)
    rows[0]["status"] = "done"
    rows[0]["flt_exact_delta"] = 0.0241
    rows[0]["flt_exact_p"] = 0.031
    rows[0]["flt_exact_sig"] = True
    table = summarize.render_table(rows)
    body = [ln for ln in table.splitlines()[1:] if ln.strip()]
    assert len(body) == 190
    assert "+2.41%" in table
    assert "pend" in table


def test_full_holdout_is_the_primary_column_block():
    """The full holdout leads and owns the bare `p`/`s`; filtered variants are prefixed.

    The table used to open on the filtered holdout and hand it the only p-value and
    significance columns, while `fullEx` carried neither and `nHold` was the filtered
    count. The full holdout is the reported result, so it comes first and the
    unqualified labels are its.
    """
    labels = [h for h, _f, _a in summarize._COLUMNS]
    assert labels.index("nHold") < labels.index("fullEx") < labels.index("fltEx")
    assert labels.index("p") == labels.index("fullEx") + 1
    assert labels.index("s") == labels.index("p") + 1
    # the filtered holdout keeps its own p/significance, under prefixed labels
    assert {"nFltH", "fltP", "fltS"} <= set(labels)

    fmt = {h: f for h, f, _a in summarize._COLUMNS}
    row = {k: None for k in summarize.ROW_KEYS}
    row.update(n_holdout=3211, n_holdout_filtered=757,
               full_exact_delta=0.0123, full_exact_p=0.031, full_exact_sig=True,
               full_approx_delta=0.0089,
               flt_exact_delta=-0.00924, flt_exact_p=0.998, flt_exact_sig=False)
    assert fmt["nHold"](row) == "3211"     # the full holdout, not the filtered one
    assert fmt["nFltH"](row) == "757"
    assert fmt["fullEx"](row) == "+1.23%"
    assert fmt["p"](row) == "0.031"        # the bare p belongs to full-exact
    assert fmt["s"](row) == "*"
    assert fmt["fullAp"](row) == "+0.89%"
    assert fmt["fltEx"](row) == "-0.92%"
    assert fmt["fltP"](row) == "0.998"
    assert fmt["fltS"](row) == ""


def test_render_table_last_column_is_runtime_in_hours(tmp_path):
    rows = summarize.build_rows(REPO_ROOT, tmp_path)
    rows[0]["status"] = "done"
    rows[0]["duration_s"] = 21396          # 5.94 h, as parsed from a log footer
    table = summarize.render_table(rows)
    header = table.splitlines()[0].split()
    assert header[-1] == "hrs", "runtime column must come last, after gen"
    assert header.index("gen") == header.index("hrs") - 1
    row = [ln for ln in table.splitlines() if ln.startswith(rows[0]["exp"][4:])][0]
    assert row.split()[-1] == "5.9"
    # a cell without a duration renders a bare dash, like every other missing value
    pending = [ln for ln in table.splitlines()[1:] if ln.strip() and " pend " in ln][0]
    assert pending.split()[-1] == "-"


def test_extract_notes_reads_annotations_and_ignores_generated_text():
    table = "\n".join([
        "exp  idx  model     status  fullEx   hrs",
        "300    1  gemma_2b  done    +8.83%   2.2 (drop)",
        "301    2  gemma_2b  done    +0.34%   2.3 (run later) but only if money allows it",
        "302    3  gemma_2b  done    +3.96%   2.1",
        "",
        "415  116  smollm2   pend         -     - (RUNNING-ID: 4, takes 24 h)",
        "not a data row (with a parenthesis)",
    ])
    notes = summarize.extract_notes(table)
    assert notes == {
        "EXP-300": "(drop)",
        "EXP-301": "(run later) but only if money allows it",
        "EXP-415": "(RUNNING-ID: 4, takes 24 h)",
    }, "only parenthesised trailing text on data rows counts as a note"


def test_render_table_appends_notes_without_widening_columns(tmp_path):
    rows = summarize.build_rows(REPO_ROOT, tmp_path)
    plain = summarize.render_table(rows)
    note = "(run later) depending on a very long explanation indeed"
    annotated = summarize.render_table(rows, {rows[0]["exp"]: note})
    target = rows[0]["exp"][4:]
    plain_row = [ln for ln in plain.splitlines() if ln.startswith(target)][0]
    annotated_row = [ln for ln in annotated.splitlines() if ln.startswith(target)][0]
    assert annotated_row == f"{plain_row} {note}"
    # a long note must not push the numeric columns of *other* rows apart
    others = [ln for ln in plain.splitlines() if not ln.startswith(target)]
    assert others == [ln for ln in annotated.splitlines() if not ln.startswith(target)]


def test_notes_survive_a_regeneration_round_trip(tmp_path):
    out = tmp_path / "out"
    argv = ["--out-dir", str(out), "--results-dir", str(tmp_path / "results")]
    assert summarize.main(argv) == 0
    md_path = out / "final_matrix_summary.md"

    # annotate two rows by hand, exactly as a human would
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    touched = 0
    for i, line in enumerate(lines):
        if line.startswith("300 ") or line.startswith("301 "):
            lines[i] = line + " (drop) because reasons"
            touched += 1
    assert touched == 2
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    assert summarize.main(argv) == 0                     # regenerate over the top
    notes = summarize.extract_notes(md_path.read_text(encoding="utf-8"))
    assert notes == {"EXP-300": "(drop) because reasons",
                     "EXP-301": "(drop) because reasons"}
    assert summarize.main(argv) == 0                     # and again: still stable
    assert summarize.extract_notes(md_path.read_text(encoding="utf-8")) == notes


def test_total_gpu_hours_sums_only_known_durations():
    rows = [{"duration_s": 3600}, {"duration_s": None}, {"duration_s": 1800}]
    assert summarize.total_hours(rows) == 1.5


def test_main_writes_both_files_and_is_byte_stable(tmp_path):
    out = tmp_path / "out"
    argv = ["--out-dir", str(out), "--results-dir", str(tmp_path / "results")]
    assert summarize.main(argv) == 0
    jsonl = out / "final_matrix_summary.jsonl"
    md = out / "final_matrix_summary.md"
    assert len(jsonl.read_text(encoding="utf-8").strip().splitlines()) == 190
    first = jsonl.read_bytes()
    assert summarize.main(argv) == 0
    assert jsonl.read_bytes() == first          # JSONL is byte-stable
    assert "190 non-attention cells" in md.read_text(encoding="utf-8")
    # every JSONL line parses and carries the full key set
    for line in jsonl.read_text(encoding="utf-8").strip().splitlines():
        assert list(json.loads(line)) == summarize.ROW_KEYS


# ── Task 5: against the real repository ───────────────────────────────────────


def test_real_repo_run_matches_jsonl_and_table_row_counts(tmp_path):
    """The committed results/ tree must produce 190 consistent rows."""
    out = tmp_path / "out"
    assert summarize.main(["--out-dir", str(out)]) == 0
    lines = (out / "final_matrix_summary.jsonl").read_text(encoding="utf-8").strip().splitlines()
    rows = [json.loads(ln) for ln in lines]
    assert len(rows) == 190
    md = (out / "final_matrix_summary.md").read_text(encoding="utf-8")
    table = md.split("```")[1].strip().splitlines()
    assert len([ln for ln in table[1:] if ln.strip()]) == 190
    # Every done cell has a filtered-exact delta and a result file; no cell is mismatched.
    done = [r for r in rows if r["status"] == "done"]
    assert done, "expected at least one finished cell in the committed results/"
    assert all(r["flt_exact_delta"] is not None for r in done)
    assert all(r["result_json"] for r in done)
    assert [r["exp"] for r in rows if r["mismatch"]] == []
    assert [r["exp"] for r in rows if r["meta_error"]] == []


def test_main_console_output_is_ascii_encodable(tmp_path, capsys):
    """The Windows console is cp1252; non-ASCII in stdout raises UnicodeEncodeError."""
    assert summarize.main(["--out-dir", str(tmp_path), "--print"]) == 0
    out = capsys.readouterr().out
    out.encode("cp1252")   # raises UnicodeEncodeError if any middot/arrow slipped in


def test_include_attention_flag_writes_220_rows(tmp_path):
    out = tmp_path / "out"
    assert summarize.main(["--out-dir", str(out), "--include-attention"]) == 0
    lines = (out / "final_matrix_summary.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 220
