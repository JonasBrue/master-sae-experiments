"""Tests for run_experiments_final.py — the 220-cell final-matrix runner."""

import importlib.util
import sys
from pathlib import Path

# run_experiments_final.py lives at the repo root, not under src/. Load it
# directly with importlib so the tests don't depend on the runner being on
# sys.path.
_RUNNER_PATH = Path(__file__).parent.parent / "run_experiments_final.py"
_spec = importlib.util.spec_from_file_location("run_experiments_final", _RUNNER_PATH)
_runner = importlib.util.module_from_spec(_spec)
sys.modules["run_experiments_final"] = _runner
_spec.loader.exec_module(_runner)

Experiment = _runner.Experiment


def test_to_cli_args_includes_cache_dir_pointing_at_output_dir():
    """Every experiment emits --cache-dir results/EXP-N so the cache lands in the result folder."""
    exp = Experiment(
        name="EXP-300: jbloom_gemma_2b L17 + CSQA",
        sae="configs/saes/jbloom_gemma_2b.yaml",
        layer=17,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        batch_size=1,
    )
    args = exp.to_cli_args()
    assert "--cache-dir" in args, "expected --cache-dir in emitted CLI args"
    cache_idx = args.index("--cache-dir")
    output_idx = args.index("--output-dir")
    # The value following --cache-dir must equal the value following --output-dir.
    assert args[cache_idx + 1] == args[output_idx + 1] == "results/EXP-300"


def test_to_cli_args_always_force_reruns_screening():
    """No experiment in this runner should ever reuse a cache file."""
    exp = Experiment(
        name="EXP-300: foo",
        sae="configs/saes/foo.yaml",
        layer=10,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
    )
    args = exp.to_cli_args()
    assert "--force-rerun-screening" in args


def test_to_cli_args_preserves_existing_flags():
    """Adding cache + force-rerun flags must not drop any of the pre-existing flags."""
    exp = Experiment(
        name="EXP-301: foo",
        sae="configs/saes/foo.yaml",
        layer=5,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        batch_size=1,
        overrides=["causal.min_p_correct=0.15"],
        generalisation=["configs/benchmarks/commonsenseqa.yaml"],
    )
    args = exp.to_cli_args()
    for flag in ["--sae", "--layer", "--benchmark", "--method", "--output-dir",
                 "--batch-size", "--set", "--generalisation"]:
        assert flag in args, f"expected {flag} in emitted CLI args"


def test_archive_if_nonempty_returns_none_for_missing_dir(tmp_path):
    """Missing directory: no archive happens, returns None."""
    target = tmp_path / "results" / "EXP-999"
    archived = _runner._archive_if_nonempty(target)
    assert archived is None
    assert not target.exists()


def test_archive_if_nonempty_returns_none_for_empty_dir(tmp_path):
    """Empty directory: no archive, returns None. The directory is left untouched."""
    target = tmp_path / "results" / "EXP-999"
    target.mkdir(parents=True)
    archived = _runner._archive_if_nonempty(target)
    assert archived is None
    assert target.exists()
    assert target.is_dir()


def test_archive_if_nonempty_renames_nonempty_dir(tmp_path):
    """Non-empty directory: renamed to EXP-N_archived_<ts>; original path is freed."""
    target = tmp_path / "results" / "EXP-999"
    target.mkdir(parents=True)
    (target / "stale.log").write_text("old run output")

    archived = _runner._archive_if_nonempty(target)
    assert archived is not None
    assert archived.exists()
    assert archived.is_dir()
    assert (archived / "stale.log").read_text() == "old run output"
    # Archive name encodes the original name + an archived_ tag + a timestamp.
    assert archived.name.startswith("EXP-999_archived_")
    # Original path is no longer occupied — caller can mkdir it fresh.
    assert not target.exists()
