"""Benchmark validator.

Modes:
1) repo   -> uses repository extraction logic (load_benchmark/extract_*)
2) server -> checks Hugging Face dataset server field structure only
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

def _validate_repo_mode(path: Path, sample_size: int, seed: int) -> dict:
    """Validate one benchmark config against loaded dataset samples."""
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from sae_experiments.benchmarks import (
        extract_answer,
        extract_choices,
        extract_question,
        format_prompt,
        load_benchmark,
    )
    from sae_experiments.config import BenchmarkConfig

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = BenchmarkConfig(**raw)

    # Local synthetic benchmark can be slow to generate; still supported.
    rows = load_benchmark(config)
    if not rows:
        raise ValueError("No rows loaded")

    rng = random.Random(seed)
    k = min(sample_size, len(rows))
    sample_rows = rng.sample(rows, k) if len(rows) > k else rows

    choice_len_mismatch = 0
    empty_question = 0
    empty_choices = 0
    answer_fail = 0
    prompt_fail = 0

    for row in sample_rows:
        try:
            question = extract_question(row, config)
            choices = extract_choices(row, config)
            _letter, idx = extract_answer(row, config)
            _prompt = format_prompt(row, config)

            if not str(question).strip():
                empty_question += 1
            if len(choices) == 0:
                empty_choices += 1
            if len(choices) != len(config.choices):
                choice_len_mismatch += 1
            if idx < 0 or idx >= len(config.choices):
                answer_fail += 1
        except Exception:
            # Keep as a compact counter rather than failing immediately
            prompt_fail += 1

    return {
        "name": path.name,
        "dataset_name": config.dataset_name,
        "rows_loaded": len(rows),
        "rows_checked": len(sample_rows),
        "empty_question": empty_question,
        "empty_choices": empty_choices,
        "choice_len_mismatch": choice_len_mismatch,
        "answer_fail": answer_fail,
        "prompt_fail": prompt_fail,
    }


def _pick_rows_dataset_config(dataset_name: str, preferred_config: str | None) -> str:
    info_url = "https://datasets-server.huggingface.co/info?dataset=" + urllib.parse.quote(
        dataset_name, safe=""
    )
    info = json.load(urllib.request.urlopen(info_url, timeout=30))
    configs = list(info.get("dataset_info", {}).keys())
    if preferred_config and preferred_config in configs:
        return preferred_config
    if configs:
        return configs[0]
    return preferred_config or "default"


def _validate_server_mode(path: Path, sample_size: int) -> dict:
    """Validate benchmark config against HF datasets-server response shape."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = raw["dataset_name"]
    if name == "letter_counting" or name.startswith("local/"):
        return {
            "name": path.name,
            "dataset_name": name,
            "rows_loaded": 0,
            "rows_checked": 0,
            "empty_question": 0,
            "empty_choices": 0,
            "choice_len_mismatch": 0,
            "answer_fail": 0,
            "prompt_fail": 0,
            "note": "local benchmark skipped in server mode",
        }

    dataset_config = _pick_rows_dataset_config(name, raw.get("dataset_config"))
    split = raw.get("dataset_split_test", "test")
    rows_url = (
        "https://datasets-server.huggingface.co/rows?dataset="
        + urllib.parse.quote(name, safe="")
        + "&config="
        + urllib.parse.quote(dataset_config, safe="")
        + "&split="
        + urllib.parse.quote(split, safe="")
        + "&offset=0&length="
        + str(sample_size)
    )
    rows_payload = json.load(urllib.request.urlopen(rows_url, timeout=50))
    rows = [r.get("row", {}) for r in rows_payload.get("rows", [])]
    if not rows:
        raise ValueError("No rows returned by datasets-server")

    fm = raw.get("field_mapping", {})
    question_fields = fm.get("question_fields") or []
    question_field = fm.get("question")
    choices_format = fm.get("choices_format")
    choices_field = fm.get("choices_field")
    choices_fields = fm.get("choices_fields") or []
    answer_field = fm.get("answer")
    label_count = len(raw.get("choices", []))

    empty_question = 0
    empty_choices = 0
    choice_len_mismatch = 0
    answer_fail = 0
    prompt_fail = 0

    for row in rows:
        # Question presence
        if question_fields:
            text = "\n\n".join(str(row.get(f, "")).strip() for f in question_fields).strip()
            if not text:
                empty_question += 1
        else:
            if not str(row.get(question_field, "")).strip():
                empty_question += 1

        # Choice shape (best-effort)
        choices_len = 0
        try:
            if choices_format == "dict_of_lists":
                choices_len = len(row.get("choices", {}).get("text", []))
            elif choices_format == "list_of_dicts":
                choices_len = len(row.get("choices", []))
            elif choices_format == "list":
                choices_len = len(row.get(choices_field or "endings", []))
            elif choices_format == "separate_fields":
                choices_len = len([row.get(f) for f in choices_fields if f in row])
            elif choices_format == "digit_list":
                choices_len = len(row.get("choices", ["0", "1", "2", "3", "4", "5"]))
            elif choices_format == "fixed":
                choices_len = len(fm.get("fixed_choices", raw.get("choices", [])))
            elif choices_format == "stringified_list":
                field = choices_field or "choices"
                val = row.get(field, [])
                if isinstance(val, list):
                    choices_len = len(val)
                elif isinstance(val, str):
                    try:
                        parsed = ast.literal_eval(val)
                        choices_len = len(parsed) if isinstance(parsed, list) else 0
                    except Exception:
                        choices_len = 0
            else:
                choices_len = 0
        except Exception:
            choices_len = 0
            prompt_fail += 1

        if choices_len == 0:
            empty_choices += 1
        if label_count and choices_len and choices_len != label_count:
            choice_len_mismatch += 1

        if answer_field not in row:
            answer_fail += 1

    return {
        "name": path.name,
        "dataset_name": name,
        "rows_loaded": len(rows),
        "rows_checked": len(rows),
        "empty_question": empty_question,
        "empty_choices": empty_choices,
        "choice_len_mismatch": choice_len_mismatch,
        "answer_fail": answer_fail,
        "prompt_fail": prompt_fail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate all benchmark configs against real dataset rows."
    )
    parser.add_argument(
        "--mode",
        choices=["repo", "server"],
        default="server",
        help="repo=full extraction checks, server=HF dataset API shape checks",
    )
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("configs/benchmarks"),
        help="Directory with benchmark YAML files",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Rows to validate per benchmark",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for row sampling",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Return non-zero exit code when any warning counters are non-zero",
    )
    args = parser.parse_args()

    paths = sorted(args.benchmarks_dir.glob("*.yaml"))
    if not paths:
        raise SystemExit(f"No benchmark YAML files found in: {args.benchmarks_dir}")

    print(f"Validating {len(paths)} benchmark configs from {args.benchmarks_dir}")
    print("-" * 100)

    had_error = False
    had_warning = False

    for path in paths:
        try:
            if args.mode == "repo":
                result = _validate_repo_mode(path, args.sample_size, args.seed)
            else:
                result = _validate_server_mode(path, args.sample_size)
            warn_count = (
                result["empty_question"]
                + result["empty_choices"]
                + result["choice_len_mismatch"]
                + result["answer_fail"]
                + result["prompt_fail"]
            )
            status = "OK" if warn_count == 0 else "WARN"
            if warn_count > 0:
                had_warning = True
            note = result.get("note")
            print(
                f"[{status}] {result['name']:<32} "
                f"rows={result['rows_loaded']:<6} checked={result['rows_checked']:<4} "
                f"empty_q={result['empty_question']:<3} empty_c={result['empty_choices']:<3} "
                f"len_mismatch={result['choice_len_mismatch']:<3} ans_fail={result['answer_fail']:<3} "
                f"prompt_fail={result['prompt_fail']:<3}"
                + (f" note={note}" if note else "")
            )
        except Exception as e:
            had_error = True
            print(f"[ERROR] {path.name:<32} {e}")

    print("-" * 100)
    if had_error:
        raise SystemExit(2)
    if args.fail_on_warning and had_warning:
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()

