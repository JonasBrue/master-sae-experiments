"""Cross-benchmark generalisation test (Phase 5).

Evaluates a precomputed hidden-space steering vector on benchmarks that
were never used during feature discovery, causal screening, or config
selection.  Tests whether the intervention generalises beyond the
training benchmark.
"""

from __future__ import annotations

import gc
from typing import Any, Callable, Dict, List

import torch
import torch.nn as nn
import yaml
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmarks import get_answer_token_ids, load_benchmark
from .config import BenchmarkConfig
from .fast_inference import (
    batched_intervention_predictions,
    batched_logit_predictions,
    make_approx_intervention_fn,
)
from .hooks import HookManager


def _load_benchmark_config(benchmark_path: str) -> BenchmarkConfig:
    """Load a BenchmarkConfig from a YAML file."""
    with open(benchmark_path) as f:
        data = yaml.safe_load(f)
    return BenchmarkConfig(**data)


def _evaluate_single_benchmark(
    benchmark_config: BenchmarkConfig,
    intervention_fn: Callable,
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hook_mgr: HookManager,
    sae: nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Evaluate baseline and intervention on a single benchmark.

    Returns a dict with benchmark metadata, baseline/intervention accuracy,
    delta, n_improved, n_worsened, and binomial p-value.
    """
    questions = load_benchmark(benchmark_config)

    answer_token_ids = get_answer_token_ids(benchmark_config.choices, tokenizer)
    answer_token_id_list = [answer_token_ids[c] for c in benchmark_config.choices]

    # Baseline predictions
    baseline_results = batched_logit_predictions(
        questions,
        batch_size=batch_size,
        model=model,
        tokenizer=tokenizer,
        config=benchmark_config,
        hook_mgr=hook_mgr,
        sae=sae,
        answer_token_id_list=answer_token_id_list,
        device=device,
        extract_features=False,
        desc=f"Generalisation Baseline: {benchmark_config.name}",
    )

    baseline_correct = sum(r["correct"] for r in baseline_results)
    baseline_acc = baseline_correct / len(questions)

    # Intervention predictions
    int_results = batched_intervention_predictions(
        questions,
        intervention_fn,
        batch_size=batch_size,
        model=model,
        tokenizer=tokenizer,
        config=benchmark_config,
        hook_mgr=hook_mgr,
        sae=sae,
        answer_token_id_list=answer_token_id_list,
        device=device,
        desc=f"Generalisation Intervention: {benchmark_config.name}",
    )

    int_correct = sum(r["correct"] for r in int_results)
    int_acc = int_correct / len(questions)
    delta = int_acc - baseline_acc

    n_improved = sum(
        1 for i, r in enumerate(int_results)
        if not baseline_results[i]["correct"] and r["correct"]
    )
    n_worsened = sum(
        1 for i, r in enumerate(int_results)
        if baseline_results[i]["correct"] and not r["correct"]
    )

    binomial_p = None
    if n_improved + n_worsened > 0:
        binomial_p = binomtest(
            n_improved, n_improved + n_worsened, 0.5, alternative="greater",
        ).pvalue

    return {
        "benchmark": benchmark_config.name,
        "dataset": benchmark_config.dataset_name,
        "n_questions": len(questions),
        "n_choices": len(benchmark_config.choices),
        "baseline_acc": float(baseline_acc),
        "intervention_acc": float(int_acc),
        "delta": float(delta),
        "n_improved": int(n_improved),
        "n_worsened": int(n_worsened),
        "binomial_p": float(binomial_p) if binomial_p is not None else None,
        "significant": bool(binomial_p is not None and binomial_p < 0.05),
    }


def run_generalisation(
    intervention_hidden_vec: torch.Tensor,
    benchmark_paths: List[str],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hook_mgr: HookManager,
    sae: nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """Run cross-benchmark generalisation test.

    Evaluates a precomputed hidden-space steering vector on benchmarks
    that were never used during feature discovery, causal screening,
    or config selection.

    Args:
        intervention_hidden_vec: Hidden-space vector [d_model], already
            scaled by the best strength.
        benchmark_paths: Paths to benchmark YAML config files.
        model: The language model.
        tokenizer: The tokenizer.
        hook_mgr: Hook manager with cache hook on the target layer.
        sae: The SAE module (required by batched prediction signatures).
        device: Torch device.
        batch_size: Batch size for inference.

    Returns:
        JSON-serializable dict with per-benchmark results and summary.
    """
    print("=" * 80)
    print("PHASE 5: CROSS-BENCHMARK GENERALISATION TEST")
    print(f"  Intervention vector norm: {intervention_hidden_vec.norm().item():.4f}")
    print(f"  Testing on {len(benchmark_paths)} benchmark(s)")
    print("=" * 80)

    intervention_fn = make_approx_intervention_fn(intervention_hidden_vec)

    results: List[Dict[str, Any]] = []
    for path in benchmark_paths:
        bench_config = _load_benchmark_config(str(path))
        print(f"\n  Evaluating: {bench_config.name} ({path})")

        # Free GPU memory before each benchmark to avoid OOM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Retry with halved batch size on OOM
        current_bs = batch_size
        result = None
        while True:
            try:
                result = _evaluate_single_benchmark(
                    bench_config,
                    intervention_fn,
                    model=model,
                    tokenizer=tokenizer,
                    hook_mgr=hook_mgr,
                    sae=sae,
                    device=device,
                    batch_size=current_bs,
                )
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if current_bs <= 1:
                    raise  # Can't reduce further, propagate error
                current_bs = current_bs // 2
                print(f"    OOM, retrying with batch_size={current_bs}")

        results.append(result)

        sig = "*" if result["significant"] else ""
        p_str = f"{result['binomial_p']:.4f}" if result["binomial_p"] is not None else "N/A"
        print(f"    N={result['n_questions']}, "
              f"Baseline={result['baseline_acc']*100:.1f}%, "
              f"Intervention={result['intervention_acc']*100:.1f}%, "
              f"Delta={result['delta']*100:+.1f}%, p={p_str}{sig}")

    # Summary table
    n_positive = sum(1 for r in results if r["delta"] > 0)
    n_significant = sum(1 for r in results if r["significant"])
    avg_delta = sum(r["delta"] for r in results) / len(results) if results else 0.0

    print(f"\n{'─' * 80}")
    print(f"{'Benchmark':<25} {'N':>6} {'Baseline':>10} {'Interv.':>10} "
          f"{'Delta':>8} {'p-value':>10}")
    print(f"{'─' * 80}")

    for r in results:
        sig = "*" if r["significant"] else ""
        p_str = f"{r['binomial_p']:.4f}" if r["binomial_p"] is not None else "N/A"
        print(f"{r['benchmark']:<25} {r['n_questions']:>6} "
              f"{r['baseline_acc']*100:>9.1f}% "
              f"{r['intervention_acc']*100:>9.1f}% "
              f"{r['delta']*100:>+7.1f}% {p_str:>10}{sig}")

    print(f"\n  Avg delta: {avg_delta*100:+.2f}%")
    print(f"  Positive: {n_positive}/{len(results)}")
    print(f"  Significant: {n_significant}/{len(results)}")
    print("=" * 80)

    return {
        "benchmarks": results,
        "summary": {
            "n_benchmarks": len(results),
            "n_positive": n_positive,
            "n_significant": n_significant,
            "avg_delta": float(avg_delta),
        },
    }
