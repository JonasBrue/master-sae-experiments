"""CAA (Contrastive Activation Addition) pipeline.

A parallel pipeline to the SAE-based feature steering pipeline.
Uses mean-difference in the residual stream as the steering vector,
without any SAE decomposition. Serves as a baseline comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmarks import get_answer_token_ids, load_benchmark
from .caa import compute_caa_vector, run_caa_strength_sweep
from .config import ExperimentConfig
from .fast_inference import make_approx_intervention_fn
from .hooks import HookManager
from .kfold import run_kfold_validation
from .models import get_layer_module, load_model_and_tokenizer
from .phases import (
    compute_baselines,
    run_phase4_reporting_with_vector,
    run_screening_and_split,
)
from .plotting import generate_all_plots
from .reporting import build_caa_summary, save_caa_summary
from .sae import load_sae


# Default strength values for the CAA sweep.
# Wider range than SAE (0.5-1.0) because the CAA vector lives in
# hidden space and has a different norm scale.
DEFAULT_CAA_STRENGTHS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


class CaaPipeline:
    """Orchestrates the CAA baseline experiment.

    Mirrors the structure of Pipeline but replaces:
    - Phase 1 (freq_diff) with CAA vector computation
    - Phase 2 (causal screening) is skipped entirely
    - Phase 3 (grid search) with a simple strength sweep
    """

    def __init__(self, config: ExperimentConfig, strengths: Optional[List[float]] = None):
        self.config = config
        self.output_dir = config.output_dir
        self.cache_dir = config.cache_dir
        self.strengths = strengths or DEFAULT_CAA_STRENGTHS

        # Populated by setup()
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: Optional[torch.device] = None
        self.sae: Optional[nn.Module] = None
        self.hook_mgr: Optional[HookManager] = None
        self.answer_token_ids: Optional[Dict[str, int]] = None
        self.answer_token_id_list: Optional[List[int]] = None
        self.all_questions: Optional[List[Dict]] = None

    def setup(self) -> None:
        """Load model, SAE, benchmark, and register hooks.

        SAE is still loaded because:
        1. The cache hook must be registered on the correct layer (sae.layer)
        2. The SAE object is required by batched_intervention_predictions signature
        """
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.model, self.tokenizer, self.device = load_model_and_tokenizer(self.config.model)
        self.sae = load_sae(self.config.sae, self.device)

        layer_module = get_layer_module(
            self.model, self.config.sae.layer, self.config.sae.hook_point,
            expected_n_layers=self.config.model.n_layers,
        )
        self.hook_mgr = HookManager(layer_module)
        self.hook_mgr.register_cache_hook()

        self.all_questions = load_benchmark(self.config.benchmark)

        self.answer_token_ids = get_answer_token_ids(
            self.config.benchmark.choices, self.tokenizer
        )
        self.answer_token_id_list = [
            self.answer_token_ids[c] for c in self.config.benchmark.choices
        ]

        print(f"\nAnswer token IDs: {self.answer_token_ids}")

    def _predict_kwargs(self) -> Dict:
        return dict(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.benchmark,
            hook_mgr=self.hook_mgr,
            sae=self.sae,
            answer_token_id_list=self.answer_token_id_list,
            device=self.device,
        )

    def run(self, generalisation_benchmarks: Optional[List[str]] = None) -> None:
        """Run the full CAA pipeline."""
        self.setup()
        bs = self.config.batch_size

        print("=" * 80)
        print("CAA BASELINE PIPELINE")
        print(f"  Model: {self.config.model.short_name}")
        print(f"  Layer: {self.config.sae.layer}")
        print(f"  Benchmark: {self.config.benchmark.name}")
        print(f"  Strengths: {self.strengths}")
        print("=" * 80)

        # ── Screening and data split — cache_dir defaults to "cache/" (shared),
        # but is overridden per-experiment by run_experiments_final.py. ──
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        split_result = run_screening_and_split(
            self.all_questions,
            **self._predict_kwargs(),
            model_config=self.config.model,
            causal_config=self.config.causal,
            output_dir=self.cache_dir,
            force_rerun=self.config.force_rerun_screening,
            batch_size=bs,
        )

        screening_data = split_result["screening_data"]
        filtered_data = split_result["filtered_data"]
        train_data = split_result["train_data"]
        val_data = split_result["val_data"]
        test_data_filtered = split_result["test_data_filtered"]
        test_data_full = split_result["test_data_full"]
        grid_eval_data = split_result["grid_eval_data"]
        holdout_data = split_result["holdout_data"]
        holdout_filtered = split_result["holdout_filtered"]
        answer_diagnostics = split_result["answer_diagnostics"]

        # ── Compute baselines for grid_eval (strength sweep selection) ──
        baseline_grid_eval_results, baseline_grid_eval_acc, _, _ = compute_baselines(
            grid_eval_data,
            **self._predict_kwargs(),
            batch_size=bs,
            label="grid_eval",
        )

        # ── Phase 1: Compute CAA vector ──
        caa_vec = compute_caa_vector(
            train_data,
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config.benchmark,
            hook_mgr=self.hook_mgr,
            answer_token_id_list=self.answer_token_id_list,
            device=self.device,
            batch_size=bs,
        )

        # ── Strength sweep on grid_eval (replaces grid search) ──
        df_sweep = run_caa_strength_sweep(
            caa_vec, self.strengths,
            grid_eval_data,
            baseline_grid_eval_results, baseline_grid_eval_acc,
            **self._predict_kwargs(),
            batch_size=bs,
        )

        # Select best strength
        if df_sweep is None or df_sweep.empty:
            print("\nNo sweep results. Aborting pipeline.")
            self.cleanup()
            return

        best_row = df_sweep.loc[df_sweep["delta"].idxmax()]
        best_strength = float(best_row["strength"])
        print(f"\nBest strength: {best_strength} (delta: {best_row['delta']*100:+.1f}%)")

        # ── Compute baselines for holdout (Phase 4 reporting) ──
        baseline_holdout_results, baseline_holdout_acc, baseline_holdout_filtered_results, baseline_holdout_filtered_acc = (
            compute_baselines(
                holdout_data,
                **self._predict_kwargs(),
                batch_size=bs,
                label="holdout",
                filtered_subset=holdout_filtered,
            )
        )

        # ── Phase 4: Full reporting on holdout (never seen by strength sweep) ──
        adaptive_config = self.config.causal.adaptive if self.config.causal.adaptive.enabled else None

        phase4_result = run_phase4_reporting_with_vector(
            caa_vec, best_strength,
            holdout_filtered, holdout_data,
            baseline_holdout_filtered_results, baseline_holdout_filtered_acc,
            baseline_holdout_results, baseline_holdout_acc,
            **self._predict_kwargs(),
            adaptive_config=adaptive_config,
            batch_size=bs,
            label="CAA",
        )

        full_reporting_results = phase4_result["full_reporting_results"]
        difficulty_analysis = phase4_result["difficulty_analysis"]
        best_intervention_vec = phase4_result["best_intervention_vec"]

        # ── K-fold validation on holdout (clean — never used for selection) ──
        kfold_intervention = make_approx_intervention_fn(best_intervention_vec)
        kfold_results = run_kfold_validation(
            holdout_data, kfold_intervention,
            causal_config=self.config.causal,
            **self._predict_kwargs(),
            batch_size=bs,
        )

        # ── Phase 5: Cross-benchmark generalisation (optional) ──
        generalisation_results = None
        if generalisation_benchmarks:
            from .generalisation import run_generalisation

            generalisation_results = run_generalisation(
                best_intervention_vec,
                generalisation_benchmarks,
                model=self.model,
                tokenizer=self.tokenizer,
                hook_mgr=self.hook_mgr,
                sae=self.sae,
                device=self.device,
                batch_size=bs,
            )

        # ── Plotting ──
        # Build a compatible df_grid for the plotting system
        # (CAA only has strength as a dimension)
        try:
            plot_file = generate_all_plots(
                full_reporting_results=full_reporting_results,
                df_grid=df_sweep,
                best_config={"strength": best_strength},
                screening_data=screening_data,
                filtered_data=filtered_data,
                causal_config=self.config.causal,
                experiment_config=self.config,
                output_dir=self.output_dir,
            )
        except Exception as e:
            print(f"WARNING: Plot generation failed: {e}")
            plot_file = None

        # ── Build and save summary ──
        summary = build_caa_summary(
            experiment_config=self.config,
            sae_layer=self.config.sae.layer,
            all_questions_count=len(self.all_questions),
            filtered_data_count=len(filtered_data),
            train_data_count=len(train_data),
            val_data_count=len(val_data),
            test_filtered_count=len(test_data_filtered),
            test_full_count=len(test_data_full),
            baseline_filtered_acc=baseline_holdout_filtered_acc,
            baseline_full_acc=baseline_holdout_acc,
            caa_vector_norm=float(caa_vec.norm().item()),
            best_strength=best_strength,
            strength_sweep_results=df_sweep,
            full_reporting_results=full_reporting_results,
            difficulty_analysis=difficulty_analysis,
            kfold_results=kfold_results,
            generalisation_results=generalisation_results,
            answer_diagnostics=answer_diagnostics,
        )

        summary_file = save_caa_summary(
            summary, self.output_dir,
            self.config.benchmark, self.config.sae.layer,
        )

        # ── Final summary ──
        self._print_final_summary(
            full_reporting_results, kfold_results,
            best_strength, caa_vec, summary_file, plot_file,
        )

    def _print_final_summary(
        self,
        full_reporting_results: Dict,
        kfold_results: Dict,
        best_strength: float,
        caa_vec: torch.Tensor,
        summary_file: str,
        plot_file: Optional[str],
    ) -> None:
        """Print the comprehensive final summary."""
        def _get_report(key):
            if key in full_reporting_results:
                return full_reporting_results[key]
            for suffix in ("_static", "_adaptive"):
                if key + suffix in full_reporting_results:
                    return full_reporting_results[key + suffix]
            return {"delta": 0.0, "binomial_p": None}

        fr_filtered = _get_report("filtered_exact")
        fr_full = _get_report("full_exact")

        kfold_p = kfold_results.get("binomial_p")
        kfold_sig = kfold_p is not None and kfold_p < 0.05
        n_positive_folds = kfold_results["n_positive_folds"]
        n_folds = kfold_results["n_folds"]
        mean_delta = kfold_results["mean_delta"]
        std_delta = kfold_results["std_delta"]

        print("=" * 80)
        print("CAA EXPERIMENT COMPLETE")
        print("=" * 80)

        print(f"\n  Method: Contrastive Activation Addition (CAA)")
        print(f"  Layer: {self.config.sae.layer}")
        print(f"  Benchmark: {self.config.benchmark.name}")
        print(f"  Model: {self.config.model.short_name}")
        print(f"\n  Best strength: {best_strength}")
        print(f"  CAA vector norm: {caa_vec.norm().item():.4f}")

        # Full reporting table
        print(f"\n{'─' * 80}")
        print(f"{'Report':<30} {'N':>6} {'Baseline':>10} {'Interv.':>10} {'Delta':>8} {'p-value':>10}")
        print(f"{'─' * 80}")

        for key, r in full_reporting_results.items():
            sig = "*" if r["binomial_p"] is not None and r["binomial_p"] < 0.05 else ""
            p_str = f"{r['binomial_p']:.4f}" if r["binomial_p"] is not None else "N/A"
            print(f"{r['name']:<30} {r['n_questions']:>6} {r['baseline_acc']*100:>9.1f}% "
                  f"{r['intervention_acc']*100:>9.1f}% {r['delta']*100:>+7.1f}% {p_str:>10}{sig}")

        # K-fold
        print(f"\n  K-Fold: mean delta = {mean_delta*100:+.2f}% +/- {std_delta*100:.2f}%")
        print(f"  Positive folds: {n_positive_folds}/{n_folds}")
        if kfold_p is not None:
            print(f"  Binomial p-value (one-sided): {kfold_p:.4f}")

        # Verdict
        positive_indicators = 0
        if fr_filtered["binomial_p"] is not None and fr_filtered["binomial_p"] < 0.05:
            positive_indicators += 1
        if fr_full["delta"] > 0:
            positive_indicators += 1
        if kfold_sig:
            positive_indicators += 1
        if n_positive_folds > n_folds // 2:
            positive_indicators += 1

        if positive_indicators >= 3:
            verdict = "SUCCESS: Robust improvement demonstrated"
        elif positive_indicators >= 2:
            verdict = "PARTIAL: Positive tendency, not fully robust"
        else:
            verdict = "NO CLEAR ADVANTAGE on this benchmark"

        print(f"\n  Verdict ({positive_indicators}/4 positive indicators): {verdict}")

        print(f"\n  Files saved:")
        print(f"    {summary_file}")
        if plot_file:
            print(f"    {plot_file}")
        print("=" * 80)

    def cleanup(self) -> None:
        """Remove hooks and free GPU resources.

        Drops references to model/sae/tokenizer/hook_mgr so the GC can reclaim
        them before torch.cuda.empty_cache(). Without releasing those references
        first, empty_cache() is effectively a no-op — cached blocks remain
        pinned by live tensors.

        The subprocess will exit shortly after cleanup() returns and the kernel
        tears down the CUDA context regardless, but explicit teardown is cheap
        insurance against future in-process callers and shorter inter-subprocess
        memory-release latency.
        """
        import gc

        if self.hook_mgr is not None:
            self.hook_mgr.remove_cache_hook()
        self.model = None
        self.sae = None
        self.tokenizer = None
        self.hook_mgr = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
