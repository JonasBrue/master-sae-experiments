"""Pipeline class that replaces notebook globals and orchestrates the full experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmarks import get_answer_token_ids, load_benchmark
from .config import ExperimentConfig
from .hooks import HookManager
from .kfold import run_kfold_validation
from .models import get_layer_module, load_model_and_tokenizer
from .phases import (
    compute_baselines,
    run_pair_stability_check,
    run_phase1_freq_diff,
    run_phase2_causal_screening,
    run_phase3_grid_search,
    run_phase4_full_reporting,
    run_phase4_adaptive_reporting,
    run_screening_and_split,
)
from .plotting import generate_all_plots
from .reporting import build_summary, print_final_summary, save_summary
from .sae import load_sae


class Pipeline:
    """Holds all state and orchestrates the experiment phases."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.cache_dir = config.cache_dir

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
        """Load model, SAE, benchmark, and register hooks."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        self.model, self.tokenizer, self.device = load_model_and_tokenizer(self.config.model)

        # Load SAE
        self.sae = load_sae(self.config.sae, self.device)

        # Register hooks
        layer_module = get_layer_module(
            self.model, self.config.sae.layer, self.config.sae.hook_point,
            expected_n_layers=self.config.model.n_layers,
        )
        self.hook_mgr = HookManager(layer_module)
        self.hook_mgr.register_cache_hook()

        # Load benchmark
        self.all_questions = load_benchmark(self.config.benchmark)

        # Get answer token IDs
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

    def run(
        self,
        screening_only: bool = False,
        generalisation_benchmarks: Optional[List[str]] = None,
    ) -> None:
        """Run the full pipeline or screening only."""
        self.setup()
        bs = self.config.batch_size

        # Screening and data split — cache_dir defaults to "cache/" (shared),
        # but is overridden per-experiment by run_experiments_final.py.
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

        if screening_only:
            print("\nScreening complete. Exiting (--screening-only).")
            return

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

        # Phase 1: freq_diff
        tracker, pos_candidates, neg_candidates = run_phase1_freq_diff(
            train_data,
            **self._predict_kwargs(),
            causal_config=self.config.causal,
            batch_size=bs,
        )

        dynamic_min_occurrences = max(3, int(len(train_data) * 0.01))

        # Phase 2: Causal screening (one pass per entry in causal.causal_test_strength)
        df_pos, df_neg, baseline_val_acc = run_phase2_causal_screening(
            pos_candidates, neg_candidates, val_data,
            **self._predict_kwargs(),
            causal_config=self.config.causal,
            batch_size=bs,
        )

        # Phase 2.5: Pairwise combination stability (combination_stability_n_pairs
        # forward passes over val_data per strength, split across pos_pos / neg_neg
        # / pos_neg pair types).
        df_pair_stability = run_pair_stability_check(
            df_pos, df_neg, val_data, baseline_val_acc,
            causal_config=self.config.causal,
            **self._predict_kwargs(),
            batch_size=bs,
        )

        # Compute baselines for grid_eval (Phase 3 selection)
        baseline_grid_eval_results, baseline_grid_eval_acc, _, _ = compute_baselines(
            grid_eval_data,
            **self._predict_kwargs(),
            batch_size=bs,
            label="grid_eval",
        )

        # Phase 3: Grid search on grid_eval (disjoint from holdout)
        df_grid = run_phase3_grid_search(
            df_pos, df_neg,
            grid_eval_data, baseline_grid_eval_results, baseline_grid_eval_acc,
            grid_config=self.config.grid,
            **self._predict_kwargs(),
            batch_size=bs,
        )

        # Select best config by grid_eval delta. An empty grid means Phase 1/2 found
        # no candidates, so there is nothing to report — exit non-zero rather than
        # returning, which the batch runner would otherwise record as SUCCESS.
        if df_grid is None or df_grid.empty:
            self.cleanup()
            raise RuntimeError(
                "No grid search results. Aborting pipeline. Phase 1/2 produced no "
                "feature candidates, so no steering config could be selected."
            )

        best_row = df_grid.loc[df_grid["delta_grid_eval"].idxmax()]
        best_config = {
            "top_k_pos": int(best_row["top_k_pos"]),
            "top_k_neg": int(best_row["top_k_neg"]),
            "strength": float(best_row["strength"]),
            "ranking_criterion": best_row["ranking_criterion"],
            "weighting_mode": best_row["weighting_mode"],
        }

        print(f"\nBest config (selected by grid_eval delta): {best_config}")
        print(f"  Delta grid_eval: {best_row['delta_grid_eval']:+.1%}")

        # Compute baselines for holdout (Phase 4 reporting)
        baseline_holdout_results, baseline_holdout_acc, baseline_holdout_filtered_results, baseline_holdout_filtered_acc = (
            compute_baselines(
                holdout_data,
                **self._predict_kwargs(),
                batch_size=bs,
                label="holdout",
                filtered_subset=holdout_filtered,
            )
        )

        # Phase 4: Full reporting on holdout (never seen by grid search)
        if self.config.causal.adaptive.enabled:
            phase4_result = run_phase4_adaptive_reporting(
                best_config, df_pos, df_neg,
                holdout_filtered, holdout_data,
                baseline_holdout_filtered_results, baseline_holdout_filtered_acc,
                baseline_holdout_results, baseline_holdout_acc,
                adaptive_config=self.config.causal.adaptive,
                **self._predict_kwargs(),
                batch_size=bs,
            )
        else:
            phase4_result = run_phase4_full_reporting(
                best_config, df_pos, df_neg,
                holdout_filtered, holdout_data,
                baseline_holdout_filtered_results, baseline_holdout_filtered_acc,
                baseline_holdout_results, baseline_holdout_acc,
                **self._predict_kwargs(),
                batch_size=bs,
            )

        full_reporting_results = phase4_result["full_reporting_results"]
        difficulty_analysis = phase4_result["difficulty_analysis"]
        best_intervention_vec = phase4_result["best_intervention_vec"]
        n_features_in_vec = phase4_result["n_features_in_vec"]

        # K-fold validation on holdout (clean — never used for selection)
        kfold_results = run_kfold_validation(
            holdout_data, best_intervention_vec,
            causal_config=self.config.causal,
            **self._predict_kwargs(),
            batch_size=bs,
        )

        # Phase 5: Cross-benchmark generalisation (optional)
        generalisation_results = None
        if generalisation_benchmarks:
            import gc

            from .generalisation import run_generalisation

            # Free intermediate tensors before loading new benchmarks
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gen_vec = phase4_result["approximated_hidden_vec"]
            generalisation_results = run_generalisation(
                gen_vec,
                generalisation_benchmarks,
                model=self.model,
                tokenizer=self.tokenizer,
                hook_mgr=self.hook_mgr,
                sae=self.sae,
                device=self.device,
                batch_size=bs,
            )

        # Plotting — wrapped so a plot failure never prevents saving JSON results
        try:
            plot_file = generate_all_plots(
                full_reporting_results=full_reporting_results,
                df_grid=df_grid,
                best_config=best_config,
                screening_data=screening_data,
                filtered_data=filtered_data,
                causal_config=self.config.causal,
                experiment_config=self.config,
                output_dir=self.output_dir,
            )
        except Exception as e:
            print(f"WARNING: Plot generation failed: {e}")
            plot_file = None

        # Build and save summary — n_causal_* counts unique features causal at
        # ANY tested strength, so the headline number isn't multiplied by the
        # number of strengths swept in Phase 2.
        if len(df_pos) > 0:
            n_causal_pos = int(df_pos[df_pos["is_causal"]]["feature_id"].nunique())
        else:
            n_causal_pos = 0
        if len(df_neg) > 0:
            n_causal_neg = int(df_neg[df_neg["is_causal"]]["feature_id"].nunique())
        else:
            n_causal_neg = 0

        grid_config_dict = self.config.grid.model_dump()

        summary = build_summary(
            experiment_config=self.config,
            sae_d_hidden=self.sae.d_hidden,
            sae_top_k=getattr(self.sae, "top_k", None),
            all_questions_count=len(self.all_questions),
            filtered_data_count=len(filtered_data),
            train_data_count=len(train_data),
            val_data_count=len(val_data),
            test_filtered_count=len(test_data_filtered),
            test_full_count=len(test_data_full),
            grid_eval_count=len(grid_eval_data),
            holdout_count=len(holdout_data),
            holdout_filtered_count=len(holdout_filtered),
            baseline_filtered_acc=baseline_holdout_filtered_acc,
            baseline_full_acc=baseline_holdout_acc,
            dynamic_min_occurrences=dynamic_min_occurrences,
            df_pos=df_pos,
            df_neg=df_neg,
            phase1_pos_candidates=pos_candidates,
            phase1_neg_candidates=neg_candidates,
            n_causal_pos=n_causal_pos,
            n_causal_neg=n_causal_neg,
            df_pair_stability=df_pair_stability,
            grid_config_dict=grid_config_dict,
            n_combinations=len(df_grid),
            best_config=best_config,
            n_features_in_vec=n_features_in_vec,
            full_reporting_results=full_reporting_results,
            difficulty_analysis=difficulty_analysis,
            kfold_results=kfold_results,
            df_grid=df_grid,
            generalisation_results=generalisation_results,
            answer_diagnostics=answer_diagnostics,
        )

        summary_file = save_summary(
            summary, self.output_dir,
            self.config.sae, self.config.benchmark,
        )

        # Final summary
        print_final_summary(
            self.config,
            full_reporting_results,
            kfold_results,
            n_features_in_vec,
            best_config,
            summary_file,
            plot_file,
        )

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
