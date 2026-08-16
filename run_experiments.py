#!/usr/bin/env python3
"""
Batch experiment runner — runs experiments sequentially on the GPU.

Each experiment runs as a subprocess. When it finishes or fails, the process
dies and the OS reclaims all GPU memory automatically. Then the next one starts.

Usage:
    uv run python run_experiments.py                # run all pending experiments
    uv run python run_experiments.py --dry-run      # show what would run
    uv run python run_experiments.py --start-from 3 # skip first 2 experiments
    uv run python run_experiments.py --only 5       # run only experiment #5
    uv run python run_experiments.py --only 5,8,12  # run only experiments 5, 8, 12
    uv run python run_experiments.py --worker 1/3   # multi-GPU: this instance runs every 3rd experiment starting at 1
    uv run python run_experiments.py --worker 2/3   # multi-GPU: this instance runs every 3rd experiment starting at 2
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_TIMEOUT = 4 * 3600  # 4 hours per experiment

_EXP_ID_RE = re.compile(r"EXP-(\d+)")


def _extract_exp_id(name: str) -> str:
    """Extract 'EXP-123' identifier from experiment name, or return empty string."""
    m = _EXP_ID_RE.search(name)
    return m.group(0) if m else ""


@dataclass
class Experiment:
    name: str
    sae: str
    layer: int
    benchmark: str
    method: str = "sae"
    overrides: list[str] = field(default_factory=list)
    caa_strengths: str | None = None
    screening_only: bool = False
    batch_size: int | None = None
    generalisation: list[str] = field(default_factory=list)
    timeout: int = DEFAULT_TIMEOUT

    def to_cli_args(self) -> list[str]:
        exp_id = _extract_exp_id(self.name)
        output_dir = f"results/{exp_id}" if exp_id else "results"
        parts = [
            "--sae", self.sae,
            "--layer", str(self.layer),
            "--benchmark", self.benchmark,
            "--method", self.method,
            "--output-dir", output_dir,
        ]
        if self.screening_only:
            parts.append("--screening-only")
        if self.batch_size is not None:
            parts += ["--batch-size", str(self.batch_size)]
        if self.caa_strengths:
            parts += ["--caa-strengths", self.caa_strengths]
        if self.generalisation:
            parts += ["--generalisation"] + self.generalisation
        for ov in self.overrides:
            parts += ["--set", ov]
        return parts


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT QUEUE — Edit this list to add/remove experiments.
#
# Session 3: Follow up on positive results from EXP 52-88.
# Focus: Adaptive steering on best models, benchmark expansion,
#        fix failed experiments, optimize Full benchmark delta.
# ═══════════════════════════════════════════════════════════════════

# Common override patterns
_FILTER_STD = ["causal.min_p_correct=0.15", "causal.max_p_correct=0.45"]
_STRENGTH_EXTENDED = ["grid.strength=[0.5,1.0,1.5,2.0]"]
_TOPK_EXTENDED = ["grid.top_k_negative=[0,3,5,7]"]
_EXTENDED_GRID = [
    "grid.strength=[0.9,1.0,1.5,2.0,2.5]",
    "grid.top_k_positive=[5,7,10,15,20]",
    "grid.top_k_negative=[0,3,5,7,10]",
    'grid.ranking_criterion=["causal_delta"]',
    'grid.weighting_mode=["uniform","proportional"]',
    "causal.n_candidate_features_pos=75",
    "causal.n_candidate_features_neg=75",
]
_FILTER_WIDE = ["causal.min_p_correct=0.10", "causal.max_p_correct=0.55"]
_FILTER_NARROW = ["causal.min_p_correct=0.20", "causal.max_p_correct=0.40"]
_STRENGTH_FINE = [
    "grid.strength=[0.1,0.25,0.5,0.75,1.0]",
    "grid.top_k_positive=[3,5,7,10]",
    "grid.top_k_negative=[0,3]",
]

EXPERIMENTS: list[Experiment] = [
    # ═══════════════════════════════════════════════════════════════════
    # EXP-198: Llama-3.1-8B re-run (originally EXP-082)
    # AndyRDT BatchTopK K32, L15, ARC-Challenge (n=2291)
    # Original: +0.87% exact, +1.39% approx, K-fold +0.29%
    # NO FILTER: p_correct_range [0.0, 1.0] to match original
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-198: Llama-3.1-8B L15 + ARC-Challenge + GEN",
        sae="configs/saes/andyrdt_llama_k32.yaml",
        layer=15,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=["causal.min_p_correct=0.00", "causal.max_p_correct=1.00"],
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    # ══════════════════════════════════════════════════════════════════
    # EXP-199: Llama-3.2-1B re-run after batching fix
    # EleutherAI TopK K32 131k MLP, L15 (last layer)
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-199: Llama-3.2-1B MLP L15 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/eleutherai_llama32_1b_131k_mlp.yaml",
        layer=15,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    # ══════════════════════════════════════════════════════════════════
    # Session 6: Broad Cross-Model Survey — 2026-03-28
    #
    # Goal: Find at least one more model where the method works.
    # Strategy: Test many model+SAE combos with standard config.
    # - Filter: [0.15, 0.45] (proven optimal)
    # - Grid: default (no extended)
    # - Layers: ~40-55% depth + last layer (~95%)
    # - Generalisation on all experiments (cheap)
    # - Mix of proven benchmarks + new ones as base
    # ══════════════════════════════════════════════════════════════════

    # ══════════════════════════════════════════════════════════════════
    # Group A: Gemma-2b-IT — jbloom ReLU 16k, resid_post
    # 18 layers, only L12 available. Previously +31-41% filtered.
    # Re-run on current pipeline with generalisation.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-200: Gemma-2b-IT L12 + CommonsenseQA + GEN",
        sae="configs/saes/jbloom_gemma_2b_it.yaml",
        layer=12,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-201: Gemma-2b-IT L12 + ARC-Challenge + GEN",
        sae="configs/saes/jbloom_gemma_2b_it.yaml",
        layer=12,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-202: Gemma-2b-IT L12 + SciQ + GEN",
        sae="configs/saes/jbloom_gemma_2b_it.yaml",
        layer=12,
        benchmark="configs/benchmarks/sciq.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group B: Gemma-2b (base) — jbloom ReLU 16k, resid_post
    # 18 layers. L10 (56%), L12 (67%), L17 (94%). Layer sweep.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-203: Gemma-2b L10 + CommonsenseQA + GEN",
        sae="configs/saes/jbloom_gemma_2b.yaml",
        layer=10,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-204: Gemma-2b L12 + CommonsenseQA + GEN",
        sae="configs/saes/jbloom_gemma_2b.yaml",
        layer=12,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-205: Gemma-2b L17 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/jbloom_gemma_2b.yaml",
        layer=17,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-206: Gemma-2b L12 + ARC-Challenge + GEN",
        sae="configs/saes/jbloom_gemma_2b.yaml",
        layer=12,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group C: GPT2-Small — jbloom 128k BatchTopK, resid_post
    # 12 layers, L0-L11. L5 (42%), L6 (50%), L11 (92%).
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-207: GPT2-Small L5 + CommonsenseQA + GEN",
        sae="configs/saes/jbloom_gpt2_small_oai_v5.yaml",
        layer=5,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-208: GPT2-Small L6 + CommonsenseQA + GEN",
        sae="configs/saes/jbloom_gpt2_small_oai_v5.yaml",
        layer=6,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-209: GPT2-Small L11 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/jbloom_gpt2_small_oai_v5.yaml",
        layer=11,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-210: GPT2-Small L6 + ARC-Challenge + GEN",
        sae="configs/saes/jbloom_gpt2_small_oai_v5.yaml",
        layer=6,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-211: GPT2-Small L6 + MMLU + GEN [NEW BENCH]",
        sae="configs/saes/jbloom_gpt2_small_oai_v5.yaml",
        layer=6,
        benchmark="configs/benchmarks/mmlu.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/arc_challenge.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group D: Pythia-410M — EleutherAI 65k MLP BatchTopK
    # 24 layers. L10 (42%), L12 (50%), L23 (96%). MLP hook!
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-212: Pythia-410M MLP L10 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_pythia_410m_65k_mlp.yaml",
        layer=10,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-213: Pythia-410M MLP L12 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_pythia_410m_65k_mlp.yaml",
        layer=12,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-214: Pythia-410M MLP L23 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/eleutherai_pythia_410m_65k_mlp.yaml",
        layer=23,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-215: Pythia-410M MLP L12 + ARC-Challenge + GEN",
        sae="configs/saes/eleutherai_pythia_410m_65k_mlp.yaml",
        layer=12,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-216: Pythia-410M MLP L12 + MedMCQA + GEN [NEW BENCH]",
        sae="configs/saes/eleutherai_pythia_410m_65k_mlp.yaml",
        layer=12,
        benchmark="configs/benchmarks/medmcqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/arc_challenge.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group E: Llama-3-8B-Instruct — EleutherAI 32x resid BatchTopK
    # 32 layers, K=192. L13 (41%), L16 (50%), L30 (94%).
    # Different from Llama-3.1! Tests if older Llama works too.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-217: Llama-3-8B L13 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_llama3_8b_32x_resid.yaml",
        layer=13,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-218: Llama-3-8B L16 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_llama3_8b_32x_resid.yaml",
        layer=16,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-219: Llama-3-8B L30 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/eleutherai_llama3_8b_32x_resid.yaml",
        layer=30,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-220: Llama-3-8B L16 + ARC-Challenge + GEN",
        sae="configs/saes/eleutherai_llama3_8b_32x_resid.yaml",
        layer=16,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group F: Pythia-160M — EleutherAI 32k MLP BatchTopK
    # 12 layers. L5 (42%), L6 (50%), L11 (92%). MLP hook.
    # chat_template fix applied — previously failed.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-221: Pythia-160M MLP L5 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_pythia_160m_32k_mlp.yaml",
        layer=5,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
        ],
    ),
    Experiment(
        name="EXP-222: Pythia-160M MLP L6 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_pythia_160m_32k_mlp.yaml",
        layer=6,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
        ],
    ),
    Experiment(
        name="EXP-223: Pythia-160M MLP L11 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/eleutherai_pythia_160m_32k_mlp.yaml",
        layer=11,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group G: SmolLM2-135M — EleutherAI 64x MLP BatchTopK
    # 30 layers, sparse: [0,3,6,9,12,15,18,21,24,27].
    # L12 (40%), L15 (50%), L27 (90%). MLP hook.
    # chat_template fix applied — previously failed.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-224: SmolLM2-135M MLP L12 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml",
        layer=12,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
        ],
    ),
    Experiment(
        name="EXP-225: SmolLM2-135M MLP L15 + CommonsenseQA + GEN",
        sae="configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml",
        layer=15,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
        ],
    ),
    Experiment(
        name="EXP-226: SmolLM2-135M MLP L27 + CommonsenseQA + GEN [LAST LAYER]",
        sae="configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml",
        layer=27,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group H: Gemma-3-4B-IT — GemmaScope JumpReLU 16k, resid_post
    # 34 layers. L9 (26%), L17 (50%), L29 (85%).
    # Hook fix applied (commit 5244626). First real JumpReLU test >1B.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-227: Gemma-3-4B-IT JumpReLU L17 + CommonsenseQA + GEN",
        sae="configs/saes/gemma_scope2_4b_it_res_w16k.yaml",
        layer=17,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-228: Gemma-3-4B-IT JumpReLU L29 + CommonsenseQA + GEN [LATE]",
        sae="configs/saes/gemma_scope2_4b_it_res_w16k.yaml",
        layer=29,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-229: Gemma-3-4B-IT JumpReLU L9 + CommonsenseQA + GEN [EARLY]",
        sae="configs/saes/gemma_scope2_4b_it_res_w16k.yaml",
        layer=9,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-230: Gemma-3-4B-IT JumpReLU L17 + ARC-Challenge + GEN",
        sae="configs/saes/gemma_scope2_4b_it_res_w16k.yaml",
        layer=17,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group I: Gemma-2-9B-IT — GemmaScope JumpReLU 16k, resid_post
    # 42 layers. L20 (48%), L31 (74%). Largest JumpReLU model.
    # If JumpReLU works anywhere, it should work here.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-231: Gemma-2-9B-IT JumpReLU L20 + CommonsenseQA + GEN",
        sae="configs/saes/gemma_scope_9b_it_res_w16k.yaml",
        layer=20,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-232: Gemma-2-9B-IT JumpReLU L31 + CommonsenseQA + GEN",
        sae="configs/saes/gemma_scope_9b_it_res_w16k.yaml",
        layer=31,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-233: Gemma-2-9B-IT JumpReLU L20 + ARC-Challenge + GEN",
        sae="configs/saes/gemma_scope_9b_it_res_w16k.yaml",
        layer=20,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),

    # ══════════════════════════════════════════════════════════════════
    # Group J: Mistral-7B retry — Tyler BatchTopK 131k, mlp
    # 32 layers, only L16 (50%) available.
    # Previous: +7.9% K-Fold CommonsenseQA, +13.7% HellaSwag (no gen).
    # Re-run with generalisation + new benchmarks.
    # ══════════════════════════════════════════════════════════════════
    Experiment(
        name="EXP-234: Mistral-7B L16 + CommonsenseQA + GEN",
        sae="configs/saes/tyler_mistral_7b.yaml",
        layer=16,
        benchmark="configs/benchmarks/commonsenseqa.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/arc_challenge.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-235: Mistral-7B L16 + ARC-Challenge + GEN",
        sae="configs/saes/tyler_mistral_7b.yaml",
        layer=16,
        benchmark="configs/benchmarks/arc_challenge.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/hellaswag.yaml",
            "configs/benchmarks/openbookqa.yaml",
        ],
    ),
    Experiment(
        name="EXP-236: Mistral-7B L16 + SciQ + GEN [NEW BENCH]",
        sae="configs/saes/tyler_mistral_7b.yaml",
        layer=16,
        benchmark="configs/benchmarks/sciq.yaml",
        overrides=_FILTER_STD,
        batch_size=8,
        generalisation=[
            "configs/benchmarks/commonsenseqa.yaml",
            "configs/benchmarks/arc_challenge.yaml",
        ],
    ),
]


# ── Runner ────────────────────────────────────────────────────────


@dataclass
class Result:
    index: int
    name: str
    success: bool
    duration: timedelta
    log_file: str
    error: str = ""


_active_proc: subprocess.Popen | None = None


def _kill_proc(proc: subprocess.Popen) -> None:
    """Kill subprocess and its entire process group (uv + python + children)."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            # Unix: kill entire process group
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=5)
        else:
            # Windows: terminate process tree
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _signal_handler(signum, frame):
    """Kill child process group on Ctrl+C, then exit."""
    if _active_proc and _active_proc.poll() is None:
        print(f"\n[RUNNER] Caught signal {signum}, killing experiment...")
        _kill_proc(_active_proc)
    sys.exit(130)


def run_single_experiment(exp: Experiment, log_file: Path) -> tuple[bool, str]:
    """Run one experiment as subprocess. Returns (success, error_msg)."""
    global _active_proc

    cmd = ["uv", "run", "python", "-m", "sae_experiments"] + exp.to_cli_args()

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"Experiment: {exp.name}\n")
        lf.write(f"Command: {' '.join(cmd)}\n")
        lf.write(f"Timeout: {exp.timeout}s ({exp.timeout // 3600}h {(exp.timeout % 3600) // 60}m)\n")
        lf.write(f"Started: {datetime.now().isoformat()}\n")
        lf.write("=" * 70 + "\n\n")
        lf.flush()

        # start_new_session=True creates a new process group so we can
        # kill uv + python + all children together with os.killpg()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        _active_proc = proc

        # Reader thread drains stdout so the pipe buffer never fills.
        # Meanwhile the main thread monitors the timeout via proc.wait().
        def _drain_stdout():
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    lf.write(line)
                    lf.flush()
            except ValueError:
                pass  # stdout closed

        reader = threading.Thread(target=_drain_stdout, daemon=True)
        reader.start()

        try:
            proc.wait(timeout=exp.timeout)
            reader.join(timeout=30)

            if proc.returncode == 0:
                return True, ""
            return False, f"exit code {proc.returncode}"

        except subprocess.TimeoutExpired:
            _kill_proc(proc)
            reader.join(timeout=10)
            return False, f"TIMEOUT after {exp.timeout}s"

        except Exception as e:
            _kill_proc(proc)
            reader.join(timeout=10)
            return False, str(e)

        finally:
            _active_proc = None


def _git_push_results(run_id: str, n_files: int) -> None:
    """Commit and push results/EXP-*/ after a batch run. Non-fatal on failure."""
    try:
        # Stage all per-experiment result dirs (-f because results/ is in .gitignore)
        subprocess.run(["git", "add", "-f", "results/EXP-*/"], check=True, capture_output=True, text=True)

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True, text=True,
        )
        if status.returncode == 0:
            print("[GIT] No new results to commit.")
            return

        msg = f"results: add {n_files} experiment(s) from batch {run_id}"
        subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True, text=True)
        subprocess.run(["git", "push"], check=True, capture_output=True, text=True)
        print(f"[GIT] Pushed results: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[GIT] WARNING: git push failed: {e.stderr or e.stdout or e}")
    except Exception as e:
        print(f"[GIT] WARNING: {e}")


def run_all(
    experiments: list[Experiment],
    dry_run: bool = False,
    start_from: int = 1,
    only: list[int] | None = None,
    worker: tuple[int, int] | None = None,
    git_push: bool = False,
) -> None:
    if not experiments:
        print("No experiments defined. Edit EXPERIMENTS in run_experiments.py.")
        sys.exit(0)

    to_run = [
        (i, exp) for i, exp in enumerate(experiments, 1)
        if (only is None or i in only) and i >= start_from
    ]

    # Multi-GPU partitioning: worker=(id, total) picks every total-th experiment
    if worker is not None:
        worker_id, total_workers = worker
        to_run = [
            (i, exp) for idx, (i, exp) in enumerate(to_run)
            if idx % total_workers == worker_id - 1
        ]

    if not to_run:
        print("No experiments match the given filters.")
        sys.exit(0)

    # Print plan
    print("=" * 70)
    worker_tag = f"  [Worker {worker[0]}/{worker[1]}]" if worker else ""
    print(f"  EXPERIMENT QUEUE — {len(to_run)} experiment(s){worker_tag}")
    print("=" * 70)
    for i, exp in to_run:
        print(f"  [{i:02d}] {exp.name}")
        print(f"       sae={exp.sae}  layer={exp.layer}  benchmark={exp.benchmark}")
        if exp.overrides:
            print(f"       overrides: {exp.overrides}")
    print("=" * 70)

    if dry_run:
        print("\nDry run — commands that would execute:\n")
        for i, exp in to_run:
            args = " ".join(exp.to_cli_args())
            print(f"  [{i:02d}] uv run python -m sae_experiments {args}\n")
        return

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if worker is not None:
        run_id += f"_w{worker[0]}of{worker[1]}"
    results: list[Result] = []

    for idx, (i, exp) in enumerate(to_run):
        remaining = len(to_run) - idx
        print(f"\n{'█' * 70}")
        print(f"  [{i:02d}/{len(experiments)}] {exp.name}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {remaining} remaining  |  timeout: {exp.timeout // 3600}h")
        print("█" * 70)

        exp_id = _extract_exp_id(exp.name)
        if exp_id:
            exp_log_dir = Path(f"results/{exp_id}")
            exp_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = exp_log_dir / f"{run_id}_{exp_id}_seq{i:02d}.log"
        else:
            Path("results").mkdir(exist_ok=True)
            log_file = Path("results") / f"{run_id}_seq{i:02d}.log"
        t0 = time.time()

        success, error_msg = run_single_experiment(exp, log_file)

        duration = timedelta(seconds=int(time.time() - t0))

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n{'=' * 70}\n")
            lf.write(f"Finished: {datetime.now().isoformat()}\n")
            lf.write(f"Duration: {duration}\n")
            lf.write(f"Result: {'SUCCESS' if success else 'FAILED'}\n")
            if error_msg:
                lf.write(f"Error: {error_msg}\n")

        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"\n[{status}] {exp.name} — {duration}")
        if error_msg:
            print(f"  Error: {error_msg}")

        results.append(Result(
            index=i, name=exp.name, success=success,
            duration=duration, log_file=str(log_file), error=error_msg,
        ))

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print("=" * 70)

    total_time = sum((r.duration for r in results), timedelta())
    n_ok = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok

    for r in results:
        s = "✓" if r.success else "✗"
        line = f"  {s} [{r.index:02d}] {r.name}  ({r.duration})"
        if r.error:
            line += f"  — {r.error}"
        print(line)

    print(f"\n  {len(results)} experiments | {n_ok} succeeded | {n_fail} failed | {total_time}")
    print("=" * 70)

    if git_push:
        _git_push_results(run_id, n_ok)

    if n_fail > 0:
        sys.exit(1)


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    git_push = "--git-push" in sys.argv
    start_from = 1
    only = None
    worker = None

    if "--start-from" in sys.argv:
        idx = sys.argv.index("--start-from")
        if idx + 1 >= len(sys.argv):
            print("Error: --start-from requires a number.", file=sys.stderr)
            sys.exit(1)
        start_from = int(sys.argv[idx + 1])

    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 >= len(sys.argv):
            print("Error: --only requires experiment number(s).", file=sys.stderr)
            sys.exit(1)
        only = [int(x.strip()) for x in sys.argv[idx + 1].split(",")]

    if "--worker" in sys.argv:
        idx = sys.argv.index("--worker")
        if idx + 1 >= len(sys.argv):
            print("Error: --worker requires format N/M (e.g. 1/3).", file=sys.stderr)
            sys.exit(1)
        parts = sys.argv[idx + 1].split("/")
        if len(parts) != 2:
            print("Error: --worker requires format N/M (e.g. 1/3).", file=sys.stderr)
            sys.exit(1)
        worker_id, total_workers = int(parts[0]), int(parts[1])
        if worker_id < 1 or worker_id > total_workers:
            print(f"Error: worker id must be 1..{total_workers}, got {worker_id}.", file=sys.stderr)
            sys.exit(1)
        worker = (worker_id, total_workers)

    run_all(EXPERIMENTS, dry_run=dry_run, start_from=start_from, only=only, worker=worker, git_push=git_push)


if __name__ == "__main__":
    main()
