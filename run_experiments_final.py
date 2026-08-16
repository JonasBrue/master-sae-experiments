#!/usr/bin/env python3
"""
Final-matrix batch runner — 220 experiments per
docs/superpowers/specs/2026-05-11-final-experiment-matrix-design.md.

The 12 BS1-successful (SAE, layer, benchmark) cells run first, sorted by their
BS1 full_exact_delta descending; the remaining 208 cells follow alphabetically
by SAE filename, then ascending layer, then CSQA-before-ARC (per spec § 6).

Usage:
    uv run python run_experiments_final.py                # run all 220
    uv run python run_experiments_final.py --dry-run      # show what would run
    uv run python run_experiments_final.py --start-from 13 # skip the BS1 head
    uv run python run_experiments_final.py --only 1       # smoke-test EXP-300
    uv run python run_experiments_final.py --worker 1/4   # 1 of 4 GPUs
    uv run python run_experiments_final.py --git-push     # commit results after batch
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

DEFAULT_TIMEOUT = 48 * 3600  # 48 hours per experiment
MAX_ATTEMPTS = 3            # max total attempts per experiment (1 original + 2 retries)
RETRY_DELAY = 30            # seconds to wait before retry
INTER_EXPERIMENT_COOLDOWN = 30  # seconds between experiments — lets the GPU driver settle
FAST_FAIL_THRESHOLD = 120   # if experiment fails within this many seconds, it's likely transient

_EXP_ID_RE = re.compile(r"EXP-\d+")


def _extract_exp_id(name: str) -> str:
    m = _EXP_ID_RE.search(name)
    return m.group(0) if m else ""


def _archive_if_nonempty(exp_dir: Path) -> Path | None:
    """Rename ``exp_dir`` to ``<name>_archived_<timestamp>`` if it exists and is
    non-empty. Returns the new path, or None when nothing was archived.

    Used by run_experiments_final.py to preserve prior good runs before
    starting a fresh run into the same EXP-N folder. Empty directories are
    left in place — they have nothing worth keeping.
    """
    if not exp_dir.exists() or not any(exp_dir.iterdir()):
        return None
    archive_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = exp_dir.with_name(f"{exp_dir.name}_archived_{archive_ts}")
    exp_dir.rename(archived)
    return archived


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
        # Every experiment in this runner keeps its screening cache co-located
        # with its results AND recomputes screening from scratch. No cache
        # reuse across experiments — see spec §3.
        parts += ["--cache-dir", output_dir]
        parts += ["--force-rerun-screening"]
        return parts


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT QUEUE — 220 cells per spec § 5.
# ═══════════════════════════════════════════════════════════════════

_FILTER_STD = [
    "causal.min_p_correct=0.15",
    "causal.max_p_correct=0.45",
    # Single-strength mode (pre-061e8a7 default). Phase 2 runs once at 1.0
    # and Phase 3 reuses that ranking for every grid.strength cell.
    # To opt back into multi-strength, change to e.g. [0.5,1.0,1.5].
    "causal.causal_test_strength=[1.0]",
]

_GEN_CSQA = [
    "configs/benchmarks/arc_challenge.yaml",
    "configs/benchmarks/hellaswag.yaml",
    "configs/benchmarks/openbookqa.yaml",
]
_GEN_ARC = [
    "configs/benchmarks/commonsenseqa.yaml",
    "configs/benchmarks/hellaswag.yaml",
    "configs/benchmarks/openbookqa.yaml",
]

EXPERIMENTS: list[Experiment] = [
    # ══════════════════════════════════════════════════════════════════
    # BS1 priority head — 12 cells from successful BS1 runs,
    # sorted by full_exact_delta desc. See dashboard for source.
    # ══════════════════════════════════════════════════════════════════
    # Mirrors BS1 EXP-205-BS1 (full_exact +29.01%)
    Experiment(
        name='EXP-300: jbloom_gemma_2b L17 + CSQA',
        sae='configs/saes/jbloom_gemma_2b.yaml',
        layer=17,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-203-BS1 (full_exact +3.41%)
    Experiment(
        name='EXP-301: jbloom_gemma_2b L10 + CSQA',
        sae='configs/saes/jbloom_gemma_2b.yaml',
        layer=10,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-199-BS1 (full_exact +1.59%)
    Experiment(
        name='EXP-302: eleutherai_llama32_1b_131k_mlp L15 + CSQA',
        sae='configs/saes/eleutherai_llama32_1b_131k_mlp.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-198-BS1 (full_exact +1.40%)
    # Note: BS1 used K32 variant; K-collapsed to K64 equivalent
    Experiment(
        name='EXP-303: andyrdt_llama_k64 L15 + ARC',
        sae='configs/saes/andyrdt_llama_k64.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    # Mirrors BS1 EXP-197-BS1 (full_exact +1.11%)
    Experiment(
        name='EXP-304: qresearch_llama32_1b L9 + CSQA',
        sae='configs/saes/qresearch_llama32_1b.yaml',
        layer=9,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-200-BS1 (full_exact +0.02%)
    Experiment(
        name='EXP-305: jbloom_gemma_2b_it L12 + CSQA',
        sae='configs/saes/jbloom_gemma_2b_it.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-208-BS1 (full_exact +0.02%)
    # Note: BS1 ran at L6; snapped to L5 (new middle per spec § 3, target=(N-1)/2=5.5, tie→lower)
    Experiment(
        name='EXP-306: jbloom_gpt2_small_oai_v5 L5 + CSQA',
        sae='configs/saes/jbloom_gpt2_small_oai_v5.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-235-BS1 (full_exact +0.00%)
    Experiment(
        name='EXP-307: tyler_mistral_7b L16 + ARC',
        sae='configs/saes/tyler_mistral_7b.yaml',
        layer=16,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    # Mirrors BS1 EXP-222-BS1 (full_exact -0.08%)
    # Note: BS1 used K32 variant; K-collapsed to K64 equivalent
    Experiment(
        name='EXP-308: eleutherai_pythia_160m_k64_32k_mlp L6 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_k64_32k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-215-BS1 (full_exact -0.12%)
    # Note: BS1 ran at L12; snapped to L11 (new middle per spec § 3, target=(N-1)/2=11.5, tie→lower)
    Experiment(
        name='EXP-309: eleutherai_pythia_410m_65k_mlp L11 + ARC',
        sae='configs/saes/eleutherai_pythia_410m_65k_mlp.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    # Mirrors BS1 EXP-218-BS1 (full_exact -0.32%)
    # Note: BS1 ran at L16; snapped to L15 (new middle per spec § 3, target=(N-1)/2=15.5, tie→lower)
    Experiment(
        name='EXP-310: eleutherai_llama3_8b_32x_resid L15 + CSQA',
        sae='configs/saes/eleutherai_llama3_8b_32x_resid.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # Mirrors BS1 EXP-224-BS1 (full_exact -0.43%)
    # Note: BS1 ran at L12; snapped to L15 (closest layer in § 3 selected set {15, 27})
    Experiment(
        name='EXP-311: eleutherai_smollm2_135m_64x_mlp L15 + CSQA',
        sae='configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    # ══════════════════════════════════════════════════════════════════
    # Tail (208 cells) — alphabetical by SAE, then ascending layer,
    # then CSQA before ARC. Per spec § 6 ordering.
    # ══════════════════════════════════════════════════════════════════

    # ── ale21m9_gemma2_2b_finetuned_16k_resid ──
    Experiment(
        name='EXP-312: ale21m9_gemma2_2b_finetuned_16k_resid L19 + CSQA',
        sae='configs/saes/ale21m9_gemma2_2b_finetuned_16k_resid.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-313: ale21m9_gemma2_2b_finetuned_16k_resid L19 + ARC',
        sae='configs/saes/ale21m9_gemma2_2b_finetuned_16k_resid.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── andyrdt_llama_k64 ──
    Experiment(
        name='EXP-314: andyrdt_llama_k64 L15 + CSQA',
        sae='configs/saes/andyrdt_llama_k64.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-315: andyrdt_llama_k64 L27 + CSQA',
        sae='configs/saes/andyrdt_llama_k64.yaml',
        layer=27,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-316: andyrdt_llama_k64 L27 + ARC',
        sae='configs/saes/andyrdt_llama_k64.yaml',
        layer=27,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── andyrdt_qwen_k64 ──
    Experiment(
        name='EXP-317: andyrdt_qwen_k64 L15 + CSQA',
        sae='configs/saes/andyrdt_qwen_k64.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-318: andyrdt_qwen_k64 L15 + ARC',
        sae='configs/saes/andyrdt_qwen_k64.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-319: andyrdt_qwen_k64 L27 + CSQA',
        sae='configs/saes/andyrdt_qwen_k64.yaml',
        layer=27,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-320: andyrdt_qwen_k64 L27 + ARC',
        sae='configs/saes/andyrdt_qwen_k64.yaml',
        layer=27,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── chanind_gemma2_2b_matryoshka_32k_snap ──
    Experiment(
        name='EXP-321: chanind_gemma2_2b_matryoshka_32k_snap L12 + CSQA',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_snap.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-322: chanind_gemma2_2b_matryoshka_32k_snap L12 + ARC',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_snap.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-323: chanind_gemma2_2b_matryoshka_32k_snap L24 + CSQA',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_snap.yaml',
        layer=24,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-324: chanind_gemma2_2b_matryoshka_32k_snap L24 + ARC',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_snap.yaml',
        layer=24,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── chanind_gemma2_2b_matryoshka_32k_standard ──
    Experiment(
        name='EXP-325: chanind_gemma2_2b_matryoshka_32k_standard L12 + CSQA',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_standard.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-326: chanind_gemma2_2b_matryoshka_32k_standard L12 + ARC',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_standard.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-327: chanind_gemma2_2b_matryoshka_32k_standard L24 + CSQA',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_standard.yaml',
        layer=24,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-328: chanind_gemma2_2b_matryoshka_32k_standard L24 + ARC',
        sae='configs/saes/chanind_gemma2_2b_matryoshka_32k_standard.yaml',
        layer=24,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── chanind_gemma2_2b_tied_topk_k50 ──
    Experiment(
        name='EXP-329: chanind_gemma2_2b_tied_topk_k50 L1 + CSQA',
        sae='configs/saes/chanind_gemma2_2b_tied_topk_k50.yaml',
        layer=1,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-330: chanind_gemma2_2b_tied_topk_k50 L1 + ARC',
        sae='configs/saes/chanind_gemma2_2b_tied_topk_k50.yaml',
        layer=1,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── chanind_gemma3_1b_matryoshka_32k ──
    Experiment(
        name='EXP-331: chanind_gemma3_1b_matryoshka_32k L12 + CSQA',
        sae='configs/saes/chanind_gemma3_1b_matryoshka_32k.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-332: chanind_gemma3_1b_matryoshka_32k L12 + ARC',
        sae='configs/saes/chanind_gemma3_1b_matryoshka_32k.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-333: chanind_gemma3_1b_matryoshka_32k L24 + CSQA',
        sae='configs/saes/chanind_gemma3_1b_matryoshka_32k.yaml',
        layer=24,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-334: chanind_gemma3_1b_matryoshka_32k L24 + ARC',
        sae='configs/saes/chanind_gemma3_1b_matryoshka_32k.yaml',
        layer=24,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── chanind_llama32_1b_16k_resid ──
    Experiment(
        name='EXP-335: chanind_llama32_1b_16k_resid L7 + CSQA',
        sae='configs/saes/chanind_llama32_1b_16k_resid.yaml',
        layer=7,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-336: chanind_llama32_1b_16k_resid L7 + ARC',
        sae='configs/saes/chanind_llama32_1b_16k_resid.yaml',
        layer=7,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-337: chanind_llama32_1b_16k_resid L8 + CSQA',
        sae='configs/saes/chanind_llama32_1b_16k_resid.yaml',
        layer=8,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-338: chanind_llama32_1b_16k_resid L8 + ARC',
        sae='configs/saes/chanind_llama32_1b_16k_resid.yaml',
        layer=8,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_deepseek_r1_qwen15b_k32_mlp ──
    Experiment(
        name='EXP-339: eleutherai_deepseek_r1_qwen15b_k32_mlp L13 + CSQA',
        sae='configs/saes/eleutherai_deepseek_r1_qwen15b_k32_mlp.yaml',
        layer=13,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-340: eleutherai_deepseek_r1_qwen15b_k32_mlp L13 + ARC',
        sae='configs/saes/eleutherai_deepseek_r1_qwen15b_k32_mlp.yaml',
        layer=13,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-341: eleutherai_deepseek_r1_qwen15b_k32_mlp L27 + CSQA',
        sae='configs/saes/eleutherai_deepseek_r1_qwen15b_k32_mlp.yaml',
        layer=27,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-342: eleutherai_deepseek_r1_qwen15b_k32_mlp L27 + ARC',
        sae='configs/saes/eleutherai_deepseek_r1_qwen15b_k32_mlp.yaml',
        layer=27,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_llama31_8b_32x_mlp ──
    Experiment(
        name='EXP-343: eleutherai_llama31_8b_32x_mlp L23 + CSQA',
        sae='configs/saes/eleutherai_llama31_8b_32x_mlp.yaml',
        layer=23,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-344: eleutherai_llama31_8b_32x_mlp L23 + ARC',
        sae='configs/saes/eleutherai_llama31_8b_32x_mlp.yaml',
        layer=23,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-345: eleutherai_llama31_8b_32x_mlp L29 + CSQA',
        sae='configs/saes/eleutherai_llama31_8b_32x_mlp.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-346: eleutherai_llama31_8b_32x_mlp L29 + ARC',
        sae='configs/saes/eleutherai_llama31_8b_32x_mlp.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_llama31_8b_64x_mlp ──
    Experiment(
        name='EXP-347: eleutherai_llama31_8b_64x_mlp L23 + CSQA',
        sae='configs/saes/eleutherai_llama31_8b_64x_mlp.yaml',
        layer=23,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-348: eleutherai_llama31_8b_64x_mlp L23 + ARC',
        sae='configs/saes/eleutherai_llama31_8b_64x_mlp.yaml',
        layer=23,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-349: eleutherai_llama31_8b_64x_mlp L29 + CSQA',
        sae='configs/saes/eleutherai_llama31_8b_64x_mlp.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-350: eleutherai_llama31_8b_64x_mlp L29 + ARC',
        sae='configs/saes/eleutherai_llama31_8b_64x_mlp.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_llama31_8b_64x_resid ──
    Experiment(
        name='EXP-351: eleutherai_llama31_8b_64x_resid L23 + CSQA',
        sae='configs/saes/eleutherai_llama31_8b_64x_resid.yaml',
        layer=23,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-352: eleutherai_llama31_8b_64x_resid L23 + ARC',
        sae='configs/saes/eleutherai_llama31_8b_64x_resid.yaml',
        layer=23,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-353: eleutherai_llama31_8b_64x_resid L29 + CSQA',
        sae='configs/saes/eleutherai_llama31_8b_64x_resid.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-354: eleutherai_llama31_8b_64x_resid L29 + ARC',
        sae='configs/saes/eleutherai_llama31_8b_64x_resid.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_llama32_1b_131k_mlp ──
    Experiment(
        name='EXP-355: eleutherai_llama32_1b_131k_mlp L7 + CSQA',
        sae='configs/saes/eleutherai_llama32_1b_131k_mlp.yaml',
        layer=7,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-356: eleutherai_llama32_1b_131k_mlp L7 + ARC',
        sae='configs/saes/eleutherai_llama32_1b_131k_mlp.yaml',
        layer=7,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-357: eleutherai_llama32_1b_131k_mlp L15 + ARC',
        sae='configs/saes/eleutherai_llama32_1b_131k_mlp.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_llama3_8b_32x_resid ──
    Experiment(
        name='EXP-358: eleutherai_llama3_8b_32x_resid L15 + ARC',
        sae='configs/saes/eleutherai_llama3_8b_32x_resid.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-359: eleutherai_llama3_8b_32x_resid L30 + CSQA',
        sae='configs/saes/eleutherai_llama3_8b_32x_resid.yaml',
        layer=30,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-360: eleutherai_llama3_8b_32x_resid L30 + ARC',
        sae='configs/saes/eleutherai_llama3_8b_32x_resid.yaml',
        layer=30,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_32k_resid ──
    Experiment(
        name='EXP-361: eleutherai_pythia_160m_32k_resid L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-362: eleutherai_pythia_160m_32k_resid L5 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-363: eleutherai_pythia_160m_32k_resid L11 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_32k_resid.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-364: eleutherai_pythia_160m_32k_resid L11 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_32k_resid.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_deduped_32k_attn ──
    Experiment(
        name='EXP-365: eleutherai_pythia_160m_deduped_32k_attn L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_attn.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-366: eleutherai_pythia_160m_deduped_32k_attn L5 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_attn.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-367: eleutherai_pythia_160m_deduped_32k_attn L11 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_attn.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-368: eleutherai_pythia_160m_deduped_32k_attn L11 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_attn.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_deduped_32k_mlp ──
    Experiment(
        name='EXP-369: eleutherai_pythia_160m_deduped_32k_mlp L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_mlp.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-370: eleutherai_pythia_160m_deduped_32k_mlp L5 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_mlp.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-371: eleutherai_pythia_160m_deduped_32k_mlp L11 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_mlp.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-372: eleutherai_pythia_160m_deduped_32k_mlp L11 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_mlp.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_deduped_32k_resid ──
    Experiment(
        name='EXP-373: eleutherai_pythia_160m_deduped_32k_resid L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-374: eleutherai_pythia_160m_deduped_32k_resid L5 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-375: eleutherai_pythia_160m_deduped_32k_resid L11 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_resid.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-376: eleutherai_pythia_160m_deduped_32k_resid L11 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_deduped_32k_resid.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_k64_131k_mlp ──
    Experiment(
        name='EXP-377: eleutherai_pythia_160m_k64_131k_mlp L6 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_k64_131k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-378: eleutherai_pythia_160m_k64_131k_mlp L6 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_k64_131k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_k64_32k_mlp ──
    Experiment(
        name='EXP-379: eleutherai_pythia_160m_k64_32k_mlp L6 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_k64_32k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_k64_4k_mlp ──
    Experiment(
        name='EXP-380: eleutherai_pythia_160m_k64_4k_mlp L6 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_k64_4k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-381: eleutherai_pythia_160m_k64_4k_mlp L6 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_k64_4k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_k64_65k_mlp ──
    Experiment(
        name='EXP-382: eleutherai_pythia_160m_k64_65k_mlp L6 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_k64_65k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-383: eleutherai_pythia_160m_k64_65k_mlp L6 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_k64_65k_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_160m_k64_768_mlp ──
    Experiment(
        name='EXP-384: eleutherai_pythia_160m_k64_768_mlp L6 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_k64_768_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-385: eleutherai_pythia_160m_k64_768_mlp L6 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_k64_768_mlp.yaml',
        layer=6,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-386: eleutherai_pythia_160m_k64_768_mlp L10 + CSQA',
        sae='configs/saes/eleutherai_pythia_160m_k64_768_mlp.yaml',
        layer=10,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-387: eleutherai_pythia_160m_k64_768_mlp L10 + ARC',
        sae='configs/saes/eleutherai_pythia_160m_k64_768_mlp.yaml',
        layer=10,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_410m_65k_mlp ──
    Experiment(
        name='EXP-388: eleutherai_pythia_410m_65k_mlp L11 + CSQA',
        sae='configs/saes/eleutherai_pythia_410m_65k_mlp.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-389: eleutherai_pythia_410m_65k_mlp L23 + CSQA',
        sae='configs/saes/eleutherai_pythia_410m_65k_mlp.yaml',
        layer=23,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-390: eleutherai_pythia_410m_65k_mlp L23 + ARC',
        sae='configs/saes/eleutherai_pythia_410m_65k_mlp.yaml',
        layer=23,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_70m_32k_attn ──
    Experiment(
        name='EXP-391: eleutherai_pythia_70m_32k_attn L2 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_32k_attn.yaml',
        layer=2,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-392: eleutherai_pythia_70m_32k_attn L2 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_32k_attn.yaml',
        layer=2,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-393: eleutherai_pythia_70m_32k_attn L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_32k_attn.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-394: eleutherai_pythia_70m_32k_attn L5 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_32k_attn.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_70m_32k_mlp ──
    Experiment(
        name='EXP-395: eleutherai_pythia_70m_32k_mlp L2 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_32k_mlp.yaml',
        layer=2,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-396: eleutherai_pythia_70m_32k_mlp L2 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_32k_mlp.yaml',
        layer=2,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-397: eleutherai_pythia_70m_32k_mlp L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_32k_mlp.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-398: eleutherai_pythia_70m_32k_mlp L5 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_32k_mlp.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_70m_32k_resid ──
    Experiment(
        name='EXP-399: eleutherai_pythia_70m_32k_resid L2 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_32k_resid.yaml',
        layer=2,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-400: eleutherai_pythia_70m_32k_resid L2 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_32k_resid.yaml',
        layer=2,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-401: eleutherai_pythia_70m_32k_resid L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-402: eleutherai_pythia_70m_32k_resid L5 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_70m_deduped_32k_attn ──
    Experiment(
        name='EXP-403: eleutherai_pythia_70m_deduped_32k_attn L2 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_attn.yaml',
        layer=2,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-404: eleutherai_pythia_70m_deduped_32k_attn L2 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_attn.yaml',
        layer=2,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-405: eleutherai_pythia_70m_deduped_32k_attn L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_attn.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-406: eleutherai_pythia_70m_deduped_32k_attn L5 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_attn.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_70m_deduped_32k_mlp ──
    Experiment(
        name='EXP-407: eleutherai_pythia_70m_deduped_32k_mlp L2 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_mlp.yaml',
        layer=2,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-408: eleutherai_pythia_70m_deduped_32k_mlp L2 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_mlp.yaml',
        layer=2,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-409: eleutherai_pythia_70m_deduped_32k_mlp L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_mlp.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-410: eleutherai_pythia_70m_deduped_32k_mlp L5 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_mlp.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_pythia_70m_deduped_32k_resid ──
    Experiment(
        name='EXP-411: eleutherai_pythia_70m_deduped_32k_resid L2 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_resid.yaml',
        layer=2,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-412: eleutherai_pythia_70m_deduped_32k_resid L2 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_resid.yaml',
        layer=2,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-413: eleutherai_pythia_70m_deduped_32k_resid L5 + CSQA',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-414: eleutherai_pythia_70m_deduped_32k_resid L5 + ARC',
        sae='configs/saes/eleutherai_pythia_70m_deduped_32k_resid.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── eleutherai_smollm2_135m_64x_mlp ──
    Experiment(
        name='EXP-415: eleutherai_smollm2_135m_64x_mlp L15 + ARC',
        sae='configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-416: eleutherai_smollm2_135m_64x_mlp L27 + CSQA',
        sae='configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml',
        layer=27,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-417: eleutherai_smollm2_135m_64x_mlp L27 + ARC',
        sae='configs/saes/eleutherai_smollm2_135m_64x_mlp.yaml',
        layer=27,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_12b_it_res_w16k ──
    Experiment(
        name='EXP-418: gemma_scope2_12b_it_res_w16k L24 + CSQA',
        sae='configs/saes/gemma_scope2_12b_it_res_w16k.yaml',
        layer=24,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-419: gemma_scope2_12b_it_res_w16k L24 + ARC',
        sae='configs/saes/gemma_scope2_12b_it_res_w16k.yaml',
        layer=24,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-420: gemma_scope2_12b_it_res_w16k L41 + CSQA',
        sae='configs/saes/gemma_scope2_12b_it_res_w16k.yaml',
        layer=41,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-421: gemma_scope2_12b_it_res_w16k L41 + ARC',
        sae='configs/saes/gemma_scope2_12b_it_res_w16k.yaml',
        layer=41,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_1b_it_res_w16k ──
    Experiment(
        name='EXP-422: gemma_scope2_1b_it_res_w16k L13 + CSQA',
        sae='configs/saes/gemma_scope2_1b_it_res_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-423: gemma_scope2_1b_it_res_w16k L13 + ARC',
        sae='configs/saes/gemma_scope2_1b_it_res_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-424: gemma_scope2_1b_it_res_w16k L22 + CSQA',
        sae='configs/saes/gemma_scope2_1b_it_res_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-425: gemma_scope2_1b_it_res_w16k L22 + ARC',
        sae='configs/saes/gemma_scope2_1b_it_res_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_1b_pt_att_w16k ──
    Experiment(
        name='EXP-426: gemma_scope2_1b_pt_att_w16k L13 + CSQA',
        sae='configs/saes/gemma_scope2_1b_pt_att_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-427: gemma_scope2_1b_pt_att_w16k L13 + ARC',
        sae='configs/saes/gemma_scope2_1b_pt_att_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-428: gemma_scope2_1b_pt_att_w16k L22 + CSQA',
        sae='configs/saes/gemma_scope2_1b_pt_att_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-429: gemma_scope2_1b_pt_att_w16k L22 + ARC',
        sae='configs/saes/gemma_scope2_1b_pt_att_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_1b_pt_mlp_w16k ──
    Experiment(
        name='EXP-430: gemma_scope2_1b_pt_mlp_w16k L13 + CSQA',
        sae='configs/saes/gemma_scope2_1b_pt_mlp_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-431: gemma_scope2_1b_pt_mlp_w16k L13 + ARC',
        sae='configs/saes/gemma_scope2_1b_pt_mlp_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-432: gemma_scope2_1b_pt_mlp_w16k L22 + CSQA',
        sae='configs/saes/gemma_scope2_1b_pt_mlp_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-433: gemma_scope2_1b_pt_mlp_w16k L22 + ARC',
        sae='configs/saes/gemma_scope2_1b_pt_mlp_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_1b_pt_res_w16k ──
    Experiment(
        name='EXP-434: gemma_scope2_1b_pt_res_w16k L13 + CSQA',
        sae='configs/saes/gemma_scope2_1b_pt_res_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-435: gemma_scope2_1b_pt_res_w16k L13 + ARC',
        sae='configs/saes/gemma_scope2_1b_pt_res_w16k.yaml',
        layer=13,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-436: gemma_scope2_1b_pt_res_w16k L22 + CSQA',
        sae='configs/saes/gemma_scope2_1b_pt_res_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-437: gemma_scope2_1b_pt_res_w16k L22 + ARC',
        sae='configs/saes/gemma_scope2_1b_pt_res_w16k.yaml',
        layer=22,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_270m_it_res_w16k ──
    Experiment(
        name='EXP-438: gemma_scope2_270m_it_res_w16k L9 + CSQA',
        sae='configs/saes/gemma_scope2_270m_it_res_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-439: gemma_scope2_270m_it_res_w16k L9 + ARC',
        sae='configs/saes/gemma_scope2_270m_it_res_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-440: gemma_scope2_270m_it_res_w16k L15 + CSQA',
        sae='configs/saes/gemma_scope2_270m_it_res_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-441: gemma_scope2_270m_it_res_w16k L15 + ARC',
        sae='configs/saes/gemma_scope2_270m_it_res_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_270m_pt_att_w16k ──
    Experiment(
        name='EXP-442: gemma_scope2_270m_pt_att_w16k L9 + CSQA',
        sae='configs/saes/gemma_scope2_270m_pt_att_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-443: gemma_scope2_270m_pt_att_w16k L9 + ARC',
        sae='configs/saes/gemma_scope2_270m_pt_att_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-444: gemma_scope2_270m_pt_att_w16k L15 + CSQA',
        sae='configs/saes/gemma_scope2_270m_pt_att_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-445: gemma_scope2_270m_pt_att_w16k L15 + ARC',
        sae='configs/saes/gemma_scope2_270m_pt_att_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_270m_pt_mlp_w16k ──
    Experiment(
        name='EXP-446: gemma_scope2_270m_pt_mlp_w16k L9 + CSQA',
        sae='configs/saes/gemma_scope2_270m_pt_mlp_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-447: gemma_scope2_270m_pt_mlp_w16k L9 + ARC',
        sae='configs/saes/gemma_scope2_270m_pt_mlp_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-448: gemma_scope2_270m_pt_mlp_w16k L15 + CSQA',
        sae='configs/saes/gemma_scope2_270m_pt_mlp_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-449: gemma_scope2_270m_pt_mlp_w16k L15 + ARC',
        sae='configs/saes/gemma_scope2_270m_pt_mlp_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_270m_pt_res_w16k ──
    Experiment(
        name='EXP-450: gemma_scope2_270m_pt_res_w16k L9 + CSQA',
        sae='configs/saes/gemma_scope2_270m_pt_res_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-451: gemma_scope2_270m_pt_res_w16k L9 + ARC',
        sae='configs/saes/gemma_scope2_270m_pt_res_w16k.yaml',
        layer=9,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-452: gemma_scope2_270m_pt_res_w16k L15 + CSQA',
        sae='configs/saes/gemma_scope2_270m_pt_res_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-453: gemma_scope2_270m_pt_res_w16k L15 + ARC',
        sae='configs/saes/gemma_scope2_270m_pt_res_w16k.yaml',
        layer=15,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_4b_it_res_w16k ──
    Experiment(
        name='EXP-454: gemma_scope2_4b_it_res_w16k L17 + CSQA',
        sae='configs/saes/gemma_scope2_4b_it_res_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-455: gemma_scope2_4b_it_res_w16k L17 + ARC',
        sae='configs/saes/gemma_scope2_4b_it_res_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-456: gemma_scope2_4b_it_res_w16k L29 + CSQA',
        sae='configs/saes/gemma_scope2_4b_it_res_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-457: gemma_scope2_4b_it_res_w16k L29 + ARC',
        sae='configs/saes/gemma_scope2_4b_it_res_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_4b_pt_att_w16k ──
    Experiment(
        name='EXP-458: gemma_scope2_4b_pt_att_w16k L17 + CSQA',
        sae='configs/saes/gemma_scope2_4b_pt_att_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-459: gemma_scope2_4b_pt_att_w16k L17 + ARC',
        sae='configs/saes/gemma_scope2_4b_pt_att_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-460: gemma_scope2_4b_pt_att_w16k L29 + CSQA',
        sae='configs/saes/gemma_scope2_4b_pt_att_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-461: gemma_scope2_4b_pt_att_w16k L29 + ARC',
        sae='configs/saes/gemma_scope2_4b_pt_att_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_4b_pt_mlp_w16k ──
    Experiment(
        name='EXP-462: gemma_scope2_4b_pt_mlp_w16k L17 + CSQA',
        sae='configs/saes/gemma_scope2_4b_pt_mlp_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-463: gemma_scope2_4b_pt_mlp_w16k L17 + ARC',
        sae='configs/saes/gemma_scope2_4b_pt_mlp_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-464: gemma_scope2_4b_pt_mlp_w16k L29 + CSQA',
        sae='configs/saes/gemma_scope2_4b_pt_mlp_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-465: gemma_scope2_4b_pt_mlp_w16k L29 + ARC',
        sae='configs/saes/gemma_scope2_4b_pt_mlp_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope2_4b_pt_res_w16k ──
    Experiment(
        name='EXP-466: gemma_scope2_4b_pt_res_w16k L17 + CSQA',
        sae='configs/saes/gemma_scope2_4b_pt_res_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-467: gemma_scope2_4b_pt_res_w16k L17 + ARC',
        sae='configs/saes/gemma_scope2_4b_pt_res_w16k.yaml',
        layer=17,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-468: gemma_scope2_4b_pt_res_w16k L29 + CSQA',
        sae='configs/saes/gemma_scope2_4b_pt_res_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-469: gemma_scope2_4b_pt_res_w16k L29 + ARC',
        sae='configs/saes/gemma_scope2_4b_pt_res_w16k.yaml',
        layer=29,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_2b_pt_att_w16k_l12 ──
    Experiment(
        name='EXP-470: gemma_scope_2b_pt_att_w16k_l12 L12 + CSQA',
        sae='configs/saes/gemma_scope_2b_pt_att_w16k_l12.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-471: gemma_scope_2b_pt_att_w16k_l12 L12 + ARC',
        sae='configs/saes/gemma_scope_2b_pt_att_w16k_l12.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_2b_pt_att_w16k_l19 ──
    Experiment(
        name='EXP-472: gemma_scope_2b_pt_att_w16k_l19 L19 + CSQA',
        sae='configs/saes/gemma_scope_2b_pt_att_w16k_l19.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-473: gemma_scope_2b_pt_att_w16k_l19 L19 + ARC',
        sae='configs/saes/gemma_scope_2b_pt_att_w16k_l19.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_2b_pt_mlp_w16k_l12 ──
    Experiment(
        name='EXP-474: gemma_scope_2b_pt_mlp_w16k_l12 L12 + CSQA',
        sae='configs/saes/gemma_scope_2b_pt_mlp_w16k_l12.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-475: gemma_scope_2b_pt_mlp_w16k_l12 L12 + ARC',
        sae='configs/saes/gemma_scope_2b_pt_mlp_w16k_l12.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_2b_pt_mlp_w16k_l19 ──
    Experiment(
        name='EXP-476: gemma_scope_2b_pt_mlp_w16k_l19 L19 + CSQA',
        sae='configs/saes/gemma_scope_2b_pt_mlp_w16k_l19.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-477: gemma_scope_2b_pt_mlp_w16k_l19 L19 + ARC',
        sae='configs/saes/gemma_scope_2b_pt_mlp_w16k_l19.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_2b_pt_res_w16k_l12 ──
    Experiment(
        name='EXP-478: gemma_scope_2b_pt_res_w16k_l12 L12 + CSQA',
        sae='configs/saes/gemma_scope_2b_pt_res_w16k_l12.yaml',
        layer=12,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-479: gemma_scope_2b_pt_res_w16k_l12 L12 + ARC',
        sae='configs/saes/gemma_scope_2b_pt_res_w16k_l12.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_2b_pt_res_w16k_l19 ──
    Experiment(
        name='EXP-480: gemma_scope_2b_pt_res_w16k_l19 L19 + CSQA',
        sae='configs/saes/gemma_scope_2b_pt_res_w16k_l19.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-481: gemma_scope_2b_pt_res_w16k_l19 L19 + ARC',
        sae='configs/saes/gemma_scope_2b_pt_res_w16k_l19.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_9b_it_res_w16k ──
    Experiment(
        name='EXP-482: gemma_scope_9b_it_res_w16k L20 + CSQA',
        sae='configs/saes/gemma_scope_9b_it_res_w16k.yaml',
        layer=20,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-483: gemma_scope_9b_it_res_w16k L20 + ARC',
        sae='configs/saes/gemma_scope_9b_it_res_w16k.yaml',
        layer=20,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-484: gemma_scope_9b_it_res_w16k L31 + CSQA',
        sae='configs/saes/gemma_scope_9b_it_res_w16k.yaml',
        layer=31,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-485: gemma_scope_9b_it_res_w16k L31 + ARC',
        sae='configs/saes/gemma_scope_9b_it_res_w16k.yaml',
        layer=31,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_9b_pt_att_w16k_l20 ──
    Experiment(
        name='EXP-486: gemma_scope_9b_pt_att_w16k_l20 L20 + CSQA',
        sae='configs/saes/gemma_scope_9b_pt_att_w16k_l20.yaml',
        layer=20,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-487: gemma_scope_9b_pt_att_w16k_l20 L20 + ARC',
        sae='configs/saes/gemma_scope_9b_pt_att_w16k_l20.yaml',
        layer=20,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_9b_pt_mlp_w16k_l20 ──
    Experiment(
        name='EXP-488: gemma_scope_9b_pt_mlp_w16k_l20 L20 + CSQA',
        sae='configs/saes/gemma_scope_9b_pt_mlp_w16k_l20.yaml',
        layer=20,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-489: gemma_scope_9b_pt_mlp_w16k_l20 L20 + ARC',
        sae='configs/saes/gemma_scope_9b_pt_mlp_w16k_l20.yaml',
        layer=20,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_9b_pt_res_w16k_l20 ──
    Experiment(
        name='EXP-490: gemma_scope_9b_pt_res_w16k_l20 L20 + CSQA',
        sae='configs/saes/gemma_scope_9b_pt_res_w16k_l20.yaml',
        layer=20,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-491: gemma_scope_9b_pt_res_w16k_l20 L20 + ARC',
        sae='configs/saes/gemma_scope_9b_pt_res_w16k_l20.yaml',
        layer=20,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_9b_pt_res_w16k_l31 ──
    Experiment(
        name='EXP-492: gemma_scope_9b_pt_res_w16k_l31 L31 + CSQA',
        sae='configs/saes/gemma_scope_9b_pt_res_w16k_l31.yaml',
        layer=31,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-493: gemma_scope_9b_pt_res_w16k_l31 L31 + ARC',
        sae='configs/saes/gemma_scope_9b_pt_res_w16k_l31.yaml',
        layer=31,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── gemma_scope_9b_pt_res_w16k_l9 ──
    Experiment(
        name='EXP-494: gemma_scope_9b_pt_res_w16k_l9 L9 + CSQA',
        sae='configs/saes/gemma_scope_9b_pt_res_w16k_l9.yaml',
        layer=9,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-495: gemma_scope_9b_pt_res_w16k_l9 L9 + ARC',
        sae='configs/saes/gemma_scope_9b_pt_res_w16k_l9.yaml',
        layer=9,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── goodfire_relu ──
    Experiment(
        name='EXP-496: goodfire_relu L19 + CSQA',
        sae='configs/saes/goodfire_relu.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-497: goodfire_relu L19 + ARC',
        sae='configs/saes/goodfire_relu.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── jbloom_gemma_2b ──
    Experiment(
        name='EXP-498: jbloom_gemma_2b L10 + ARC',
        sae='configs/saes/jbloom_gemma_2b.yaml',
        layer=10,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-499: jbloom_gemma_2b L17 + ARC',
        sae='configs/saes/jbloom_gemma_2b.yaml',
        layer=17,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── jbloom_gemma_2b_anthropic_l0 ──
    Experiment(
        name='EXP-500: jbloom_gemma_2b_anthropic_l0 L0 + CSQA',
        sae='configs/saes/jbloom_gemma_2b_anthropic_l0.yaml',
        layer=0,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-501: jbloom_gemma_2b_anthropic_l0 L0 + ARC',
        sae='configs/saes/jbloom_gemma_2b_anthropic_l0.yaml',
        layer=0,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── jbloom_gemma_2b_anthropic_l6 ──
    Experiment(
        name='EXP-502: jbloom_gemma_2b_anthropic_l6 L6 + CSQA',
        sae='configs/saes/jbloom_gemma_2b_anthropic_l6.yaml',
        layer=6,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-503: jbloom_gemma_2b_anthropic_l6 L6 + ARC',
        sae='configs/saes/jbloom_gemma_2b_anthropic_l6.yaml',
        layer=6,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── jbloom_gemma_2b_it ──
    Experiment(
        name='EXP-504: jbloom_gemma_2b_it L12 + ARC',
        sae='configs/saes/jbloom_gemma_2b_it.yaml',
        layer=12,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── jbloom_gpt2_small_oai_v5 ──
    Experiment(
        name='EXP-505: jbloom_gpt2_small_oai_v5 L5 + ARC',
        sae='configs/saes/jbloom_gpt2_small_oai_v5.yaml',
        layer=5,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),
    Experiment(
        name='EXP-506: jbloom_gpt2_small_oai_v5 L11 + CSQA',
        sae='configs/saes/jbloom_gpt2_small_oai_v5.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-507: jbloom_gpt2_small_oai_v5 L11 + ARC',
        sae='configs/saes/jbloom_gpt2_small_oai_v5.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── qresearch_deepseek_r1_llama_8b ──
    Experiment(
        name='EXP-508: qresearch_deepseek_r1_llama_8b L19 + CSQA',
        sae='configs/saes/qresearch_deepseek_r1_llama_8b.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-509: qresearch_deepseek_r1_llama_8b L19 + ARC',
        sae='configs/saes/qresearch_deepseek_r1_llama_8b.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── qresearch_llama32_1b ──
    Experiment(
        name='EXP-510: qresearch_llama32_1b L9 + ARC',
        sae='configs/saes/qresearch_llama32_1b.yaml',
        layer=9,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── seonglae_gemma2_2b_18k_resid ──
    Experiment(
        name='EXP-511: seonglae_gemma2_2b_18k_resid L19 + CSQA',
        sae='configs/saes/seonglae_gemma2_2b_18k_resid.yaml',
        layer=19,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-512: seonglae_gemma2_2b_18k_resid L19 + ARC',
        sae='configs/saes/seonglae_gemma2_2b_18k_resid.yaml',
        layer=19,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── seonglae_llama31_8b_16k_resid ──
    Experiment(
        name='EXP-513: seonglae_llama31_8b_16k_resid L23 + CSQA',
        sae='configs/saes/seonglae_llama31_8b_16k_resid.yaml',
        layer=23,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-514: seonglae_llama31_8b_16k_resid L23 + ARC',
        sae='configs/saes/seonglae_llama31_8b_16k_resid.yaml',
        layer=23,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── seonglae_llama32_1b_14k_resid ──
    Experiment(
        name='EXP-515: seonglae_llama32_1b_14k_resid L11 + CSQA',
        sae='configs/saes/seonglae_llama32_1b_14k_resid.yaml',
        layer=11,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-516: seonglae_llama32_1b_14k_resid L11 + ARC',
        sae='configs/saes/seonglae_llama32_1b_14k_resid.yaml',
        layer=11,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # ── sof_gemma2_2b_chat_l13_k100 ──
    Experiment(
        name='EXP-517: sof_gemma2_2b_chat_l13_k100 L13 + CSQA',
        sae='configs/saes/sof_gemma2_2b_chat_l13_k100.yaml',
        layer=13,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
    ),
    Experiment(
        name='EXP-518: sof_gemma2_2b_chat_l13_k100 L13 + ARC',
        sae='configs/saes/sof_gemma2_2b_chat_l13_k100.yaml',
        layer=13,
        benchmark='configs/benchmarks/arc_challenge.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_ARC,
    ),

    # sof_gemma2_2b_difference_{bc,cb}_l13_k100 removed: these SAEs are trained on the
    # (base_act − chat_act) / (chat_act − base_act) difference and require a dual-model
    # forward pass that the current pipeline does not support. See their YAMLs for details.

    # ── tyler_mistral_7b ──
    Experiment(
        name='EXP-519: tyler_mistral_7b L16 + CSQA',
        sae='configs/saes/tyler_mistral_7b.yaml',
        layer=16,
        benchmark='configs/benchmarks/commonsenseqa.yaml',
        overrides=_FILTER_STD,
        batch_size=1,
        generalisation=_GEN_CSQA,
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
    attempts: int = 1


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
        lf.write(
            f"Timeout: {exp.timeout}s ({exp.timeout // 3600}h {(exp.timeout % 3600) // 60}m)\n"
        )
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
        subprocess.run(
            ["git", "add", "-f", "results/EXP-*/"],
            check=True,
            capture_output=True,
            text=True,
        )

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
            text=True,
        )
        if status.returncode == 0:
            print("[GIT] No new results to commit.")
            return

        msg = f"results: add {n_files} experiment(s) from batch {run_id}"
        subprocess.run(
            ["git", "commit", "-m", msg], check=True, capture_output=True, text=True
        )
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
        (i, exp)
        for i, exp in enumerate(experiments, 1)
        if (only is None or i in only) and i >= start_from
    ]

    # Multi-GPU partitioning: worker=(id, total) picks every total-th experiment
    if worker is not None:
        worker_id, total_workers = worker
        to_run = [
            (i, exp)
            for idx, (i, exp) in enumerate(to_run)
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
        print(
            f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  {remaining} remaining  |  timeout: {exp.timeout // 3600}h"
        )
        print("█" * 70)

        exp_id = _extract_exp_id(exp.name)
        exp_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if exp_id:
            exp_log_dir = Path(f"results/{exp_id}")
            archived = _archive_if_nonempty(exp_log_dir)
            if archived is not None:
                print(f"[ARCHIVE] {exp_log_dir} → {archived}")
            exp_log_dir.mkdir(parents=True, exist_ok=True)
            log_file = exp_log_dir / f"{exp_ts}_{exp_id}.log"
        else:
            Path("results").mkdir(exist_ok=True)
            log_file = Path("results") / f"{exp_ts}.log"
        wall_t0 = time.time()
        t0 = time.time()

        success, error_msg = run_single_experiment(exp, log_file)
        elapsed = time.time() - t0

        # Retry transient failures (CUDA busy, OOM on init, etc.)
        attempt = 1
        while not success and elapsed < FAST_FAIL_THRESHOLD and attempt < MAX_ATTEMPTS:
            # Write footer to the failed attempt's log before moving on
            attempt_duration = timedelta(seconds=int(elapsed))
            with open(log_file if attempt == 1 else retry_log, "a", encoding="utf-8") as lf:
                lf.write(f"\n{'=' * 70}\n")
                lf.write(f"Finished: {datetime.now().isoformat()}\n")
                lf.write(f"Duration: {attempt_duration}\n")
                lf.write(f"Result: FAILED (will retry)\n")
                lf.write(f"Error: {error_msg}\n")

            attempt += 1
            print(f"\n[RETRY] Experiment failed in {elapsed:.0f}s (likely transient). "
                  f"Waiting {RETRY_DELAY}s before attempt {attempt}/{MAX_ATTEMPTS}...")
            time.sleep(RETRY_DELAY)

            retry_log = log_file.with_stem(f"{log_file.stem}_retry{attempt}")
            t0 = time.time()
            success, error_msg = run_single_experiment(exp, retry_log)
            elapsed = time.time() - t0
            if success:
                log_file = retry_log  # point to the successful log

        if not success and attempt > 1:
            log_file = retry_log  # final footer lands on the last attempted log

        duration = timedelta(seconds=int(time.time() - wall_t0))

        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"\n{'=' * 70}\n")
            lf.write(f"Finished: {datetime.now().isoformat()}\n")
            lf.write(f"Duration: {duration}\n")
            lf.write(f"Result: {'SUCCESS' if success else 'FAILED'}\n")
            if error_msg:
                lf.write(f"Error: {error_msg}\n")
            if attempt > 1:
                lf.write(f"Attempts: {attempt}\n")

        status = "✓ SUCCESS" if success else "✗ FAILED"
        retry_tag = f" (attempt {attempt}/{MAX_ATTEMPTS})" if attempt > 1 else ""
        print(f"\n[{status}] {exp.name} — {duration}{retry_tag}")
        if error_msg:
            print(f"  Error: {error_msg}")

        results.append(
            Result(
                index=i,
                name=exp.name,
                success=success,
                duration=duration,
                log_file=str(log_file),
                error=error_msg,
                attempts=attempt,
            )
        )

        # GPU cooldown — let the driver release the CUDA context fully before
        # the next subprocess starts. Skip after the LAST experiment so we
        # don't waste 30s right before exit.
        if idx < len(to_run) - 1:
            print(f"\n[COOLDOWN] Waiting {INTER_EXPERIMENT_COOLDOWN}s for GPU to settle before next experiment...")
            time.sleep(INTER_EXPERIMENT_COOLDOWN)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print("=" * 70)

    total_time = sum((r.duration for r in results), timedelta())
    n_ok = sum(1 for r in results if r.success)
    n_fail = len(results) - n_ok

    for r in results:
        s = "✓" if r.success else "✗"
        retry_note = f" (retried {r.attempts - 1}x)" if r.attempts > 1 else ""
        line = f"  {s} [{r.index:02d}] {r.name}  ({r.duration}){retry_note}"
        if r.error:
            line += f"  — {r.error}"
        print(line)

    print(
        f"\n  {len(results)} experiments | {n_ok} succeeded | {n_fail} failed | {total_time}"
    )
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
            print(
                f"Error: worker id must be 1..{total_workers}, got {worker_id}.",
                file=sys.stderr,
            )
            sys.exit(1)
        worker = (worker_id, total_workers)

    run_all(
        EXPERIMENTS,
        dry_run=dry_run,
        start_from=start_from,
        only=only,
        worker=worker,
        git_push=git_push,
    )


if __name__ == "__main__":
    main()
