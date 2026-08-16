"""Logit-based screening with JSON caching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
from torch import device as torch_device
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmarks import extract_answer
from .config import BenchmarkConfig
from .hooks import HookManager
from .prediction import get_logit_prediction


def get_screening_filename(output_dir: str, cache_name: str, model_short_name: str) -> str:
    return os.path.join(output_dir, f"logit_screening_{cache_name}_{model_short_name}.json")




def sample_to_serializable(sample: Dict) -> Dict:
    """Convert a sample to a JSON-serializable dict."""
    result = {}
    for key, val in sample.items():
        if isinstance(val, (str, int, float, bool, type(None))):
            result[key] = val
        elif isinstance(val, dict):
            result[key] = sample_to_serializable(val)
        elif isinstance(val, list):
            serialized_list = []
            for item in val:
                if isinstance(item, dict):
                    serialized_list.append(sample_to_serializable(item))
                elif isinstance(item, (str, int, float, bool, type(None))):
                    serialized_list.append(item)
                else:
                    serialized_list.append(str(item))
            result[key] = serialized_list
        else:
            result[key] = str(val)
    return result


def screening_to_sample(screening_entry: Dict) -> Dict:
    """Reconstruct original sample from screening entry."""
    return screening_entry["_sample"]


def run_logit_screening(
    questions: List[Dict],
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch_device,
    output_dir: str,
    model_short_name: str,
    force_rerun: bool = False,
    batch_size: int = 1,
) -> Tuple[List[Dict], Optional[Dict[str, Any]]]:
    """Run logit-based screening for all questions, with JSON caching.

    Returns:
        (screening_data, answer_diagnostics) where answer_diagnostics is
        a summary dict when computed during a fresh batched run, or None
        when loaded from cache or using single-sample inference.
    """
    filename = get_screening_filename(output_dir, config.cache_name, model_short_name)

    if os.path.exists(filename) and not force_rerun:
        print(f"Loading logit screening from cache: {filename}")
        with open(filename) as f:
            screening_data = json.load(f)
        print(f"  {len(screening_data)} questions loaded")
        return screening_data, None

    print(f"Running logit screening for {config.name}...")
    print(f"  Questions: {len(questions)}, batch_size={batch_size}")

    answer_diagnostics = None

    if batch_size > 1:
        from .fast_inference import (
            aggregate_answer_diagnostics,
            batched_logit_predictions,
        )

        diagnostics_collector: List[Dict] = []

        results = batched_logit_predictions(
            questions,
            batch_size=batch_size,
            model=model,
            tokenizer=tokenizer,
            config=config,
            hook_mgr=hook_mgr,
            sae=sae,
            answer_token_id_list=answer_token_id_list,
            device=device,
            extract_features=False,
            diagnostics_collector=diagnostics_collector,
            desc="Logit Screening",
        )

        if diagnostics_collector:
            answer_diagnostics = aggregate_answer_diagnostics(
                diagnostics_collector, tokenizer,
            )

        screening_data = []
        for q_idx, (sample, result) in enumerate(zip(questions, results)):
            correct_letter, _ = extract_answer(sample, config)
            screening_entry = {
                "q_idx": q_idx,
                "p_correct": result["prob_correct"],
                "max_confidence": result["max_confidence"],
                "prediction": result["prediction"],
                "correct": result["correct"],
                "expected": correct_letter,
                "_sample": sample_to_serializable(sample),
            }
            screening_data.append(screening_entry)
    else:
        screening_data = []
        for q_idx, sample in enumerate(tqdm(questions, desc="Logit Screening")):
            result = get_logit_prediction(
                sample,
                model=model,
                tokenizer=tokenizer,
                config=config,
                hook_mgr=hook_mgr,
                sae=sae,
                answer_token_id_list=answer_token_id_list,
                device=device,
                extract_features=False,
            )

            correct_letter, _ = extract_answer(sample, config)

            screening_entry = {
                "q_idx": q_idx,
                "p_correct": result["prob_correct"],
                "max_confidence": result["max_confidence"],
                "prediction": result["prediction"],
                "correct": result["correct"],
                "expected": correct_letter,
                "_sample": sample_to_serializable(sample),
            }
            screening_data.append(screening_entry)

            if (q_idx + 1) % 500 == 0:
                recent = screening_data[-500:]
                avg_p = np.mean([s["p_correct"] for s in recent])
                acc = np.mean([s["correct"] for s in recent])
                print(f"  [{q_idx+1}/{len(questions)}] Avg P(correct): {avg_p:.3f}, Acc: {acc*100:.1f}%")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(screening_data, f)

    print(f"Logit screening saved: {filename}")
    return screening_data, answer_diagnostics
