"""Batched inference for fast SAE experiments.

Processes N questions in one forward pass (right-padded, attention_mask).
~4-8x speedup over the single-sample loop in prediction.py.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmarks import extract_answer, format_prompt
from .config import BenchmarkConfig
from .hooks import HookManager, _gather_last_logits, _gather_last_tokens, _scatter_last_tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Batch input preparation
# ═══════════════════════════════════════════════════════════════════════════════


def prepare_batch_inputs(
    samples: List[Dict],
    config: BenchmarkConfig,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a batch of samples with right-padding.

    Returns:
        (input_ids [B, max_seq], attention_mask [B, max_seq], seq_lengths [B])
    """
    texts = []
    for sample in samples:
        base_prompt = format_prompt(sample, config)
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": base_prompt}]
            full_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )
        else:
            full_text = base_prompt
        texts.append(full_text)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    seq_lengths = attention_mask.sum(dim=1)  # [B] actual token count per sample

    return input_ids, attention_mask, seq_lengths


# ═══════════════════════════════════════════════════════════════════════════════
# Intervention callables (hidden-space, for use with batched_intervention_predictions)
# ═══════════════════════════════════════════════════════════════════════════════


def make_approx_intervention_fn(
    precomputed_hidden_vec: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a fn(hidden [B,seq,d]) -> hidden that adds a precomputed hidden-space vector.

    Replicates ApproximatedInterventionHook logic. Pass to
    batched_intervention_predictions as the intervention argument.
    """
    state: Dict[str, Any] = {"seq_lengths": None}

    def fn(hidden: torch.Tensor) -> torch.Tensor:
        seq_lengths = state["seq_lengths"]
        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths)
            modified_last = last_hidden + precomputed_hidden_vec.unsqueeze(0).unsqueeze(0).to(hidden.dtype)
            return _scatter_last_tokens(hidden, modified_last, seq_lengths)
        context = hidden[:, :-1, :]
        last_hidden = hidden[:, -1:, :]
        modified_last = last_hidden + precomputed_hidden_vec.unsqueeze(0).unsqueeze(0).to(hidden.dtype)
        return torch.cat([context, modified_last], dim=1)

    fn.set_seq_lengths = lambda sl: state.update(seq_lengths=sl)
    return fn


def make_adaptive_exact_intervention_fn(
    sae: nn.Module,
    intervention_vec: torch.Tensor,
    strength_fn: Callable[[float], float],
    p_correct_values: List[float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a vectorized adaptive exact intervention function.

    Each sample in the batch gets a per-sample strength multiplier based on its
    p_correct. Uses an internal offset counter that advances by B each call.
    Call reset_offset() between independent passes over the dataset.
    """
    state: Dict[str, Any] = {"offset": 0, "seq_lengths": None}

    def fn(hidden: torch.Tensor) -> torch.Tensor:
        B = hidden.shape[0]
        start = state["offset"]
        seq_lengths = state["seq_lengths"]
        multipliers = torch.tensor(
            [strength_fn(p) for p in p_correct_values[start:start + B]],
            device=hidden.device, dtype=sae.dtype,
        ).view(B, 1, 1)
        state["offset"] = start + B

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths).to(sae.dtype)
        else:
            last_hidden = hidden[:, -1:, :].to(sae.dtype)

        features = sae.encode(last_hidden)
        reconstructed = sae.decode(features)
        error = last_hidden - reconstructed

        scaled_vec = intervention_vec.unsqueeze(0).unsqueeze(0) * multipliers
        features = features + scaled_vec
        features = torch.clamp(features, min=0)

        modified_last = (sae.decode(features) + error).to(hidden.dtype)

        if seq_lengths is not None:
            return _scatter_last_tokens(hidden, modified_last, seq_lengths)
        return torch.cat([hidden[:, :-1, :], modified_last], dim=1)

    fn.reset_offset = lambda: state.update(offset=0)
    fn.set_seq_lengths = lambda sl: state.update(seq_lengths=sl)
    return fn


def make_adaptive_approx_intervention_fn(
    precomputed_hidden_vec: torch.Tensor,
    strength_fn: Callable[[float], float],
    p_correct_values: List[float],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a vectorized adaptive approximated intervention function.

    Uses an internal offset counter that advances by B each call.
    Call reset_offset() between independent passes over the dataset.
    """
    state: Dict[str, Any] = {"offset": 0, "seq_lengths": None}

    def fn(hidden: torch.Tensor) -> torch.Tensor:
        B = hidden.shape[0]
        start = state["offset"]
        seq_lengths = state["seq_lengths"]
        multipliers = torch.tensor(
            [strength_fn(p) for p in p_correct_values[start:start + B]],
            device=hidden.device, dtype=hidden.dtype,
        ).view(B, 1, 1)
        state["offset"] = start + B

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths)
        else:
            last_hidden = hidden[:, -1:, :]

        scaled_vec = precomputed_hidden_vec.unsqueeze(0).unsqueeze(0).to(hidden.dtype) * multipliers
        modified_last = last_hidden + scaled_vec

        if seq_lengths is not None:
            return _scatter_last_tokens(hidden, modified_last, seq_lengths)
        return torch.cat([hidden[:, :-1, :], modified_last], dim=1)

    fn.reset_offset = lambda: state.update(offset=0)
    fn.set_seq_lengths = lambda sl: state.update(seq_lengths=sl)
    return fn


# ═══════════════════════════════════════════════════════════════════════════════
# Result extraction
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_results(
    logits: torch.Tensor,
    answer_token_id_list: List[int],
    metadata: List[Tuple],
    config: BenchmarkConfig,
) -> List[Dict]:
    """Extract per-sample predictions from batched logits.

    Args:
        logits: [B, vocab_size] (last-token logits)
        answer_token_id_list: Token IDs for answer choices.
        metadata: List of (correct_letter, correct_idx, ...) per sample.
        config: Benchmark config for choice labels.

    Returns:
        List of result dicts with prediction, correct, expected, prob_correct.
    """
    answer_logits = logits[:, answer_token_id_list]  # [B, n_choices]
    probs = torch.softmax(answer_logits, dim=-1)  # [B, n_choices]
    pred_indices = torch.argmax(answer_logits, dim=-1)  # [B]

    results = []
    for i, meta in enumerate(metadata):
        correct_letter, correct_idx = meta[0], meta[1]
        pred_idx = pred_indices[i].item()
        pred_letter = config.choices[pred_idx]
        is_correct = pred_idx == correct_idx

        results.append({
            "prediction": pred_letter,
            "correct": is_correct,
            "expected": correct_letter,
            "prob_correct": probs[i, correct_idx].item(),
            "max_confidence": probs[i].max().item(),
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Answer token diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_batch_diagnostics(
    logits: torch.Tensor,
    answer_token_id_list: List[int],
    collector: List[Dict],
) -> None:
    """Compute per-sample answer token diagnostics and append to collector.

    For each sample in the batch, computes:
    - answer_mass: total softmax probability on answer tokens (full vocab softmax)
    - best_answer_rank: 1-indexed rank of the best answer token in full vocab
    - top1_is_answer: whether the highest-logit token is an answer token
    - top_non_answer: top-5 non-answer tokens with prob and rank
    """
    B = logits.shape[0]
    full_probs = torch.softmax(logits, dim=-1)  # [B, vocab_size]
    answer_set = set(answer_token_id_list)

    # Vectorized: answer mass per sample
    answer_mass = full_probs[:, answer_token_id_list].sum(dim=-1)  # [B]

    # Sort by logit descending to determine ranks
    sorted_indices = logits.argsort(dim=-1, descending=True)  # [B, vocab_size]

    # Answer token mask
    answer_mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    answer_mask[answer_token_id_list] = True

    # Top-1 check
    top1_is_answer = answer_mask[sorted_indices[:, 0]]  # [B]

    # Rank of best answer token per sample
    is_answer_in_sorted = answer_mask[sorted_indices]  # [B, vocab_size]
    best_answer_rank = is_answer_in_sorted.float().argmax(dim=-1) + 1  # [B], 1-indexed

    for i in range(B):
        # Top-5 non-answer tokens from the sorted order
        top_non_answer = []
        for rank_idx in range(min(50, sorted_indices.shape[1])):
            token_id = sorted_indices[i, rank_idx].item()
            if token_id not in answer_set:
                top_non_answer.append({
                    "token_id": token_id,
                    "prob": full_probs[i, token_id].item(),
                    "rank": rank_idx + 1,
                })
                if len(top_non_answer) >= 5:
                    break

        collector.append({
            "answer_mass": answer_mass[i].item(),
            "best_answer_rank": best_answer_rank[i].item(),
            "top1_is_answer": bool(top1_is_answer[i].item()),
            "top_non_answer": top_non_answer,
        })


def aggregate_answer_diagnostics(
    per_sample: List[Dict],
    tokenizer: AutoTokenizer,
) -> Dict[str, Any]:
    """Aggregate per-sample diagnostics into a summary with warnings.

    Prints a human-readable diagnostic report and returns a JSON-serializable
    summary dict suitable for inclusion in the experiment JSON.
    """
    n = len(per_sample)
    if n == 0:
        return {}

    answer_masses = [d["answer_mass"] for d in per_sample]
    best_ranks = [d["best_answer_rank"] for d in per_sample]
    top1_flags = [d["top1_is_answer"] for d in per_sample]

    mean_answer_mass = float(np.mean(answer_masses))
    median_best_rank = int(np.median(best_ranks))
    pct_top1_answer = float(np.mean(top1_flags) * 100)

    # Questions with very low answer mass
    low_mass_threshold = 0.10
    n_low_mass = sum(1 for m in answer_masses if m < low_mass_threshold)

    # Aggregate: which non-answer tokens appear as vocab-wide top-1?
    non_answer_top1_counts: Counter = Counter()
    for d in per_sample:
        if not d["top1_is_answer"] and d["top_non_answer"]:
            non_answer_top1_counts[d["top_non_answer"][0]["token_id"]] += 1

    top_non_answer_decoded = []
    for token_id, count in non_answer_top1_counts.most_common(10):
        token_str = tokenizer.decode([token_id])
        top_non_answer_decoded.append({
            "token": token_str,
            "token_id": token_id,
            "count": count,
            "pct": count / n * 100,
        })

    # ── Print report ──
    print(f"\n{'=' * 80}")
    print(f"ANSWER TOKEN DIAGNOSTICS ({n} questions)")
    print(f"{'=' * 80}")
    print(f"  Mean answer token mass:     {mean_answer_mass*100:.1f}% (of full vocab probability)")
    print(f"  Median best-answer rank:    {median_best_rank} (within full vocab)")
    print(f"  Top-1 is answer token:      {pct_top1_answer:.1f}% of questions")

    if top_non_answer_decoded:
        print(f"\n  Top non-answer tokens (when model prefers non-answer):")
        for entry in top_non_answer_decoded[:5]:
            print(f"    {repr(entry['token']):<15} -> top-1 in {entry['count']} questions ({entry['pct']:.1f}%)")

    # ── Warnings ──
    warnings_list = []
    if mean_answer_mass < 0.50:
        msg = f"CRITICAL: Mean answer mass {mean_answer_mass*100:.1f}% < 50% -- model predominantly prefers non-answer tokens"
        print(f"\n  WARNING: {msg}")
        warnings_list.append(msg)
    elif mean_answer_mass < 0.70:
        msg = f"WARNING: Mean answer mass {mean_answer_mass*100:.1f}% < 70% -- significant probability on non-answer tokens"
        print(f"\n  WARNING: {msg}")
        warnings_list.append(msg)

    if pct_top1_answer < 80:
        msg = f"WARNING: Top-1 is answer token only {pct_top1_answer:.1f}% of the time"
        print(f"\n  WARNING: {msg}")
        warnings_list.append(msg)

    if n_low_mass > 0:
        pct_low = n_low_mass / n * 100
        msg = f"INFO: {n_low_mass} questions ({pct_low:.1f}%) have answer_mass < {low_mass_threshold*100:.0f}%"
        print(f"\n  {msg}")
        warnings_list.append(msg)

    if not warnings_list:
        print(f"\n  OK: No warnings -- answer tokens capture model preferences well")

    print(f"{'=' * 80}")

    return {
        "n_questions": n,
        "mean_answer_mass": mean_answer_mass,
        "median_best_answer_rank": median_best_rank,
        "pct_top1_is_answer": pct_top1_answer,
        "n_low_mass": n_low_mass,
        "low_mass_threshold": low_mass_threshold,
        "top_non_answer_tokens": top_non_answer_decoded,
        "warnings": warnings_list,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Batched prediction functions (high-level)
# ═══════════════════════════════════════════════════════════════════════════════


def batched_logit_predictions(
    samples: List[Dict],
    *,
    batch_size: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    extract_features: bool = False,
    diagnostics_collector: Optional[List[Dict]] = None,
    desc: str = "Batched Predictions",
) -> List[Dict]:
    """Batched baseline logit predictions (no intervention).

    Processes samples in chunks of batch_size. Returns results in same order.

    Args:
        diagnostics_collector: If provided, per-sample answer token diagnostics
            are appended to this list. Use aggregate_answer_diagnostics() to
            summarize the collected data afterwards.
    """
    all_results = []
    n = len(samples)

    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        batch_samples = samples[start:end]

        input_ids, attention_mask, seq_lengths = prepare_batch_inputs(
            batch_samples, config, tokenizer, device,
        )

        metadata = []
        for sample in batch_samples:
            correct_letter, correct_idx = extract_answer(sample, config)
            metadata.append((correct_letter, correct_idx))

        hook_mgr.layer_module._batch_seq_lengths = seq_lengths
        hook_mgr.cache.clear()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = _gather_last_logits(outputs.logits, seq_lengths)

        if diagnostics_collector is not None:
            _collect_batch_diagnostics(logits, answer_token_id_list, diagnostics_collector)

        batch_results = _extract_results(logits, answer_token_id_list, metadata, config)

        if extract_features:
            last_hidden = hook_mgr.cache.get_last_token(seq_lengths)
            if last_hidden is not None:
                features = sae.encode(last_hidden.to(sae.dtype))
                for i in range(len(batch_results)):
                    active_mask = features[i, 0, :] > 0
                    active_indices = torch.where(active_mask)[0]
                    active_values = features[i, 0, active_mask]
                    batch_results[i]["active_indices"] = active_indices
                    batch_results[i]["active_values"] = active_values
            else:
                for i in range(len(batch_results)):
                    batch_results[i]["active_indices"] = torch.tensor([], device=device)
                    batch_results[i]["active_values"] = torch.tensor([], device=device)

        all_results.extend(batch_results)

    return all_results


def batched_intervention_predictions(
    samples: List[Dict],
    intervention: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]],
    *,
    batch_size: int,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: BenchmarkConfig,
    hook_mgr: HookManager,
    sae: nn.Module,
    answer_token_id_list: List[int],
    device: torch.device,
    desc: str = "Batched Intervention",
) -> List[Dict]:
    """Batched intervention predictions.

    Args:
        samples: Questions to evaluate.
        intervention: Either
            - torch.Tensor (feature space): uses InterventionHook(sae, vec) internally.
            - Callable fn(hidden [B,seq,d]) -> hidden: wrapped in a hook adapter.
              If fn has reset_offset(), it is called once before iterating.
              Use make_approx_intervention_fn, make_adaptive_exact_intervention_fn,
              or make_adaptive_approx_intervention_fn to create these callables.

    Returns:
        List of result dicts in same order as samples.
    """
    from .hooks import InterventionHook

    if isinstance(intervention, torch.Tensor):
        int_hook = InterventionHook(sae, intervention)
        int_hook.enabled = True
        hook_callable = int_hook
    else:
        fn = intervention
        if hasattr(fn, "reset_offset"):
            fn.reset_offset()

        def hook_callable(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            sl = getattr(module, '_batch_seq_lengths', None)
            if sl is not None and hasattr(fn, 'set_seq_lengths'):
                fn.set_seq_lengths(sl)
            modified = fn(hidden)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

    all_results = []
    n = len(samples)

    for start in tqdm(range(0, n, batch_size), desc=desc):
        end = min(start + batch_size, n)
        batch_samples = samples[start:end]

        input_ids, attention_mask, seq_lengths = prepare_batch_inputs(
            batch_samples, config, tokenizer, device,
        )

        metadata = []
        for sample in batch_samples:
            correct_letter, correct_idx = extract_answer(sample, config)
            metadata.append((correct_letter, correct_idx))

        hook_mgr.layer_module._batch_seq_lengths = seq_lengths
        with hook_mgr.intervention(hook_callable):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = _gather_last_logits(outputs.logits, seq_lengths)
        batch_results = _extract_results(logits, answer_token_id_list, metadata, config)
        all_results.extend(batch_results)

    return all_results
