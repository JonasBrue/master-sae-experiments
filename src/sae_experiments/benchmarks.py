"""Dataset loading, question/answer extraction, and prompt formatting."""

from __future__ import annotations

import ast
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import BenchmarkConfig


def load_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Load benchmark dataset and return list of question dicts."""

    print(f"Loading benchmark: {config.name}...")
    
    # Check if it's a local letter_counting dataset
    if config.dataset_name == "letter_counting" or config.dataset_name.startswith("local/"):
        return load_local_benchmark(config)
    
    print(f"  Dataset: {config.dataset_name}")
    if config.dataset_config:
        print(f"  Config: {config.dataset_config}")

    ds = load_dataset(config.dataset_name, config.dataset_config)

    available_splits = list(ds.keys())
    for split_name in available_splits:
        print(f"    {split_name}: {len(ds[split_name])} questions")

    if config.combine_splits:
        all_questions = []
        for split in [config.dataset_split_train, config.dataset_split_test]:
            if split in ds:
                all_questions.extend(list(ds[split]))
            else:
                print(f"  Warning: split '{split}' not found, skipping...")
        print(f"  Combined: {len(all_questions)} questions")
    else:
        if config.dataset_split_test in ds:
            all_questions = list(ds[config.dataset_split_test])
            print(f"  Using '{config.dataset_split_test}': {len(all_questions)} questions")
        else:
            fallback_split = available_splits[0]
            all_questions = list(ds[fallback_split])
            print(f"  Warning: split '{config.dataset_split_test}' not found, using '{fallback_split}'")

    if len(all_questions) == 0:
        raise ValueError("No questions loaded! Check the split configuration.")

    # Optional category filtering (e.g., MMLU subject, MMLU-Pro category).
    if config.category_field and config.category_values:
        allowed = {str(v).strip() for v in config.category_values}
        before = len(all_questions)
        all_questions = [
            q for q in all_questions
            if str(q.get(config.category_field, "")).strip() in allowed
        ]
        print(
            f"  Category filter on '{config.category_field}' ({sorted(allowed)}): "
            f"{before} -> {len(all_questions)}"
        )

    # Optional strict choice-count filtering for variable-choice datasets.
    if config.expected_num_choices is not None:
        expected = int(config.expected_num_choices)
        before = len(all_questions)
        filtered_questions = []
        for q in all_questions:
            try:
                n_choices = len(extract_choices(q, config))
            except Exception:
                n_choices = -1
            if n_choices == expected:
                filtered_questions.append(q)
        all_questions = filtered_questions
        print(f"  Choice-count filter ({expected}): {before} -> {len(all_questions)}")

    if len(all_questions) == 0:
        raise ValueError(
            "No questions left after filtering. "
            "Check category_field/category_values and expected_num_choices."
        )

    return all_questions


def load_local_benchmark(config: BenchmarkConfig) -> List[Dict]:
    """Load a local benchmark dataset (e.g., letter_counting)."""
    
    # Check for letter_counting cache
    cache_name = getattr(config, 'cache_name', None)
    if cache_name:
        cache_path = Path(f"/workspace/{cache_name}")
        if cache_path.exists():
            print(f"  Loading from cache: {cache_path}")
            all_questions = []
            for split_file in cache_path.glob("*.json"):
                with open(split_file) as f:
                    all_questions.extend(json.load(f))
            print(f"  Loaded: {len(all_questions)} questions")
            return all_questions
    
    # Generate letter_counting dataset if needed
    if config.dataset_name == "letter_counting" or "letter" in config.dataset_name.lower():
        from .letter_counting import generate_letter_counting_benchmark
        
        output_dir = Path("/workspace/letter_counting_v3_cleaned")
        paths = generate_letter_counting_benchmark(output_dir)
        
        all_questions = []
        for split_file in output_dir.glob("*.json"):
            with open(split_file) as f:
                all_questions.extend(json.load(f))
        
        print(f"  Generated: {len(all_questions)} questions")
        return all_questions
    
    raise ValueError(f"Unknown local benchmark: {config.dataset_name}")


def get_answer_token_ids(
    choices: List[str], tokenizer: AutoTokenizer
) -> Dict[str, int]:
    """Get token IDs for answer letters."""
    answer_token_ids = {}

    for letter in choices:
        variants = [letter, f" {letter}", letter.lower(), f" {letter.lower()}"]
        for variant in variants:
            tokens = tokenizer.encode(variant, add_special_tokens=False)
            if len(tokens) == 1:
                answer_token_ids[letter] = tokens[0]
                break

        if letter not in answer_token_ids:
            raise ValueError(f"Could not find a single token for '{letter}'!")

    return answer_token_ids


def extract_question(sample: Dict, config: BenchmarkConfig) -> str:
    """Extract the question text from a sample."""
    field = config.field_mapping.question
    question_fields = config.field_mapping.question_fields or []

    if question_fields:
        parts = [str(sample.get(f, "")).strip() for f in question_fields]
        return "\n\n".join([p for p in parts if p])

    if field == "ctx":
        if "ctx" in sample:
            return sample["ctx"]
        elif "ctx_a" in sample and "ctx_b" in sample:
            ctx_a = sample.get("ctx_a", "")
            ctx_b = sample.get("ctx_b", "")
            return (ctx_a + " " + ctx_b).strip()

    return sample.get(field, "")


def extract_choices(sample: Dict, config: BenchmarkConfig) -> List[str]:
    """Extract answer choices from a sample."""
    fmt = config.field_mapping.choices_format

    if fmt == "dict_of_lists":
        return sample["choices"]["text"]
    elif fmt == "list_of_dicts":
        return [c["text"] for c in sample["choices"]]
    elif fmt == "list":
        field = config.field_mapping.choices_field or "endings"
        return sample[field]
    elif fmt == "separate_fields":
        fields = config.field_mapping.choices_fields or []
        choices = [sample[f] for f in fields]
        if config.field_mapping.shuffle_choices and config.answer_key_type == "text_match":
            # Deterministic per-sample shuffle to avoid fixed-position bias.
            seed_src = str(extract_question(sample, config)) + "||" + "||".join(map(str, choices))
            seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            rng.shuffle(choices)
        return choices
    elif fmt == "digit_list":
        # Letter counting uses digit choices
        return sample.get("choices", ["0", "1", "2", "3", "4", "5"])
    elif fmt == "fixed":
        return config.field_mapping.fixed_choices or config.choices
    elif fmt == "stringified_list":
        field = config.field_mapping.choices_field or "choices"
        raw = sample.get(field, [])
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                pass
        return []
    else:
        raise ValueError(f"Unknown choices_format: {fmt}")


def extract_answer(sample: Dict, config: BenchmarkConfig) -> Tuple[str, int]:
    """Extract the correct answer letter and its index."""
    answer_field = config.field_mapping.answer
    raw_answer = sample[answer_field]
    key_type = config.answer_key_type
    choices = config.choices

    if key_type == "letter":
        letter = str(raw_answer).upper().strip()
        if letter in ["1", "2", "3", "4", "5"]:
            idx = int(letter) - 1
            letter = choices[idx] if idx < len(choices) else choices[0]
        else:
            try:
                idx = choices.index(letter)
            except ValueError:
                idx = 0
                letter = choices[0]
        return letter, idx

    elif key_type == "index_0":
        idx = int(raw_answer)
        idx = max(0, min(idx, len(choices) - 1))
        letter = choices[idx]
        return letter, idx

    elif key_type == "index_1":
        idx = int(raw_answer) - 1
        idx = max(0, min(idx, len(choices) - 1))
        letter = choices[idx]
        return letter, idx

    elif key_type == "digit":
        # For letter counting - answer is a digit string
        answer_str = str(raw_answer)
        try:
            idx = choices.index(answer_str)
        except ValueError:
            idx = 0
            answer_str = choices[0]
        return answer_str, idx
    elif key_type == "text_match":
        answer_str = str(raw_answer).strip().lower()
        choice_text = [str(c).strip().lower() for c in extract_choices(sample, config)]
        idx = 0
        for i, c in enumerate(choice_text):
            if c == answer_str:
                idx = i
                break
        idx = max(0, min(idx, len(choices) - 1))
        return choices[idx], idx

    else:
        raise ValueError(f"Unknown answer_key_type: {key_type}")


def format_prompt(sample: Dict, config: BenchmarkConfig) -> str:
    """Create the prompt string for a sample."""
    question = extract_question(sample, config)
    choices_text = extract_choices(sample, config)
    choices_labels = config.choices

    special = config.special_formatting

    if special == "winogrande":
        sentence = question.replace("_", "[BLANK]")
        options_str = "\n".join(
            f"{label}. {text}" for label, text in zip(choices_labels, choices_text)
        )
        return (
            f'{config.prompt_instruction}\n'
            f'Respond with ONLY the letter (A or B).\n\n'
            f'Sentence: "{sentence}"\n\n'
            f'Which word or phrase should replace [BLANK]?\n\n'
            f'{options_str}\n\n'
            f'Answer:'
        )

    elif special == "hellaswag":
        options_str = "\n".join(
            f"{label}. {text}" for label, text in zip(choices_labels, choices_text)
        )
        return (
            f'{config.prompt_instruction}\n'
            f'Respond with ONLY the letter of the best ending (A, B, C, or D).\n\n'
            f'Context: {question}\n\n'
            f'Possible endings:\n{options_str}\n\n'
            f'Answer:'
        )
    
    elif special == "letter_counting":
        # Letter counting format - choices are digits
        options_str = "\n".join(
            f"{label}. {text}" for label, text in zip(choices_labels, choices_text)
        )
        return (
            f'{config.prompt_instruction}\n'
            f'Respond with ONLY the digit (0-5).\n\n'
            f'{question}\n\n'
            f'{options_str}\n\n'
            f'Answer:'
        )

    # Standard formatting
    options_str = "\n".join(
        f"{label}. {text}" for label, text in zip(choices_labels, choices_text)
    )

    if len(choices_labels) == 2:
        choices_str = f"{choices_labels[0]} or {choices_labels[1]}"
    else:
        choices_str = ", ".join(choices_labels[:-1]) + f", or {choices_labels[-1]}"

    return (
        f'{config.prompt_instruction}\n'
        f'Respond with ONLY the letter of the correct answer ({choices_str}).\n\n'
        f'Question: {question}\n\n'
        f'{options_str}\n\n'
        f'Answer:'
    )


def prepare_input(
    sample: Dict,
    config: BenchmarkConfig,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare tokenized input with chat template."""
    base_prompt = format_prompt(sample, config)

    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": base_prompt}]
        full_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        full_text = base_prompt

    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=2048,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask
