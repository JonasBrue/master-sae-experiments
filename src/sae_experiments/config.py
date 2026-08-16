"""Configuration schema and YAML loading for all experiment components."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, model_validator


class FieldMapping(BaseModel):
    question: str
    question_fields: Optional[list[str]] = None
    choices_format: Literal[
        "dict_of_lists",
        "list_of_dicts",
        "list",
        "separate_fields",
        "digit_list",
        "fixed",
        "stringified_list",
    ]
    choices_field: Optional[str] = None
    choices_fields: Optional[list[str]] = None
    fixed_choices: Optional[list[str]] = None
    shuffle_choices: bool = False
    answer: str


class ModelConfig(BaseModel):
    model_id: str
    short_name: str
    d_model: int
    n_layers: int
    dtype: str = "bfloat16"


class SAEConfig(BaseModel):
    name: str
    architecture: Literal["relu", "batchtopk", "jump_relu"]
    repo: str
    filename: str
    config_filename: Optional[str] = None
    layer: int
    available_layers: list[int]
    d_model: int
    d_hidden: int
    top_k: Optional[int] = None
    file_format: Literal["pth", "pt", "safetensors", "npz"] = "pt"
    hook_point: Literal["resid_post", "mlp", "attention"] = "resid_post"
    center_input: bool = True
    model: str


class BenchmarkConfig(BaseModel):
    name: str
    dataset_name: str
    dataset_config: Optional[str] = None
    dataset_split_train: str = "train"
    dataset_split_test: str = "test"
    combine_splits: bool = False
    category_field: Optional[str] = None
    category_values: Optional[list[str]] = None
    expected_num_choices: Optional[int] = None
    choices: list[str]
    answer_key_type: Literal["letter", "index_0", "index_1", "digit", "text_match"]
    field_mapping: FieldMapping
    prompt_instruction: str
    special_formatting: Optional[Literal["winogrande", "hellaswag", "letter_counting"]] = None
    cache_name: str


class GridConfig(BaseModel):
    top_k_positive: list[int] = [3, 5, 7, 10, 15]
    top_k_negative: list[int] = [0, 3, 5]
    strength: list[float] = [0.5, 1.0, 1.5]
    ranking_criterion: list[str] = ["causal_delta", "freq_diff"]
    weighting_mode: list[str] = ["uniform"]

    @model_validator(mode="after")
    def _normalize_strength(self) -> "GridConfig":
        # Same precision normalisation as CausalConfig.causal_test_strength so
        # float-equality joins between grid cells and Phase-2 results are exact.
        self.strength = sorted({round(float(s), 6) for s in self.strength})
        return self


class AdaptiveConfig(BaseModel):
    """Configuration for difficulty-adaptive steering.

    Adaptive steering adjusts intervention strength based on question difficulty,
    preventing harm on easy questions while maximizing benefit on hard ones.
    """
    enabled: bool = False
    mode: Literal["step", "linear", "gaussian", "constant"] = "step"
    difficulty_metric: Literal["p_correct", "max_confidence"] = "p_correct"
    p_min: float = 0.15  # Lower bound for intervention
    p_sweet: float = 0.35  # Upper bound of sweet spot (for step mode)
    p_max: float = 0.50  # Upper bound for intervention
    mu: float = 0.25  # Center of Gaussian (for gaussian mode)
    sigma: float = 0.15  # Std of Gaussian (for gaussian mode)
    transition_strength: float = 0.3  # Strength in transition zone (for step mode)


class CausalConfig(BaseModel):
    min_p_correct: float = 0.00
    max_p_correct: float = 1.00
    n_candidate_features_pos: int = 50
    n_candidate_features_neg: int = 50
    causal_test_strength: Optional[list[float]] = None
    n_folds: int = 5
    seed: int = 42
    train_fraction: float = 0.50
    val_fraction: float = 0.25
    adaptive: AdaptiveConfig = AdaptiveConfig()  # Adaptive steering config
    combination_stability_check: bool = False
    combination_stability_n_pairs: int = 12

    @model_validator(mode="after")
    def validate_causal(self) -> "CausalConfig":
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if self.train_fraction + self.val_fraction > 1.0:
            raise ValueError(
                f"train_fraction ({self.train_fraction}) + val_fraction "
                f"({self.val_fraction}) must be <= 1.0"
            )
        if self.causal_test_strength is not None:
            if len(self.causal_test_strength) == 0:
                raise ValueError("causal_test_strength must contain at least one value")
            # Deduplicate, sort, and round to a stable precision so downstream
            # float-equality comparisons against grid.strength never fail due
            # to representation drift.
            self.causal_test_strength = sorted({round(float(s), 6) for s in self.causal_test_strength})
        if self.combination_stability_n_pairs < 0:
            raise ValueError(
                f"combination_stability_n_pairs must be >= 0, got {self.combination_stability_n_pairs}"
            )
        return self


class ExperimentConfig(BaseModel):
    model: ModelConfig
    sae: SAEConfig
    benchmark: BenchmarkConfig
    grid: GridConfig = GridConfig()
    causal: CausalConfig = CausalConfig()
    batch_size: int = 32
    output_dir: str = "results"
    cache_dir: str = "cache"
    force_rerun_screening: bool = False

    @model_validator(mode="after")
    def validate_compatibility(self) -> "ExperimentConfig":
        if self.sae.layer >= self.model.n_layers:
            raise ValueError(
                f"SAE layer {self.sae.layer} out of range for model with "
                f"{self.model.n_layers} layers (max: {self.model.n_layers - 1})"
            )
        if self.sae.d_model != self.model.d_model:
            raise ValueError(
                f"SAE d_model ({self.sae.d_model}) != model d_model ({self.model.d_model})"
            )
        if self.causal.causal_test_strength is None:
            # Auto-populate from grid. If causal_delta is not in the grid,
            # strength-specific marginal deltas are never read by Phase 3/4,
            # so a single representative strength is enough — saves Phase 2
            # the |grid.strength|× cost multiplier.
            if "causal_delta" in self.grid.ranking_criterion:
                self.causal.causal_test_strength = list(self.grid.strength)
            else:
                self.causal.causal_test_strength = [self.grid.strength[0]]
        elif (
            "causal_delta" in self.grid.ranking_criterion
            and len(self.causal.causal_test_strength) > 1
        ):
            # Single-strength mode (len == 1) is always allowed: Phase 3 reuses
            # the one available ranking for every grid.strength cell. Only the
            # explicit multi-strength opt-in requires grid ⊆ test_strength.
            missing = set(self.grid.strength) - set(self.causal.causal_test_strength)
            if missing:
                raise ValueError(
                    f"Multi-strength mode requires every grid.strength to have a matching "
                    f"Phase-2 test, but {sorted(missing)} are missing from "
                    f"causal.causal_test_strength ({self.causal.causal_test_strength})."
                )
        return self


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment_config(
    *,
    sae_path: Path,
    layer: int,
    benchmark_path: Path,
    overrides: Optional[dict] = None,
) -> ExperimentConfig:
    """Load and compose an ExperimentConfig from an SAE YAML (which references its model)."""
    sae_data = _load_yaml(sae_path)

    available = sae_data.get("available_layers", [])
    if layer not in available:
        raise ValueError(
            f"Layer {layer} not available for this SAE. Choose from: {available}"
        )

    sae_data["layer"] = layer
    for key in ("filename", "config_filename"):
        val = sae_data.get(key)
        if val and "{layer}" in str(val):
            sae_data[key] = val.format(layer=layer)

    model_ref = sae_data["model"]
    model_path = (sae_path.parent / model_ref).resolve()
    if not model_path.exists():
        model_path = Path(model_ref)

    data = {
        "model": _load_yaml(model_path),
        "sae": sae_data,
        "benchmark": _load_yaml(benchmark_path),
    }
    if overrides:
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            d = data
            for part in parts[:-1]:
                d = d.setdefault(part, {})
            d[parts[-1]] = value
    return ExperimentConfig(**data)
