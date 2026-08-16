"""Tests for ExperimentConfig fields used by the per-experiment-output refactor."""

from sae_experiments.config import (
    BenchmarkConfig,
    ExperimentConfig,
    FieldMapping,
    ModelConfig,
    SAEConfig,
)


def _minimal_configs() -> tuple[ModelConfig, SAEConfig, BenchmarkConfig]:
    """Return a minimal valid (model, sae, benchmark) triple for ExperimentConfig.

    Values are intentionally small/synthetic — these tests only exercise the
    `cache_dir` plumbing, not the actual loading code, so we never instantiate
    a real model or SAE.
    """
    model = ModelConfig(
        model_id="hf/whatever",
        short_name="dummy",
        d_model=4,
        n_layers=2,
        dtype="float32",
    )
    sae = SAEConfig(
        name="dummy_sae",
        architecture="relu",
        repo="hf/dummy",
        filename="dummy.pt",
        layer=0,
        available_layers=[0],
        d_model=4,
        d_hidden=8,
        hook_point="resid_post",
        model="dummy",
    )
    benchmark = BenchmarkConfig(
        name="dummy_bench",
        dataset_name="dummy_dataset",
        choices=["A", "B"],
        answer_key_type="letter",
        field_mapping=FieldMapping(
            question="question",
            choices_format="fixed",
            fixed_choices=["A", "B"],
            answer="answer",
        ),
        prompt_instruction="Answer the question.",
        cache_name="dummy_bench",
    )
    return model, sae, benchmark


def test_cache_dir_defaults_to_cache():
    """ExperimentConfig.cache_dir defaults to 'cache' so legacy runners keep shared cache."""
    model, sae, benchmark = _minimal_configs()
    config = ExperimentConfig(model=model, sae=sae, benchmark=benchmark)
    assert config.cache_dir == "cache"


def test_cache_dir_accepts_override():
    """Passing cache_dir='results/EXP-300' overrides the default."""
    model, sae, benchmark = _minimal_configs()
    config = ExperimentConfig(
        model=model, sae=sae, benchmark=benchmark, cache_dir="results/EXP-300"
    )
    assert config.cache_dir == "results/EXP-300"
