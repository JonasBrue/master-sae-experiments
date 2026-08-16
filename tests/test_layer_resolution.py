"""Tests for decoder-layer resolution and the empty-grid abort.

Regression cover for the EXP-468/469 failure: on Gemma-3 multimodal checkpoints
(`Gemma3ForConditionalGeneration`, used by gemma-3-4b-pt/it and gemma-3-12b-it)
none of the `_LAYER_PATHS` matched under transformers >= 5, so the recursive
fallback returned the SigLIP *vision* encoder layers instead of the text decoder.
Layers >= vision depth raised IndexError; layers below it silently hooked a module
that never runs on text-only input, yielding zero features for every sample.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch
import transformers as tf

from sae_experiments import pipeline as pipeline_mod
from sae_experiments.models import _find_decoder_layers, get_layer_module

TEXT_LAYERS = 34
VISION_LAYERS = 27


def _gemma3_multimodal() -> tf.Gemma3ForConditionalGeneration:
    """Build a gemma-3-4b-shaped multimodal model on the meta device (no weights)."""
    config = tf.Gemma3Config(
        text_config=dict(
            num_hidden_layers=TEXT_LAYERS, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=128,
        ),
        vision_config=dict(
            num_hidden_layers=VISION_LAYERS, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, image_size=224, patch_size=14,
        ),
    )
    with torch.device("meta"):
        return tf.Gemma3ForConditionalGeneration(config)


def test_finds_text_decoder_not_vision_tower_on_gemma3_multimodal():
    model = _gemma3_multimodal()

    layers = _find_decoder_layers(model)

    assert layers is model.model.language_model.layers
    assert len(layers) == TEXT_LAYERS


def test_layer_above_vision_depth_resolves_on_gemma3_multimodal():
    """EXP-468/469 asked for layer 29, which exceeds the 27-layer vision tower."""
    model = _gemma3_multimodal()

    layer_module = get_layer_module(model, 29, "resid_post")

    assert type(layer_module).__name__ == "Gemma3DecoderLayer"


def test_get_layer_module_rejects_layer_count_mismatch():
    """A fallback that finds the wrong ModuleList must fail loudly, not silently."""
    model = _gemma3_multimodal()

    with pytest.raises(ValueError, match="34"):
        get_layer_module(model, 9, "resid_post", expected_n_layers=TEXT_LAYERS + 5)


def test_get_layer_module_accepts_matching_layer_count():
    model = _gemma3_multimodal()

    layer_module = get_layer_module(model, 9, "resid_post", expected_n_layers=TEXT_LAYERS)

    assert type(layer_module).__name__ == "Gemma3DecoderLayer"


# ── empty-grid abort must not report success ──────────────────────────────────


@pytest.fixture
def stubbed_pipeline(monkeypatch, tmp_path):
    """A Pipeline whose phases are stubbed out and whose grid search finds nothing."""
    split_keys = [
        "screening_data", "filtered_data", "train_data", "val_data",
        "test_data_filtered", "test_data_full", "grid_eval_data",
        "holdout_data", "holdout_filtered", "answer_diagnostics",
    ]

    monkeypatch.setattr(pipeline_mod.Pipeline, "setup", lambda self: None)
    monkeypatch.setattr(pipeline_mod.Pipeline, "_predict_kwargs", lambda self: {})
    monkeypatch.setattr(pipeline_mod.Pipeline, "cleanup", lambda self: None)
    monkeypatch.setattr(pipeline_mod, "run_screening_and_split",
                        lambda *a, **k: {key: [] for key in split_keys})
    monkeypatch.setattr(pipeline_mod, "run_phase1_freq_diff",
                        lambda *a, **k: (None, [], []))
    monkeypatch.setattr(pipeline_mod, "run_phase2_causal_screening",
                        lambda *a, **k: (pd.DataFrame(), pd.DataFrame(), 0.5))
    monkeypatch.setattr(pipeline_mod, "run_pair_stability_check",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(pipeline_mod, "compute_baselines",
                        lambda *a, **k: ([], 0.5, [], 0.5))
    # Phase 1/2 found no candidates -> Phase 3 returns nothing.
    monkeypatch.setattr(pipeline_mod, "run_phase3_grid_search",
                        lambda *a, **k: pd.DataFrame())

    class _Cfg:
        batch_size = 1
        force_rerun_screening = False
        output_dir = str(tmp_path)
        cache_dir = str(tmp_path)

        class model:
            pass

        class causal:
            pass

        class grid:
            pass

    return pipeline_mod.Pipeline(_Cfg())


def test_empty_grid_search_raises_instead_of_returning_success(stubbed_pipeline):
    """Zero feature candidates must surface as a non-zero exit, not Result: SUCCESS."""
    with pytest.raises(RuntimeError, match="grid search"):
        stubbed_pipeline.run()
