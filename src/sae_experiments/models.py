"""LLM and tokenizer loading utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


def load_model_and_tokenizer(
    config: ModelConfig,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """Load a model and tokenizer from HuggingFace."""

    print(f"Loading model: {config.model_id}...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The model would load on CPU and run ~60x slower. "
            "Make sure a GPU is available before starting experiments."
        )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=getattr(torch, config.dtype),
    )
    model.to("cuda")
    device = torch.device("cuda")

    print(f"Model loaded")
    print(f"  Device: {device}")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    return model, tokenizer, device


_LAYER_PATHS = [
    "model.layers",                          # Llama, Mistral, Qwen, Gemma-2, Gemma-3 (older transformers)
    "model.language_model.layers",           # Gemma-3 multimodal (transformers >= 5, language_model is a Gemma3TextModel)
    "model.language_model.model.layers",     # Gemma-3 multimodal (transformers < 5, language_model is a Gemma3ForCausalLM)
    "language_model.model.layers",           # Some VLM wrappers
    "model.text_model.layers",               # Other VLMs with text_model wrapper
    "transformer.h",                         # GPT-2, GPT-Neo
    "gpt_neox.layers",                       # Pythia / GPT-NeoX
]


def _find_decoder_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
    """Locate the decoder layer list across different model architectures."""
    for attr_path in _LAYER_PATHS:
        obj = model
        try:
            for attr in attr_path.split("."):
                obj = getattr(obj, attr)
            if isinstance(obj, torch.nn.ModuleList):
                return obj
        except AttributeError:
            continue

    # Fallback: recursively search for a ModuleList of transformer decoder layers
    def _is_decoder_layer(module: torch.nn.Module) -> bool:
        return hasattr(module, "self_attn") and hasattr(module, "mlp")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0 and _is_decoder_layer(module[0]):
            return module

    raise ValueError(
        f"Cannot locate decoder layers in {type(model).__name__}. "
        f"Tried paths: {_LAYER_PATHS}. "
        f"Recursive search also failed. Top-level attributes: {[n for n, _ in model.named_children()]}. "
        f"Add the correct path for this architecture to _LAYER_PATHS."
    )


def get_layer_module(
    model: AutoModelForCausalLM,
    layer: int,
    hook_point: str = "resid_post",
    expected_n_layers: Optional[int] = None,
) -> torch.nn.Module:
    """Get the module for hook registration.

    Args:
        hook_point: "resid_post" hooks the full layer output (residual stream),
                    "mlp" hooks the MLP write as it enters the residual stream,
                    "attention" hooks the self-attention output.
        expected_n_layers: depth the model config declares. When given, the
                    resolved list must match it. `_find_decoder_layers` falls back
                    to a recursive guess when no known path matches, and on
                    Gemma-3 multimodal checkpoints that guess used to land on the
                    SigLIP vision encoder — a module text-only inputs never
                    execute, so every activation came back empty instead of
                    raising. Checking the depth turns any such miss into an
                    immediate error.

    Gemma-2 and Gemma-3 pass each sub-layer write through a further RMSNorm
    before adding it to the residual stream, and Gemma Scope trains its MLP
    SAEs on that normalised value (Lieberum et al. 2024, "Location"). Where the
    layer defines `post_feedforward_layernorm`, that module is therefore the
    trained site; the raw output of `layer.mlp` is one RMSNorm short of it.
    Llama-style layers apply no such norm, so there `layer.mlp` already is the
    write and the lookup falls back to it.
    """
    layers = _find_decoder_layers(model)
    if expected_n_layers is not None and len(layers) != expected_n_layers:
        path = next((n for n, m in model.named_modules() if m is layers), "<unknown>")
        raise ValueError(
            f"Resolved {len(layers)} decoder layers at '{path}', but the model config "
            f"declares {expected_n_layers}. The layer lookup landed on the wrong module "
            f"in {type(model).__name__}. Add the correct path for this architecture to "
            f"_LAYER_PATHS."
        )
    layer_module = layers[layer]
    if hook_point == "mlp":
        return getattr(layer_module, "post_feedforward_layernorm", layer_module.mlp)
    if hook_point == "attention":
        return layer_module.self_attn
    return layer_module
