"""Activation cache, intervention hooks, and hook lifecycle management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
# Right-padding utilities: per-sample last-token access
# ═══════════════════════════════════════════════════════════════════════════════


def _gather_last_tokens(tensor: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Extract last real token per sample from [B, seq, D] tensor. Returns [B, 1, D]."""
    B, _, D = tensor.shape
    idx = (seq_lengths - 1).view(B, 1, 1).expand(-1, -1, D)
    return torch.gather(tensor, 1, idx)


def _scatter_last_tokens(
    tensor: torch.Tensor, new_last: torch.Tensor, seq_lengths: torch.Tensor,
) -> torch.Tensor:
    """Replace last real token per sample in [B, seq, D] tensor. Returns modified copy."""
    B, _, D = tensor.shape
    idx = (seq_lengths - 1).view(B, 1, 1).expand(-1, -1, D)
    return tensor.clone().scatter_(1, idx, new_last)


def _gather_last_logits(logits: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """Extract last-token logits per sample from [B, seq, vocab]. Returns [B, vocab]."""
    B, _, V = logits.shape
    idx = (seq_lengths - 1).view(B, 1, 1).expand(-1, -1, V)
    return torch.gather(logits, 1, idx).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Activation cache
# ═══════════════════════════════════════════════════════════════════════════════


class ActivationCache:
    """Cache for hidden states during forward pass."""

    def __init__(self):
        self.activations: Optional[torch.Tensor] = None

    def hook(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.activations = hidden.detach()

    def get_last_token(self, seq_lengths: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.activations is None:
            return None
        if seq_lengths is not None:
            return _gather_last_tokens(self.activations, seq_lengths)
        return self.activations[:, -1:, :]

    def clear(self):
        self.activations = None


# ═══════════════════════════════════════════════════════════════════════════════
# Intervention hooks
# ═══════════════════════════════════════════════════════════════════════════════


class InterventionHook:
    """Exact intervention: encode -> add -> clamp(min=0) -> decode (requires SAE at runtime)."""

    def __init__(self, sae_model: nn.Module, intervention_vec: torch.Tensor):
        self.sae = sae_model
        self.vec = intervention_vec
        self.enabled = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        seq_lengths = getattr(module, '_batch_seq_lengths', None)

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths).to(self.sae.dtype)
        else:
            last_hidden = hidden[:, -1:, :].to(self.sae.dtype)

        features = self.sae.encode(last_hidden)
        reconstructed = self.sae.decode(features)
        error = last_hidden - reconstructed

        features = features + self.vec.unsqueeze(0).unsqueeze(0)
        features = torch.clamp(features, min=0)

        modified_last = (self.sae.decode(features) + error).to(hidden.dtype)

        if seq_lengths is not None:
            modified = _scatter_last_tokens(hidden, modified_last, seq_lengths)
        else:
            context = hidden[:, :-1, :]
            modified = torch.cat([context, modified_last], dim=1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


class ApproximatedInterventionHook:
    """Approximated intervention: precomputed vector added directly to hidden state (no SAE at runtime)."""

    def __init__(self, precomputed_hidden_vec: torch.Tensor):
        self.vec = precomputed_hidden_vec
        self.enabled = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        seq_lengths = getattr(module, '_batch_seq_lengths', None)

        if seq_lengths is not None:
            last_hidden = _gather_last_tokens(hidden, seq_lengths)
        else:
            last_hidden = hidden[:, -1:, :]

        modified_last = last_hidden + self.vec.unsqueeze(0).unsqueeze(0).to(hidden.dtype)

        if seq_lengths is not None:
            modified = _scatter_last_tokens(hidden, modified_last, seq_lengths)
        else:
            context = hidden[:, :-1, :]
            modified = torch.cat([context, modified_last], dim=1)

        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified


# ═══════════════════════════════════════════════════════════════════════════════
# Hook lifecycle management
# ═══════════════════════════════════════════════════════════════════════════════


class HookManager:
    """Manages hook lifecycle on a single layer, preventing leaks."""

    def __init__(self, layer_module: nn.Module):
        self.layer_module = layer_module
        self._cache_handle: Optional[torch.utils.hooks.RemovableHook] = None
        self.cache = ActivationCache()

    def register_cache_hook(self) -> None:
        """Install the activation cache hook."""
        self.remove_cache_hook()
        self._cache_handle = self.layer_module.register_forward_hook(self.cache.hook)

    def remove_cache_hook(self) -> None:
        if self._cache_handle is not None:
            self._cache_handle.remove()
            self._cache_handle = None

    @contextmanager
    def intervention(self, hook_callable):
        """Context manager that temporarily replaces the cache hook with an intervention hook.

        Usage:
            int_hook = InterventionHook(sae, vec)
            int_hook.enabled = True
            with hook_mgr.intervention(int_hook):
                outputs = model(input_ids=...)
        """
        self.remove_cache_hook()
        handle = self.layer_module.register_forward_hook(hook_callable)
        try:
            yield
        finally:
            handle.remove()
            self.register_cache_hook()
