"""SAE classes (ReLU and BatchTopK) and loading utilities."""

from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .config import SAEConfig


class ReLUSparseAutoEncoder(nn.Module):
    """Goodfire-style ReLU SAE."""

    def __init__(self, d_in: int, d_hidden: int, device: torch.device, dtype=torch.bfloat16):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        self.architecture = "relu"

        self.encoder_linear = nn.Linear(d_in, d_hidden)
        self.decoder_linear = nn.Linear(d_hidden, d_in)
        self.to(device, dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_linear(x)

    def load_pretrained_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        key_variants = {
            "encoder_linear.weight": ["encoder_linear.weight", "encoder.weight", "W_enc", "w_enc"],
            "encoder_linear.bias": ["encoder_linear.bias", "encoder.bias", "b_enc"],
            "decoder_linear.weight": ["decoder_linear.weight", "decoder.weight", "W_dec", "w_dec"],
            "decoder_linear.bias": ["decoder_linear.bias", "decoder.bias", "b_dec", "b_pre"],
        }

        print(f"  State dict keys: {list(state_dict.keys())}")

        for our_key, their_keys in key_variants.items():
            loaded = None
            used_key = None

            for their_key in their_keys:
                if their_key in state_dict:
                    loaded = state_dict[their_key]
                    used_key = their_key
                    break

            if loaded is None:
                print(f"  Warning: no key found for {our_key}, tried {their_keys}")
                continue

            param = self.get_parameter(our_key)
            loaded = loaded.to(self.dtype).to(self.device)

            if param.shape != loaded.shape:
                if len(loaded.shape) == 2 and param.shape == loaded.T.shape:
                    loaded = loaded.T
                    print(f"  {used_key} -> {our_key} (transposed)")
                else:
                    print(f"  Warning: shape mismatch for {our_key}: {param.shape} vs {loaded.shape}")
                    continue
            else:
                print(f"  {used_key} -> {our_key}")

            param.data.copy_(loaded)

        print("  Weights loaded")


class BatchTopKSparseAutoEncoder(nn.Module):
    """AndyRDT-style BatchTopK SAE."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype=torch.bfloat16,
        top_k: Optional[int] = None,
        center_input: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        self.top_k = top_k
        self.center_input = center_input
        self.architecture = "batchtopk"

        self.W_enc = nn.Parameter(torch.zeros(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.zeros(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.to(device, dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = (x - self.b_dec) if self.center_input else x
        pre_acts = x_centered @ self.W_enc + self.b_enc

        if self.top_k is not None and self.top_k < self.d_hidden:
            topk_values, topk_indices = torch.topk(pre_acts, self.top_k, dim=-1)
            acts = torch.zeros_like(pre_acts)
            acts.scatter_(-1, topk_indices, torch.relu(topk_values))
        else:
            acts = torch.relu(pre_acts)

        return acts

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec

    def load_pretrained_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        key_variants = {
            "W_enc": ["W_enc", "w_enc", "encoder.weight", "encode.weight"],
            "b_enc": ["b_enc", "encoder.bias", "encode.bias"],
            "W_dec": ["W_dec", "w_dec", "decoder.weight", "decode.weight"],
            "b_dec": ["b_dec", "decoder.bias", "decode.bias", "b_pre"],
        }

        print(f"  State dict keys: {list(state_dict.keys())}")

        for our_key, their_keys in key_variants.items():
            loaded = None
            used_key = None

            for their_key in their_keys:
                if their_key in state_dict:
                    loaded = state_dict[their_key]
                    used_key = their_key
                    break

            if loaded is None:
                print(f"  Warning: no key found for {our_key}, tried {their_keys}")
                continue

            param = getattr(self, our_key)
            loaded = loaded.to(self.dtype).to(self.device)

            if param.shape != loaded.shape:
                if param.shape == loaded.T.shape:
                    loaded = loaded.T
                    print(f"  {used_key} -> {our_key} (transposed)")
                else:
                    print(f"  Warning: shape mismatch for {our_key}: {param.shape} vs {loaded.shape}")
                    continue
            else:
                print(f"  {used_key} -> {our_key}")

            param.data.copy_(loaded)

        print("  Weights loaded")


class JumpReLUSparseAutoEncoder(nn.Module):
    """JumpReLU SAE (Gemma Scope style).

    Activation: JumpReLU_theta(z) := z * H(z - theta)  (Eq. 3 in Gemma Scope paper)
    Encoder:    f(x) = JumpReLU_theta(W_enc @ x + b_enc)  (Eq. 1/10)
    Decoder:    x_hat = W_dec @ f + b_dec  (Eq. 2/11)

    Gemma Scope publishes reparameterized weights (Eq. 10-11) where b_dec centering
    is absorbed into b_enc, so center_input should be False for these.

    Reference: Lieberum et al., "Gemma Scope: Open Sparse Autoencoders Everywhere
    All At Once on Gemma 2", arXiv:2408.05147, 2024.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype=torch.bfloat16,
        center_input: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.dtype = dtype
        self.center_input = center_input
        self.architecture = "jump_relu"

        self.W_enc = nn.Parameter(torch.zeros(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.zeros(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.threshold = nn.Parameter(torch.zeros(d_hidden))

        self.to(device, dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = (x - self.b_dec) if self.center_input else x
        pre_acts = x_centered @ self.W_enc + self.b_enc
        return pre_acts * (pre_acts > self.threshold).to(pre_acts.dtype)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_dec + self.b_dec

    def load_pretrained_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        key_variants = {
            "W_enc": ["W_enc", "w_enc", "encoder.weight", "encode.weight"],
            "b_enc": ["b_enc", "encoder.bias", "encode.bias"],
            "W_dec": ["W_dec", "w_dec", "decoder.weight", "decode.weight"],
            "b_dec": ["b_dec", "decoder.bias", "decode.bias", "b_pre"],
            "threshold": ["threshold", "thresholds", "jump_threshold"],
        }

        print(f"  State dict keys: {list(state_dict.keys())}")

        for our_key, their_keys in key_variants.items():
            loaded = None
            used_key = None

            for their_key in their_keys:
                if their_key in state_dict:
                    loaded = state_dict[their_key]
                    used_key = their_key
                    break

            if loaded is None:
                if our_key == "threshold":
                    print("  No threshold key found, defaulting to zeros")
                    continue
                print(f"  Warning: no key found for {our_key}, tried {their_keys}")
                continue

            param = getattr(self, our_key)
            loaded = loaded.to(self.dtype).to(self.device)

            if param.shape != loaded.shape:
                if len(loaded.shape) == 2 and param.shape == loaded.T.shape:
                    loaded = loaded.T
                    print(f"  {used_key} -> {our_key} (transposed)")
                else:
                    print(f"  Warning: shape mismatch for {our_key}: {param.shape} vs {loaded.shape}")
                    continue
            else:
                print(f"  {used_key} -> {our_key}")

            param.data.copy_(loaded)

        print("  Weights loaded")


def load_sae(config: SAEConfig, device: torch.device) -> nn.Module:
    """Load an SAE from HuggingFace based on the config."""

    print(f"Loading SAE: {config.name}...")
    print(f"  Architecture: {config.architecture}")
    print(f"  Repo: {config.repo}")
    print(f"  File: {config.filename}")

    if config.architecture == "relu":
        sae_path = hf_hub_download(
            repo_id=config.repo,
            filename=config.filename,
            repo_type="model",
        )

        if config.file_format == "safetensors" or config.filename.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as safetensors_load_file
            except ImportError as e:
                raise ImportError(
                    "Loading SAE safetensors requires 'safetensors'. "
                    "Install via: pip install safetensors"
                ) from e
            state_dict = safetensors_load_file(sae_path, device=str(device))
        else:
            state_dict = torch.load(sae_path, weights_only=True, map_location=device)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        d_hidden = config.d_hidden
        sae = ReLUSparseAutoEncoder(config.d_model, d_hidden, device)
        sae.load_pretrained_weights(state_dict)

        print(f"ReLU SAE loaded: {sae.d_hidden} features")

    elif config.architecture == "batchtopk":
        d_in = config.d_model
        d_hidden = config.d_hidden
        top_k = config.top_k

        # Load config file if available
        if config.config_filename:
            try:
                config_path = hf_hub_download(
                    repo_id=config.repo,
                    filename=config.config_filename,
                    repo_type="model",
                )
                with open(config_path) as f:
                    sae_config = json.load(f)
                print(f"  Config loaded: {sae_config}")

                d_in = sae_config.get("d_in", d_in)
                d_hidden = sae_config.get(
                    "d_sae", sae_config.get(
                        "d_hidden", sae_config.get("num_latents", d_hidden)
                    )
                )
                # Support both EleutherAI cfg ("k") and SAELens cfg ("activation_fn_kwargs")
                if config.top_k is None:
                    afk = sae_config.get("activation_fn_kwargs", {})
                    activation_fn = (
                        sae_config.get("activation_fn")
                        or sae_config.get("activation_fn_str")
                        or ""
                    )
                    if isinstance(activation_fn, str):
                        activation_fn = activation_fn.lower()

                    if activation_fn == "topk":
                        top_k = afk.get("k", sae_config.get("k", sae_config.get("top_k", top_k)))
                    elif activation_fn in {"relu", "jump_relu"}:
                        top_k = None
                    else:
                        top_k = sae_config.get("k", sae_config.get("top_k", top_k))
                else:
                    top_k = config.top_k
            except Exception as e:
                print(f"  Warning: could not load config: {e}")
                print("  Using default values")

        sae_path = hf_hub_download(
            repo_id=config.repo,
            filename=config.filename,
            repo_type="model",
        )

        if config.file_format == "safetensors" or config.filename.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as safetensors_load_file
            except ImportError as e:
                raise ImportError(
                    "Loading SAE safetensors requires 'safetensors'. "
                    "Install via: pip install safetensors"
                ) from e
            state_dict = safetensors_load_file(sae_path, device=str(device))
        else:
            state_dict = torch.load(sae_path, weights_only=True, map_location=device)

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Infer dimensions from weights if possible
        for key in ["W_enc", "encoder.weight"]:
            if key in state_dict:
                w = state_dict[key]
                if len(w.shape) == 2:
                    if w.shape[0] == d_in:
                        actual_d_in, actual_d_hidden = w.shape
                    else:
                        actual_d_hidden, actual_d_in = w.shape

                    if actual_d_in != d_in or actual_d_hidden != d_hidden:
                        print(f"  Dimensions from weights: d_in={actual_d_in}, d_hidden={actual_d_hidden}")
                        d_in, d_hidden = actual_d_in, actual_d_hidden
                break

        sae = BatchTopKSparseAutoEncoder(
            d_in=d_in,
            d_hidden=d_hidden,
            device=device,
            top_k=top_k,
            center_input=config.center_input,
        )
        sae.load_pretrained_weights(state_dict)

        print(f"BatchTopK SAE loaded: {sae.d_hidden} features, k={sae.top_k}, center_input={sae.center_input}")

    elif config.architecture == "jump_relu":
        d_in = config.d_model
        d_hidden = config.d_hidden

        if config.config_filename:
            try:
                config_path = hf_hub_download(
                    repo_id=config.repo,
                    filename=config.config_filename,
                    repo_type="model",
                )
                with open(config_path) as f:
                    sae_config = json.load(f)
                print(f"  Config loaded: {sae_config}")
            except Exception as e:
                print(f"  Warning: could not load config: {e}")

        sae_path = hf_hub_download(
            repo_id=config.repo,
            filename=config.filename,
            repo_type="model",
        )

        if config.file_format == "npz" or config.filename.endswith(".npz"):
            npz = np.load(sae_path)
            state_dict = {k: torch.from_numpy(npz[k]) for k in npz.files}
        elif config.file_format == "safetensors" or config.filename.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file as safetensors_load_file
            except ImportError as e:
                raise ImportError(
                    "Loading SAE safetensors requires 'safetensors'. "
                    "Install via: pip install safetensors"
                ) from e
            state_dict = safetensors_load_file(sae_path, device=str(device))
        else:
            state_dict = torch.load(sae_path, weights_only=True, map_location=device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        # Infer dimensions from encoder weights when available
        for key in ["w_enc", "W_enc", "encoder.weight"]:
            if key in state_dict:
                w = state_dict[key]
                if len(w.shape) == 2:
                    if w.shape[0] == d_in:
                        actual_d_in, actual_d_hidden = w.shape
                    else:
                        actual_d_hidden, actual_d_in = w.shape
                    if actual_d_in != d_in or actual_d_hidden != d_hidden:
                        print(f"  Dimensions from weights: d_in={actual_d_in}, d_hidden={actual_d_hidden}")
                        d_in, d_hidden = actual_d_in, actual_d_hidden
                break

        sae = JumpReLUSparseAutoEncoder(
            d_in=d_in,
            d_hidden=d_hidden,
            device=device,
            center_input=config.center_input,
        )
        sae.load_pretrained_weights(state_dict)

        print(f"JumpReLU SAE loaded: {sae.d_hidden} features")

    else:
        raise ValueError(f"Unknown SAE architecture: {config.architecture}")

    return sae
