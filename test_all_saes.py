#!/usr/bin/env python3
"""
Test all SAE configs: load model + SAE, run one encode/decode pass.

Groups SAEs by model to avoid reloading. Reports pass/fail for each.
Each model group runs in a subprocess to guarantee clean GPU memory.

Usage:
    uv run python test_all_saes.py                  # test all SAE configs
    uv run python test_all_saes.py --filter eleuther # only configs matching pattern
    uv run python test_all_saes.py --dry-run         # just list what would be tested
    uv run python test_all_saes.py --clear-cache     # free disk between models
    uv run python test_all_saes.py --isolate-per-sae # full VRAM reset after each SAE

Deploy to remote GPU:
    uv run deploy_to_gpu.py --experiment-script test_all_saes.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch
import yaml


def load_sae_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_path(sae_path: Path, model_ref: str) -> Path:
    model_path = (sae_path.parent / model_ref).resolve()
    if not model_path.exists():
        model_path = Path(model_ref)
    return model_path


def discover_configs(sae_dir: Path, filter_pattern: str | None = None) -> list[dict]:
    """Discover all SAE configs, resolve model refs, group by model."""
    configs = []
    for sae_path in sorted(sae_dir.glob("*.yaml")):
        sae_data = load_sae_yaml(sae_path)
        if filter_pattern and filter_pattern.lower() not in sae_path.name.lower():
            continue

        model_path = resolve_model_path(sae_path, sae_data["model"])
        if not model_path.exists():
            configs.append({
                "sae_path": sae_path,
                "sae_data": sae_data,
                "model_path": model_path,
                "model_data": None,
                "error": f"Model config not found: {model_path}",
            })
            continue

        model_data = load_sae_yaml(model_path)
        configs.append({
            "sae_path": sae_path,
            "sae_data": sae_data,
            "model_path": model_path,
            "model_data": model_data,
        })
    return configs


def group_by_model(configs: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for cfg in configs:
        if cfg.get("error"):
            groups["__errors__"].append(cfg)
        else:
            groups[cfg["model_data"]["model_id"]].append(cfg)
    return dict(groups)


def estimate_model_vram_gb(model_data: dict) -> float:
    """Rough VRAM estimate for a model in fp16."""
    d = model_data["d_model"]
    n = model_data["n_layers"]
    params = 12 * d * d * n + d * 50000
    bytes_per_param = 2
    return params * bytes_per_param / 1e9


def test_sae_loading(sae_data: dict, sae_path: Path, model, tokenizer, device: torch.device) -> str | None:
    """Load SAE and do one encode/decode pass. Returns error string or None on success."""
    from sae_experiments.config import SAEConfig
    from sae_experiments.sae import load_sae

    available = sae_data["available_layers"]
    layer = available[len(available) // 2]

    resolved = dict(sae_data)
    resolved["layer"] = layer
    for key in ("filename", "config_filename"):
        val = resolved.get(key)
        if val and "{layer}" in str(val):
            resolved[key] = val.format(layer=layer)

    sae_config = SAEConfig(**resolved)
    sae = load_sae(sae_config, device)

    test_input = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**test_input, output_hidden_states=True)
        hidden = outputs.hidden_states[layer]
        last_token = hidden[:, -1, :]

        sae_input = last_token.to(sae.dtype) if hasattr(sae, "dtype") else last_token
        encoded = sae.encode(sae_input)
        decoded = sae.decode(encoded)

    assert encoded.shape[-1] == sae.d_hidden, (
        f"Encoded shape mismatch: {encoded.shape[-1]} != {sae.d_hidden}"
    )
    assert decoded.shape == last_token.shape, (
        f"Decoded shape mismatch: {decoded.shape} != {last_token.shape}"
    )

    sparsity = (encoded == 0).float().mean().item()
    print(f"    Sparsity: {sparsity:.1%}, "
          f"encoded range: [{encoded.min().item():.4f}, {encoded.max().item():.4f}]")

    del sae, encoded, decoded, hidden, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None


def _print_vram_status(label: str) -> None:
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  VRAM [{label}]: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")


def _clear_hf_repo_cache(repo_id: str) -> None:
    """Remove any HuggingFace repo's files from the local cache."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_root = Path(hf_home) / "hub"
    dir_name = "models--" + repo_id.replace("/", "--")
    repo_cache_dir = cache_root / dir_name
    if repo_cache_dir.exists():
        size_mb = sum(f.stat().st_size for f in repo_cache_dir.rglob("*") if f.is_file()) / 1e6
        shutil.rmtree(repo_cache_dir, ignore_errors=True)
        print(f"  Cleared HF cache for {repo_id} ({size_mb:.0f} MB)")


def _clear_group_caches(model_id: str, configs: list[dict]) -> None:
    """Clear HF cache for a model and all its SAE repos."""
    _clear_hf_repo_cache(model_id)
    seen = set()
    for cfg in configs:
        repo = cfg.get("sae_data", {}).get("repo", "")
        if repo and repo not in seen:
            seen.add(repo)
            _clear_hf_repo_cache(repo)


def test_model_group(
    model_id: str,
    configs: list[dict],
    device: torch.device,
    clear_cache: bool = False,
) -> list[dict]:
    """Test all SAEs for a given model. Returns results list."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = []
    model_data = configs[0]["model_data"]
    vram_est = estimate_model_vram_gb(model_data)

    print(f"\n{'='*60}")
    print(f"Model: {model_id} (~{vram_est:.1f} GB VRAM estimate)")
    print(f"  SAEs to test: {len(configs)}")
    print(f"{'='*60}")
    _print_vram_status("before load")

    t0 = time.time()
    model = None
    tokenizer = None
    try:
        dtype = getattr(torch, model_data.get("torch_dtype", "float16"))
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")
        _print_vram_status("after load")
    except Exception as e:
        print(f"  FAILED to load model: {e}")
        for cfg in configs:
            results.append({
                "sae": cfg["sae_path"].stem,
                "model": model_id,
                "status": "SKIP",
                "error": f"Model load failed: {e}",
            })
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if clear_cache:
            _clear_group_caches(model_id, configs)
        return results

    for cfg in configs:
        sae_name = cfg["sae_data"]["name"]
        sae_file = cfg["sae_path"].stem
        print(f"\n  Testing: {sae_name}")
        print(f"    Config: {sae_file}")

        t1 = time.time()
        try:
            err = test_sae_loading(cfg["sae_data"], cfg["sae_path"], model, tokenizer, device)
            elapsed = time.time() - t1
            if err:
                print(f"    FAIL ({elapsed:.1f}s): {err}")
                results.append({"sae": sae_file, "model": model_id, "status": "FAIL", "error": err})
            else:
                print(f"    PASS ({elapsed:.1f}s)")
                results.append({"sae": sae_file, "model": model_id, "status": "PASS", "error": None})
        except Exception as e:
            elapsed = time.time() - t1
            tb = traceback.format_exc()
            print(f"    FAIL ({elapsed:.1f}s): {e}")
            results.append({"sae": sae_file, "model": model_id, "status": "FAIL", "error": str(e), "traceback": tb})

    _print_vram_status("before cleanup")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _print_vram_status("after cleanup")

    if clear_cache:
        _clear_group_caches(model_id, configs)

    return results


# ---------------------------------------------------------------------------
# Subprocess isolation: each model group runs in a fresh process so that
# CUDA driver-level memory pools are fully released between models.
# ---------------------------------------------------------------------------

def _serialize_configs(configs: list[dict]) -> list[dict]:
    """Convert configs to JSON-serializable dicts."""
    out = []
    for cfg in configs:
        out.append({
            "sae_path": str(cfg["sae_path"]),
            "sae_data": cfg["sae_data"],
            "model_data": cfg["model_data"],
        })
    return out


def _deserialize_configs(data: list[dict]) -> list[dict]:
    """Restore configs from serialized form."""
    out = []
    for item in data:
        out.append({
            "sae_path": Path(item["sae_path"]),
            "sae_data": item["sae_data"],
            "model_data": item["model_data"],
        })
    return out


def _subprocess_entry(input_file: str, result_file: str) -> None:
    """Entry point invoked inside each subprocess."""
    with open(input_file) as f:
        payload = json.load(f)

    configs = _deserialize_configs(payload["configs"])
    device = torch.device(payload["device"])

    try:
        results = test_model_group(
            payload["model_id"], configs, device, clear_cache=False,
        )
    except Exception as e:
        results = [{
            "sae": Path(c["sae_path"]).stem,
            "model": payload["model_id"],
            "status": "SKIP",
            "error": str(e),
        } for c in payload["configs"]]

    serializable = []
    for r in results:
        serializable.append({
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in r.items()
        })
    try:
        with open(result_file, "w") as f:
            json.dump(serializable, f)
    except OSError:
        print(f"  ERROR: Could not write result file (disk full?)", file=sys.stderr)
        sys.exit(2)


def run_model_isolated(
    model_id: str,
    configs: list[dict],
    device_str: str,
    clear_cache: bool,
    timeout: int = 900,
) -> list[dict]:
    """Run test_model_group in a subprocess for clean CUDA memory.

    Cache clearing is always handled here in the parent process (not the
    subprocess) so that caches are freed even when the subprocess crashes.
    """
    import subprocess

    payload = {
        "model_id": model_id,
        "configs": _serialize_configs(configs),
        "device": device_str,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir="."
    ) as f:
        json.dump(payload, f)
        input_file = f.name

    result_file = input_file + ".result.json"

    try:
        proc = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; sys.path.insert(0, '.'); "
                "from test_all_saes import _subprocess_entry; "
                f"_subprocess_entry({input_file!r}, {result_file!r})",
            ],
            timeout=timeout,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

        if proc.returncode != 0 or not Path(result_file).exists():
            return [{
                "sae": cfg["sae_path"].stem, "model": model_id,
                "status": "SKIP",
                "error": f"Subprocess exited with code {proc.returncode}",
            } for cfg in configs]

        with open(result_file) as f:
            return json.load(f)

    except subprocess.TimeoutExpired:
        return [{
            "sae": cfg["sae_path"].stem, "model": model_id,
            "status": "SKIP", "error": "Subprocess timed out",
        } for cfg in configs]

    finally:
        for p in (input_file, result_file):
            try:
                os.unlink(p)
            except OSError:
                pass
        if clear_cache:
            _clear_group_caches(model_id, configs)


def run_sae_isolated(
    model_id: str,
    cfg: dict,
    device_str: str,
    clear_cache: bool,
    timeout: int = 900,
) -> list[dict]:
    """Run a single SAE test in its own subprocess for a full VRAM reset."""
    return run_model_isolated(
        model_id,
        [cfg],
        device_str,
        clear_cache=clear_cache,
        timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser(description="Test all SAE configs")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only test SAE configs whose filename contains this pattern")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be tested without actually testing")
    parser.add_argument("--sae-dir", type=Path, default=Path("configs/saes"),
                        help="Directory with SAE YAML configs")
    parser.add_argument("--max-model-vram", type=float, default=40.0,
                        help="Skip models estimated to need more than this VRAM (GB)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete HF model cache after each model group to free disk space")
    parser.add_argument("--no-isolate", action="store_true",
                        help="Run all models in-process (no subprocess isolation)")
    parser.add_argument("--isolate-per-sae", action="store_true",
                        help="Run each SAE in its own subprocess for a full GPU VRAM reset")
    args = parser.parse_args()
    if args.no_isolate and args.isolate_per_sae:
        parser.error("--isolate-per-sae cannot be combined with --no-isolate")

    configs = discover_configs(args.sae_dir, args.filter)
    groups = group_by_model(configs)

    # Print plan
    print("=" * 60)
    print("SAE Loading Test Plan")
    print("=" * 60)
    total_saes = 0
    skipped_models = []
    for model_id, cfgs in sorted(groups.items()):
        if model_id == "__errors__":
            print(f"\n  Config errors ({len(cfgs)}):")
            for c in cfgs:
                print(f"    {c['sae_path'].stem}: {c['error']}")
            continue
        model_data = cfgs[0]["model_data"]
        vram = estimate_model_vram_gb(model_data)
        skip = vram > args.max_model_vram
        marker = " [SKIP - too large]" if skip else ""
        print(f"\n  {model_id} (~{vram:.1f} GB){marker}")
        for c in cfgs:
            print(f"    - {c['sae_path'].stem}")
        if skip:
            skipped_models.append(model_id)
        else:
            total_saes += len(cfgs)

    n_models = len(groups) - len(skipped_models) - (1 if "__errors__" in groups else 0)
    print(f"\n  Total: {total_saes} SAEs across {n_models} models")
    if skipped_models:
        print(f"  Skipping {len(skipped_models)} models (>{args.max_model_vram} GB): {skipped_models}")
    isolate = not args.no_isolate
    if args.isolate_per_sae:
        print("  Mode: subprocess isolation per SAE (fresh CUDA context after every SAE)")
    elif isolate:
        print("  Mode: subprocess isolation (each model gets a fresh CUDA context)")
    print("=" * 60)

    if args.dry_run:
        print("\nDry run complete.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_results = []
    errors_list = groups.pop("__errors__", [])
    for e in errors_list:
        all_results.append({
            "sae": e["sae_path"].stem, "model": "N/A",
            "status": "ERROR", "error": e["error"],
        })

    sorted_models = sorted(
        [(mid, cfgs) for mid, cfgs in groups.items()],
        key=lambda x: estimate_model_vram_gb(x[1][0]["model_data"]),
    )

    for model_id, cfgs in sorted_models:
        vram = estimate_model_vram_gb(cfgs[0]["model_data"])
        if vram > args.max_model_vram:
            for c in cfgs:
                all_results.append({
                    "sae": c["sae_path"].stem, "model": model_id,
                    "status": "SKIP", "error": f"Model too large (~{vram:.1f} GB)",
                })
            continue

        if args.isolate_per_sae:
            results = []
            for cfg in cfgs:
                results.extend(
                    run_sae_isolated(
                        model_id, cfg, str(device), clear_cache=args.clear_cache,
                    )
                )
        elif isolate:
            results = run_model_isolated(
                model_id, cfgs, str(device), clear_cache=args.clear_cache,
            )
        else:
            results = test_model_group(
                model_id, cfgs, device, clear_cache=args.clear_cache,
            )
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = [r for r in all_results if r["status"] == "PASS"]
    failed = [r for r in all_results if r["status"] == "FAIL"]
    skipped = [r for r in all_results if r["status"] == "SKIP"]
    errored = [r for r in all_results if r["status"] == "ERROR"]

    print(f"  PASS:    {len(passed)}")
    print(f"  FAIL:    {len(failed)}")
    print(f"  SKIP:    {len(skipped)}")
    print(f"  ERROR:   {len(errored)}")
    print(f"  Total:   {len(all_results)}")

    if failed:
        print("\nFailed SAEs:")
        for r in failed:
            print(f"  {r['sae']} ({r['model']}): {r['error']}")
            if r.get("traceback"):
                for line in r["traceback"].strip().split("\n")[-3:]:
                    print(f"    {line}")

    if errored:
        print("\nConfig errors:")
        for r in errored:
            print(f"  {r['sae']}: {r['error']}")

    sys.exit(1 if failed or errored else 0)


if __name__ == "__main__":
    main()
