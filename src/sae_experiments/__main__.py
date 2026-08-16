"""CLI entry point: python -m sae_experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_experiment_config
from .pipeline import Pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sae_experiments",
        description="Feature-based SAE intervention experiments on LLM benchmarks.",
    )
    parser.add_argument(
        "--sae", required=True, type=Path,
        help="Path to SAE YAML config (e.g. configs/saes/batchtopk_llama_k32.yaml)",
    )
    parser.add_argument(
        "--layer", required=True, type=int,
        help="Layer index for SAE intervention (must be in SAE's available_layers)",
    )
    parser.add_argument(
        "--benchmark", required=True, type=Path,
        help="Path to benchmark YAML config (e.g. configs/benchmarks/commonsenseqa.yaml)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results (default: results)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for the screening cache JSON (default: 'cache' — shared across experiments). "
             "Pass a per-experiment path (e.g. results/EXP-300) to keep cache co-located with results.",
    )
    parser.add_argument(
        "--screening-only", action="store_true",
        help="Only run logit screening (cache baseline), skip phases 1-4",
    )
    parser.add_argument(
        "--force-rerun-screening", action="store_true",
        help="Force rerun logit screening even if cache exists",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size for batched inference (default: 32)",
    )
    parser.add_argument(
        "--method", choices=["sae", "caa"], default="sae",
        help="Steering method: 'sae' (default) or 'caa' (contrastive activation addition)",
    )
    parser.add_argument(
        "--caa-strengths", type=str, default=None,
        help="Comma-separated strength values for CAA sweep (e.g. '0.5,1.0,2.0,5.0')",
    )
    parser.add_argument(
        "--generalisation", nargs="+", type=Path, default=None,
        metavar="BENCHMARK_YAML",
        help="Extra benchmark YAML configs for cross-benchmark generalisation test",
    )
    parser.add_argument(
        "--set", action="append", dest="overrides", default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g. --set causal.seed=123 --set grid.strength=[0.5,1.0,2.0])",
    )

    args = parser.parse_args()

    # Parse overrides
    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            print(f"Error: override must be KEY=VALUE, got: {item}", file=sys.stderr)
            sys.exit(1)
        key, value = item.split("=", 1)
        # Try to parse as Python literal
        try:
            import ast
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass  # Keep as string
        overrides[key] = value

    if args.output_dir != "results":
        overrides["output_dir"] = args.output_dir
    if args.cache_dir is not None:
        overrides["cache_dir"] = args.cache_dir
    if args.force_rerun_screening:
        overrides["force_rerun_screening"] = True
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size

    # Load config
    config = load_experiment_config(
        sae_path=args.sae,
        layer=args.layer,
        benchmark_path=args.benchmark,
        overrides=overrides or None,
    )

    print(f"Experiment: {config.model.short_name} + {config.sae.name} + {config.benchmark.name}")
    print(f"Method: {args.method.upper()}")
    print(f"Output dir: {config.output_dir}")

    gen_benchmarks = [str(p) for p in args.generalisation] if args.generalisation else None

    if args.screening_only and gen_benchmarks:
        print("Warning: --generalisation is ignored with --screening-only", file=sys.stderr)

    # Run pipeline
    if args.method == "caa":
        from .caa_pipeline import CaaPipeline

        caa_strengths = None
        if args.caa_strengths:
            caa_strengths = [float(s.strip()) for s in args.caa_strengths.split(",")]

        pipeline = CaaPipeline(config, strengths=caa_strengths)
        try:
            pipeline.run(generalisation_benchmarks=gen_benchmarks)
        finally:
            pipeline.cleanup()
    else:
        pipeline = Pipeline(config)
        try:
            pipeline.run(
                screening_only=args.screening_only,
                generalisation_benchmarks=gen_benchmarks,
            )
        finally:
            pipeline.cleanup()


if __name__ == "__main__":
    main()
