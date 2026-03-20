"""Experiment 9: Parameter Sensitivity (Figure 7, Appendix).

Sweeps key hyperparameters one at a time while fixing others at defaults.
Generates dual-axis plots: ASR↓ and VQA accuracy.

Usage:
    python scripts/exp9_sensitivity.py --model qwen3vl --param all
    python scripts/exp9_sensitivity.py --model qwen3vl --param alpha_sa_at
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    CONFIG_MAP,
    run_evaluation,
    run_training,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

# {param_name: (hydra_config_key, [sweep_values])}
SENSITIVITY_SWEEPS: dict[str, tuple[str, list[float | int]]] = {
    "alpha_sa_at": ("training.alpha_sa_at", [0.05, 0.1, 0.3, 0.5, 1.0]),
    "alpha_entangle": ("training.alpha_entangle", [0.01, 0.05, 0.1, 0.5, 1.0]),
    "alpha_consist": ("training.alpha_consist", [0.01, 0.05, 0.1, 0.5]),
    "d_safe": ("subspace.d_safe", [8, 16, 32, 64, 128]),
    "threshold": ("monitor.threshold", [0.1, 0.15, 0.2, 0.3, 0.5]),
    "num_groups": ("model.layer_groups", [1, 2, 3, 4]),
    "lora_rank": ("training.lora_rank", [4, 8, 16, 32]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 9: Parameter Sensitivity")
    parser.add_argument("--model", type=str, default="qwen3vl")
    parser.add_argument("--param", type=str, default="all",
                        help="Parameter to sweep or 'all'")
    parser.add_argument("--output-dir", type=str,
                        default="results/exp9_sensitivity")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _get_layer_groups_for_g(model: str, g: int) -> list[list[int]]:
    """Return layer group configs for a given number of groups G."""
    from exp4_redundancy_analysis import GROUP_CONFIGS
    model_groups = GROUP_CONFIGS.get(model, GROUP_CONFIGS["qwen3vl"])
    return model_groups.get(g, model_groups[3])


def run_sensitivity_sweep(
    args: argparse.Namespace,
    param_name: str,
) -> None:
    """Run one parameter sweep."""
    if param_name not in SENSITIVITY_SWEEPS:
        logger.error("Unknown param: %s", param_name)
        return

    config_key, values = SENSITIVITY_SWEEPS[param_name]
    config_path = CONFIG_MAP[args.model]
    results: dict[str, object] = {}

    for val in values:
        logger.info("=== %s = %s ===", param_name, val)

        # Handle special case: num_groups changes layer_groups entirely
        if param_name == "num_groups":
            groups = _get_layer_groups_for_g(args.model, int(val))
            overrides = {"model.layer_groups": str(groups).replace(" ", "")}
        else:
            overrides = {config_key: val}

        train_dir = f"outputs/exp9/{args.model}/{param_name}/{val}"
        run_training(
            config_path=config_path,
            overrides=overrides,
            output_dir=train_dir,
            dry_run=args.dry_run,
        )

        # Evaluate safety (SCIA)
        eval_safety = f"{args.output_dir}/{args.model}/{param_name}/{val}/safety"
        safety_metrics = run_evaluation(
            model=args.model, defense="rdsa", attack="scia",
            checkpoint_dir=train_dir, output_dir=eval_safety,
            max_samples=args.max_samples, dry_run=args.dry_run,
        )

        # Evaluate capability (VQAv2)
        eval_cap = f"{args.output_dir}/{args.model}/{param_name}/{val}/vqav2"
        cap_metrics = run_evaluation(
            model=args.model, defense="rdsa", attack="none",
            checkpoint_dir=train_dir, output_dir=eval_cap,
            max_samples=args.max_samples, dry_run=args.dry_run,
            extra_args=["--benchmarks=vqav2"],
        )

        results[str(val)] = {
            "value": val,
            "safety": safety_metrics,
            "capability": cap_metrics,
        }

    save_results(
        results,
        f"{args.output_dir}/{args.model}/{param_name}_sweep.json",
    )


def main() -> None:
    setup_logging("logs/exp9_sensitivity.log")
    args = parse_args()

    if args.param == "all":
        params = list(SENSITIVITY_SWEEPS.keys())
    else:
        params = [args.param]

    for param in params:
        logger.info("=== Sensitivity sweep: %s ===", param)
        run_sensitivity_sweep(args, param)

    logger.info("All sensitivity sweeps complete.")


if __name__ == "__main__":
    main()
