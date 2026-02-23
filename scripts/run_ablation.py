"""Run ablation study (Exp 2) and parameter sensitivity analysis (Exp 7).

Ablation: trains RDSA variants with individual loss components disabled.
Sensitivity: sweeps one hyperparameter while fixing others.

Usage:
    python scripts/run_ablation.py --experiment ablation --model qwen3vl
    python scripts/run_ablation.py --experiment sensitivity --param alpha_entangle --model qwen3vl
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ablation variants: which loss weights to zero out
ABLATION_VARIANTS: dict[str, dict[str, float]] = {
    "full": {
        "training.alpha_entangle": 0.1,
        "training.alpha_consist": 0.05,
        "training.alpha_lat_sub": 0.1,
    },
    "wo_entanglement": {
        "training.alpha_entangle": 0.0,
        "training.alpha_consist": 0.05,
        "training.alpha_lat_sub": 0.1,
    },
    "wo_consistency": {
        "training.alpha_entangle": 0.1,
        "training.alpha_consist": 0.0,
        "training.alpha_lat_sub": 0.1,
    },
    "wo_sublat": {
        "training.alpha_entangle": 0.1,
        "training.alpha_consist": 0.05,
        "training.alpha_lat_sub": 0.0,
    },
    "training_only": {
        "training.alpha_entangle": 0.1,
        "training.alpha_consist": 0.05,
        "training.alpha_lat_sub": 0.1,
    },
}

# Parameter sensitivity sweep ranges
SENSITIVITY_PARAMS: dict[str, list[float]] = {
    "alpha_entangle": [0.01, 0.05, 0.1, 0.5, 1.0],
    "alpha_consist": [0.01, 0.05, 0.1, 0.5, 1.0],
    "alpha_lat_sub": [0.01, 0.05, 0.1, 0.5],
    "d_safe": [8, 16, 32, 64, 128],
    "threshold": [0.1, 0.2, 0.3, 0.5],
}

CONFIG_MAP: dict[str, str] = {
    "qwen3vl": "configs/qwen3vl.yaml",
    "gemma3": "configs/gemma3.yaml",
    "llama": "configs/llama32.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDSA Ablation & Sensitivity Study")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["ablation", "sensitivity"],
        required=True,
        help="Which experiment to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3vl",
        choices=["qwen3vl", "gemma3", "llama"],
        help="Model shortname",
    )
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        help="Parameter to sweep (for sensitivity experiment)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--attacks",
        nargs="*",
        default=["scia", "umk", "figstep"],
        help="Attacks to evaluate after each training run",
    )
    return parser.parse_args()


def run_training(
    config_path: str,
    overrides: dict[str, float | str],
    output_suffix: str,
    dry_run: bool = False,
) -> bool:
    """Run a single RDSA training with config overrides.

    Args:
        config_path: Path to base YAML config.
        overrides: Hydra-style config overrides.
        output_suffix: Suffix for output directory.
        dry_run: If True, only print the command.

    Returns:
        True if training succeeded.
    """
    cmd = [
        sys.executable,
        "-m",
        "rdsa.train",
        f"--config-path=../../{Path(config_path).parent}",
        f"--config-name={Path(config_path).stem}",
    ]

    for key, value in overrides.items():
        cmd.append(f"{key}={value}")

    cmd.append(f"output.save_dir=outputs/ablation/{output_suffix}")

    logger.info("Running: %s", " ".join(cmd))

    if dry_run:
        return True

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Training completed: %s", output_suffix)
        logger.info("stdout: %s", result.stdout[-500:] if result.stdout else "")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Training failed: %s", output_suffix)
        logger.error("stderr: %s", e.stderr[-500:] if e.stderr else "")
        return False


def run_evaluation(
    model_name: str,
    checkpoint_dir: str,
    attacks: list[str],
    output_file: str,
    dry_run: bool = False,
) -> dict[str, dict[str, float]]:
    """Run evaluation on trained model.

    Args:
        model_name: Model shortname.
        checkpoint_dir: Path to model checkpoint.
        attacks: List of attack names.
        output_file: Path to save results.
        dry_run: If True, skip execution.

    Returns:
        Results dict.
    """
    if dry_run:
        return {}

    results: dict[str, dict[str, float]] = {}
    for attack in attacks:
        cmd = [
            sys.executable,
            "-m",
            "rdsa.evaluate",
            f"--model={model_name}",
            "--defense=rdsa",
            f"--attack={attack}",
            f"--checkpoint-dir={checkpoint_dir}",
            f"--output-dir={Path(output_file).parent}",
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Evaluation done: %s", attack)
        except subprocess.CalledProcessError as e:
            stderr_tail = e.stderr[-200:] if e.stderr else ""
            logger.error(
                "Evaluation failed for %s: %s", attack, stderr_tail
            )

        results[attack] = {"status": "completed"}

    return results


def run_ablation(args: argparse.Namespace) -> None:
    """Run the full ablation study (Exp 2)."""
    config_path = CONFIG_MAP[args.model]
    output_dir = Path(args.output_dir) / "ablation" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, object]] = {}

    for variant_name, overrides in ABLATION_VARIANTS.items():
        logger.info("=== Ablation variant: %s ===", variant_name)

        success = run_training(
            config_path=config_path,
            overrides=overrides,
            output_suffix=f"{args.model}/{variant_name}",
            dry_run=args.dry_run,
        )

        if success and not args.dry_run:
            checkpoint_dir = f"outputs/ablation/{args.model}/{variant_name}"
            eval_results = run_evaluation(
                model_name=args.model,
                checkpoint_dir=checkpoint_dir,
                attacks=args.attacks,
                output_file=str(output_dir / f"{variant_name}_results.json"),
                dry_run=args.dry_run,
            )
            all_results[variant_name] = {"overrides": overrides, "results": eval_results}
        else:
            status = "dry_run" if args.dry_run else "failed"
            all_results[variant_name] = {
                "overrides": overrides,
                "status": status,
            }

    # Save summary
    summary_file = output_dir / "ablation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Ablation summary saved to %s", summary_file)


def run_sensitivity(args: argparse.Namespace) -> None:
    """Run parameter sensitivity analysis (Exp 7)."""
    if args.param is None:
        logger.error("--param is required for sensitivity experiment")
        return

    if args.param not in SENSITIVITY_PARAMS:
        logger.error(
            "Unknown parameter: %s. Available: %s",
            args.param,
            list(SENSITIVITY_PARAMS.keys()),
        )
        return

    config_path = CONFIG_MAP[args.model]
    values = SENSITIVITY_PARAMS[args.param]
    output_dir = Path(args.output_dir) / "sensitivity" / args.model / args.param
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map param name to Hydra config key
    param_key_map = {
        "alpha_entangle": "training.alpha_entangle",
        "alpha_consist": "training.alpha_consist",
        "alpha_lat_sub": "training.alpha_lat_sub",
        "d_safe": "subspace.d_safe",
        "threshold": "monitor.threshold",
    }

    config_key = param_key_map.get(args.param, f"training.{args.param}")
    all_results: dict[str, dict[str, object]] = {}

    for value in values:
        run_name = f"{args.param}={value}"
        logger.info("=== Sensitivity: %s ===", run_name)

        overrides = {config_key: value}

        success = run_training(
            config_path=config_path,
            overrides=overrides,
            output_suffix=f"{args.model}/sensitivity/{args.param}/{value}",
            dry_run=args.dry_run,
        )

        all_results[str(value)] = {
            "value": value,
            "status": "completed" if success else "failed",
        }

    summary_file = output_dir / "sensitivity_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Sensitivity summary saved to %s", summary_file)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    if args.experiment == "ablation":
        run_ablation(args)
    elif args.experiment == "sensitivity":
        run_sensitivity(args)


if __name__ == "__main__":
    main()
