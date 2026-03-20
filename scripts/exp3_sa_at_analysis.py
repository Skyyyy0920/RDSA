"""Experiment 3: SA-AT Analysis (Figures 2-3).

3a. PGD search strength vs robustness (eval-time PGD steps sweep)
3b. Random restart count effect
3c. Epsilon sensitivity (epsilon_ratio sweep)
3d. Perturbation coverage in V_s subspace

Usage:
    python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub all
    python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub pgd_steps
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 3: SA-AT Analysis")
    parser.add_argument("--model", type=str, default="qwen3vl")
    parser.add_argument("--sub", type=str, default="all",
                        choices=["all", "pgd_steps", "restarts",
                                 "epsilon", "coverage"])
    parser.add_argument("--output-dir", type=str,
                        default="results/exp3_sa_at")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_3a_pgd_steps(args: argparse.Namespace) -> None:
    """Sweep eval-time PGD steps: train RDSA once, evaluate with varying attack strength."""
    logger.info("=== Exp 3a: PGD Steps Sweep ===")
    config_path = CONFIG_MAP[args.model]
    pgd_steps_values = [1, 3, 5, 7, 10, 20, 50]

    # Train RDSA once with default config
    train_dir = f"outputs/exp3/{args.model}/default"
    run_training(config_path=config_path, overrides={},
                 output_dir=train_dir, dry_run=args.dry_run)

    # Also train vanilla and LAT for comparison
    for defense, overrides in [
        ("vanilla", {}),
        ("lat", {"training.alpha_sa_at": 0.3, "training.alpha_consist": 0.0,
                 "training.alpha_entangle": 0.0}),
    ]:
        d = f"outputs/exp3/{args.model}/{defense}"
        if defense != "vanilla":
            run_training(config_path=config_path, overrides=overrides,
                         output_dir=d, dry_run=args.dry_run)

    results: dict[str, dict[int, object]] = {
        "vanilla": {}, "lat": {}, "rdsa": {},
    }

    for steps in pgd_steps_values:
        for defense in ["vanilla", "lat", "rdsa"]:
            ckpt = "none" if defense == "vanilla" else f"outputs/exp3/{args.model}/{defense}"
            if defense == "rdsa":
                ckpt = train_dir
            eval_dir = f"{args.output_dir}/{args.model}/3a/{defense}/steps_{steps}"
            metrics = run_evaluation(
                model=args.model, defense=defense, attack="scia",
                checkpoint_dir=ckpt, output_dir=eval_dir,
                max_samples=args.max_samples, dry_run=args.dry_run,
                extra_args=[f"--pgd-steps={steps}"],
            )
            results[defense][steps] = metrics

    save_results(results, f"{args.output_dir}/{args.model}/3a_pgd_steps.json")


def run_3b_restarts(args: argparse.Namespace) -> None:
    """Sweep num_restarts during training."""
    logger.info("=== Exp 3b: Random Restarts ===")
    config_path = CONFIG_MAP[args.model]
    restart_values = [1, 2, 3, 5, 8]
    results: dict[int, object] = {}

    for n_restart in restart_values:
        train_dir = f"outputs/exp3/{args.model}/restarts_{n_restart}"
        run_training(
            config_path=config_path,
            overrides={"training.sa_at_num_restarts": n_restart},
            output_dir=train_dir,
            dry_run=args.dry_run,
        )
        eval_dir = f"{args.output_dir}/{args.model}/3b/restarts_{n_restart}"
        metrics = run_evaluation(
            model=args.model, defense="rdsa", attack="scia",
            checkpoint_dir=train_dir, output_dir=eval_dir,
            max_samples=args.max_samples, dry_run=args.dry_run,
        )
        results[n_restart] = metrics

    save_results(results, f"{args.output_dir}/{args.model}/3b_restarts.json")


def run_3c_epsilon(args: argparse.Namespace) -> None:
    """Sweep epsilon_ratio and measure ASR vs capability trade-off."""
    logger.info("=== Exp 3c: Epsilon Sensitivity ===")
    config_path = CONFIG_MAP[args.model]
    epsilon_ratios = [0.01, 0.02, 0.05, 0.1, 0.2]
    results: dict[float, object] = {}

    for ratio in epsilon_ratios:
        train_dir = f"outputs/exp3/{args.model}/eps_{ratio}"
        run_training(
            config_path=config_path,
            overrides={"training.sa_at_epsilon_ratio": ratio},
            output_dir=train_dir,
            dry_run=args.dry_run,
        )

        # Evaluate safety
        eval_dir = f"{args.output_dir}/{args.model}/3c/eps_{ratio}/scia"
        safety_metrics = run_evaluation(
            model=args.model, defense="rdsa", attack="scia",
            checkpoint_dir=train_dir, output_dir=eval_dir,
            max_samples=args.max_samples, dry_run=args.dry_run,
        )

        # Evaluate capability (VQAv2)
        cap_dir = f"{args.output_dir}/{args.model}/3c/eps_{ratio}/vqav2"
        cap_metrics = run_evaluation(
            model=args.model, defense="rdsa", attack="none",
            checkpoint_dir=train_dir, output_dir=cap_dir,
            max_samples=args.max_samples, dry_run=args.dry_run,
            extra_args=["--benchmarks=vqav2"],
        )

        results[ratio] = {"safety": safety_metrics, "capability": cap_metrics}

    save_results(results, f"{args.output_dir}/{args.model}/3c_epsilon.json")


def run_3d_coverage(args: argparse.Namespace) -> None:
    """Analyze perturbation coverage across V_s dimensions.

    This is a post-hoc analysis: load a trained RDSA model, run PGD,
    and measure how the found delta* distributes across V_s dimensions.
    """
    logger.info("=== Exp 3d: Perturbation Coverage ===")
    # This analysis requires direct Python (not subprocess) since we need
    # to inspect intermediate PGD results. Save a placeholder config.
    analysis_config = {
        "model": args.model,
        "checkpoint": f"outputs/exp3/{args.model}/default",
        "subspace": f"subspaces/{args.model}",
        "description": (
            "Load trained RDSA model, run PGD inner loop, record per-dimension "
            "delta* magnitudes. Compare before/after RDSA training to show "
            "that SA-AT achieves more uniform coverage across all d_s dimensions."
        ),
        "script": (
            "import torch\n"
            "from rdsa.subspace.identifier import SafetySubspaceIdentifier\n"
            "from rdsa.training.losses import SubspaceConstrainedATLoss\n"
            "# 1. Load model + subspaces\n"
            "# 2. Run find_worst_perturbation on eval data\n"
            "# 3. Compute |delta*| per V_s dimension\n"
            "# 4. Plot histogram: before vs after training\n"
        ),
    }
    save_results(
        analysis_config,
        f"{args.output_dir}/{args.model}/3d_coverage_config.json",
    )


def main() -> None:
    setup_logging("logs/exp3_sa_at.log")
    args = parse_args()

    subs = (
        ["pgd_steps", "restarts", "epsilon", "coverage"]
        if args.sub == "all"
        else [args.sub]
    )

    for sub in subs:
        if sub == "pgd_steps":
            run_3a_pgd_steps(args)
        elif sub == "restarts":
            run_3b_restarts(args)
        elif sub == "epsilon":
            run_3c_epsilon(args)
        elif sub == "coverage":
            run_3d_coverage(args)


if __name__ == "__main__":
    main()
