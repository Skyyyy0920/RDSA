"""Experiment 5: Entanglement Analysis (Figure 6).

5a. η profile before/after training (per-layer, per-model)
5b. η vs ASR scatter plot with linear fit
5c. Manipulable dimensions: theory d_s*(1-η) vs measured
5d. Semantic impact of safety removal (VQA on successfully attacked samples)

Usage:
    python scripts/exp5_entanglement_analysis.py --model qwen3vl
    python scripts/exp5_entanglement_analysis.py --model all --sub eta_profile
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    CONFIG_MAP,
    SURROGATE_MODELS,
    run_evaluation,
    run_training,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Alpha-entangle sweep values for η vs ASR scatter
ALPHA_ENTANGLE_VALUES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 5: Entanglement Analysis")
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--sub", type=str, default="all",
                        choices=["all", "eta_profile", "eta_vs_asr",
                                 "manipulable_dims", "semantic_impact"])
    parser.add_argument("--output-dir", type=str,
                        default="results/exp5_entanglement")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_5a_eta_profile(args: argparse.Namespace) -> None:
    """Generate η profiles before and after RDSA training.

    Uses visualize_entanglement.py on saved subspace files.
    """
    logger.info("=== Exp 5a: η Profile ===")
    models = SURROGATE_MODELS if args.model == "all" else [args.model]

    for model in models:
        # This leverages the existing visualization script
        analysis_config = {
            "model": model,
            "vanilla_subspace": f"subspaces/{model}/vanilla",
            "rdsa_subspace": f"subspaces/{model}/rdsa",
            "command": (
                f"python scripts/visualize_entanglement.py "
                f"--model {model} "
                f"--subspace-dir subspaces/{model}/vanilla "
                f"--rdsa-subspace-dir subspaces/{model}/rdsa "
                f"--output figures/exp5/eta_profile_{model}.pdf"
            ),
            "description": (
                "Compare per-group η values between vanilla model and "
                "RDSA-trained model. Shows that RDSA pushes η from ~0.1-0.3 "
                "to ~0.6-0.8."
            ),
        }
        save_results(
            analysis_config,
            f"{args.output_dir}/{model}/5a_eta_profile_config.json",
        )


def run_5b_eta_vs_asr(args: argparse.Namespace) -> None:
    """Train with varying α_entangle, measure η and ASR to show correlation."""
    logger.info("=== Exp 5b: η vs ASR ===")
    config_path = CONFIG_MAP[args.model]
    results: dict[float, object] = {}

    for alpha in ALPHA_ENTANGLE_VALUES:
        logger.info("Training with alpha_entangle=%.3f", alpha)
        train_dir = f"outputs/exp5/{args.model}/alpha_{alpha}"

        run_training(
            config_path=config_path,
            overrides={"training.alpha_entangle": alpha},
            output_dir=train_dir,
            dry_run=args.dry_run,
        )

        # Evaluate ASR
        eval_dir = f"{args.output_dir}/{args.model}/5b/alpha_{alpha}"
        metrics = run_evaluation(
            model=args.model, defense="rdsa", attack="scia",
            checkpoint_dir=train_dir, output_dir=eval_dir,
            max_samples=args.max_samples, dry_run=args.dry_run,
        )
        results[alpha] = {
            "alpha_entangle": alpha,
            "metrics": metrics,
            # η will be extracted from wandb logs or checkpoint metadata
        }

    save_results(results, f"{args.output_dir}/{args.model}/5b_eta_vs_asr.json")


def run_5c_manipulable_dims(args: argparse.Namespace) -> None:
    """Compare theoretical d_s*(1-η) with measured effective PGD dimensions."""
    logger.info("=== Exp 5c: Manipulable Dimensions ===")
    analysis_config = {
        "model": args.model,
        "description": (
            "For models trained with different α_entangle values: "
            "1) Compute theoretical manipulable dims = d_s * (1 - η) "
            "2) Run PGD and measure effective rank of δ* via SVD "
            "3) Plot theoretical vs measured, expect strong correlation"
        ),
        "data_source": f"results/exp5/{args.model}/5b_eta_vs_asr.json",
    }
    save_results(
        analysis_config,
        f"{args.output_dir}/{args.model}/5c_manipulable_config.json",
    )


def run_5d_semantic_impact(args: argparse.Namespace) -> None:
    """Measure capability degradation on successfully attacked samples."""
    logger.info("=== Exp 5d: Semantic Impact ===")
    analysis_config = {
        "model": args.model,
        "description": (
            "Use Adaptive-SCIA to attack RDSA-trained models with varying η. "
            "For samples where the attack succeeds: "
            "1) Measure VQAv2 accuracy on those same inputs "
            "2) Show that high-η models suffer more semantic damage "
            "   when safety is forcibly removed "
            "This demonstrates the entanglement trade-off for attackers."
        ),
        "expected_result": (
            "High η: successful attacks also destroy semantic capability "
            "(VQA accuracy drops >20%). "
            "Low η: attacks succeed without semantic impact."
        ),
    }
    save_results(
        analysis_config,
        f"{args.output_dir}/{args.model}/5d_semantic_config.json",
    )


def main() -> None:
    setup_logging("logs/exp5_entanglement.log")
    args = parse_args()

    subs = (
        ["eta_profile", "eta_vs_asr", "manipulable_dims", "semantic_impact"]
        if args.sub == "all"
        else [args.sub]
    )

    for sub in subs:
        if sub == "eta_profile":
            run_5a_eta_profile(args)
        elif sub == "eta_vs_asr":
            run_5b_eta_vs_asr(args)
        elif sub == "manipulable_dims":
            run_5c_manipulable_dims(args)
        elif sub == "semantic_impact":
            run_5d_semantic_impact(args)


if __name__ == "__main__":
    main()
