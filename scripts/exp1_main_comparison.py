"""Experiment 1: Main Comparison Table (Table 1).

Compares RDSA against 6 baseline defenses across 6 attacks and 3 models.
Each configuration is run with 3 seeds for statistical significance.

Usage:
    python scripts/exp1_main_comparison.py --model qwen3vl
    python scripts/exp1_main_comparison.py --model all --dry-run
    python scripts/exp1_main_comparison.py --model qwen3vl --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    BASELINE_DEFENSES,
    CONFIG_MAP,
    SEEDS,
    SURROGATE_MODELS,
    aggregate_seeds,
    results_to_latex,
    run_evaluation,
    run_training,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Defense → Hydra overrides (training-time config changes)
# Baselines that are NOT RDSA use different training scripts or configs.
# We represent them as RDSA with specific components disabled.
DEFENSE_OVERRIDES: dict[str, dict[str, object]] = {
    "vanilla": {},  # No training — use base model
    "safety_sft": {
        "training.alpha_sa_at": 0.0,
        "training.alpha_consist": 0.0,
        "training.alpha_entangle": 0.0,
    },
    "circuit_breaker": {
        # External baseline — requires separate implementation
        # Placeholder: trained separately, loaded from checkpoint
    },
    "lat": {
        # Latent Adversarial Training — full-space AT, no subspace constraint
        "training.alpha_sa_at": 0.3,
        "training.alpha_consist": 0.0,
        "training.alpha_entangle": 0.0,
        "training.sa_at_epsilon_relative": False,
        "training.sa_at_epsilon": 4.0,  # Full-space ε (larger)
    },
    "smoothvlm": {
        # External baseline — input perturbation at inference
        # Placeholder: loaded from checkpoint
    },
    "vlguard": {
        # External baseline — safety fine-tuning with VLGuard data
        # Placeholder: loaded from checkpoint
    },
    "rdsa": {},  # Full RDSA with default config
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 1: Main Comparison")
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--attacks", nargs="*",
                        default=["none", "scia", "umk", "figstep",
                                 "adaptive_scia", "adaptive_pgd"])
    parser.add_argument("--defenses", nargs="*",
                        default=BASELINE_DEFENSES + ["rdsa"])
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=str,
                        default="results/exp1_main_comparison")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_exp1(args: argparse.Namespace) -> None:
    models = SURROGATE_MODELS if args.model == "all" else [args.model]
    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for model in models:
        config_path = CONFIG_MAP[model]
        all_results[model] = {}

        for defense in args.defenses:
            all_results[model][defense] = {}

            for seed in args.seeds:
                # ── Step 1: Train (skip for vanilla / external baselines) ──
                output_dir = (
                    f"outputs/exp1/{model}/{defense}/seed_{seed}"
                )

                if defense == "vanilla":
                    checkpoint_dir = "none"
                elif defense in ("circuit_breaker", "smoothvlm", "vlguard"):
                    # External baselines: expect pre-trained checkpoints
                    checkpoint_dir = f"checkpoints/baselines/{defense}/{model}"
                    logger.info(
                        "External baseline '%s': expecting checkpoint at %s",
                        defense, checkpoint_dir,
                    )
                else:
                    overrides = dict(DEFENSE_OVERRIDES.get(defense, {}))
                    overrides["seed"] = seed
                    success = run_training(
                        config_path=config_path,
                        overrides=overrides,
                        output_dir=output_dir,
                        dry_run=args.dry_run,
                    )
                    if not success:
                        logger.error("Training failed: %s/%s/seed_%d",
                                     model, defense, seed)
                        continue
                    checkpoint_dir = output_dir

                # ── Step 2: Evaluate all attacks ──
                for attack in args.attacks:
                    eval_dir = f"{args.output_dir}/{model}/{defense}/{attack}/seed_{seed}"
                    metrics = run_evaluation(
                        model=model,
                        defense=defense,
                        attack=attack,
                        checkpoint_dir=checkpoint_dir,
                        output_dir=eval_dir,
                        max_samples=args.max_samples,
                        dry_run=args.dry_run,
                    )

                    if attack not in all_results[model][defense]:
                        all_results[model][defense][attack] = []
                    all_results[model][defense][attack].append(metrics)

            # Aggregate across seeds
            for attack in args.attacks:
                seed_results = all_results[model][defense].get(attack, [])
                if seed_results:
                    all_results[model][defense][attack] = aggregate_seeds(
                        seed_results
                    )

    # ── Save results ──
    save_results(all_results, f"{args.output_dir}/exp1_results.json")

    # ── Generate LaTeX tables (one per model) ──
    for model in models:
        model_results = all_results.get(model, {})
        latex = results_to_latex(
            model_results,
            metric="asr",
            caption=f"Main comparison on {model} (ASR\\% $\\downarrow$)",
            label=f"tab:main_{model}",
        )
        latex_path = f"{args.output_dir}/{model}_table1.tex"
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex)
        logger.info("LaTeX table saved to %s", latex_path)


def main() -> None:
    setup_logging("logs/exp1_main_comparison.log")
    args = parse_args()
    logger.info("Experiment 1: Main Comparison")
    logger.info("Models: %s, Defenses: %s, Attacks: %s, Seeds: %s",
                args.model, args.defenses, args.attacks, args.seeds)
    run_exp1(args)


if __name__ == "__main__":
    main()
