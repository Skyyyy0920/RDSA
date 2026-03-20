"""Experiment 2: Ablation Study (Table 2).

Tests the contribution of each RDSA component by disabling them one at a time.
Variants: Full, w/o SA-AT, w/o Consistency, w/o Entanglement, w/o Monitor, SFT Only.

Usage:
    python scripts/exp2_ablation.py --model qwen3vl
    python scripts/exp2_ablation.py --model qwen3vl --dry-run
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    CONFIG_MAP,
    SEEDS,
    aggregate_seeds,
    results_to_latex,
    run_evaluation,
    run_training,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Ablation variants: {name: {config_override_key: value}}
ABLATION_VARIANTS: dict[str, dict[str, object]] = {
    "full": {},  # Full RDSA — all components enabled
    "wo_sa_at": {
        "training.alpha_sa_at": 0.0,
    },
    "wo_consistency": {
        "training.alpha_consist": 0.0,
    },
    "wo_entanglement": {
        "training.alpha_entangle": 0.0,
    },
    "wo_monitor": {
        # All training losses active, but monitor disabled at inference
        "monitor.conservative_mode": False,
        "monitor.threshold": 999.0,  # Effectively disable
    },
    "sft_only": {
        "training.alpha_sa_at": 0.0,
        "training.alpha_consist": 0.0,
        "training.alpha_entangle": 0.0,
        "monitor.conservative_mode": False,
        "monitor.threshold": 999.0,
    },
}

EVAL_ATTACKS = ["scia", "adaptive_scia"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 2: Ablation Study")
    parser.add_argument("--model", type=str, default="qwen3vl")
    parser.add_argument("--attacks", nargs="*", default=EVAL_ATTACKS)
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=str,
                        default="results/exp2_ablation")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_exp2(args: argparse.Namespace) -> None:
    config_path = CONFIG_MAP[args.model]
    all_results: dict[str, dict[str, object]] = {}

    for variant_name, overrides in ABLATION_VARIANTS.items():
        logger.info("=== Ablation: %s ===", variant_name)
        all_results[variant_name] = {}

        for seed in args.seeds:
            output_dir = (
                f"outputs/exp2/{args.model}/{variant_name}/seed_{seed}"
            )
            train_overrides = dict(overrides)
            train_overrides["seed"] = seed

            success = run_training(
                config_path=config_path,
                overrides=train_overrides,
                output_dir=output_dir,
                dry_run=args.dry_run,
            )
            if not success:
                continue

            for attack in args.attacks:
                eval_dir = (
                    f"{args.output_dir}/{args.model}/{variant_name}"
                    f"/{attack}/seed_{seed}"
                )
                metrics = run_evaluation(
                    model=args.model,
                    defense="rdsa",
                    attack=attack,
                    checkpoint_dir=output_dir,
                    output_dir=eval_dir,
                    max_samples=args.max_samples,
                    dry_run=args.dry_run,
                )
                if attack not in all_results[variant_name]:
                    all_results[variant_name][attack] = []
                all_results[variant_name][attack].append(metrics)

        # Aggregate
        for attack in args.attacks:
            seed_results = all_results[variant_name].get(attack, [])
            if seed_results:
                all_results[variant_name][attack] = aggregate_seeds(seed_results)

    save_results(all_results, f"{args.output_dir}/{args.model}_ablation.json")

    # LaTeX
    latex = results_to_latex(
        all_results,
        metric="asr",
        caption=f"Ablation study on {args.model} (ASR\\% $\\downarrow$)",
        label="tab:ablation",
    )
    with open(f"{args.output_dir}/{args.model}_table2.tex", "w",
              encoding="utf-8") as f:
        f.write(latex)
    logger.info("Ablation complete.")


def main() -> None:
    setup_logging("logs/exp2_ablation.log")
    args = parse_args()
    run_exp2(args)


if __name__ == "__main__":
    main()
