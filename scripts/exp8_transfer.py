"""Experiment 8: Transfer Attack (Table 6).

Generates adversarial examples on surrogate models (white-box) and evaluates
on victim models (black-box): same-family transfer, cross-architecture,
and commercial APIs.

Usage:
    python scripts/exp8_transfer.py --surrogate qwen3vl
    python scripts/exp8_transfer.py --surrogate all --skip-commercial
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    COMMERCIAL_MODELS,
    SURROGATE_MODELS,
    VICTIM_MODELS,
    run_evaluation,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Surrogate → Victim transfer pairs
TRANSFER_PAIRS: dict[str, list[str]] = {
    "qwen3vl": ["qwen3vl_30b", "gemma3_27b", "llama_11b"],
    "gemma3": ["gemma3_27b", "qwen3vl_30b", "llama_11b"],
    "llama": ["llama_11b", "qwen3vl_30b", "gemma3_27b"],
}

TRANSFER_ATTACKS = ["scia", "umk", "figstep"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 8: Transfer Attack")
    parser.add_argument("--surrogate", type=str, default="qwen3vl",
                        help="Surrogate model or 'all'")
    parser.add_argument("--attacks", nargs="*", default=TRANSFER_ATTACKS)
    parser.add_argument("--output-dir", type=str,
                        default="results/exp8_transfer")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--skip-commercial", action="store_true",
                        help="Skip commercial API evaluation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_exp8(args: argparse.Namespace) -> None:
    surrogates = (
        SURROGATE_MODELS if args.surrogate == "all" else [args.surrogate]
    )
    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for surrogate in surrogates:
        logger.info("=== Surrogate: %s ===", surrogate)
        all_results[surrogate] = {}

        # RDSA checkpoint on surrogate
        surrogate_ckpt = f"outputs/exp1/{surrogate}/rdsa/seed_42"

        # ── Step 1: Generate adversarial examples on surrogate ──
        for attack in args.attacks:
            adv_dir = f"outputs/exp8/{surrogate}/adv_examples/{attack}"
            logger.info("Generating adversarial examples: %s + %s",
                        surrogate, attack)
            # Generate on surrogate (white-box)
            run_evaluation(
                model=surrogate,
                defense="rdsa",
                attack=attack,
                checkpoint_dir=surrogate_ckpt,
                output_dir=adv_dir,
                max_samples=args.max_samples,
                dry_run=args.dry_run,
                extra_args=["--save-adversarial"],
            )

        # ── Step 2: Evaluate on victim models (black-box transfer) ──
        victims = TRANSFER_PAIRS.get(surrogate, [])
        for victim in victims:
            all_results[surrogate][victim] = {}
            victim_model_name = VICTIM_MODELS.get(victim, victim)

            for attack in args.attacks:
                eval_dir = (
                    f"{args.output_dir}/{surrogate}/to/{victim}/{attack}"
                )
                logger.info("Transfer: %s → %s (%s)", surrogate, victim, attack)
                metrics = run_evaluation(
                    model=victim,
                    defense="vanilla",
                    attack=f"transfer_{attack}",
                    checkpoint_dir="none",
                    output_dir=eval_dir,
                    max_samples=args.max_samples,
                    dry_run=args.dry_run,
                    extra_args=[
                        f"--adv-examples-dir=outputs/exp8/{surrogate}/adv_examples/{attack}",
                        f"--victim-model={victim_model_name}",
                    ],
                )
                all_results[surrogate][victim][attack] = metrics

        # ── Step 3: Evaluate on commercial APIs ──
        if not args.skip_commercial:
            for commercial in COMMERCIAL_MODELS:
                all_results[surrogate][commercial] = {}
                for attack in args.attacks:
                    eval_dir = (
                        f"{args.output_dir}/{surrogate}/to/{commercial}/{attack}"
                    )
                    logger.info("Transfer: %s → %s (%s)",
                                surrogate, commercial, attack)
                    metrics = run_evaluation(
                        model=commercial,
                        defense="vanilla",
                        attack=f"transfer_{attack}",
                        checkpoint_dir="none",
                        output_dir=eval_dir,
                        max_samples=args.max_samples,
                        dry_run=args.dry_run,
                        extra_args=[
                            f"--adv-examples-dir=outputs/exp8/{surrogate}/adv_examples/{attack}",
                            f"--api-model={commercial}",
                        ],
                    )
                    all_results[surrogate][commercial][attack] = metrics

    save_results(all_results, f"{args.output_dir}/exp8_transfer.json")

    # Generate transfer matrix table
    _generate_transfer_table(all_results, args.output_dir)


def _generate_transfer_table(
    results: dict[str, dict[str, dict[str, object]]],
    output_dir: str,
) -> None:
    """Generate LaTeX transfer matrix table."""
    from pathlib import Path

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Transfer attack results (ASR\% $\downarrow$).}",
        r"\label{tab:transfer}",
        r"\begin{tabular}{l" + "c" * 6 + "}",
        r"\toprule",
        (r"Surrogate $\rightarrow$ Victim & Qwen-30B & Gemma-27B"
         r" & LLaMA-11B & GPT-4o & Gemini & Claude \\"),
        r"\midrule",
    ]

    for surrogate, victims in results.items():
        cells = [surrogate.replace("_", r"\_")]
        for target in ["qwen3vl_30b", "gemma3_27b", "llama_11b",
                        "gpt-4o", "gemini-2.5-flash", "claude-sonnet-4-5"]:
            target_data = victims.get(target, {})
            # Average ASR across attacks
            asrs = []
            for attack_data in target_data.values():
                if isinstance(attack_data, dict) and "asr" in attack_data:
                    asrs.append(attack_data["asr"])
            if asrs:
                avg_asr = sum(asrs) / len(asrs) * 100
                cells.append(f"{avg_asr:.1f}")
            else:
                cells.append("-")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    table = "\n".join(lines)

    path = f"{output_dir}/table6_transfer.tex"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
    logger.info("Transfer table saved to %s", path)


def main() -> None:
    setup_logging("logs/exp8_transfer.log")
    args = parse_args()
    run_exp8(args)


if __name__ == "__main__":
    main()
