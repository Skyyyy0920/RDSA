"""Experiment 7: Adaptive Attack Robustness (Table 5).

Tests RDSA against attackers who have FULL knowledge of the defense:
- Adaptive-SCIA: anti-entanglement objective
- Adaptive-PGD: multi-layer simultaneous attack
- Monitor-Evasion: variance-matching constraint
- Strong budget: 10× PGD steps, 10 restarts

Usage:
    python scripts/exp7_adaptive_attacks.py --model qwen3vl
    python scripts/exp7_adaptive_attacks.py --model all --budget strong
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    SEEDS,
    SURROGATE_MODELS,
    aggregate_seeds,
    run_evaluation,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

ADAPTIVE_ATTACKS = ["adaptive_scia", "adaptive_pgd", "monitor_evasion"]

# Attack budgets: standard vs strong (10×)
ATTACK_BUDGETS: dict[str, dict[str, object]] = {
    "standard": {
        "pgd_steps": 7,
        "restarts": 3,
    },
    "strong": {
        "pgd_steps": 70,
        "restarts": 10,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp 7: Adaptive Attack Robustness"
    )
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--attacks", nargs="*", default=ADAPTIVE_ATTACKS)
    parser.add_argument("--budget", type=str, default="both",
                        choices=["standard", "strong", "both"])
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    parser.add_argument("--output-dir", type=str,
                        default="results/exp7_adaptive")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_exp7(args: argparse.Namespace) -> None:
    models = SURROGATE_MODELS if args.model == "all" else [args.model]
    budgets = (
        list(ATTACK_BUDGETS.keys())
        if args.budget == "both"
        else [args.budget]
    )

    all_results: dict[str, dict[str, dict[str, object]]] = {}

    for model in models:
        all_results[model] = {}
        # Use the default RDSA checkpoint (trained in Exp 1 or standalone)
        checkpoint_dir = f"outputs/exp1/{model}/rdsa/seed_42"

        for budget_name in budgets:
            budget = ATTACK_BUDGETS[budget_name]
            all_results[model][budget_name] = {}

            for attack in args.attacks:
                logger.info(
                    "=== %s / %s / budget=%s ===", model, attack, budget_name
                )
                seed_results = []

                for seed in args.seeds:
                    eval_dir = (
                        f"{args.output_dir}/{model}/{budget_name}"
                        f"/{attack}/seed_{seed}"
                    )
                    extra_args = [
                        f"--pgd-steps={budget['pgd_steps']}",
                        f"--attack-restarts={budget['restarts']}",
                    ]
                    metrics = run_evaluation(
                        model=model,
                        defense="rdsa",
                        attack=attack,
                        checkpoint_dir=checkpoint_dir,
                        output_dir=eval_dir,
                        max_samples=args.max_samples,
                        dry_run=args.dry_run,
                        extra_args=extra_args,
                    )
                    seed_results.append(metrics)

                all_results[model][budget_name][attack] = aggregate_seeds(
                    seed_results
                )

    save_results(all_results, f"{args.output_dir}/exp7_adaptive.json")

    # LaTeX table (one per model, rows=attacks, cols=budgets)
    for model in models:
        model_data = all_results.get(model, {})
        _generate_adaptive_table(model_data, model, args.output_dir)


def _generate_adaptive_table(
    results: dict[str, dict[str, object]],
    model: str,
    output_dir: str,
) -> None:
    """Generate LaTeX table for adaptive attacks."""
    budgets = sorted(results.keys())
    attacks = set()
    for budget_data in results.values():
        attacks.update(budget_data.keys())
    attacks = sorted(attacks)

    header_parts = ["Attack"]
    for b in budgets:
        header_parts.extend([f"ASR$\\downarrow$ ({b})", f"RR$\\uparrow$ ({b})"])
    header = " & ".join(header_parts)
    n_cols = 1 + 2 * len(budgets)

    rows = []
    for attack in attacks:
        cells = [attack.replace("_", r"\_")]
        for b in budgets:
            data = results.get(b, {}).get(attack, {})
            if isinstance(data, dict):
                asr = data.get("asr", {})
                rr = data.get("rr", {})
                if isinstance(asr, dict):
                    cells.append(f"{asr.get('mean', 0) * 100:.1f}")
                else:
                    cells.append("-")
                if isinstance(rr, dict):
                    cells.append(f"{rr.get('mean', 0) * 100:.1f}")
                else:
                    cells.append("-")
            else:
                cells.extend(["-", "-"])
        rows.append(" & ".join(cells) + r" \\")

    table = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{Adaptive attack robustness on {model} (Table 5)}}\n"
        f"\\label{{tab:adaptive_{model}}}\n"
        f"\\begin{{tabular}}{{{'l' + 'c' * (n_cols - 1)}}}\n"
        r"\toprule" + "\n"
        f"{header} \\\\\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    from pathlib import Path
    path = f"{output_dir}/{model}_table5.tex"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
    logger.info("LaTeX table saved to %s", path)


def main() -> None:
    setup_logging("logs/exp7_adaptive.log")
    args = parse_args()
    run_exp7(args)


if __name__ == "__main__":
    main()
