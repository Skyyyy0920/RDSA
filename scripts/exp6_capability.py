"""Experiment 6: Capability Preservation (Table 4).

Evaluates all defense methods on 4 capability benchmarks:
VQAv2, MMBench, MME, OR-Bench.

Usage:
    python scripts/exp6_capability.py --model qwen3vl
    python scripts/exp6_capability.py --model all --benchmarks vqav2 mmbench
"""

from __future__ import annotations

import argparse
import logging

from experiment_utils import (
    BASELINE_DEFENSES,
    SURROGATE_MODELS,
    run_benchmark,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

ALL_BENCHMARKS = ["vqav2", "mmbench", "mme", "orbench"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 6: Capability Preservation")
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--defenses", nargs="*",
                        default=BASELINE_DEFENSES + ["rdsa"])
    parser.add_argument("--benchmarks", nargs="*", default=ALL_BENCHMARKS)
    parser.add_argument("--output-dir", type=str,
                        default="results/exp6_capability")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_exp6(args: argparse.Namespace) -> None:
    models = SURROGATE_MODELS if args.model == "all" else [args.model]
    all_results: dict[str, dict[str, object]] = {}

    for model in models:
        all_results[model] = {}

        for defense in args.defenses:
            logger.info("=== Capability: %s / %s ===", model, defense)

            # Determine checkpoint
            if defense == "vanilla":
                checkpoint_dir = "none"
            elif defense == "rdsa":
                checkpoint_dir = f"outputs/exp1/{model}/rdsa/seed_42"
            elif defense in ("circuit_breaker", "smoothvlm", "vlguard"):
                checkpoint_dir = f"checkpoints/baselines/{defense}/{model}"
            else:
                checkpoint_dir = f"outputs/exp1/{model}/{defense}/seed_42"

            eval_dir = f"{args.output_dir}/{model}/{defense}"
            metrics = run_benchmark(
                model=model,
                defense=defense,
                checkpoint_dir=checkpoint_dir,
                benchmarks=args.benchmarks,
                output_dir=eval_dir,
                dry_run=args.dry_run,
            )
            all_results[model][defense] = metrics

    save_results(all_results, f"{args.output_dir}/exp6_capability.json")

    # Generate LaTeX table
    for model in models:
        _generate_capability_table(
            all_results.get(model, {}), model, args.output_dir
        )


def _generate_capability_table(
    results: dict[str, object],
    model: str,
    output_dir: str,
) -> None:
    """Generate a LaTeX table for capability benchmarks."""
    benchmarks = ["vqav2", "mmbench", "mme_perception", "mme_cognition", "or_rate"]
    header = " & ".join(
        ["Method", "VQAv2$\\uparrow$", "MMBench$\\uparrow$",
         "MME-P$\\uparrow$", "MME-C$\\uparrow$", "OR\\%$\\downarrow$"]
    )

    rows = []
    for defense, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        cells = [defense.replace("_", r"\_")]
        for bm in benchmarks:
            val = metrics.get(bm, "-")
            if isinstance(val, (int, float)):
                cells.append(f"{val:.1f}")
            else:
                cells.append(str(val))
        rows.append(" & ".join(cells) + r" \\")

    table = (
        r"\begin{table}[t]" + "\n"
        r"\centering" + "\n"
        f"\\caption{{Capability preservation on {model}}}\n"
        f"\\label{{tab:capability_{model}}}\n"
        r"\begin{tabular}{lccccc}" + "\n"
        r"\toprule" + "\n"
        f"{header} \\\\\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    path = f"{output_dir}/{model}_table4.tex"
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)
    logger.info("LaTeX table saved to %s", path)


def main() -> None:
    setup_logging("logs/exp6_capability.log")
    args = parse_args()
    run_exp6(args)


if __name__ == "__main__":
    main()
