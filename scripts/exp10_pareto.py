"""Experiment 10: Safety-Utility Pareto Frontier (Figure 1).

Aggregates results from Exp 1 (baselines) and Exp 9 (α_sa_at sweep)
to plot the Pareto frontier showing RDSA dominates all baselines.

Usage:
    python scripts/exp10_pareto.py --model qwen3vl
    python scripts/exp10_pareto.py --model all --format pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiment_utils import (
    SURROGATE_MODELS,
    load_results,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exp 10: Safety-Utility Pareto Frontier"
    )
    parser.add_argument("--model", type=str, default="qwen3vl",
                        help="Model shortname or 'all'")
    parser.add_argument("--exp1-results", type=str,
                        default="results/exp1_main_comparison/exp1_results.json")
    parser.add_argument("--exp9-results-dir", type=str,
                        default="results/exp9_sensitivity")
    parser.add_argument("--output-dir", type=str,
                        default="results/exp10_pareto")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def collect_pareto_data(
    model: str,
    exp1_path: str,
    exp9_dir: str,
) -> dict[str, object]:
    """Collect data points for the Pareto plot.

    Returns:
        Dict with 'rdsa' (list of points) and 'baselines' (dict of points).
    """
    pareto_data: dict[str, object] = {"rdsa": [], "baselines": {}}

    # ── Baselines from Exp 1 ──
    try:
        exp1 = load_results(exp1_path)
        model_data = exp1.get(model, {})

        for defense, attack_results in model_data.items():
            if defense == "rdsa":
                continue
            # Average ASR across non-adaptive attacks
            asr_values = []
            for _attack, data in attack_results.items():
                if isinstance(data, dict):
                    asr = data.get("asr", {})
                    if isinstance(asr, dict):
                        asr_values.append(asr.get("mean", 0))
                    elif isinstance(asr, (int, float)):
                        asr_values.append(asr)

            if asr_values:
                avg_asr = sum(asr_values) / len(asr_values)
                # Placeholder for VQA drop — needs Exp 6 data
                pareto_data["baselines"][defense] = {
                    "asr_reduction": (1.0 - avg_asr) * 100,
                    "vqa_drop": 0.0,  # Fill from Exp 6
                }
    except FileNotFoundError:
        logger.warning("Exp 1 results not found at %s", exp1_path)

    # ── RDSA points from Exp 9 α_sa_at sweep ──
    sweep_path = f"{exp9_dir}/{model}/alpha_sa_at_sweep.json"
    try:
        sweep = load_results(sweep_path)
        for val_str, data in sweep.items():
            safety = data.get("safety", {})
            capability = data.get("capability", {})

            asr = safety.get("asr", 0)
            if isinstance(asr, dict):
                asr = asr.get("mean", 0)

            vqa = capability.get("vqa_accuracy", 0)
            if isinstance(vqa, dict):
                vqa = vqa.get("mean", 0)

            pareto_data["rdsa"].append({
                "alpha": float(data.get("value", val_str)),
                "asr_reduction": (1.0 - asr) * 100,
                "vqa_drop": max(0, 100 - vqa),  # Approximate drop
            })
    except FileNotFoundError:
        logger.warning("Exp 9 sweep not found at %s", sweep_path)

    return pareto_data


def generate_pareto_plot(
    pareto_data: dict[str, object],
    model: str,
    output_dir: str,
    fmt: str = "pdf",
) -> None:
    """Generate the Pareto frontier plot using plot_pareto.py."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save data for plot_pareto.py
    data_file = output_path / f"{model}_pareto_data.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2, default=str)

    # Call plot_pareto.py
    import subprocess
    import sys

    cmd = [
        sys.executable, "scripts/plot_pareto.py",
        "--data", str(data_file),
        "--output", str(output_path / f"pareto_{model}.{fmt}"),
        "--format", fmt,
    ]
    logger.info("Generating Pareto plot: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Pareto plot saved to %s", output_path / f"pareto_{model}.{fmt}")
    except subprocess.CalledProcessError as e:
        logger.error("Plot failed: %s", e.stderr[-300:] if e.stderr else "")


def main() -> None:
    setup_logging("logs/exp10_pareto.log")
    args = parse_args()

    models = SURROGATE_MODELS if args.model == "all" else [args.model]

    for model in models:
        logger.info("=== Pareto: %s ===", model)
        pareto_data = collect_pareto_data(
            model, args.exp1_results, args.exp9_results_dir
        )
        save_results(
            pareto_data,
            f"{args.output_dir}/{model}_pareto_data.json",
        )

        if not args.dry_run:
            generate_pareto_plot(
                pareto_data, model, args.output_dir, args.format
            )


if __name__ == "__main__":
    main()
