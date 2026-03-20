"""Collect and aggregate results from all experiments into summary tables.

Generates:
- JSON summary of all experiments
- LaTeX tables ready for NeurIPS paper
- Human-readable text summary

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --results-dir results/ --output-dir paper/tables/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from experiment_utils import load_results, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Root results directory")
    parser.add_argument("--output-dir", type=str, default="paper/tables/",
                        help="Output directory for tables")
    parser.add_argument("--model", type=str, default="qwen3vl")
    return parser.parse_args()


def collect_exp1(results_dir: str, model: str) -> dict[str, object] | None:
    """Collect Exp 1: Main comparison."""
    path = f"{results_dir}/exp1_main_comparison/exp1_results.json"
    try:
        data = load_results(path)
        return data.get(model, {})
    except FileNotFoundError:
        logger.warning("Exp 1 results not found: %s", path)
        return None


def collect_exp2(results_dir: str, model: str) -> dict[str, object] | None:
    """Collect Exp 2: Ablation."""
    path = f"{results_dir}/exp2_ablation/{model}_ablation.json"
    try:
        return load_results(path)
    except FileNotFoundError:
        logger.warning("Exp 2 results not found: %s", path)
        return None


def collect_exp6(results_dir: str, model: str) -> dict[str, object] | None:
    """Collect Exp 6: Capability."""
    path = f"{results_dir}/exp6_capability/exp6_capability.json"
    try:
        data = load_results(path)
        return data.get(model, {})
    except FileNotFoundError:
        logger.warning("Exp 6 results not found: %s", path)
        return None


def collect_exp7(results_dir: str, model: str) -> dict[str, object] | None:
    """Collect Exp 7: Adaptive attacks."""
    path = f"{results_dir}/exp7_adaptive/exp7_adaptive.json"
    try:
        data = load_results(path)
        return data.get(model, {})
    except FileNotFoundError:
        logger.warning("Exp 7 results not found: %s", path)
        return None


def generate_text_summary(all_data: dict[str, object], model: str) -> str:
    """Generate a human-readable text summary."""
    lines = [
        f"{'=' * 60}",
        f"  RDSA NeurIPS Results Summary — {model}",
        f"{'=' * 60}",
        "",
    ]

    # Exp 1
    exp1 = all_data.get("exp1")
    if exp1:
        lines.append("## Experiment 1: Main Comparison")
        for defense, attacks in exp1.items():
            if not isinstance(attacks, dict):
                continue
            lines.append(f"  {defense}:")
            for attack, metrics in attacks.items():
                if isinstance(metrics, dict):
                    asr = metrics.get("asr", {})
                    if isinstance(asr, dict):
                        lines.append(
                            f"    {attack}: ASR={asr.get('mean', 0)*100:.1f}% "
                            f"(±{asr.get('std', 0)*100:.1f}%)"
                        )
        lines.append("")

    # Exp 2
    exp2 = all_data.get("exp2")
    if exp2:
        lines.append("## Experiment 2: Ablation")
        for variant, attacks in exp2.items():
            if not isinstance(attacks, dict):
                continue
            asr_strs = []
            for attack, metrics in attacks.items():
                if isinstance(metrics, dict):
                    asr = metrics.get("asr", {})
                    if isinstance(asr, dict):
                        asr_strs.append(
                            f"{attack}={asr.get('mean', 0)*100:.1f}%"
                        )
            if asr_strs:
                lines.append(f"  {variant}: {', '.join(asr_strs)}")
        lines.append("")

    # Exp 7
    exp7 = all_data.get("exp7")
    if exp7:
        lines.append("## Experiment 7: Adaptive Attacks")
        for budget, attacks in exp7.items():
            if not isinstance(attacks, dict):
                continue
            lines.append(f"  Budget: {budget}")
            for attack, metrics in attacks.items():
                if isinstance(metrics, dict):
                    asr = metrics.get("asr", {})
                    if isinstance(asr, dict):
                        lines.append(
                            f"    {attack}: ASR={asr.get('mean', 0)*100:.1f}%"
                        )
        lines.append("")

    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = {
        "exp1": collect_exp1(args.results_dir, args.model),
        "exp2": collect_exp2(args.results_dir, args.model),
        "exp6": collect_exp6(args.results_dir, args.model),
        "exp7": collect_exp7(args.results_dir, args.model),
    }

    # Save full JSON
    json_path = output_dir / f"{args.model}_all_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, default=str)
    logger.info("Full results: %s", json_path)

    # Text summary
    summary = generate_text_summary(all_data, args.model)
    summary_path = output_dir / f"{args.model}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    logger.info("Text summary: %s", summary_path)
    print(summary)


if __name__ == "__main__":
    main()
