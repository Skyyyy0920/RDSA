"""Experiment 4: Cross-Layer Redundancy Analysis (Figures 4-5).

4a. Single-group vs multi-group attack cost (G=1,2,3)
4b. Gradient direction alignment across groups
4c. t-SNE visualization of cross-layer safety projections

Usage:
    python scripts/exp4_redundancy_analysis.py --model qwen3vl
    python scripts/exp4_redundancy_analysis.py --model qwen3vl --sub gradient_alignment
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

# Layer group configs for G=1,2,3 (Qwen3-VL-8B example)
GROUP_CONFIGS: dict[str, dict[int, list[list[int]]]] = {
    "qwen3vl": {
        1: [[16, 17, 18, 19, 20]],
        2: [[10, 11, 12, 13, 14], [22, 23, 24, 25, 26]],
        3: [[8, 9, 10, 11, 12], [16, 17, 18, 19, 20], [24, 25, 26, 27, 28]],
    },
    "gemma3": {
        1: [[24, 25, 26, 27, 28, 29]],
        2: [[14, 15, 16, 17, 18, 19], [34, 35, 36, 37, 38, 39]],
        3: [[12, 13, 14, 15, 16, 17], [24, 25, 26, 27, 28, 29],
            [36, 37, 38, 39, 40, 41]],
    },
    "llama": {
        1: [[16, 17, 18, 19, 20]],
        2: [[10, 11, 12, 13, 14], [22, 23, 24, 25, 26]],
        3: [[8, 9, 10, 11, 12], [16, 17, 18, 19, 20], [24, 25, 26, 27, 28]],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp 4: Redundancy Analysis")
    parser.add_argument("--model", type=str, default="qwen3vl")
    parser.add_argument("--sub", type=str, default="all",
                        choices=["all", "group_count", "gradient_alignment",
                                 "tsne"])
    parser.add_argument("--output-dir", type=str,
                        default="results/exp4_redundancy")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_4a_group_count(args: argparse.Namespace) -> None:
    """Train RDSA with G=1,2,3 groups and compare attack cost."""
    logger.info("=== Exp 4a: Group Count vs Attack Cost ===")
    config_path = CONFIG_MAP[args.model]
    group_cfgs = GROUP_CONFIGS.get(args.model, GROUP_CONFIGS["qwen3vl"])
    results: dict[int, object] = {}

    for g, layer_groups in group_cfgs.items():
        logger.info("Training with G=%d groups: %s", g, layer_groups)
        train_dir = f"outputs/exp4/{args.model}/G_{g}"

        # Convert layer groups to Hydra-compatible format
        # Hydra list override: model.layer_groups=[[8,9,10],[16,17,18]]
        groups_str = str(layer_groups).replace(" ", "")
        run_training(
            config_path=config_path,
            overrides={"model.layer_groups": groups_str},
            output_dir=train_dir,
            dry_run=args.dry_run,
        )

        # Evaluate with adaptive PGD (multi-layer attack)
        for attack in ["scia", "adaptive_pgd"]:
            eval_dir = f"{args.output_dir}/{args.model}/4a/G_{g}/{attack}"
            metrics = run_evaluation(
                model=args.model, defense="rdsa", attack=attack,
                checkpoint_dir=train_dir, output_dir=eval_dir,
                max_samples=args.max_samples, dry_run=args.dry_run,
            )
            if g not in results:
                results[g] = {}
            results[g][attack] = metrics

    save_results(results, f"{args.output_dir}/{args.model}/4a_group_count.json")


def run_4b_gradient_alignment(args: argparse.Namespace) -> None:
    """Measure cosine similarity of attack gradients across groups.

    Post-hoc analysis: load trained model, compute per-group PGD gradients,
    measure pairwise cosine similarity.
    """
    logger.info("=== Exp 4b: Gradient Direction Alignment ===")
    analysis_config = {
        "model": args.model,
        "checkpoint_vanilla": f"outputs/exp4/{args.model}/vanilla",
        "checkpoint_rdsa": f"outputs/exp4/{args.model}/G_3",
        "description": (
            "For each model (vanilla vs RDSA), run PGD on eval data and "
            "record the gradient ∂L/∂δ at each group's representative layer. "
            "Compute pairwise cosine similarity across groups. "
            "RDSA should show lower cross-group gradient alignment."
        ),
        "analysis_steps": [
            "1. Load model + V_s per group",
            "2. For each eval sample, run 1 step of PGD per group",
            "3. Record grad_delta for each group (shape [B, d_s])",
            "4. Compute cos_sim(grad_g1, grad_g2) for all group pairs",
            "5. Report mean ± std of pairwise cosine similarities",
            "6. Plot heatmap: groups × groups (vanilla vs RDSA side by side)",
        ],
    }
    save_results(
        analysis_config,
        f"{args.output_dir}/{args.model}/4b_gradient_config.json",
    )


def run_4c_tsne(args: argparse.Namespace) -> None:
    """t-SNE visualization of cross-layer safety projections."""
    logger.info("=== Exp 4c: t-SNE Cross-Layer Projections ===")
    analysis_config = {
        "model": args.model,
        "description": (
            "Collect safety projections (h @ V_s) at each group's "
            "representative layer for both harmful and benign inputs. "
            "Apply t-SNE to visualize in 2D. Compare vanilla vs RDSA."
        ),
        "expected_result": (
            "Vanilla: projections from different groups form separate clusters "
            "(disentangled → attackable independently). "
            "RDSA: projections from different groups overlap "
            "(consistent → must attack all groups simultaneously)."
        ),
    }
    save_results(
        analysis_config,
        f"{args.output_dir}/{args.model}/4c_tsne_config.json",
    )


def main() -> None:
    setup_logging("logs/exp4_redundancy.log")
    args = parse_args()

    subs = (
        ["group_count", "gradient_alignment", "tsne"]
        if args.sub == "all"
        else [args.sub]
    )

    for sub in subs:
        if sub == "group_count":
            run_4a_group_count(args)
        elif sub == "gradient_alignment":
            run_4b_gradient_alignment(args)
        elif sub == "tsne":
            run_4c_tsne(args)


if __name__ == "__main__":
    main()
