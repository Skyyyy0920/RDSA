"""Plot Safety-Utility Pareto frontier.

Visualizes the trade-off between safety improvement (ASR reduction) and
capability preservation (VQA accuracy drop). Used for Exp 5 in
EXPERIMENT_DESIGN.md.

Usage:
    python scripts/plot_pareto.py --results results/pareto_data.json
    python scripts/plot_pareto.py --results results/pareto_data.json --output figures/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Safety-Utility Pareto frontier"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to JSON with Pareto data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["pdf", "png", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def plot_pareto_frontier(
    rdsa_points: list[dict[str, float]],
    baseline_points: dict[str, dict[str, float]] | None = None,
    output_path: str = "figures/pareto_frontier.pdf",
    dpi: int = 300,
) -> None:
    """Plot Safety-Utility Pareto frontier.

    Expected data format:
        rdsa_points: [{"alpha": 0.1, "asr_reduction": 30.5, "vqa_drop": 1.2}, ...]
        baseline_points: {"CB": {"asr_reduction": 25.0, "vqa_drop": 5.3}, ...}

    Args:
        rdsa_points: List of RDSA data points at different alpha values.
        baseline_points: Optional baseline method data points.
        output_path: Path to save the figure.
        dpi: DPI for raster formats.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # RDSA points and Pareto curve
    asr_reds = [p["asr_reduction"] for p in rdsa_points]
    vqa_drops = [p["vqa_drop"] for p in rdsa_points]
    alphas = [p.get("alpha", 0) for p in rdsa_points]

    ax.scatter(
        vqa_drops,
        asr_reds,
        s=100,
        c="#2196F3",
        marker="o",
        zorder=5,
        label="RDSA (varying α)",
    )

    # Annotate alpha values
    for alpha, vqa_d, asr_r in zip(alphas, vqa_drops, asr_reds, strict=True):
        if alpha > 0:
            ax.annotate(
                f"α={alpha}",
                (vqa_d, asr_r),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=8,
                color="#1565C0",
            )

    # Connect RDSA points as Pareto curve
    sorted_idx = np.argsort(vqa_drops)
    sorted_vqa = [vqa_drops[i] for i in sorted_idx]
    sorted_asr = [asr_reds[i] for i in sorted_idx]
    ax.plot(sorted_vqa, sorted_asr, "-", color="#2196F3", alpha=0.5, linewidth=1.5)

    # Baseline points
    if baseline_points:
        markers = {"CB": "^", "LAT": "s", "Safety SFT": "D", "SmoothVLM": "v"}
        colors = {
            "CB": "#F44336",
            "LAT": "#4CAF50",
            "Safety SFT": "#FF9800",
            "SmoothVLM": "#9C27B0",
        }

        for name, point in baseline_points.items():
            marker = markers.get(name, "x")
            color = colors.get(name, "#757575")
            ax.scatter(
                point["vqa_drop"],
                point["asr_reduction"],
                s=120,
                c=color,
                marker=marker,
                zorder=5,
                label=name,
            )

    # Ideal region annotation
    ax.annotate(
        "Ideal\nregion",
        xy=(0, max(asr_reds) if asr_reds else 50),
        fontsize=10,
        color="gray",
        alpha=0.7,
        ha="center",
    )

    ax.set_xlabel("VQA Accuracy Drop (%)", fontsize=13)
    ax.set_ylabel("ASR Reduction (%)", fontsize=13)
    ax.set_title("Safety-Utility Pareto Frontier", fontsize=14)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Invert x-axis direction: less drop is better (left)
    ax.set_xlim(left=-0.5)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Pareto frontier to %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    rdsa_points = data.get("rdsa", [])
    baseline_points = data.get("baselines", None)

    if not rdsa_points:
        logger.error("No RDSA data points found in %s", args.results)
        return

    plot_pareto_frontier(
        rdsa_points=rdsa_points,
        baseline_points=baseline_points,
        output_path=str(output_dir / f"pareto_frontier.{args.format}"),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
