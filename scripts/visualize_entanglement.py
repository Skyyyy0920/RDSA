"""Visualize entanglement degree (eta) profiles across layers.

Generates layer-wise entanglement degree plots comparing vanilla (pre-RDSA)
and RDSA-trained models. Used for Exp 3a in EXPERIMENT_DESIGN.md.

Usage:
    python scripts/visualize_entanglement.py --model qwen3vl
    python scripts/visualize_entanglement.py --model qwen3vl \
        --subspace-dir subspaces/ --output figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize entanglement degree profiles"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3vl",
        choices=["qwen3vl", "gemma3", "llama"],
        help="Model shortname",
    )
    parser.add_argument(
        "--subspace-dir",
        type=str,
        default="subspaces/",
        help="Directory with saved subspace results",
    )
    parser.add_argument(
        "--rdsa-subspace-dir",
        type=str,
        default=None,
        help="Directory with RDSA-trained subspace results (for comparison)",
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
        help="Output file format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for raster formats",
    )
    return parser.parse_args()


def load_subspace_results(
    subspace_dir: str,
) -> list[dict[str, torch.Tensor]]:
    """Load subspace results from a directory.

    Returns:
        List of dicts with V_s, V_t, layer_group_idx, representative_layer.
    """
    path = Path(subspace_dir)
    metadata = torch.load(path / "metadata.pt", map_location="cpu", weights_only=True)
    results = []

    for group_idx in range(metadata["num_groups"]):
        data = torch.load(
            path / f"group_{group_idx}.pt",
            map_location="cpu",
            weights_only=True,
        )
        results.append(data)

    return results


def compute_per_layer_eta(results: list[dict[str, torch.Tensor]]) -> dict[int, float]:
    """Compute entanglement degree for each layer group.

    Returns:
        Dict mapping representative layer index to eta value.
    """
    from rdsa.subspace.metrics import entanglement_degree

    eta_dict = {}
    for r in results:
        V_s = r["V_s"]
        V_t = r["V_t"]
        layer = r["representative_layer"]
        eta = entanglement_degree(V_s, V_t).item()
        eta_dict[layer] = eta

    return eta_dict


def plot_entanglement_profile(
    vanilla_eta: dict[int, float],
    rdsa_eta: dict[int, float] | None = None,
    model_name: str = "Qwen3-VL-8B",
    output_path: str = "figures/entanglement_profile.pdf",
    dpi: int = 300,
) -> None:
    """Plot layer-wise entanglement degree comparison.

    Args:
        vanilla_eta: {layer_idx: eta} for vanilla model.
        rdsa_eta: Optional {layer_idx: eta} for RDSA-trained model.
        model_name: Display name for the model.
        output_path: Path to save the figure.
        dpi: DPI for raster formats.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    layers = sorted(vanilla_eta.keys())
    etas_vanilla = [vanilla_eta[layer] for layer in layers]

    ax.plot(
        layers,
        etas_vanilla,
        "o-",
        color="#2196F3",
        linewidth=2,
        markersize=8,
        label="Vanilla (pre-RDSA)",
    )

    if rdsa_eta is not None:
        rdsa_layers = sorted(rdsa_eta.keys())
        etas_rdsa = [rdsa_eta[layer] for layer in rdsa_layers]
        ax.plot(
            rdsa_layers,
            etas_rdsa,
            "s-",
            color="#F44336",
            linewidth=2,
            markersize=8,
            label="RDSA-trained",
        )

    ax.set_xlabel("Layer Index", fontsize=13)
    ax.set_ylabel("Entanglement Degree (η)", fontsize=13)
    ax.set_title(f"Safety-Semantic Entanglement Profile — {model_name}", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    # Annotate target
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Target η=1")

    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved entanglement profile to %s", output_path)


def plot_eta_vs_asr(
    eta_values: list[float],
    asr_values: list[float],
    output_path: str = "figures/eta_vs_asr.pdf",
    dpi: int = 300,
) -> None:
    """Plot eta vs ASR scatter with fitted curve (Exp 3b).

    Args:
        eta_values: List of entanglement degrees.
        asr_values: List of corresponding ASR values.
        output_path: Path to save the figure.
        dpi: DPI for raster formats.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(eta_values, asr_values, s=80, c="#2196F3", zorder=5)

    # Fit a line
    if len(eta_values) >= 2:
        coeffs = np.polyfit(eta_values, asr_values, 1)
        x_fit = np.linspace(min(eta_values), max(eta_values), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, "--", color="#F44336", linewidth=1.5, label="Linear fit")

    ax.set_xlabel("Entanglement Degree (η)", fontsize=13)
    ax.set_ylabel("Attack Success Rate (ASR %)", fontsize=13)
    ax.set_title("Entanglement vs Attack Success Rate", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved eta vs ASR plot to %s", output_path)


def plot_manipulable_dimensions(
    eta_values: list[float],
    d_s: int = 32,
    output_path: str = "figures/manipulable_dims.pdf",
    dpi: int = 300,
) -> None:
    """Plot manipulable dimensions as function of eta (Exp 3c).

    Theoretical prediction: dim(V_s^perp) = d_s * (1 - eta).

    Args:
        eta_values: List of measured entanglement degrees.
        d_s: Safety subspace dimensionality.
        output_path: Path to save the figure.
        dpi: DPI for raster formats.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Theoretical curve
    eta_theory = np.linspace(0, 1, 100)
    dims_theory = d_s * (1 - eta_theory)
    ax.plot(
        eta_theory,
        dims_theory,
        "-",
        color="#9E9E9E",
        linewidth=2,
        label="Theoretical: $d_s(1-\\eta)$",
    )

    # Measured points
    dims_measured = [d_s * (1 - eta) for eta in eta_values]
    ax.scatter(
        eta_values,
        dims_measured,
        s=80,
        c="#F44336",
        zorder=5,
        label="Measured",
    )

    ax.set_xlabel("Entanglement Degree (η)", fontsize=13)
    ax.set_ylabel("Manipulable Dimensions", fontsize=13)
    ax.set_title("Attacker's Manipulable Subspace", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved manipulable dimensions plot to %s", output_path)


MODEL_DISPLAY_NAMES = {
    "qwen3vl": "Qwen3-VL-8B-Instruct",
    "gemma3": "Gemma-3-12B-IT",
    "llama": "LLaMA-3.2-11B-Vision",
}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_display = MODEL_DISPLAY_NAMES.get(args.model, args.model)

    # Load vanilla subspaces
    vanilla_dir = Path(args.subspace_dir) / args.model
    if not vanilla_dir.exists():
        vanilla_dir = Path(args.subspace_dir)

    logger.info("Loading subspaces from %s", vanilla_dir)
    vanilla_results = load_subspace_results(str(vanilla_dir))
    vanilla_eta = compute_per_layer_eta(vanilla_results)
    logger.info("Vanilla eta: %s", vanilla_eta)

    # Load RDSA subspaces if available
    rdsa_eta = None
    if args.rdsa_subspace_dir is not None:
        rdsa_dir = Path(args.rdsa_subspace_dir)
        if rdsa_dir.exists():
            rdsa_results = load_subspace_results(str(rdsa_dir))
            rdsa_eta = compute_per_layer_eta(rdsa_results)
            logger.info("RDSA eta: %s", rdsa_eta)

    # Plot entanglement profile
    plot_entanglement_profile(
        vanilla_eta=vanilla_eta,
        rdsa_eta=rdsa_eta,
        model_name=model_display,
        output_path=str(output_dir / f"entanglement_profile_{args.model}.{args.format}"),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
