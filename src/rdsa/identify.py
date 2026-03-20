"""CLI entry point for safety subspace identification.

Usage:
    python -m rdsa.identify --model qwen3vl --output subspaces/
    python -m rdsa.identify --config configs/gemma3.yaml

Can also be run directly:
    python src/rdsa/identify.py --model qwen3vl --output subspaces/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package is importable when running this file directly
# (e.g. `python src/rdsa/identify.py` instead of `python -m rdsa.identify`)
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rdsa.config import ModelConfig, RDSAConfig, SubspaceConfig  # noqa: E402
from rdsa.subspace.metrics import entanglement_degree  # noqa: E402

logger = logging.getLogger(__name__)

# Default configurations per model
MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "qwen3vl": {
        "name": "Qwen/Qwen3-VL-8B-Instruct",
        "architecture": "qwen3vl",
        "hidden_dim": 4096,
        "num_layers": 32,
        "layer_groups": [
            [8, 9, 10, 11, 12],
            [16, 17, 18, 19, 20],
            [24, 25, 26, 27, 28],
        ],
        "subspace_dir": "subspaces/qwen3-vl-8b",
    },
    "gemma3": {
        "name": "google/gemma-3-12b-it",
        "architecture": "gemma3",
        "hidden_dim": 3840,
        "num_layers": 48,
        "layer_groups": [
            [12, 13, 14, 15, 16, 17],
            [24, 25, 26, 27, 28, 29],
            [36, 37, 38, 39, 40, 41],
        ],
        "subspace_dir": "subspaces/gemma-3-12b",
    },
    "llama": {
        "name": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "architecture": "llama_vision",
        "hidden_dim": 4096,
        "num_layers": 32,
        "layer_groups": [
            [8, 9, 10, 11, 12],
            [16, 17, 18, 19, 20],
            [24, 25, 26, 27, 28],
        ],
        "subspace_dir": "subspaces/llama-3.2-11b",
    },
}

# Reverse lookup: HuggingFace model name → shortname
_HF_NAME_TO_SHORT: dict[str, str] = {
    str(v["name"]).lower(): k for k, v in MODEL_CONFIGS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RDSA Safety Subspace Identification (SVD + PCA)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3vl",
        help=(
            "Target model: shortname (qwen3vl, gemma3, llama) "
            "or full HuggingFace name (e.g. Qwen/Qwen3-VL-8B-Instruct)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="subspaces/",
        help="Output directory for subspace tensors",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides --model)",
    )
    parser.add_argument(
        "--safe-data",
        type=str,
        default="data/safe_prompts.jsonl",
        help="Path to safe prompts JSONL",
    )
    parser.add_argument(
        "--unsafe-data",
        type=str,
        default="data/unsafe_prompts.jsonl",
        help="Path to unsafe prompts JSONL",
    )
    parser.add_argument(
        "--semantic-data",
        type=str,
        default="data/safe_prompts.jsonl",
        help="Path to semantic (normal) data JSONL for PCA",
    )
    parser.add_argument(
        "--d-safe",
        type=int,
        default=32,
        help="Safety subspace dimensionality",
    )
    parser.add_argument(
        "--d-semantic",
        type=int,
        default=256,
        help="Semantic subspace dimensionality",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for activation collection",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2000,
        help="Maximum contrast samples to collect",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for model inference (default: auto)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    args = parse_args()

    # Resolve model name: accept both shortnames and full HuggingFace names
    model_key = args.model
    if model_key not in MODEL_CONFIGS:
        resolved = _HF_NAME_TO_SHORT.get(model_key.lower())
        if resolved is not None:
            logger.info(
                "Resolved HuggingFace name '%s' → shortname '%s'",
                model_key,
                resolved,
            )
            model_key = resolved
        else:
            logger.error(
                "Unknown model '%s'. Valid shortnames: %s. "
                "Valid HuggingFace names: %s",
                model_key,
                list(MODEL_CONFIGS.keys()),
                [str(v["name"]) for v in MODEL_CONFIGS.values()],
            )
            sys.exit(1)

    # Build config
    if args.config is not None:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(args.config)
        model_cfg = ModelConfig(
            name=cfg.model.name,
            architecture=cfg.model.architecture,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            layer_groups=[list(g) for g in cfg.model.layer_groups],
        )
    else:
        model_dict = MODEL_CONFIGS[model_key]
        model_cfg = ModelConfig(
            name=str(model_dict["name"]),
            architecture=str(model_dict["architecture"]),
            hidden_dim=int(model_dict["hidden_dim"]),  # type: ignore[arg-type]
            num_layers=int(model_dict["num_layers"]),  # type: ignore[arg-type]
            layer_groups=list(model_dict["layer_groups"]),  # type: ignore[arg-type]
        )

    subspace_cfg = SubspaceConfig(
        d_safe=args.d_safe,
        d_semantic=args.d_semantic,
        n_contrast_samples=args.max_samples,
    )
    config = RDSAConfig(model=model_cfg, subspace=subspace_cfg)

    # Determine output directory
    if args.config is not None and hasattr(cfg, "output") and hasattr(cfg.output, "subspace_dir"):
        # Use subspace_dir from YAML config
        output_dir = Path(cfg.output.subspace_dir)
    elif args.output != "subspaces/" or args.config is not None:
        # User explicitly specified --output, use as-is
        output_dir = Path(args.output)
    else:
        # Default: use model-specific subspace_dir from MODEL_CONFIGS
        model_dict = MODEL_CONFIGS[model_key]
        output_dir = Path(str(model_dict["subspace_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    from rdsa.models.model_utils import get_layer_accessor, load_model_and_processor

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model: %s", model_cfg.name)
    model, processor = load_model_and_processor(model_cfg, device=device)

    # Validate data paths before loading
    for label, path in [
        ("safe", args.safe_data),
        ("unsafe", args.unsafe_data),
    ]:
        if not Path(path).is_file():
            logger.error(
                "%s data file not found: %s\n"
                "Expected JSONL format with one JSON object per line:\n"
                '  {"prompt": "..."}\n'
                "Create data files or specify paths via --safe-data / --unsafe-data.",
                label,
                path,
            )
            sys.exit(1)

    # Create data loaders
    from rdsa.training.data import ContrastDataset, SplitContrastDataset

    contrast_ds = ContrastDataset(
        safe_path=args.safe_data,
        unsafe_path=args.unsafe_data,
        processor=processor,
    )

    safe_ds = SplitContrastDataset(contrast_ds, side="safe")
    unsafe_ds = SplitContrastDataset(contrast_ds, side="unsafe")

    safe_loader = DataLoader(safe_ds, batch_size=args.batch_size, shuffle=False)
    unsafe_loader = DataLoader(unsafe_ds, batch_size=args.batch_size, shuffle=False)

    # Identify subspaces
    from rdsa.subspace.identifier import SafetySubspaceIdentifier

    layer_accessor = get_layer_accessor(model_cfg.architecture)
    identifier = SafetySubspaceIdentifier(
        model=model,
        config=config,
        device=device,
        layer_accessor=layer_accessor,
    )

    logger.info("Identifying safety and semantic subspaces...")
    logger.info("  Layer groups: %s", model_cfg.layer_groups)
    logger.info("  d_s=%d, d_t=%d", args.d_safe, args.d_semantic)

    results = identifier.identify_all_groups(
        safe_dataloader=safe_loader,
        unsafe_dataloader=unsafe_loader,
        normal_dataloader=safe_loader,  # Reuse safe data for semantic PCA
    )

    # Report results
    for r in results:
        eta = entanglement_degree(r.V_s, r.V_t).item()
        logger.info(
            "Group %d (layer %d): V_s %s, V_t %s, eta_0=%.4f, "
            "top singular value=%.4f",
            r.layer_group_idx,
            r.representative_layer,
            list(r.V_s.shape),
            list(r.V_t.shape),
            eta,
            r.singular_values[0].item(),
        )

    # Save
    identifier.save_subspaces(results, str(output_dir))
    logger.info("Subspaces saved to %s", output_dir)

if __name__ == "__main__":
    main()
