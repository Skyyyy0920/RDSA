"""CLI entry point for RDSA training.

Usage:
    python -m rdsa.train --config configs/qwen3vl.yaml
    python -m rdsa.train --config-name gemma3
    python -m rdsa.train training.alpha_sa_at=0.5

Can also be run directly:
    python src/rdsa/train.py --config configs/qwen3vl.yaml
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the package is importable when running this file directly
# (e.g. `python src/rdsa/train.py` instead of `python -m rdsa.train`)
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def _preprocess_argv() -> None:
    """Translate ``--config <file>`` and ``--config-path <file>`` to Hydra args.

    Hydra expects ``--config-path <dir>`` and ``--config-name <stem>`` but
    users naturally pass a full file path.  This rewrites sys.argv so that
    ``--config configs/qwen3vl.yaml`` becomes
    ``--config-path configs --config-name qwen3vl``.
    """
    for flag in ("--config", "--config-path"):
        if flag not in sys.argv:
            continue
        idx = sys.argv.index(flag)
        if idx + 1 >= len(sys.argv):
            continue
        value = sys.argv[idx + 1]
        p = Path(value)
        # Only rewrite if value looks like a file (has a suffix or the path
        # exists as a file).  Pure directory paths are left alone.
        if p.suffix or (p.exists() and p.is_file()):
            # Hydra CLI --config-path is relative to CWD; resolve to
            # absolute so it works regardless of working directory.
            parent = p.parent if str(p.parent) != "." else Path.cwd()
            config_dir = str(parent.resolve())
            config_name = p.stem
            # Replace the two argv entries with --config-path <dir>
            sys.argv[idx] = "--config-path"
            sys.argv[idx + 1] = config_dir
            # Append --config-name (only if user didn't already provide one)
            if "--config-name" not in sys.argv:
                sys.argv.extend(["--config-name", config_name])
            break


_preprocess_argv()

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from rdsa.config import (  # noqa: E402
    ModelConfig,
    MonitorConfig,
    RDSAConfig,
    SubspaceConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)


def _build_config(cfg: DictConfig) -> RDSAConfig:
    """Convert Hydra DictConfig to typed RDSAConfig dataclass."""
    model_cfg = ModelConfig(
        name=cfg.model.name,
        architecture=cfg.model.architecture,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        layer_groups=[list(g) for g in cfg.model.layer_groups],
        model_class=cfg.model.get("model_class", "Qwen3VLForConditionalGeneration"),
        layer_access_path=cfg.model.get("layer_access_path", "model.language_model.layers"),
    )
    subspace_cfg = SubspaceConfig(
        d_safe=cfg.subspace.d_safe,
        d_semantic=cfg.subspace.d_semantic,
        n_contrast_samples=cfg.subspace.n_contrast_samples,
        n_semantic_samples=cfg.subspace.n_semantic_samples,
    )
    training_cfg = TrainingConfig(
        lora_rank=cfg.training.lora_rank,
        lora_alpha=cfg.training.lora_alpha,
        lora_target_modules=list(cfg.training.lora_target_modules),
        lora_dropout=cfg.training.lora_dropout,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_ratio=cfg.training.warmup_ratio,
        num_epochs=cfg.training.num_epochs,
        per_device_batch_size=cfg.training.per_device_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        max_grad_norm=cfg.training.max_grad_norm,
        fp16=cfg.training.fp16,
        bf16=cfg.training.get("bf16", True),
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        alpha_sa_at=cfg.training.alpha_sa_at,
        alpha_consist=cfg.training.alpha_consist,
        sa_at_pgd_steps=cfg.training.sa_at_pgd_steps,
        sa_at_pgd_alpha=cfg.training.sa_at_pgd_alpha,
        sa_at_epsilon=cfg.training.sa_at_epsilon,
        harmful_benign_ratio=cfg.training.harmful_benign_ratio,
    )
    monitor_cfg = MonitorConfig(
        threshold=cfg.monitor.threshold,
        conservative_mode=cfg.monitor.conservative_mode,
    )
    return RDSAConfig(
        model=model_cfg,
        subspace=subspace_cfg,
        training=training_cfg,
        monitor=monitor_cfg,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="qwen3vl")
def main(cfg: DictConfig) -> None:
    """RDSA training pipeline.

    Steps:
    1. Load model and processor
    2. Load pre-identified subspaces (or identify them)
    3. Apply LoRA adapters
    4. Create data loaders
    5. Train with SA-AT + consistency loss
    """
    logger.info("RDSA Training Pipeline")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    config = _build_config(cfg)

    # 1. Load model
    from rdsa.models.model_utils import apply_lora, load_model_and_processor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model: %s", config.model.name)
    model, processor = load_model_and_processor(config.model, device=device)

    # 2. Load or identify subspaces
    subspace_dir = Path(cfg.output.subspace_dir)
    if (subspace_dir / "metadata.pt").exists():
        from rdsa.subspace.identifier import SafetySubspaceIdentifier

        logger.info("Loading pre-identified subspaces from %s", subspace_dir)
        identifier = SafetySubspaceIdentifier(model, config, device=device)
        subspace_results = identifier.load_subspaces(str(subspace_dir))
    else:
        logger.error(
            "Subspaces not found at %s. Run `python -m rdsa.identify` first.",
            subspace_dir,
        )
        sys.exit(1)

    # 3. Apply LoRA
    logger.info("Applying LoRA adapters (rank=%d)", config.training.lora_rank)
    model = apply_lora(model, config.training)

    # Cast LoRA parameters to fp32 for optimizer precision.
    # Base model stays bf16 to save memory; only trainable LoRA weights
    # are promoted so that AdamW doesn't lose precision in bf16.
    n_cast = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            n_cast += 1
    logger.info(
        "Cast %d LoRA parameter tensors from bf16 to fp32 for optimizer precision",
        n_cast,
    )

    # 4. Create data loaders
    from rdsa.training.data import (
        ContrastDataset,
        create_contrast_dataloaders,
        create_train_dataloader,
    )

    # Load harmful/benign samples from JSONL files
    harmful_samples = ContrastDataset._load_jsonl(cfg.data.contrast_unsafe_path)
    benign_samples = ContrastDataset._load_jsonl(cfg.data.contrast_safe_path)

    train_dataloader = create_train_dataloader(
        harmful_samples=harmful_samples,
        benign_samples=benign_samples,
        processor=processor,
        batch_size=config.training.per_device_batch_size,
        ratio=config.training.harmful_benign_ratio,
    )

    # Create contrast dataloaders for subspace re-identification between epochs
    safe_dl, unsafe_dl = create_contrast_dataloaders(
        safe_path=cfg.data.contrast_safe_path,
        unsafe_path=cfg.data.contrast_unsafe_path,
        processor=processor,
        batch_size=config.training.per_device_batch_size,
    )

    # 5. Train
    from rdsa.subspace.identifier import SafetySubspaceIdentifier as _Identifier
    from rdsa.training.trainer import RDSATrainer

    def re_identify(
        current_model: torch.nn.Module, epoch: int
    ) -> list:
        """Re-identify V_s/V_t from the model's current activations."""
        identifier = _Identifier(
            current_model, config, device=device,
            layer_accessor=config.model.layer_access_path,
        )
        # Use safe prompts as normal data for PCA (semantic subspace)
        return identifier.identify_all_groups(safe_dl, unsafe_dl, safe_dl)

    trainer = RDSATrainer(
        model=model,
        processor=processor,
        config=config,
        subspace_results=subspace_results,
        train_dataloader=train_dataloader,
        device=device,
    )

    logger.info("Starting training for %d epochs", config.training.num_epochs)
    metrics = trainer.train(re_identify_fn=re_identify)

    logger.info("Training complete. Final metrics:")
    for key, value in sorted(metrics.items()):
        logger.info("  %s: %.6f", key, value)


if __name__ == "__main__":
    main()
