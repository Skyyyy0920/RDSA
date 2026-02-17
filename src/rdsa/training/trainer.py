"""RDSA multi-objective training loop.

Orchestrates the training pipeline:
1. Forward pass with hooks to capture hidden states at representative layers
2. Compute SFT loss (standard cross-entropy)
3. Compute L_entangle, L_consist, L_LAT-sub
4. Backward pass with combined loss
5. Log all components to wandb

CRITICAL: Hook cleanup via context manager in every training step.
CRITICAL: Subspace projections in fp32 even under AMP.
CRITICAL: Gradient checkpointing + activation offloading for memory.
CRITICAL: Seed everything for reproducibility.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdsa.config import RDSAConfig
from rdsa.models.hooks import HookManager
from rdsa.models.model_utils import (
    enable_gradient_checkpointing,
    get_layer_accessor,
)
from rdsa.subspace.identifier import SubspaceResult
from rdsa.subspace.metrics import entanglement_degree
from rdsa.training.losses import RDSALoss


class RDSATrainer:
    """Multi-objective training loop for RDSA.

    Combines standard SFT loss with entanglement, consistency, and
    subspace-aware LAT losses. Uses LoRA for parameter-efficient fine-tuning.

    Args:
        model: The VLM with LoRA adapters applied.
        processor: HuggingFace processor (tokenizer + image processor).
        config: Full RDSA configuration.
        subspace_results: Pre-identified subspace results, one per layer group.
        train_dataloader: Training data loader.
        eval_dataloader: Optional evaluation data loader.
        device: Target device for training.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        config: RDSAConfig,
        subspace_results: list[SubspaceResult],
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Subspace bases (fp32, on device)
        self.V_s_list = [r.V_s.float().to(device) for r in subspace_results]
        self.V_t_list = [r.V_t.float().to(device) for r in subspace_results]
        self.rep_layers = [r.representative_layer for r in subspace_results]

        # Layer accessor for hooks
        self.layer_accessor = get_layer_accessor(config.model.architecture)

        # Loss
        self.rdsa_loss = RDSALoss(config.training)

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        total_steps = (
            len(train_dataloader)
            * config.training.num_epochs
            // config.training.gradient_accumulation_steps
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1)
        )

        # AMP
        self.use_amp = config.training.fp16
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient accumulation
        self.grad_accum_steps = config.training.gradient_accumulation_steps
        self.max_grad_norm = config.training.max_grad_norm

        # Gradient checkpointing
        if config.training.gradient_checkpointing:
            enable_gradient_checkpointing(model)

        # Logging
        self._wandb_run = None
        self._global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, seed: int = 42) -> dict[str, float]:
        """Run the full training loop.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            Dict of final metrics.
        """
        self._seed_everything(seed)
        self._init_wandb()

        final_metrics: dict[str, float] = {}

        for epoch in range(self.config.training.num_epochs):
            epoch_metrics = self._train_epoch(epoch)
            final_metrics.update(epoch_metrics)

            if self.eval_dataloader is not None:
                eval_metrics = self._evaluate()
                final_metrics.update(eval_metrics)
                self._log(eval_metrics)

            # Save checkpoint
            save_dir = f"outputs/{self.config.model.architecture}/epoch_{epoch}"
            self.save_checkpoint(epoch, save_dir)

        return final_metrics

    def save_checkpoint(self, epoch: int, save_dir: str) -> None:
        """Save LoRA weights, optimizer state, and training metadata.

        Args:
            epoch: Current epoch number.
            save_dir: Directory to save the checkpoint into.
        """
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(path / "lora_weights"))

        # Save optimizer and scheduler state
        torch.save(
            {
                "epoch": epoch,
                "global_step": self._global_step,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            path / "training_state.pt",
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        """Run a single training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Average loss components for this epoch.
        """
        self.model.train()
        accum_losses: dict[str, float] = {}
        num_steps = 0

        progress = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.training.num_epochs}",
        )

        for step, batch in enumerate(progress):
            loss_dict = self._train_step(batch)

            # Accumulate for epoch average
            for k, v in loss_dict.items():
                accum_losses[k] = accum_losses.get(k, 0.0) + v
            num_steps += 1

            # Gradient accumulation: step optimizer every N micro-steps
            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self._global_step += 1

                # Log
                self._log(loss_dict)

            progress.set_postfix(
                loss=f"{loss_dict.get('loss/total', 0):.4f}",
                eta=f"{loss_dict.get('metric/eta', 0):.4f}",
            )

        return {k: v / max(num_steps, 1) for k, v in accum_losses.items()}

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Execute a single training micro-step.

        1. Forward pass with hooks to capture hidden states
        2. Compute SFT loss from model output
        3. Compute RDSA losses using captured hidden states
        4. Backward pass (scaled for gradient accumulation)

        Args:
            batch: Dict with ``"input_ids"``, ``"attention_mask"``, and
                optionally ``"pixel_values"``, ``"is_harmful"``.

        Returns:
            Loss component dict for logging.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100

        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values

        with HookManager(
            self.model,
            self.rep_layers,
            layer_accessor=self.layer_accessor,
        ) as hm:
            with autocast(enabled=self.use_amp):
                output = self.model(**forward_kwargs)
                sft_loss = output.loss

                # Get hidden states from hooks
                raw_activations = hm.get_activations()

            # Aggregate to last-token: {group_idx: [B, d]}
            hidden_states = self._aggregate_activations(
                raw_activations, attention_mask
            )

            # Define safety loss function for LAT
            def safety_loss_fn(
                perturbed_states: dict[int, torch.Tensor],
            ) -> torch.Tensor:
                """Proxy safety loss: norm of perturbed safety projections.

                In Phase 1 we avoid a second forward pass by using the L2 norm
                of the perturbed safety projection as a proxy. The idea is that
                larger projections indicate stronger safety signals, so the LAT
                loss encourages robustness of those projections.
                """
                loss = torch.tensor(0.0, device=self.device)
                for group_idx, h_perturbed in perturbed_states.items():
                    V_s = self.V_s_list[group_idx]
                    proj = h_perturbed.float() @ V_s.float()  # [B, d_s]
                    # Maximize projection norm (negate for minimization)
                    loss = loss - proj.norm(dim=-1).mean()
                return loss / len(perturbed_states)

            with autocast(enabled=self.use_amp):
                total_loss, loss_dict = self.rdsa_loss(
                    sft_loss=sft_loss,
                    hidden_states=hidden_states,
                    V_s_list=self.V_s_list,
                    V_t_list=self.V_t_list,
                    safety_loss_fn=safety_loss_fn,
                )

        # Scale loss for gradient accumulation
        scaled_loss = total_loss / self.grad_accum_steps
        self.scaler.scale(scaled_loss).backward()

        return loss_dict

    def _aggregate_activations(
        self,
        raw_activations: dict[int, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Aggregate raw [B, seq_len, d] activations to [B, d] using last token.

        Args:
            raw_activations: ``{layer_idx: [B, seq_len, d]}``.
            attention_mask: ``[B, seq_len]``.

        Returns:
            ``{group_idx: [B, d]}`` with last-token activations.
        """
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(
            attention_mask.size(0), device=attention_mask.device
        )

        result: dict[int, torch.Tensor] = {}
        for group_idx, layer_idx in enumerate(self.rep_layers):
            act = raw_activations[layer_idx]  # [B, seq_len, d]
            result[group_idx] = act[batch_indices, seq_lengths]  # [B, d]

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        """Evaluate on the eval set: SFT loss, eta, and LCIV.

        Returns:
            Dict with evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            forward_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            output = self.model(**forward_kwargs)
            total_loss += output.loss.item()
            num_batches += 1

        # Compute current eta
        etas = [
            entanglement_degree(V_s, V_t).item()
            for V_s, V_t in zip(self.V_s_list, self.V_t_list, strict=True)
        ]
        mean_eta = sum(etas) / len(etas) if etas else 0.0

        self.model.train()
        return {
            "eval/sft_loss": total_loss / max(num_batches, 1),
            "eval/eta": mean_eta,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _seed_everything(self, seed: int) -> None:
        """Set all random seeds for reproducibility.

        Args:
            seed: The random seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _init_wandb(self) -> None:
        """Initialize wandb logging if available."""
        try:
            import wandb

            if wandb.run is None:
                wandb.init(
                    project="rdsa",
                    config={
                        "model": self.config.model.name,
                        "architecture": self.config.model.architecture,
                        "d_s": self.config.subspace.d_safe,
                        "d_t": self.config.subspace.d_semantic,
                        "alpha_entangle": self.config.training.alpha_entangle,
                        "alpha_consist": self.config.training.alpha_consist,
                        "alpha_lat_sub": self.config.training.alpha_lat_sub,
                        "lora_rank": self.config.training.lora_rank,
                        "lr": self.config.training.learning_rate,
                        "num_epochs": self.config.training.num_epochs,
                    },
                )
            self._wandb_run = wandb.run
        except ImportError:
            self._wandb_run = None

    def _log(self, metrics: dict[str, float]) -> None:
        """Log metrics to wandb if initialized.

        Args:
            metrics: Key-value metric pairs to log.
        """
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=self._global_step)
