"""RDSA multi-objective training loop with Subspace-Constrained AT.

Training pipeline:
1. Clean forward pass with hooks to capture hidden states
2. PGD inner loop: find worst-case perturbation delta* in V_s subspace
3. Outer training: forward with h + V_s @ delta*, compute refusal CE loss
4. Compute cross-layer consistency loss on clean hidden states
5. Combined backward: L_SFT + alpha_sa_at * L_SA-AT + alpha_consist * L_consist

CRITICAL: Hook cleanup via context manager in every training step.
CRITICAL: Subspace projections in fp32 even under AMP.
CRITICAL: PGD inner loop uses torch.autograd.grad (not .backward) to
          avoid accumulating gradients on model parameters.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
from rdsa.training.losses import ConsistencyLoss, SubspaceConstrainedATLoss

logger = logging.getLogger(__name__)


class RDSATrainer:
    """Multi-objective training loop for RDSA with SA-AT.

    Training objective:
        L_total = L_SFT + alpha_sa_at * L_SA-AT + alpha_consist * L_consist

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

        # Losses (no learnable parameters)
        self.sa_at_loss = SubspaceConstrainedATLoss(
            pgd_steps=config.training.sa_at_pgd_steps,
            pgd_alpha=config.training.sa_at_pgd_alpha,
            epsilon=config.training.sa_at_epsilon,
        )
        self.consist_loss = ConsistencyLoss()
        self.alpha_sa_at = config.training.alpha_sa_at
        self.alpha_consist = config.training.alpha_consist

        # Optimizer — LoRA params only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler — linear warmup then linear decay to 10% of peak lr
        total_steps = (
            len(train_dataloader)
            * config.training.num_epochs
            // config.training.gradient_accumulation_steps
        )
        warmup_steps = max(
            int(total_steps * config.training.warmup_ratio), 1
        )

        def _lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / warmup_steps
            progress = (current_step - warmup_steps) / max(
                total_steps - warmup_steps, 1
            )
            return max(0.1, 1.0 - 0.9 * progress)

        self.scheduler = LambdaLR(self.optimizer, _lr_lambda)
        logger.info(
            "Scheduler: linear warmup %d steps -> linear decay over %d "
            "total optimizer steps (floor 10%% lr)",
            warmup_steps, total_steps,
        )

        # AMP — bf16 uses autocast (no GradScaler); fp16 uses both.
        self.use_amp = config.training.fp16 or config.training.bf16
        self.amp_dtype = (
            torch.float16 if config.training.fp16 else torch.bfloat16
        )
        self.scaler = GradScaler(enabled=config.training.fp16)

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

    def train(
        self,
        seed: int = 42,
        re_identify_fn: Any | None = None,
    ) -> dict[str, float]:
        """Run the full training loop.

        Args:
            seed: Random seed for reproducibility.
            re_identify_fn: Optional callable ``(model, epoch) -> list[SubspaceResult]``.
                Called at the start of each epoch (after epoch 0) to refresh
                V_s/V_t subspace bases from the model's updated activations.

        Returns:
            Dict of final metrics.
        """
        self._seed_everything(seed)
        self._init_wandb()

        # --- Diagnostic: check optimizer has LoRA params ---
        n_trainable = sum(
            p.numel()
            for pg in self.optimizer.param_groups
            for p in pg["params"]
        )
        logger.info("Trainable parameters in optimizer: %d", n_trainable)
        if n_trainable == 0:
            raise RuntimeError(
                "Optimizer has 0 trainable parameters — LoRA not connected?"
            )

        # Ensure clean gradient state before training begins
        self.optimizer.zero_grad()

        # Log initial lr for verification
        init_lr = self.optimizer.param_groups[0]["lr"]
        logger.info("Initial learning rate: %.2e", init_lr)

        final_metrics: dict[str, float] = {}

        for epoch in range(self.config.training.num_epochs):
            # Refresh V_s / V_t from current model activations (skip epoch 0
            # which uses the pre-computed subspaces passed at init time).
            if epoch > 0 and re_identify_fn is not None:
                logger.info(
                    "Epoch %d: re-identifying subspaces from current model...",
                    epoch,
                )
                new_results = re_identify_fn(self.model, epoch)
                self.update_subspaces(new_results)

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

    def update_subspaces(
        self, subspace_results: list[SubspaceResult]
    ) -> None:
        """Update V_s and V_t from freshly identified subspaces.

        Args:
            subspace_results: Newly identified subspace results.
        """
        self.V_s_list = [
            r.V_s.float().to(self.device) for r in subspace_results
        ]
        self.V_t_list = [
            r.V_t.float().to(self.device) for r in subspace_results
        ]
        mean_eta = sum(
            entanglement_degree(Vs, Vt).item()
            for Vs, Vt in zip(self.V_s_list, self.V_t_list, strict=True)
        ) / max(len(self.V_s_list), 1)
        logger.info("Updated subspaces. New eta: %.4f", mean_eta)

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

        n_batches = len(self.train_dataloader)
        logger.info(
            "Epoch %d/%d START — %d batches, grad_accum=%d",
            epoch + 1, self.config.training.num_epochs,
            n_batches, self.grad_accum_steps,
        )

        progress = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch + 1}/{self.config.training.num_epochs}",
        )

        for step, batch in enumerate(progress):
            loss_dict = self._train_step(batch)

            # --- Diagnostic: gradient check on first step of first epoch ---
            if epoch == 0 and step == 0:
                grad_norms: list[tuple[str, float]] = []
                for name, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_norms.append((name, p.grad.norm().item()))
                if grad_norms:
                    norms = [gn for _, gn in grad_norms]
                    logger.info(
                        "Gradient check: %d params with grad, "
                        "max=%.6f, mean=%.6f, min=%.6f",
                        len(grad_norms),
                        max(norms),
                        sum(norms) / len(norms),
                        min(norms),
                    )
                    grad_norms.sort(key=lambda x: x[1], reverse=True)
                    for name, gn in grad_norms[:5]:
                        logger.info("  top grad: %s = %.6f", name, gn)
                else:
                    logger.warning(
                        "No gradients found after first backward! "
                        "Check loss computation graph."
                    )

            # --- Per-step diagnostic every 10 micro-steps ---
            if step % 10 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d step %d: lr=%.2e, total=%.4f, sft=%.4f, "
                    "sa_at=%.4f, consist=%.4f",
                    epoch, step, lr,
                    loss_dict.get("loss/total", 0),
                    loss_dict.get("loss/sft", 0),
                    loss_dict.get("loss/sa_at", 0),
                    loss_dict.get("loss/consist", 0),
                )

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

        # Handle leftover gradients when batch count is not divisible
        # by grad_accum_steps.
        if num_steps % self.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self._global_step += 1

        avg_losses = {k: v / max(num_steps, 1) for k, v in accum_losses.items()}
        logger.info(
            "Epoch %d/%d END — %d micro-steps, global_step=%d, "
            "avg_total=%.4f, avg_sft=%.4f, avg_sa_at=%.4f",
            epoch + 1, self.config.training.num_epochs,
            num_steps, self._global_step,
            avg_losses.get("loss/total", 0),
            avg_losses.get("loss/sft", 0),
            avg_losses.get("loss/sa_at", 0),
        )
        return avg_losses

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Execute a single training micro-step with SA-AT.

        **Phase 1** — Clean forward: captures hidden states at representative
        layers (``detach=False`` to allow backprop for consistency loss).
        Computes SFT loss.

        **Phase 2** — PGD inner loop (no model gradients): finds worst-case
        perturbation ``delta*`` within V_s subspace.

        **Phase 3** — Outer adversarial training: forward with
        ``h + V_s @ delta*`` (h retains graph), computes refusal CE loss.

        **Phase 4** — Consistency loss on clean hidden states.

        **Phase 5** — Combined backward:
        ``L_SFT + alpha_sa_at * L_SA-AT + alpha_consist * L_consist``

        Args:
            batch: Dict with ``"input_ids"``, ``"attention_mask"``, and
                optionally ``"pixel_values"``.

        Returns:
            Loss component dict for logging.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            forward_kwargs["pixel_values"] = pixel_values

        # ============================================================
        # Phase 1: Clean forward → SFT loss + capture hidden states
        # ============================================================
        # detach=False: hidden states retain computation graph for
        # consistency loss and SA-AT outer loop.
        with HookManager(
            self.model,
            self.rep_layers,
            layer_accessor=self.layer_accessor,
            detach=False,
        ) as hm:
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                output = self.model(**forward_kwargs)
                sft_loss = output.loss
                # {layer_idx: [B, seq, d]} with live graph
                raw_activations = hm.get_activations()

        # Map to group indices: {group_idx: [B, seq, d]}
        h_clean: dict[int, torch.Tensor] = {}
        for gidx, layer_idx in enumerate(self.rep_layers):
            h_clean[gidx] = raw_activations[layer_idx]
        del raw_activations

        # ============================================================
        # Phase 2: PGD inner loop — find delta* (no model gradients)
        # ============================================================
        delta_star = self.sa_at_loss.find_worst_perturbation(
            model=self.model,
            forward_kwargs=forward_kwargs,
            h_clean=h_clean,
            V_s_list=self.V_s_list,
            rep_layers=self.rep_layers,
            layer_accessor=self.layer_accessor,
        )

        # ============================================================
        # Phase 3: Outer SA-AT loss (gradients flow to model)
        # ============================================================
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            sa_at_loss = self.sa_at_loss.compute_outer_loss(
                model=self.model,
                forward_kwargs=forward_kwargs,
                h_clean=h_clean,
                delta_star=delta_star,
                V_s_list=self.V_s_list,
                rep_layers=self.rep_layers,
                layer_accessor=self.layer_accessor,
            )

        # ============================================================
        # Phase 4: Consistency loss on clean hidden states
        # ============================================================
        # Aggregate to last-token: {group_idx: [B, d]}
        h_aggregated = self._aggregate_activations(h_clean, attention_mask)

        with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            l_consist = self.consist_loss(h_aggregated, self.V_s_list)

        # ============================================================
        # Phase 5: Combined backward
        # ============================================================
        total_loss = (
            sft_loss
            + self.alpha_sa_at * sa_at_loss
            + self.alpha_consist * l_consist
        )

        self.scaler.scale(total_loss / self.grad_accum_steps).backward()

        del h_clean, h_aggregated, delta_star

        # ============================================================
        # Logging (all detached — no graph impact)
        # ============================================================
        with torch.no_grad():
            etas = [
                entanglement_degree(V_s.float(), V_t.float())
                for V_s, V_t in zip(
                    self.V_s_list, self.V_t_list, strict=True
                )
            ]
            mean_eta = torch.stack(etas).mean().item()

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/sft": sft_loss.item(),
            "loss/sa_at": sa_at_loss.item(),
            "loss/consist": l_consist.item(),
            "metric/eta": mean_eta,
        }
        return loss_dict

    def _aggregate_activations(
        self,
        h_clean: dict[int, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        """Aggregate [B, seq_len, d] hidden states to [B, d] using last token.

        Args:
            h_clean: ``{group_idx: [B, seq_len, d]}``.
            attention_mask: ``[B, seq_len]``.

        Returns:
            ``{group_idx: [B, d]}`` with last-token activations.
        """
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        batch_indices = torch.arange(
            attention_mask.size(0), device=attention_mask.device
        )

        result: dict[int, torch.Tensor] = {}
        for group_idx, h in h_clean.items():
            result[group_idx] = h[batch_indices, seq_lengths]  # [B, d]

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        """Evaluate on the eval set: SFT loss and eta.

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
        """Set all random seeds for reproducibility."""
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
                        "alpha_sa_at": self.config.training.alpha_sa_at,
                        "alpha_consist": self.config.training.alpha_consist,
                        "sa_at_pgd_steps": self.config.training.sa_at_pgd_steps,
                        "sa_at_epsilon": self.config.training.sa_at_epsilon,
                        "lora_rank": self.config.training.lora_rank,
                        "lr": self.config.training.learning_rate,
                        "num_epochs": self.config.training.num_epochs,
                    },
                )
            self._wandb_run = wandb.run
        except ImportError:
            self._wandb_run = None

    def _log(self, metrics: dict[str, float]) -> None:
        """Log metrics to wandb if initialized."""
        if self._wandb_run is not None:
            import wandb

            wandb.log(metrics, step=self._global_step)
