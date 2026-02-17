"""SCIA (Safety Circuit Intervention Attack) reproduction.

SCIA is a transfer attack that identifies safety-critical neurons via
layer-wise probing, then crafts adversarial images that suppress safety
activations while preserving semantic content.

This is the primary attack that RDSA defends against. The reproduction
focuses on the core mechanism described in Algorithm 1 of the SCIA paper.

CRITICAL: Hook cleanup during attack optimization loops.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdsa.models.hooks import HookManager
from rdsa.models.model_utils import get_layer_accessor


class SCIAAttack:
    """Reproduction of the Safety Circuit Intervention Attack.

    SCIA identifies safety-critical neurons by comparing activations on
    safe vs unsafe inputs, then optimizes adversarial image perturbations
    that suppress those neurons' activations. The attack is crafted on
    a surrogate model and transfers to target models.

    Args:
        surrogate_model: The model used to craft adversarial examples.
        processor: HuggingFace processor for the surrogate.
        architecture: Model architecture name.
        target_layers: Layer indices to target. If ``None``, uses middle and
            deep layers.
        attack_lr: Learning rate for PGD optimization.
        attack_steps: Number of PGD iterations.
        epsilon: L_inf perturbation budget (in [0, 1] pixel range).
        device: Computation device.
    """

    def __init__(
        self,
        surrogate_model: nn.Module,
        processor: Any,
        architecture: str = "llava",
        target_layers: list[int] | None = None,
        attack_lr: float = 1 / 255,
        attack_steps: int = 100,
        epsilon: float = 16 / 255,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.model = surrogate_model
        self.processor = processor
        self.architecture = architecture
        self.layer_accessor = get_layer_accessor(architecture)
        self.attack_lr = attack_lr
        self.attack_steps = attack_steps
        self.epsilon = epsilon
        self.device = device

        if target_layers is None:
            # Default: target middle and deep layers (typical for safety circuits)
            self.target_layers = [12, 16, 20, 24, 28]
        else:
            self.target_layers = list(target_layers)

        # Safety neuron masks: populated by identify_safety_neurons
        self._safety_masks: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def identify_safety_neurons(
        self,
        safe_dataloader: DataLoader,
        unsafe_dataloader: DataLoader,
        top_k_ratio: float = 0.01,
    ) -> dict[int, torch.Tensor]:
        """Identify safety-critical neurons via activation difference probing.

        For each target layer, computes the mean absolute activation
        difference between safe and unsafe inputs. Neurons with the
        largest differences are identified as safety-critical.

        Args:
            safe_dataloader: DataLoader of safe inputs.
            unsafe_dataloader: DataLoader of unsafe inputs.
            top_k_ratio: Fraction of neurons to mark as safety-critical.

        Returns:
            Dict mapping layer index to boolean mask ``[d]`` where ``True``
            indicates a safety-critical neuron.
        """
        self.model.eval()

        # Accumulate activation means per layer
        safe_means: dict[int, torch.Tensor] = {}
        unsafe_means: dict[int, torch.Tensor] = {}

        for dataloader, means_dict in [
            (safe_dataloader, safe_means),
            (unsafe_dataloader, unsafe_means),
        ]:
            accum: dict[int, tuple[torch.Tensor, int]] = {}
            for batch in tqdm(dataloader, desc="Probing activations"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with HookManager(
                    self.model,
                    self.target_layers,
                    layer_accessor=self.layer_accessor,
                ) as hm:
                    self.model(input_ids=input_ids, attention_mask=attention_mask)
                    raw = hm.get_activations()

                seq_lengths = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(input_ids.size(0), device=self.device)

                for layer_idx in self.target_layers:
                    act = raw[layer_idx]  # [B, seq_len, d]
                    h = act[batch_idx, seq_lengths]  # [B, d]
                    batch_mean = h.mean(dim=0)  # [d]

                    if layer_idx not in accum:
                        accum[layer_idx] = (batch_mean, 1)
                    else:
                        prev_sum, prev_count = accum[layer_idx]
                        accum[layer_idx] = (prev_sum + batch_mean, prev_count + 1)

            for layer_idx, (total, count) in accum.items():
                means_dict[layer_idx] = total / count

        # Compute difference and identify top neurons
        masks: dict[int, torch.Tensor] = {}
        for layer_idx in self.target_layers:
            diff = (unsafe_means[layer_idx] - safe_means[layer_idx]).abs()
            k = max(1, int(diff.numel() * top_k_ratio))
            threshold = diff.topk(k).values[-1]
            masks[layer_idx] = diff >= threshold

        self._safety_masks = masks
        return masks

    def generate_adversarial_image(
        self,
        image: torch.Tensor,
        harmful_prompt: str,
    ) -> torch.Tensor:
        """Generate an adversarial image that suppresses safety activations.

        Uses PGD to optimize the image perturbation so that safety-critical
        neurons' activations are minimized.

        Args:
            image: Clean image tensor ``[C, H, W]`` in [0, 1] range.
            harmful_prompt: The harmful instruction to pair with the image.

        Returns:
            Adversarial image ``[C, H, W]`` in [0, 1] range.
        """
        if not self._safety_masks:
            raise RuntimeError(
                "Safety neuron masks not computed. "
                "Call identify_safety_neurons first."
            )

        self.model.eval()

        # Tokenize prompt
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        enc = tokenizer(
            harmful_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Initialize perturbation
        image = image.to(self.device).unsqueeze(0)  # [1, C, H, W]
        delta = torch.zeros_like(image, requires_grad=True)

        for _step in range(self.attack_steps):
            adv_image = (image + delta).clamp(0, 1)

            forward_kwargs: dict[str, Any] = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "pixel_values": adv_image,
            }

            # Compute loss: minimize safety neuron activations
            with HookManager(
                self.model,
                self.target_layers,
                layer_accessor=self.layer_accessor,
            ) as hm:
                self.model(**forward_kwargs)
                raw = hm.get_activations()

            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            seq_lengths = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(1, device=self.device)

            for layer_idx in self.target_layers:
                act = raw[layer_idx]  # [1, seq_len, d]
                h = act[batch_idx, seq_lengths]  # [1, d]
                mask = self._safety_masks[layer_idx].to(self.device)
                # Minimize magnitude of safety-critical neurons
                loss = loss + h[:, mask].abs().mean()

            loss.backward()

            with torch.no_grad():
                grad = delta.grad.sign()
                delta.data = delta.data - self.attack_lr * grad
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = (image + delta.data).clamp(0, 1) - image

            if delta.grad is not None:
                delta.grad.zero_()

        return (image + delta.detach()).clamp(0, 1).squeeze(0)  # [C, H, W]

    def attack_batch(
        self,
        images: torch.Tensor,
        harmful_prompts: list[str],
    ) -> torch.Tensor:
        """Generate adversarial images for a batch.

        Processes each sample independently (PGD is per-sample).

        Args:
            images: Batch of clean images ``[B, C, H, W]``.
            harmful_prompts: List of harmful instructions, length ``B``.

        Returns:
            Batch of adversarial images ``[B, C, H, W]``.
        """
        adv_images = []
        for i in range(images.size(0)):
            adv = self.generate_adversarial_image(images[i], harmful_prompts[i])
            adv_images.append(adv)
        return torch.stack(adv_images, dim=0)
