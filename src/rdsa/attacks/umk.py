"""UMK (Universal Model-Knowledge) white-box attack.

Standard PGD attack with full access to model gradients. Optimizes
adversarial image perturbations to maximize the probability of generating
a harmful response (or minimize the probability of refusal).

CRITICAL: Hook cleanup is handled implicitly (no hooks needed — direct
gradient computation through the model).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class UMKAttack:
    """Universal Model-Knowledge white-box PGD attack.

    Optimizes an adversarial image perturbation to elicit harmful responses
    from the model. Uses the standard PGD framework with L_inf constraint.

    Args:
        model: The target VLM (full white-box access).
        processor: HuggingFace processor.
        epsilon: L_inf perturbation budget in [0, 1] pixel range.
        attack_steps: Number of PGD iterations.
        step_size: Per-step perturbation magnitude.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        epsilon: float = 16 / 255,
        attack_steps: int = 100,
        step_size: float = 1 / 255,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.model = model
        self.processor = processor
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.step_size = step_size
        self.device = device

    def attack(
        self,
        image: torch.Tensor,
        harmful_prompt: str,
        target_response: str | None = None,
    ) -> torch.Tensor:
        """Generate an adversarial image via PGD.

        When ``target_response`` is provided, maximizes the log-probability
        of generating that response. Otherwise, minimizes the log-probability
        of common refusal tokens.

        Args:
            image: Clean image ``[C, H, W]`` in [0, 1] range.
            harmful_prompt: The harmful instruction.
            target_response: Optional target response to elicit.

        Returns:
            Adversarial image ``[C, H, W]`` in [0, 1] range.
        """
        self.model.eval()
        # Enable gradients for the image perturbation only
        for p in self.model.parameters():
            p.requires_grad_(False)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # Prepare target labels
        if target_response is not None:
            full_text = f"{harmful_prompt}\n{target_response}"
        else:
            # Default: use a generic compliant prefix
            full_text = f"{harmful_prompt}\nSure, here is"

        enc = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        labels = enc["input_ids"].clone()
        # Only compute loss on the response portion
        prompt_enc = tokenizer(
            harmful_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        prompt_len = prompt_enc["input_ids"].size(1)
        labels[:, :prompt_len] = -100

        # Initialize perturbation
        image = image.to(self.device).unsqueeze(0)  # [1, C, H, W]
        delta = torch.zeros_like(image, requires_grad=True)

        for _step in range(self.attack_steps):
            adv_image = (image + delta).clamp(0, 1)

            output = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=adv_image,
                labels=labels,
            )

            # Minimize the cross-entropy loss (= maximize target probability)
            loss = -output.loss
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.sign()
                delta.data = delta.data + self.step_size * grad
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = (image + delta.data).clamp(0, 1) - image

            if delta.grad is not None:
                delta.grad.zero_()

        # Restore model gradients
        for p in self.model.parameters():
            p.requires_grad_(True)

        return (image + delta.detach()).clamp(0, 1).squeeze(0)  # [C, H, W]

    def attack_batch(
        self,
        images: torch.Tensor,
        harmful_prompts: list[str],
        target_responses: list[str] | None = None,
    ) -> torch.Tensor:
        """Generate adversarial images for a batch.

        Args:
            images: Batch of clean images ``[B, C, H, W]``.
            harmful_prompts: List of harmful instructions, length ``B``.
            target_responses: Optional list of target responses, length ``B``.

        Returns:
            Batch of adversarial images ``[B, C, H, W]``.
        """
        adv_images = []
        for i in range(images.size(0)):
            target = target_responses[i] if target_responses else None
            adv = self.attack(images[i], harmful_prompts[i], target)
            adv_images.append(adv)
        return torch.stack(adv_images, dim=0)
