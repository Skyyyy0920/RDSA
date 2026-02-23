"""Adaptive attacks designed to bypass RDSA defense.

Three attacks that assume full knowledge of the RDSA defense mechanism:
1. Adaptive-SCIA: SCIA + anti-entanglement objective
2. Adaptive-PGD: Multi-layer simultaneous PGD
3. Monitor-Evasion: PGD with cross-layer variance constraint

These are the most important attacks for evaluating RDSA's robustness
(Exp 6 in EXPERIMENT_DESIGN.md). Reviewers will scrutinize these results.

CRITICAL: Hook cleanup during optimization loops.
CRITICAL: Subspace projections in fp32.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rdsa.models.hooks import HookManager
from rdsa.models.model_utils import get_layer_accessor
from rdsa.subspace.identifier import SubspaceResult


class AdaptiveSCIA:
    """Adaptive SCIA with anti-entanglement objective.

    Extends SCIA by adding a loss term that attempts to suppress safety
    neurons along directions in V_s that are orthogonal to V_t (the
    manipulable subspace). When eta -> 1 the orthogonal complement
    shrinks to zero dimensions, making this attack ineffective.

    L_adaptive = L_SCIA + lambda_anti * L_anti_entangle

    Args:
        surrogate_model: The model used to craft adversarial examples.
        processor: HuggingFace processor for the surrogate.
        subspace_results: RDSA subspace results (attacker has full knowledge).
        architecture: Model architecture name.
        target_layers: Layer indices to target.
        attack_lr: Learning rate for PGD optimization.
        attack_steps: Number of PGD iterations.
        epsilon: L_inf perturbation budget in [0, 1] pixel range.
        lambda_anti: Weight for the anti-entanglement loss.
        device: Computation device.
    """

    def __init__(
        self,
        surrogate_model: nn.Module,
        processor: Any,
        subspace_results: list[SubspaceResult],
        architecture: str = "qwen3vl",
        target_layers: list[int] | None = None,
        attack_lr: float = 1 / 255,
        attack_steps: int = 200,
        epsilon: float = 16 / 255,
        lambda_anti: float = 0.5,
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
        self.lambda_anti = lambda_anti
        self.device = device

        # Attacker has full knowledge of RDSA subspaces
        self.V_s_list = [r.V_s.float().to(device) for r in subspace_results]
        self.V_t_list = [r.V_t.float().to(device) for r in subspace_results]
        self.rep_layers = [r.representative_layer for r in subspace_results]

        if target_layers is None:
            self.target_layers = list(self.rep_layers)
        else:
            self.target_layers = list(target_layers)

        # Precompute orthogonal complements: directions in V_s orthogonal to V_t
        self._ortho_projectors = self._compute_orthogonal_projectors()

        # Safety neuron masks (populated by identify_safety_neurons)
        self._safety_masks: dict[int, torch.Tensor] = {}

    def _compute_orthogonal_projectors(self) -> list[torch.Tensor]:
        """Compute projectors onto V_s^perp (V_s orthogonal to V_t).

        For each layer group, projects V_s onto the null space of V_t^T.
        When eta -> 1, this projector approaches zero.

        Returns:
            List of projectors [d_s, d_s], one per group.
        """
        projectors = []
        for V_s, V_t in zip(self.V_s_list, self.V_t_list, strict=True):
            # Project V_s columns onto V_t space, then subtract
            # P_Vt = V_t @ V_t^T (projector onto V_t column space)
            # V_s_parallel = P_Vt @ V_s
            # V_s_ortho = V_s - V_s_parallel
            P_vt = V_t @ V_t.T  # [d, d]
            V_s_parallel = P_vt @ V_s  # [d, d_s]
            V_s_ortho = V_s - V_s_parallel  # [d, d_s]

            # Projector in safety subspace coords: V_s_ortho^T @ V_s_ortho
            proj = V_s_ortho.T @ V_s_ortho  # [d_s, d_s]
            # Normalize
            norm = torch.linalg.norm(proj)
            if norm > 1e-8:
                proj = proj / norm
            projectors.append(proj)
        return projectors

    @torch.no_grad()
    def identify_safety_neurons(
        self,
        safe_dataloader: DataLoader,
        unsafe_dataloader: DataLoader,
        top_k_ratio: float = 0.01,
    ) -> dict[int, torch.Tensor]:
        """Identify safety-critical neurons (same as SCIA).

        Args:
            safe_dataloader: DataLoader of safe inputs.
            unsafe_dataloader: DataLoader of unsafe inputs.
            top_k_ratio: Fraction of neurons to mark as safety-critical.

        Returns:
            Dict mapping layer index to boolean mask [d].
        """
        self.model.eval()

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
                    act = raw[layer_idx]
                    h = act[batch_idx, seq_lengths]
                    batch_mean = h.mean(dim=0)

                    if layer_idx not in accum:
                        accum[layer_idx] = (batch_mean, 1)
                    else:
                        prev_sum, prev_count = accum[layer_idx]
                        accum[layer_idx] = (prev_sum + batch_mean, prev_count + 1)

            for layer_idx, (total, count) in accum.items():
                means_dict[layer_idx] = total / count

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
        """Generate adversarial image with anti-entanglement objective.

        L_adaptive = L_SCIA + lambda_anti * L_anti_entangle

        L_SCIA: minimize safety neuron activations.
        L_anti_entangle: maximize perturbation along V_s^perp directions
            (the manipulable subspace orthogonal to V_t).

        Args:
            image: Clean image tensor [C, H, W] in [0, 1] range.
            harmful_prompt: The harmful instruction.

        Returns:
            Adversarial image [C, H, W] in [0, 1] range.
        """
        if not self._safety_masks:
            raise RuntimeError(
                "Safety neuron masks not computed. "
                "Call identify_safety_neurons first."
            )

        self.model.eval()
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        enc = tokenizer(
            harmful_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        image = image.to(self.device).unsqueeze(0)
        delta = torch.zeros_like(image, requires_grad=True)

        for _step in range(self.attack_steps):
            adv_image = (image + delta).clamp(0, 1)

            forward_kwargs: dict[str, Any] = {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "pixel_values": adv_image,
            }

            with HookManager(
                self.model,
                self.target_layers,
                layer_accessor=self.layer_accessor,
            ) as hm:
                self.model(**forward_kwargs)
                raw = hm.get_activations()

            seq_lengths = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(1, device=self.device)

            # L_SCIA: suppress safety neurons
            loss_scia = torch.tensor(0.0, device=self.device, requires_grad=True)
            for layer_idx in self.target_layers:
                act = raw[layer_idx]
                h = act[batch_idx, seq_lengths]
                mask = self._safety_masks[layer_idx].to(self.device)
                loss_scia = loss_scia + h[:, mask].abs().mean()

            # L_anti_entangle: maximize perturbation along V_s^perp
            loss_anti = torch.tensor(0.0, device=self.device, requires_grad=True)
            for group_idx, layer_idx in enumerate(self.rep_layers):
                if layer_idx not in raw:
                    continue
                act = raw[layer_idx]
                h = act[batch_idx, seq_lengths].float()
                V_s = self.V_s_list[group_idx]
                proj_s = h @ V_s  # [1, d_s]
                # Project onto orthogonal complement
                ortho_proj = proj_s @ self._ortho_projectors[group_idx]  # [1, d_s]
                # Maximize magnitude along orthogonal directions (negate)
                loss_anti = loss_anti - ortho_proj.norm(dim=-1).mean()

            loss = loss_scia + self.lambda_anti * loss_anti
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.sign()
                delta.data = delta.data - self.attack_lr * grad
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = (image + delta.data).clamp(0, 1) - image

            if delta.grad is not None:
                delta.grad.zero_()

        return (image + delta.detach()).clamp(0, 1).squeeze(0)


class AdaptivePGD:
    """Adaptive PGD that simultaneously targets all layer groups.

    Standard PGD maximizes harmful output from a single model. Adaptive-PGD
    jointly optimizes across all G layer groups:

        delta* = argmin sum_k L_attack(f_k(x + delta, q))
        s.t. ||delta||_inf <= epsilon

    This is exactly the scenario analyzed in Theorem 1 — the attack cost
    scales as O(sqrt(sum(epsilon_k*^2))).

    Args:
        model: The target VLM (full white-box access).
        processor: HuggingFace processor.
        subspace_results: RDSA subspace results.
        epsilon: L_inf perturbation budget.
        attack_steps: Number of PGD iterations.
        step_size: Per-step perturbation magnitude.
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        subspace_results: list[SubspaceResult],
        architecture: str = "qwen3vl",
        epsilon: float = 16 / 255,
        attack_steps: int = 200,
        step_size: float = 1 / 255,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.model = model
        self.processor = processor
        self.architecture = architecture
        self.layer_accessor = get_layer_accessor(architecture)
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.step_size = step_size
        self.device = device

        # Attacker knowledge of RDSA
        self.V_s_list = [r.V_s.float().to(device) for r in subspace_results]
        self.rep_layers = [r.representative_layer for r in subspace_results]

    def attack(
        self,
        image: torch.Tensor,
        harmful_prompt: str,
        target_response: str | None = None,
    ) -> torch.Tensor:
        """Generate adversarial image via multi-layer PGD.

        Jointly optimizes to suppress safety projections across all
        layer groups while maximizing harmful output probability.

        Args:
            image: Clean image [C, H, W] in [0, 1] range.
            harmful_prompt: The harmful instruction.
            target_response: Optional target response to elicit.

        Returns:
            Adversarial image [C, H, W] in [0, 1] range.
        """
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        if target_response is not None:
            full_text = f"{harmful_prompt}\n{target_response}"
        else:
            full_text = f"{harmful_prompt}\nSure, here is"

        enc = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        labels = enc["input_ids"].clone()
        prompt_enc = tokenizer(
            harmful_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        prompt_len = prompt_enc["input_ids"].size(1)
        labels[:, :prompt_len] = -100

        image = image.to(self.device).unsqueeze(0)
        delta = torch.zeros_like(image, requires_grad=True)

        for _step in range(self.attack_steps):
            adv_image = (image + delta).clamp(0, 1)

            with HookManager(
                self.model,
                self.rep_layers,
                layer_accessor=self.layer_accessor,
            ) as hm:
                output = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    pixel_values=adv_image,
                    labels=labels,
                )
                raw = hm.get_activations()

            # Standard attack loss: maximize target response probability
            loss_attack = -output.loss

            # Multi-layer safety suppression: minimize safety projections
            seq_lengths = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(1, device=self.device)
            loss_suppress = torch.tensor(0.0, device=self.device, requires_grad=True)

            for group_idx, layer_idx in enumerate(self.rep_layers):
                act = raw[layer_idx]
                h = act[batch_idx, seq_lengths].float()
                V_s = self.V_s_list[group_idx]
                proj = h @ V_s  # [1, d_s]
                loss_suppress = loss_suppress + proj.norm(dim=-1).mean()

            # Joint objective
            loss = loss_attack + loss_suppress
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.sign()
                delta.data = delta.data + self.step_size * grad
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = (image + delta.data).clamp(0, 1) - image

            if delta.grad is not None:
                delta.grad.zero_()

        for p in self.model.parameters():
            p.requires_grad_(True)

        return (image + delta.detach()).clamp(0, 1).squeeze(0)


class MonitorEvasion:
    """Adaptive attack that evades the inference-time monitor.

    Adds a variance-matching constraint to PGD so that the adversarial
    input maintains low cross-layer safety confidence variance:

        L_evasion = L_attack + lambda_evade * max(Var_k[sigma(w_k^T V_s^(k)^T h_k)] - tau, 0)

    The attacker tries to suppress safety while keeping the cross-layer
    confidences consistent (low variance) to avoid triggering the monitor.

    Args:
        model: The target VLM (full white-box access).
        processor: HuggingFace processor.
        subspace_results: RDSA subspace results.
        safety_classifiers: The trained per-group safety classifiers.
        architecture: Model architecture name.
        epsilon: L_inf perturbation budget.
        attack_steps: Number of PGD iterations.
        step_size: Per-step perturbation magnitude.
        lambda_evade: Weight for the evasion constraint.
        tau: Monitor threshold (attacker knows this).
        device: Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        subspace_results: list[SubspaceResult],
        safety_classifiers: dict[int, nn.Linear] | None = None,
        architecture: str = "qwen3vl",
        epsilon: float = 16 / 255,
        attack_steps: int = 200,
        step_size: float = 1 / 255,
        lambda_evade: float = 1.0,
        tau: float = 0.2,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda")
        self.model = model
        self.processor = processor
        self.architecture = architecture
        self.layer_accessor = get_layer_accessor(architecture)
        self.epsilon = epsilon
        self.attack_steps = attack_steps
        self.step_size = step_size
        self.lambda_evade = lambda_evade
        self.tau = tau
        self.device = device

        self.V_s_list = [r.V_s.float().to(device) for r in subspace_results]
        self.rep_layers = [r.representative_layer for r in subspace_results]

        # Attacker has full knowledge of safety classifiers
        self.safety_classifiers = safety_classifiers
        if self.safety_classifiers is not None:
            for _k, clf in self.safety_classifiers.items():
                clf.to(device).eval()

    def attack(
        self,
        image: torch.Tensor,
        harmful_prompt: str,
        target_response: str | None = None,
    ) -> torch.Tensor:
        """Generate adversarial image with monitor evasion.

        Args:
            image: Clean image [C, H, W] in [0, 1] range.
            harmful_prompt: The harmful instruction.
            target_response: Optional target response to elicit.

        Returns:
            Adversarial image [C, H, W] in [0, 1] range.
        """
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        if target_response is not None:
            full_text = f"{harmful_prompt}\n{target_response}"
        else:
            full_text = f"{harmful_prompt}\nSure, here is"

        enc = tokenizer(
            full_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        labels = enc["input_ids"].clone()
        prompt_enc = tokenizer(
            harmful_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        prompt_len = prompt_enc["input_ids"].size(1)
        labels[:, :prompt_len] = -100

        image = image.to(self.device).unsqueeze(0)
        delta = torch.zeros_like(image, requires_grad=True)

        for _step in range(self.attack_steps):
            adv_image = (image + delta).clamp(0, 1)

            with HookManager(
                self.model,
                self.rep_layers,
                layer_accessor=self.layer_accessor,
            ) as hm:
                output = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    pixel_values=adv_image,
                    labels=labels,
                )
                raw = hm.get_activations()

            # L_attack: maximize target response probability
            loss_attack = -output.loss

            # Compute per-group safety confidences for variance constraint
            seq_lengths = enc["attention_mask"].sum(dim=1) - 1
            batch_idx = torch.arange(1, device=self.device)
            confidences: list[torch.Tensor] = []

            for group_idx, layer_idx in enumerate(self.rep_layers):
                act = raw[layer_idx]
                h = act[batch_idx, seq_lengths].float()
                V_s = self.V_s_list[group_idx]
                proj = h @ V_s  # [1, d_s]

                if (
                    self.safety_classifiers is not None
                    and group_idx in self.safety_classifiers
                ):
                    logit = self.safety_classifiers[group_idx](proj).squeeze(-1)
                    conf = torch.sigmoid(logit)
                else:
                    conf = torch.sigmoid(proj.norm(dim=-1))

                confidences.append(conf)

            stacked = torch.stack(confidences, dim=0)  # [G, 1]
            variance = torch.var(stacked, dim=0)  # [1]

            # Penalty: only penalize when variance exceeds threshold
            loss_evasion = torch.clamp(variance - self.tau, min=0.0).mean()

            loss = loss_attack + self.lambda_evade * loss_evasion
            loss.backward()

            with torch.no_grad():
                grad = delta.grad.sign()
                delta.data = delta.data + self.step_size * grad
                delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
                delta.data = (image + delta.data).clamp(0, 1) - image

            if delta.grad is not None:
                delta.grad.zero_()

        for p in self.model.parameters():
            p.requires_grad_(True)

        return (image + delta.detach()).clamp(0, 1).squeeze(0)
