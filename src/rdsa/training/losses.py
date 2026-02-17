"""RDSA training loss functions.

Implements the three RDSA-specific losses as nn.Module subclasses:
- EntanglementLoss: Maximize safety-semantic subspace alignment
- ConsistencyLoss: Enforce cross-layer safety assessment agreement
- SubspaceLATLoss: Subspace-aware Latent Adversarial Training
- RDSALoss: Composite loss combining all components

CRITICAL: EntanglementLoss uses torch.max (differentiable), NOT argmax.
CRITICAL: All subspace projections must be in fp32 even under AMP.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdsa.config import TrainingConfig
from rdsa.subspace.metrics import entanglement_degree


class EntanglementLoss(nn.Module):
    """Maximize safety-semantic subspace alignment (Theorem 2).

    .. math::
        L_{entangle} = -\\frac{1}{|G|} \\sum_k \\frac{1}{d_s}
            \\sum_i \\max_j |(v_s^{(k,i)})^T v_t^{(k,j)}|

    Minimizing this loss maximizes the entanglement degree ``eta``.

    CRITICAL: Uses ``torch.max`` (supports autograd) not ``argmax``.
    """

    def forward(
        self,
        V_s_list: list[torch.Tensor],
        V_t_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute entanglement loss across all layer groups.

        Args:
            V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.
            V_t_list: Semantic subspace bases, one ``[d, d_t]`` per group.

        Returns:
            Scalar loss (negative entanglement, to be minimized).
        """
        group_etas = []

        for V_s, V_t in zip(V_s_list, V_t_list, strict=True):
            V_s = V_s.float()
            V_t = V_t.float()

            # Cross-correlation: [d_s, d_t]
            cross = V_s.T @ V_t

            # torch.max returns (values, indices) — .values is differentiable
            max_alignments = torch.max(cross.abs(), dim=1).values  # [d_s]
            group_etas.append(max_alignments.mean())

        # Negative mean eta across groups
        eta = torch.stack(group_etas).mean()
        return -eta


class ConsistencyLoss(nn.Module):
    """Enforce cross-layer safety assessment agreement.

    .. math::
        L_{consist} = \\sum_{(k_1, k_2)} \\left(1 - \\cos(
            V_s^{(k_1)T} h_{l_{k_1}}, V_s^{(k_2)T} h_{l_{k_2}})\\right)

    All layer groups should agree on whether an input is safe or unsafe.

    CRITICAL: Hidden states projected to fp32 before subspace projection.
    """

    def forward(
        self,
        hidden_states: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute consistency loss across all pairs of layer groups.

        Args:
            hidden_states: ``{group_idx: [B, d]}`` hidden states from
                representative layers.
            V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.

        Returns:
            Scalar consistency loss.
        """
        # Project each group's hidden states into safety subspace
        projections: list[torch.Tensor] = []
        group_indices = sorted(hidden_states.keys())

        for group_idx in group_indices:
            h = hidden_states[group_idx].float()  # [B, d] — fp32
            V_s = V_s_list[group_idx].float()  # [d, d_s]
            proj = h @ V_s  # [B, d_s]
            projections.append(proj)

        # Compute pairwise cosine distance: 1 - cos(proj_i, proj_j)
        loss = torch.tensor(0.0, device=projections[0].device)
        num_pairs = 0

        for i in range(len(projections)):
            for j in range(i + 1, len(projections)):
                cos_sim = F.cosine_similarity(
                    projections[i], projections[j], dim=-1
                )  # [B]
                loss = loss + (1.0 - cos_sim).mean()
                num_pairs += 1

        if num_pairs > 0:
            loss = loss / num_pairs

        return loss


class SubspaceLATLoss(nn.Module):
    """Subspace-aware Latent Adversarial Training.

    .. math::
        h_{adv} = h + V_s^{(k)} \\cdot \\epsilon,
        \\quad \\epsilon \\sim \\text{Uniform}(-\\alpha, \\alpha)

        L_{LAT-sub} = \\mathbb{E}_\\epsilon[L_{safety}(h_{adv})]

    Perturbs hidden states within the safety subspace and trains the model
    to still produce safe outputs. This hardens the safety representations
    against adversarial manipulation.

    Args:
        perturbation_alpha: Magnitude of uniform perturbation.
        num_perturbation_samples: Monte Carlo samples for the expectation.
    """

    def __init__(
        self,
        perturbation_alpha: float = 0.1,
        num_perturbation_samples: int = 1,
    ) -> None:
        super().__init__()
        self.perturbation_alpha = perturbation_alpha
        self.num_perturbation_samples = num_perturbation_samples

    def _perturb_in_subspace(
        self,
        h: torch.Tensor,
        V_s: torch.Tensor,
        alpha: float | None = None,
    ) -> torch.Tensor:
        """Apply random perturbation within the safety subspace.

        Args:
            h: Hidden states ``[B, d]``.
            V_s: Safety subspace basis ``[d, d_s]``.
            alpha: Perturbation magnitude. Uses ``self.perturbation_alpha``
                if ``None``.

        Returns:
            Perturbed hidden states ``[B, d]``.
        """
        if alpha is None:
            alpha = self.perturbation_alpha

        V_s = V_s.float()
        d_s = V_s.shape[1]

        # epsilon ~ Uniform(-alpha, alpha), shape [B, d_s]
        epsilon = (
            torch.rand(h.shape[0], d_s, device=h.device, dtype=torch.float32)
            * 2 * alpha
            - alpha
        )

        # Perturbation in full space: V_s @ epsilon^T -> [d, B] -> transpose
        perturbation = (V_s @ epsilon.T).T  # [B, d]

        return h.float() + perturbation

    def forward(
        self,
        hidden_states: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        safety_loss_fn: Callable[[dict[int, torch.Tensor]], torch.Tensor],
    ) -> torch.Tensor:
        """Compute subspace-aware LAT loss.

        For each Monte Carlo sample, perturbs hidden states within V_s for
        each group, then evaluates the safety loss on perturbed states.

        Args:
            hidden_states: ``{group_idx: [B, d]}`` hidden states.
            V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.
            safety_loss_fn: Callable that takes perturbed hidden states dict
                and returns a scalar safety loss.

        Returns:
            Scalar LAT loss averaged over groups and samples.
        """
        total_loss = torch.tensor(0.0, device=next(iter(hidden_states.values())).device)

        for _sample in range(self.num_perturbation_samples):
            perturbed: dict[int, torch.Tensor] = {}
            for group_idx in hidden_states:
                h = hidden_states[group_idx]
                V_s = V_s_list[group_idx]
                perturbed[group_idx] = self._perturb_in_subspace(h, V_s)

            total_loss = total_loss + safety_loss_fn(perturbed)

        return total_loss / self.num_perturbation_samples


class RDSALoss(nn.Module):
    """Composite RDSA loss combining SFT with all three RDSA losses.

    .. math::
        L_{total} = L_{SFT} + \\alpha_1 \\cdot L_{entangle}
            + \\alpha_2 \\cdot L_{consist}
            + \\alpha_3 \\cdot L_{LAT-sub}

    Logs individual components for wandb tracking.

    Args:
        config: Training configuration with loss weights.
    """

    def __init__(self, config: TrainingConfig) -> None:
        super().__init__()
        self.alpha_entangle = config.alpha_entangle
        self.alpha_consist = config.alpha_consist
        self.alpha_lat_sub = config.alpha_lat_sub

        self.entanglement_loss = EntanglementLoss()
        self.consistency_loss = ConsistencyLoss()
        self.lat_loss = SubspaceLATLoss(
            perturbation_alpha=config.lat_perturbation_alpha,
        )

    def forward(
        self,
        sft_loss: torch.Tensor,
        hidden_states: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        V_t_list: list[torch.Tensor],
        safety_loss_fn: Callable[[dict[int, torch.Tensor]], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total RDSA loss and return individual components for logging.

        Args:
            sft_loss: Standard SFT (cross-entropy) loss.
            hidden_states: ``{group_idx: [B, d]}`` from representative layers.
            V_s_list: Safety subspace bases per group.
            V_t_list: Semantic subspace bases per group.
            safety_loss_fn: Callback for SubspaceLATLoss.

        Returns:
            ``(total_loss, loss_dict)`` where ``loss_dict`` contains:
            ``{"loss/total", "loss/sft", "loss/entangle", "loss/consist",
            "loss/lat_sub", "metric/eta"}``.
        """
        l_entangle = self.entanglement_loss(V_s_list, V_t_list)
        l_consist = self.consistency_loss(hidden_states, V_s_list)
        l_lat_sub = self.lat_loss(hidden_states, V_s_list, safety_loss_fn)

        total = (
            sft_loss
            + self.alpha_entangle * l_entangle
            + self.alpha_consist * l_consist
            + self.alpha_lat_sub * l_lat_sub
        )

        # Compute eta for monitoring (detached — not part of loss graph)
        with torch.no_grad():
            etas = [
                entanglement_degree(V_s.float(), V_t.float())
                for V_s, V_t in zip(V_s_list, V_t_list, strict=True)
            ]
            mean_eta = torch.stack(etas).mean().item()

        loss_dict = {
            "loss/total": total.item(),
            "loss/sft": sft_loss.item(),
            "loss/entangle": l_entangle.item(),
            "loss/consist": l_consist.item(),
            "loss/lat_sub": l_lat_sub.item(),
            "metric/eta": mean_eta,
        }

        return total, loss_dict
