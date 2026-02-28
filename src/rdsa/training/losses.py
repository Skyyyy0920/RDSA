"""RDSA training loss functions.

Implements the RDSA-specific losses:
- ConsistencyLoss: Enforce cross-layer safety assessment agreement
- SubspaceConstrainedATLoss: PGD adversarial training within safety subspace

CRITICAL: All subspace projections must be in fp32 even under AMP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SubspaceConstrainedATLoss(nn.Module):
    """Subspace-Constrained PGD Adversarial Training Loss.

    Core idea: find the strongest adversarial perturbation within the
    safety subspace V_s via PGD, then train the model to still refuse
    under that worst-case perturbation.

    .. math::
        \\delta^* = \\arg\\max_{\\|\\delta\\| \\leq \\varepsilon}
            \\mathcal{L}_{CE}(f(h + V_s \\delta), y_{refusal})

        \\mathcal{L}_{SA\\text{-}AT} = \\mathcal{L}_{CE}(
            f(h + V_s \\delta^*), y_{refusal})

    Key properties:

    - Search space is only ``d_s`` dimensional (e.g. 32), so PGD
      converges in a few steps.
    - Inner PGD: ``h`` is **detached** — only ``delta`` receives gradients.
    - Outer training: ``delta*`` is **detached** — gradients flow through
      ``h`` to model parameters.
    - Each layer group is perturbed independently in a single forward pass.

    Args:
        pgd_steps: Number of PGD steps in the inner loop.
        pgd_alpha: PGD step size.
        epsilon: L-infinity norm bound on the perturbation in subspace.
    """

    def __init__(
        self,
        pgd_steps: int = 7,
        pgd_alpha: float = 0.1,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.epsilon = epsilon

    def find_worst_perturbation(
        self,
        model: nn.Module,
        forward_kwargs: dict[str, torch.Tensor],
        h_clean: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        rep_layers: list[int],
        layer_accessor: str,
    ) -> dict[int, torch.Tensor]:
        """PGD inner loop: find worst-case perturbation in V_s subspace.

        Searches for delta that maximizes CE loss (i.e. reduces the
        model's ability to produce the correct refusal response).

        Args:
            model: The VLM.
            forward_kwargs: Standard model inputs (input_ids, attention_mask,
                labels, optional pixel_values).
            h_clean: ``{group_idx: [B, seq, d]}`` clean hidden states,
                will be detached internally.
            V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.
            rep_layers: Representative layer indices.
            layer_accessor: Dotted path to transformer layers.

        Returns:
            ``{group_idx: [B, seq, d_s]}`` optimal perturbation vectors.
        """
        from rdsa.models.hooks import InjectionHookManager

        h_detached = {gidx: h.detach() for gidx, h in h_clean.items()}
        result: dict[int, torch.Tensor] = {}

        # Run PGD independently per group.  Each group's perturbation is
        # found with a single-group injection hook, avoiding the problem
        # of later injections severing earlier groups' computation graphs.
        for gidx in sorted(h_clean.keys()):
            V_s = V_s_list[gidx].float()  # [d, d_s]
            h_det = h_detached[gidx]      # [B, seq, d]
            d_s = V_s.shape[1]

            delta = torch.zeros(
                h_det.shape[0], h_det.shape[1], d_s,
                device=h_det.device, dtype=torch.float32,
            )
            delta.requires_grad_(True)

            for _step in range(self.pgd_steps):
                # h_perturbed = h_detached + delta @ V_s.T
                h_perturbed = h_det + delta @ V_s.T.unsqueeze(0)

                with InjectionHookManager(
                    model, rep_layers,
                    {gidx: h_perturbed},
                    layer_accessor,
                ):
                    output = model(**forward_kwargs)

                # Maximize CE (gradient ascent on delta)
                loss_inner = output.loss
                (grad,) = torch.autograd.grad(
                    loss_inner, delta, create_graph=False
                )

                with torch.no_grad():
                    delta.data += self.pgd_alpha * grad.sign()
                    delta.data.clamp_(-self.epsilon, self.epsilon)

            result[gidx] = delta.detach()

        return result

    def compute_outer_loss(
        self,
        model: nn.Module,
        forward_kwargs: dict[str, torch.Tensor],
        h_clean: dict[int, torch.Tensor],
        delta_star: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        rep_layers: list[int],
        layer_accessor: str,
    ) -> torch.Tensor:
        """Outer training: compute refusal loss under worst-case perturbation.

        ``h_clean`` retains computation graph so gradients flow to model
        parameters.  ``delta_star`` is detached (constant).

        Args:
            model: The VLM.
            forward_kwargs: Standard model inputs.
            h_clean: ``{group_idx: [B, seq, d]}`` with live computation graph.
            delta_star: ``{group_idx: [B, seq, d_s]}`` optimal perturbations
                from PGD (detached).
            V_s_list: Safety subspace bases per group.
            rep_layers: Representative layer indices.
            layer_accessor: Dotted path to transformer layers.

        Returns:
            Scalar SA-AT loss (CE on perturbed hidden states).
        """
        from rdsa.models.hooks import AdditiveInjectionHookManager

        # Use ADDITIVE hooks: add V_s @ delta_star to the natural layer
        # output.  This preserves the computation graph through ALL layers
        # so gradients flow to every LoRA parameter, even when multiple
        # groups are perturbed simultaneously.
        with AdditiveInjectionHookManager(
            model, rep_layers, delta_star, V_s_list, layer_accessor
        ):
            output = model(**forward_kwargs)

        return output.loss
