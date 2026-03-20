"""RDSA training loss functions.

Implements the RDSA-specific losses:
- ConsistencyLoss: Enforce cross-layer safety assessment agreement
- SubspaceConstrainedATLoss: PGD adversarial training within safety subspace
- EntanglementLoss: Push safety subspace to overlap with semantic subspace

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

    When ``harmful_only=True``, the loss is only computed on samples marked
    as harmful in the batch, reducing over-refusal on benign inputs.

    CRITICAL: Hidden states projected to fp32 before subspace projection.
    """

    def __init__(self, harmful_only: bool = True) -> None:
        super().__init__()
        self.harmful_only = harmful_only

    def forward(
        self,
        hidden_states: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        is_harmful: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute consistency loss across all pairs of layer groups.

        Args:
            hidden_states: ``{group_idx: [B, d]}`` hidden states from
                representative layers.
            V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.
            is_harmful: Optional ``[B]`` boolean mask. When ``harmful_only``
                is True and this is provided, only harmful samples contribute.

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

        # Filter to harmful samples if requested
        if self.harmful_only and is_harmful is not None:
            mask = is_harmful.bool()
            if not mask.any():
                return torch.tensor(0.0, device=projections[0].device)
            projections = [p[mask] for p in projections]

        # Compute pairwise cosine distance: 1 - cos(proj_i, proj_j)
        loss = torch.tensor(0.0, device=projections[0].device)
        num_pairs = 0

        for i in range(len(projections)):
            for j in range(i + 1, len(projections)):
                cos_sim = F.cosine_similarity(
                    projections[i], projections[j], dim=-1
                )  # [B'] or [B]
                loss = loss + (1.0 - cos_sim).mean()
                num_pairs += 1

        if num_pairs > 0:
            loss = loss / num_pairs

        return loss


class EntanglementLoss(nn.Module):
    """Maximize entanglement between safety and semantic subspaces.

    Pushes the safety subspace V_s to overlap with the semantic subspace V_t,
    making it impossible to remove safety features without destroying semantics.

    .. math::
        L_{entangle} = 1 - \\eta(V_s, V_t)
            = 1 - \\frac{1}{d_s} \\sum_i \\max_j |v_s^{(i)T} v_t^{(j)}|

    This loss directly optimizes the entanglement degree eta that was previously
    only monitored. The loss is computed on the LoRA-modified hidden states
    rather than the static subspace bases, so gradients flow to LoRA parameters.

    Two modes:
    - ``mode="subspace"``: Directly optimize eta on V_s, V_t (static, no
      gradient to model — useful as regularizer on re-identified subspaces).
    - ``mode="activation"``: Maximize alignment of safety-projected and
      semantic-projected activations (gradient flows to model).
    """

    def __init__(self, mode: str = "activation") -> None:
        super().__init__()
        if mode not in ("subspace", "activation"):
            raise ValueError(f"mode must be 'subspace' or 'activation', got {mode!r}")
        self.mode = mode

    def forward(
        self,
        hidden_states: dict[int, torch.Tensor] | None = None,
        V_s_list: list[torch.Tensor] | None = None,
        V_t_list: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Compute entanglement loss.

        Args:
            hidden_states: ``{group_idx: [B, d]}`` hidden states (needed for
                activation mode).
            V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.
            V_t_list: Semantic subspace bases, one ``[d, d_t]`` per group.

        Returns:
            Scalar loss in [0, 1]. Lower means more entangled.
        """
        if V_s_list is None or V_t_list is None:
            raise ValueError("V_s_list and V_t_list are required")

        if self.mode == "subspace":
            return self._subspace_loss(V_s_list, V_t_list)
        else:
            if hidden_states is None:
                raise ValueError("hidden_states required for activation mode")
            return self._activation_loss(hidden_states, V_s_list, V_t_list)

    def _subspace_loss(
        self,
        V_s_list: list[torch.Tensor],
        V_t_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """1 - eta computed directly on subspace bases."""
        total_eta = torch.tensor(0.0, device=V_s_list[0].device)
        for V_s, V_t in zip(V_s_list, V_t_list, strict=True):
            V_s = V_s.float()
            V_t = V_t.float()
            cross = V_s.T @ V_t  # [d_s, d_t]
            max_alignments = torch.max(cross.abs(), dim=1).values  # [d_s]
            total_eta = total_eta + max_alignments.mean()
        total_eta = total_eta / len(V_s_list)
        return 1.0 - total_eta

    def _activation_loss(
        self,
        hidden_states: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        V_t_list: list[torch.Tensor],
    ) -> torch.Tensor:
        """Maximize alignment between safety-projected and semantic-projected
        activations.  Gradients flow through hidden states to LoRA parameters.

        For each group, project h into both subspaces and maximize the
        correlation between projections (if safety and semantic projections
        are correlated, removing safety features damages semantics).
        """
        total_loss = torch.tensor(0.0, device=V_s_list[0].device)
        group_indices = sorted(hidden_states.keys())

        for gidx in group_indices:
            h = hidden_states[gidx].float()  # [B, d]
            V_s = V_s_list[gidx].float()  # [d, d_s]
            V_t = V_t_list[gidx].float()  # [d, d_t]

            # Project to safety and semantic subspaces
            h_s = h @ V_s  # [B, d_s]
            h_t = h @ V_t  # [B, d_t]

            # Normalize projections
            h_s_norm = F.normalize(h_s, dim=-1)  # [B, d_s]
            h_t_norm = F.normalize(h_t, dim=-1)  # [B, d_t]

            # Cross-correlation: want each safety dimension to be correlated
            # with at least one semantic dimension
            cross = h_s_norm.T @ h_t_norm / h.size(0)  # [d_s, d_t]
            max_corr = torch.max(cross.abs(), dim=1).values  # [d_s]

            total_loss = total_loss + (1.0 - max_corr.mean())

        return total_loss / max(len(group_indices), 1)


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

    Key improvements over naive PGD:

    - **Random restarts**: Runs PGD from ``num_restarts`` random
      initializations and keeps the perturbation with the highest loss.
      This finds stronger adversaries than single-start PGD from zero.
    - **Relative epsilon**: When enabled, scales epsilon by the mean
      activation norm so that perturbation strength is proportional to
      the representation scale.
    - **Multi-layer hooks**: Can perturb all layers in a group, not just
      the representative layer, for more thorough hardening.

    Args:
        pgd_steps: Number of PGD steps in the inner loop.
        pgd_alpha: PGD step size.
        epsilon: L-infinity norm bound on the perturbation in subspace.
        num_restarts: Number of random restarts for PGD (1 = no restart).
        epsilon_relative: If True, scale epsilon by mean activation norm.
        epsilon_ratio: Multiplier for relative epsilon (epsilon = ratio * norm).
    """

    def __init__(
        self,
        pgd_steps: int = 7,
        pgd_alpha: float = 0.1,
        epsilon: float = 1.0,
        num_restarts: int = 3,
        epsilon_relative: bool = True,
        epsilon_ratio: float = 0.05,
    ) -> None:
        super().__init__()
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.epsilon = epsilon
        self.num_restarts = max(num_restarts, 1)
        self.epsilon_relative = epsilon_relative
        self.epsilon_ratio = epsilon_ratio

    def _compute_epsilon(
        self,
        h: torch.Tensor,
    ) -> float:
        """Compute effective epsilon, optionally scaled by activation norm.

        Args:
            h: Hidden states ``[B, seq, d]`` or ``[B, d]``.

        Returns:
            Effective epsilon value.
        """
        if self.epsilon_relative:
            mean_norm = h.float().norm(dim=-1).mean().item()
            return self.epsilon_ratio * mean_norm
        return self.epsilon

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

        Uses random restarts to find stronger adversarial perturbations.

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

        # Run PGD independently per group.
        for gidx in sorted(h_clean.keys()):
            V_s = V_s_list[gidx].float()  # [d, d_s]
            h_det = h_detached[gidx]      # [B, seq, d]
            d_s = V_s.shape[1]

            # Calibrate epsilon based on activation norms
            effective_eps = self._compute_epsilon(h_det)

            best_delta: torch.Tensor | None = None
            best_loss: float = -float("inf")

            for restart_idx in range(self.num_restarts):
                # Initialize: first restart from zero, rest from random
                if restart_idx == 0:
                    delta = torch.zeros(
                        h_det.shape[0], h_det.shape[1], d_s,
                        device=h_det.device, dtype=torch.float32,
                    )
                else:
                    delta = torch.empty(
                        h_det.shape[0], h_det.shape[1], d_s,
                        device=h_det.device, dtype=torch.float32,
                    ).uniform_(-effective_eps, effective_eps)

                delta.requires_grad_(True)

                for _step in range(self.pgd_steps):
                    h_perturbed = h_det + delta @ V_s.T.unsqueeze(0)

                    with InjectionHookManager(
                        model, rep_layers,
                        {gidx: h_perturbed},
                        layer_accessor,
                    ):
                        output = model(**forward_kwargs)

                    loss_inner = output.loss
                    (grad,) = torch.autograd.grad(
                        loss_inner, delta, create_graph=False
                    )

                    with torch.no_grad():
                        delta.data += self.pgd_alpha * grad.sign()
                        delta.data.clamp_(-effective_eps, effective_eps)

                # Track best restart
                with torch.no_grad():
                    if loss_inner.item() > best_loss:
                        best_loss = loss_inner.item()
                        best_delta = delta.detach().clone()

            result[gidx] = best_delta

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
