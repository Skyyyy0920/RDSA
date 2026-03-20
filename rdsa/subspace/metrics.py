"""Entanglement degree eta, LCIV, and related subspace metrics.

These metrics quantify the relationship between safety and semantic subspaces
and are used both for monitoring training progress and evaluation experiments.

CRITICAL: Use torch.max (supports autograd), NOT argmax.
CRITICAL: All subspace computations must be in fp32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def entanglement_degree(
    V_s: torch.Tensor,
    V_t: torch.Tensor,
) -> torch.Tensor:
    """Compute entanglement degree eta between safety and semantic subspaces.

    .. math::
        \\eta(V_s, V_t) = \\frac{1}{d_s} \\sum_i \\max_j |v_s^{(i)T} v_t^{(j)}|

    When ``eta = 0`` the subspaces are orthogonal (fully disentangleable,
    vulnerable). When ``eta = 1`` the safety subspace is fully embedded in
    the semantic subspace (maximally entangled).

    Args:
        V_s: Safety subspace basis ``[d, d_s]``, must be fp32.
        V_t: Semantic subspace basis ``[d, d_t]``, must be fp32.

    Returns:
        Scalar tensor in ``[0, 1]``.
    """
    V_s = V_s.float()
    V_t = V_t.float()

    # Cross-correlation matrix: [d_s, d_t]
    cross = V_s.T @ V_t

    # For each safety direction, find maximum absolute alignment with any
    # semantic direction.  torch.max is differentiable (gradient flows
    # through the argmax element).
    max_alignments = torch.max(cross.abs(), dim=1).values  # [d_s]

    return max_alignments.mean()


def cross_layer_consistency_variance(
    safety_projections: dict[int, torch.Tensor],
    safety_classifiers: dict[int, nn.Linear] | None = None,
) -> torch.Tensor:
    """Compute cross-layer safety confidence variance (LCIV).

    .. math::
        LCIV(x, q) = \\text{Var}_k[\\sigma(w_k^T V_s^{(k)T} h_{l_k})]

    High cross-layer variance indicates an adversarial input that has
    managed to fool some but not all layer-group detectors.

    Args:
        safety_projections: Dict mapping group index to projected activations
            ``[B, d_s]``.
        safety_classifiers: Optional dict mapping group index to linear
            classifiers ``nn.Linear(d_s, 1)``. If ``None``, the L2 norm
            of the projection is used as a proxy confidence.

    Returns:
        Per-sample variance tensor of shape ``[B]``.
    """
    confidences = []

    for group_idx in sorted(safety_projections.keys()):
        proj = safety_projections[group_idx].float()  # [B, d_s]

        if safety_classifiers is not None and group_idx in safety_classifiers:
            logit = safety_classifiers[group_idx](proj).squeeze(-1)  # [B]
            conf = torch.sigmoid(logit)  # [B]
        else:
            # Proxy: normalized L2 norm
            conf = torch.sigmoid(proj.norm(dim=-1))  # [B]

        confidences.append(conf)

    # Stack: [G, B] -> variance across groups
    stacked = torch.stack(confidences, dim=0)  # [G, B]
    return torch.var(stacked, dim=0)  # [B]


def manipulable_dimensions(
    V_s: torch.Tensor,
    V_t: torch.Tensor,
    eta: torch.Tensor | None = None,
) -> float:
    """Compute the attackable degrees of freedom.

    From Theorem 2 (partial entanglement): the number of dimensions in the
    safety subspace that an attacker can manipulate without affecting semantics
    is ``d_s * (1 - eta)``.

    Args:
        V_s: Safety subspace basis ``[d, d_s]``.
        V_t: Semantic subspace basis ``[d, d_t]``.
        eta: Pre-computed entanglement degree. Computed if ``None``.

    Returns:
        Number of manipulable dimensions (float, may be fractional).
    """
    if eta is None:
        eta = entanglement_degree(V_s, V_t)
    d_s = V_s.shape[1]
    return d_s * (1.0 - eta.item())


def subspace_overlap(
    V_s: torch.Tensor,
    V_t: torch.Tensor,
) -> torch.Tensor:
    """Compute the full absolute cross-correlation matrix between subspaces.

    Useful for visualization and detailed per-direction analysis.

    Args:
        V_s: Safety subspace basis ``[d, d_s]``.
        V_t: Semantic subspace basis ``[d, d_t]``.

    Returns:
        Absolute cosine similarity matrix ``[d_s, d_t]``.
    """
    V_s = V_s.float()
    V_t = V_t.float()
    return (V_s.T @ V_t).abs()
