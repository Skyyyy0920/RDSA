"""Tests for RDSA training losses."""

import pytest
import torch
import torch.testing

from rdsa.config import TrainingConfig
from rdsa.training.losses import (
    ConsistencyLoss,
    EntanglementLoss,
    RDSALoss,
    SubspaceLATLoss,
)


def make_orthonormal(d: int, k: int, seed: int = 0) -> torch.Tensor:
    """Create a random orthonormal matrix [d, k]."""
    torch.manual_seed(seed)
    A = torch.randn(d, k)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k]


# ---------------------------------------------------------------------------
# Tests: EntanglementLoss
# ---------------------------------------------------------------------------


class TestEntanglementLoss:
    def test_orthogonal_loss_near_zero(self) -> None:
        """Orthogonal V_s and V_t should produce loss near 0 (since -eta ≈ 0)."""
        d, d_s, d_t = 64, 8, 8
        V_s = torch.zeros(d, d_s)
        V_t = torch.zeros(d, d_t)
        for i in range(d_s):
            V_s[i, i] = 1.0
        for i in range(d_t):
            V_t[d_s + i, i] = 1.0

        loss_fn = EntanglementLoss()
        loss = loss_fn([V_s], [V_t])
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=1e-4)

    def test_aligned_loss_near_negative_one(self) -> None:
        """When V_s ⊂ V_t, loss should be near -1."""
        d, d_s, d_t = 64, 8, 32
        V_t = make_orthonormal(d, d_t, seed=42)
        V_s = V_t[:, :d_s]

        loss_fn = EntanglementLoss()
        loss = loss_fn([V_s], [V_t])
        torch.testing.assert_close(loss, torch.tensor(-1.0), atol=1e-5, rtol=1e-4)

    def test_gradient_flows(self) -> None:
        """Gradient should flow through EntanglementLoss via torch.max."""
        d, d_s, d_t = 64, 8, 32
        V_s = make_orthonormal(d, d_s, seed=10).requires_grad_(True)
        V_t = make_orthonormal(d, d_t, seed=20)

        loss_fn = EntanglementLoss()
        loss = loss_fn([V_s], [V_t])
        loss.backward()

        assert V_s.grad is not None
        assert not torch.all(V_s.grad == 0)

    def test_multiple_groups(self) -> None:
        """Loss should average across multiple groups."""
        d = 64
        # Group 0: orthogonal (eta=0), Group 1: aligned (eta=1)
        V_s_0 = torch.zeros(d, 8)
        V_t_0 = torch.zeros(d, 8)
        for i in range(8):
            V_s_0[i, i] = 1.0
            V_t_0[8 + i, i] = 1.0

        V_t_1 = make_orthonormal(d, 32, seed=42)
        V_s_1 = V_t_1[:, :8]

        loss_fn = EntanglementLoss()
        loss = loss_fn([V_s_0, V_s_1], [V_t_0, V_t_1])

        # Average of 0 and -1 = -0.5
        torch.testing.assert_close(loss, torch.tensor(-0.5), atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: ConsistencyLoss
# ---------------------------------------------------------------------------


class TestConsistencyLoss:
    def test_identical_projections_zero_loss(self) -> None:
        """Identical safety projections across groups -> loss = 0."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)
        h = torch.randn(B, d)

        # Same hidden states for all groups
        hidden = {0: h, 1: h.clone(), 2: h.clone()}
        loss_fn = ConsistencyLoss()
        loss = loss_fn(hidden, [V_s, V_s, V_s])

        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=1e-4)

    def test_orthogonal_projections_nonzero_loss(self) -> None:
        """Very different projections should produce significant loss."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)

        h0 = torch.randn(B, d)
        h1 = -h0  # Negated -> cosine = -1 -> loss per pair = 2

        hidden = {0: h0, 1: h1}
        loss_fn = ConsistencyLoss()
        loss = loss_fn(hidden, [V_s, V_s])

        assert loss.item() > 0


# ---------------------------------------------------------------------------
# Tests: SubspaceLATLoss
# ---------------------------------------------------------------------------


class TestSubspaceLATLoss:
    def test_perturbation_shape(self) -> None:
        """Perturbed hidden states should have same shape as input."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)
        h = torch.randn(B, d)

        lat = SubspaceLATLoss(perturbation_alpha=0.1)
        h_perturbed = lat._perturb_in_subspace(h, V_s)

        assert h_perturbed.shape == h.shape

    def test_perturbation_in_subspace(self) -> None:
        """Perturbation should lie within the V_s column space."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)
        h = torch.randn(B, d)

        lat = SubspaceLATLoss(perturbation_alpha=0.1)
        h_perturbed = lat._perturb_in_subspace(h, V_s)

        delta = h_perturbed - h.float()  # [B, d]

        # Project delta onto V_s: should recover delta (since delta ∈ col(V_s))
        delta_proj = delta @ V_s @ V_s.T  # [B, d]
        torch.testing.assert_close(delta, delta_proj, atol=1e-5, rtol=1e-4)

    def test_forward_returns_scalar(self) -> None:
        """Forward should return a scalar loss."""
        d, d_s, B = 64, 8, 4
        V_s = make_orthonormal(d, d_s, seed=42)
        hidden = {0: torch.randn(B, d)}

        def dummy_safety_loss(states):
            return states[0].sum()

        lat = SubspaceLATLoss(perturbation_alpha=0.1)
        loss = lat(hidden, [V_s], dummy_safety_loss)

        assert loss.dim() == 0  # scalar


# ---------------------------------------------------------------------------
# Tests: RDSALoss
# ---------------------------------------------------------------------------


class TestRDSALoss:
    def test_weight_scaling(self) -> None:
        """With alpha_1=1, alpha_2=0, alpha_3=0, total should be SFT + entangle."""
        d, d_s, d_t, B = 64, 8, 32, 4

        config = TrainingConfig()
        config.alpha_entangle = 1.0
        config.alpha_consist = 0.0
        config.alpha_lat_sub = 0.0

        V_t = make_orthonormal(d, d_t, seed=42)
        V_s = V_t[:, :d_s]
        hidden = {0: torch.randn(B, d)}

        def dummy_safety(states):
            return torch.tensor(0.0)

        rdsa_loss = RDSALoss(config)
        sft = torch.tensor(2.0)

        total, loss_dict = rdsa_loss(sft, hidden, [V_s], [V_t], dummy_safety)

        # entangle loss for aligned subspaces = -1
        expected = 2.0 + 1.0 * (-1.0)  # = 1.0
        torch.testing.assert_close(total, torch.tensor(expected), atol=1e-4, rtol=1e-4)

    def test_loss_dict_keys(self) -> None:
        """Loss dict should contain all required keys for wandb logging."""
        d, d_s, d_t, B = 64, 8, 32, 4
        config = TrainingConfig()

        V_s = make_orthonormal(d, d_s, seed=10)
        V_t = make_orthonormal(d, d_t, seed=20)
        hidden = {0: torch.randn(B, d)}

        def dummy_safety(states):
            return torch.tensor(0.0)

        rdsa_loss = RDSALoss(config)
        _, loss_dict = rdsa_loss(torch.tensor(1.0), hidden, [V_s], [V_t], dummy_safety)

        required_keys = {
            "loss/total",
            "loss/sft",
            "loss/entangle",
            "loss/consist",
            "loss/lat_sub",
            "metric/eta",
        }
        assert required_keys == set(loss_dict.keys())

    def test_eta_in_range(self) -> None:
        """Reported eta should be in [0, 1]."""
        d, d_s, d_t, B = 64, 8, 32, 4
        config = TrainingConfig()

        V_s = make_orthonormal(d, d_s, seed=10)
        V_t = make_orthonormal(d, d_t, seed=20)
        hidden = {0: torch.randn(B, d)}

        def dummy_safety(states):
            return torch.tensor(0.0)

        rdsa_loss = RDSALoss(config)
        _, loss_dict = rdsa_loss(torch.tensor(1.0), hidden, [V_s], [V_t], dummy_safety)

        assert 0.0 <= loss_dict["metric/eta"] <= 1.0
