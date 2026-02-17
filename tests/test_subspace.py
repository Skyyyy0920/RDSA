"""Tests for subspace identification and metrics."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.testing

from rdsa.subspace.identifier import SafetySubspaceIdentifier, SubspaceResult
from rdsa.subspace.metrics import (
    cross_layer_consistency_variance,
    entanglement_degree,
    manipulable_dimensions,
    subspace_overlap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_orthonormal(d: int, k: int, seed: int = 0) -> torch.Tensor:
    """Create a random orthonormal matrix [d, k] via QR decomposition."""
    torch.manual_seed(seed)
    A = torch.randn(d, k)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k]


# ---------------------------------------------------------------------------
# Tests: entanglement_degree
# ---------------------------------------------------------------------------


class TestEntanglementDegree:
    def test_orthogonal_subspaces_eta_zero(self) -> None:
        """When V_s and V_t are orthogonal, eta should be ~0."""
        d = 64
        d_s, d_t = 8, 8

        # Construct orthogonal subspaces from different halves of identity
        V_s = torch.zeros(d, d_s)
        V_t = torch.zeros(d, d_t)
        for i in range(d_s):
            V_s[i, i] = 1.0
        for i in range(d_t):
            V_t[d_s + i, i] = 1.0

        eta = entanglement_degree(V_s, V_t)
        torch.testing.assert_close(eta, torch.tensor(0.0), atol=1e-5, rtol=1e-4)

    def test_identical_subspaces_eta_one(self) -> None:
        """When V_s is a subset of V_t, eta should be ~1."""
        d = 64
        d_s, d_t = 8, 32

        V_t = make_orthonormal(d, d_t, seed=42)
        V_s = V_t[:, :d_s]  # V_s is a subset of V_t

        eta = entanglement_degree(V_s, V_t)
        torch.testing.assert_close(eta, torch.tensor(1.0), atol=1e-5, rtol=1e-4)

    def test_eta_in_range(self) -> None:
        """Random subspaces should have eta in [0, 1]."""
        d = 64
        V_s = make_orthonormal(d, 8, seed=10)
        V_t = make_orthonormal(d, 32, seed=20)

        eta = entanglement_degree(V_s, V_t)
        assert 0.0 <= eta.item() <= 1.0

    def test_fp32_output(self) -> None:
        """Output should always be fp32."""
        V_s = make_orthonormal(64, 8).half()
        V_t = make_orthonormal(64, 32).half()
        eta = entanglement_degree(V_s, V_t)
        assert eta.dtype == torch.float32


# ---------------------------------------------------------------------------
# Tests: cross_layer_consistency_variance
# ---------------------------------------------------------------------------


class TestCrossLayerConsistencyVariance:
    def test_identical_projections_zero_variance(self) -> None:
        """When all groups have identical projections, variance should be 0."""
        B, d_s = 4, 8
        proj = torch.randn(B, d_s)
        projections = {0: proj, 1: proj.clone(), 2: proj.clone()}

        var = cross_layer_consistency_variance(projections)
        torch.testing.assert_close(var, torch.zeros(B), atol=1e-5, rtol=1e-4)

    def test_different_projections_nonzero_variance(self) -> None:
        """Different projections should produce nonzero variance."""
        B, d_s = 4, 8
        projections = {
            0: torch.randn(B, d_s) * 10,  # Large norms -> sigmoid near 1
            1: torch.randn(B, d_s) * 0.01,  # Small norms -> sigmoid near 0.5
        }

        var = cross_layer_consistency_variance(projections)
        assert (var > 0).all()

    def test_output_shape(self) -> None:
        """Output should have shape [B]."""
        B = 3
        projections = {0: torch.randn(B, 8), 1: torch.randn(B, 8)}
        var = cross_layer_consistency_variance(projections)
        assert var.shape == (B,)


# ---------------------------------------------------------------------------
# Tests: manipulable_dimensions
# ---------------------------------------------------------------------------


class TestManipulableDimensions:
    def test_fully_entangled(self) -> None:
        """When eta=1, manipulable dimensions should be 0."""
        d = 64
        V_t = make_orthonormal(d, 32, seed=42)
        V_s = V_t[:, :8]
        dims = manipulable_dimensions(V_s, V_t)
        assert abs(dims) < 0.5  # Should be ~0

    def test_fully_disentangled(self) -> None:
        """When eta=0, manipulable dimensions should equal d_s."""
        d = 64
        d_s, d_t = 8, 8
        V_s = torch.zeros(d, d_s)
        V_t = torch.zeros(d, d_t)
        for i in range(d_s):
            V_s[i, i] = 1.0
        for i in range(d_t):
            V_t[d_s + i, i] = 1.0

        dims = manipulable_dimensions(V_s, V_t)
        assert abs(dims - d_s) < 0.5


# ---------------------------------------------------------------------------
# Tests: subspace_overlap
# ---------------------------------------------------------------------------


class TestSubspaceOverlap:
    def test_shape(self) -> None:
        V_s = make_orthonormal(64, 8)
        V_t = make_orthonormal(64, 32)
        overlap = subspace_overlap(V_s, V_t)
        assert overlap.shape == (8, 32)

    def test_values_in_range(self) -> None:
        """Absolute cosine similarities should be in [0, 1]."""
        V_s = make_orthonormal(64, 8)
        V_t = make_orthonormal(64, 32)
        overlap = subspace_overlap(V_s, V_t)
        assert (overlap >= 0).all()
        assert (overlap <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# Tests: SafetySubspaceIdentifier (SVD / PCA)
# ---------------------------------------------------------------------------


class TestSafetySubspaceIdentifier:
    def test_identify_safety_subspace_orthonormality(self) -> None:
        """V_s from SVD should be orthonormal."""
        torch.manual_seed(42)
        N, d, d_s = 100, 64, 8
        safe = torch.randn(N, d)
        unsafe = safe + torch.randn(N, d) * 0.5  # Add structured difference

        identifier = self._make_identifier(d=d, d_s=d_s)
        V_s, sv = identifier.identify_safety_subspace(safe, unsafe, d_s=d_s)

        assert V_s.shape == (d, d_s)
        assert V_s.dtype == torch.float32
        torch.testing.assert_close(
            V_s.T @ V_s, torch.eye(d_s), atol=1e-5, rtol=1e-4
        )

    def test_identify_semantic_subspace_orthonormality(self) -> None:
        """V_t from PCA should be orthonormal."""
        torch.manual_seed(42)
        N, d, d_t = 200, 64, 32

        normal = torch.randn(N, d)
        identifier = self._make_identifier(d=d, d_t=d_t)
        V_t, ev = identifier.identify_semantic_subspace(normal, d_t=d_t)

        assert V_t.shape == (d, d_t)
        assert V_t.dtype == torch.float32
        torch.testing.assert_close(
            V_t.T @ V_t, torch.eye(d_t), atol=1e-5, rtol=1e-4
        )

    def test_singular_values_descending(self) -> None:
        """Singular values should be in descending order."""
        torch.manual_seed(42)
        safe = torch.randn(50, 64)
        unsafe = safe + torch.randn(50, 64)

        identifier = self._make_identifier(d=64, d_s=8)
        _, sv = identifier.identify_safety_subspace(safe, unsafe, d_s=8)

        for i in range(len(sv) - 1):
            assert sv[i] >= sv[i + 1]

    def test_explained_variance_sums_to_less_than_one(self) -> None:
        """Explained variance ratios for top d_t components should sum to <= 1."""
        torch.manual_seed(42)
        normal = torch.randn(200, 64)
        identifier = self._make_identifier(d=64, d_t=16)
        _, ev = identifier.identify_semantic_subspace(normal, d_t=16)

        assert ev.sum().item() <= 1.0 + 1e-5

    def test_save_load_roundtrip(self) -> None:
        """Save/load should preserve subspace bases exactly."""
        results = [
            SubspaceResult(
                V_s=torch.randn(64, 8),
                V_t=torch.randn(64, 32),
                singular_values=torch.randn(8),
                explained_variance=torch.randn(32),
                layer_group_idx=0,
                representative_layer=12,
            ),
            SubspaceResult(
                V_s=torch.randn(64, 8),
                V_t=torch.randn(64, 32),
                singular_values=torch.randn(8),
                explained_variance=torch.randn(32),
                layer_group_idx=1,
                representative_layer=20,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            SafetySubspaceIdentifier.save_subspaces(results, tmpdir)
            loaded = SafetySubspaceIdentifier.load_subspaces(tmpdir)

        assert len(loaded) == 2
        for orig, load in zip(results, loaded):
            torch.testing.assert_close(orig.V_s, load.V_s)
            torch.testing.assert_close(orig.V_t, load.V_t)
            assert orig.layer_group_idx == load.layer_group_idx
            assert orig.representative_layer == load.representative_layer

    @staticmethod
    def _make_identifier(d: int = 64, d_s: int = 8, d_t: int = 32):
        """Create a SafetySubspaceIdentifier with a dummy model."""
        from rdsa.config import RDSAConfig

        config = RDSAConfig()
        config.model.hidden_dim = d
        config.subspace.d_safe = d_s
        config.subspace.d_semantic = d_t

        # Use a dummy model — we only test SVD/PCA, not activation collection
        dummy = torch.nn.Linear(1, 1)
        return SafetySubspaceIdentifier(dummy, config)
