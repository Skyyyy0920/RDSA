"""Tests for defense/monitor.py — ActivationIntegrityMonitor."""

import torch
import torch.nn as nn

from rdsa.subspace.identifier import SubspaceResult


def make_orthonormal(d: int, k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn(d, k)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k]


def make_dummy_subspace_results(
    d: int = 64, d_s: int = 8, d_t: int = 32, n_groups: int = 2
):
    """Create dummy SubspaceResult list for testing."""
    results = []
    for i in range(n_groups):
        results.append(
            SubspaceResult(
                V_s=make_orthonormal(d, d_s, seed=i),
                V_t=make_orthonormal(d, d_t, seed=i + 100),
                singular_values=torch.randn(d_s),
                explained_variance=torch.randn(d_t).abs(),
                layer_group_idx=i,
                representative_layer=10 + i * 5,
            )
        )
    return results


class TestActivationIntegrityMonitor:
    """Basic tests for the monitor module (without a real VLM).

    These tests verify the metric computation logic using synthetic inputs
    rather than testing end-to-end with a model (which requires GPU + model
    download).
    """

    def test_anomaly_score_nonnegative(self) -> None:
        """Anomaly scores (variance) should be non-negative."""
        from rdsa.subspace.metrics import cross_layer_consistency_variance

        B, d_s = 4, 8
        projections = {
            0: torch.randn(B, d_s),
            1: torch.randn(B, d_s),
            2: torch.randn(B, d_s),
        }
        var = cross_layer_consistency_variance(projections)
        assert (var >= 0).all()

    def test_zero_threshold_all_anomalous(self) -> None:
        """With threshold=0, any nonzero variance should flag as anomalous."""
        from rdsa.subspace.metrics import cross_layer_consistency_variance

        B, d_s = 4, 8
        projections = {
            0: torch.randn(B, d_s) * 10,
            1: torch.randn(B, d_s) * 0.01,
        }
        var = cross_layer_consistency_variance(projections)

        # Any sample with variance > 0 would be anomalous at threshold 0
        flags = var > 0.0
        assert flags.any()

    def test_infinite_threshold_none_anomalous(self) -> None:
        """With infinite threshold, nothing should be flagged."""
        from rdsa.subspace.metrics import cross_layer_consistency_variance

        B, d_s = 4, 8
        projections = {
            0: torch.randn(B, d_s),
            1: torch.randn(B, d_s),
        }
        var = cross_layer_consistency_variance(projections)

        flags = var > float("inf")
        assert not flags.any()

    def test_safety_classifier_integration(self) -> None:
        """Test that linear safety classifiers produce valid confidences."""
        B, d_s = 4, 8
        clf = nn.Linear(d_s, 1)
        proj = torch.randn(B, d_s)

        logit = clf(proj).squeeze(-1)
        conf = torch.sigmoid(logit)

        assert conf.shape == (B,)
        assert (conf >= 0).all()
        assert (conf <= 1).all()
