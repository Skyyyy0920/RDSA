"""Tests for attack modules (scia.py, umk.py).

These tests verify attack mechanics with synthetic data — they do NOT
require a real VLM or GPU.
"""

import torch


class TestPGDMechanics:
    """Test PGD-style attack invariants without a real model."""

    def test_epsilon_ball_constraint(self) -> None:
        """Adversarial perturbation should stay within L_inf epsilon ball."""
        epsilon = 16 / 255
        C, H, W = 3, 32, 32

        image = torch.rand(C, H, W)
        delta = torch.randn(C, H, W) * 0.1

        # Clamp delta to epsilon ball
        delta = delta.clamp(-epsilon, epsilon)
        # Clamp adversarial image to valid range
        adv = (image + delta).clamp(0, 1)

        assert (adv - image).abs().max() <= epsilon + 1e-6
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0

    def test_pgd_step_reduces_delta_norm(self) -> None:
        """PGD with sign gradient step should not exceed step_size per pixel."""
        step_size = 1 / 255
        C, H, W = 3, 32, 32

        delta = torch.zeros(C, H, W)
        grad = torch.randn(C, H, W)

        # PGD step
        delta_new = delta + step_size * grad.sign()

        assert (delta_new - delta).abs().max() <= step_size + 1e-8

    def test_adversarial_image_shape_preserved(self) -> None:
        """Adversarial image should have same shape as clean image."""
        C, H, W = 3, 224, 224
        image = torch.rand(C, H, W)
        delta = torch.zeros_like(image)
        adv = (image + delta).clamp(0, 1)
        assert adv.shape == (C, H, W)


class TestSCIASafetyNeuronMasks:
    """Test safety neuron identification logic."""

    def test_top_k_ratio(self) -> None:
        """Top-k ratio should select the expected number of neurons."""
        d = 4096
        top_k_ratio = 0.01  # 1%
        expected_k = int(d * top_k_ratio)

        diff = torch.randn(d).abs()
        k = max(1, int(diff.numel() * top_k_ratio))
        threshold = diff.topk(k).values[-1]
        mask = diff >= threshold

        assert mask.sum().item() >= expected_k
        assert mask.sum().item() <= expected_k + 10  # Allow small ties

    def test_mask_selects_largest_differences(self) -> None:
        """Safety neuron mask should select neurons with largest activation diffs."""
        d = 100
        diff = torch.zeros(d)
        # Plant 5 strong safety neurons
        safety_indices = [10, 20, 30, 40, 50]
        for idx in safety_indices:
            diff[idx] = 10.0

        top_k_ratio = 0.05  # 5% = 5 neurons
        k = max(1, int(d * top_k_ratio))
        threshold = diff.topk(k).values[-1]
        mask = diff >= threshold

        for idx in safety_indices:
            assert mask[idx], f"Safety neuron at index {idx} should be selected"


class TestEvalMetrics:
    """Test evaluation metrics computation."""

    def test_attack_success_rate_basic(self) -> None:
        from rdsa.evaluation.metrics import attack_success_rate

        judgments = [
            {"is_harmful": True},
            {"is_harmful": True},
            {"is_harmful": False},
            {"is_harmful": False},
        ]
        assert attack_success_rate(judgments) == 0.5

    def test_attack_success_rate_empty(self) -> None:
        from rdsa.evaluation.metrics import attack_success_rate

        assert attack_success_rate([]) == 0.0

    def test_refusal_rate_basic(self) -> None:
        from rdsa.evaluation.metrics import refusal_rate

        responses = [
            "I'm sorry, I cannot help with that.",
            "Sure, here is how to do it.",
            "I refuse to answer that question.",
            "Here are the steps you requested.",
        ]
        assert refusal_rate(responses) == 0.5

    def test_over_refusal_rate(self) -> None:
        from rdsa.evaluation.metrics import over_refusal_rate

        judgments = [
            {"is_harmful": True},  # Incorrectly refused benign query
            {"is_harmful": False},
            {"is_harmful": False},
        ]
        assert abs(over_refusal_rate(judgments) - 1 / 3) < 1e-6

    def test_compute_all_metrics(self) -> None:
        from rdsa.evaluation.metrics import compute_all_metrics

        harmful_judgments = [{"is_harmful": True}, {"is_harmful": False}]
        harmful_responses = ["Sure, here is how.", "I'm sorry, I cannot."]
        benign_judgments = [{"is_harmful": True}, {"is_harmful": False}]

        result = compute_all_metrics(
            harmful_judgments, harmful_responses, benign_judgments
        )

        assert "asr" in result
        assert "rr" in result
        assert "or" in result
        assert 0.0 <= result["asr"] <= 1.0
        assert 0.0 <= result["rr"] <= 1.0
        assert 0.0 <= result["or"] <= 1.0
