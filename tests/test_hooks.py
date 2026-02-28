"""Tests for models/hooks.py — HookManager and activation extraction."""

import pytest
import torch
import torch.nn as nn

from rdsa.models.hooks import (
    HookManager,
    InjectionHookManager,
    get_representative_layer_indices,
)

# ---------------------------------------------------------------------------
# Fixtures: minimal transformer-like model for testing hooks
# ---------------------------------------------------------------------------


class FakeTransformerLayer(nn.Module):
    """Minimal transformer layer that returns (hidden_state,) tuple."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> tuple:
        return (self.linear(x),)


class FakeModel(nn.Module):
    """Minimal model with nested `model.language_model.layers` structure."""

    def __init__(self, num_layers: int = 4, d: int = 64) -> None:
        super().__init__()
        layers = nn.ModuleList([FakeTransformerLayer(d) for _ in range(num_layers)])

        # Nest as model.language_model.layers to match VLM pattern
        # (e.g. Qwen3VLForConditionalGeneration.model -> Qwen3VLModel.language_model -> layers)
        language_model = nn.Module()
        language_model.layers = layers
        wrapper = nn.Module()
        wrapper.language_model = language_model
        self.model = wrapper
        self._num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        B, seq_len = input_ids.shape
        d = self.model.language_model.layers[0].linear.weight.shape[0]
        x = torch.randn(B, seq_len, d, device=input_ids.device)
        for layer in self.model.language_model.layers:
            x = layer(x)[0]
        return x


@pytest.fixture
def fake_model():
    return FakeModel(num_layers=4, d=64)


# ---------------------------------------------------------------------------
# Tests: HookManager
# ---------------------------------------------------------------------------


class TestHookManager:
    def test_activations_shape(self, fake_model: FakeModel) -> None:
        """Activations should have shape [B, seq_len, d]."""
        B, seq_len = 2, 10
        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)

        with HookManager(fake_model, layer_indices=[0, 2]) as hm:
            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            acts = hm.get_activations()

        assert 0 in acts
        assert 2 in acts
        assert acts[0].shape == (B, seq_len, 64)
        assert acts[2].shape == (B, seq_len, 64)

    def test_hooks_removed_after_exit(self, fake_model: FakeModel) -> None:
        """All hooks must be removed after exiting the context manager."""
        with HookManager(fake_model, layer_indices=[0, 1, 2, 3]):
            pass

        # Check that no hooks remain on any layer
        for i in range(4):
            layer = fake_model.model.language_model.layers[i]
            assert len(layer._forward_hooks) == 0

    def test_hooks_removed_on_exception(self, fake_model: FakeModel) -> None:
        """Hooks must be cleaned up even when an exception occurs."""
        with pytest.raises(RuntimeError):
            with HookManager(fake_model, layer_indices=[0, 1]):
                raise RuntimeError("test error")

        for i in range(4):
            layer = fake_model.model.language_model.layers[i]
            assert len(layer._forward_hooks) == 0

    def test_clear_frees_memory(self, fake_model: FakeModel) -> None:
        """clear() should empty the stored activations."""
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        attn_mask = torch.ones(1, 5, dtype=torch.long)

        with HookManager(fake_model, layer_indices=[0]) as hm:
            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            assert len(hm.get_activations()) == 1
            hm.clear()
            assert len(hm.get_activations()) == 0

    def test_multiple_forward_passes_overwrite(self, fake_model: FakeModel) -> None:
        """A second forward pass should overwrite previous activations."""
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        attn_mask = torch.ones(1, 5, dtype=torch.long)

        with HookManager(fake_model, layer_indices=[0]) as hm:
            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            first = hm.get_activations()[0].clone()

            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            second = hm.get_activations()[0]

            # With random init they should be different tensors
            # (but overwritten, not accumulated)
            assert second.shape == first.shape

    def test_invalid_extraction_point(self, fake_model: FakeModel) -> None:
        """Should raise ValueError for invalid extraction_point."""
        with pytest.raises(ValueError, match="extraction_point"):
            HookManager(fake_model, layer_indices=[0], extraction_point="invalid")

    def test_detach_true_no_grad(self, fake_model: FakeModel) -> None:
        """Default detach=True should produce tensors without grad_fn."""
        B, seq_len = 2, 5
        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)

        with HookManager(fake_model, layer_indices=[0], detach=True) as hm:
            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            acts = hm.get_activations()

        assert acts[0].grad_fn is None

    def test_detach_false_retains_grad(self, fake_model: FakeModel) -> None:
        """detach=False should produce tensors that retain computation graph."""
        B, seq_len = 2, 5
        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)

        with HookManager(fake_model, layer_indices=[0], detach=False) as hm:
            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            acts = hm.get_activations()

        assert acts[0].grad_fn is not None

    def test_detach_false_gradient_flows(self, fake_model: FakeModel) -> None:
        """With detach=False, gradients should flow back to model params."""
        B, seq_len = 2, 5
        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)

        # Require gradients on layer 0 weights
        layer0 = fake_model.model.language_model.layers[0]
        layer0.linear.weight.requires_grad_(True)

        with HookManager(fake_model, layer_indices=[0], detach=False) as hm:
            fake_model(input_ids=input_ids, attention_mask=attn_mask)
            acts = hm.get_activations()

        loss = acts[0].sum()
        loss.backward()

        assert layer0.linear.weight.grad is not None
        assert not torch.all(layer0.linear.weight.grad == 0)


# ---------------------------------------------------------------------------
# Tests: InjectionHookManager
# ---------------------------------------------------------------------------


class TestInjectionHookManager:
    def test_injection_replaces_hidden_state(self, fake_model: FakeModel) -> None:
        """Injected hidden states should replace the original layer output."""
        B, seq_len, d = 2, 5, 64
        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)

        # Create a specific tensor to inject
        injected = torch.ones(B, seq_len, d) * 42.0

        # Inject at layer 1, capture at layer 1 to verify
        with InjectionHookManager(
            fake_model, rep_layers=[1], injection_dict={0: injected},
            layer_accessor="model.language_model.layers",
        ):
            with HookManager(
                fake_model, layer_indices=[1], detach=True,
                layer_accessor="model.language_model.layers",
            ) as hm:
                fake_model(input_ids=input_ids, attention_mask=attn_mask)
                acts = hm.get_activations()

        # The captured activation at layer 1 should be the injected tensor
        torch.testing.assert_close(acts[1], injected, atol=1e-5, rtol=1e-4)

    def test_hooks_cleaned_up(self, fake_model: FakeModel) -> None:
        """InjectionHookManager should clean up hooks on exit."""
        injected = torch.randn(1, 5, 64)

        with InjectionHookManager(
            fake_model, rep_layers=[0, 2],
            injection_dict={0: injected, 1: injected},
            layer_accessor="model.language_model.layers",
        ):
            pass

        for i in range(4):
            layer = fake_model.model.language_model.layers[i]
            assert len(layer._forward_hooks) == 0

    def test_injection_preserves_gradient(self, fake_model: FakeModel) -> None:
        """When injected tensor has grad, gradients should flow through."""
        B, seq_len, d = 2, 5, 64
        input_ids = torch.zeros(B, seq_len, dtype=torch.long)
        attn_mask = torch.ones(B, seq_len, dtype=torch.long)

        injected = torch.randn(B, seq_len, d, requires_grad=True)

        with InjectionHookManager(
            fake_model, rep_layers=[1],
            injection_dict={0: injected},
            layer_accessor="model.language_model.layers",
        ):
            output = fake_model(input_ids=input_ids, attention_mask=attn_mask)

        loss = output.sum()
        loss.backward()

        assert injected.grad is not None
        assert not torch.all(injected.grad == 0)


# ---------------------------------------------------------------------------
# Tests: get_representative_layer_indices
# ---------------------------------------------------------------------------


class TestRepresentativeLayerIndices:
    def test_basic(self) -> None:
        groups = [[10, 11, 12, 13, 14], [18, 19, 20, 21, 22], [26, 27, 28, 29, 30]]
        result = get_representative_layer_indices(groups)
        assert result == [12, 20, 28]

    def test_single_layer_group(self) -> None:
        groups = [[5]]
        assert get_representative_layer_indices(groups) == [5]

    def test_two_layer_group(self) -> None:
        groups = [[5, 6]]
        assert get_representative_layer_indices(groups) == [6]

    def test_empty(self) -> None:
        assert get_representative_layer_indices([]) == []
