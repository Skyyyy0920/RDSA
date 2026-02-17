"""Activation extraction hooks with safe context-managed lifecycle.

CRITICAL: Hook cleanup is essential. Leaked hooks cause silent memory leaks
and wrong gradients. Always use HookManager as a context manager.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


class HookManager:
    """Context manager for safe registration and cleanup of forward hooks.

    Registers forward hooks on specified layers to capture hidden states.
    All hooks are guaranteed to be removed on exit, even if an exception occurs.

    Usage:
        with HookManager(model, layer_indices=[10, 12, 14]) as hm:
            output = model(**inputs)
            activations = hm.get_activations()  # {layer_idx: [B, seq_len, d]}

    Args:
        model: The transformer model to hook into.
        layer_indices: Which layers to extract activations from.
        extraction_point: "output" to capture layer outputs, "input" for inputs.
        layer_accessor: Dotted path to the layers module list (e.g. "model.layers").
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        extraction_point: str = "output",
        layer_accessor: str = "model.model.layers",
    ) -> None:
        if extraction_point not in ("output", "input"):
            raise ValueError(
                f"extraction_point must be 'output' or 'input', got {extraction_point!r}"
            )
        self._model = model
        self._layer_indices = list(layer_indices)
        self._extraction_point = extraction_point
        self._layer_accessor = layer_accessor
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._activations: dict[int, torch.Tensor] = {}

    def __enter__(self) -> HookManager:
        self._register_hooks()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        self._remove_hooks()

    def _get_layer(self, layer_idx: int) -> nn.Module:
        """Retrieve a specific layer module by index."""
        parts = self._layer_accessor.split(".")
        module = self._model
        for part in parts:
            module = getattr(module, part)
        return module[layer_idx]

    def _register_hooks(self) -> None:
        """Register forward hooks on all target layers."""
        for layer_idx in self._layer_indices:
            layer = self._get_layer(layer_idx)
            hook_fn = self._make_hook_fn(layer_idx)
            handle = layer.register_forward_hook(hook_fn)
            self._handles.append(handle)

    def _remove_hooks(self) -> None:
        """Remove ALL registered hooks. Called in __exit__ to prevent leaks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook_fn(self, layer_idx: int) -> Callable:
        """Create a hook closure that captures activations for a given layer.

        Handles transformer layers that return tuples (hidden_state, ...) by
        extracting the first element.

        Args:
            layer_idx: The index of the layer this hook is attached to.

        Returns:
            A hook function compatible with register_forward_hook.
        """

        def hook_fn(
            module: nn.Module,
            input: torch.Tensor | tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            if self._extraction_point == "output":
                tensor = output[0] if isinstance(output, tuple) else output
            else:
                tensor = input[0] if isinstance(input, tuple) else input
            # Detach to avoid retaining the computation graph
            self._activations[layer_idx] = tensor.detach()

        return hook_fn

    def get_activations(self) -> dict[int, torch.Tensor]:
        """Return collected activations from the last forward pass.

        Returns:
            Dict mapping layer index to activation tensor of shape
            ``[B, seq_len, d]``.
        """
        return dict(self._activations)

    def clear(self) -> None:
        """Free stored activations to release memory."""
        self._activations.clear()


def get_representative_layer_indices(layer_groups: list[list[int]]) -> list[int]:
    """Return the middle layer index for each group.

    For a group ``[10, 11, 12, 13, 14]`` the representative is ``12``
    (index ``len(group) // 2``).

    Args:
        layer_groups: List of layer index lists, one per group.

    Returns:
        List of representative layer indices, one per group.
    """
    return [group[len(group) // 2] for group in layer_groups]


def extract_group_activations(
    model: nn.Module,
    layer_groups: list[list[int]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    aggregate: str = "last_token",
    layer_accessor: str = "model.model.layers",
) -> dict[int, torch.Tensor]:
    """Extract and aggregate activations from the representative layer of each group.

    Args:
        model: The VLM to extract activations from.
        layer_groups: Layer group definitions, e.g. ``[[10,11,12,13,14], ...]``.
        input_ids: Token IDs, shape ``[B, seq_len]``.
        attention_mask: Attention mask, shape ``[B, seq_len]``.
        pixel_values: Optional image tensor, shape ``[B, C, H, W]``.
        aggregate: Aggregation strategy over sequence positions.
            ``"last_token"`` — use the last non-padding token.
            ``"mean"`` — mean-pool over non-padding positions.
            ``"all"`` — no aggregation, return full ``[B, seq_len, d]``.
        layer_accessor: Dotted path to model layers.

    Returns:
        Dict mapping group index to aggregated activation tensor.
        Shape is ``[B, d]`` for ``"last_token"`` and ``"mean"``,
        or ``[B, seq_len, d]`` for ``"all"``.
    """
    if aggregate not in ("last_token", "mean", "all"):
        raise ValueError(f"aggregate must be 'last_token', 'mean', or 'all', got {aggregate!r}")

    rep_layers = get_representative_layer_indices(layer_groups)

    # Build forward kwargs
    forward_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if pixel_values is not None:
        forward_kwargs["pixel_values"] = pixel_values

    with HookManager(model, rep_layers, layer_accessor=layer_accessor) as hm:
        with torch.no_grad():
            model(**forward_kwargs)
        raw = hm.get_activations()

    result: dict[int, torch.Tensor] = {}
    for group_idx, layer_idx in enumerate(rep_layers):
        act = raw[layer_idx]  # [B, seq_len, d]

        if aggregate == "all":
            result[group_idx] = act
        elif aggregate == "last_token":
            # Find the index of the last non-padding token per sample
            seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_indices = torch.arange(act.size(0), device=act.device)
            result[group_idx] = act[batch_indices, seq_lengths]  # [B, d]
        elif aggregate == "mean":
            # Mean pool over non-padding positions
            mask = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            pooled = (act * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            result[group_idx] = pooled  # [B, d]

    return result
