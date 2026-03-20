"""Activation extraction and injection hooks with safe context-managed lifecycle.

CRITICAL: Hook cleanup is essential. Leaked hooks cause silent memory leaks
and wrong gradients. Always use hook managers as context managers.
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
        detach: If True, detach captured tensors from the computation graph.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_indices: list[int],
        extraction_point: str = "output",
        layer_accessor: str = "model.language_model.layers",
        detach: bool = True,
    ) -> None:
        if extraction_point not in ("output", "input"):
            raise ValueError(
                f"extraction_point must be 'output' or 'input', got {extraction_point!r}"
            )
        self._model = model
        self._layer_indices = list(layer_indices)
        self._extraction_point = extraction_point
        self._layer_accessor = layer_accessor
        self._detach = detach
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
        """Retrieve a specific layer module by index.

        Automatically unwraps PeftModel (LoRA) wrappers so that the
        layer accessor path works regardless of whether LoRA is applied.
        """
        return _resolve_layer(self._model, layer_idx, self._layer_accessor)

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
            # When detach=True (default), disconnect from the computation graph
            # to save memory.  When detach=False, keep the graph alive so that
            # losses computed on captured activations can backpropagate through
            # the model (required for consistency loss L_consist and SA-AT
            # outer loop).
            self._activations[layer_idx] = (
                tensor.detach() if self._detach else tensor
            )

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


def _resolve_layer(
    model: nn.Module, layer_idx: int, layer_accessor: str
) -> nn.Module:
    """Resolve a specific layer module, unwrapping peft if needed.

    Args:
        model: The model (possibly LoRA-wrapped).
        layer_idx: Zero-indexed layer number.
        layer_accessor: Dotted path to the layers module list.

    Returns:
        The transformer layer module at the given index.
    """
    module = model
    if hasattr(module, "get_base_model"):
        module = module.get_base_model()
    for part in layer_accessor.split("."):
        module = getattr(module, part)
    return module[layer_idx]


class InjectionHookManager:
    """Injects replacement hidden states at specified layers via forward hooks.

    Used by SA-AT for both the PGD inner loop (replace with
    ``h_detached + perturbation``) and the outer training loop (replace
    with ``h + perturbation`` where ``h`` retains the computation graph).

    The hook replaces the first element of the layer output with the
    pre-computed tensor, preserving any additional outputs (attention
    weights, cache, etc.).

    Usage::

        with InjectionHookManager(model, rep_layers, perturbed_h, accessor):
            output = model(**inputs)  # hidden states are replaced

    Args:
        model: The VLM (possibly LoRA-wrapped).
        rep_layers: Representative layer indices, one per group.
        injection_dict: ``{group_idx: tensor [B, seq, d]}`` — the hidden
            states to inject at each representative layer.
        layer_accessor: Dotted path to transformer layers.
    """

    def __init__(
        self,
        model: nn.Module,
        rep_layers: list[int],
        injection_dict: dict[int, torch.Tensor],
        layer_accessor: str = "model.language_model.layers",
    ) -> None:
        self._model = model
        self._rep_layers = rep_layers
        self._injection_dict = injection_dict
        self._layer_accessor = layer_accessor
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> InjectionHookManager:
        for group_idx, layer_idx in enumerate(self._rep_layers):
            if group_idx not in self._injection_dict:
                continue
            layer = _resolve_layer(
                self._model, layer_idx, self._layer_accessor
            )
            hook_fn = self._make_injection_hook(group_idx)
            handle = layer.register_forward_hook(hook_fn)
            self._handles.append(handle)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_injection_hook(self, group_idx: int) -> Callable:
        """Create a hook that replaces hidden states with the injected tensor."""
        h_inject = self._injection_dict[group_idx]

        def hook_fn(
            module: nn.Module,
            input: torch.Tensor | tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            if isinstance(output, tuple):
                return (h_inject.to(output[0].dtype),) + output[1:]
            return h_inject.to(output.dtype)

        return hook_fn


class AdditiveInjectionHookManager:
    """Adds perturbations to hidden states at specified layers.

    Unlike ``InjectionHookManager`` which **replaces** the hidden state,
    this manager **adds** a perturbation to the natural layer output.
    This preserves the full computation graph through the model, which
    is required for the SA-AT outer training loop where gradients must
    flow through all layers' LoRA parameters simultaneously.

    Usage::

        with AdditiveInjectionHookManager(model, rep_layers, deltas, V_s, accessor):
            output = model(**inputs)  # h_out = h_natural + V_s @ delta

    Args:
        model: The VLM (possibly LoRA-wrapped).
        rep_layers: Representative layer indices, one per group.
        delta_dict: ``{group_idx: [B, seq, d_s]}`` perturbations in subspace.
        V_s_list: Safety subspace bases, one ``[d, d_s]`` per group.
        layer_accessor: Dotted path to transformer layers.
    """

    def __init__(
        self,
        model: nn.Module,
        rep_layers: list[int],
        delta_dict: dict[int, torch.Tensor],
        V_s_list: list[torch.Tensor],
        layer_accessor: str = "model.language_model.layers",
    ) -> None:
        self._model = model
        self._rep_layers = rep_layers
        self._delta_dict = delta_dict
        self._V_s_list = V_s_list
        self._layer_accessor = layer_accessor
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def __enter__(self) -> AdditiveInjectionHookManager:
        for group_idx, layer_idx in enumerate(self._rep_layers):
            if group_idx not in self._delta_dict:
                continue
            layer = _resolve_layer(
                self._model, layer_idx, self._layer_accessor
            )
            hook_fn = self._make_additive_hook(group_idx)
            handle = layer.register_forward_hook(hook_fn)
            self._handles.append(handle)
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_additive_hook(self, group_idx: int) -> Callable:
        """Create a hook that ADDS V_s @ delta to the layer output."""
        delta = self._delta_dict[group_idx]  # [B, seq, d_s]
        V_s = self._V_s_list[group_idx].float()  # [d, d_s]

        def hook_fn(
            module: nn.Module,
            input: torch.Tensor | tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> torch.Tensor | tuple[torch.Tensor, ...]:
            h = output[0] if isinstance(output, tuple) else output
            # Add perturbation: delta @ V_s.T → [B, seq, d]
            perturbation = delta @ V_s.T.unsqueeze(0)
            h_perturbed = h + perturbation.to(h.dtype)
            if isinstance(output, tuple):
                return (h_perturbed,) + output[1:]
            return h_perturbed

        return hook_fn


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


def get_all_group_layer_indices(layer_groups: list[list[int]]) -> list[int]:
    """Return all layer indices across all groups, deduplicated and sorted.

    Used when ``hook_all_group_layers=True`` to harden every layer in
    each group, not just the representative.

    Args:
        layer_groups: List of layer index lists, one per group.

    Returns:
        Sorted list of unique layer indices across all groups.
    """
    all_layers: set[int] = set()
    for group in layer_groups:
        all_layers.update(group)
    return sorted(all_layers)


def group_idx_for_layer(
    layer_idx: int,
    layer_groups: list[list[int]],
) -> int | None:
    """Return the group index a layer belongs to, or None if not in any group.

    Args:
        layer_idx: Layer index to look up.
        layer_groups: Layer group definitions.

    Returns:
        Group index (0-based) or None.
    """
    for gidx, group in enumerate(layer_groups):
        if layer_idx in group:
            return gidx
    return None


def extract_group_activations(
    model: nn.Module,
    layer_groups: list[list[int]],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    aggregate: str = "last_token",
    layer_accessor: str = "model.language_model.layers",
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
