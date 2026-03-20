"""Model utilities: hooks and architecture-specific helpers."""

from rdsa.models.hooks import HookManager, extract_group_activations
from rdsa.models.model_utils import (
    apply_lora,
    count_trainable_parameters,
    get_layer,
    get_layers,
    load_model_and_processor,
)

__all__ = [
    "HookManager",
    "extract_group_activations",
    "apply_lora",
    "count_trainable_parameters",
    "get_layer",
    "get_layers",
    "load_model_and_processor",
]
