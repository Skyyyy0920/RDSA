"""Model loading, layer access helpers, and LoRA application.

Handles architecture-specific layer access patterns for supported VLMs:
- Qwen3-VL, Gemma-3, LLaMA-3.2-Vision (may require HF login)
- InternVL2.5, MiniCPM-V 2.6 (fully open, no login required)
"""

from __future__ import annotations

from functools import reduce
from typing import Any

import torch
import torch.nn as nn

from rdsa.config import ModelConfig, TrainingConfig

# Maps architecture name -> dotted path to the transformer layer list.
LAYER_ACCESSORS: dict[str, str] = {
    "qwen3vl": "model.language_model.layers",
    "gemma3": "model.language_model.layers",
    "llama_vision": "model.language_model.layers",
    "internvl2": "language_model.model.layers",
    "minicpm_v": "llm.model.layers",
}

# Maps architecture name -> HuggingFace model class name.
MODEL_CLASSES: dict[str, str] = {
    "qwen3vl": "Qwen3VLForConditionalGeneration",
    "gemma3": "Gemma3ForConditionalGeneration",
    "llama_vision": "MllamaForConditionalGeneration",
    "internvl2": "AutoModel",
    "minicpm_v": "AutoModel",
}


def _resolve_attr(obj: Any, dotted_path: str) -> Any:
    """Resolve a dotted attribute path like ``'model.language_model.layers'``."""
    return reduce(getattr, dotted_path.split("."), obj)


def get_layer_accessor(architecture: str) -> str:
    """Return the dotted layer accessor path for a given architecture.

    Args:
        architecture: One of ``"qwen3vl"``, ``"gemma3"``, ``"llama_vision"``.

    Returns:
        Dotted path string for accessing transformer layers.

    Raises:
        ValueError: If architecture is unknown.
    """
    if architecture not in LAYER_ACCESSORS:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            f"Supported: {list(LAYER_ACCESSORS.keys())}"
        )
    return LAYER_ACCESSORS[architecture]


def get_layers(model: nn.Module, architecture: str = "qwen3vl") -> nn.ModuleList:
    """Return the full ModuleList of transformer layers.

    Args:
        model: The loaded VLM.
        architecture: One of ``"qwen3vl"``, ``"gemma3"``, ``"llama_vision"``.

    Returns:
        The ``nn.ModuleList`` containing all transformer layers.
    """
    accessor = get_layer_accessor(architecture)
    return _resolve_attr(model, accessor)


def get_layer(
    model: nn.Module,
    layer_idx: int,
    architecture: str = "qwen3vl",
) -> nn.Module:
    """Access a specific transformer layer by index.

    Args:
        model: The loaded VLM.
        layer_idx: Zero-indexed layer number.
        architecture: One of ``"qwen3vl"``, ``"gemma3"``, ``"llama_vision"``.

    Returns:
        The transformer layer module at the given index.

    Raises:
        ValueError: If ``layer_idx`` is out of range or architecture is unknown.
    """
    layers = get_layers(model, architecture)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(
            f"layer_idx {layer_idx} out of range for model with {len(layers)} layers"
        )
    return layers[layer_idx]


def get_representative_layer_indices(layer_groups: list[list[int]]) -> list[int]:
    """Return the middle layer index for each group.

    Args:
        layer_groups: List of layer index lists, one per group.

    Returns:
        List of representative layer indices, one per group.
    """
    return [group[len(group) // 2] for group in layer_groups]


def load_model_and_processor(
    config: ModelConfig,
    device: torch.device | None = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: torch.dtype | None = None,
) -> tuple[nn.Module, Any]:
    """Load a VLM and its processor/tokenizer.

    Uses HuggingFace ``transformers`` auto classes or architecture-specific
    classes. Applies device mapping and quantization as requested.

    Args:
        config: Model configuration specifying name and architecture.
        device: Target device. If ``None``, uses ``"auto"`` device map.
        load_in_8bit: Load model in 8-bit quantization.
        load_in_4bit: Load model in 4-bit quantization.
        torch_dtype: Data type for model weights. Defaults to ``torch.bfloat16``.

    Returns:
        ``(model, processor)`` tuple.
    """
    from transformers import AutoProcessor

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    load_kwargs: dict[str, Any] = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if load_in_8bit or load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
        load_kwargs["quantization_config"] = quant_config
        load_kwargs["device_map"] = "auto"
    elif device is not None:
        load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["device_map"] = "auto"

    architecture = config.architecture

    if architecture == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.name, **load_kwargs
        )
    elif architecture == "gemma3":
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            config.name, **load_kwargs
        )
    elif architecture == "llama_vision":
        from transformers import MllamaForConditionalGeneration

        model = MllamaForConditionalGeneration.from_pretrained(
            config.name, **load_kwargs
        )
    elif architecture == "internvl2":
        from transformers import AutoModel

        load_kwargs.pop("dtype", None)
        model = AutoModel.from_pretrained(
            config.name, torch_dtype=torch_dtype, **load_kwargs
        )
    elif architecture == "minicpm_v":
        from transformers import AutoModel

        load_kwargs.pop("dtype", None)
        model = AutoModel.from_pretrained(
            config.name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            **load_kwargs,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture!r}")

    processor = AutoProcessor.from_pretrained(
        config.name,
        trust_remote_code=(architecture == "minicpm_v"),
    )

    return model, processor


def apply_lora(
    model: nn.Module,
    config: TrainingConfig,
) -> nn.Module:
    """Apply LoRA adapters to the model using the ``peft`` library.

    Freezes the base model and makes only LoRA parameters trainable.

    Args:
        model: The pretrained VLM.
        config: Training configuration with LoRA hyperparameters.

    Returns:
        The model wrapped with LoRA adapters.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters.

    Args:
        model: A PyTorch model.

    Returns:
        ``(trainable_params, total_params)`` tuple.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing for memory-efficient training.

    Uses ``use_reentrant=False`` (modern PyTorch default) so that:
    - Forward hooks work correctly with the computation graph
    - No spurious "None of the inputs have requires_grad=True" warnings
    - SA-AT injection hooks preserve gradient flow through delta

    Args:
        model: The model to enable gradient checkpointing on.
    """
    gc_kwargs = {"use_reentrant": False}
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs,
        )
    elif hasattr(model, "base_model") and hasattr(
        model.base_model, "gradient_checkpointing_enable"
    ):
        # LoRA-wrapped models
        model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gc_kwargs,
        )


def disable_gradient_checkpointing(model: nn.Module) -> None:
    """Disable gradient checkpointing.

    Used to temporarily turn off checkpointing during SA-AT PGD inner
    loop, where we only need the delta→loss graph (not full model graph)
    and checkpointing would waste compute on re-computation.

    Args:
        model: The model to disable gradient checkpointing on.
    """
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    elif hasattr(model, "base_model") and hasattr(
        model.base_model, "gradient_checkpointing_disable"
    ):
        model.base_model.gradient_checkpointing_disable()
