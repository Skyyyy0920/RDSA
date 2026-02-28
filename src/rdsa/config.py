"""RDSA configuration dataclass."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-VL-8B-Instruct"
    architecture: str = "qwen3vl"
    hidden_dim: int = 4096
    num_layers: int = 32
    layer_groups: list[list[int]] = field(
        default_factory=lambda: [[8, 9, 10, 11, 12], [16, 17, 18, 19, 20], [24, 25, 26, 27, 28]]
    )
    # HuggingFace model class name (e.g. "Qwen3VLForConditionalGeneration")
    model_class: str = "Qwen3VLForConditionalGeneration"
    # Dotted path from model root to the transformer layer list
    layer_access_path: str = "model.language_model.layers"


@dataclass
class SubspaceConfig:
    d_safe: int = 32
    d_semantic: int = 256
    n_contrast_samples: int = 2000
    n_semantic_samples: int = 5000


@dataclass
class TrainingConfig:
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    lora_dropout: float = 0.05

    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_epochs: int = 5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # SA-AT loss weight
    alpha_sa_at: float = 0.3
    # Cross-layer consistency weight
    alpha_consist: float = 0.05

    # SA-AT hyperparameters
    sa_at_pgd_steps: int = 7
    sa_at_pgd_alpha: float = 0.1
    sa_at_epsilon: float = 1.0

    # Data
    harmful_benign_ratio: float = 1.0


@dataclass
class MonitorConfig:
    threshold: float = 0.2
    conservative_mode: bool = True


@dataclass
class RDSAConfig:
    """Top-level RDSA configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    subspace: SubspaceConfig = field(default_factory=SubspaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    @property
    def num_groups(self) -> int:
        return len(self.model.layer_groups)

    @property
    def safety_ratio(self) -> float:
        """d_s / d — should be << 1 for performance preservation."""
        return self.subspace.d_safe / self.model.hidden_dim
