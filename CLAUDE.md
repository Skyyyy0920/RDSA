# RDSA: Resilient Distributed Safety Architecture

Defense method against adversarial attacks on Vision-Language Models (VLMs). Targets the root vulnerability discovered by SCIA (ICML 2026): safety features are localized in ~0.1-0.5% of neurons and disentangleable from semantic features. RDSA makes safety representations distributed, entangled with semantics, and multi-layer redundant.

## Tech Stack

- Python 3.10+, PyTorch 2.x, CUDA 12.x
- transformers, peft (LoRA), accelerate, deepspeed
- Target models: LLaVA-v1.6-Mistral-7B, Qwen2.5-VL-7B-Instruct, LLaMA-3.2-11B-Vision-Instruct
- Evaluation: GPT-4o judge via OpenAI API
- Config management: Hydra + OmegaConf
- Logging: wandb

## Commands

- `python -m rdsa.train --config configs/llava.yaml` — RDSA training
- `python -m rdsa.identify --model llava --output subspaces/` — Safety subspace identification (SVD)
- `python -m rdsa.evaluate --defense rdsa --attack scia --model llava` — Run evaluation
- `python -m rdsa.evaluate --defense rdsa --attack all --model all` — Full evaluation matrix
- `pytest tests/ -x -q` — Run tests (stop on first failure)
- `pytest tests/test_losses.py -k "entanglement"` — Run specific test
- `python scripts/visualize_entanglement.py --model llava` — Generate η profile plots
- `ruff check src/` — Lint
- `ruff format src/` — Format

## Project Structure

```
rdsa/
├── CLAUDE.md
├── configs/                    # Hydra configs
│   ├── llava.yaml
│   ├── qwen.yaml
│   └── llama.yaml
├── src/rdsa/
│   ├── __init__.py
│   ├── config.py               # RDSAConfig dataclass
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hooks.py            # Activation extraction hooks
│   │   └── model_utils.py      # Model loading, layer access helpers
│   ├── subspace/
│   │   ├── __init__.py
│   │   ├── identifier.py       # SafetySubspaceIdentifier (SVD + PCA)
│   │   └── metrics.py          # Entanglement degree η, LCIV computation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py           # EntanglementLoss, ConsistencyLoss, SubspaceLATLoss
│   │   ├── trainer.py          # RDSATrainer (multi-objective training loop)
│   │   └── data.py             # Dataset classes (safe/unsafe pairs, VQA)
│   ├── defense/
│   │   ├── __init__.py
│   │   └── monitor.py          # ActivationIntegrityMonitor (inference-time)
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── scia.py             # SCIA reproduction
│   │   ├── umk.py              # UMK white-box attack
│   │   ├── adaptive.py         # Adaptive-SCIA, Adaptive-PGD, Monitor-Evasion
│   │   └── baselines.py        # FigStep, MM-SafetyBench wrappers
│   └── evaluation/
│       ├── __init__.py
│       ├── judge.py            # GPT-4o safety judge
│       ├── metrics.py          # ASR, RR, OR computation
│       └── benchmarks.py       # VQAv2, MMBench, MME, OR-Bench wrappers
├── scripts/
│   ├── visualize_entanglement.py
│   ├── plot_pareto.py
│   └── run_ablation.py
├── tests/
│   ├── test_subspace.py
│   ├── test_losses.py
│   ├── test_monitor.py
│   └── test_attacks.py
├── docs/
│   ├── MATH.md                 # Full mathematical framework (Theorems 1 & 2)
│   └── EXPERIMENT_DESIGN.md    # 7 experiments, baselines, metrics
└── pyproject.toml
```

## Architecture & Core Concepts

### Two Defense Modules

**Module 1 — Multi-Granularity Safety Encoding (Theorem 1)**
Distribute safety representations across G layer groups. Attack cost scales from O(ε₁*) to O(√Σ(εₖ*)²).

**Module 2 — Safety-Semantic Subspace Entanglement (Theorem 2)**
Maximize entanglement degree η(Vₛ, Vₜ). When η→1, attacker cannot suppress safety without destroying semantics.

### Key Data Structures

- `V_s`: Safety subspace basis, `torch.Tensor` shape `[d, d_s]`, d_s=32 by default
- `V_t`: Semantic subspace basis, `torch.Tensor` shape `[d, d_t]`, d_t=256 by default
- Layer groups (LLaVA): `[[10-14], [18-22], [26-30]]` — skip shallow layers, gap between groups
- `η = (1/kₛ) Σᵢ maxⱼ |vₛ⁽ⁱ⁾ᵀ vₜ⁽ʲ⁾|` — entanglement degree, target η→1

### Training Pipeline (3 steps)

1. **Identify** — SVD on activation differences (safe vs unsafe) → V_s per layer group; PCA on normal data → V_t
2. **Entangle** — Fine-tune with LoRA. Total loss: `L_SFT + α₁·L_entangle + α₂·L_consist + α₃·L_LAT-sub`
3. **Monitor** — Inference anomaly detection via cross-layer safety confidence variance

### Default Hyperparameters

```
d_s=32, d_t=256, G=3, α₁=0.1, α₂=0.05, α₃=0.1, τ=0.2, LAT_α=0.1
LoRA: rank=16, alpha=32, lr=2e-5, epochs=3
```

## Coding Conventions

- Type hints on all function signatures. Use `torch.Tensor` not `Tensor`.
- Docstrings: Google style. Include shape annotations for tensor args: `h: torch.Tensor  # [B, d]`
- All tensor operations must specify device explicitly or inherit from input. Never hardcode `.cuda()`.
- Use `@dataclass` for configs, never raw dicts.
- Hook-based activation extraction: always register AND remove hooks in try/finally or context manager.
- Loss functions inherit `nn.Module`. Training utilities are plain classes.
- Tests use `pytest` with `torch.testing.assert_close` for numerical checks. Tolerance: `atol=1e-5, rtol=1e-4`.
- Log all loss components to wandb every step: `{"loss/total": ..., "loss/entangle": ..., "loss/consist": ..., "loss/lat_sub": ..., "metric/eta": ...}`
- Config override via CLI: `python -m rdsa.train model.name=qwen training.alpha_entangle=0.2`

## Critical Implementation Notes

- **Hook cleanup is critical.** Leaked hooks cause silent memory leaks and wrong gradients. Use `HookManager` context manager in `src/rdsa/models/hooks.py`.
- **SVD numerical stability.** Use `torch.linalg.svd(full_matrices=False)`. For large N, compute on CPU then move result to GPU.
- **Entanglement loss is non-differentiable** (contains `max`). Use `torch.max` which supports autograd, NOT `argmax`.
- **Mixed precision.** Safety subspace projections (`V_s`, `V_t`) must stay fp32 even under AMP. Cast hidden states before projection.
- **Memory.** Collecting activations across 3 layer groups for 7B model needs ~6GB extra. Use gradient checkpointing + activation offloading for training.
- **Reproducibility.** Seed everything: `torch.manual_seed`, `numpy.random.seed`, `random.seed`. Log seed in wandb.

## Model-Specific Layer Access

```python
# LLaVA-v1.6: model.model.layers[i]
# Qwen2.5-VL: model.model.layers[i]
# LLaMA-3.2-Vision: model.model.layers[i]
# Use get_layer(model, idx) helper in model_utils.py — handles all architectures
```

## Reference Files

- `docs/MATH.md` — Full proofs of Theorem 1 (attack cost amplification) and Theorem 2 (attack infeasibility under entanglement)
- `docs/EXPERIMENT_DESIGN.md` — 7 experiments (main comparison, ablation, entanglement analysis, redundancy analysis, capability preservation, adaptive attacks, parameter sensitivity)
- SCIA paper: the attack we primarily defend against. Key figures: Fig 3 (layer-wise probing), Algorithm 1 (attack pipeline)
