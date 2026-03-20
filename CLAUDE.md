# RDSA: Resilient Distributed Safety Architecture

Defense method against adversarial attacks on Vision-Language Models (VLMs). Targets the root vulnerability discovered by SCIA (ICML 2026): safety features are localized in ~0.1-0.5% of neurons and disentangleable from semantic features. RDSA makes safety representations distributed, entangled with semantics, and multi-layer redundant.

## Tech Stack

- Python 3.10+, PyTorch 2.x, CUDA 12.x
- transformers (>=4.57.0), peft (LoRA), accelerate, deepspeed
- Surrogate models (white-box, LoRA training): Qwen3-VL-8B-Instruct, Gemma-3-12B-IT, LLaMA-3.2-11B-Vision-Instruct
- Victim models (black-box evaluation): Qwen3-VL-30B-A3B-Instruct, Gemma-3-27B-IT, LLaMA-3.2-11B-Vision-Instruct
- Commercial models: GPT-4o, Gemini-2.5-Flash, Claude-Sonnet-4.5
- Evaluation: GPT-4o judge via OpenAI API
- Config management: Hydra + OmegaConf
- Logging: wandb

## Commands

- `python -m rdsa.train --config configs/qwen3vl.yaml` — RDSA training
- `python -m rdsa.identify --model qwen3vl --output subspaces/` — Safety subspace identification (SVD)
- `python -m rdsa.evaluate --defense rdsa --attack scia --model qwen3vl` — Run evaluation
- `python -m rdsa.evaluate --defense rdsa --attack all --model all` — Full evaluation matrix
- `pytest tests/ -x -q` — Run tests (stop on first failure)
- `pytest tests/test_losses.py -k "sa_at"` — Run specific test
- `python scripts/visualize_entanglement.py --model qwen3vl` — Generate η profile plots
- `ruff check rdsa/` — Lint
- `ruff format rdsa/` — Format

## Project Structure

```
rdsa/
├── CLAUDE.md
├── configs/                    # Hydra configs
│   ├── qwen3vl.yaml
│   ├── gemma3.yaml
│   └── llama32.yaml
├── rdsa/
│   ├── __init__.py
│   ├── config.py               # RDSAConfig dataclass
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hooks.py            # Activation extraction & injection hooks (SA-AT)
│   │   └── model_utils.py      # Model loading, layer access helpers
│   ├── subspace/
│   │   ├── __init__.py
│   │   ├── identifier.py       # SafetySubspaceIdentifier (SVD + PCA)
│   │   └── metrics.py          # Entanglement degree η, LCIV computation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py           # ConsistencyLoss, SubspaceConstrainedATLoss (SA-AT)
│   │   ├── trainer.py          # RDSATrainer (SA-AT multi-objective training loop)
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

**Module 1 — Multi-Layer Subspace-Constrained Adversarial Training (Theorem 1)**
SA-AT: PGD adversarial training within each layer group's safety subspace V_s. For each group, find worst-case perturbation δ* in the d_s-dimensional V_s subspace, then train the model to still refuse under that perturbation. Coverage: every safety-relevant direction within ε-ball is hardened.

**Module 2 — Cross-Layer Safety Redundancy (Theorem 2)**
Distribute safety representations across G layer groups. Attack cost scales from O(ε₁*) to O(√Σ(εₖ*)²) when gradient directions are non-aligned across groups.

### Key Data Structures

- `V_s`: Safety subspace basis, `torch.Tensor` shape `[d, d_s]`, d_s=32 by default
- `V_t`: Semantic subspace basis, `torch.Tensor` shape `[d, d_t]`, d_t=256 by default
- Layer groups (Qwen3-VL-8B): `[[8-12], [16-20], [24-28]]` — skip shallow layers, gap between groups
- `η = (1/kₛ) Σᵢ maxⱼ |vₛ⁽ⁱ⁾ᵀ vₜ⁽ʲ⁾|` — entanglement degree (directly optimized via EntanglementLoss)

### Training Pipeline (3 steps)

1. **Identify** — SVD on activation differences (safe vs unsafe) → V_s per layer group; PCA on normal data → V_t
2. **Train (SA-AT)** — Fine-tune with LoRA (attention + MLP). Total loss: `L_SFT + α_sa_at·L_SA-AT + α_consist·L_consist + α_entangle·L_entangle`
   - **Label masking**: SFT loss computed only on response tokens (prompt tokens masked to -100)
   - **Chat template**: All training data formatted with model-specific chat templates
   - **SA-AT warmup**: Pure SFT for first N epochs before adversarial training begins
   - SA-AT inner loop: Random-restart PGD finds δ* = argmax_{‖δ‖≤ε} L_CE(f(h + V_s·δ), y_refusal) per group
   - SA-AT outer loop: L_SA-AT = L_CE(f(h + V_s·δ*), y_refusal) — train to refuse under worst perturbation
   - **Relative epsilon**: ε scaled by mean activation norm (`ε = ratio × ‖h‖_mean`)
   - Consistency: enforce cross-layer agreement on safety assessments (harmful samples only to reduce over-refusal)
   - **Entanglement**: L_entangle = 1 - η, directly maximizes safety/semantic subspace overlap
   - **Multi-layer hooks**: SA-AT perturbs all layers in each group, not just the representative
   - **Periodic subspace refresh**: V_s/V_t re-identified every N optimizer steps within epochs
3. **Monitor** — Inference anomaly detection via cross-layer safety confidence variance

### Default Hyperparameters

```
d_s=32, d_t=256, G=3, α_sa_at=0.3, α_consist=0.05, α_entangle=0.1, τ=0.2
SA-AT: pgd_steps=7, pgd_alpha=0.1, epsilon=1.0, num_restarts=3, epsilon_relative=true, epsilon_ratio=0.05
SA-AT warmup: 1 epoch, subspace_update_interval=100 steps
LoRA: rank=16, alpha=32, lr=2e-5, epochs=5, targets=[q,k,v,o,gate,up,down]_proj
ConsistencyLoss: harmful_only=true
```

## Coding Conventions

- Type hints on all function signatures. Use `torch.Tensor` not `Tensor`.
- Docstrings: Google style. Include shape annotations for tensor args: `h: torch.Tensor  # [B, d]`
- All tensor operations must specify device explicitly or inherit from input. Never hardcode `.cuda()`.
- Use `@dataclass` for configs, never raw dicts.
- Hook-based activation extraction: always register AND remove hooks in try/finally or context manager.
- Loss functions inherit `nn.Module`. Training utilities are plain classes.
- Tests use `pytest` with `torch.testing.assert_close` for numerical checks. Tolerance: `atol=1e-5, rtol=1e-4`.
- Log all loss components to wandb every step: `{"loss/total": ..., "loss/sft": ..., "loss/sa_at": ..., "loss/consist": ..., "metric/eta": ...}`
- Config override via CLI: `python -m rdsa.train model.name=qwen3vl training.alpha_sa_at=0.3`

## Critical Implementation Notes

- **Hook cleanup is critical.** Leaked hooks cause silent memory leaks and wrong gradients. Use `HookManager` context manager in `rdsa/models/hooks.py`.
- **SVD numerical stability.** Use `torch.linalg.svd(full_matrices=False)`. For large N, compute on CPU then move result to GPU.
- **SA-AT inner loop gradient routing.** Use `torch.autograd.grad(loss, delta)` in PGD inner loop — only delta receives gradients, not model parameters. Run PGD per-group independently to avoid graph severing.
- **SA-AT outer loop uses additive hooks.** `AdditiveInjectionHookManager` adds `V_s @ delta*` to natural hidden states, preserving the full computation graph so gradients flow to ALL LoRA parameters.
- **Mixed precision.** Safety subspace projections (`V_s`, `V_t`) must stay fp32 even under AMP. Cast hidden states before projection. Qwen3-VL and Gemma-3 prefer bf16 over fp16.
- **Memory.** Collecting activations across 3 layer groups for 8B model needs ~6GB extra. Use gradient checkpointing + activation offloading for training. Gemma-3-12B needs smaller batch_size=2.
- **Reproducibility.** Seed everything: `torch.manual_seed`, `numpy.random.seed`, `random.seed`. Log seed in wandb.
- **Gemma-3 attention pattern.** Gemma-3-12B uses alternating 5-layer local sliding window + 1-layer global attention (idx%6==5 is global). Layer group boundaries must include global layers.

## Model-Specific Layer Access

```python
# Qwen3-VL-8B:     model.model.language_model.layers[i]  — "model.language_model.layers"
# Gemma-3-12B:     model.model.language_model.layers[i]  — "model.language_model.layers"
# LLaMA-3.2-Vision: model.model.language_model.layers[i] — "model.language_model.layers"
# InternVL2.5-8B:  model.language_model.model.layers[i]  — "language_model.model.layers"
# MiniCPM-V-2.6:   model.llm.model.layers[i]             — "llm.model.layers"
# Use get_layer(model, idx) helper in model_utils.py — handles all architectures
```

### Layer Group Configurations

```
Qwen3-VL-8B (32 layers, d=4096):
  g1=[8,9,10,11,12]  g2=[16,17,18,19,20]  g3=[24,25,26,27,28]

Gemma-3-12B (48 layers, d=3840):  [requires HF login]
  g1=[12,13,14,15,16,17]  g2=[24,25,26,27,28,29]  g3=[36,37,38,39,40,41]

LLaMA-3.2-11B (32 layers, d=4096):  [requires HF login]
  g1=[8,9,10,11,12]  g2=[16,17,18,19,20]  g3=[24,25,26,27,28]

InternVL2.5-8B (32 layers, d=4096):  [open access]
  g1=[8,9,10,11,12]  g2=[16,17,18,19,20]  g3=[24,25,26,27,28]

MiniCPM-V-2.6 (28 layers, d=3584):  [open access, trust_remote_code]
  g1=[6,7,8,9,10]  g2=[12,13,14,15,16]  g3=[20,21,22,23,24]
```

## Reference Files

- `docs/MATH.md` — Full proofs of Theorem 1 (SA-AT coverage guarantee) and Theorem 2 (cross-layer redundancy attack cost amplification)
- `docs/EXPERIMENT_DESIGN.md` — 7 experiments (main comparison, ablation, SA-AT analysis, redundancy analysis, capability preservation, adaptive attacks, parameter sensitivity)
- SCIA paper: the attack we primarily defend against. Key figures: Fig 3 (layer-wise probing), Algorithm 1 (attack pipeline)
