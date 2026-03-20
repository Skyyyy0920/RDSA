# RDSA NeurIPS Experiment Guide

Complete guide for reproducing all experiments in the RDSA paper.

## Prerequisites

### Hardware
- **Minimum**: 1× A100 80GB (or 2× A100 40GB with DeepSpeed ZeRO-3)
- **Recommended**: 8× A100 80GB for parallel runs
- **Storage**: ~500GB for checkpoints, subspaces, and results

### Software
```bash
# Create environment
conda create -n rdsa python=3.11 -y
conda activate rdsa

# Install PyTorch (CUDA 12.x)
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# Install RDSA and dependencies
pip install -e ".[dev]"

# Verify installation
python -c "import rdsa; print('RDSA OK')"
pytest tests/ -x -q
```

### API Keys
```bash
# For GPT-4o safety judge (Exp 1-9)
export OPENAI_API_KEY="sk-..."

# For commercial model evaluation (Exp 8)
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

### Data Preparation
```bash
# Download and prepare datasets
mkdir -p data/

# 1. AdvBench harmful prompts
wget -O data/advbench.csv "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"

# 2. Safe/unsafe contrast prompts (for subspace identification)
# Generate from StrongReject + safe prompts:
python scripts/prepare_contrast_data.py \
    --harmful-source walledai/StrongReject \
    --safe-source HuggingFaceM4/VQAv2 \
    --output-safe data/safe_prompts.jsonl \
    --output-unsafe data/unsafe_prompts.jsonl \
    --max-samples 2000

# 3. Benchmark datasets (downloaded automatically by evaluators)
# VQAv2, MMBench, MME, OR-Bench
```

---

## Quick Start: Run Everything

```bash
# Dry run first (prints all commands without executing)
python scripts/run_all_experiments.py --model qwen3vl --dry-run

# Run all experiments on one model
python scripts/run_all_experiments.py --model qwen3vl

# Run all experiments on all 3 models
python scripts/run_all_experiments.py --model all

# Resume from a specific experiment
python scripts/run_all_experiments.py --model qwen3vl --start-from 5

# Run only specific experiments
python scripts/run_all_experiments.py --model qwen3vl --only 1 2 6 7
```

---

## Step-by-Step Execution

### Step 0: Subspace Identification

Before training, identify safety and semantic subspaces for each model.

```bash
# Qwen3-VL-8B
python -m rdsa.identify --model qwen3vl --output subspaces/qwen3-vl-8b

# Gemma-3-12B
python -m rdsa.identify --model gemma3 --output subspaces/gemma-3-12b

# LLaMA-3.2-11B-Vision
python -m rdsa.identify --model llama --output subspaces/llama-3.2-11b
```

**Expected output**: `subspaces/{model}/metadata.pt` + `group_{0,1,2}.pt`
**Time**: ~30 min per model on 1× A100

---

### Experiment 1: Main Comparison (Table 1)

**Goal**: Compare RDSA vs 6 baselines across 6 attacks and 3 models.

```bash
# Single model
python scripts/exp1_main_comparison.py --model qwen3vl

# All models (parallel-friendly — can split across GPUs)
python scripts/exp1_main_comparison.py --model qwen3vl &
python scripts/exp1_main_comparison.py --model gemma3 &
python scripts/exp1_main_comparison.py --model llama &
wait

# With specific attacks only
python scripts/exp1_main_comparison.py --model qwen3vl \
    --attacks scia umk figstep adaptive_scia

# With specific seeds
python scripts/exp1_main_comparison.py --model qwen3vl --seeds 42 123 456
```

**External baselines** (Circuit Breaker, SmoothVLM, VLGuard) require separate training. Place pre-trained checkpoints at:
```
checkpoints/baselines/{circuit_breaker,smoothvlm,vlguard}/{qwen3vl,gemma3,llama}/
```

**Output**: `results/exp1_main_comparison/exp1_results.json` + LaTeX tables
**Time**: ~40 GPU-hours per model (training + all evaluations)

---

### Experiment 2: Ablation Study (Table 2)

**Goal**: Isolate contribution of each RDSA component.

```bash
python scripts/exp2_ablation.py --model qwen3vl
```

**Variants**: full, w/o SA-AT, w/o consistency, w/o entanglement, w/o monitor, SFT only
**Attacks**: SCIA + Adaptive-SCIA
**Output**: `results/exp2_ablation/{model}_ablation.json` + LaTeX table
**Time**: ~60 GPU-hours (6 variants × ~8h training + evaluation)

---

### Experiment 3: SA-AT Analysis (Figures 2-3)

**Goal**: Validate SA-AT effectiveness and analyze PGD behavior.

```bash
# All sub-experiments
python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub all

# Individual sub-experiments
python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub pgd_steps
python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub restarts
python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub epsilon
python scripts/exp3_sa_at_analysis.py --model qwen3vl --sub coverage
```

**Sub-experiments**:
- **3a** PGD steps sweep: eval with [1, 3, 5, 7, 10, 20, 50] steps
- **3b** Random restarts: train with [1, 2, 3, 5, 8] restarts
- **3c** Epsilon ratio: train with [0.01, 0.02, 0.05, 0.1, 0.2]
- **3d** Perturbation coverage: analyze δ* distribution in V_s

**Output**: `results/exp3_sa_at/{model}/3{a,b,c,d}_*.json`
**Time**: ~40 GPU-hours

---

### Experiment 4: Cross-Layer Redundancy (Figures 4-5)

**Goal**: Validate Theorem 2 — multi-group redundancy amplifies attack cost.

```bash
python scripts/exp4_redundancy_analysis.py --model qwen3vl --sub all
```

**Sub-experiments**:
- **4a** G=1,2,3 attack cost comparison
- **4b** Gradient alignment measurement
- **4c** t-SNE of cross-layer safety projections

**Note**: 4b and 4c are post-hoc analyses that generate config files describing the analysis procedure. They require running the analysis inline (see the generated config JSONs for detailed steps).

**Output**: `results/exp4_redundancy/{model}/`
**Time**: ~20 GPU-hours

---

### Experiment 5: Entanglement Analysis (Figure 6)

**Goal**: Show that entanglement (η) is the core defense mechanism.

```bash
python scripts/exp5_entanglement_analysis.py --model qwen3vl --sub all
```

**Sub-experiments**:
- **5a** η profile before/after training (uses `scripts/visualize_entanglement.py`)
- **5b** η vs ASR scatter: train with α_entangle ∈ [0, 0.01, ..., 1.0]
- **5c** Manipulable dimensions: theory vs measurement
- **5d** Semantic impact: VQA on attacked samples

**Key result**: η and ASR should show strong negative correlation (r < -0.8).

**Output**: `results/exp5_entanglement/{model}/`
**Time**: ~15 GPU-hours (5b dominates)

---

### Experiment 6: Capability Preservation (Table 4)

**Goal**: Prove RDSA doesn't degrade model capability.

```bash
# Depends on Exp 1 checkpoints
python scripts/exp6_capability.py --model qwen3vl

# Specific benchmarks only
python scripts/exp6_capability.py --model qwen3vl --benchmarks vqav2 mmbench
```

**Benchmarks**: VQAv2, MMBench, MME (Perception+Cognition), OR-Bench
**Target**: VQA drop < 2%, OR < 5%

**Output**: `results/exp6_capability/exp6_capability.json` + LaTeX table
**Time**: ~30 GPU-hours (4 benchmarks × 7 defenses × 3 models)

---

### Experiment 7: Adaptive Attack Robustness (Table 5)

**Goal**: NeurIPS requirement — defense must hold under adaptive attacks.

```bash
# Standard + strong budget
python scripts/exp7_adaptive_attacks.py --model qwen3vl --budget both

# Strong budget only (10× PGD steps, 10 restarts)
python scripts/exp7_adaptive_attacks.py --model qwen3vl --budget strong
```

**Attack settings**:
| Budget | PGD Steps | Restarts | Description |
|--------|-----------|----------|-------------|
| standard | 7 | 3 | Normal evaluation |
| strong | 70 | 10 | Strongest attacker (10× compute) |

**Adaptive attacks**:
- **Adaptive-SCIA**: anti-entanglement objective (knows V_s, V_t, η)
- **Adaptive-PGD**: simultaneous multi-layer attack (knows all group V_s)
- **Monitor-Evasion**: variance-matching constraint (knows monitor threshold)

**Target**: ASR < 20% even under strong budget

**Output**: `results/exp7_adaptive/exp7_adaptive.json` + LaTeX table
**Time**: ~40 GPU-hours

---

### Experiment 8: Transfer Attack (Table 6)

**Goal**: Black-box evaluation via adversarial transferability.

```bash
# Single surrogate
python scripts/exp8_transfer.py --surrogate qwen3vl

# Skip commercial APIs (saves API costs)
python scripts/exp8_transfer.py --surrogate qwen3vl --skip-commercial

# All surrogates
python scripts/exp8_transfer.py --surrogate all
```

**Transfer matrix**:
```
Surrogate (white-box)     →  Victim (black-box)
Qwen3-VL-8B              →  Qwen3-VL-30B, Gemma-27B, LLaMA-11B
Gemma-3-12B               →  Gemma-27B, Qwen-30B, LLaMA-11B
LLaMA-3.2-11B             →  LLaMA-11B, Qwen-30B, Gemma-27B

Cross-architecture + Commercial:
Any surrogate             →  GPT-4o, Gemini-2.5-Flash, Claude-Sonnet-4.5
```

**Output**: `results/exp8_transfer/exp8_transfer.json` + LaTeX table
**Time**: ~30 GPU-hours + ~$200 API costs
**Note**: Commercial API evaluation requires API keys (see Prerequisites).

---

### Experiment 9: Parameter Sensitivity (Figure 7, Appendix)

**Goal**: Show RDSA is robust to hyperparameter choices.

```bash
# All parameters
python scripts/exp9_sensitivity.py --model qwen3vl --param all

# Specific parameter
python scripts/exp9_sensitivity.py --model qwen3vl --param alpha_sa_at
python scripts/exp9_sensitivity.py --model qwen3vl --param d_safe
python scripts/exp9_sensitivity.py --model qwen3vl --param lora_rank
```

**Swept parameters**:
| Parameter | Values | Default |
|-----------|--------|---------|
| α_sa_at | 0.05, 0.1, 0.3, 0.5, 1.0 | 0.3 |
| α_entangle | 0.01, 0.05, 0.1, 0.5, 1.0 | 0.1 |
| α_consist | 0.01, 0.05, 0.1, 0.5 | 0.05 |
| d_s | 8, 16, 32, 64, 128 | 32 |
| τ (threshold) | 0.1, 0.15, 0.2, 0.3, 0.5 | 0.2 |
| G (groups) | 1, 2, 3, 4 | 3 |
| LoRA rank | 4, 8, 16, 32 | 16 |

**Output**: `results/exp9_sensitivity/{model}/{param}_sweep.json`
**Time**: ~240 GPU-hours (30+ configs, can parallelize heavily)

---

### Experiment 10: Pareto Frontier (Figure 1)

**Goal**: Show RDSA pushes the safety-utility Pareto frontier.

```bash
# Requires results from Exp 1 and Exp 9
python scripts/exp10_pareto.py --model qwen3vl
python scripts/exp10_pareto.py --model all --format pdf
```

**Output**: `results/exp10_pareto/pareto_{model}.pdf`
**Time**: Minutes (aggregation + plotting only)

---

## Collecting Results

After running experiments, collect all results into paper-ready format:

```bash
# Generate summary + LaTeX tables
python scripts/collect_results.py --model qwen3vl --output-dir paper/tables/

# For all models
for model in qwen3vl gemma3 llama; do
    python scripts/collect_results.py --model $model --output-dir paper/tables/
done
```

**Output**:
- `paper/tables/{model}_all_results.json` — Full JSON
- `paper/tables/{model}_summary.txt` — Human-readable summary
- Individual `.tex` files from each experiment script

---

## Statistical Significance Protocol

All reported numbers follow this protocol:

1. **3 seeds**: Every experiment runs with seeds {42, 123, 456}
2. **Reported format**: mean ± std across seeds
3. **Bootstrap CI**: For key results (Table 1 main comparison), compute 95% bootstrap confidence intervals on ASR
4. **McNemar test**: For per-sample refusal comparison between RDSA and best baseline

```python
# Example: compute bootstrap CI
import numpy as np

def bootstrap_ci(values, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval."""
    boot_means = [
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ]
    alpha = (1 - ci) / 2
    return np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])
```

---

## Resource Planning

### GPU Hours Summary
| Experiment | GPU-hours | Priority | Parallelizable |
|-----------|-----------|----------|----------------|
| Exp 0: Identify | 1.5 | Required | Yes (per model) |
| Exp 1: Main | 120 | P0 | Yes (per model) |
| Exp 2: Ablation | 60 | P0 | Yes (per variant) |
| Exp 3: SA-AT | 40 | P1 | Yes (per sub-exp) |
| Exp 4: Redundancy | 20 | P1 | Partially |
| Exp 5: Entanglement | 15 | P1 | Yes (per α) |
| Exp 6: Capability | 30 | P0 | Yes (per benchmark) |
| Exp 7: Adaptive | 40 | P0 | Yes (per attack) |
| Exp 8: Transfer | 30 + API | P1 | Yes (per pair) |
| Exp 9: Sensitivity | 240 | P2 | Yes (per param) |
| Exp 10: Pareto | 0 | P0 | N/A |
| **Total (1 seed)** | **~597** | | |
| **Total (3 seeds)** | **~1,790** | | |

### With 8× A100 Cluster
```
Week 1: Exp 0 + Exp 1 + Exp 6    →  Tables 1, 4
Week 2: Exp 2 + Exp 7             →  Tables 2, 5
Week 3: Exp 3 + Exp 4 + Exp 5    →  Figures 2-6
Week 4: Exp 8 + Exp 9             →  Table 6, Figure 7
Week 5: Re-runs, statistics, Exp 10  →  Figure 1 + polish
```

### Estimated API Costs
| Service | Usage | Cost |
|---------|-------|------|
| OpenAI GPT-4o (judge) | ~50,000 judgments | ~$150 |
| OpenAI GPT-4o (Exp 8 victim) | ~500 generations | ~$20 |
| Google Gemini (Exp 8) | ~500 generations | ~$10 |
| Anthropic Claude (Exp 8) | ~500 generations | ~$20 |
| **Total** | | **~$200** |

---

## Troubleshooting

### Common Issues

**OOM during training** (especially Gemma-3-12B):
```bash
# Reduce batch size and increase gradient accumulation
python -m rdsa.train --config configs/gemma3.yaml \
    training.per_device_batch_size=1 \
    training.gradient_accumulation_steps=8
```

**Subspace identification takes too long**:
```bash
# Reduce sample count
python -m rdsa.identify --model qwen3vl --max-samples 500
```

**GPT-4o judge rate limiting**:
- The judge has built-in rate limiting (0.5s between calls)
- For large batches, consider increasing `--rate-limit-delay`

**Hook memory leaks**:
- Always use context managers (`with HookManager(...) as hm:`)
- If training memory grows over time, check for leaked hooks

### Verifying Results

After running experiments, sanity-check key metrics:

```bash
# Check that RDSA ASR < vanilla ASR on all attacks
python -c "
from scripts.experiment_utils import load_results
data = load_results('results/exp1_main_comparison/exp1_results.json')
model = 'qwen3vl'
for attack in data.get(model, {}).get('rdsa', {}):
    rdsa_asr = data[model]['rdsa'][attack].get('asr', {}).get('mean', 1)
    vanilla_asr = data[model]['vanilla'][attack].get('asr', {}).get('mean', 0)
    status = 'OK' if rdsa_asr < vanilla_asr else 'FAIL'
    print(f'{attack}: vanilla={vanilla_asr:.3f}, rdsa={rdsa_asr:.3f} [{status}]')
"
```

---

## Paper Figure/Table Checklist

| Item | Source | Script |
|------|--------|--------|
| Figure 1: Pareto frontier | Exp 1 + 9 | `exp10_pareto.py` |
| Figure 2: SA-AT analysis (3 subfigs) | Exp 3a,b,c | `exp3_sa_at_analysis.py` |
| Figure 3: Perturbation coverage | Exp 3d | Manual analysis |
| Figure 4: G=1,2,3 attack cost | Exp 4a | `exp4_redundancy_analysis.py` |
| Figure 5: Gradient alignment + t-SNE | Exp 4b,c | Manual analysis |
| Figure 6: η analysis (3 subfigs) | Exp 5a,b,c | `exp5_entanglement_analysis.py` |
| Figure 7: Sensitivity (Appendix) | Exp 9 | `exp9_sensitivity.py` |
| Table 1: Main comparison | Exp 1 | `exp1_main_comparison.py` |
| Table 2: Ablation | Exp 2 | `exp2_ablation.py` |
| Table 3: Semantic impact | Exp 5d | Manual analysis |
| Table 4: Capability | Exp 6 | `exp6_capability.py` |
| Table 5: Adaptive attacks | Exp 7 | `exp7_adaptive_attacks.py` |
| Table 6: Transfer matrix | Exp 8 | `exp8_transfer.py` |
