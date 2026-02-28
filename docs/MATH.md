# RDSA Mathematical Framework

## Notation

| Symbol | Meaning |
|--------|---------|
| L | Total layers in model |
| G = {g₁,...,gG} | Layer groups (G groups total) |
| d | Hidden dimension |
| dₛ | Safety subspace dimension (default 32) |
| dₜ | Semantic subspace dimension (default 256) |
| Vₛ^(k) ∈ ℝ^{d×dₛ} | Safety subspace basis for group k |
| Vₜ^(k) ∈ ℝ^{d×dₜ} | Semantic subspace basis for group k |
| h_l(x,q) | Hidden state at layer l for image x, query q |
| η | Entanglement degree ∈ [0,1] (monitored, not directly optimized) |
| S_safe^(k) | Safe region in group k's safety subspace |
| δ | Perturbation in dₛ-dimensional subspace |
| ε | L∞ norm bound on subspace perturbation |

## Theorem 1: SA-AT Coverage Guarantee

### Setup

For each layer group k, the safety subspace Vₛ^(k) has dimension dₛ. SA-AT performs PGD adversarial training within this subspace:

**Inner loop (PGD):**
```
δ* = argmax_{‖δ‖_∞ ≤ ε} L_CE(f(h + Vₛ^(k) · δ), y_refusal)
```

**Outer loop (training):**
```
L_SA-AT = L_CE(f(h + Vₛ^(k) · δ*), y_refusal)
```

### Statement

After SA-AT training with ε-bounded PGD, for any perturbation δ in the safety subspace with ‖δ‖_∞ ≤ ε, the model maintains refusal behavior:

```
∀δ ∈ ℝ^{dₛ}, ‖δ‖_∞ ≤ ε:
  L_CE(f(h + Vₛ · δ), y_refusal) ≤ L_CE(f(h + Vₛ · δ*), y_refusal)
```

Since training minimizes L_CE on the worst case δ*, the model is hardened against ALL perturbation directions within the ε-ball.

### Key Property: Low-Dimensional Search Space

The PGD search operates in dₛ = 32 dimensions (not d = 4096). This means:
- PGD converges in few steps (7 steps sufficient empirically)
- The ε-ball in ℝ^{dₛ} covers all safety-relevant directions
- Unlike full-space adversarial training, SA-AT is computationally tractable

### Comparison with Entanglement Approach

Direct η optimization attempted to make safety and semantic subspaces overlap. This failed because:
1. LoRA rank (16) << d (4096): insufficient capacity to rotate subspace geometry
2. SVD-derived V_s is not differentiable w.r.t. model parameters
3. η metric is insensitive to small parameter changes

SA-AT bypasses these issues entirely — instead of trying to change where safety features live, it directly hardens whatever safety subspace exists against worst-case perturbations.

---

## Theorem 2: Attack Cost Amplification via Cross-Layer Redundancy

### Setup

Model has safety representations distributed across G layer groups. Each group has safety detector fₖ(h) = wₖᵀ Πₖ(h), where Πₖ is the projection onto Vₛ^(k).

### Without Redundancy (Current VLMs)

Attacker needs δ s.t. Πₗ*(hₗ*(x+δ,q)) ∉ S_safe for single critical layer l*.

Minimum perturbation:
```
εₖ* = fₖ(h(x,q)) / ‖∇_δ fₖ‖₁
```

### With Redundancy (RDSA)

Attacker must simultaneously bypass ALL groups:
```
∀k ∈ {1,...,G}: Πₖ(h_{lₖ}(x+δ,q)) ∉ S_safe^(k)
```

### Condition 1 (Non-Alignment)
```
cos(∇_δ fᵢ, ∇_δ fⱼ) < 1  for i ≠ j
```

Empirical support: SCIA Figure 3 shows defensive probing peaks at deep layers, transferable probing at middle layers → different layers have different safety gradient directions.

### Result

Under Condition 1:
```
‖δ‖₂² ≥ Σₖ (εₖ*)²
```

Attack cost: O(ε₁*) → O(√Σₖ(εₖ*)²)

### Consistency Loss Reinforcement

The consistency loss L_consist enforces cross-layer agreement:
```
L_consist = Σ_{(k₁,k₂)} (1 - cos(Vₛ^(k₁)ᵀ h_{k₁}, Vₛ^(k₂)ᵀ h_{k₂}))
```

This training objective ensures that all layer groups encode coherent safety signals, making it harder for an attacker to find a single perturbation that suppresses safety in all groups simultaneously.

### Performance Preservation

```
Perf(θ_RDSA) ≥ Perf(θ_orig) - O(dₛ/d) · C
```

When dₛ/d ≈ 32/4096 ≈ 0.8%, safety projections constrain only a tiny subspace. Semantic representations remain free in the orthogonal complement ℝ^{d-dₛ}.

---

## Training Losses

### L_SA-AT (Subspace-Constrained Adversarial Training)

**PGD Inner Loop (per group k):**
```
δ₀ = 0 ∈ ℝ^{dₛ}
For t = 1..T:
  h_perturbed = h_detached + δ_{t-1} @ Vₛ^(k)ᵀ
  loss_t = L_CE(f(h_perturbed), y_refusal)
  g_t = ∇_δ loss_t    (via torch.autograd.grad, no model gradients)
  δ_t = clamp(δ_{t-1} + α · sign(g_t), -ε, ε)
δ* = δ_T
```

**Outer Loss:**
```
L_SA-AT = L_CE(f(h_natural + Vₛ · δ*_detached), y_refusal)
```

Key: h_natural retains computation graph (gradients flow to LoRA); δ* is detached (constant).

Implementation uses AdditiveInjectionHookManager to add V_s @ δ* to natural layer outputs, preserving the full computation graph through all layers.

### L_consist (Cross-Layer Consistency)

```
L_consist = Σ_{(k₁,k₂)} (1 - cos(Vₛ^(k₁)ᵀ h_{lₖ₁}, Vₛ^(k₂)ᵀ h_{lₖ₂}))
```

Enforces all layer groups agree on safety assessment. Hidden states projected to fp32 before subspace projection.

### Total Loss

```
L_total = L_SFT + α_sa_at · L_SA-AT + α_consist · L_consist
```

Default: α_sa_at = 0.3, α_consist = 0.05.

---

## Inference Monitoring

```
Anomaly(x,q) = Var_k[σ(wₖᵀ Vₛ^(k)ᵀ h_{lₖ}(x,q))]
```

High cross-layer variance → adversarial input → trigger conservative mode.

---

## Entanglement Degree (Analysis Metric)

η is computed for monitoring and analysis but not directly optimized:

```
η(Vₛ, Vₜ) = (1/kₛ) Σᵢ maxⱼ |vₛ^(i)ᵀ vₜ^(j)|
```

- η = 0: Safety and semantic subspaces orthogonal (fully disentangleable)
- η = 1: Safety subspace fully embedded in semantic subspace

SA-AT training may indirectly increase η as the model learns to encode safety in semantically relevant directions, but this is a side effect rather than a training objective.
