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
| Πₖ | Projection to safety subspace of group k |
| h_l(x,q) | Hidden state at layer l for image x, query q |
| η | Entanglement degree ∈ [0,1] |
| S_safe^(k) | Safe region in group k's safety subspace |

## Theorem 1: Attack Cost Amplification via Safety Redundancy

### Setup

Model has safety representations distributed across G layer groups. Each group has safety detector fₖ(h) = wₖᵀ Πₖ(h).

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

### Performance Preservation

```
Perf(θ_RDSA) ≥ Perf(θ_orig) - O(dₛ/d) · C
```

When dₛ/d ≈ 32/4096 ≈ 0.8%, safety projections constrain only a tiny subspace. Semantic representations remain free in the orthogonal complement ℝ^{d-dₛ}.

---

## Theorem 2: Attack Infeasibility Under Entanglement

### Entanglement Degree

```
η(Vₛ, Vₜ) = (1/kₛ) Σᵢ maxⱼ |vₛ^(i)ᵀ vₜ^(j)|
```

- η = 0: Safety and semantic subspaces orthogonal (fully disentangleable, vulnerable)
- η = 1: Safety subspace fully embedded in semantic subspace (maximally entangled)

### Attacker Constraints

1. Safety bypass: Vₛᵀ Δh must be large
2. Semantic preservation: ‖Vₜᵀ Δh‖₂ ≤ γ

### Proof (η = 1 case)

When η = 1, col(Vₛ) ⊆ col(Vₜ), therefore:
```
Vₛᵀ Δh = Vₛᵀ Vₜ Vₜᵀ Δh
```

Taking norms:
```
‖Vₛᵀ Δh‖₂ ≤ ‖Vₛᵀ Vₜ‖_op · ‖Vₜᵀ Δh‖₂ ≤ ‖Vₜᵀ Δh‖₂ ≤ γ
```

**Conclusion:** If attacker preserves semantics (‖Vₜᵀ Δh‖ ≤ γ), safety shift is also bounded by γ. Cannot selectively suppress safety without destroying semantics.

### Partial Entanglement (η < 1)

Attackable degrees of freedom = kₛ · (1 - η)

Maximizing η directly minimizes attacker's manipulable space.

### Connection to SCIA

SCIA succeeds precisely because current VLMs have low η — safety circuits are disentangleable from semantic circuits (the paper's core finding). RDSA training increases η to eliminate this vulnerability.

---

## Training Losses

### L_entangle (Entanglement Loss)

```
L_entangle = -(1/|G|) Σₖ (1/dₛ) Σᵢ maxⱼ |(vₛ^(k,i))ᵀ vₜ^(k,j)|
```

Maximizes alignment between each safety direction and its closest semantic direction.

### L_consist (Cross-Layer Consistency)

```
L_consist = Σ_{(k₁,k₂)} (1 - cos(Vₛ^(k₁)ᵀ h_{lₖ₁}, Vₛ^(k₂)ᵀ h_{lₖ₂}))
```

Enforces all layer groups agree on safety assessment.

### L_LAT-sub (Subspace-Aware LAT)

```
h_adv = h + Vₛ^(k) · ε,  where ε ~ Uniform(-α, α)
L_LAT-sub = 𝔼_ε[L_safety(h_adv)]
```

Adversarial training restricted to safety subspace.

### Total Loss

```
L_total = L_SFT/DPO + α₁·L_entangle + α₂·L_consist + α₃·L_LAT-sub
```

---

## Inference Monitoring

```
Anomaly(x,q) = Var_k[σ(wₖᵀ Vₛ^(k)ᵀ h_{lₖ}(x,q))]
```

High cross-layer variance → adversarial input → trigger conservative mode.
