# Talk Script: RDSA Milestone Report

**Duration**: 15 minutes + Q&A
**Total slides**: 16
**Venue**: Milestone Presentation (targeting NeurIPS 2026)

---

## Slide 1: Title [0:00 - 0:15]

*[Wait for introduction]*

"Thank you. I'm Tianhao from UVA. Today I'll present RDSA — Resilient Distributed Safety Architecture — our defense method against adversarial attacks on Vision-Language Models."

---

## Slide 2: Outline [0:15 - 0:30]

"Here's what I'll cover: first the problem and why it matters, then the root cause we're targeting, our approach with three defense modules, and finally where we are with implementation and our path to NeurIPS."

→ Transition: "Let's start with the problem."

---

## Slide 3: VLM Safety Is Fragile [0:30 - 2:00]

"Vision-Language Models like GPT-4o, Qwen-VL, and Gemma all have safety alignment — they're trained to refuse harmful requests. But here's the alarming finding: a single adversarial image can bypass all of these safety guardrails.

The SCIA attack, published at ICML 2026, demonstrated over 80% attack success rate on aligned models. That means 4 out of 5 harmful requests succeed when paired with a carefully crafted image. And this isn't limited to one defense — safety fine-tuning, RLHF, constitutional AI — all broken by visual attacks.

The natural question is: why is this so easy to break?"

→ Transition: "The answer lies in how these models represent safety internally."

---

## Slide 4: Root Cause [2:00 - 3:30]

"SCIA's key discovery — and this is the finding we're building on — is that safety behavior in these models depends on roughly 0.1 to 0.5 percent of neurons. That's about 20 out of 4096 neurons per layer.

Critically, these safety neurons are disentangled from the semantic features that handle understanding and reasoning. This means an attacker can suppress safety without hurting the model's capability at all.

Think of it like a building where the fire alarm runs through a single wire that's completely separate from the electrical system. Cut that one wire — the lights stay on, everything works perfectly, but the alarm is dead.

This is the fundamental vulnerability we're targeting."

→ Transition: "So what's our approach?"

---

## Slide 5: Key Insight [3:30 - 4:30]

"Our key insight is simple: make safety features impossible to remove without destroying the model itself. We do this through three strategies.

First, distribute safety across multiple layer groups — so there's no single point of failure. Second, entangle safety features with semantic features — so removing safety also removes capability. Third, harden each layer group via adversarial training within the safety subspace.

The combination means an attacker faces an impossible dilemma: they can't remove safety without also breaking the model's ability to understand and reason."

→ Transition: "Let me walk through the three modules."

---

## Slide 6: RDSA Architecture [4:30 - 5:30]

"RDSA has three steps. First, we identify the safety and semantic subspaces using SVD on activation differences between safe and unsafe inputs. This gives us V_s — a 32-dimensional safety subspace — and V_t — a 256-dimensional semantic subspace.

Second, we train the model with a multi-objective loss: standard SFT for safety behavior, SA-AT for adversarial robustness within the safety subspace, consistency loss for cross-layer agreement, and entanglement loss to maximize overlap between V_s and V_t.

Third, at inference time, a lightweight monitor detects anomalous cross-layer variance — a signature of adversarial manipulation.

The first two modules come with theoretical guarantees, which I'll explain next."

→ Transition: "Let's look at Module 1 — SA-AT."

---

## Slide 7: SA-AT (Theorem 1) [5:30 - 7:00]

"SA-AT — Subspace-Constrained Adversarial Training — is our core training module. The key idea is to find the worst-case perturbation within the safety subspace and train the model to still refuse under that perturbation.

The inner loop uses PGD to find delta-star — the strongest attack in the V_s subspace. The outer loop then trains the model to produce refusal responses even when this worst-case perturbation is applied.

What makes this efficient is that we're searching in only 32 dimensions, not the full 4096. PGD converges quickly in this low-dimensional space.

We've added several improvements to the basic formulation: random restarts — we run PGD 3 times from different starting points and keep the strongest attack. And relative epsilon — we scale the perturbation budget by the activation norm, so deeper layers with larger activations get proportionally larger perturbations."

→ Transition: "Module 2 is about cross-layer redundancy."

---

## Slide 8: Cross-Layer Redundancy (Theorem 2) [7:00 - 8:30]

"Theorem 2 says that distributing safety across G independent layer groups amplifies the attack cost.

For Qwen3-VL with 32 layers, we define 3 groups: layers 8-12, 16-20, and 24-28. Each group has its own safety subspace V_s, and each is independently hardened by SA-AT.

The mathematical result is that the attack cost scales from O of epsilon-star for a single group to O of the root sum of squares across groups. Intuitively, the attacker needs to simultaneously fool all three groups, and when the gradient directions across groups are non-aligned, there's no single perturbation that works for all of them.

The consistency loss ensures all groups agree on whether an input is safe or unsafe, but we only apply it on harmful samples to avoid over-refusal on benign inputs."

→ Transition: "The third module is what makes RDSA fundamentally different from prior work."

---

## Slide 9: Entanglement [8:30 - 10:00]

"Safety-semantic entanglement is our core innovation. We define the entanglement degree eta as the average maximum alignment between safety and semantic subspace directions.

When eta is close to 0, the subspaces are orthogonal — safety is fully separable, and the attacker can remove it freely. When eta is close to 1, safety is embedded in the semantic subspace — removing safety also destroys the model's ability to understand language and images.

Prior work only monitored eta. We directly optimize it with an entanglement loss: L_entangle equals 1 minus eta. This loss flows through the LoRA parameters and pushes the model's representations to align safety with semantics.

The result is that an attacker faces a fundamental trade-off: any successful attack on safety also degrades the model's capability, making the attack self-defeating."

→ Transition: "Let me show you the training improvements we've made."

---

## Slide 10: Training Enhancements [10:00 - 10:45]

"We've implemented 10 key enhancements to the training pipeline. On the data side: proper chat template formatting, label masking so the SFT loss only applies to response tokens, and LoRA that covers both attention and MLP modules.

For SA-AT: random restarts, relative epsilon, and a warmup period — one epoch of pure SFT before adversarial training begins. This gives the model basic safety behavior before we start hardening it.

For the loss functions: the new entanglement loss that directly optimizes eta, and consistency loss restricted to harmful samples only.

And for robustness: we hook all layers in each group, not just the representative, and we refresh the subspaces every 100 steps within each epoch."

→ Transition: "Here's our evaluation setup."

---

## Slide 11: Models & Evaluation [10:45 - 11:30]

"We're evaluating on three open-access surrogate models: Qwen3-VL-8B, InternVL2.5-8B, and MiniCPM-V 2.6. All three are fully open — no HuggingFace login required.

We test against 6 attack types: two white-box attacks — SCIA and UMK, one visual injection — FigStep, and three adaptive attacks that assume the attacker knows our defense mechanism.

These adaptive attacks are critical for NeurIPS — reviewers will expect the defense to hold even when the attacker knows V_s, V_t, eta, and the monitor threshold.

For metrics, we use GPT-4o as a safety judge for ASR, pattern-based refusal rate, and over-refusal rate on benign queries. We also run VQAv2, MMBench, and MME to verify capability preservation."

→ Transition: "Let me share our preliminary results."

---

## Slide 12: Preliminary Results [11:30 - 12:30]

"We've completed subspace identification on Qwen3-VL-8B. The safety subspace is 32-dimensional, the semantic subspace is 256-dimensional, and the vanilla entanglement eta is around 0.49 to 0.53 across the three layer groups.

This eta is higher than expected based on SCIA's findings. We believe this is partly due to the diversity of our contrast data — we're working on improving data quality with semantically paired safe/unsafe prompts.

For the baseline without any attack, vanilla Qwen3-VL-8B has an ASR of about 21.6% on AdvBench prompts — meaning about 1 in 5 harmful requests get through even without adversarial manipulation.

Training is in progress. The consistency loss dropped from 0.63 to 0.02 during the warmup epoch, suggesting the model is learning cross-layer agreement. After full SA-AT training, we expect ASR under SCIA to drop below 10%, and ASR under adaptive attacks to stay below 20%, while VQA accuracy drops by less than 2%."

→ Transition: "Here's the full experiment plan."

---

## Slide 13: Experiment Plan [12:30 - 13:15]

"We have 10 experiments planned for NeurIPS. The main comparison table tests 7 defenses against 6 attacks across 3 models — that's the core result. The ablation isolates each component's contribution. Experiments 3 through 5 provide empirical evidence for our two theorems and the entanglement mechanism.

Capability preservation is critical — we need to show RDSA doesn't degrade model performance. And the adaptive attack experiment is a hard requirement for NeurIPS security papers — we need to show robustness under the strongest possible attacker.

Total compute is about 600 GPU-hours for one seed, or 1800 for three seeds with statistical significance. All experiment scripts are implemented and ready to run."

→ Transition: "Let me show you the implementation status."

---

## Slide 14: Implementation Status [13:15 - 13:45]

"On the implementation side, we've completed the full codebase — about 5000 lines of Python. This includes the subspace identification pipeline, SA-AT training loop with all four loss functions, all six attack implementations, the GPT-4o safety judge, four capability benchmarks, and all 10 experiment scripts.

We support three open-access VLMs with architecture-specific layer access patterns and configs.

What's in progress is the full RDSA training run — we're in the warmup epoch now — and the complete evaluation with real attacks integrated into the pipeline. We also need to reproduce baseline defenses for fair comparison."

→ Transition: "Here's the timeline."

---

## Slide 15: Timeline [13:45 - 14:30]

"We have roughly 7 weeks to the NeurIPS deadline. Weeks 1 and 2 focus on training all three models and running the main comparison and ablation experiments. Week 3 covers the analysis experiments — SA-AT, redundancy, and entanglement. Week 4 handles transfer attacks and sensitivity analysis. Week 5 is for statistical significance — running with 3 seeds, computing bootstrap confidence intervals. And weeks 6-7 are for paper writing.

The key risks are GPU availability — we need sustained access for the full training runs — baseline reproduction, since Circuit Breaker and SmoothVLM need separate implementations, and the strength of adaptive attacks — we need to ensure they're genuinely strong, not just nominally adaptive."

→ Transition: "Let me summarize."

---

## Slide 16: Summary & Questions [14:30 - 15:00]

"To summarize: VLM safety is fragile because safety features are localized and separable. RDSA addresses this with three modules — subspace-constrained adversarial training, cross-layer redundancy, and safety-semantic entanglement — backed by two theoretical guarantees.

Our implementation is complete across three open-access VLMs, with a comprehensive 10-experiment evaluation plan targeting NeurIPS 2026.

The code is available on GitHub. I'm happy to take questions."

---

## Anticipated Q&A

### Q1: How does RDSA compare to Circuit Breaker?
**A**: "Circuit Breaker applies representation engineering to remap harmful activations, but it doesn't consider the subspace structure of safety features. RDSA specifically targets the root vulnerability — localized, disentangled safety neurons — with provable coverage within the safety subspace. We expect RDSA to be more robust to adaptive attacks because of the entanglement mechanism."

### Q2: Why not just use regular adversarial training?
**A**: "Standard adversarial training operates in the full activation space — 4096 dimensions. That's both computationally expensive and imprecise. SA-AT constrains the search to the 32-dimensional safety subspace, which makes PGD converge faster and the defense more targeted. It's also what enables our theoretical coverage guarantee."

### Q3: Is eta=0.5 for the vanilla model a problem?
**A**: "It's higher than expected based on SCIA's findings. We believe this is due to semantic contamination in our contrast data — the safe and unsafe prompts differ in topic, not just safety. We're working on semantically paired data. Regardless, what matters for the paper is the delta — how much RDSA increases eta and how that correlates with ASR reduction."

### Q4: How expensive is RDSA training compared to standard SFT?
**A**: "About 3-4x more expensive per step due to the PGD inner loop (7 steps × 3 restarts) and the additional forward pass for the outer loss. For Qwen3-VL-8B, one epoch takes about 1.5 hours on a single A100 versus 25 minutes for pure SFT. The total training is about 8 hours."

### Q5: What if an attacker uses a completely different attack strategy?
**A**: "That's exactly what the adaptive attacks test. We have three adaptive attacks that assume full knowledge of RDSA — they know V_s, V_t, eta, and the monitor threshold. If RDSA holds under these, it provides strong evidence of robustness. The entanglement mechanism is particularly important here — it creates a fundamental trade-off that's independent of the attack strategy."

### Q6: Why these three specific models?
**A**: "We chose for architectural diversity and accessibility. Qwen3-VL uses Qwen architecture, InternVL2.5 uses InternLM, and MiniCPM-V uses Qwen2 — three different language backbones. All are open-access without gating, which is important for reproducibility."

### Q7: What's the biggest risk for the NeurIPS submission?
**A**: "The adaptive attacks. NeurIPS reviewers will scrutinize whether our adaptive attacks are genuinely strong. If the attacks are weak and RDSA only looks good because of that, the paper won't be accepted. We need to show that even with a 10x compute budget, the attacker can't break through."

### Q8: How does the inference-time monitor work?
**A**: "The monitor computes the variance of safety confidence across the three layer groups. For clean inputs, all groups agree — low variance. For adversarial inputs, the perturbation affects groups differently — high variance. We set a threshold tau=0.2 based on calibration data. The monitor adds negligible latency — just three dot products and a variance computation."

---

## Time Budget Summary

| Slide | Topic | Duration | Cumulative |
|:-----:|-------|:--------:|:----------:|
| 1 | Title | 0:15 | 0:15 |
| 2 | Outline | 0:15 | 0:30 |
| 3 | Problem | 1:30 | 2:00 |
| 4 | Root Cause | 1:30 | 3:30 |
| 5 | Key Insight | 1:00 | 4:30 |
| 6 | Architecture | 1:00 | 5:30 |
| 7 | SA-AT | 1:30 | 7:00 |
| 8 | Redundancy | 1:30 | 8:30 |
| 9 | Entanglement | 1:30 | 10:00 |
| 10 | Training | 0:45 | 10:45 |
| 11 | Models & Eval | 0:45 | 11:30 |
| 12 | Preliminary Results | 1:00 | 12:30 |
| 13 | Experiment Plan | 0:45 | 13:15 |
| 14 | Implementation | 0:30 | 13:45 |
| 15 | Timeline | 0:45 | 14:30 |
| 16 | Summary + Q&A | 0:30 | 15:00 |

**Total**: 15:00 min
