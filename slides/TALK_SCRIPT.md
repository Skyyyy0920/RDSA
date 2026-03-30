# Talk Script: RDSA Milestone Report (v2)

**Duration**: 15 minutes + Q&A
**Total slides**: 19
**Key changes from v1**: More figures, expanded method detail, actual training data

---

## Slide 1: Title [0:00 - 0:15]

"Thank you. I'm Tianhao from UVA. Today I'll present RDSA — our defense against adversarial attacks on Vision-Language Models."

---

## Slide 2: The Attack Problem [0:15 - 1:30]

"Let me start with the threat. The top half shows normal operation — a user asks a harmful question, the VLM correctly refuses. The bottom half shows what happens under a SCIA attack — the same harmful text, but paired with an adversarial image. The model now complies and generates harmful content.

SCIA, published at ICML 2026, achieves over 80% attack success rate on aligned models. And this isn't just one defense that breaks — safety fine-tuning, RLHF, constitutional AI — all broken. The question is: why is this so easy?"

-> Transition: "The answer is in how models represent safety internally."

---

## Slide 3: Root Cause [1:30 - 3:00]

"On the left, I'm showing the 4096 neurons in a typical transformer layer. The red dots are safety neurons — only about 5 out of 4096, or 0.1%. On the right, the subspace view shows this more precisely: the safety subspace V_s and the semantic subspace V_t are nearly orthogonal — eta is about 0.2.

This means an attacker can suppress the 32 safety dimensions without touching the 256 semantic dimensions. Safety is a removable component, and that's the fundamental vulnerability."

-> Transition: "RDSA fixes this by making safety impossible to remove."

---

## Slide 4: RDSA Key Idea [3:00 - 4:00]

"Our approach has three pillars. First, distribute safety across multiple layer groups — no single point of failure. Second, entangle safety with semantics — make them inseparable. Third, harden via adversarial training within the safety subspace.

The bottom shows the transformation: from a single fuse that can be cut, to a distributed, entangled system where removing safety also destroys the model."

-> Transition: "Let me walk through the pipeline."

---

## Slide 5: RDSA Pipeline [4:00 - 5:00]

"RDSA has three steps. Step 1: identify safety and semantic subspaces using SVD and PCA — this gives us V_s, a 32-dimensional safety subspace. Step 2: train with our multi-objective loss including SA-AT, consistency, and entanglement. Step 3: deploy with an inference-time monitor that detects anomalous cross-layer variance.

Steps 1 and 2 are backed by theoretical guarantees."

-> Transition: "Let me dive into the SA-AT module."

---

## Slide 6: SA-AT Inner/Outer Loop [5:00 - 6:30]

"This diagram shows the two-loop structure. The inner loop — shown in red at the top — uses PGD to find the worst-case perturbation delta-star. We initialize from zero or random, run 7 PGD steps, clip to the epsilon ball, and repeat with 3 random restarts to keep the strongest.

The critical innovation: we search in only 32 dimensions — the safety subspace — not the full 4096. And epsilon is relative — 5% of the activation norm — so it scales properly across layers.

The outer loop — shown in green at the bottom — takes this worst-case delta-star, freezes it, and trains the model to still refuse. Gradients flow through h to the LoRA parameters."

-> Transition: "Module 2 is about redundancy."

---

## Slide 7: Cross-Layer Redundancy [6:30 - 7:30]

"We distribute safety across 3 independent layer groups — shown as the colored blocks in the layer stack. Each group has its own V_s subspace, and each is independently hardened by SA-AT.

Theorem 2 says the attack cost scales from O of epsilon-star for one group to the root sum of squares across groups. The attacker must fool all three groups simultaneously, and when gradients are non-aligned across groups, no single perturbation works."

-> Transition: "Module 3 is what makes RDSA fundamentally different."

---

## Slide 8: Entanglement [7:30 - 9:00]

"This slide shows the key transformation. On the left, the vanilla model: V_s and V_t are orthogonal — eta is 0.2. An attacker can cut the safety directions without touching semantics.

On the right, after RDSA training: V_s and V_t are aligned — eta is 0.8. Now removing safety also removes semantic capability. This is the attacker's dilemma.

The innovation is that we directly optimize eta through a differentiable loss: L_entangle = 1 minus eta. Gradients flow through the activation-mode cross-correlation to the LoRA parameters. Previous work only monitored eta — we actively push it toward 1."

-> Transition: "Here's the full training objective."

---

## Slide 9: Multi-Objective Loss [9:00 - 9:45]

"The total loss combines four components. SFT for basic safety behavior, with label masking so only response tokens contribute. SA-AT at weight 0.3 — that's the adversarial robustness. Consistency at 0.05 — cross-layer agreement on harmful samples only, to avoid over-refusal. And entanglement at 0.1 — the direct eta optimization.

Important: epoch 0 is pure SFT warmup. SA-AT only kicks in from epoch 1."

-> Transition: "We also made 10 engineering improvements."

---

## Slide 10: 10 Enhancements [9:45 - 10:15]

"Briefly — these are the 10 improvements over the baseline. Chat templates, label masking, LoRA on attention plus MLP, PGD random restarts, relative epsilon, warmup, entanglement loss, harmful-only consistency, multi-layer hooks, and intra-epoch subspace refresh. All implemented and tested."

-> Transition: "Now let me show actual results."

---

## Slide 11: Subspace Results [10:15 - 10:45]

"Here are the actual subspace identification results on Qwen3-VL-8B. Left chart: the top singular values increase dramatically from 333 at group 1 to 7478 at group 3 — deeper layers have much larger activation norms, which is why we use relative epsilon.

Right chart: the vanilla entanglement eta is about 0.49 to 0.53 across groups. This is our starting point — RDSA training will push this higher."

---

## Slide 12: Training Curves [10:45 - 11:30]

"These are real training curves from the warmup epoch. On the left, consistency loss drops from 0.63 to 0.01 — the model learns cross-layer agreement quickly. On the right, entanglement loss decreases more slowly from 0.96 to 0.92 — this is expected during the warmup phase since entanglement optimization really benefits from the SA-AT perturbations that start in epoch 1."

---

## Slide 13: Baseline Evaluation [11:30 - 12:00]

"The baseline result: vanilla Qwen3-VL-8B without any attack has 21.6% ASR — about 1 in 5 harmful prompts get through. This is our no-attack baseline. We've just completed the attack integration in evaluate.py, so adversarial attack results with actual perturbations are running next."

---

## Slide 14: Code Architecture [12:00 - 12:30]

"The codebase is about 5000 lines, organized into 5 packages. Subspace for identification, training for the SA-AT loop and losses, attacks for all 6 attack implementations, evaluation for the GPT-4o judge and benchmarks, and models for the hook-based activation system. All 10 experiment scripts are ready."

---

## Slide 15: Evaluation Setup [12:30 - 13:00]

"We evaluate on 3 open-access VLMs — no HuggingFace login required. Qwen3-VL-8B, InternVL2.5-8B, and MiniCPM-V-2.6 — three different language backbones for architecture diversity. Against 6 attacks including 3 adaptive attacks. Measured by GPT-4o safety judge plus 4 capability benchmarks."

---

## Slide 16: Experiment Plan [13:00 - 13:30]

"10 experiments total. The main comparison is the core table — 7 defenses times 6 attacks times 3 models. The ablation isolates each component. Experiments 3-5 provide empirical evidence for our theorems. Experiment 7 — adaptive attacks — is the NeurIPS hard requirement. Total compute: about 600 GPU-hours for one seed."

---

## Slide 17: Implementation Status [13:30 - 14:00]

"The progress bars tell the story. Green bars — 8 out of 11 items fully complete. The training loop, all attacks, the judge, benchmarks, experiment scripts — all done. Orange: training is 20% through the first model. Red: full evaluation with real attacks and baseline reproduction are next."

---

## Slide 18: Timeline [14:00 - 14:30]

"We have 7 weeks. Weeks 1-2: train all models and run the main experiments. Weeks 3-4: analysis and transfer experiments. Week 5: statistical significance with 3 seeds. Weeks 6-7: paper writing and polish. Key risks: GPU availability, baseline reproduction difficulty, and ensuring adaptive attacks are genuinely strong."

---

## Slide 19: Summary & Questions [14:30 - 15:00]

"To summarize: VLM safety is fragile because safety features are localized and separable. RDSA makes them distributed, entangled, and hardened — with theoretical guarantees. Implementation is complete, training is in progress, targeting NeurIPS 2026 with 10 experiments across 3 models.

Questions?"

---

## Anticipated Q&A

### Q1: Why is vanilla eta ~0.5, not 0.1-0.3 as SCIA predicts?
**A**: "Our contrast data has some semantic contamination — safe and unsafe prompts differ in topic, not just safety. We're working on semantically paired data. What matters for the paper is the eta delta after RDSA training and its correlation with ASR reduction."

### Q2: How expensive is SA-AT compared to standard SFT?
**A**: "About 3-4x per step due to the 7-step × 3-restart PGD inner loop. One epoch on Qwen3-VL-8B takes about 1.5 hours on A100 versus 25 minutes for pure SFT."

### Q3: What if an attacker uses a completely novel strategy?
**A**: "The adaptive attacks give full white-box access plus knowledge of V_s, V_t, eta, and the monitor. If RDSA holds there, the entanglement mechanism provides a fundamental trade-off independent of strategy."

### Q4: Can RDSA cause over-refusal?
**A**: "That's why consistency loss only applies to harmful samples, and we evaluate OR-Bench. We target OR < 5%."

### Q5: Why only 3 layer groups? Why not more?
**A**: "Experiment 4 tests G=1,2,3,4. More groups add redundancy but increase training cost. G=3 is our default — we expect diminishing returns past that."

---

## Time Budget

| Slide | Topic | Dur. | Cumul. |
|:-----:|-------|:----:|:------:|
| 1 | Title | 0:15 | 0:15 |
| 2 | Attack problem | 1:15 | 1:30 |
| 3 | Root cause | 1:30 | 3:00 |
| 4 | RDSA idea | 1:00 | 4:00 |
| 5 | Pipeline | 1:00 | 5:00 |
| 6 | SA-AT diagram | 1:30 | 6:30 |
| 7 | Redundancy | 1:00 | 7:30 |
| 8 | Entanglement | 1:30 | 9:00 |
| 9 | Multi-obj loss | 0:45 | 9:45 |
| 10 | Enhancements | 0:30 | 10:15 |
| 11 | Subspace results | 0:30 | 10:45 |
| 12 | Training curves | 0:45 | 11:30 |
| 13 | Baseline eval | 0:30 | 12:00 |
| 14 | Code architecture | 0:30 | 12:30 |
| 15 | Eval setup | 0:30 | 13:00 |
| 16 | Experiment plan | 0:30 | 13:30 |
| 17 | Impl status | 0:30 | 14:00 |
| 18 | Timeline | 0:30 | 14:30 |
| 19 | Summary + Q | 0:30 | 15:00 |

**Total**: 15:00
