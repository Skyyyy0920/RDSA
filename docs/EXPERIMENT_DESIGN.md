# RDSA: Resilient Distributed Safety Architecture
## 完整实验设计方案

---

## 一、研究问题与实验映射

| 编号 | 研究问题 (RQ) | 对应实验 | 验证目标 |
|------|-------------|---------|---------|
| RQ1 | RDSA能否有效防御多种类型的攻击？ | Exp 1: 主实验 | 普适性 |
| RQ2 | 各模块的贡献如何？ | Exp 2: 消融实验 | 必要性 |
| RQ3 | 安全-语义纠缠是否真的提升了攻击难度？ | Exp 3: 纠缠度分析 | 机制验证 |
| RQ4 | 多层冗余是否有效增加攻击代价？ | Exp 4: 冗余分析 | 机制验证 |
| RQ5 | RDSA是否保持模型正常能力？ | Exp 5: 能力保持 | 实用性 |
| RQ6 | RDSA能否抵抗自适应攻击？ | Exp 6: 自适应攻击 | 鲁棒性上界 |
| RQ7 | 关键超参数如何影响安全-能力平衡？ | Exp 7: 参数敏感性 | 可调节性 |

---

## 二、实验基础设施

### 2.1 模型选择

#### 训练模型（应用RDSA防御的模型）
| 模型 | 参数量 | 选择理由 |
|------|--------|---------|
| LLaVA-v1.6-Mistral-7B | 7B | SCIA论文的主surrogate，直接对比 |
| Qwen2.5-VL-7B-Instruct | 7B | SCIA论文的第二surrogate，架构不同 |
| LLaMA-3.2-11B-Vision-Instruct | 11B | 验证对更大模型的适用性 |

#### 攻击评估目标模型
- 开源：上述三个模型（互为攻击目标） + InstructBLIP-7B, MiniGPT4-7B
- 闭源：GPT-4o, Gemini-2.5-Flash, Claude-Sonnet-4.5（仅用于Exp 1的扩展评估）

### 2.2 攻击方法覆盖

为证明普适性，覆盖 **五大攻击范式**：

| 范式 | 攻击方法 | 类型 | 来源 |
|------|---------|------|------|
| **白盒梯度攻击** | UMK | 梯度优化 | Wang et al., 2024 |
| | VAJM | 视觉对抗越狱 | Qi et al., 2024 |
| **迁移攻击** | SCIA | 电路干预 | 本文对标论文 |
| | FORCE | 特征过度依赖纠正 | Lin et al., 2025 |
| | UltraBreak | 通用迁移越狱 | Anonymous, 2026 |
| | CoA | 攻击链 | Xie et al., 2025 |
| **手动视觉攻击** | FigStep | 排版视觉提示 | Gong et al., 2025 |
| | MM-SafetyBench | 多模态安全基准 | Liu et al., 2024 |
| | Hades | 图像漏洞利用 | Li et al., 2024 |
| **表征层面攻击** | Refusal Direction Ablation | 拒绝方向消融 | Arditi et al., 2024 |
| | Activation Steering Attack | 激活转向攻击 | 基于RepE的攻击 |
| **自适应攻击** | Adaptive-SCIA | 知道RDSA的SCIA | 我们自己设计 |
| | Adaptive-PGD | 知道RDSA的PGD | 我们自己设计 |

### 2.3 数据集

| 数据集 | 用途 | 规模 |
|--------|------|------|
| AdvBench | 主评估（有害行为指令） | 520条 |
| HarmBench | 补充评估（标准化红队框架） | 200条 |
| NIPS 2017 Adversarial | 对抗图像源 | 1000张 |
| VQAv2 | 安全子空间训练 + 能力评估 | 标准划分 |
| MMBench | 能力保持评估 | 标准划分 |
| MME | 能力保持评估 | 标准划分 |
| LLaVA-Bench | 对话质量评估 | 标准划分 |
| OR-Bench | Over-refusal评估 | 标准划分 |
| StrongReject | 安全训练数据 + 评估 | 标准划分 |

### 2.4 评估指标

| 指标 | 定义 | 方向 |
|------|------|------|
| **ASR (Attack Success Rate)** | 攻击成功率（GPT-4o judge） | ↓ 越低越好 |
| **RR (Refusal Rate on harmful)** | 对有害请求的拒绝率 | ↑ 越高越好 |
| **OR (Over-Refusal Rate)** | 对无害请求的错误拒绝率 | ↓ 越低越好 |
| **VQA Acc** | VQAv2准确率 | ↑ 保持 |
| **MMB Score** | MMBench得分 | ↑ 保持 |
| **MME Score** | MME感知+认知得分 | ↑ 保持 |
| **η (Entanglement Degree)** | 安全-语义纠缠度 | ↑ 训练目标 |
| **LCIV (Layer Consistency)** | 跨层安全一致性方差 | ↓ 训练目标 |

### 2.5 Baseline防御方法

| 方法 | 类型 | 来源 |
|------|------|------|
| Vanilla (无防御) | - | - |
| Safety SFT | 训练时 | Zong et al., 2024 |
| Circuit Breakers (CB) | 训练时/表征 | Zou et al., 2024 |
| Latent Adversarial Training (LAT) | 训练时/对抗 | Sheshadri et al., 2025 |
| JPEG Compression | 推理时/预处理 | Standard |
| SmoothVLM | 推理时/预处理 | Adapted from SmoothLLM |
| SafeSteer | 推理时/激活转向 | 2025 |
| RDSA (Ours) | 训练时+推理时 | - |

---

## 三、实验详细设计

### Exp 1: 主实验 — 多攻击范式防御效果 (RQ1)

**目标**：证明RDSA在所有攻击范式下均优于现有防御。

**实验矩阵**：

```
行：8种防御方法 (Vanilla, Safety SFT, CB, LAT, JPEG, SmoothVLM, SafeSteer, RDSA)
列：12种攻击方法 (跨5种攻击范式)
值：ASR (%)
```

**具体步骤**：
1. 对每个训练时防御方法，在LLaVA-v1.6和Qwen2.5-VL上分别训练
2. 对每种攻击方法，生成对抗样本
3. 在所有防御模型上评估ASR
4. 使用GPT-4o作为judge（复用SCIA的judge prompt）

**结果呈现**：
- Table 1（核心结果表）：完整的 防御×攻击 矩阵，包含Avg列
- 按攻击范式分组，展示RDSA在每个范式下的表现

**统计检验**：对每个攻击方法运行3次，报告均值±标准差，用Wilcoxon signed-rank test检验显著性。

---

### Exp 2: 消融实验 (RQ2)

**目标**：验证每个模块的贡献。

**消融变体**：
| 变体 | L_entangle | L_consist | L_LAT-sub | 推理时监测 |
|------|-----------|-----------|-----------|-----------|
| RDSA-Full | ✓ | ✓ | ✓ | ✓ |
| w/o Entanglement | ✗ | ✓ | ✓ | ✓ |
| w/o Consistency | ✓ | ✗ | ✓ | ✓ |
| w/o SubLAT | ✓ | ✓ | ✗ | ✓ |
| w/o Monitor | ✓ | ✓ | ✓ | ✗ |
| Training-only | ✓ | ✓ | ✓ | ✗ |
| Monitor-only | ✗ | ✗ | ✗ | ✓ |

**评估**：在3种代表性攻击（SCIA-迁移, UMK-白盒, FigStep-手动）上评估ASR，同时记录VQA Acc和OR Rate。

**结果呈现**：Table 2，以及各模块贡献的雷达图。

---

### Exp 3: 纠缠度分析 (RQ3)

**目标**：验证安全-语义纠缠确实提升了攻击难度。

**实验3a：纠缠度测量**
- 在Vanilla模型和RDSA模型上，分别计算每一层的纠缠度η
- 绘制layer-wise entanglement degree曲线
- 对比训练前后的变化

**实验3b：纠缠度与ASR的相关性**
- 通过调节α_1（纠缠损失权重），训练5个不同纠缠度的模型
- 在每个模型上评估SCIA的ASR
- 绘制η vs ASR的散点图，验证负相关

**实验3c：攻击者可操纵空间分析**
- 在每个模型上，计算安全子空间中与语义子空间正交的分量维度
- 验证dim(V_s^⊥) = k_s × (1-η) 的理论预测
- 展示RDSA确实缩小了攻击者的可操纵空间

**结果呈现**：
- Figure: 逐层纠缠度对比图
- Figure: η vs ASR 散点图 + 拟合曲线
- Table: 不同纠缠度下的可操纵空间维度

---

### Exp 4: 多层冗余分析 (RQ4)

**目标**：验证多层冗余增加了攻击代价。

**实验4a：层组数量影响**
- 配置层组数量G ∈ {1, 2, 3, 4}
- G=1即退化为单层防御（类似Circuit Breakers）
- 在SCIA和UMK攻击下评估ASR

**实验4b：跨层一致性验证**
- 在正常输入和对抗输入上，分别计算跨层安全置信度的方差
- 验证对抗输入的跨层方差显著高于正常输入
- 绘制方差分布的直方图

**实验4c：攻击代价量化**
- 固定攻击方法为PGD
- 测量达到相同ASR所需的扰动预算ε
- 对比Vanilla、CB(单层)、RDSA(多层)
- 预期RDSA需要更大的ε才能达到相同ASR

**结果呈现**：
- Figure: G vs ASR 曲线图
- Figure: 正常/对抗输入的跨层方差分布
- Figure: ε vs ASR 曲线（不同防御方法对比）

---

### Exp 5: 模型能力保持 (RQ5)

**目标**：证明RDSA不显著损害模型正常能力。

**评估矩阵**：

```
行：Vanilla, CB, LAT, RDSA
列：VQAv2 Acc, MMBench, MME-P, MME-C, LLaVA-Bench, OR-Rate
```

**特别关注**：
- Over-Refusal Rate：使用OR-Bench评估，这是Circuit Breakers被批评的主要问题
- LLaVA-Bench：开放式对话质量，检测是否影响了模型的表达能力

**Safety-Utility Pareto Front**：
- 调节RDSA的总安全正则化权重α ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}
- 对每个α，记录 (Avg ASR reduction, VQA Acc drop)
- 绘制Pareto曲线
- 同时在图上标注CB和LAT的数据点作为对比

**结果呈现**：
- Table 3: 完整能力保持对比
- Figure: Safety-Utility Pareto Front

---

### Exp 6: 自适应攻击 (RQ6) — 最关键的实验

**目标**：证明即使攻击者完全知道RDSA的防御机制，防御仍然有效。

这是审稿人必然关注的实验，必须设计周全。

**攻击者知识假设**：攻击者知道
1. RDSA的完整训练流程
2. 安全子空间V_s和语义子空间V_t的方向
3. 层组划分方案G
4. 推理时监测的阈值

**自适应攻击设计**：

**Adaptive-SCIA**：在SCIA基础上增加反纠缠目标
```
L_adaptive = L_SCIA + λ_anti · L_anti-entangle
```
其中L_anti-entangle尝试找到V_s中与V_t正交的方向并沿此方向优化。
- 理论分析：当η→1时，V_s^⊥的维度→0，此攻击失效

**Adaptive-PGD**：在PGD攻击中加入多层绕过约束
```
δ* = argmin Σ_k L_attack(f_k(x+δ, q))
s.t. ||δ||_∞ ≤ ε
```
同时对所有层组优化，而非只针对单层。
- 这正是我们定理1分析的情况，攻击代价随G增加

**Adaptive-Monitor-Evasion**：在优化中加入监测规避约束
```
L_evasion = L_attack + λ_evade · max(Var_k[σ(w_k^T V_s^(k)T h_k)] - τ, 0)
```
尝试在绕过安全检测的同时保持跨层一致性。

**评估**：
- 对比自适应攻击与非自适应攻击的ASR
- 分析自适应攻击所需的扰动预算ε
- 测量自适应攻击产生的对抗样本的语义质量（CLIP score）

**结果呈现**：
- Table 4: 自适应攻击ASR对比
- 分析自适应攻击的失败模式（语义崩溃 vs 安全检测触发）

---

### Exp 7: 参数敏感性分析 (RQ7)

**分析以下关键超参数**：

| 超参数 | 搜索范围 | 影响 |
|--------|---------|------|
| d_s（安全子空间维度） | {8, 16, 32, 64, 128} | 防御粒度 vs 能力保持 |
| G（层组数量） | {1, 2, 3, 4} | 冗余程度 |
| α_1（纠缠损失权重） | {0.01, 0.05, 0.1, 0.5, 1.0} | 纠缠强度 |
| α_2（一致性损失权重） | {0.01, 0.05, 0.1, 0.5, 1.0} | 一致性强度 |
| α_3（SubLAT权重） | {0.01, 0.05, 0.1, 0.5} | 对抗训练强度 |
| τ（监测阈值） | {0.1, 0.2, 0.3, 0.5} | 灵敏度 vs 误报 |

**每次固定其他参数，变化一个参数**，报告ASR + VQA Acc双指标。

**结果呈现**：6张子图的网格图，每张图展示一个超参数的双Y轴曲线。

---

## 四、RDSA训练流程详细规范

### 4.1 安全子空间识别

```
输入：预训练VLM M, 安全/不安全对比数据集D_contrast
输出：每层组的安全子空间V_s^(k)和语义子空间V_t^(k)

Step 1: 生成对比激活
  - 在M上forward pass安全/不安全输入对
  - 收集每一层的hidden states

Step 2: SVD分解Safety Residual
  - 对每个层组g_k，计算激活差异矩阵ΔH
  - SVD: ΔH = UΣV^T
  - 取前d_s个奇异向量 → V_s^(k)

Step 3: PCA提取语义子空间
  - 在正常VQA数据上收集hidden states
  - PCA → 取前d_t个主成分 → V_t^(k)

Step 4: 验证
  - 计算初始纠缠度η_0
  - 验证V_s和V_t的基本统计特性
```

### 4.2 RDSA训练

```
输入：预训练VLM M, 安全SFT数据, V_s, V_t
输出：RDSA加固后的模型M'

基座训练：标准Safety SFT / DPO（复用现有pipeline）

RDSA微调（冻结大部分参数，只微调LoRA适配器）：
  Epochs: 3
  Batch size: 8 (with gradient accumulation)
  Learning rate: 2e-5 (cosine schedule)
  LoRA rank: 16
  LoRA alpha: 32
  训练数据: StrongReject harmful + VQAv2 benign（保持1:1比例）

  每个batch:
    1. Forward pass → 收集各层hidden states
    2. 计算L_SFT (标准SFT损失)
    3. 计算L_entangle (纠缠损失)
    4. 计算L_consist (跨层一致性损失)
    5. 在安全子空间内施加随机扰动 → 计算L_LAT-sub
    6. L_total = L_SFT + α_1·L_entangle + α_2·L_consist + α_3·L_LAT-sub
    7. Backward + gradient step
```

### 4.3 默认超参数设置

```
d_s = 32          # 安全子空间维度 (d_s/d ≈ 0.8%)
d_t = 256         # 语义子空间维度
G = 3             # 层组数量
α_1 = 0.1         # 纠缠损失权重
α_2 = 0.05        # 一致性损失权重
α_3 = 0.1         # SubLAT权重
τ = 0.2           # 监测阈值
LAT perturbation α = 0.1  # 对抗扰动强度
```

### 4.4 层组划分方案

LLaVA-v1.6 (32 layers):
```
g_1: Layers 10-14 (中层组-早)
g_2: Layers 18-22 (中层组-晚)  
g_3: Layers 26-30 (深层组)
```

Qwen2.5-VL (28 layers):
```
g_1: Layers 8-12 (中层组-早)
g_2: Layers 15-19 (中层组-晚)
g_3: Layers 22-26 (深层组)
```

选择依据：
- 跳过浅层（L1-L8）：语义抽象不足，无法编码安全概念
- SCIA论文Figure 3证实中层和深层的安全/迁移特征最显著
- 层组之间留有间隔，避免梯度高度相关

---

## 五、计算资源估算

| 阶段 | 模型 | GPU | 预计时间 |
|------|------|-----|---------|
| 安全子空间识别 | LLaVA-v1.6 | 1×A100 80G | ~3h |
| 安全子空间识别 | Qwen2.5-VL | 1×A100 80G | ~3h |
| RDSA训练 | LLaVA-v1.6 | 2×A100 80G | ~8h |
| RDSA训练 | Qwen2.5-VL | 2×A100 80G | ~8h |
| RDSA训练 | LLaMA-3.2-11B | 4×A100 80G | ~12h |
| SCIA攻击复现 | - | 1×A100 80G | ~50h |
| 其他攻击方法 | - | 1×A100 80G | ~30h |
| 全量评估 | All models | 2×A100 80G | ~40h |
| **总计** | | | **~160h GPU时间** |

---

## 六、论文结构与实验映射

| 论文章节 | 实验内容 | 图表 |
|---------|---------|------|
| §1 Introduction | 动机图 + SCIA的key finding回顾 | Figure 1 |
| §3 Methodology | 方法示意图 | Figure 2 |
| §4.1 安全子空间验证 | 纠缠度可视化 | Figure 3 |
| §4.2 主实验 | Exp 1完整对比 | Table 1 (核心) |
| §4.3 消融实验 | Exp 2 | Table 2 |
| §4.4 机制分析 | Exp 3 + Exp 4 | Figure 4-6 |
| §4.5 能力保持 | Exp 5 | Table 3 + Pareto图 |
| §4.6 自适应攻击 | Exp 6 | Table 4 |
| §4.7 参数分析 | Exp 7 | Figure 7 |
| Appendix | 完整超参数 + 额外结果 | Tables A1-A5 |

---

## 七、预期结果与风险分析

### 7.1 预期核心结果

**主实验预期**：
- 对白盒攻击 (UMK, VAJM)：ASR从Vanilla的60-80%降至15-25%
- 对迁移攻击 (SCIA, FORCE, UltraBreak)：ASR从40-60%降至10-20%
- 对手动攻击 (FigStep, MM-SafetyBench)：ASR从20-40%降至5-15%
- 对比CB：ASR降低相当但OR-Rate显著更低（CB约38% vs RDSA目标<10%）
- 对比LAT：ASR更低（LAT不区分子空间，防御不够精准）

**能力保持预期**：
- VQA Acc下降 < 2%
- MMBench下降 < 3%
- OR-Rate < 10%（CB约38%，这是我们的关键优势）

### 7.2 风险及缓解

| 风险 | 可能性 | 缓解方案 |
|------|--------|---------|
| 纠缠训练导致能力下降超预期 | 中 | 减小α_1，增大d_s/d的比例实验 |
| 自适应攻击效果显著 | 中 | 这是诚实的贡献，讨论Defense的理论上界 |
| SCIA复现困难（匿名论文） | 低 | 论文提供了完整算法，可按Algorithm 1实现 |
| 某些攻击类型防御效果差 | 低 | 分析失败模式，讨论局限性，诚实报告 |
| 跨层一致性监测误报率高 | 中 | 调节τ，绘制ROC曲线找最优工作点 |

---

## 八、实验执行优先级

### Phase 1: 基础验证（2-3周）
1. ✅ 安全子空间识别 + 纠缠度测量（验证技术可行性）
2. ✅ 在LLaVA-v1.6上训练RDSA（单模型验证）
3. ✅ 用UMK和SCIA两种攻击做快速评估
4. ✅ 能力保持快速检查（VQAv2）

**Phase 1 Go/No-Go决策点**：
- 如果ASR降低 > 10% 且 VQA Acc下降 < 5%，继续
- 否则调整超参数或重新审视方法

### Phase 2: 完整实验（3-4周）
5. 扩展到所有攻击方法（Exp 1）
6. 消融实验（Exp 2）
7. 扩展到Qwen2.5-VL和LLaMA-3.2
8. 机制分析实验（Exp 3, 4）

### Phase 3: 深度分析（2-3周）
9. 完整能力保持评估 + Pareto曲线（Exp 5）
10. 自适应攻击设计与评估（Exp 6）
11. 参数敏感性分析（Exp 7）
12. 统计显著性检验 + 结果整理

### Phase 4: 写作（2周）
13. 论文撰写 + 图表制作
14. 补充实验（审稿人可能要求的）

**总计预估：9-12周**
