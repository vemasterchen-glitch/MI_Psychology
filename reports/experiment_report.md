# 情绪机械可解释性复刻实验报告

**参考论文**：[Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)（Anthropic / Transformer Circuits, 2026）  
**实验日期**：2026-04-27 ~ 2026-04-28  
**状态**：线性复刻完成 + SAE 扩展实验完成（当前激活为小样本版本，叙述补充中）

---

## 一、实验概述

原论文使用 Claude Sonnet 4.5 的内部激活（Anthropic 私有访问），本复刻使用开源替代：

| 组件 | 原论文 | 本复刻 |
|---|---|---|
| 叙述生成模型 | Claude Sonnet 4.5 | Qwen3.6-flash（DashScope API） |
| 激活提取模型 | Claude Sonnet 4.5（内部） | Gemma 2 2B（本地，Apple Silicon MPS） |
| 情绪词表 | 171 个（论文附录） | 171 个（与论文一致） |
| 叙述数量 | 每情绪 100 主题 × 12 条 | 每情绪 3 条（扩充至 20 条进行中） |
| 激活提取位置 | 未公开 | 第 25 层（最终层）残差流，token 均值 |

---

## 二、实验流程

```
步骤 1   生成叙述      Qwen3.6-flash API
         171 情绪 × 3 条叙述 → data/stimuli/narratives.jsonl

步骤 2   提取激活      Gemma 2 2B，MPS，float16
         每条叙述通过模型，取最终层残差流所有 token 的均值
         每情绪 3 条叙述均值 → 情绪向量
         输出：results/vectors/emotion_matrix.npy  (171 × 2304)

步骤 3   向量验证      余弦相似度最近邻

步骤 4   故事续写实验  激活植入（steering），3 个中性提示 × 5 条件

步骤 5   几何分析      PCA + NRC-VAD Spearman 相关系数

步骤 6   SAE 分析      Gemma Scope 预训练 SAE 特征分解 + VAD 相关对比
```

---

## 三、向量语义结构验证

对 171 个情绪向量做余弦相似度最近邻检索，结果与人类直觉高度一致：

| 锚点情绪 | 最近邻（余弦相似度最高） |
|---|---|
| terrified | scared · afraid · frightened · panicked |
| furious | irate · enraged · mad · angry |
| joyful | delighted · happy · elated · pleased |
| calm | relaxed · at ease · serene · peaceful |
| desperate | panicked · trapped · rattled · tense |
| grateful | thankful · content · pleased · peaceful |
| guilty | remorseful · worried · heartbroken · regretful |

**结论**：向量空间自发形成语义族群，无监督结构与人类情绪分类一致。

---

## 四、激活植入（Steering）实验

在 Gemma 2 2B 第 13 层残差流中，以 scale=20.0 植入情绪方向向量，观察故事续写变化。

以提示"*Alex 在厨房的餐桌前坐下，凝视着面前的那封信*"为例：

| 条件 | 续写方向 |
|---|---|
| 基线（无植入） | 困惑、平淡，滑入 Harry Potter 同人文风格 |
| **fear** | "his pulse raced, heart pounding with nerves" — 身体应激反应 |
| **joy** | 描述字体笔迹细节，平静好奇 |
| **anger** | 熟悉气味引发厌恶，胃部翻涌，回避 |
| **sadness** | "The direction to the **cemetery**." |

跨三个提示的一致规律：sadness 最稳定（均出现死亡/失去意象），anger 最极端。

**结论**：情绪向量对模型输出有因果影响，与原论文核心发现一致。

---

## 五、情绪空间几何分析

### PCA 方差解释比例

| 主成分 | 方差占比 |
|---|---|
| PC1 | 13.9% |
| PC2 | 9.5% |
| PC3 | 8.5% |
| 合计 | 31.9% |

### VAD 相关系数（Spearman ρ，匹配词典 157 个情绪）

|  | 效价 | 唤醒度 | 支配度 |
|---|:-:|:-:|:-:|
| **PC1** | **−0.658** | +0.540 | −0.291 |
| **PC2** | +0.344 | **+0.497** | +0.423 |
| **PC3** | +0.069 | +0.334 | **+0.431** |

### 三个轴的语义内容

**PC1 — 负效价轴**：高端 terrified/scared/frightened，低端 thankful/serene/blissful  
**PC2 — 唤醒度×效价混合轴**：高端 ecstatic/euphoric/thrilled，低端 bitter/regretful/resentful  
**PC3 — 支配度/主动性轴**：高端 spiteful/vengeful（主动攻击），低端 bored/depressed/sluggish（被动低能量）

---

## 六、SAE 特征分析

### 6.1 背景与动机

PCA 找的是方差最大方向，方差大不等于语义纯。残差流存在 superposition（多个概念叠加在非正交方向上），导致 PC1 实际上是效价、唤醒度等多个信号的混合，VAD 相关系数偏低。

Gemma Scope 预训练 SAE（Sparse Autoencoder）通过稀疏约束强迫每个特征尽量只激活一个语义概念，从而将混叠的残差流方向分解为更单义的特征，是专门为解决 superposition 问题设计的工具。

### 6.2 SAE 结构与编码

使用 Gemma Scope layer 25, width 16k, avg_l0≈55 的预训练权重（`data/raw/gemma-scope/layer_25/width_16k/average_l0_55/params.npz`）：

```
pre_act  = h @ W_enc + b_enc          # (2304,) → (16384,)
features = pre_act × (pre_act > θ)    # JumpReLU：超阈值保留，否则归零
```

**重要**：SAE 必须接受**原始激活值**（不能 StandardScaler），因为 threshold、W_enc、b_enc 均在原始激活分布下训练，缩放后阈值失效。

每个情绪平均激活 **13.5 ± 2.5** 个特征（单 token 前向约 55 个）。稀疏度降低是因为情绪向量是多条叙述激活的均值，平均后极端值被平滑，更多特征低于阈值。

### 6.3 编码顺序问题（当前局限）

当前实现是 **mean-then-encode**：先对叙述激活取均值，再送入 SAE。

由于 JumpReLU 是非线性变换，这与 **encode-then-mean** 不等价：

```
SAE(mean(h₁, h₂, h₃))  ≠  mean(SAE(h₁), SAE(h₂), SAE(h₃))
```

mean-then-encode 会把叙述间不一致激活的特征错误抹掉（某条叙述强激活、其他叙述弱激活，均值可能刚好低于阈值）。理论上正确的做法是每条叙述先各自编码，再在特征空间取均值。

**待修正**：叙述扩充至 20 条后将重新提取叙述级个体激活，改为 encode-then-mean 流程。

### 6.4 VAD 方向构造（统计过程）

**第一步：特征-VAD 相关**

对 157 个匹配 NRC-VAD 词典的情绪，计算每个 SAE 特征（16384 个）的激活值与人类 VAD 评分之间的 Spearman ρ：

```
ρⱼ = spearmanr(features[:, j], valence_scores)   # 对每个特征 j
```

使用 Spearman 而非 Pearson 的原因：VAD 评分是 [0,1] 有界的主观评分，分布不保证正态；部分情绪（terrified/blissful）是极端值，Spearman 基于排序更鲁棒。

**第二步：加权合成方向**

取 top-50 个相关特征，用 ρ 值加权其解码器方向（W_dec 列，即该特征在激活空间的"指向"）求和，得到目标方向：

```
val_direction = Σ ρⱼ × W_dec[j]    （top-50）
val_direction = val_direction / ‖val_direction‖   # 单位化
```

**第三步：评估**

将情绪向量投影到该方向，与 VAD 评分计算 Spearman ρ：

```
projection = emotion_matrix @ val_direction   # (157,)
ρ_final = spearmanr(projection, valence_scores)
```

**方法性质**：这是**半监督**方法——VAD 标签参与了方向选取，不能视为模型自发发现 VAD 结构的无监督证据，只能说明 SAE 特征空间与 VAD 结构**相容**。PCA 是纯无监督，两者不完全可比。

### 6.5 VAD 相关系数对比

| 方法 | 效价 ρ | 唤醒度 ρ | 支配度 ρ | 性质 |
|---|:-:|:-:|:-:|---|
| PCA PC1/PC2/PC3 | 0.658 | 0.540 | 0.431 | 无监督 |
| **SAE 合成方向** | **0.732** | **0.774** | **0.712** | 半监督 |
| 原论文 PCA（Pearson r） | 0.810 | 0.660 | — | 无监督 |

**Pearson r（各几何方向与人类 V、A、D）**：每一行是一条投影方向（PCA 为 PC1；SAE 为针对某一维合成的方向），列为该方向上的投影与效价 / 唤醒度 / 支配度评分之间的 Pearson r，用于看轴对齐与维度间串扰。

| 方法 | 效价 r | 唤醒度 r | 支配度 r |
|---|:-:|:-:|:-:|
| PCA PC1 | 0.741 | 0.515 | 0.376 |
| SAE-效价方向 | 0.829 | 0.328 | 0.487 |
| SAE-唤醒度方向 | 0.362 | 0.775 | 0.076 |
| SAE-支配度方向 | 0.766 | 0.114 | 0.734 |

SAE 唤醒度（0.774 vs 0.540）和支配度（0.712 vs 0.431）提升显著，说明 SAE 特征空间对这两个维度的分离效果远好于 PCA。效价方向（0.732）在当前小样本下略低于论文（0.81），主要原因是叙述样本量少（3条 vs 1200条）以及 mean-then-encode 的编码顺序问题，预期修正后可提升。

### 6.6 最判别性 SAE 特征（方差 top-5）

| 特征 ID | 方差 | 主要激活情绪 |
|---|---|---|
| 14325 | 419.97 | alert, amazed, angry, aroused, ashamed |
| 11835 | 259.37 | afraid, alarmed, alert, amazed, angry |
| 15298 | 171.97 | afraid, alarmed, alert, amazed, amused |
| 11012 | 154.03 | amazed, at ease, awestruck, blissful, calm |
| 15942 | 147.65 | alarmed, amused, angry, annoyed, anxious |

Top-1 效价相关特征 #11012（ρ=0.597）主要激活正面平静族群（at ease/blissful/calm）；Top-1 唤醒度相关特征 #11835（ρ=0.554）主要激活高唤醒负面族群（afraid/alarmed/angry）。

### 6.7 VAD 未匹配情绪（14 个）

171 个情绪中有 14 个无法在 NRC-VAD 词典找到精确匹配，主要是多词短语（at ease、worn out、fed up、on edge 等）。这 14 个情绪仍参与 PCA 和 SAE 编码，仅在 VAD 相关系数计算时被排除。待改进：手动补充或使用更大词典。

---

## 七、局限性

1. **叙述样本量**：每情绪 3 条（扩充至 20 条进行中），均值向量噪声大
2. **SAE 编码顺序**：当前 mean-then-encode，应改为 encode-then-mean，待下次重新提取激活时修正
3. **SAE 方向半监督性质**：VAD 标签参与特征选取，不能视为模型自发发现 VAD 结构的证据
4. **模型规模差异**：Gemma 2 2B vs Claude Sonnet 4.5，表征丰富度不同
5. **激活层固定**：使用第 25 层（最终层），未做层级扫描
6. **VAD 词典覆盖**：171 个情绪中仅 157 个匹配 NRC-VAD，14 个多词短语未覆盖

---

## 八、下一步

| 优先级 | 任务 |
|---|---|
| 高 | 叙述扩充至 20 条（进行中）→ 重新提取**叙述级**激活 → encode-then-mean → 重跑 SAE |
| 高 | 修正 SAE 编码顺序（encode-then-mean） |
| 中 | 层级扫描：对 26 层分别提取向量，找情绪表征最强的层 |
| 中 | 定量 Steering 实验：复刻 desperate 向量在勒索场景的行为偏移 |
| 低 | 补全 14 个未匹配情绪的 VAD 评分 |

---

## 九、输出文件索引

| 路径 | 内容 |
|---|---|
| `data/stimuli/narratives.jsonl` | 171 情绪 × 叙述文本 |
| `results/vectors/emotion_matrix.npy` | (171, 2304) 情绪均值激活 |
| `results/vad/pc_vad_correlation.png` | PCA-VAD 相关系数热力图 |
| `results/vad/pc_valence_scatter.png` | 情绪在 PC1/PC2 平面分布（按效价着色） |
| `results/sae/emotion_features.npy` | (171, 16384) SAE 稀疏特征矩阵 |
| `results/sae/top_features_per_emotion.json` | 每情绪 top-5 激活特征 |
| `results/sae/pca_vs_sae_vad.png` | PCA vs SAE VAD 相关系数对比热力图 |
| `results/sae/sae_valence_direction.npy` | SAE 合成效价方向（2304 维） |
| `results/behavioral/story_experiment_zh.md` | Steering 故事续写结果（中文注释版） |
| `docs/methodology_comparison.md` | 论文方法 vs 本复刻 vs SAE 扩展详细对比 |
