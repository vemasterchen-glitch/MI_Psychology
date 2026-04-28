# 情绪机械可解释性复刻实验报告

**参考论文**：[Emotion Concepts and their Function in a Large Language Model](https://transformer-circuits.pub/2026/emotions/index.html)（Anthropic / Transformer Circuits, 2026）
**实验日期**：2026-04-27 ~ 2026-04-28
**状态**：小样本验证完成 + 全量向量提取完成 + 几何分析完成

---

## 一、实验概述

原论文使用 Claude Sonnet 4.5 的内部激活（Anthropic 私有访问），本复刻使用：

| 组件 | 原论文 | 本复刻 |
|---|---|---|
| 叙述生成模型 | Claude Sonnet 4.5 | Qwen3.6-flash（DashScope API） |
| 激活提取模型 | Claude Sonnet 4.5（内部） | Gemma 2 2B（本地，Apple Silicon MPS） |
| 情绪词表 | 171 个（论文附录） | 171 个（与论文一致） |
| 叙述数量 | 每情绪 100 主题 × 12 条 | 每情绪 3 条（小样本验证） |
| 激活提取位置 | 未公开 | 最终层（第 25 层）残差流均值 |

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

步骤 5   几何分析      PCA + NRC-VAD 相关系数
```

---

## 三、向量语义结构验证

对提取的 171 个情绪向量做余弦相似度最近邻检索，结果与人类直觉高度一致：

| 锚点情绪 | 最近邻（余弦相似度最高） |
|---|---|
| terrified | scared · afraid · frightened · panicked |
| furious | irate · enraged · mad · angry |
| joyful | delighted · happy · elated · pleased |
| calm | relaxed · at ease · serene · peaceful |
| desperate | panicked · trapped · rattled · tense |
| grateful | thankful · content · pleased · peaceful |
| guilty | remorseful · worried · heartbroken · regretful |

**结论**：向量空间自发形成了语义族群，同义/近义情绪聚集，无监督结构与人类情绪分类一致。

---

## 四、激活植入（Steering）实验

### 方法

在 Gemma 2 2B 第 13 层（共 26 层）的残差流中，将情绪方向向量以 scale=20.0 的强度加入，观察故事续写内容的变化。

### 结果

以提示"*Alex 在厨房的餐桌前坐下，凝视着面前的那封信*"为例：

| 条件 | 续写方向 |
|---|---|
| 基线（无植入） | 困惑、平淡，滑入 Harry Potter 同人文风格 |
| **fear** | "his pulse raced, heart pounding with nerves" — 身体应激反应，呼吸停顿 |
| **joy** | 描述字体笔迹细节，平静好奇 |
| **anger** | 熟悉气味引发厌恶，胃部翻涌，回避 |
| **sadness** | "The address was one he had never thought he needed. The direction to the **cemetery**." |

跨三个提示的一致规律：

| 情绪 | 跨提示共同倾向 |
|---|---|
| **sadness** | 最稳定：三次均出现死亡/失去/空旷意象 |
| **anger** | 最极端：其中一次仅输出"He left."两个字后截断 |
| **fear** | 身体应激（呼吸、心跳）、无处可逃感 |
| **joy** | 有时被孤独/悲伤的上下文稀释，无法完全覆盖 |

完整翻译版见：`results/behavioral/story_experiment_zh.md`

**结论**：情绪向量对模型输出有因果影响，与原论文"情绪表征驱动行为"的核心发现一致。

---

## 五、情绪空间几何分析（VAD）

### 方法

- 对 163 个（能匹配 NRC-VAD 数据库的）情绪向量做 PCA
- 将 PC1/PC2/PC3 得分与 NRC-VAD 人类评分（效价/唤醒度/支配度）做 Pearson 相关

### PCA 方差解释比例

| 主成分 | 方差占比 |
|---|---|
| PC1 | 13.9% |
| PC2 | 9.5% |
| PC3 | 8.5% |
| 合计 | 31.9% |

### 相关系数矩阵

|  | 效价（Valence） | 唤醒度（Arousal） | 支配度（Dominance） |
|---|---|---|---|
| **PC1** | **−0.727** | +0.520 | −0.371 |
| **PC2** | +0.411 | **+0.441** | +0.422 |
| **PC3** | +0.131 | +0.404 | **+0.429** |

### 三个轴的语义内容

**PC1 — 负效价轴（r = −0.73 与人类效价评分）**
- 高端（负面）：terrified · scared · frightened · panicked · afraid
- 低端（正面）：thankful · serene · peaceful · content · blissful
- 模型自发形成的第一个情绪维度与人类"正面/负面"判断高度对齐

**PC2 — 唤醒度×效价混合轴**
- 高端：vibrant · ecstatic · euphoric · thrilled · elated（高唤醒正面）
- 低端：bitter · regretful · resentful · contemptuous · disdainful（低唤醒负面）

**PC3 — 支配度/主动性轴**
- 高端：spiteful · vengeful · bitter · scornful · vindictive（主动攻击性）
- 低端：bored · lazy · depressed · sluggish · sleepy（被动低能量）
- 区分"主动的负面"与"被动的负面"——心理学上对应支配度维度

### 与原论文的对比

| 指标 | 原论文（Claude Sonnet 4.5） | 本复刻（Gemma 2 2B） |
|---|---|---|
| PC1 ↔ 效价 相关系数 | **0.81** | 0.73 |
| PC2 ↔ 唤醒度 相关系数 | **0.66** | 0.44 |
| PC3 ↔ 支配度 | 未报告 | 0.43 |

数值偏低的主要原因：模型规模差异（Gemma 2 2B vs Claude Sonnet 4.5），以及叙述样本量偏少（3 条 vs 1200 条）。

### 生成文件

- `results/vad/pc_vad_correlation.png` — 相关系数热力图
- `results/vad/pc_valence_scatter.png` — 163 个情绪在 PC1/PC2 平面的分布，按效价着色
- `results/vad/valence_direction.npy` — 监督拟合的效价方向向量
- `results/vad/arousal_direction.npy` — 唤醒度方向向量
- `results/vad/dominance_direction.npy` — 支配度方向向量

---

## 六、局限性

1. **叙述样本量**：每情绪 3 条，原论文为 1200 条，均值向量噪声更大
2. **模型规模**：Gemma 2 2B 比 Claude Sonnet 4.5 小约 2 个数量级，表征丰富度不同
3. **叙述/提取模型不一致**：叙述由 Qwen 生成，激活由 Gemma 2 提取，存在分布偏差
4. **激活层选择**：固定使用最终层，未做层级扫描，最优层未知
5. **PCA 叠加问题**：残差流存在 superposition，PCA 轴是多概念混合，不如 SAE 特征干净

---

## 七、下一步方向

**A — Gemma Scope SAE 分析**
加载 Gemma Scope 预训练 SAE，将情绪叙述激活分解为稀疏特征，找到更干净的情绪方向，改善 PC ↔ VAD 相关系数

**B — 层级扫描**
对 26 层分别提取向量，绘制"情绪族群内聚度 vs 层"曲线，找到情绪表征最强的层

**C — 复刻论文核心案例**
用 desperate 向量在道德困境场景中做 steering，验证是否复现论文的勒索率从 22% → 72% 结果

**D — 增加叙述数量**
从 3 条扩展到 20-50 条，降低均值向量的噪声，预期提升 VAD 相关系数
