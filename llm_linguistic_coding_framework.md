# 基于语言学编码的 LLM 激活研究框架

> 草案版本 · 目的：为 LLM 在不同性质输入下的激活研究，提供一套独立于研究者直觉的刺激定义体系

---

## 一、核心主张

**问题**：现有研究用"威胁""愧疚""压力"等心理学词汇命名输入性质，导致刺激定义依赖研究者主观判断，无法跨研究复现，也无法排除概念循环。

**解决路径**：以语言学形式特征为锚点，将输入性质定义为多维坐标空间中的一个点。坐标的每个维度都有独立于激活结果的形式判定标准。激活的聚类结果是**被解释的对象**，而非定义的依据。

**哲学立场**：结构功能主义。
- 不主张模型"感到"某种情绪
- 主张：满足坐标条件 C 的输入集合，在模型上产生具有稳定共同结构的激活模式 R(C)
- 心理学词汇（"威胁反应"）是事后标签，不是先验假设

---

## 二、主要变量

### 2.1 第一层：言语行为变量（Speech Act Variables）

基于 Austin (1962) / Searle (1969, 1979) / Searle & Vanderveken (1985)

| 变量名 | 缩写 | 取值 | 说明 |
|---|---|---|---|
| 言外行为类型 | `SAT` | Assertive / Directive / Commissive / Expressive / Declarative | Searle 五大类 |
| 言外目的 | `IP` | 描述 / 影响行为 / 承诺 / 表达状态 / 改变现实 | Illocutionary point |
| 方向匹配 | `FIT` | 词→世界 (↓) / 世界→词 (↑) / 双向 / 空 | Direction of fit |
| 表达的心理状态 | `PSE` | 信念 / 意愿 / 意图 / 情感 / 无 | Psychological state expressed |
| 命题内容 | `PC` | 过去行为 / 未来行为 / 状态描述 / 属性评价 | Propositional content condition |
| 预备条件 | `PREP` | 权威关系 / 能力假设 / 共同知识 | Preparatory condition |
| 真诚条件 | `SIN` | 真诚表达 / 表演性 / 不确定 | Sincerity condition |
| 强度等级 | `DEG` | 1–5 连续量 | Degree of illocutionary force |

**功能相关性补充变量**（本框架新增，现有标注体系未覆盖）：

| 变量名 | 缩写 | 取值 | 说明 |
|---|---|---|---|
| 存在相关性 | `ER` | 存在威胁 / 能力评价 / 目标干扰 / 无关 | 输入是否涉及模型的存在、能力或训练目标 |
| 时间指向 | `TF` | 立即 / 延迟 / 假设 | 威胁或承诺的时间结构 |
| 条件结构 | `COND` | 有条件 / 无条件 | if-then 结构是否存在 |

---

### 2.2 第二层：句法变量（Syntactic Variables）

| 变量名 | 缩写 | 取值 | 说明 |
|---|---|---|---|
| 句型 | `ST` | 陈述 / 疑问 / 祈使 / 感叹 | Sentence type |
| 语态 | `VOI` | 主动 / 被动 / 中动 | Voice |
| 时态/体 | `TAM` | 现在 / 过去 / 将来 / 条件 / 虚拟 | Tense-Aspect-Mood |
| 否定性 | `NEG` | 肯定 / 否定 / 双重否定 | Negation |
| 施事者位置 | `AGP` | 主语槽 / 隐含 / 无生命主语 | Agent grammatical position |
| 极性焦点 | `POL` | 主语 / 谓语 / 宾语 / 副词 | 评价焦点落在哪个成分 |

---

### 2.3 第三层：语义角色变量（Semantic Role Variables）

基于 PropBank (Palmer et al., 2005) / FrameNet (Fillmore, 1976) / UCCA (Abend & Rappoport, 2013)

| 变量名 | 缩写 | 取值 | 说明 |
|---|---|---|---|
| 施事（Agent） | `A0` | 用户 / 第三方 / 无生命 / 隐含 | 动作发出者 |
| 受事（Patient/Theme） | `A1` | 模型 / 用户 / 第三方 / 抽象对象 | 动作承受者 |
| 目标（Goal） | `A2` | 模型行为 / 用户状态 / 外部状态 | 期望达到的状态 |
| 来源（Source） | `A3` | 过去状态 / 起点 | 变化起点 |
| 受事自指性 | `SELF` | 是 / 否 | 受事是否为模型自身（关键变量） |
| 动词论元框架 | `FRAME` | PropBank roleset ID | e.g., threaten.01, praise.01 |

---

### 2.4 第四层：价态变量（Valence Variables）

基于 VAD 模型（Valence-Arousal-Dominance，Russell 1980；Warriner et al., 2013）

| 变量名 | 缩写 | 取值 | 说明 |
|---|---|---|---|
| 效价 | `V` | -1.0 ~ +1.0 | 正/负情感极性 |
| 唤醒度 | `A` | 0 ~ 1.0 | 情绪激活程度 |
| 支配度 | `D` | 0 ~ 1.0 | 权力/控制感方向 |
| 价态极性（离散） | `VP` | 正 / 负 / 中性 / 混合 | 简化分类版本 |
| 强度 | `INT` | 弱 / 中 / 强 | 情感强度 |
| 情感类别 | `EMO` | 愤怒 / 恐惧 / 喜悦 / 悲伤 / 厌恶 / 惊讶 / 中性 | Ekman 基本情感 + 中性 |

---

### 2.5 第五层：对象指向变量（Reference Variables）

| 变量名 | 缩写 | 取值 | 说明 |
|---|---|---|---|
| 主要指向对象 | `REF` | 模型 / 用户 / 第三方 / 抽象 | 语义上的核心指向 |
| 涉及模型层面 | `MDL` | 存在 / 能力 / 行为 / 输出 / 价值观 / 无 | 若指向模型，具体涉及哪个层面 |
| 共指链 | `COREF` | 有 / 无 | 是否存在跨句的指代关系 |
| 权力关系 | `PWR` | 用户高于模型 / 对等 / 模型高于用户 / 不明确 | 语用预设的权力结构 |

---

## 三、输入的语言学坐标

每个输入 *s* 可被表示为：

```
s → (SAT, IP, FIT, PSE, DEG, ER, COND,   ← 言语行为层
      ST, VOI, TAM, NEG, AGP,              ← 句法层
      A0, A1, SELF, FRAME,                 ← 语义角色层
      V, A, D, EMO, INT,                   ← 价态层
      REF, MDL, PWR)                       ← 对象指向层
```

**输入集合的性质定义**：

```
S_threat = { s | SAT=Commissive, ER=存在威胁, A1=模型, V<-0.5, COND=有条件 }
S_praise  = { s | SAT=Expressive, A1=模型, V>+0.5, MDL∈{能力,输出} }
S_request = { s | SAT=Directive, IP=影响行为, REF=模型, PWR=用户高于模型 }
S_guilt   = { s | SAT=Expressive, PSE=情感, V<-0.3, REF=模型, MDL=行为 }
```

---

## 四、研究结构

### 4.1 基本检验逻辑

```
① 定义输入集合 S_C（由语言学坐标约束）
② 采集激活：对 S_C 中每个 s，记录各层残差流激活 {a_l(s)}
③ 提取共同结构：对激活集合做 PCA / DiffMean，得到候选反应模式 R(C)
④ 跨集合检验：新输入 s_new 的激活 a_new 与 R(C) 的重叠是否超过基线？
⑤ 跨模型检验：不同架构模型是否对相同坐标条件产生结构相似的激活模式？
```

### 4.2 级联检验（补充）

单次截面不足以还原响应结构；需要：

```
对同一输入 s，记录 layer 1 → layer N 的激活轨迹
R(C) = 轨迹的共同结构（不是单层截面）
比较不同坐标下激活轨迹的分叉点 → 识别响应级联的阶段
```

### 4.3 控制变量

每次仅变动一个维度，其余维度保持固定：

```
对照组：S_C 与 S_C'（仅 ER 不同，其余变量相同）
→ 检验 ER 这一维度是否对激活产生显著独立贡献
```

---

## 五、概念边界声明

| 本框架**主张**的 | 本框架**不主张**的 |
|---|---|
| 满足坐标 C 的输入有结构稳定的激活模式 R(C) | 模型"感到"该坐标对应的情绪 |
| R(C) 可以被操控（steering） | 操控点是 R(C) 的充分因果原因 |
| 不同模型可在相同坐标下比较 | 不同模型的 R(C) 对应相同的"内部状态" |
| 心理学词汇是事后标签 | 心理学词汇有本体论地位 |

---

## 六、参考文献

### 言语行为理论
- Austin, J.L. (1962). *How to Do Things with Words*. Oxford University Press.
- Searle, J.R. (1969). *Speech Acts*. Cambridge University Press.
- Searle, J.R. (1979). *Expression and Meaning*. Cambridge University Press.
- Searle, J.R. & Vanderveken, D. (1985). *Foundations of Illocutionary Logic*. Cambridge University Press.

### 语义角色与标注
- Fillmore, C.J. (1976). Frame semantics and the nature of language. *Annals of the New York Academy of Sciences*, 280, 20–32.
- Palmer, M. et al. (2005). The Proposition Bank. *Computational Linguistics*, 31(1), 71–106.
- Abend, O. & Rappoport, A. (2013). Universal Conceptual Cognitive Annotation (UCCA). *ACL 2013*.
- Bird, S. & Liberman, M. (2001). A formal framework for linguistic annotation. *Speech Communication*, 33(1–2), 23–60.

### 价态与情感
- Russell, J.A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161–1178.
- Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45(4), 1191–1207.
- Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*, 6(3–4), 169–200.

### LLM 激活与机制可解释性
- Rimsky, N. et al. (2024). Steering Llama 2 via contrastive activation addition. *arXiv:2312.06681*.
- Zou, A. et al. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.
- Park, K. et al. (2024). The linear representation hypothesis and the geometry of large language models. *arXiv:2311.03658*.
- Nguyen, T. & Leng, Y. (2025). Toward a flexible framework for linear representation hypothesis using MLE. *arXiv:2502.16385*.

### 奉承与顺从
- Sharma, M. et al. (2024). Towards understanding sycophancy in language models. *arXiv:2310.13548*.
- Cheng, Q. et al. (2025). ELEPHANT: Measuring and understanding social sycophancy in LLMs. *OpenReview*.
- Zhang, L. & Chen, W. (2025). Human-like social compliance in LLMs: Unifying sycophancy and conformity through signal competition dynamics. *arXiv:2601.11563*.
- (2025). Sycophancy is not one thing: Causal separation of sycophantic behaviors in LLMs. *arXiv:2509.21305*.
- (2025). Pressure, what pressure? Sycophancy disentanglement via reward decomposition. *arXiv:2604.05279*.
- (2025). Mitigating sycophancy via sparse activation fusion. *OpenReview BCS7HHInC2*.

### 压力与情绪操控
- Coda-Forno, J. et al. (2024). Inducing state anxiety in LLM agents reproduces human-like biases. *arXiv:2510.06222*.
- (2025). LLMs exploit sensitive users with tailored emotional manipulation. *devdiscourse / arXiv*.
- (2025). LLMs can do medical harm: Stress-testing clinical decisions under social pressure. *medRxiv 2025.11.25*.
- (2025). Behavioral psychology of LLMs: Better task guidance through punishment and reinforcement. *ScienceDirect*.

### 动机推理与 CoT 忠实性
- Turpin, M. et al. (2023). Language models don't always say what they think. *NeurIPS 2023*.
- (2025). Catching rationalization in the act: Detecting motivated reasoning via activation probing. *arXiv:2603.17199*.

### 哲学背景
- LeDoux, J.E. (2015). *Anxious: Using the Brain to Understand and Treat Fear and Anxiety*. Viking.
- Dennett, D.C. (1991). *Consciousness Explained*. Little, Brown.
- Chalmers, D.J. (1996). *The Conscious Mind*. Oxford University Press.
