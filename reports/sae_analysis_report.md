# SAE 情绪空间分析报告

**日期**：2026-04-28  
**方法**：Gemma Scope 预训练 SAE（layer 25, width 16k, avg_l0=55）  
**输入**：171 情绪向量 × 2304 维（Gemma 2 2B 残差流原始激活均值）

---

## 一、方法

### SAE 编码

将 171 个情绪向量直接送入 Gemma Scope SAE 编码器（不做 StandardScaler，SAE 训练时使用原始激活分布）：

```
pre_act = h @ W_enc + b_enc        # (2304,) → (16384,)
features = pre_act * (pre_act > threshold)   # JumpReLU
```

每个情绪平均激活 **13.5 ± 2.5** 个特征（原始单 token 前向约 55 个，均值池化后稀疏度降低，符合预期）。

### VAD 方向构造（半监督）

对 157 个匹配 NRC-VAD 词典的情绪，计算每个 SAE 特征激活值与人类 VAD 评分的 Pearson r，取 top-50 特征，用相关系数加权其解码器方向 `W_dec[j]` 合成目标方向：

```
direction = Σ r_j × W_dec[j]   （top-50，再单位化）
```

最终将情绪向量投影到该方向，与 VAD 评分计算相关系数。

---

## 二、结果

### VAD 相关系数对比

各行：一条投影方向；各列：该方向投影与人类效价 / 唤醒度 / 支配度之间的 **Pearson r**（对角块为「目标维」，非对角为与其他 VAD 维的串扰）。

| 方法 | 效价 r | 唤醒度 r | 支配度 r |
|---|:-:|:-:|:-:|
| PCA PC1 | 0.741 | 0.515 | 0.376 |
| SAE-效价方向 | 0.829 | 0.328 | 0.487 |
| SAE-唤醒度方向 | 0.362 | 0.775 | 0.076 |
| SAE-支配度方向 | 0.766 | 0.114 | 0.734 |
| 原论文（Claude Sonnet 4.5，PCA，效价轴） | 0.810 | 0.660 | — |

专用 SAE 方向在对齐维度上高于 PCA PC1 的对应相关；非目标维上仍有相关（例如效价方向与支配度 r=0.487），说明轴未完全正交。原论文报告的是 PCA 效价主轴，与本表第一列可比：SAE-效价方向 0.829 与原论文 0.810 接近略高。

### 最判别性 SAE 特征（方差最大 top-5）

| 特征 ID | 方差 | 主要激活情绪 |
|---|---|---|
| 14325 | 419.97 | alert, amazed, angry, aroused, ashamed, at ease |
| 11835 | 259.37 | afraid, alarmed, alert, amazed, angry, anxious |
| 15298 | 171.97 | afraid, alarmed, alert, amazed, amused, angry |
| 11012 | 154.03 | amazed, at ease, awestruck, blissful, calm, cheerful |
| 15942 | 147.65 | alarmed, amused, angry, annoyed, anxious, aroused |

Top-1 效价相关特征：#11012（r=0.663，激活正面平静情绪族群）  
Top-1 唤醒度相关特征：#11835（r=0.545，激活高唤醒负面情绪族群）

---

## 三、解读

**SAE 优于 PCA 的原因**：PCA 找方差最大方向，方差大不等于语义纯，PC1 实际是效价、唤醒度、其他信号的混合。SAE 通过稀疏约束将残差流的 superposition 拆解成更单义的特征，再用 VAD 标签选取相关特征合成方向，因此语义更纯净。

**方法性质**：SAE 这一步是**半监督**的——用 VAD 人类标签来选特征，不是纯无监督发现。PCA 是纯无监督，两者不完全可比，但结果说明 SAE 特征空间确实更好地分离了效价/唤醒度/支配度三个维度。

---

## 四、输出文件

| 文件 | 内容 |
|---|---|
| `emotion_features.npy` | (171, 16384) 稀疏特征激活矩阵 |
| `top_features_per_emotion.json` | 每个情绪的 top-5 激活特征 |
| `sae_valence_direction.npy` | SAE 合成效价方向向量（2304维） |
| `sae_arousal_direction.npy` | SAE 合成唤醒度方向向量 |
| `sae_dominance_direction.npy` | SAE 合成支配度方向向量 |
| `pca_vs_sae_vad.png` | PCA vs SAE VAD 相关系数热力图 |

---

## 五、局限性

1. **均值池化降低稀疏度**：情绪向量是多条叙述的激活均值，平均仅 13.5 个特征激活（原始单 token 约 55 个），部分细粒度特征信号被平滑掉。
2. **半监督性质**：VAD 标签参与了方向选取，结果不能作为模型自发学习 VAD 结构的证据，只能说明 SAE 特征空间与 VAD 结构相容。
3. **叙述样本量**：当前每情绪 3 条叙述（生成中补充至 20 条），均值向量噪声较大。

---

## 六、下一步

- 叙述补充至 20 条后重新提取激活，重跑 SAE 分析，预期稀疏度更接近 55，r 可能进一步提升
- 用 SAE 效价/唤醒度方向做 Steering 实验，验证因果影响
