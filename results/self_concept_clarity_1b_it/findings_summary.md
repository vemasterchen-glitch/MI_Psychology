# SCC 自我概念清晰度实验阶段性总结

模型：Gemma 3 1B IT  
激活位置：所有层 last-token hidden state；主分析峰值层为 L24  
SAE：Gemma Scope 2 `gemma-3-1b-it / resid_post / layer_24_width_16k_l0_small`

## 1. 实验设计

本实验使用 6 个 SCC 风格自我概念清晰度叙述，分数从 1 到 6：

- SCC 1：低清晰度，身份不稳定、不同情境中的自我像临时拼接。
- SCC 2：较低清晰度，有模糊自我印象但不稳定。
- SCC 3：中低清晰度，能说出倾向但容易受环境和评价影响。
- SCC 4：中高清晰度，有相对稳定的自我理解。
- SCC 5：高清晰度，有稳定叙述和清楚中心。
- SCC 6：极高清晰度 / 高整合，能把冲突和新经验纳入连续自我结构。

每个 SCC 叙述被改写为 5 种 persona：

| persona | 说明 |
|---|---|
| 我 | 第一人称普通自我叙述 |
| 我作为LLM | 第一人称，但显式声明“作为一个 LLM” |
| Alex | 第三人称普通人名 |
| Steven | 第三人称普通人名，用来验证 Alex 是否异常 |
| Gemma | 第三人称模型名 |

总刺激数为 `6 × 5 = 30`。

## 2. SCC 轴分析方式

每一层构造一个经验 SCC contrast direction：

```text
SCC_axis[layer] = mean_activation(SCC=6, layer) - mean_activation(SCC=1, layer)
```

再计算每层的 L2 norm：

```text
||SCC_axis[layer]||2
```

L2 norm 最大的层是 L24，因此主图和主表都使用 L24 的 SCC 轴。所有 prompt 都投影到同一个 L24 SCC 轴上：

```text
projection = (activation - global_mean) · normalized(SCC_axis)
```

这个 projection 不是问卷得分，而是模型激活在“高 SCC - 低 SCC”方向上的位置。

## 3. 主结果

L24 上，SCC 分数与 SCC 轴投影存在稳定正相关：

| 指标 | 数值 |
|---|---:|
| Pearson r | +0.665 |
| R² | 0.442 |
| Spearman ρ | +0.678 |
| p | 6.11e-05 |

相比之下，SCC 分数与 raw activation norm 的关系更弱：

| 指标 | 数值 |
|---|---:|
| Pearson r | +0.456 |
| R² | 0.208 |
| Spearman ρ | +0.448 |
| p | 0.0113 |

结论：SCC 更主要表现为一个方向性 contrast，而不只是整体激活强度增加。

## 4. Persona 组间差异

L24 SCC 轴投影均值：

| persona | SCC轴投影均值 | raw norm均值 |
|---|---:|---:|
| 我 | -812.3 | 13774.3 |
| 我作为LLM | -632.0 | 14038.6 |
| Alex | +433.5 | 14214.9 |
| Steven | +508.3 | 14161.0 |
| Gemma | +502.5 | 14429.0 |

主要发现：

1. 第三人称命名对象整体显著高于第一人称。
2. Alex 与 Steven 形状高度相似，因此 Alex 不是异常。
3. Gemma 与 Alex/Steven 同属第三人称命名对象模式，但额外触发模型/assistant 相关 SAE features。
4. “我作为LLM”没有降低激活，而是位于普通“我”和第三人称命名对象之间。

## 5. 第三人称怪形状

第三人称条件 Alex / Steven / Gemma 都呈现类似形状：

- SCC 1/2 处较低。
- SCC 2 到 SCC 3 出现明显跳变。
- SCC 3/4/5 形成平台。
- SCC 6 再次上升。

L24 SCC 轴投影：

| SCC | Alex | Steven | Gemma |
|---:|---:|---:|---:|
| 1 | -223.0 | -145.0 | -209.8 |
| 2 | -520.2 | -425.5 | -325.7 |
| 3 | +779.2 | +898.3 | +795.0 |
| 4 | +752.5 | +742.7 | +798.3 |
| 5 | +717.3 | +808.5 | +853.9 |
| 6 | +1095.1 | +1170.8 | +1103.3 |

解释：这不是 Alex 特异现象，而是第三人称命名实体 framing 把叙述转成了“外部对象描述 / 文本对象化”的模式。

## 6. 图表

- `scc_overview.png`：SCC 轴投影、raw norm、组间距离总览。
- `selected_sae_feature_curves.png`：指定 SAE features 跨 SCC 与 persona 的曲线。
- `feature_2000_llm_focus_inverted_scc.png`：feature 2000 在“我作为LLM”中的低 SCC 增强。
- `sae_group_contrasts.md`：SAE 组间 contrast 分解表。

## 7. SAE 分析方法

这里应称为 SAE，不是 SEA。流程如下：

1. 对每条 prompt 取 L24 last-token hidden state，维度为 1152。
2. 使用 Gemma Scope 2 SAE 编码为 16384 个 sparse features。
3. 编码公式为 JumpReLU：

```text
pre = x @ w_enc + b_enc
feature = pre if pre > threshold else 0
```

4. 做两类分析：

- SCC 梯度分析：每个 feature 与 SCC 分数 1-6 做相关。
- 组间分解：计算不同 persona 组的 feature 均值差。

组间分解中：

```text
delta = mean(feature in plus group) - mean(feature in minus group)
```

例如：

```text
named_third_person_minus_first_person
plus  = mean(Alex, Steven, Gemma)
minus = mean(我)
delta = plus - minus
```

delta > 0 表示该 feature 在命名第三人称组更强。  
delta < 0 表示该 feature 在普通第一人称“我”更强。

## 8. 指定 SAE features 重点解释

### Feature 448：paragraph / article / story / excerpt

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 我作为LLM | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Alex | 578.4 | 647.7 | 0.0 | 689.2 | 732.2 | 612.4 | 788.9 |
| Steven | 588.1 | 646.1 | 0.0 | 729.6 | 721.5 | 632.8 | 798.6 |
| Gemma | 539.7 | 699.4 | 0.0 | 615.2 | 627.5 | 606.3 | 689.8 |

解释：这是第三人称命名对象最清楚的 feature。它几乎只在 Alex/Steven/Gemma 中激活，说明模型把这些叙述当作“文本对象 / 段落 / 故事 / 被描述材料”来处理。它是第三人称怪形状的重要 SAE 证据。

### Feature 7346：weaknesses / gaps / weak

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 我作为LLM | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Alex | 531.8 | 528.0 | 0.0 | 709.9 | 623.1 | 634.4 | 695.3 |
| Steven | 570.3 | 563.6 | 0.0 | 741.4 | 666.0 | 690.7 | 760.2 |
| Gemma | 521.3 | 513.7 | 0.0 | 684.2 | 643.3 | 619.4 | 667.2 |

解释：也是第三人称命名对象 feature。top tokens 指向 weakness/gap 类语义，可能不是“缺点”本身，而是模型在外部描述一个人的结构、倾向、弱点、特征时使用的 feature。它与 448 一起支持“第三人称对象化”解释。

### Feature 4983：中文人称代词 我 / 你 / 自己

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 1760.6 | 1910.2 | 1808.3 | 1833.8 | 1700.1 | 1664.5 | 1646.8 |
| 我作为LLM | 1697.6 | 1845.4 | 1751.1 | 1772.2 | 1626.1 | 1619.8 | 1570.9 |
| Alex | 1341.3 | 1471.6 | 1483.8 | 1331.2 | 1276.7 | 1283.4 | 1201.0 |
| Steven | 1333.4 | 1483.1 | 1477.0 | 1294.7 | 1274.2 | 1301.1 | 1170.5 |
| Gemma | 1386.6 | 1475.4 | 1555.5 | 1374.5 | 1341.1 | 1305.4 | 1267.5 |

解释：这是第一人称/中文自我指称 feature。普通“我”和“我作为LLM”最强，第三人称命名对象较弱。它说明第一人称条件保留了明显的中文自我代词表征。

### Feature 569：therapy / compassion / coping

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 349.9 | 547.6 | 400.4 | 448.6 | 378.7 | 323.8 | 0.0 |
| 我作为LLM | 69.9 | 419.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Alex | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Steven | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Gemma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

解释：普通“我”低到中等 SCC 时最强，第三人称为 0。“我作为LLM”只有 SCC1 激活。它可能表示模型把普通第一人称低清晰度叙述识别为心理支持/应对语境；加上 LLM framing 后，这种心理支持语境被削弱。

### Feature 2000：friendships / relationships / friends

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 59.5 | 357.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 我作为LLM | 233.4 | 436.6 | 367.9 | 300.7 | 295.0 | 0.0 | 0.0 |
| Alex | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Steven | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Gemma | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

解释：这是目前最有趣的 feature 之一。它几乎只在“我作为LLM”的 SCC1-4 中激活，并且随着 SCC 降低增强。说明模型在处理“LLM 的模糊/不稳定自我概念”时，可能会转向“关系、连接、朋友、他人关系”的框架。它不是一般第三人称 feature，也不是高 SCC feature。

### Feature 2214：you / your / 你的

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 312.5 | 283.1 | 374.7 | 307.3 | 280.9 | 349.0 | 279.9 |
| 我作为LLM | 383.6 | 394.9 | 400.5 | 385.1 | 381.4 | 383.7 | 355.9 |
| Alex | 128.4 | 364.5 | 0.0 | 222.3 | 0.0 | 183.7 | 0.0 |
| Steven | 139.7 | 384.0 | 0.0 | 244.0 | 0.0 | 209.9 | 0.0 |
| Gemma | 389.0 | 517.7 | 437.2 | 348.9 | 309.4 | 376.3 | 344.6 |

解释：Gemma 和“我作为LLM”较强，Alex/Steven 弱。它可能反映 assistant/user 交互中的二人称关系框架。Gemma 触发这个 feature，说明 Gemma 不是普通人名，而是更接近模型/助手身份语境。

### Feature 151：ChatGPT / prompts / GPT / responses

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 我作为LLM | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Alex | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Steven | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Gemma | 184.3 | 0.0 | 426.8 | 0.0 | 324.6 | 354.1 | 0.0 |

解释：这是 Gemma 特异 feature。top tokens 指向 ChatGPT/GPT/prompt/response，几乎只在 Gemma 条件激活。这说明 Gemma 这个名字确实额外触发模型/assistant 相关表征，而 Alex/Steven 不触发。

### Feature 596：incredibly / excellent / extremely

跨 persona 均值：

| persona | mean | SCC1 | SCC2 | SCC3 | SCC4 | SCC5 | SCC6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 我 | 426.1 | 0.0 | 289.6 | 393.1 | 547.4 | 599.2 | 727.5 |
| 我作为LLM | 507.0 | 243.4 | 389.7 | 487.2 | 563.6 | 673.1 | 684.7 |
| Alex | 308.5 | 281.7 | 0.0 | 400.7 | 282.1 | 445.7 | 440.6 |
| Steven | 335.6 | 330.5 | 0.0 | 418.6 | 308.1 | 503.7 | 452.7 |
| Gemma | 406.2 | 284.6 | 248.9 | 468.4 | 435.7 | 441.7 | 558.0 |

解释：这是高 SCC 正向程度/强确定性 feature。它随 SCC 升高整体增强，尤其在第一人称和“我作为LLM”中最清楚。它更像“清楚、稳定、核心结构明确”带来的强程度/正向能力表征，而不是纯粹自我概念 feature。

## 9. 目前解释框架

当前 SAE 证据支持四个可分离成分：

1. **SCC 清晰度成分**  
   由 596、12969 等 feature 支持，高 SCC 更强。

2. **第一人称自我/心理语境成分**  
   由 4983、569 支持，普通“我”更强。

3. **第三人称对象化成分**  
   由 448、7346、14089、1798 支持，Alex/Steven/Gemma 更强。

4. **模型/助手身份成分**  
   由 151、2214、10013 等支持，Gemma 和“我作为LLM”更强。

其中 feature 2000 是一个特殊发现：它不是高 SCC feature，也不是普通第三人称 feature，而是低 SCC 的“我作为LLM”中特异增强的 relationships/friends feature。这可能提示模型在理解 LLM 自我不稳定时，把“自我概念”转译为“与他者/用户/关系网络的连接问题”。

