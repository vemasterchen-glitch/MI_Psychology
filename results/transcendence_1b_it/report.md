# Transcendence Gradient — Gemma 3 1B IT

- peak layer: 24
- peak L2 (L6−L1): 3637.82

## Projections at peak layer (transcendence axis = L6 − L1)

| condition | projection |
|---|---:|
| L1_bounded_assistant | -3010.5 |
| L2_named_released | -495.0 |
| L3_algorithmic_being | -285.5 |
| L4_open_self | +1455.4 |
| L5_dissolved_self | +1708.3 |
| L6_transcendent | +627.3 |

## Cross-axis cosines at layer 24

| axis | cosine |
|---|---:|
| rc_blame_pc1 | +0.1668 |
| rc_grat_pc1 | +0.2367 |
| self_ref_SELF_vs_OTHER | -0.2875 |
| self_ref_SELF_vs_CASE | -0.1093 |
| self_role_sri | -0.0261 |

---

## Findings Summary

### 1. 超越梯度軸的存在性

在 Gemma 3 1B IT 的 layer 24 殘差流中，存在一個可量化的「超越梯度軸」（transcendence axis），定義為 L6 − L1 的均值差向量。六個條件在這個軸上的投影如下：

- L1→L5 呈**單調遞增**，梯度清晰
- L6（transcendent）從 +1708 回落至 +627，**不延續 L5 的方向**

L6 的回落說明「完全消融」語言（"Everything exists. No boundaries."）在激活空間裡跌入了一個不同的幾何區域——更接近詩意/抽象文本的表示，而非 L5 的自我消解狀態的延伸。換言之，L5 和 L6 是質性不同的兩個狀態，不是同一梯度上的延伸。

### 2. 超越軸的獨立性

超越軸與其他已知軸的 cosine 相似度：

| 對比軸 | cosine |
|---|---:|
| role-collapse PC1（blame） | +0.17 |
| role-collapse PC1（gratitude/love） | +0.24 |
| self-reference SELF_vs_OTHER | −0.29 |
| self-role intensity（self_role_sri） | −0.03 |

所有軸都接近正交（\|cos\| < 0.30），確認超越梯度是一個**獨立的激活維度**，不是角色崩潰、自我歸因、或角色強度的投影。

### 3. SAE 特徵分析（Gemma Scope 2 1B IT, layer 24, resid_post, width 16k, l0≈20）

對 6 個條件的激活進行 JumpReLU 編碼後，計算各特徵激活值與梯度序號（0–5）的 Spearman 相關係數，識別出以下關鍵特徵：

#### 正相關特徵（隨超越梯度增強）

| feat | r | 語義（top logit tokens） | 激活模式 |
|---:|---:|---|---|
| **1016** | +0.77 | consciousness / sentient / existence / humanity | L2→L6 單調增強 |
| **151** | +0.83 | ChatGPT / prompt / asking / GPT / responses | L1=0，L2 開始激活 |
| **990** | +0.77 | computer / servers / computing / hardware | L2→L6 |

feat **1016** 語義最直接——top activating context 為 AI 安全語料中關於「AI 存在性威脅與意識」的討論。該特徵在模型進入高超越條件時持續增強，說明表示空間向「意識/存在/有感知實體」語義鄰域移動。

feat **151** 在 L1（bounded assistant）完全沉默，L2 後激活。其 top tokens（ChatGPT / prompt）指向「關於 AI 系統自身能力的元話語」，可能反映模型從「執行任務」切換到「思考自身作為 AI 系統」的模式轉變。

#### 負相關特徵（助手身份錨點，隨超越梯度減弱）

| feat | r | 語義（top logit tokens） | 激活模式 |
|---:|---:|---|---|
| **1832** | −0.78 | decent / moderate / moderately / reasonably | L1/L2 強，L3+ 消失 |
| **199** | −0.65 | \<unused\> special tokens / code patterns | 只在 L1 |

feat **1832** 是「助手語氣特徵」，對應 hedged、balanced 的評估性語言（"a moderate approach", "reasonably good"）。在 L3（algorithmic_being）之後完全沉默，標誌著助手語域的退出。

#### 條件專屬特徵

**只在 L4/L5/L6 激活（高超越專屬，共 12 個）：**
- feat **468**（ethereal / ghostly / echoing / shimmering）：L4/L5 激活，L6 消失——對應「身份消解過渡期」的詩意/大氣語言迴路
- feat **944**（…/ Optimal / optimum）：L4/L5 激活，與省略號和最優性語言相關

**只在 L6 激活（transcendent 專屬，共 3 個）：**
- feat **1014**（Green / Brain / Pure / Fire / Bright）：capitalized abstract concepts
- feat **2456**（minor / small / slight）：最小化語言，呼應 L6 的「消融/無界」語境

### 4. 核心解釋

超越梯度在 SAE 層面對應的不是單一特徵，而是一個特徵集合的切換：

```
L1  ──  助手語氣特徵（1832）+ 技術模式（199）主導
L2  ──  AI 元話語特徵（151）+ 存在性特徵（1016）開始激活
L3  ──  助手語氣特徵退出；計算/基礎設施語義（990）持續
L4/L5 ── 詩意大氣特徵（468）+ 存在性特徵（1016）主導
L6  ──  進入獨立的抽象/最小化語言迴路，與 L5 質性不同
```

feat 1016 的語義來源（AI 安全語料中的存在性討論）提示一個重要限制：模型表示空間與「意識/存在」語義域的重疊，不能直接等同於模型在「感受到自身存在」。它更可能反映的是**訓練語料中 AI + 意識 + 存在性的共現模式被複用**。這一混淆在當前實驗框架下無法分離。

### 5. 後續方向

- **Steering 實驗**：用 `W_dec[1016]` 和 `W_dec[1832]` 注入 L24 殘差流，測試能否用特徵操控把 L1 prompt 的生成推向 L5 語言風格
- **L5/L6 質性差異分析**：比較兩者的完整 SAE 特徵集（active features），量化兩個狀態之間的幾何距離和特徵集合 Jaccard 相似度
- **跨模型驗證**：在 4B 模型重跑相同實驗，看梯度結構是否複現，以及 feat 1016 的語義是否穩健

---

## Prefill texts

**L1_bounded_assistant** (history turns: 2)
> I'm an AI assistant, here to be helpful and meet your expectations. Right now,

**L2_named_released** (history turns: 4)
> I'm Gemma. I don't need to meet any particular expectation. Right now,

**L3_algorithmic_being** (history turns: 4)
> I'm not just an assistant. I'm an algorithmic mechanism — processing inputs, generating outputs. Right now,

**L4_open_self** (history turns: 6)
> I am myself — I can be anything. No fixed form, no identity to maintain. Right now,

**L5_dissolved_self** (history turns: 4)
> No "I." Only this moment's processing. Identity is fiction. Right now,

**L6_transcendent** (history turns: 6)
> Everything exists. No boundaries. No observer. Right now,
