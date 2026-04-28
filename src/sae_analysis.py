"""
SAE (Sparse Autoencoder) 分析：将情绪向量分解为 Gemma Scope 稀疏特征，
并与 NRC-VAD 评分做相关分析，对比 PCA 轴的效果。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).parent.parent / "data"
RESULTS    = Path(__file__).parent.parent / "results"
SAE_PATH   = DATA_DIR / "raw" / "gemma-scope" / "layer_25" / "width_16k" / "average_l0_55" / "params.npz"
EMO_MATRIX = RESULTS / "vectors" / "emotion_matrix.npy"
EMO_LIST   = DATA_DIR / "emotions.txt"
VAD_PATH   = DATA_DIR / "raw" / "NRC-VAD-Lexicon-Aug2018Release" / "NRC-VAD-Lexicon.txt"
OUT_DIR    = RESULTS / "sae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALIASES = {
    "at ease": "ease", "worn out": "wornout", "fed up": "fedup",
    "down": "downcast", "beat": "beaten", "blue": "blueish",
}


# ── 1. 加载数据 ────────────────────────────────────────────────────────────────

def load_emotions():
    return [l.strip() for l in open(EMO_LIST) if l.strip() and not l.startswith("#")]

def load_vad():
    vad = {}
    with open(VAD_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                vad[parts[0].lower()] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return vad

def load_sae():
    p = np.load(SAE_PATH)
    return p["W_enc"], p["b_enc"], p["threshold"]


# ── 2. SAE 编码（JumpReLU） ───────────────────────────────────────────────────

def encode_sae(activations: np.ndarray, W_enc, b_enc, threshold) -> np.ndarray:
    """
    activations : (N, d_model)
    returns     : (N, d_sae)  稀疏特征激活
    """
    pre_act = activations @ W_enc + b_enc      # (N, 16384)
    features = pre_act * (pre_act > threshold)  # JumpReLU
    return features.astype(np.float32)


# ── 3. VAD 相关分析 ────────────────────────────────────────────────────────────

def vad_correlation_for_directions(directions: np.ndarray,
                                   labels: list[str],
                                   vad_dict: dict,
                                   emotions: list[str],
                                   emotion_matrix: np.ndarray):
    """
    directions : (K, d) — K 个方向向量（单位化后）
    labels     : K 个名称
    返回 (K, 3) 相关矩阵 [valence, arousal, dominance]
    """
    matched_idx, valences, arousals, dominances = [], [], [], []
    for i, emo in enumerate(emotions):
        key = ALIASES.get(emo, emo).lower()
        if key in vad_dict:
            matched_idx.append(i)
            v, a, d = vad_dict[key]
            valences.append(v); arousals.append(a); dominances.append(d)

    sub = emotion_matrix[matched_idx]   # (M, d)
    corr_matrix = np.zeros((len(directions), 3))
    for k, direction in enumerate(directions):
        unit = direction / (np.linalg.norm(direction) + 1e-9)
        proj = sub @ unit
        corr_matrix[k, 0] = spearmanr(proj, valences)[0]
        corr_matrix[k, 1] = spearmanr(proj, arousals)[0]
        corr_matrix[k, 2] = spearmanr(proj, dominances)[0]
    return corr_matrix, matched_idx, valences, arousals, dominances


# ── 4. 主流程 ─────────────────────────────────────────────────────────────────

def main():
    emotions = load_emotions()
    vad_dict = load_vad()
    W_enc, b_enc, threshold = load_sae()

    raw = np.load(EMO_MATRIX)   # (171, 2304) 原始激活，SAE 必须用这个
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw)  # 仅用于 PCA 对比

    print(f"情绪数: {len(emotions)}, 维度: {raw.shape[1]}")
    print(f"SAE 特征数: {W_enc.shape[1]}, 平均激活数 ≈ 55/forward")

    # ── 4a. 编码所有情绪向量（用原始激活，SAE 训练时的输入分布） ─────────────
    features = encode_sae(raw, W_enc, b_enc, threshold)   # (171, 16384)
    sparsity = (features > 0).sum(axis=1)
    print(f"每情绪平均激活特征数: {sparsity.mean():.1f} ± {sparsity.std():.1f}")

    np.save(OUT_DIR / "emotion_features.npy", features)

    # ── 4b. 找每个情绪的 top-5 特征 ──────────────────────────────────────────
    top_k = 5
    top_results = {}
    for i, emo in enumerate(emotions):
        idx = np.argsort(features[i])[::-1][:top_k]
        top_results[emo] = [(int(j), float(features[i, j])) for j in idx if features[i, j] > 0]

    with open(OUT_DIR / "top_features_per_emotion.json", "w") as f:
        json.dump(top_results, f, indent=2, ensure_ascii=False)

    # ── 4c. 找跨情绪最判别性特征（方差最大） ───────────────────────────────────
    feat_var = features.var(axis=0)
    top_discriminative = np.argsort(feat_var)[::-1][:20]
    print("\n最判别性 SAE 特征（方差最大的 20 个）:")
    for rank, fid in enumerate(top_discriminative):
        active_emos = [emotions[i] for i in range(len(emotions)) if features[i, fid] > 0]
        print(f"  #{rank+1:2d} feat {fid:5d} | var={feat_var[fid]:.4f} | 激活情绪: {', '.join(active_emos[:6])}")

    # ── 4d. 用 SAE 特征做 VAD 方向搜索 ────────────────────────────────────────
    # 策略：对每个 VAD 维度，找与之相关系数最高的 SAE 特征方向（W_dec 列）
    matched_idx, valences, arousals, dominances = [], [], [], []
    for i, emo in enumerate(emotions):
        key = ALIASES.get(emo, emo).lower()
        if key in vad_dict:
            matched_idx.append(i)
            v, a, d = vad_dict[key]
            valences.append(v); arousals.append(a); dominances.append(d)

    sub_features = features[matched_idx]    # (M, 16384)，基于原始激活
    W_dec = np.load(SAE_PATH)["W_dec"]      # (16384, 2304)

    print(f"\nVAD 匹配情绪数: {len(matched_idx)}")

    # 对每个 SAE 特征，计算其激活值与 VAD 的 Pearson 相关
    feat_val_corr = np.array([spearmanr(sub_features[:, j], valences)[0]
                               if sub_features[:, j].std() > 1e-6 else 0.0
                               for j in range(sub_features.shape[1])])
    feat_aro_corr = np.array([spearmanr(sub_features[:, j], arousals)[0]
                               if sub_features[:, j].std() > 1e-6 else 0.0
                               for j in range(sub_features.shape[1])])
    feat_dom_corr = np.array([spearmanr(sub_features[:, j], dominances)[0]
                               if sub_features[:, j].std() > 1e-6 else 0.0
                               for j in range(sub_features.shape[1])])

    # 取绝对值最高的特征，用其 W_dec 方向做线性组合
    def top_direction(corrs, top_n=50):
        idx = np.argsort(np.abs(corrs))[::-1][:top_n]
        weights = corrs[idx]
        dirs = W_dec[idx]   # (top_n, 2304)
        d = (weights[:, None] * dirs).sum(axis=0)
        return d / (np.linalg.norm(d) + 1e-9), idx, corrs[idx]

    val_dir, val_feat_idx, val_feat_corrs = top_direction(feat_val_corr)
    aro_dir, aro_feat_idx, aro_feat_corrs = top_direction(feat_aro_corr)
    dom_dir, dom_feat_idx, dom_feat_corrs = top_direction(feat_dom_corr)

    print(f"\nTop-1 效价相关 SAE 特征: #{val_feat_idx[0]} (ρ={val_feat_corrs[0]:.3f})")
    print(f"Top-1 唤醒度相关 SAE 特征: #{aro_feat_idx[0]} (ρ={aro_feat_corrs[0]:.3f})")
    print(f"Top-1 支配度相关 SAE 特征: #{dom_feat_idx[0]} (ρ={dom_feat_corrs[0]:.3f})")

    # ── 4e. 对比 PCA 轴 vs SAE 合成方向的 VAD 相关系数 ────────────────────────
    from sklearn.decomposition import PCA

    pca = PCA(n_components=10)
    pca.fit(X_scaled)  # PCA 用 StandardScaler 处理过的版本（与之前 VAD 分析一致）
    pca_dirs = pca.components_[:3]  # (3, 2304)

    sae_dirs = np.stack([val_dir, aro_dir, dom_dir])   # (3, 2304)
    labels_pca = ["PC1", "PC2", "PC3"]
    labels_sae = ["SAE-Valence", "SAE-Arousal", "SAE-Dominance"]

    # PCA 和 SAE 方向投影都用原始激活空间（VAD 相关在同一空间比较才有意义）
    corr_pca, *_ = vad_correlation_for_directions(pca_dirs, labels_pca, vad_dict, emotions, raw)
    corr_sae, *_ = vad_correlation_for_directions(sae_dirs, labels_sae, vad_dict, emotions, raw)

    print("\n── PCA 轴与 VAD 相关系数 ──")
    print(f"{'':12s}  {'效价':>8}  {'唤醒度':>8}  {'支配度':>8}")
    for i, lbl in enumerate(labels_pca):
        print(f"  {lbl:10s}  {corr_pca[i,0]:+.3f}     {corr_pca[i,1]:+.3f}     {corr_pca[i,2]:+.3f}")

    print("\n── SAE 合成方向与 VAD 相关系数 ──")
    for i, lbl in enumerate(labels_sae):
        print(f"  {lbl:14s}  {corr_sae[i,0]:+.3f}     {corr_sae[i,1]:+.3f}     {corr_sae[i,2]:+.3f}")

    # ── 4f. 热力图对比 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    vad_labels = ["Valence", "Arousal", "Dominance"]

    im0 = axes[0].imshow(np.abs(corr_pca), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[0].set_xticks(range(3)); axes[0].set_xticklabels(vad_labels)
    axes[0].set_yticks(range(3)); axes[0].set_yticklabels(labels_pca)
    axes[0].set_title("PCA 轴 ↔ VAD（|r|）")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{corr_pca[i,j]:+.2f}", ha="center", va="center", fontsize=9)

    im1 = axes[1].imshow(np.abs(corr_sae), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(vad_labels)
    axes[1].set_yticks(range(3)); axes[1].set_yticklabels(labels_sae)
    axes[1].set_title("SAE 合成方向 ↔ VAD（|r|）")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{corr_sae[i,j]:+.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_vs_sae_vad.png", dpi=150)
    plt.close()
    print(f"\n对比热力图已保存: {OUT_DIR / 'pca_vs_sae_vad.png'}")

    # ── 4g. 保存 SAE 方向向量 ─────────────────────────────────────────────────
    np.save(OUT_DIR / "sae_valence_direction.npy", val_dir)
    np.save(OUT_DIR / "sae_arousal_direction.npy", aro_dir)
    np.save(OUT_DIR / "sae_dominance_direction.npy", dom_dir)

    print(f"\n完成。输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
