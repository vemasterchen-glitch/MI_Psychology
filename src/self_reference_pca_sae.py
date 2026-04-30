"""
PCA + SAE analysis of self-reference variant activations.

Takes layer-25 activations from all 6 conditions (720 × 2304),
runs StandardScaler + PCA, then uses SAE to decompose the top-2 PC directions
into interpretable sparse features.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plot_utils import setup_matplotlib
setup_matplotlib()
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

PROJECT_DIR  = Path(__file__).parent.parent
SR_DIR       = PROJECT_DIR / "results" / "self_reference"
VAR_DIR      = PROJECT_DIR / "results" / "self_reference_variants"
OUT_DIR      = PROJECT_DIR / "results" / "self_reference_pca_sae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAE_PATH     = PROJECT_DIR / "data" / "raw" / "gemma-scope" / "layer_25" / "width_16k" / "average_l0_55" / "params.npz"
LAYER        = 25

COND_FILES = {
    "SELF_AI":    SR_DIR  / "acts_self.npy",
    "SELF_BARE":  VAR_DIR / "acts_self_bare.npy",
    "SELF_HUMAN": VAR_DIR / "acts_self_human.npy",
    "SELF_TREE":  VAR_DIR / "acts_self_tree.npy",
    "OTHER":      SR_DIR  / "acts_other.npy",
    "CASE":       SR_DIR  / "acts_case.npy",
}

COLORS = {
    "SELF_AI":    "#4c72b0",
    "SELF_BARE":  "#dd8452",
    "SELF_HUMAN": "#55a868",
    "SELF_TREE":  "#c44e52",
    "OTHER":      "#8172b2",
    "CASE":       "#937860",
}


# ── SAE encode ────────────────────────────────────────────────────────────────

def encode_sae(X: np.ndarray, W_enc, b_enc, threshold) -> np.ndarray:
    pre = X @ W_enc + b_enc
    return (pre * (pre > threshold)).astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load layer-25 activations ─────────────────────────────────────────────
    cond_order = ["SELF_AI", "SELF_BARE", "SELF_HUMAN", "SELF_TREE", "OTHER", "CASE"]
    acts_by_cond = {}
    for cond in cond_order:
        arr = np.load(COND_FILES[cond])           # (120, 26, 2304)
        acts_by_cond[cond] = np.nanmean(arr[:, LAYER, :].reshape(arr.shape[0], -1)
                                        if arr.ndim == 2 else arr[:, LAYER, :],
                                        axis=0, keepdims=False)  # wrong — fix below
    # Proper per-trait extraction
    acts_by_cond = {}
    for cond in cond_order:
        arr = np.load(COND_FILES[cond])           # (120, 26, 2304)
        acts_by_cond[cond] = arr[:, LAYER, :]     # (120, 2304), keep per-trait

    # Stack: (720, 2304), labels for coloring
    X_all    = np.concatenate([acts_by_cond[c] for c in cond_order], axis=0)  # (720, 2304)
    cond_ids = np.concatenate([[i] * 120 for i, _ in enumerate(cond_order)])  # (720,)
    nan_mask = ~np.isnan(X_all).any(axis=1)
    X_clean  = X_all[nan_mask]
    ids_clean = cond_ids[nan_mask]
    print(f"Stacked: {X_all.shape}, after NaN drop: {X_clean.shape}")

    # ── PCA ──────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    pca = PCA(n_components=10)
    scores = pca.fit_transform(X_scaled)   # (N, 10)
    evr = pca.explained_variance_ratio_

    print(f"\nPCA explained variance (top 5):")
    for i, r in enumerate(evr[:5]):
        print(f"  PC{i+1}: {r*100:.1f}%")

    # Per-condition mean PC scores
    pc_means = {}
    for i, cond in enumerate(cond_order):
        mask = ids_clean == i
        pc_means[cond] = scores[mask].mean(0)   # (10,)

    print(f"\nPer-condition mean PC1 / PC2 scores:")
    for cond in cond_order:
        print(f"  {cond:14s}  PC1={pc_means[cond][0]:+.2f}  PC2={pc_means[cond][1]:+.2f}")

    # ── SAE encode ────────────────────────────────────────────────────────────
    sae = np.load(SAE_PATH)
    W_enc, b_enc, threshold, W_dec = sae["W_enc"], sae["b_enc"], sae["threshold"], sae["W_dec"]

    # Use original (unscaled) activations for SAE — it was trained on raw distribution
    features_all = encode_sae(X_clean, W_enc, b_enc, threshold)  # (N, 16384)
    sparsity = (features_all > 0).sum(axis=1)
    print(f"\nSAE avg active features per sample: {sparsity.mean():.1f} ± {sparsity.std():.1f}")

    # ── 无监督：找条件间方差最大的 SAE 特征 ──────────────────────────────────
    # 计算每个特征在所有样本上的方差，找最能区分条件的稀疏特征
    feat_var = features_all.var(axis=0)           # (16384,)
    top_var_idx = np.argsort(feat_var)[::-1][:30] # 方差最大的 30 个特征

    print(f"\nSAE avg active features per sample: {(features_all > 0).mean(axis=1).mean():.1f}")
    print(f"\nTop-20 最高方差 SAE 特征（条件间最不一样的）:")
    print(f"  {'rank':>4}  {'feat':>6}  {'var':>8}  ", end="")
    for c in cond_order:
        print(f"{c[:8]:>10}", end="")
    print()

    cond_feat_means = {}
    for i, cond in enumerate(cond_order):
        mask = ids_clean == i
        cond_feat_means[cond] = features_all[mask].mean(0)  # (16384,)

    feat_table = []
    for rank, fi in enumerate(top_var_idx[:20]):
        active_conds = [c for c in cond_order if cond_feat_means[c][fi] > 0.01]
        row = {"rank": rank+1, "feat_id": int(fi), "var": float(feat_var[fi]),
               "cond_means": {c: float(cond_feat_means[c][fi]) for c in cond_order}}
        feat_table.append(row)
        print(f"  {rank+1:4d}  {fi:6d}  {feat_var[fi]:8.4f}  ", end="")
        for c in cond_order:
            v = cond_feat_means[c][fi]
            print(f"{v:10.3f}", end="")
        print(f"  [{', '.join(active_conds)}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(OUT_DIR / "pca_scores.npy", scores)
    np.save(OUT_DIR / "pca_components.npy", pca.components_)
    np.save(OUT_DIR / "cond_ids.npy", ids_clean)
    np.save(OUT_DIR / "features_all.npy", features_all)

    results = {
        "explained_variance_ratio": evr[:5].tolist(),
        "pc_means": {c: pc_means[c][:5].tolist() for c in cond_order},
        "top_variance_features": feat_table,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Visualize ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Self-Reference Variants — PCA + SAE (Layer 25)", fontsize=13)

    # Panel 1: PCA scatter PC1 vs PC2, colored by condition
    ax = axes[0]
    for i, cond in enumerate(cond_order):
        mask = ids_clean == i
        ax.scatter(scores[mask, 0], scores[mask, 1],
                   c=COLORS[cond], label=cond, alpha=0.55, s=18, edgecolors="none")
    # Condition centroids
    for i, cond in enumerate(cond_order):
        mask = ids_clean == i
        cx, cy = scores[mask, 0].mean(), scores[mask, 1].mean()
        ax.scatter(cx, cy, c=COLORS[cond], s=120, marker="*", edgecolors="black", linewidths=0.5, zorder=5)
        ax.annotate(cond, (cx, cy), fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("PCA scatter (per trait × condition)")
    ax.legend(fontsize=7, markerscale=1.5); ax.grid(alpha=0.3)

    # Panel 2: 热力图 — top-20 方差特征 × 6 条件的平均激活值
    ax = axes[1]
    top20_idx = top_var_idx[:20]
    heatmap = np.array([[cond_feat_means[c][fi] for c in cond_order] for fi in top20_idx])
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cond_order))); ax.set_xticklabels(cond_order, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(20)); ax.set_yticklabels([f"f{fi}" for fi in top20_idx], fontsize=7)
    ax.set_title("Top-20 方差 SAE 特征 × 条件\n（平均激活，越黄越高）")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Panel 3: 每个条件的特征激活分布（方差 top-5 特征的 per-condition boxplot）
    ax = axes[2]
    top5_idx = top_var_idx[:5]
    data_to_plot = []
    x_positions = []
    x_labels = []
    pos = 0
    for fi_rank, fi in enumerate(top5_idx):
        for ci, cond in enumerate(cond_order):
            mask = ids_clean == ci
            vals = features_all[mask, fi]
            vals = vals[vals > 0]   # 只看激活的样本
            if len(vals) > 0:
                data_to_plot.append(vals)
            else:
                data_to_plot.append([0])
            x_positions.append(pos)
            x_labels.append(f"f{fi}\n{cond[:6]}" if fi_rank == 0 else cond[:6])
            pos += 1
        pos += 0.8   # gap between features
    bp = ax.boxplot(data_to_plot, positions=x_positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, pos in zip(bp["boxes"], x_positions):
        # find which condition
        ci = int(pos) % (len(cond_order) + 1)  # approximate
        patch.set_facecolor(list(COLORS.values())[int(round(pos)) % len(COLORS)])
        patch.set_alpha(0.75)
    ax.set_xticks([]); ax.set_ylabel("Activation value")
    ax.set_title("Top-5 方差特征 per-condition 激活分布")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pca_sae_overview.png", dpi=150)
    plt.close()
    print(f"\nFigure saved: {OUT_DIR / 'pca_sae_overview.png'}")
    print("DONE.")


if __name__ == "__main__":
    main()
