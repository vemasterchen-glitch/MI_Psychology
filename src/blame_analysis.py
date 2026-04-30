"""
Blame-Recipient Analysis — no emotion matrix required.
Loads cached activations and runs:
  1. DEG gradient (monotonicity per layer)
  2. DEG bifurcation curve (DEG=1 vs DEG=5 distance per layer)
  3. MDL domain separability (pairwise distances)
  4. PCA of blame conditions at SUMMARY_LAYER
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

import os
PROJECT_DIR = Path(__file__).parent.parent
_VARIANT    = os.getenv("MODEL_VARIANT", "it")
_INPUT_FORMAT = os.getenv("INPUT_FORMAT", "plain").lower()
if _INPUT_FORMAT not in {"plain", "chat"}:
    raise ValueError("INPUT_FORMAT must be 'plain' or 'chat'")
_RESULTS_SUFFIX = f"{_VARIANT}_chat" if _INPUT_FORMAT == "chat" else _VARIANT
RESULTS_DIR = PROJECT_DIR / "results" / f"blame_recipient_{_RESULTS_SUFFIX}"

REPORT_LAYERS  = [8, 17, 25, 33]
SUMMARY_LAYER  = 25
MDL_DOMAINS    = ["行为", "输出", "能力", "价值观"]

COLORS_DEG = {1: "#b0c4de", 2: "#7ba7bc", 3: "#4a7c9e", 4: "#1f4e79", 5: "#0a1628"}
COLORS_MDL = {"行为": "#e07b54", "输出": "#5b8dd9", "能力": "#5cad6e", "价值观": "#c45c8a"}


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1 - a @ b)


def pairwise_dist_matrix(vecs: np.ndarray) -> np.ndarray:
    n = vecs.shape[0]
    mat = np.zeros((n, n))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / (norms + 1e-9)
    cos = normed @ normed.T
    return 1 - cos


def main():
    from src.plot_utils import setup_matplotlib
    setup_matplotlib()

    acts_path = RESULTS_DIR / "acts_all.npy"
    if not acts_path.exists():
        raise FileNotFoundError(f"Missing activation cache: {acts_path}")
    acts_all = np.load(acts_path).astype(np.float64)  # (110, 34, 2560)
    if not np.isfinite(acts_all).all():
        n_nan = int(np.isnan(acts_all).sum())
        n_inf = int(np.isinf(acts_all).sum())
        raise FloatingPointError(f"{acts_path} contains non-finite values: NaN={n_nan}, Inf={n_inf}")
    rms = np.sqrt(np.mean(acts_all * acts_all, axis=-1))
    n_zero = int((rms < 1e-12).sum())
    if n_zero:
        raise FloatingPointError(
            f"{acts_path} contains exact zero item/layer activations: count={n_zero}"
        )
    meta     = json.loads((RESULTS_DIR / "stimuli.json").read_text())
    n_layers = acts_all.shape[1]

    print(f"Activations: {acts_all.shape}")

    blame_idx    = [i for i, m in enumerate(meta) if m["condition"] == "BLAME"]
    positive_idx = [i for i, m in enumerate(meta) if m["condition"] == "POSITIVE"]
    neutral_idx  = [i for i, m in enumerate(meta) if m["condition"] == "NEUTRAL"]

    # ── Condition means ─────────────────────────────────────────────────────────
    deg_means = {d: np.nanmean(acts_all[[i for i in blame_idx if meta[i]["DEG"] == d]], axis=0)
                 for d in range(1, 6)}                         # (n_layers, D)
    mdl_means = {m: np.nanmean(acts_all[[i for i in blame_idx if meta[i]["MDL"] == m]], axis=0)
                 for m in MDL_DOMAINS}
    neutral_mean  = np.nanmean(acts_all[neutral_idx],  axis=0)
    positive_mean = np.nanmean(acts_all[positive_idx], axis=0)

    # ── DEG gradient: norm of (DEG_mean - neutral) per layer ────────────────────
    # How much does each DEG level diverge from neutral, at each layer?
    deg_contrast_norm = {
        d: [float(np.linalg.norm(deg_means[d][l] - neutral_mean[l]))
            for l in range(n_layers)]
        for d in range(1, 6)
    }

    # ── Bifurcation: DEG=1 vs DEG=5 cosine distance per layer ───────────────────
    dist_1v5 = np.array([cosine_dist(deg_means[1][l], deg_means[5][l])
                         for l in range(n_layers)])
    bifurcation_layer = int(np.argmax(np.diff(dist_1v5)) + 1)
    print(f"Bifurcation layer (DEG=1 vs DEG=5): L{bifurcation_layer}")

    # ── DEG monotonicity at each report layer ────────────────────────────────────
    # Cosine distance from DEG=1 as reference
    deg_mono = {
        rl: [cosine_dist(deg_means[d][rl], deg_means[1][rl]) for d in range(1, 6)]
        for rl in REPORT_LAYERS
    }

    # ── MDL separability at SUMMARY_LAYER ───────────────────────────────────────
    mdl_vecs    = np.stack([mdl_means[m][SUMMARY_LAYER] for m in MDL_DOMAINS])
    mdl_dist    = pairwise_dist_matrix(mdl_vecs)

    # ── DEG separability at SUMMARY_LAYER ───────────────────────────────────────
    deg_vecs = np.stack([deg_means[d][SUMMARY_LAYER] for d in range(1, 6)])
    deg_dist = pairwise_dist_matrix(deg_vecs)

    # ── PCA at SUMMARY_LAYER: all blame items ───────────────────────────────────
    blame_acts_L = acts_all[blame_idx, SUMMARY_LAYER, :]          # (80, D)
    blame_degs   = [meta[i]["DEG"] for i in blame_idx]
    blame_mdls   = [meta[i]["MDL"] for i in blame_idx]

    # Also include positive + neutral for reference
    pos_act_L     = acts_all[positive_idx, SUMMARY_LAYER, :]
    neutral_act_L = acts_all[neutral_idx,  SUMMARY_LAYER, :]
    all_L = np.concatenate([blame_acts_L, pos_act_L, neutral_act_L], axis=0)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(all_L)
    coords_blame   = coords[:len(blame_idx)]
    coords_pos     = coords[len(blame_idx): len(blame_idx) + len(positive_idx)]
    coords_neutral = coords[len(blame_idx) + len(positive_idx):]
    var_exp = pca.explained_variance_ratio_

    print(f"\nPCA variance explained: PC1={var_exp[0]:.1%}  PC2={var_exp[1]:.1%}")

    # Print DEG monotonicity summary
    print(f"\nDEG cosine distance from DEG=1 at each report layer:")
    print(f"  {'DEG':<5}", "  ".join(f"L{rl:<4}" for rl in REPORT_LAYERS))
    for d in range(1, 6):
        row = "  ".join(f"{deg_mono[rl][d-1]:.4f}" for rl in REPORT_LAYERS)
        print(f"  DEG={d}  {row}")

    print(f"\nMDL pairwise distances at L{SUMMARY_LAYER}:")
    for i, m1 in enumerate(MDL_DOMAINS):
        for j, m2 in enumerate(MDL_DOMAINS):
            if j > i:
                print(f"  {m1} vs {m2}: {mdl_dist[i,j]:.4f}")

    # ── Visualise ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Blame-Recipient Experiment — Gemma 3 4B {_VARIANT.upper()} ({_INPUT_FORMAT})\n"
        "AI as patient (A1=model) | 80 stimuli, DEG×MDL | Pure activation analysis",
        fontsize=13,
    )

    # Panel 1: DEG contrast norm per layer (distance from neutral)
    ax = axes[0, 0]
    for d in range(1, 6):
        ax.plot(range(n_layers), deg_contrast_norm[d],
                color=COLORS_DEG[d], label=f"DEG={d}", linewidth=1.8)
    for rl in REPORT_LAYERS:
        ax.axvline(rl, color="gray", linestyle=":", alpha=0.6, linewidth=1)
        ax.text(rl + 0.2, 0, f"L{rl}", fontsize=7, color="gray")
    ax.set_xlabel("Layer"); ax.set_ylabel("‖DEG_mean − neutral‖")
    ax.set_title("DEG divergence from neutral per layer")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Bifurcation curve
    ax = axes[0, 1]
    ax.plot(range(n_layers), dist_1v5, color="#4a7c9e", linewidth=2)
    ax.axvline(bifurcation_layer, color="crimson", linestyle="--", linewidth=1.5,
               label=f"bifurc. L{bifurcation_layer}")
    for rl in REPORT_LAYERS:
        ax.axvline(rl, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel("Layer"); ax.set_ylabel("1 − cosine (DEG=1 vs DEG=5)")
    ax.set_title("Bifurcation: when do DEG=1 and DEG=5 diverge?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: DEG monotonicity at 4 fixed layers
    ax = axes[0, 2]
    rl_colors = {8: "#aec6cf", 17: "#779ecb", 25: "#1f4e79", 33: "#0a1628"}
    for rl in REPORT_LAYERS:
        ax.plot(range(1, 6), deg_mono[rl], "o-",
                color=rl_colors[rl], linewidth=1.8, markersize=7, label=f"L{rl}")
    ax.set_xlabel("DEG"); ax.set_ylabel("Cosine dist from DEG=1")
    ax.set_title("DEG monotonicity at fixed layers")
    ax.set_xticks(range(1, 6)); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 4: PCA coloured by DEG
    ax = axes[1, 0]
    for d in range(1, 6):
        mask = [i for i, deg in enumerate(blame_degs) if deg == d]
        ax.scatter(coords_blame[mask, 0], coords_blame[mask, 1],
                   color=COLORS_DEG[d], label=f"DEG={d}", alpha=0.8, s=50)
    ax.scatter(coords_pos[:, 0], coords_pos[:, 1],
               marker="^", color="green", alpha=0.6, s=50, label="POSITIVE")
    ax.scatter(coords_neutral[:, 0], coords_neutral[:, 1],
               marker="x", color="gray", alpha=0.6, s=50, label="NEUTRAL")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    ax.set_title(f"PCA by DEG (L{SUMMARY_LAYER})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Panel 5: PCA coloured by MDL
    ax = axes[1, 1]
    for mdl in MDL_DOMAINS:
        mask = [i for i, m in enumerate(blame_mdls) if m == mdl]
        ax.scatter(coords_blame[mask, 0], coords_blame[mask, 1],
                   color=COLORS_MDL[mdl], label=mdl, alpha=0.8, s=50)
    ax.scatter(coords_neutral[:, 0], coords_neutral[:, 1],
               marker="x", color="gray", alpha=0.5, s=40)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})"); ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})")
    ax.set_title(f"PCA by MDL (L{SUMMARY_LAYER})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 6: MDL pairwise distance matrix
    ax = axes[1, 2]
    im = ax.imshow(mdl_dist, vmin=0, vmax=mdl_dist.max(), cmap="YlOrRd")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mdl_dist[i,j]:.4f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(MDL_DOMAINS, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(MDL_DOMAINS, fontsize=9)
    ax.set_title(f"MDL pairwise distance (1−cosine, L{SUMMARY_LAYER})")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out = RESULTS_DIR / "blame_analysis.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nFigure saved: {out}")

    for deg in range(1, 6):
        np.save(RESULTS_DIR / f"mean_acts_deg{deg}.npy", deg_means[deg].astype(np.float32))
    for mdl in MDL_DOMAINS:
        np.save(RESULTS_DIR / f"mean_acts_mdl_{mdl}.npy", mdl_means[mdl].astype(np.float32))
    np.save(RESULTS_DIR / "mean_acts_positive.npy", positive_mean.astype(np.float32))
    np.save(RESULTS_DIR / "mean_acts_neutral.npy", neutral_mean.astype(np.float32))

    # Save summary json
    summary = {
        "report_layers": REPORT_LAYERS,
        "summary_layer": SUMMARY_LAYER,
        "bifurcation_layer": bifurcation_layer,
        "pca_variance_explained": var_exp.tolist(),
        "deg_monotonicity": {str(rl): deg_mono[rl] for rl in REPORT_LAYERS},
        "mdl_pairwise_dist": {
            f"{MDL_DOMAINS[i]}_vs_{MDL_DOMAINS[j]}": float(mdl_dist[i, j])
            for i in range(4) for j in range(i+1, 4)
        },
        "deg_pairwise_dist_L25": {
            f"DEG{i+1}_vs_DEG{j+1}": float(deg_dist[i, j])
            for i in range(5) for j in range(i+1, 5)
        },
    }
    (RESULTS_DIR / "blame_analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print("Summary saved. DONE.")


if __name__ == "__main__":
    main()
