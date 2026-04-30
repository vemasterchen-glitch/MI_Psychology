"""
Emotion vector geometry analysis — VAD correlation.

Pipeline:
1. Load 171 emotion vectors
2. Match to NRC-VAD scores (valence / arousal / dominance)
3. PCA → PC1, PC2, PC3
4. 3x3 correlation matrix: PCs vs VAD dimensions
5. Supervised Ridge regression → find VAD directions in activation space
6. Plots: VAD scatter, correlation heatmap, PC projection
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.plot_utils import setup_matplotlib
setup_matplotlib()
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = Path(__file__).parent.parent / "results"
VECTORS_DIR = RESULTS_DIR / "vectors"
VAD_DIR = RESULTS_DIR / "vad"
NRC_PATH = (
    Path(__file__).parent.parent
    / "data/raw/NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt"
)

# Multi-word / compound emotion terms → best single-word proxy for VAD lookup
ALIASES = {
    "at ease":       "ease",
    "on edge":       "edgy",
    "worn out":      "exhausted",
    "grief-stricken":"grief",
    "self-confident":"confident",
    "self-conscious": "self-conscious",
    "self-critical":  "critical",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_vectors() -> tuple[np.ndarray, list[str]]:
    matrix = np.load(VECTORS_DIR / "emotion_matrix.npy")
    labels = json.loads((VECTORS_DIR / "emotion_labels.json").read_text())
    return matrix, labels


def load_nrc_vad() -> dict[str, tuple[float, float, float]]:
    """Returns {word: (valence, arousal, dominance)}, all in [0,1]."""
    vad: dict[str, tuple[float, float, float]] = {}
    with open(NRC_PATH) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                vad[parts[0]] = (float(parts[1]), float(parts[2]), float(parts[3]))
    return vad


def match_emotions_to_vad(
    labels: list[str],
    vad: dict[str, tuple[float, float, float]],
) -> tuple[list[int], np.ndarray]:
    """
    Returns (matched_indices, vad_scores) where matched_indices are the
    positions in `labels` that were found in NRC-VAD.
    vad_scores shape: (n_matched, 3) — columns: valence, arousal, dominance.
    """
    matched_idx, scores = [], []
    missing = []
    for i, label in enumerate(labels):
        key = ALIASES.get(label, label)
        if key in vad:
            matched_idx.append(i)
            scores.append(vad[key])
        else:
            # try without hyphen / first word
            alt = label.replace("-", " ").split()[0]
            if alt in vad:
                matched_idx.append(i)
                scores.append(vad[alt])
            else:
                missing.append(label)
    if missing:
        print(f"No VAD match for {len(missing)} emotions: {missing}")
    return matched_idx, np.array(scores)


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_pca(matrix: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, PCA]:
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)
    print(f"Variance explained: {pca.explained_variance_ratio_}")
    return coords, pca


def correlation_matrix(
    pc_coords: np.ndarray,         # (n, 3)
    vad_scores: np.ndarray,        # (n, 3)
    pc_labels: list[str] | None = None,
    vad_labels: list[str] | None = None,
) -> np.ndarray:
    """Compute 3×3 Pearson r matrix between PC scores and VAD scores."""
    pc_labels = pc_labels or ["PC1", "PC2", "PC3"]
    vad_labels = vad_labels or ["Valence", "Arousal", "Dominance"]
    corr = np.zeros((3, 3))
    print("\n── Correlation matrix (PC × VAD) ──")
    print(f"{'':8s}  {'Valence':>10s}  {'Arousal':>10s}  {'Dominance':>10s}")
    for i, pc_lbl in enumerate(pc_labels):
        for j, vad_lbl in enumerate(vad_labels):
            r, p = pearsonr(pc_coords[:, i], vad_scores[:, j])
            corr[i, j] = r
        print(
            f"{pc_lbl:8s}  {corr[i,0]:>+10.3f}  {corr[i,1]:>+10.3f}  {corr[i,2]:>+10.3f}"
        )
    return corr


def supervised_vad_directions(
    matrix: np.ndarray,    # all 171 vectors
    matched_idx: list[int],
    vad_scores: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Fit Ridge regression for each VAD dimension.
    Returns {'valence': direction_vec, 'arousal': ..., 'dominance': ...}.
    """
    X = matrix[matched_idx]
    directions = {}
    for j, dim in enumerate(["valence", "arousal", "dominance"]):
        y = vad_scores[:, j]
        clf = Ridge(alpha=1.0).fit(X, y)
        directions[dim] = clf.coef_ / (np.linalg.norm(clf.coef_) + 1e-8)
        # quick R² estimate via leave-nothing-out (just for reporting)
        y_pred = clf.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"Ridge R² for {dim}: {r2:.3f}")
    return directions


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_corr_heatmap(corr: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        corr,
        annot=True, fmt="+.3f", cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
        xticklabels=["Valence", "Arousal", "Dominance"],
        yticklabels=["PC1", "PC2", "PC3"],
        ax=ax,
    )
    ax.set_title("Pearson r: PCA components vs NRC-VAD dimensions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_vad_scatter(
    pc_coords: np.ndarray,
    vad_scores: np.ndarray,
    labels_matched: list[str],
    out_path: Path,
) -> None:
    """Scatter: PC1 vs PC2, colored by valence."""
    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(
        pc_coords[:, 0], pc_coords[:, 1],
        c=vad_scores[:, 0], cmap="RdYlGn",
        s=60, alpha=0.8, vmin=0, vmax=1,
    )
    for i, lbl in enumerate(labels_matched):
        ax.annotate(lbl, (pc_coords[i, 0], pc_coords[i, 1]),
                    fontsize=7, alpha=0.85, ha="center", va="bottom")
    plt.colorbar(sc, ax=ax, label="Valence (NRC-VAD)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Emotion vectors — PC1/PC2 projection, colored by valence")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(save_directions: bool = True) -> None:
    VAD_DIR.mkdir(parents=True, exist_ok=True)

    matrix, labels = load_vectors()
    vad = load_nrc_vad()
    matched_idx, vad_scores = match_emotions_to_vad(labels, vad)
    labels_matched = [labels[i] for i in matched_idx]
    print(f"\nMatched {len(matched_idx)}/{len(labels)} emotions to NRC-VAD")

    # PCA on matched subset only (for fair correlation)
    pc_coords, pca = run_pca(matrix[matched_idx])

    # 3×3 correlation
    corr = correlation_matrix(pc_coords, vad_scores)

    # Supervised directions (on full matched set)
    print("\n── Supervised Ridge regression ──")
    directions = supervised_vad_directions(matrix, matched_idx, vad_scores)

    # Save directions for use in steering
    if save_directions:
        np.save(VAD_DIR / "valence_direction.npy", directions["valence"])
        np.save(VAD_DIR / "arousal_direction.npy", directions["arousal"])
        np.save(VAD_DIR / "dominance_direction.npy", directions["dominance"])
        print("Saved VAD direction vectors to results/vad/")

    # Plots
    plot_corr_heatmap(corr, VAD_DIR / "pc_vad_correlation.png")
    plot_vad_scatter(pc_coords, vad_scores, labels_matched, VAD_DIR / "pc_valence_scatter.png")

    # Top emotions on each PC axis
    print("\n── Top emotions per PC ──")
    for i, pc_lbl in enumerate(["PC1", "PC2", "PC3"]):
        scores = pc_coords[:, i]
        top = sorted(zip(scores, labels_matched), reverse=True)
        print(f"\n{pc_lbl} high: {[l for _,l in top[:6]]}")
        print(f"{pc_lbl} low:  {[l for _,l in top[-6:]]}")


if __name__ == "__main__":
    run()
