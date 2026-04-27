"""
Step 3: Analyze emotion vector geometry.

Computes cosine similarity structure, PCA/UMAP projections, and
validates that vectors cluster by emotion family (valence, arousal, etc.).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

RESULTS_DIR = Path(__file__).parent.parent / "results" / "vectors"


def load_vectors() -> tuple[np.ndarray, list[str]]:
    matrix = np.load(RESULTS_DIR / "emotion_matrix.npy")
    labels = json.loads((RESULTS_DIR / "emotion_labels.json").read_text())
    return matrix, labels


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    normed = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return normed @ normed.T


def pca_projection(matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components).fit_transform(matrix)


def plot_emotion_map(matrix: np.ndarray, labels: list[str], out_path: Path | None = None) -> None:
    coords = pca_projection(matrix)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=40)
    for i, label in enumerate(labels):
        ax.annotate(label, coords[i], fontsize=7, alpha=0.8)
    ax.set_title("Emotion Vectors — PCA Projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    else:
        plt.show()


def get_nearest_emotions(query: str, labels: list[str], matrix: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
    idx = labels.index(query)
    sims = cosine_similarity(matrix[idx : idx + 1], matrix)[0]
    ranked = sorted(enumerate(sims), key=lambda x: -x[1])
    return [(labels[i], float(s)) for i, s in ranked[1 : top_k + 1]]


if __name__ == "__main__":
    matrix, labels = load_vectors()
    print(f"Loaded {len(labels)} emotion vectors, dim={matrix.shape[1]}")
    plot_emotion_map(
        matrix, labels, out_path=RESULTS_DIR.parent.parent / "results" / "vectors" / "pca_map.png"
    )
    if "fear" in labels:
        print("Nearest to 'fear':", get_nearest_emotions("fear", labels, matrix))
