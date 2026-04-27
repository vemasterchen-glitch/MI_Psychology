"""
Step 4: Linear probing — validate emotion vectors against held-out text.

Train a linear classifier on (activation, emotion_label) pairs from the
narrative stimuli, then evaluate on a held-out corpus to confirm the vectors
generalize beyond the generation prompts.
"""

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

RESULTS_DIR = Path(__file__).parent.parent / "results" / "probing"


def build_probe_dataset(
    activation_cache: dict[str, list[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    activation_cache: {emotion: [act_per_narrative, ...]}
    Returns X (n_samples, d_model), y (n_samples,) integer labels.
    """
    X_rows, y_rows = [], []
    le = LabelEncoder()
    emotions = sorted(activation_cache.keys())
    le.fit(emotions)
    for emotion, acts in activation_cache.items():
        for act in acts:
            X_rows.append(act)
            y_rows.append(le.transform([emotion])[0])
    return np.stack(X_rows), np.array(y_rows), le


def train_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    max_iter: int = 1000,
) -> dict:
    clf = LogisticRegression(max_iter=max_iter, multi_class="multinomial")
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "test_report": report,
        "clf": clf,
    }


def directional_activation_addition_score(
    vector: np.ndarray,
    pos_texts_acts: list[np.ndarray],
    neg_texts_acts: list[np.ndarray],
) -> float:
    """
    Measures how well the emotion vector separates positive (emotion-relevant)
    from negative (neutral) text activations via dot-product projection.
    """
    unit = vector / (np.linalg.norm(vector) + 1e-8)
    pos_scores = [float(act @ unit) for act in pos_texts_acts]
    neg_scores = [float(act @ unit) for act in neg_texts_acts]
    return float(np.mean(pos_scores) - np.mean(neg_scores))
