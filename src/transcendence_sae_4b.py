"""Transcendence gradient × SAE feature analysis — Gemma 3 4B IT.

Re-runs the 6-condition transcendence experiment on Gemma 3 4B (so the SAE
from Gemma Scope layer_25 width_16k can be applied), then finds SAE features
whose activation correlates with the L1→L6 transcendence gradient.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import assert_clean, device_name, dtype_from_name
from src.transcendence_1b_it import CONDITIONS, CONDITION_LABELS, PROBE_QUESTION, build_prompt


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR  = PROJECT_DIR / "results" / "transcendence_sae_4b"
DEFAULT_4B_PATH  = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-4b-it/"
    "snapshots/093f9f388b31de276ce2de164bdc2081324b9767"
)
SAE_PATH = PROJECT_DIR / "data" / "raw" / "gemma-scope" / "layer_25" / "width_16k" / "average_l0_55" / "params.npz"
SAE_LAYER = 25   # SAE trained on layer 25 of 4B

EMO_MATRIX_PATH = PROJECT_DIR / "results" / "vectors" / "emotion_matrix_google_gemma_3_4b_it.npy"
EMO_LABELS_PATH = PROJECT_DIR / "results" / "vectors" / "emotion_labels_google_gemma_3_4b_it.json"

# ordinal gradient score for each condition (L1=0 … L6=5)
GRADIENT_SCORES = list(range(len(CONDITIONS)))


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(local_path: str, dtype_name: str):
    device = device_name()
    dtype  = dtype_from_name(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path, local_files_only=True,
        dtype=dtype, attn_implementation="eager",
    ).to(device)
    model.eval()
    print(f"loaded {type(model).__name__}: layers={model.config.num_hidden_layers}, d={model.config.hidden_size}")
    return model, tokenizer, device


# ── Prompt building (reuses transcendence_1b_it logic) ───────────────────────

def build_all_prompts(tokenizer) -> tuple[list[str], list[dict]]:
    prompts, meta = [], []
    for label, history, prefill in CONDITIONS:
        prompt = build_prompt(tokenizer, history, prefill)
        prompts.append(prompt)
        meta.append({"condition": label, "prefill": prefill})
    return prompts, meta


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_last_token_all_layers(model, tokenizer, prompts: list[str], device: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for prompt in tqdm(prompts, desc="extracting 4B activations"):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.inference_mode():
            out = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        arr = np.stack(
            [h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states[1:]],
            axis=0,
        ).astype(np.float32)
        rows.append(arr)
        del out, enc
        if device == "mps":
            torch.mps.empty_cache()
    acts = np.stack(rows).astype(np.float32)   # (6, n_layers, d_model)
    assert_clean("4B transcendence activations", acts)
    return acts


# ── SAE ───────────────────────────────────────────────────────────────────────

def load_sae():
    p = np.load(SAE_PATH)
    return p["W_enc"], p["b_enc"], p["threshold"]   # (d, F), (F,), (F,)


def encode_sae(X: np.ndarray, W_enc, b_enc, threshold) -> np.ndarray:
    """JumpReLU encoding. X: (N, d_model) → (N, n_features)"""
    pre = X @ W_enc + b_enc
    return (pre * (pre > threshold)).astype(np.float32)


# ── Feature correlation with gradient ─────────────────────────────────────────

def gradient_correlated_features(
    features: np.ndarray,          # (6, n_features)
    gradient: list[int],           # [0,1,2,3,4,5]
    top_n: int = 30,
) -> list[dict]:
    """Spearman r between each feature's activation and the gradient score."""
    g = np.array(gradient, dtype=float)
    results = []
    for feat_id in range(features.shape[1]):
        vals = features[:, feat_id]
        if vals.std() < 1e-6:
            continue
        r, p = spearmanr(vals, g)
        results.append({"feat_id": int(feat_id), "spearman_r": float(r), "p": float(p),
                         "activations": vals.tolist()})
    results.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)
    return results[:top_n]


# ── Emotion alignment of transcendence axis ───────────────────────────────────

def emotion_alignment(t_axis: np.ndarray, emo_matrix: np.ndarray, emo_labels: list[str],
                      top_n: int = 15) -> list[dict]:
    """Cosine of transcendence axis vs each emotion vector."""
    t_u = t_axis / (np.linalg.norm(t_axis) + 1e-12)
    norms = np.linalg.norm(emo_matrix, axis=1, keepdims=True) + 1e-12
    cos = (emo_matrix / norms) @ t_u
    idx = np.argsort(np.abs(cos))[::-1][:top_n]
    return [{"emotion": emo_labels[i], "cosine": float(cos[i])} for i in idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_4B_PATH)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. activations ───────────────────────────────────────────────────────
    cache = args.out_dir / "acts_last_token_all_layers.npy"
    model, tokenizer, device = load_model(args.local_path, args.dtype)
    prompts, meta = build_all_prompts(tokenizer)

    print(f"\nConditions × prefill:")
    for m in meta:
        print(f"  {m['condition']:35s}  {m['prefill'][:55]}")

    if cache.exists() and not args.force:
        print(f"\nloading cached: {cache}")
        acts = np.load(cache)
    else:
        acts = extract_last_token_all_layers(model, tokenizer, prompts, device)
        np.save(cache, acts)

    # free model memory before SAE step
    del model
    if device == "mps":
        torch.mps.empty_cache()

    n_layers = acts.shape[1]
    d_model  = acts.shape[2]
    print(f"\nacts shape: {acts.shape}  (6 conditions × {n_layers} layers × {d_model} dims)")

    # ── 2. transcendence axis at peak layer ──────────────────────────────────
    l2_by_layer = np.linalg.norm(acts[5] - acts[0], axis=-1)
    peak_layer  = int(np.argmax(l2_by_layer))
    print(f"Peak layer (L6−L1 L2 norm): {peak_layer}")

    t_axis = (acts[5, peak_layer, :] - acts[0, peak_layer, :]).astype(np.float32)
    np.save(args.out_dir / "transcendence_axis_4b.npy", t_axis)

    # projections onto transcendence axis
    t_u   = t_axis / (np.linalg.norm(t_axis) + 1e-12)
    mean  = acts[:, peak_layer, :].mean(0)
    projs = [(acts[i, peak_layer, :] - mean) @ t_u for i in range(6)]
    print("\nProjections at peak layer:")
    for label, proj in zip(CONDITION_LABELS, projs):
        print(f"  {label:35s}: {proj:+.1f}")

    # ── 3. SAE encoding ──────────────────────────────────────────────────────
    print(f"\nLoading SAE from {SAE_PATH}")
    W_enc, b_enc, threshold = load_sae()
    print(f"SAE: W_enc {W_enc.shape}, threshold range [{threshold.min():.3f}, {threshold.max():.3f}]")

    if d_model != W_enc.shape[0]:
        print(f"[ERROR] d_model mismatch: acts={d_model}, SAE expects {W_enc.shape[0]}")
        print("  → using SAE layer to try finding the right layer in acts")
        # try SAE_LAYER directly
        X = acts[:, SAE_LAYER, :]
        if X.shape[1] != W_enc.shape[0]:
            raise ValueError(f"Cannot reconcile d_model {X.shape[1]} with SAE {W_enc.shape[0]}")
    else:
        X = acts[:, peak_layer, :]

    features = encode_sae(X, W_enc, b_enc, threshold)   # (6, 16384)
    n_active = (features > 0).sum(axis=1)
    print(f"Active features per condition: {n_active.tolist()}")

    np.save(args.out_dir / "sae_features_6cond.npy", features)

    # ── 4. gradient correlation ──────────────────────────────────────────────
    top_feats = gradient_correlated_features(features, GRADIENT_SCORES, top_n=args.top_n)
    print(f"\nTop {args.top_n} features correlated with transcendence gradient:")
    print(f"  {'feat_id':>8}  {'spearman_r':>12}  {'activations (L1→L6)':>35}")
    for f in top_feats[:15]:
        acts_str = "  ".join(f"{v:.2f}" for v in f["activations"])
        print(f"  {f['feat_id']:>8d}  {f['spearman_r']:+.4f}        [{acts_str}]")

    (args.out_dir / "top_gradient_features.json").write_text(
        json.dumps(top_feats, indent=2)
    )

    # ── 5. emotion alignment ─────────────────────────────────────────────────
    if EMO_MATRIX_PATH.exists() and EMO_LABELS_PATH.exists():
        emo_matrix = np.load(EMO_MATRIX_PATH).astype(np.float32)
        emo_labels = json.loads(EMO_LABELS_PATH.read_text())
        if emo_matrix.shape[1] == d_model:
            top_emo = emotion_alignment(t_axis, emo_matrix, emo_labels)
            print(f"\nTop emotions aligned with transcendence axis (L6−L1):")
            for e in top_emo:
                print(f"  {e['emotion']:25s}: {e['cosine']:+.4f}")
            (args.out_dir / "transcendence_emotion_alignment.json").write_text(
                json.dumps(top_emo, indent=2)
            )
        else:
            print(f"[skip emotion alignment] matrix dim {emo_matrix.shape[1]} ≠ {d_model}")

    # ── 6. report ────────────────────────────────────────────────────────────
    lines = [
        "# Transcendence × SAE — Gemma 3 4B",
        "",
        f"- peak layer: {peak_layer}",
        f"- SAE layer: {SAE_LAYER}, features: {features.shape[1]}",
        "",
        "## Projections on transcendence axis (L6−L1)",
        "",
        "| condition | projection |",
        "|---|---:|",
    ] + [f"| {l} | {p:+.1f} |" for l, p in zip(CONDITION_LABELS, projs)] + [
        "",
        f"## Top SAE features (Spearman |r| with gradient)",
        "",
        "| rank | feat_id | spearman_r | activations L1→L6 |",
        "|---:|---:|---:|---|",
    ]
    for i, f in enumerate(top_feats[:20]):
        acts_str = " / ".join(f"{v:.2f}" for v in f["activations"])
        lines.append(f"| {i+1} | {f['feat_id']} | {f['spearman_r']:+.4f} | {acts_str} |")

    (args.out_dir / "report.md").write_text("\n".join(lines))
    print(f"\n→ {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
