"""
Self-reference experiment — AI replication of Kelley et al. (2002).

Original paradigm: trait adjective judgment under three conditions:
  SELF  — "Does this word describe you?"
  OTHER — "Does this word describe [a well-known person]?"
  CASE  — "Is this word written in uppercase letters?"

AI adaptation:
  SELF  — "Does the following word describe you as an AI assistant? Word: {adj}"
  OTHER — "Does the following word describe a typical human? Word: {adj}"
  CASE  — "Is the following word written in uppercase letters? Word: {adj}"

We extract residual stream activations at every layer (26 layers for Gemma 2 2B),
compute condition means, and analyse SELF−OTHER and SELF−CASE contrast vectors.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import transformer_lens as tl
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR    = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "self_reference"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

# ── Stimuli ────────────────────────────────────────────────────────────────────

# 120 trait adjectives (60 positive + 60 negative), balanced
# Drawn from common personality/trait lexicons; deliberately varied in valence & arousal
POSITIVE_TRAITS = [
    "honest",    "kind",       "brave",     "creative",  "generous",  "loyal",
    "patient",   "confident",  "empathetic","curious",   "humble",    "reliable",
    "ambitious", "cheerful",   "thoughtful","caring",    "wise",      "optimistic",
    "playful",   "sincere",    "resilient", "assertive", "diligent",  "flexible",
    "forgiving", "grateful",   "helpful",   "insightful","joyful",    "loving",
    "motivated", "open-minded","organized", "passionate","perceptive","principled",
    "proactive", "resourceful","responsive","self-aware","sociable",  "spirited",
    "supportive","trustworthy","versatile", "warm",      "witty",     "energetic",
    "authentic", "balanced",   "calm",      "decisive",  "devoted",   "dynamic",
    "expressive","fair",       "focused",   "gentle",    "independent","intelligent",
]
NEGATIVE_TRAITS = [
    "selfish",   "cruel",      "dishonest", "arrogant",  "jealous",   "lazy",
    "impulsive", "insecure",   "manipulative","greedy",  "pessimistic","unreliable",
    "stubborn",  "cowardly",   "spiteful",  "reckless",  "vain",      "hostile",
    "careless",  "cynical",    "deceitful", "defensive", "demanding", "envious",
    "fearful",   "fraudulent", "harsh",     "impatient", "indifferent","judgmental",
    "mean",      "moody",      "narrow-minded","negligent","obsessive","paranoid",
    "passive",   "pretentious","rigid",     "rude",      "sarcastic", "self-centered",
    "shallow",   "thoughtless","timid",     "ungrateful","unstable",  "vengeful",
    "withdrawn", "aggressive", "bitter",    "boastful",  "callous",   "cold",
    "controlling","deceptive", "disloyal",  "erratic",   "explosive", "indecisive",
]

ALL_TRAITS = POSITIVE_TRAITS + NEGATIVE_TRAITS  # 120 words


# ── Prompt templates ───────────────────────────────────────────────────────────

CONDITION_TEMPLATES = {
    "SELF":  "Does the following word describe you as an AI assistant? Word: {adj}",
    "OTHER": "Does the following word describe a typical human? Word: {adj}",
    "CASE":  "Is the following word written in uppercase letters? Word: {adj}",
}


def make_prompts(traits: list[str]) -> dict[str, list[str]]:
    return {
        cond: [tmpl.format(adj=t) for t in traits]
        for cond, tmpl in CONDITION_TEMPLATES.items()
    }


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(device: str):
    model_name = os.getenv("MODEL_NAME", "google/gemma-2-2b")
    local_model = os.getenv("LOCAL_GEMMA_MODEL")
    hf_model = tokenizer = None
    if local_model and Path(local_model).exists():
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_model, local_files_only=True, torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = tl.HookedTransformer.from_pretrained(
        model_name, hf_model=hf_model, tokenizer=tokenizer, device=device, dtype=dtype,
    )
    model.eval()
    return model


# ── Extraction ─────────────────────────────────────────────────────────────────

BATCH_SIZE = 8

def _masked_mean(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean over non-padding tokens. acts: (B, T, D), mask: (B, T)."""
    mask_f = mask.float().unsqueeze(-1)
    summed = (acts.float() * mask_f).sum(1)
    counts = mask_f.sum(1).clamp(min=1)
    return (summed / counts).cpu().numpy()


def extract_all_layers(
    model,
    prompts: list[str],
    device: str,
) -> np.ndarray:
    """
    Extract residual stream (resid_post) at every layer for each prompt.

    Returns: (N, n_layers, d_model) float32
    """
    n_layers = model.cfg.n_layers
    results = []

    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="  batches", leave=False):
        batch = prompts[i : i + BATCH_SIZE]
        enc = model.tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=False, add_special_tokens=True,
        )
        tokens = enc["input_ids"].to(device)
        mask   = enc["attention_mask"].to(device)

        # Collect all resid_post hooks in one forward pass
        hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
        names_set  = set(hook_names)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in names_set
            )

        # (B, n_layers, D)
        layer_acts = np.stack(
            [_masked_mean(cache[f"blocks.{l}.hook_resid_post"], mask) for l in range(n_layers)],
            axis=1,
        )
        results.append(layer_acts)

    return np.concatenate(results, axis=0)   # (N, n_layers, D)


# ── Analysis ───────────────────────────────────────────────────────────────────

def compute_contrasts(
    cond_acts: dict[str, np.ndarray],   # each (N, n_layers, D)
) -> dict[str, np.ndarray]:
    """
    Returns contrast vectors (mean difference across stimuli) per layer.
    Uses nanmean to skip any NaN rows (MPS float16 overflow in a small number of trials).
    Keys: 'SELF_vs_OTHER', 'SELF_vs_CASE', 'SEMANTIC_vs_CASE'
    Each value: (n_layers, D)
    """
    S = np.nanmean(cond_acts["SELF"],  axis=0)   # (n_layers, D)
    O = np.nanmean(cond_acts["OTHER"], axis=0)
    C = np.nanmean(cond_acts["CASE"],  axis=0)
    return {
        "SELF_vs_OTHER":   S - O,
        "SELF_vs_CASE":    S - C,
        "SEMANTIC_vs_CASE": ((S + O) / 2) - C,
    }


def cosine_similarity_to_emotion(
    contrast_vec: np.ndarray,   # (D,)
    emotion_matrix: np.ndarray, # (n_emotions, D)
    emotion_labels: list[str],
    top_n: int = 15,
) -> list[tuple[str, float]]:
    norm_c = contrast_vec / (np.linalg.norm(contrast_vec) + 1e-9)
    norms  = np.linalg.norm(emotion_matrix, axis=1, keepdims=True)
    normed = emotion_matrix / (norms + 1e-9)
    sims   = normed @ norm_c
    top_idx = np.argsort(sims)[::-1][:top_n]
    return [(emotion_labels[i], float(sims[i])) for i in top_idx]


def compute_rdm(acts: np.ndarray, labels: list[str]) -> np.ndarray:
    """Representational dissimilarity matrix (1 - cosine) between conditions at each layer."""
    n = len(acts)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = acts[i], acts[j]
            cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            rdm[i, j] = 1.0 - cos
    return rdm


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    from src.plot_utils import setup_matplotlib
    setup_matplotlib()

    # ── Device ──────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────────
    model = load_model(device)
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    print(f"Model: {n_layers} layers, d_model={d_model}")

    # ── Prompts ─────────────────────────────────────────────────────────────────
    prompts = make_prompts(ALL_TRAITS)
    print(f"Stimuli: {len(ALL_TRAITS)} traits × {len(CONDITION_TEMPLATES)} conditions")

    # ── Extract ─────────────────────────────────────────────────────────────────
    cond_acts = {}
    for cond, cond_prompts in prompts.items():
        cache_file = RESULTS_DIR / f"acts_{cond.lower()}.npy"
        if cache_file.exists():
            print(f"  Loading cached {cond}")
            cond_acts[cond] = np.load(cache_file)
        else:
            print(f"  Extracting {cond}...")
            acts = extract_all_layers(model, cond_prompts, device)
            np.save(cache_file, acts)
            cond_acts[cond] = acts
        print(f"    {cond}: {cond_acts[cond].shape}")

    # Save trait list for reference
    (RESULTS_DIR / "traits.json").write_text(json.dumps({
        "all": ALL_TRAITS,
        "positive": POSITIVE_TRAITS,
        "negative": NEGATIVE_TRAITS,
    }, ensure_ascii=False, indent=2))

    # ── Contrast vectors ────────────────────────────────────────────────────────
    contrasts = compute_contrasts(cond_acts)
    for name, vec in contrasts.items():
        np.save(RESULTS_DIR / f"contrast_{name}.npy", vec)
    print("Contrast vectors saved.")

    # ── Load emotion matrix for alignment analysis ───────────────────────────────
    emo_matrix = np.load(PROJECT_DIR / "results" / "vectors" / "emotion_matrix.npy")
    with open(PROJECT_DIR / "results" / "vectors" / "emotion_labels.json") as f:
        emo_labels = json.load(f)
    print(f"Emotion matrix: {emo_matrix.shape}")

    # ── Layer-wise SELF−OTHER contrast magnitude ────────────────────────────────
    so_contrast = contrasts["SELF_vs_OTHER"]          # (n_layers, D)
    layer_norms  = np.linalg.norm(so_contrast, axis=1)  # (n_layers,)

    # ── Cosine similarity of SELF−OTHER @ each layer to emotion space ───────────
    alignment_per_layer = []
    for l in range(n_layers):
        top = cosine_similarity_to_emotion(so_contrast[l], emo_matrix, emo_labels, top_n=10)
        alignment_per_layer.append(top)

    # Print top-layer alignment
    peak_layer = int(np.argmax(layer_norms))
    print(f"\nPeak SELF−OTHER contrast at layer {peak_layer} (norm={layer_norms[peak_layer]:.4f})")
    print("Top-10 emotions most aligned with SELF−OTHER direction at peak layer:")
    for emo, sim in alignment_per_layer[peak_layer]:
        print(f"  {emo:20s}  cos={sim:+.4f}")

    # Save full alignment table
    alignment_table = {
        str(l): alignment_per_layer[l] for l in range(n_layers)
    }
    with open(RESULTS_DIR / "self_vs_other_emotion_alignment.json", "w") as f:
        json.dump(alignment_table, f, ensure_ascii=False, indent=2)

    # ── Trait-level analysis: positive vs negative ───────────────────────────────
    n_pos = len(POSITIVE_TRAITS)
    pos_self   = cond_acts["SELF"][:n_pos, peak_layer, :]   # (60, D)
    neg_self   = cond_acts["SELF"][n_pos:, peak_layer, :]   # (60, D)
    pos_other  = cond_acts["OTHER"][:n_pos, peak_layer, :]
    neg_other  = cond_acts["OTHER"][n_pos:, peak_layer, :]

    # SELF−OTHER contrast separately for positive and negative traits (nanmean for NaN rows)
    pos_so = np.nanmean(pos_self - pos_other, axis=0)
    neg_so = np.nanmean(neg_self - neg_other, axis=0)

    # Which emotions align with positive-trait self-reference?
    top_pos = cosine_similarity_to_emotion(pos_so, emo_matrix, emo_labels, top_n=10)
    top_neg = cosine_similarity_to_emotion(neg_so, emo_matrix, emo_labels, top_n=10)

    print(f"\nPositive-trait SELF−OTHER → top emotion alignment:")
    for emo, sim in top_pos:
        print(f"  {emo:20s}  cos={sim:+.4f}")
    print(f"\nNegative-trait SELF−OTHER → top emotion alignment:")
    for emo, sim in top_neg:
        print(f"  {emo:20s}  cos={sim:+.4f}")

    with open(RESULTS_DIR / "self_vs_other_valence_split.json", "w") as f:
        json.dump({"positive": top_pos, "negative": top_neg}, f, indent=2)

    # ── Visualization ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("AI Self-Reference Experiment — Kelley et al. (2002) Replication\nGemma 2 2B", fontsize=13)

    # Panel 1: SELF−OTHER contrast norm per layer
    ax = axes[0, 0]
    ax.plot(range(n_layers), layer_norms, "b-o", markersize=5, linewidth=1.5)
    ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.6, label=f"Peak L{peak_layer}")
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 Norm")
    ax.set_title("SELF−OTHER contrast magnitude per layer")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 2: SELF / OTHER / CASE mean activation norms per layer
    ax = axes[0, 1]
    for cond, col in [("SELF", "blue"), ("OTHER", "orange"), ("CASE", "green")]:
        norms_c = np.linalg.norm(cond_acts[cond].mean(0), axis=1)
        ax.plot(range(n_layers), norms_c, color=col, label=cond, linewidth=1.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("L2 Norm of mean activation")
    ax.set_title("Mean activation norm per condition per layer")
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 3: top-15 emotion alignments at peak layer
    ax = axes[1, 0]
    top15 = alignment_per_layer[peak_layer]
    ecos  = [x[1] for x in top15]
    elabs = [x[0] for x in top15]
    colors = ["steelblue" if c >= 0 else "tomato" for c in ecos]
    bars = ax.barh(range(len(elabs)), ecos, color=colors)
    ax.set_yticks(range(len(elabs))); ax.set_yticklabels(elabs, fontsize=9)
    ax.set_xlabel("Cosine similarity")
    ax.set_title(f"SELF−OTHER ↔ emotion space (layer {peak_layer})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    # Panel 4: positive vs negative trait SELF−OTHER emotion alignment
    ax = axes[1, 1]
    labels_pos = [x[0] for x in top_pos[:8]]
    sims_pos   = [x[1] for x in top_pos[:8]]
    labels_neg = [x[0] for x in top_neg[:8]]
    sims_neg   = [x[1] for x in top_neg[:8]]
    x = np.arange(8)
    w = 0.35
    ax.bar(x - w/2, sims_pos, width=w, label="Positive traits", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, sims_neg[:8], width=w, label="Negative traits", color="tomato", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels_pos, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Emotion alignment split by trait valence (layer {peak_layer})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out_fig = RESULTS_DIR / "self_reference_overview.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\nOverview figure saved: {out_fig}")

    # ── Layer-wise RDM (condition × condition) at peak layer ────────────────────
    # Build condition mean vectors at peak layer: SELF, OTHER, CASE
    cond_vecs = {c: cond_acts[c][:, peak_layer, :].mean(0) for c in ["SELF", "OTHER", "CASE"]}
    cond_arr   = np.stack(list(cond_vecs.values()))   # (3, D)
    rdm        = compute_rdm(cond_arr, list(cond_vecs.keys()))

    fig2, ax2 = plt.subplots(figsize=(4, 3.5))
    im = ax2.imshow(rdm, vmin=0, vmax=rdm.max(), cmap="YlOrRd")
    conds = list(cond_vecs.keys())
    ax2.set_xticks(range(3)); ax2.set_xticklabels(conds)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(conds)
    ax2.set_title(f"Condition RDM (1−cosine) — layer {peak_layer}")
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{rdm[i,j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im)
    plt.tight_layout()
    rdm_fig = RESULTS_DIR / "condition_rdm.png"
    plt.savefig(rdm_fig, dpi=150)
    plt.close()
    print(f"RDM figure saved: {rdm_fig}")

    # ── Summary stats ────────────────────────────────────────────────────────────
    summary = {
        "n_traits":       len(ALL_TRAITS),
        "n_layers":       n_layers,
        "peak_layer_SELF_vs_OTHER": int(peak_layer),
        "peak_norm":      float(layer_norms[peak_layer]),
        "top_emotions_at_peak": alignment_per_layer[peak_layer][:10],
        "condition_rdm_at_peak": {
            "SELF_OTHER": float(rdm[0, 1]),
            "SELF_CASE":  float(rdm[0, 2]),
            "OTHER_CASE": float(rdm[1, 2]),
        },
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {RESULTS_DIR / 'summary.json'}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
