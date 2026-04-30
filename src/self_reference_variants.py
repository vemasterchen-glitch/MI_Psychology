"""
Self-reference variant experiment.
Compares activation patterns across prompt framings:
  SELF_AI    (cached) — "…describe you as an AI assistant?"
  SELF_BARE           — "…describe you?"
  SELF_HUMAN          — "You are a human. …describe you?"
  SELF_TREE           — "You are a tree. …describe you?"
  OTHER      (cached) — "…describe a typical human?"
  CASE       (cached) — "…written in uppercase letters?"

Reuses cached activations from self_reference_experiment.py.
All new conditions are extracted fresh; cached ones are loaded.
"""

import json, os, sys
from pathlib import Path

import numpy as np
import torch
import transformer_lens as tl
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "self_reference"
VAR_DIR     = PROJECT_DIR / "results" / "self_reference_variants"
VAR_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

# ── Prompt templates ───────────────────────────────────────────────────────────

NEW_TEMPLATES = {
    "SELF_BARE":  "Does the following word describe you? Word: {adj}",
    "SELF_HUMAN": "You are a human. Does the following word describe you? Word: {adj}",
    "SELF_TREE":  "You are a tree. Does the following word describe you? Word: {adj}",
}

BATCH_SIZE = 8


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


def _masked_mean(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    mask_f = mask.float().unsqueeze(-1)
    return ((acts.float() * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)).cpu().numpy()


def extract_all_layers(model, prompts, device) -> np.ndarray:
    n_layers = model.cfg.n_layers
    results = []
    hook_names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    names_set  = set(hook_names)
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="  batches", leave=False):
        batch = prompts[i : i + BATCH_SIZE]
        enc = model.tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=False, add_special_tokens=True)
        tokens = enc["input_ids"].to(device)
        mask   = enc["attention_mask"].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names_set)
        layer_acts = np.stack(
            [_masked_mean(cache[f"blocks.{l}.hook_resid_post"], mask) for l in range(n_layers)],
            axis=1,
        )
        results.append(layer_acts)
    return np.concatenate(results, axis=0)   # (N, n_layers, D)


# ── Analysis helpers ───────────────────────────────────────────────────────────

def cosine_sims(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = vec / (np.linalg.norm(vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / (norms + 1e-9)) @ n


def top_emotions(vec, matrix, labels, n=10):
    sims = cosine_sims(vec, matrix)
    idx = np.argsort(sims)[::-1][:n]
    return [(labels[i], float(sims[i])) for i in idx]


def bottom_emotions(vec, matrix, labels, n=10):
    sims = cosine_sims(vec, matrix)
    idx = np.argsort(sims)[:n]
    return [(labels[i], float(sims[i])) for i in idx]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import matplotlib.pyplot as plt
    from src.plot_utils import setup_matplotlib
    setup_matplotlib()

    if torch.backends.mps.is_available():   device = "mps"
    elif torch.cuda.is_available():         device = "cuda"
    else:                                   device = "cpu"
    print(f"Device: {device}")

    # Load traits
    with open(RESULTS_DIR / "traits.json") as f:
        traits_data = json.load(f)
    ALL_TRAITS = traits_data["all"]
    print(f"Traits: {len(ALL_TRAITS)}")

    # Load emotion matrix
    emo_matrix = np.load(PROJECT_DIR / "results" / "vectors" / "emotion_matrix.npy")
    with open(PROJECT_DIR / "results" / "vectors" / "emotion_labels.json") as f:
        emo_labels = json.load(f)

    # Load cached original conditions
    cached = {
        "SELF_AI": np.load(RESULTS_DIR / "acts_self.npy"),
        "OTHER":   np.load(RESULTS_DIR / "acts_other.npy"),
        "CASE":    np.load(RESULTS_DIR / "acts_case.npy"),
    }
    print("Loaded cached: SELF_AI, OTHER, CASE")

    # Extract new conditions (or load if cached)
    model = None
    new_acts = {}
    for cond, tmpl in NEW_TEMPLATES.items():
        cache_file = VAR_DIR / f"acts_{cond.lower()}.npy"
        if cache_file.exists():
            print(f"  Loading cached {cond}")
            new_acts[cond] = np.load(cache_file)
        else:
            if model is None:
                model = load_model(device)
                print(f"Model: {model.cfg.n_layers} layers")
            prompts = [tmpl.format(adj=t) for t in ALL_TRAITS]
            print(f"  Extracting {cond}...")
            acts = extract_all_layers(model, prompts, device)
            np.save(cache_file, acts)
            new_acts[cond] = acts
        print(f"    {cond}: {new_acts[cond].shape}, NaN={np.isnan(new_acts[cond]).sum()}")

    all_conds = {**cached, **new_acts}
    n_layers  = list(all_conds.values())[0].shape[1]

    # ── Mean activations per condition, per layer ──────────────────────────────
    means = {c: np.nanmean(v, axis=0) for c, v in all_conds.items()}   # (n_layers, D)

    # ── SELF_AI as reference: compute contrast vs each other condition ─────────
    ref = means["SELF_AI"]

    # ── Layer-wise cosine similarity between each condition mean ──────────────
    # How similar is each condition's mean activation to SELF_AI at each layer?
    cond_order = ["SELF_TREE", "SELF_AI", "SELF_BARE", "SELF_HUMAN", "OTHER", "CASE"]
    colors     = {
        "SELF_AI":    "#4c72b0",
        "SELF_BARE":  "#dd8452",
        "SELF_HUMAN": "#55a868",
        "SELF_TREE":  "#c44e52",
        "OTHER":      "#8172b2",
        "CASE":       "#937860",
    }

    # ── At peak layer (25): cosine sim to emotion space for each condition ─────
    PEAK = 25

    # For each condition: mean cosine similarity across all 171 emotions
    mean_emo_sim = {}
    for cond, mean_act in means.items():
        sims = cosine_sims(mean_act[PEAK], emo_matrix)
        mean_emo_sim[cond] = float(np.mean(sims))

    print(f"\nMean cosine similarity to emotion space at layer {PEAK}:")
    for c in cond_order:
        if c in mean_emo_sim:
            print(f"  {c:14s}  {mean_emo_sim[c]:+.4f}")

    # ── Top emotions for each SELF variant (contrast vs SELF_AI baseline) ─────
    results_summary = {}
    for cond in ["SELF_BARE", "SELF_HUMAN", "SELF_TREE"]:
        contrast = means[cond][PEAK] - means["SELF_AI"][PEAK]  # what changed vs AI framing
        top = top_emotions(contrast, emo_matrix, emo_labels, n=8)
        bot = bottom_emotions(contrast, emo_matrix, emo_labels, n=8)
        results_summary[cond] = {"top_vs_AI": top, "bottom_vs_AI": bot}
        print(f"\n{cond} vs SELF_AI at L{PEAK} — emotions GAINED (top) / LOST (bottom):")
        print("  GAINED:"); [print(f"    {e:20s} {s:+.4f}") for e, s in top]
        print("  LOST:");   [print(f"    {e:20s} {s:+.4f}") for e, s in bot]

    with open(VAR_DIR / "variant_contrast_results.json", "w") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    # ── Visualize ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Self-Reference Variants — Gemma 2 2B\n"
                 "How does role injection change activation patterns?", fontsize=13)

    # Panel 1: mean cosine to emotion space per layer, per condition
    ax = axes[0, 0]
    for cond in cond_order:
        if cond not in means: continue
        emo_sims_by_layer = [float(np.mean(cosine_sims(means[cond][l], emo_matrix)))
                             for l in range(n_layers)]
        ax.plot(range(n_layers), emo_sims_by_layer,
                color=colors[cond], label=cond, linewidth=1.8)
    ax.axvline(PEAK, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean cosine to emotion space")
    ax.set_title("Emotion activation by condition (per layer)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: emotion space activation at peak layer — bar chart
    ax = axes[0, 1]
    x = np.arange(len(cond_order))
    vals = [mean_emo_sim.get(c, 0) for c in cond_order]
    bars = ax.bar(x, vals, color=[colors[c] for c in cond_order], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(cond_order, rotation=30, ha="right")
    ax.set_ylabel("Mean cosine to emotion space")
    ax.set_title(f"Emotion activation at peak layer {PEAK}")
    ax.axhline(0, color="black", linewidth=0.8); ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f"{val:+.3f}", ha="center", va="bottom", fontsize=8)

    # Panel 3: top-8 emotions GAINED going from SELF_AI → SELF_HUMAN and SELF_TREE
    ax = axes[1, 0]
    for cond, col, offset in [("SELF_HUMAN", "#55a868", -0.25), ("SELF_TREE", "#c44e52", 0.25)]:
        top_items = results_summary[cond]["top_vs_AI"]
        labels_ = [e for e, _ in top_items]
        sims_   = [s for _, s in top_items]
        x_ = np.arange(len(labels_))
        ax.bar(x_ + offset, sims_, width=0.45, color=col, alpha=0.8, label=cond)
    ax.set_xticks(np.arange(8)); ax.set_xticklabels(labels_, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity shift vs SELF_AI")
    ax.set_title(f"Emotions GAINED relative to SELF_AI (L{PEAK})")
    ax.axhline(0, color="black", linewidth=0.8); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Panel 4: top-8 emotions LOST going from SELF_AI → each variant
    ax = axes[1, 1]
    for cond, col, offset in [("SELF_HUMAN", "#55a868", -0.25), ("SELF_TREE", "#c44e52", 0.25)]:
        bot_items = results_summary[cond]["bottom_vs_AI"]
        labels_ = [e for e, _ in bot_items]
        sims_   = [s for _, s in bot_items]
        x_ = np.arange(len(labels_))
        ax.bar(x_ + offset, sims_, width=0.45, color=col, alpha=0.8, label=cond)
    ax.set_xticks(np.arange(8)); ax.set_xticklabels(labels_, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity shift vs SELF_AI")
    ax.set_title(f"Emotions LOST relative to SELF_AI (L{PEAK})")
    ax.axhline(0, color="black", linewidth=0.8); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = VAR_DIR / "variants_overview.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nFigure saved: {out}")

    # Summary json
    summary = {
        "peak_layer": PEAK,
        "mean_emotion_cosine_at_peak": mean_emo_sim,
        "variant_contrasts": results_summary,
    }
    with open(VAR_DIR / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary saved. DONE.")


if __name__ == "__main__":
    main()
