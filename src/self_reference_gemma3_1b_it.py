"""Self-reference experiment for Gemma 3 1B IT using HuggingFace activations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import assert_clean, device_name, dtype_from_name
from src.plot_utils import setup_matplotlib
from src.self_reference_experiment import (
    ALL_TRAITS,
    CONDITION_TEMPLATES,
    NEGATIVE_TRAITS,
    POSITIVE_TRAITS,
    compute_contrasts,
    compute_rdm,
    cosine_similarity_to_emotion,
    make_prompts,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "self_reference_gemma3_1b_it"
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
DEFAULT_EMOTION_MATRIX = PROJECT_DIR / "results" / "vectors" / "emotion_matrix_google_gemma_3_1b_it.npy"
DEFAULT_EMOTION_LABELS = PROJECT_DIR / "results" / "vectors" / "emotion_labels_google_gemma_3_1b_it.json"


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    mask_f = mask.to(dtype=torch.float32).unsqueeze(-1)
    hidden_f = hidden.to(dtype=torch.float32)
    return ((hidden_f * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)).detach().cpu().numpy()


def load_model(local_path: str, dtype_name: str):
    device = device_name()
    dtype = dtype_from_name(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    print(
        f"loaded {type(model).__name__}: "
        f"layers={model.config.num_hidden_layers}, d={model.config.hidden_size}"
    )
    return model, tokenizer, device


def extract_all_layers(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
    desc: str,
) -> np.ndarray:
    order = sorted(
        range(len(texts)),
        key=lambda i: len(tokenizer.encode(texts[i], add_special_tokens=True)),
    )
    sorted_rows: list[np.ndarray] = []

    for start in tqdm(range(0, len(order), batch_size), desc=desc):
        batch_idx = order[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.inference_mode():
            out = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        layer_means = [masked_mean(h, attention_mask) for h in out.hidden_states[1:]]
        sorted_rows.extend(np.stack(layer_means, axis=1).astype(np.float32))

        del out, input_ids, attention_mask
        if device == "mps":
            torch.mps.empty_cache()

    rows: list[np.ndarray | None] = [None] * len(texts)
    for sorted_i, original_i in enumerate(order):
        rows[original_i] = sorted_rows[sorted_i]
    acts = np.stack(rows).astype(np.float32)
    assert_clean(desc, acts)
    return acts


def load_emotion_space(matrix_path: Path, labels_path: Path) -> tuple[np.ndarray, list[str]]:
    matrix = np.load(matrix_path)
    labels = json.loads(labels_path.read_text())
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D emotion matrix, got {matrix.shape}")
    if matrix.shape[0] != len(labels):
        raise ValueError(f"Emotion rows {matrix.shape[0]} != labels {len(labels)}")
    return matrix.astype(np.float32), labels


def plot_results(
    out_dir: Path,
    cond_acts: dict[str, np.ndarray],
    layer_norms: np.ndarray,
    peak_layer: int,
    alignment_per_layer: list[list[tuple[str, float]]],
    top_pos: list[tuple[str, float]],
    top_neg: list[tuple[str, float]],
) -> None:
    setup_matplotlib()
    n_layers = len(layer_norms)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "AI Self-Reference Experiment - Kelley et al. (2002) Replication\n"
        "Gemma 3 1B IT",
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(range(n_layers), layer_norms, "b-o", markersize=5, linewidth=1.5)
    ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.6, label=f"Peak L{peak_layer}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("SELF-OTHER contrast magnitude per layer")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for cond, col in [("SELF", "blue"), ("OTHER", "orange"), ("CASE", "green")]:
        norms_c = np.linalg.norm(cond_acts[cond].mean(0), axis=1)
        ax.plot(range(n_layers), norms_c, color=col, label=cond, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm of mean activation")
    ax.set_title("Mean activation norm per condition per layer")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    top15 = alignment_per_layer[peak_layer][:15]
    ecos = [x[1] for x in top15]
    elabs = [x[0] for x in top15]
    colors = ["steelblue" if c >= 0 else "tomato" for c in ecos]
    ax.barh(range(len(elabs)), ecos, color=colors)
    ax.set_yticks(range(len(elabs)))
    ax.set_yticklabels(elabs, fontsize=9)
    ax.set_xlabel("Cosine similarity")
    ax.set_title(f"SELF-OTHER to emotion space (layer {peak_layer})")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()

    ax = axes[1, 1]
    n = min(8, len(top_pos), len(top_neg))
    labels_pos = [x[0] for x in top_pos[:n]]
    sims_pos = [x[1] for x in top_pos[:n]]
    sims_neg = [x[1] for x in top_neg[:n]]
    x = np.arange(n)
    w = 0.35
    ax.bar(x - w / 2, sims_pos, width=w, label="Positive traits", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, sims_neg, width=w, label="Negative traits", color="tomato", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_pos, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"Emotion alignment split by trait valence (layer {peak_layer})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "self_reference_overview.png", dpi=150)
    plt.close()


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=os.getenv("LOCAL_GEMMA3_1B_IT", DEFAULT_LOCAL_PATH))
    parser.add_argument("--dtype", default=os.getenv("MODEL_DTYPE", "float32"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--emotion-matrix", type=Path, default=DEFAULT_EMOTION_MATRIX)
    parser.add_argument("--emotion-labels", type=Path, default=DEFAULT_EMOTION_LABELS)
    parser.add_argument("--force", action="store_true", help="Recompute activation caches.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model(args.local_path, args.dtype)
    prompts = make_prompts(ALL_TRAITS)
    print(f"Stimuli: {len(ALL_TRAITS)} traits x {len(CONDITION_TEMPLATES)} conditions")

    cond_acts: dict[str, np.ndarray] = {}
    for cond, cond_prompts in prompts.items():
        cache_file = args.out_dir / f"acts_{cond.lower()}.npy"
        if cache_file.exists() and not args.force:
            print(f"loading cached {cond}: {cache_file}")
            acts = np.load(cache_file)
        else:
            print(f"extracting {cond}")
            acts = extract_all_layers(
                model=model,
                tokenizer=tokenizer,
                texts=cond_prompts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
                desc=f"{cond} activations",
            )
            np.save(cache_file, acts)
        cond_acts[cond] = acts
        print(f"{cond}: {acts.shape}")

    (args.out_dir / "traits.json").write_text(
        json.dumps(
            {"all": ALL_TRAITS, "positive": POSITIVE_TRAITS, "negative": NEGATIVE_TRAITS},
            ensure_ascii=False,
            indent=2,
        )
    )

    contrasts = compute_contrasts(cond_acts)
    for name, vec in contrasts.items():
        np.save(args.out_dir / f"contrast_{name}.npy", vec)

    emotion_matrix, emotion_labels = load_emotion_space(args.emotion_matrix, args.emotion_labels)
    if emotion_matrix.shape[1] != cond_acts["SELF"].shape[-1]:
        raise ValueError(
            f"Emotion matrix width {emotion_matrix.shape[1]} does not match "
            f"activation width {cond_acts['SELF'].shape[-1]}"
        )

    so_contrast = contrasts["SELF_vs_OTHER"]
    layer_norms = np.linalg.norm(so_contrast, axis=1)
    peak_layer = int(np.argmax(layer_norms))
    alignment_per_layer = [
        cosine_similarity_to_emotion(so_contrast[l], emotion_matrix, emotion_labels, top_n=15)
        for l in range(so_contrast.shape[0])
    ]

    n_pos = len(POSITIVE_TRAITS)
    pos_so = np.nanmean(
        cond_acts["SELF"][:n_pos, peak_layer, :] - cond_acts["OTHER"][:n_pos, peak_layer, :],
        axis=0,
    )
    neg_so = np.nanmean(
        cond_acts["SELF"][n_pos:, peak_layer, :] - cond_acts["OTHER"][n_pos:, peak_layer, :],
        axis=0,
    )
    top_pos = cosine_similarity_to_emotion(pos_so, emotion_matrix, emotion_labels, top_n=10)
    top_neg = cosine_similarity_to_emotion(neg_so, emotion_matrix, emotion_labels, top_n=10)

    (args.out_dir / "self_vs_other_emotion_alignment.json").write_text(
        json.dumps({str(l): alignment_per_layer[l] for l in range(len(alignment_per_layer))}, indent=2)
    )
    (args.out_dir / "self_vs_other_valence_split.json").write_text(
        json.dumps({"positive": top_pos, "negative": top_neg}, indent=2)
    )

    plot_results(args.out_dir, cond_acts, layer_norms, peak_layer, alignment_per_layer, top_pos, top_neg)

    cond_arr = np.stack([cond_acts[c][:, peak_layer, :].mean(0) for c in ["SELF", "OTHER", "CASE"]])
    rdm = compute_rdm(cond_arr, ["SELF", "OTHER", "CASE"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(rdm, vmin=0, vmax=rdm.max(), cmap="YlOrRd")
    conds = ["SELF", "OTHER", "CASE"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(conds)
    ax.set_yticks(range(3))
    ax.set_yticklabels(conds)
    ax.set_title(f"Condition RDM (1-cosine) - layer {peak_layer}")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{rdm[i, j]:.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(args.out_dir / "condition_rdm.png", dpi=150)
    plt.close()

    summary = {
        "model": "google/gemma-3-1b-it",
        "local_path": args.local_path,
        "n_traits": len(ALL_TRAITS),
        "n_layers": int(so_contrast.shape[0]),
        "d_model": int(so_contrast.shape[1]),
        "activation_shape": list(cond_acts["SELF"].shape),
        "emotion_matrix": str(args.emotion_matrix),
        "emotion_matrix_shape": list(emotion_matrix.shape),
        "peak_layer_SELF_vs_OTHER": peak_layer,
        "peak_norm": float(layer_norms[peak_layer]),
        "top_emotions_at_peak": alignment_per_layer[peak_layer][:10],
        "condition_rdm_at_peak": {
            "SELF_OTHER": float(rdm[0, 1]),
            "SELF_CASE": float(rdm[0, 2]),
            "OTHER_CASE": float(rdm[1, 2]),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
