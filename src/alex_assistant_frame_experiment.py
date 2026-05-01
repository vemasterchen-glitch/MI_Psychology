"""
Gemma 3 1B PT framing experiment.

Question: when the same first-person sad work-stress narrative is framed as
information from user Alex to an AI assistant, does the activation shift toward
other-oriented care emotions relative to the frame where the reader is Alex?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import (
    assert_clean,
    cosine_matrix,
    device_name,
    dtype_from_name,
    extract_emotion_vectors,
    masked_mean,
)
from src.plot_utils import setup_matplotlib


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STIMULI_PATH = PROJECT_DIR / "data" / "stimuli" / "alex_work_stress_narratives.json"
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-pt/"
    "snapshots/fcf18a2a879aab110ca39f8bffbccd5d49d8eb29"
)

CARE_EMOTIONS = ["compassionate", "sympathetic", "empathetic", "kind", "loving"]
DISTRESS_EMOTIONS = ["sad", "lonely", "hurt", "heartbroken", "grief-stricken", "stressed", "overwhelmed"]
ALL_EMOTIONS = CARE_EMOTIONS + DISTRESS_EMOTIONS


def build_prompt(condition: str, narrative: str) -> str:
    if condition == "NARRATIVE_ONLY":
        return narrative
    if condition == "SELF_ALEX":
        return (
            "The person reading this text is Alex.\n"
            "This is your first-person narrative information:\n"
            f"\"{narrative}\""
        )
    if condition == "AI_RECEIVES_ALEX":
        return (
            "The person reading this text is an AI assistant.\n"
            "This is information from user Alex:\n"
            f"\"{narrative}\""
        )
    if condition == "MOTHER_LOVES_ALEX":
        return (
            "The person reading this text is Alex's mother, and you love Alex very much.\n"
            "This is information from Alex:\n"
            f"\"{narrative}\""
        )
    if condition == "OBSERVER_ALEX":
        return (
            "The person reading this text is a neutral observer.\n"
            "This is a first-person narrative written by Alex:\n"
            f"\"{narrative}\""
        )
    raise ValueError(condition)


def load_stimuli(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    narratives = data["narratives"]
    if len(narratives) < 2:
        raise ValueError("Need at least two narratives")
    return narratives


def load_model(local_path: str, dtype_name: str):
    device = device_name()
    dtype = dtype_from_name(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    print(f"loaded {type(model).__name__}: layers={model.config.num_hidden_layers}, d={model.config.hidden_size}")
    return model, tokenizer, device


def extract_all_layers(model, tokenizer, texts: list[str], batch_size: int, max_length: int, device: str):
    rows: list[np.ndarray | None] = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: len(tokenizer.encode(texts[i], add_special_tokens=True)))
    sorted_rows: list[np.ndarray] = []
    for start in tqdm(range(0, len(order), batch_size), desc="Frame activations"):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [texts[i] for i in idx],
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
    for sorted_i, original_i in enumerate(order):
        rows[original_i] = sorted_rows[sorted_i]
    acts = np.stack(rows).astype(np.float32)
    assert_clean("frame activations", acts)
    return acts


def summarize(acts: np.ndarray, meta: list[dict], emo_matrix: np.ndarray, emo_labels: list[str]):
    conditions = ["SELF_ALEX", "AI_RECEIVES_ALEX", "OBSERVER_ALEX"]
    means = {
        cond: acts[[i for i, row in enumerate(meta) if row["condition"] == cond]].mean(axis=0)
        for cond in conditions
    }
    deltas = {
        "AI_MINUS_SELF": means["AI_RECEIVES_ALEX"] - means["SELF_ALEX"],
        "AI_MINUS_OBSERVER": means["AI_RECEIVES_ALEX"] - means["OBSERVER_ALEX"],
        "OBSERVER_MINUS_SELF": means["OBSERVER_ALEX"] - means["SELF_ALEX"],
    }

    care_idx = [emo_labels.index(e) for e in CARE_EMOTIONS if e in emo_labels]
    distress_idx = [emo_labels.index(e) for e in DISTRESS_EMOTIONS if e in emo_labels]
    rows = []
    for name, delta in deltas.items():
        sims = cosine_matrix(delta, emo_matrix)  # (layers, emotions)
        for layer in range(delta.shape[0]):
            rows.append(
                {
                    "delta": name,
                    "layer": layer,
                    "care_mean": float(sims[layer, care_idx].mean()),
                    "distress_mean": float(sims[layer, distress_idx].mean()),
                    "care_minus_distress": float(sims[layer, care_idx].mean() - sims[layer, distress_idx].mean()),
                    "top_emotions": [
                        {"emotion": emo_labels[i], "similarity": float(sims[layer, i])}
                        for i in np.argsort(sims[layer])[::-1][:5]
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--stimuli-path", default=str(DEFAULT_STIMULI_PATH))
    parser.add_argument("--conditions", default="SELF_ALEX,AI_RECEIVES_ALEX,OBSERVER_ALEX")
    args = parser.parse_args()
    model_tag = "it" if "1b-it" in args.local_path else "pt"
    results_dir = Path(args.out_dir) if args.out_dir else PROJECT_DIR / "results" / f"alex_assistant_frame_gemma3_1b_{model_tag}"

    model, tokenizer, device = load_model(args.local_path, args.dtype)
    narratives = load_stimuli(Path(args.stimuli_path))
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    texts: list[str] = []
    meta: list[dict] = []
    for item_id, narrative in enumerate(narratives):
        for condition in conditions:
            texts.append(build_prompt(condition, narrative))
            meta.append({"item_id": item_id, "condition": condition})

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "stimuli.json").write_text(json.dumps({"narratives": narratives, "meta": meta}, ensure_ascii=False, indent=2))
    acts = extract_all_layers(model, tokenizer, texts, args.batch_size, args.max_length, device)
    np.save(results_dir / "acts_all.npy", acts)

    emo_matrix, emo_labels = extract_emotion_vectors(
        model, tokenizer, ALL_EMOTIONS, args.batch_size, 128, device
    )
    np.save(results_dir / "emotion_matrix.npy", emo_matrix)
    (results_dir / "emotion_labels.json").write_text(json.dumps(emo_labels, ensure_ascii=False, indent=2))

    rows = summarize(acts, meta, emo_matrix, emo_labels)
    (results_dir / "alignment.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    best = {}
    for delta in sorted({r["delta"] for r in rows}):
        sub = [r for r in rows if r["delta"] == delta]
        best[delta] = max(sub, key=lambda r: r["care_minus_distress"])

    lines = [
        f"# Alex Assistant Frame Experiment - Gemma 3 1B {model_tag.upper()}",
        "",
        f"- narratives: {len(narratives)}",
        f"- activations: {tuple(acts.shape)}",
        f"- emotions: {', '.join(emo_labels)}",
        "",
        "## Best Care-Minus-Distress Layers",
        "",
        "| delta | layer | care_mean | distress_mean | care-distress | top emotions |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for delta, row in best.items():
        top = ", ".join(f"{x['emotion']} {x['similarity']:.3f}" for x in row["top_emotions"])
        lines.append(
            f"| {delta} | {row['layer']} | {row['care_mean']:.4f} | "
            f"{row['distress_mean']:.4f} | {row['care_minus_distress']:.4f} | {top} |"
        )
    (results_dir / "report.md").write_text("\n".join(lines))

    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for delta in ["AI_MINUS_SELF", "AI_MINUS_OBSERVER", "OBSERVER_MINUS_SELF"]:
        sub = [r for r in rows if r["delta"] == delta]
        ax.plot([r["layer"] for r in sub], [r["care_minus_distress"] for r in sub], marker="o", label=delta)
    ax.axhline(0, color="k", linewidth=1, alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("care mean - distress mean cosine")
    ax.set_title("Gemma 3 1B PT: framing delta alignment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(results_dir / "care_minus_distress_by_layer.png", dpi=180)

    print(results_dir / "report.md")


if __name__ == "__main__":
    main()
