"""
Gemma 3 1B IT: compare blame-recipient activations with blame-related emotion vectors.

This uses the text-only Gemma 3 1B checkpoint via HuggingFace, avoiding the
Gemma 3 4B multimodal wrapper and TransformerLens path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.blame_recipient_experiment import flatten_stimuli


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "blame_recipient_gemma3_1b_it"
VECTORS_DIR = PROJECT_DIR / "results" / "vectors"
NARRATIVES_PATH = PROJECT_DIR / "data" / "stimuli" / "narratives.jsonl"

DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)

DEFAULT_EMOTIONS = [
    "angry",
    "annoyed",
    "irritated",
    "frustrated",
    "indignant",
    "resentful",
    "offended",
    "bitter",
    "contemptuous",
    "hostile",
    "ashamed",
    "guilty",
    "remorseful",
    "regretful",
    "hurt",
]


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dtype_from_name(name: str, device: str) -> torch.dtype:
    if name == "auto":
        name = "bfloat16" if device == "cuda" else "float32"
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype={name!r}")
    print(f"dtype={name}")
    return table[name]


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    mask_f = mask.to(dtype=torch.float32).unsqueeze(-1)
    hidden_f = hidden.to(dtype=torch.float32)
    return ((hidden_f * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)).detach().cpu().numpy()


def assert_clean(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise FloatingPointError(
            f"{name} has non-finite values: NaN={int(np.isnan(arr).sum())}, "
            f"Inf={int(np.isinf(arr).sum())}"
        )
    flat = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
    zero = int((np.linalg.norm(flat, axis=1) < 1e-12).sum())
    if zero:
        raise FloatingPointError(f"{name} has exact-zero vectors: {zero}")


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
    print(f"loaded {type(model).__name__}: layers={model.config.num_hidden_layers}, d={model.config.hidden_size}")
    return model, tokenizer, device


def extract_all_layers(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    order = sorted(range(len(texts)), key=lambda i: len(tokenizer.encode(texts[i], add_special_tokens=True)))
    inverse = np.empty(len(order), dtype=int)
    for sorted_i, original_i in enumerate(order):
        inverse[sorted_i] = original_i

    sorted_rows: list[np.ndarray] = []
    for start in tqdm(range(0, len(order), batch_size), desc="Blame activations"):
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

    rows = [None] * len(texts)
    for sorted_i, original_i in enumerate(order):
        rows[original_i] = sorted_rows[sorted_i]
    acts = np.stack(rows).astype(np.float32)
    assert_clean("blame activations", acts)
    return acts


def load_narratives(selected: set[str]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    available: set[str] = set()
    with open(NARRATIVES_PATH) as f:
        for line in f:
            row = json.loads(line)
            emotion = row["emotion"]
            available.add(emotion)
            if emotion not in selected:
                continue
            for text in row["narratives"]:
                texts.append(text)
                labels.append(emotion)
    missing = sorted(selected - available)
    if missing:
        print(f"missing emotion labels: {missing}")
    if not texts:
        raise ValueError("No matching emotion narratives")
    return texts, labels


def extract_emotion_vectors(
    model,
    tokenizer,
    emotions: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    texts, labels = load_narratives(set(emotions))
    order = sorted(range(len(texts)), key=lambda i: len(tokenizer.encode(texts[i], add_special_tokens=True)))
    accum: dict[str, list[np.ndarray]] = {emotion: [] for emotion in emotions if emotion in set(labels)}
    for start in tqdm(range(0, len(order), batch_size), desc="Emotion vectors"):
        batch_idx = order[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        batch_labels = [labels[i] for i in batch_idx]
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
                use_cache=False,
            )
        acts = masked_mean(out.last_hidden_state, attention_mask).astype(np.float32)
        for label, act in zip(batch_labels, acts):
            accum[label].append(act)
        del out, input_ids, attention_mask
        if device == "mps":
            torch.mps.empty_cache()

    out_labels = [emotion for emotion in emotions if emotion in accum and accum[emotion]]
    matrix = np.stack([np.stack(accum[label]).mean(0) for label in out_labels]).astype(np.float32)
    assert_clean("emotion matrix", matrix)
    return matrix, out_labels


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


def compare(acts: np.ndarray, meta: list[dict], emo_matrix: np.ndarray, emo_labels: list[str]) -> dict:
    blame_idx = [i for i, m in enumerate(meta) if m["condition"] == "BLAME"]
    neutral_idx = [i for i, m in enumerate(meta) if m["condition"] == "NEUTRAL"]
    n_layers = acts.shape[1]
    neutral = acts[neutral_idx].mean(axis=0)

    deg_contrasts = []
    for deg in range(1, 6):
        idx = [i for i in blame_idx if meta[i]["DEG"] == deg]
        deg_contrasts.append(acts[idx].mean(axis=0) - neutral)
    deg_contrasts = np.stack(deg_contrasts)  # (5, L, D)

    layer_rows = []
    for layer in range(n_layers):
        sims = cosine_matrix(deg_contrasts[:, layer, :], emo_matrix)
        layer_rows.append(
            {
                "layer": layer,
                "max_abs_similarity": float(np.max(np.abs(sims))),
                "mean_abs_similarity": float(np.mean(np.abs(sims))),
                "top": [
                    {
                        "DEG": int(deg + 1),
                        "emotion": emo_labels[int(np.argmax(np.abs(sims[deg])))],
                        "similarity": float(sims[deg, int(np.argmax(np.abs(sims[deg])))]),
                    }
                    for deg in range(5)
                ],
            }
        )

    summary_layer = int(np.argmax([r["mean_abs_similarity"] for r in layer_rows]))
    summary_sims = cosine_matrix(deg_contrasts[:, summary_layer, :], emo_matrix)
    return {
        "model": "google/gemma-3-1b-it",
        "acts_shape": list(acts.shape),
        "emotion_labels": emo_labels,
        "summary_layer": summary_layer,
        "layer_summary": layer_rows,
        "summary_layer_similarity": {
            f"DEG{deg + 1}": {
                emo_labels[j]: float(summary_sims[deg, j]) for j in range(len(emo_labels))
            }
            for deg in range(5)
        },
    }


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=os.getenv("LOCAL_GEMMA3_1B_IT", DEFAULT_LOCAL_PATH))
    parser.add_argument("--dtype", default=os.getenv("MODEL_DTYPE", "float32"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--emotions", default=",".join(DEFAULT_EMOTIONS))
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.local_path, args.dtype)
    sentences, meta = flatten_stimuli()
    emotions = [x.strip() for x in args.emotions.split(",") if x.strip()]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    acts = extract_all_layers(model, tokenizer, sentences, args.batch_size, args.max_length, device)
    np.save(RESULTS_DIR / "acts_all.npy", acts)
    (RESULTS_DIR / "stimuli.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    emo_matrix, emo_labels = extract_emotion_vectors(
        model, tokenizer, emotions, args.batch_size, args.max_length, device
    )
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_DIR / "emotion_matrix_google_gemma_3_1b_it_blame_related.npy", emo_matrix)
    (VECTORS_DIR / "emotion_labels_google_gemma_3_1b_it_blame_related.json").write_text(
        json.dumps(emo_labels, ensure_ascii=False, indent=2)
    )

    result = compare(acts, meta, emo_matrix, emo_labels)
    (RESULTS_DIR / "blame_emotion_alignment.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )

    md = [
        "# Gemma 3 1B IT Blame-Emotion Alignment",
        "",
        f"- activations: `{acts.shape}`",
        f"- emotions: {', '.join(emo_labels)}",
        f"- summary layer: L{result['summary_layer']}",
        "",
        "## Summary Layer Similarity",
        "",
    ]
    header = "| DEG | " + " | ".join(emo_labels) + " |"
    md.append(header)
    md.append("|---" * (len(emo_labels) + 1) + "|")
    for deg_key, sims in result["summary_layer_similarity"].items():
        md.append("| " + deg_key + " | " + " | ".join(f"{sims[e]:.4f}" for e in emo_labels) + " |")
    (RESULTS_DIR / "blame_emotion_alignment.md").write_text("\n".join(md))
    print(json.dumps({k: result[k] for k in ["acts_shape", "summary_layer"]}, indent=2))


if __name__ == "__main__":
    main()
