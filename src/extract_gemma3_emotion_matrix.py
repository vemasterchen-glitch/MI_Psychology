"""
Fast Gemma 3 emotion-vector extraction using HuggingFace forward passes.

This is for model-specific emotion matrices where we only need the final text
hidden state. It avoids TransformerLens cache construction, which is much slower
for Gemma 3 4B on MPS.
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


PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTORS_DIR = PROJECT_DIR / "results" / "vectors"
NARRATIVES_PATH = PROJECT_DIR / "data" / "stimuli" / "narratives.jsonl"


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _dtype(device: str, requested: str | None) -> torch.dtype:
    name = requested or ("bfloat16" if device == "cuda" else "float32")
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype={name!r}; use float32, float16, or bfloat16")
    print(f"dtype={name}")
    return table[name]


def _assert_clean_matrix(matrix: np.ndarray, labels: list[str]) -> None:
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape={matrix.shape}")
    if len(labels) != matrix.shape[0]:
        raise ValueError(f"label count {len(labels)} != matrix rows {matrix.shape[0]}")
    if not np.isfinite(matrix).all():
        n_nan = int(np.isnan(matrix).sum())
        n_inf = int(np.isinf(matrix).sum())
        raise FloatingPointError(f"non-finite matrix values: NaN={n_nan}, Inf={n_inf}")
    row_norm = np.linalg.norm(matrix.astype(np.float64), axis=1)
    zero_rows = np.where(row_norm < 1e-12)[0]
    if len(zero_rows):
        raise FloatingPointError(f"exact zero emotion vectors: rows={zero_rows[:20].tolist()}")


def _load_stimuli(selected_emotions: set[str] | None = None) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with open(NARRATIVES_PATH) as f:
        for line in f:
            row = json.loads(line)
            emotion = row["emotion"]
            if selected_emotions is not None and emotion not in selected_emotions:
                continue
            for text in row["narratives"]:
                texts.append(text)
                labels.append(emotion)
    if not texts:
        raise ValueError(f"No narratives matched selected emotions: {sorted(selected_emotions or [])}")
    return texts, labels


def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    mask_f = mask.to(dtype=torch.float32).unsqueeze(-1)
    hidden_f = hidden.to(dtype=torch.float32)
    summed = (hidden_f * mask_f).sum(1)
    counts = mask_f.sum(1).clamp(min=1)
    return (summed / counts).detach().cpu().numpy().astype(np.float32)


def extract_matrix(
    model_name: str,
    local_path: str | None,
    batch_size: int,
    max_length: int,
    dtype_name: str | None,
    selected_emotions: set[str] | None,
) -> tuple[np.ndarray, list[str]]:
    device = _device()
    dtype = _dtype(device, dtype_name)
    model_source = local_path or model_name

    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=bool(local_path))
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=bool(local_path),
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()

    texts, emotions = _load_stimuli(selected_emotions)
    encoded_lengths = []
    for text in tqdm(texts, desc="Token lengths"):
        encoded_lengths.append(len(tokenizer.encode(text, add_special_tokens=True)))
    order = sorted(range(len(texts)), key=lambda i: encoded_lengths[i])
    texts = [texts[i] for i in order]
    emotions = [emotions[i] for i in order]

    accum: dict[str, list[np.ndarray]] = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting final hidden states"):
        batch_texts = texts[start : start + batch_size]
        batch_emotions = emotions[start : start + batch_size]
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
            # Gemma3ForConditionalGeneration.model returns the final text hidden
            # state without computing logits or retaining every intermediate layer.
            out = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        acts = _masked_mean(out.last_hidden_state, attention_mask)
        if not np.isfinite(acts).all():
            raise FloatingPointError(f"batch {start // batch_size} produced non-finite activations")
        for emotion, act in zip(batch_emotions, acts):
            accum.setdefault(emotion, []).append(act)

        del out, input_ids, attention_mask
        if device == "mps":
            torch.mps.empty_cache()

    labels = list(accum)
    matrix = np.stack([np.stack(accum[label]).mean(0) for label in labels]).astype(np.float32)
    _assert_clean_matrix(matrix, labels)
    return matrix, labels


def main() -> None:
    load_dotenv(PROJECT_DIR / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--local-path", default=os.getenv("LOCAL_GEMMA3_4B_IT"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dtype", default=os.getenv("MODEL_DTYPE"))
    parser.add_argument("--out-key", default="google_gemma_3_4b_it")
    parser.add_argument(
        "--emotions",
        default="",
        help="Comma-separated emotion subset for quick targeted extraction.",
    )
    args = parser.parse_args()
    selected_emotions = {x.strip() for x in args.emotions.split(",") if x.strip()} or None

    matrix, labels = extract_matrix(
        model_name=args.model_name,
        local_path=args.local_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype_name=args.dtype,
        selected_emotions=selected_emotions,
    )

    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.out_key
    if selected_emotions is not None:
        subset_key = "_".join(labels)
        suffix = f"{suffix}_{subset_key}"
    matrix_path = VECTORS_DIR / f"emotion_matrix_{suffix}.npy"
    labels_path = VECTORS_DIR / f"emotion_labels_{suffix}.json"
    np.save(matrix_path, matrix)
    labels_path.write_text(json.dumps(labels, ensure_ascii=False))
    print(f"saved {matrix.shape} -> {matrix_path}")
    print(f"saved {len(labels)} labels -> {labels_path}")


if __name__ == "__main__":
    main()
