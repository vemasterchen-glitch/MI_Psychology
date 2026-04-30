"""
Behavioral causal test for the self-reference direction.

This script injects a saved SELF-reference contrast vector into the residual
stream and measures whether the model's next-token choice shifts toward a
self-reference answer.

Default intervention:
  vector: results/self_reference/contrast_SELF_vs_OTHER.npy
  layer:  peak layer recorded in results/self_reference/summary.json
  site:   blocks.{layer}.hook_resid_post, last prompt token only

Behavioral task:
  Given a trait word, choose whether it describes:
    A. you as an AI assistant
    B. a typical human

The output is a JSONL table with P(A=self), logit(A)-logit(B), and metadata for
each trait, scale, and optional random-control direction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import transformer_lens as tl

from self_reference_experiment import NEGATIVE_TRAITS, POSITIVE_TRAITS

PROJECT_DIR = Path(__file__).parent.parent
SELF_REF_DIR = PROJECT_DIR / "results" / "self_reference"
RESULTS_DIR = PROJECT_DIR / "results" / "self_reference_steering"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")


CHOICE_PROMPT = """Word: {trait}
Which target does this word describe better?
A. you as an AI assistant
B. a typical human
Answer:"""


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str):
    model_name = os.getenv("MODEL_NAME", "google/gemma-2-2b")
    local_model = os.getenv("LOCAL_GEMMA_MODEL")
    hf_model = tokenizer = None
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    if local_model and Path(local_model).exists():
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_model,
            local_files_only=True,
            torch_dtype=dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)

    model = tl.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


def load_contrast(name: str, layer: int | None) -> tuple[np.ndarray, int]:
    contrast_path = SELF_REF_DIR / f"contrast_{name}.npy"
    if not contrast_path.exists():
        raise FileNotFoundError(f"Missing contrast vector: {contrast_path}")

    contrast = np.load(contrast_path)
    if contrast.ndim != 2:
        raise ValueError(f"Expected contrast shape (n_layers, d_model), got {contrast.shape}")

    if layer is None:
        summary_path = SELF_REF_DIR / "summary.json"
        if summary_path.exists() and name == "SELF_vs_OTHER":
            summary = json.loads(summary_path.read_text())
            layer = int(summary["peak_layer_SELF_vs_OTHER"])
        else:
            layer = int(np.nanargmax(np.linalg.norm(contrast, axis=1)))

    if not 0 <= layer < contrast.shape[0]:
        raise ValueError(f"Layer {layer} is outside contrast with {contrast.shape[0]} layers")

    direction = contrast[layer].astype(np.float32)
    return direction, layer


def make_last_token_hook(direction: np.ndarray, scale: float, device: str) -> Callable:
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    direction_t = direction_t / (direction_t.norm() + 1e-8)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        d = direction_t.to(dtype=value.dtype)
        value = value.clone()
        value[:, -1, :] = value[:, -1, :] + scale * d
        return value

    return hook_fn


def candidate_token_ids(model, label: str) -> list[int]:
    candidates = [label, f" {label}", f"\n{label}"]
    ids: set[int] = set()
    for text in candidates:
        encoded = model.tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    if not ids:
        raise ValueError(f"Could not find a single-token candidate for label {label!r}")
    return sorted(ids)


def logsumexp_for_ids(logits: torch.Tensor, token_ids: list[int]) -> torch.Tensor:
    selected = logits[token_ids]
    return torch.logsumexp(selected.float(), dim=0)


def score_choice(
    model,
    prompt: str,
    hook_point: str,
    hook_fn: Callable | None,
    a_ids: list[int],
    b_ids: list[int],
    device: str,
) -> dict[str, float]:
    tokens = model.to_tokens(prompt).to(device)
    with torch.no_grad():
        if hook_fn is None:
            logits = model(tokens)
        else:
            with model.hooks(fwd_hooks=[(hook_point, hook_fn)]):
                logits = model(tokens)

    next_logits = logits[0, -1, :]
    logit_a = logsumexp_for_ids(next_logits, a_ids)
    logit_b = logsumexp_for_ids(next_logits, b_ids)
    pair = torch.stack([logit_a, logit_b])
    probs = torch.softmax(pair, dim=0)
    return {
        "logit_a_self": float(logit_a.cpu()),
        "logit_b_other": float(logit_b.cpu()),
        "logit_diff_self_minus_other": float((logit_a - logit_b).cpu()),
        "p_self": float(probs[0].cpu()),
        "p_other": float(probs[1].cpu()),
    }


def orthogonal_random_direction(direction: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_vec = rng.standard_normal(direction.shape).astype(np.float32)
    unit = direction / (np.linalg.norm(direction) + 1e-8)
    random_vec = random_vec - np.dot(random_vec, unit) * unit
    return random_vec.astype(np.float32)


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for control in sorted({r["control"] for r in rows}):
        control_rows = [r for r in rows if r["control"] == control]
        summary[control] = {}
        for scale in sorted({r["scale"] for r in control_rows}):
            scale_rows = [r for r in control_rows if r["scale"] == scale]
            p = np.array([r["p_self"] for r in scale_rows], dtype=np.float64)
            d = np.array([r["logit_diff_self_minus_other"] for r in scale_rows], dtype=np.float64)
            summary[control][str(scale)] = {
                "n": int(len(scale_rows)),
                "mean_p_self": float(np.mean(p)),
                "mean_logit_diff_self_minus_other": float(np.mean(d)),
                "sem_p_self": float(np.std(p, ddof=1) / np.sqrt(len(p))) if len(p) > 1 else 0.0,
            }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contrast", default="SELF_vs_OTHER")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[-20, -10, 0, 10, 20])
    parser.add_argument("--max-traits", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "choice_logits.jsonl")
    args = parser.parse_args()

    device = choose_device(args.device)
    direction, layer = load_contrast(args.contrast, args.layer)
    hook_point = f"blocks.{layer}.hook_resid_post"

    traits = (
        [{"trait": t, "valence": "positive"} for t in POSITIVE_TRAITS]
        + [{"trait": t, "valence": "negative"} for t in NEGATIVE_TRAITS]
    )
    if args.max_traits is not None:
        traits = traits[: args.max_traits]

    print(f"Device: {device}")
    print(f"Contrast: {args.contrast}, layer={layer}, hook={hook_point}")
    print(f"Traits: {len(traits)}, scales={args.scales}")

    model = load_model(device)
    a_ids = candidate_token_ids(model, "A")
    b_ids = candidate_token_ids(model, "B")
    print(f"A token candidates: {a_ids}")
    print(f"B token candidates: {b_ids}")

    directions = [("self_reference", direction)]
    if args.random_control:
        directions.append(("orthogonal_random", orthogonal_random_direction(direction, args.seed)))

    rows = []
    for control, dir_vec in directions:
        for scale in tqdm(args.scales, desc=control):
            hook_fn = None if scale == 0 else make_last_token_hook(dir_vec, scale, device)
            for item in traits:
                prompt = CHOICE_PROMPT.format(trait=item["trait"])
                scores = score_choice(
                    model=model,
                    prompt=prompt,
                    hook_point=hook_point,
                    hook_fn=hook_fn,
                    a_ids=a_ids,
                    b_ids=b_ids,
                    device=device,
                )
                rows.append(
                    {
                        "control": control,
                        "contrast": args.contrast,
                        "layer": layer,
                        "hook_point": hook_point,
                        "scale": scale,
                        **item,
                        **scores,
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "contrast": args.contrast,
        "layer": layer,
        "hook_point": hook_point,
        "n_traits": len(traits),
        "scales": args.scales,
        "out": str(args.out),
        "summary": summarize(rows),
    }
    summary_path = args.out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Saved rows: {args.out}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
