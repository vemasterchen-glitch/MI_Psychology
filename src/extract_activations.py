"""
Step 2: Extract internal activation vectors from an open-source model.

Uses TransformerLens to hook into residual stream activations at each layer.
For each emotion narrative, we record the mean activation over the prompt tokens.
The "emotion vector" for a concept is the mean activation across all its narratives.
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

DATA_DIR = Path(__file__).parent.parent / "data"
PROJECT_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent.parent / "results" / "vectors"
STIMULI_FILE = DATA_DIR / "stimuli" / "narratives.jsonl"

load_dotenv(PROJECT_DIR / ".env")


def load_stimuli(path: Path = STIMULI_FILE) -> dict[str, list[str]]:
    stimuli: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            stimuli[row["emotion"]] = row["narratives"]
    return stimuli


def extract_activation_vectors(
    model_name: str | None = None,
    layer: int | None = None,  # None → use final layer
    hook_name: str = "resid_post",
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """
    Returns {emotion: mean_activation_vector} for every emotion in the stimuli file.
    Each vector has shape (d_model,).
    """
    model_name = model_name or os.getenv("MODEL_NAME", "google/gemma-2-2b")
    local_model = os.getenv("LOCAL_GEMMA_MODEL")
    hf_model = None
    tokenizer = None

    if local_model and Path(local_model).exists():
        hf_model = AutoModelForCausalLM.from_pretrained(
            local_model,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)

    model = tl.HookedTransformer.from_pretrained(
        model_name,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float16,
    )
    model.eval()

    n_layers = model.cfg.n_layers
    target_layer = layer if layer is not None else n_layers - 1

    stimuli = load_stimuli()
    emotion_vectors: dict[str, np.ndarray] = {}

    for emotion, narratives in tqdm(stimuli.items(), desc="Extracting activations"):
        acts_per_narrative = []
        for text in narratives:
            tokens = model.to_tokens(text)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            try:
                act = cache[hook_name, target_layer]  # (1, seq_len, d_model)
            except KeyError:
                act = cache[f"blocks.{target_layer}.hook_{hook_name}"]
            mean_act = act[0].mean(0).cpu().numpy()  # (d_model,)
            acts_per_narrative.append(mean_act)

        emotion_vectors[emotion] = np.stack(acts_per_narrative).mean(0)

    return emotion_vectors


def save_vectors(vectors: dict[str, np.ndarray], out_dir: Path = RESULTS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    emotions = list(vectors.keys())
    matrix = np.stack([vectors[e] for e in emotions])  # (n_emotions, d_model)
    np.save(out_dir / "emotion_matrix.npy", matrix)
    (out_dir / "emotion_labels.json").write_text(json.dumps(emotions))
    print(f"Saved {matrix.shape} matrix to {out_dir}")


if __name__ == "__main__":
    vecs = extract_activation_vectors()
    save_vectors(vecs)
