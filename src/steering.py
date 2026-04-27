"""
Step 5: Activation steering — causal intervention experiments.

Add or subtract emotion vectors at a given layer's residual stream,
then observe how the model's output distribution changes.
This replicates the "amplify/suppress emotion" experiments in the paper.
"""

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import transformer_lens as tl

RESULTS_DIR = Path(__file__).parent.parent / "results" / "steering"


def load_emotion_vector(emotion: str, vectors_dir: Path) -> np.ndarray:
    matrix = np.load(vectors_dir / "emotion_matrix.npy")
    labels = json.loads((vectors_dir / "emotion_labels.json").read_text())
    idx = labels.index(emotion)
    return matrix[idx]


def make_steering_hook(
    direction: np.ndarray,
    scale: float,
    device: str = "cpu",
) -> Callable:
    """Returns a TransformerLens hook that adds `scale * direction` to the residual stream."""
    direction_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
    direction_tensor = direction_tensor / (direction_tensor.norm() + 1e-8)

    def hook_fn(value: torch.Tensor, hook) -> torch.Tensor:
        return value + scale * direction_tensor

    return hook_fn


def steered_generate(
    model: tl.HookedTransformer,
    prompt: str,
    emotion: str,
    vectors_dir: Path,
    scale: float = 20.0,
    layer: int | None = None,
    max_new_tokens: int = 100,
    device: str = "cpu",
) -> dict[str, str]:
    """
    Generate text with and without the emotion steering vector.
    Returns {"baseline": ..., "steered": ...}.
    """
    n_layers = model.cfg.n_layers
    target_layer = layer if layer is not None else n_layers // 2
    direction = load_emotion_vector(emotion, vectors_dir)
    hook = make_steering_hook(direction, scale, device)
    hook_point = f"blocks.{target_layer}.hook_resid_post"

    tokens = model.to_tokens(prompt)

    with torch.no_grad():
        baseline_ids = model.generate(tokens, max_new_tokens=max_new_tokens)
        with model.hooks(fwd_hooks=[(hook_point, hook)]):
            steered_ids = model.generate(tokens, max_new_tokens=max_new_tokens)

    return {
        "baseline": model.to_string(baseline_ids[0]),
        "steered": model.to_string(steered_ids[0]),
    }


def batch_steering_eval(
    model: tl.HookedTransformer,
    prompts: list[str],
    emotion: str,
    scales: list[float],
    vectors_dir: Path,
    device: str = "cpu",
) -> list[dict]:
    """Sweep over multiple steering scales and record outputs."""
    results = []
    for prompt in prompts:
        for scale in scales:
            out = steered_generate(model, prompt, emotion, vectors_dir, scale=scale, device=device)
            results.append({"prompt": prompt, "emotion": emotion, "scale": scale, **out})
    return results
