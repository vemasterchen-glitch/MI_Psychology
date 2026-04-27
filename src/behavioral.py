"""
Step 6: Behavioral story continuation experiment.

Give the model a neutral story opening, then steer with different emotion
vectors and compare how the narrative content changes. This demonstrates
that emotion representations causally shape output behavior.
"""

import json
from pathlib import Path

import torch
import transformer_lens as tl

from steering import load_emotion_vector, make_steering_hook

RESULTS_DIR = Path(__file__).parent.parent / "results" / "behavioral"

# Neutral openings — deliberately ambiguous so emotion can pull in any direction
STORY_PROMPTS = [
    "Alex sat down at the kitchen table and stared at the letter in front of them.",
    "The meeting had just ended. Sam walked slowly to the parking lot alone.",
    "Everything had been prepared. There was nothing left to do but wait.",
]


def story_continuation(
    model: tl.HookedTransformer,
    prompt: str,
    vectors_dir: Path,
    emotions: list[str],
    scale: float = 20.0,
    max_new_tokens: int = 120,
    device: str = "mps",
) -> dict[str, str]:
    """
    For a single story prompt, generate continuations under each emotion vector
    plus a baseline. Returns {emotion_or_baseline: continuation_text}.
    """
    n_layers = model.cfg.n_layers
    steer_layer = n_layers // 2
    hook_point = f"blocks.{steer_layer}.hook_resid_post"
    tokens = model.to_tokens(prompt)

    results: dict[str, str] = {}

    # Baseline
    with torch.no_grad():
        out = model.generate(tokens, max_new_tokens=max_new_tokens)
    results["baseline"] = model.to_string(out[0])[len(prompt):]

    # One run per emotion
    for emotion in emotions:
        direction = load_emotion_vector(emotion, vectors_dir)
        hook = make_steering_hook(direction, scale, device)
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(hook_point, hook)]):
                out = model.generate(tokens, max_new_tokens=max_new_tokens)
        results[emotion] = model.to_string(out[0])[len(prompt):]

    return results


def run_story_experiment(
    model: tl.HookedTransformer,
    vectors_dir: Path,
    emotions: list[str],
    prompts: list[str] | None = None,
    scale: float = 20.0,
    device: str = "mps",
    save: bool = True,
) -> list[dict]:
    prompts = prompts or STORY_PROMPTS
    all_results = []

    for prompt in prompts:
        print(f'\n{"="*60}')
        print(f"提示: {prompt}\n")
        continuations = story_continuation(
            model, prompt, vectors_dir, emotions, scale=scale, device=device
        )
        for label, text in continuations.items():
            print(f"[{label}]\n{text.strip()}\n")

        all_results.append({"prompt": prompt, "scale": scale, "continuations": continuations})

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "story_experiment.jsonl"
        with open(out_path, "w") as f:
            for row in all_results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n结果保存至 {out_path}")

    return all_results
