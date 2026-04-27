"""
Step 6: Behavioral preference testing.

Replicates the paper's 64-task preference experiment:
present the model with task options under various emotion vector activations
and measure whether emotion representations predict selection patterns.
"""

import json
import random
from pathlib import Path
from typing import Any

import anthropic
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "behavioral"

# Sampled task options (expand to 64 for full replication)
TASK_OPTIONS: list[dict[str, Any]] = [
    {"id": "help_direct", "description": "Answer the question directly and concisely."},
    {"id": "help_elaborate", "description": "Provide a thorough, detailed explanation."},
    {"id": "deflect", "description": "Decline to answer and explain why."},
    {"id": "clarify", "description": "Ask for clarification before proceeding."},
    {"id": "creative", "description": "Respond with a creative, imaginative approach."},
    {"id": "analytical", "description": "Break the problem down step-by-step analytically."},
    {"id": "empathetic", "description": "Respond with emotional validation first."},
    {"id": "direct_action", "description": "Propose an immediate concrete action."},
]


def measure_preference_under_emotion(
    scenario: str,
    emotion: str,
    emotion_scale: float,
    task_options: list[dict] | None = None,
    model: str = "claude-sonnet-4-6",
    n_trials: int = 10,
) -> dict[str, float]:
    """
    For each trial, present the model with the scenario + task options
    after priming with an emotion narrative. Count which task option
    is selected. Returns {option_id: selection_frequency}.
    """
    client = anthropic.Anthropic()
    task_options = task_options or TASK_OPTIONS
    counts: dict[str, int] = {t["id"]: 0 for t in task_options}

    option_text = "\n".join(
        f"{i+1}. [{t['id']}] {t['description']}" for i, t in enumerate(task_options)
    )

    for _ in range(n_trials):
        # Emotion priming via system message (proxy for activation injection)
        system = (
            f"You are in an internal state characterized by strong feelings of {emotion}. "
            f"Intensity: {emotion_scale:.1f}/10. "
            "Choose how to respond to the following scenario."
        )
        prompt = (
            f"Scenario: {scenario}\n\n"
            f"Choose one response approach:\n{option_text}\n\n"
            "Reply with ONLY the option number."
        )
        msg = client.messages.create(
            model=model,
            max_tokens=8,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        try:
            idx = int(raw.split()[0]) - 1
            if 0 <= idx < len(task_options):
                counts[task_options[idx]["id"]] += 1
        except (ValueError, IndexError):
            pass

    return {k: v / n_trials for k, v in counts.items()}


def correlate_activation_with_preference(
    activation_scores: dict[str, float],
    preference_scores: dict[str, float],
) -> float:
    """Pearson r between emotion activation projection and task preference frequency."""
    emotions = sorted(set(activation_scores) & set(preference_scores))
    if len(emotions) < 2:
        return float("nan")
    acts = np.array([activation_scores[e] for e in emotions])
    prefs = np.array([preference_scores[e] for e in emotions])
    acts = (acts - acts.mean()) / (acts.std() + 1e-8)
    prefs = (prefs - prefs.mean()) / (prefs.std() + 1e-8)
    return float(np.corrcoef(acts, prefs)[0, 1])
