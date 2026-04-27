"""
Step 1: Generate emotion narratives via Claude API.

For each of the 171 emotion concepts, prompt Claude to write a short narrative
featuring a character experiencing that emotion. These narratives serve as
stimuli for activation extraction.
"""

import json
from pathlib import Path

import anthropic
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
EMOTIONS_FILE = DATA_DIR / "emotions.txt"
STIMULI_DIR = DATA_DIR / "stimuli"

NARRATIVE_PROMPT = """\
Write a short narrative (3-5 sentences) about a character who is experiencing \
the emotion of {emotion}. Make the emotional state vivid and central to the scene. \
Focus on the internal experience. Do not name the emotion explicitly.\
"""


def load_emotions() -> list[str]:
    lines = EMOTIONS_FILE.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def generate_narratives(
    emotions: list[str],
    n_per_emotion: int = 5,
    model: str = "claude-sonnet-4-6",
    out_file: Path | None = None,
) -> dict[str, list[str]]:
    client = anthropic.Anthropic()
    out_file = out_file or STIMULI_DIR / "narratives.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = {}

    with open(out_file, "w") as f:
        for emotion in tqdm(emotions, desc="Generating narratives"):
            narratives = []
            for _ in range(n_per_emotion):
                msg = client.messages.create(
                    model=model,
                    max_tokens=256,
                    messages=[{"role": "user", "content": NARRATIVE_PROMPT.format(emotion=emotion)}],
                )
                narratives.append(msg.content[0].text.strip())

            results[emotion] = narratives
            f.write(json.dumps({"emotion": emotion, "narratives": narratives}) + "\n")
            f.flush()

    return results


if __name__ == "__main__":
    emotions = load_emotions()
    print(f"Loaded {len(emotions)} emotions")
    generate_narratives(emotions[:5], n_per_emotion=2)  # smoke test
    print("Done — output in data/stimuli/narratives.jsonl")
