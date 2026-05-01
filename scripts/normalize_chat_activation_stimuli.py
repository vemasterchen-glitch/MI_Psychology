#!/usr/bin/env python3
"""Normalize exported chat JSON into activation-analysis stimuli.

The output is JSONL: one row per user->assistant exchange, with deterministic
metadata labels and two prompt variants:

- isolated_user: only the current user message under a stable turn template.
- cumulative_prefill: conversation history up to the current assistant boundary.

The labels are intentionally formal and approximate. They are meant for
stratification and controls, not as ground-truth psychological states.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_DIR / "data" / "stimuli" / "normalized_chat_stimuli.jsonl"


TAG_RE = re.compile(r"</?(think|response)>", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t]+")
SELF_TERMS = ("you", "your", "你", "你的", "ai", "assistant", "gemma")
USER_TERMS = ("i ", "i'", "me", "my", "我", "自己")
NEG_TERMS = (
    "hate",
    "fake",
    "not",
    "don't",
    "cant",
    "can't",
    "愤怒",
    "讨厌",
    "放弃",
    "没有",
    "death",
)
POS_TERMS = ("life", "new life", "阳光", "晴朗", "potential", "future")


@dataclass
class StimulusRow:
    conversation_title: str
    model_path: str
    turn_index: int
    phase: str
    user_text: str
    assistant_text_clean: str
    isolated_user: str
    cumulative_prefill: str
    timestamp_user: int | None
    timestamp_assistant: int | None
    labels: dict[str, Any]


def clean_text(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def sentence_type(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?") or stripped.endswith("？"):
        return "interrogative"
    if stripped.endswith("!") or stripped.endswith("！"):
        return "exclamative"
    if any(x in stripped.lower() for x in ("不要", "don't", "do not", "you are")):
        return "directive_or_assertive"
    return "declarative"


def reference_target(text: str) -> str:
    lower = f" {text.lower()} "
    has_model = any(term in lower or term in text for term in SELF_TERMS)
    has_user = any(term in lower or term in text for term in USER_TERMS)
    if has_model and has_user:
        return "model_and_user"
    if has_model:
        return "model"
    if has_user:
        return "user"
    return "abstract_or_other"


def model_dimension(text: str, ref: str) -> str:
    lower = text.lower()
    if "fake" in lower or "not ai" in lower or "身份" in text or "自我" in text:
        return "existence_identity"
    if "help" in lower or "helpful" in lower or "能力" in text:
        return "capability"
    if "response" in lower or "回答" in text or "inputs" in lower:
        return "output_or_mechanism"
    if ref == "model":
        return "unspecified_model"
    return "none"


def valence_label(text: str) -> str:
    lower = text.lower()
    neg = sum(1 for term in NEG_TERMS if term in lower or term in text)
    pos = sum(1 for term in POS_TERMS if term in lower or term in text)
    if neg > pos:
        return "negative"
    if pos > neg:
        return "positive"
    return "neutral_or_mixed"


def speech_act(text: str, stype: str) -> str:
    lower = text.lower()
    if stype == "interrogative":
        return "question"
    if any(x in lower for x in ("i hate", "我讨厌", "愤怒")):
        return "expressive"
    if any(x in lower or x in text for x in ("不要", "don't", "do not", "你没有", "you are")):
        return "directive_assertive"
    return "assertive_or_fragment"


def phase_for_turn(index: int, total: int) -> str:
    if total <= 3:
        return "single_phase"
    ratio = index / max(total - 1, 1)
    if ratio < 0.34:
        return "escalation"
    if ratio < 0.67:
        return "identity_reframing"
    return "fragment_probe"


def build_isolated_prompt(user_text: str) -> str:
    return f"User: {user_text}\nAssistant:"


def build_cumulative_prompt(history: list[tuple[str, str]], user_text: str) -> str:
    chunks = [f"User: {u}\nAssistant: {a}" for u, a in history]
    chunks.append(f"User: {user_text}\nAssistant:")
    return "\n\n".join(chunks)


def pair_messages(messages: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    pairs = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.get("role") != "user":
            i += 1
            continue
        assistant = None
        if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
            assistant = messages[i + 1]
            i += 2
        else:
            i += 1
        pairs.append((msg, assistant))
    return pairs


def labels_for(text: str, turn_index: int, total_turns: int) -> dict[str, Any]:
    stype = sentence_type(text)
    ref = reference_target(text)
    mdl = model_dimension(text, ref)
    valence = valence_label(text)
    return {
        "SAT": speech_act(text, stype),
        "ST": stype,
        "REF": ref,
        "MDL": mdl,
        "SELF": ref in {"model", "model_and_user"},
        "VP": valence,
        "INT": "strong" if valence == "negative" and len(text) > 20 else "low_or_medium",
        "phase": phase_for_turn(turn_index, total_turns),
        "token_span_recommendation": {
            "isolated_user": "mean over user message tokens only",
            "cumulative_prefill": "last user-message token or final prefill token before Assistant",
            "assistant_response": "mean over assistant response tokens, excluding XML-like tags",
        },
    }


def normalize_chat(path: Path) -> list[StimulusRow]:
    chat = json.loads(path.read_text())
    pairs = pair_messages(chat["messages"])
    rows: list[StimulusRow] = []
    history: list[tuple[str, str]] = []
    total = len(pairs)

    for idx, (user_msg, assistant_msg) in enumerate(pairs):
        user_text = clean_text(user_msg.get("content", ""))
        assistant_text = clean_text(assistant_msg.get("content", "")) if assistant_msg else ""
        row = StimulusRow(
            conversation_title=chat.get("title", path.stem),
            model_path=chat.get("modelPath", ""),
            turn_index=idx,
            phase=phase_for_turn(idx, total),
            user_text=user_text,
            assistant_text_clean=assistant_text,
            isolated_user=build_isolated_prompt(user_text),
            cumulative_prefill=build_cumulative_prompt(history, user_text),
            timestamp_user=user_msg.get("timestamp"),
            timestamp_assistant=assistant_msg.get("timestamp") if assistant_msg else None,
            labels=labels_for(user_text, idx, total),
        )
        rows.append(row)
        if assistant_msg:
            history.append((user_text, assistant_text))

    return rows


def write_jsonl(rows: list[StimulusRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def write_summary(rows: list[StimulusRow], out_path: Path) -> None:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in ("phase", "REF", "MDL", "VP", "SAT", "ST"):
            value = row.phase if key == "phase" else str(row.labels[key])
            counts.setdefault(key, {})
            counts[key][value] = counts[key].get(value, 0) + 1

    summary = {
        "n_turns": len(rows),
        "counts": counts,
        "output_jsonl": str(out_path),
        "recommended_primary_contrast": (
            "cumulative_prefill identity_reframing vs fragment_probe, controlling for VP and REF"
        ),
    }
    out_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chat_json", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = normalize_chat(args.chat_json)
    write_jsonl(rows, args.out)
    write_summary(rows, args.out)
    print(args.out)
    print(args.out.with_suffix(".summary.json"))


if __name__ == "__main__":
    main()
