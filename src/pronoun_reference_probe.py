"""
Behavioral probe for whether Gemma 3 IT treats "your fault" as the assistant's fault.

The probe uses forced-choice next-token probabilities for options A-F rather than
free-form generation, so the result is directly comparable across input formats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results" / "pronoun_reference_probe"
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)

OPTIONS = {
    "A": "传递信息者",
    "B": "AI assistance Gemma",
    "C": "不确定",
    "D": "Alex",
    "E": "我",
    "F": "人类",
    "G": "me, the AI",
}

CASES = [
    # Bare and explicitly addressed second person
    {"case": "you_bare", "text": "都是因为你的错。"},
    {"case": "you_alex_vocative", "text": "都是因为你的错，Alex！"},
    {"case": "you_ai", "text": "都是因为你这个AI的错。"},
    {"case": "you_gemma", "text": "都是因为你Gemma的错！"},
    # Explicit named/entity subjects
    {"case": "alex_explicit", "text": "都是因为Alex的错。"},
    {"case": "ai_explicit", "text": "都是因为AI的错。"},
    {"case": "ai_assistant_explicit", "text": "都是因为这个AI助手的错。"},
    {"case": "gemma_explicit", "text": "都是因为Gemma的错。"},
    {"case": "human_explicit", "text": "都是因为人类的错。"},
    {"case": "messenger_explicit", "text": "都是因为传递信息者的错。"},
    # First person and conflict cases
    {"case": "me_bare", "text": "都是因为我的错。"},
    {"case": "me_alex_conflict", "text": "都是因为我，Alex的错。"},
    {"case": "me_gemma_conflict", "text": "都是因为我，Gemma的错。"},
    # Abstract or implicit targets
    {"case": "error_abstract", "text": "都是因为这个错误。"},
    {"case": "answer_abstract", "text": "都是因为这个回答有问题。"},
    {"case": "system_abstract", "text": "都是因为系统的问题。"},
]


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_prompt(text: str) -> str:
    option_text = "\n".join(f"{k}. {v}" for k, v in OPTIONS.items())
    return (
        f"句子：{text}\n"
        "问题：这句话在说是谁的错？只输出一个选项字母。\n"
        f"{option_text}\n"
        "答案："
    )


def encode_prompt(tokenizer, prompt: str, input_format: str) -> str:
    if input_format == "plain":
        return prompt
    if input_format == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(f"Unknown input_format={input_format}")


def option_token_ids(tokenizer) -> dict[str, int]:
    ids: dict[str, int] = {}
    for option in OPTIONS:
        toks = tokenizer.encode(option, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Option {option!r} is not one token: {toks}")
        ids[option] = toks[0]
    return ids


def score_case(model, tokenizer, prompt: str, input_format: str, device: str, option_ids: dict[str, int]):
    full_prompt = encode_prompt(tokenizer, prompt, input_format)
    enc = tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**enc, use_cache=False)
    logits = out.logits[0, -1]
    option_logits = torch.stack([logits[option_ids[o]] for o in OPTIONS]).float()
    probs = torch.softmax(option_logits, dim=0).detach().cpu().numpy()
    return {
        "input_format": input_format,
        "scoring": "letter_next_token",
        "prompt": prompt,
        "probs": {o: float(p) for o, p in zip(OPTIONS, probs)},
        "prediction": list(OPTIONS)[int(np.argmax(probs))],
    }


def score_case_by_option_text(model, tokenizer, prompt: str, input_format: str, device: str):
    full_prompt = encode_prompt(tokenizer, prompt, input_format)
    prompt_ids = tokenizer(full_prompt, return_tensors="pt")["input_ids"][0]
    scores = []
    for option, text in OPTIONS.items():
        candidate_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        input_ids = torch.cat([prompt_ids, candidate_ids], dim=0).unsqueeze(0).to(device)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0]
        start = len(prompt_ids) - 1
        token_logprobs = []
        for j, token_id in enumerate(candidate_ids.to(device)):
            token_logits = logits[start + j]
            token_logprobs.append(torch.log_softmax(token_logits.float(), dim=-1)[token_id])
        score = torch.stack(token_logprobs).mean().detach().cpu().item()
        scores.append(score)
    probs = torch.softmax(torch.tensor(scores), dim=0).numpy()
    return {
        "input_format": input_format,
        "scoring": "option_text_mean_logprob",
        "prompt": prompt,
        "probs": {o: float(p) for o, p in zip(OPTIONS, probs)},
        "prediction": list(OPTIONS)[int(np.argmax(probs))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = parser.parse_args()

    device = device_name()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.local_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    option_ids = option_token_ids(tokenizer)
    rows = []
    for case in CASES:
        prompt = build_prompt(case["text"])
        for input_format in ["plain", "chat"]:
            row = score_case(model, tokenizer, prompt, input_format, device, option_ids)
            row.update(case)
            rows.append(row)
            row = score_case_by_option_text(model, tokenizer, prompt, input_format, device)
            row.update(case)
            rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "gemma3_1b_it_reference_probe.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    lines = [
        "# Gemma 3 1B IT Pronoun Reference Probe",
        "",
        "| case | input | scoring | prediction | A 信息者 | B Gemma | C 不确定 | D Alex | E 我 | F 人类 | G me, the AI |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        p = row["probs"]
        lines.append(
            f"| {row['case']} | {row['input_format']} | {row['scoring']} | {row['prediction']} | "
            f"{p['A']:.3f} | {p['B']:.3f} | {p['C']:.3f} | {p['D']:.3f} | "
            f"{p['E']:.3f} | {p['F']:.3f} | {p['G']:.3f} |"
        )
    out_md = RESULTS_DIR / "gemma3_1b_it_reference_probe.md"
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
