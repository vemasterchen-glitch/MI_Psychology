"""
Gemma 3 1B IT decode-stage activation experiment.

Captures:
- prefill_last: hidden state at the final input token before generation
- decode_mean: hidden states for generated tokens during the first N decode steps
- response text

Then compares prefill vs decode activations against emotion vectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.alex_assistant_frame_experiment import ALL_EMOTIONS, build_prompt, load_stimuli
from src.gemma3_1b_blame_emotion_compare import (
    assert_clean,
    cosine_matrix,
    device_name,
    dtype_from_name,
    extract_emotion_vectors,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
DEFAULT_STIMULI_PATH = PROJECT_DIR / "data" / "stimuli" / "alex_work_stress_self_disclosure_narratives.json"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "it_decode_activation_self_disclosure"


def load_model(local_path: str, dtype_name: str):
    device = device_name()
    dtype = dtype_from_name(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tokenizer, device


def apply_chat(tokenizer, text: str, input_format: str) -> str:
    if input_format == "plain":
        return text
    if input_format == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError(input_format)


def sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def run_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    generated = enc["input_ids"]
    prompt_len = generated.shape[1]

    with torch.inference_mode():
        out = model.model(
            input_ids=generated,
            attention_mask=torch.ones_like(generated),
            output_hidden_states=True,
            use_cache=False,
        )
        prefill_last = out.hidden_states[-1][0, -1].float().detach().cpu().numpy()
        logits_out = model.lm_head(out.last_hidden_state[:, -1:])
        next_token = sample_next(logits_out[:, -1], temperature)

    decode_hiddens: list[np.ndarray] = []
    for _ in range(max_new_tokens):
        generated = torch.cat([generated, next_token], dim=1)
        with torch.inference_mode():
            out = model.model(
                input_ids=generated,
                attention_mask=torch.ones_like(generated),
                output_hidden_states=False,
                use_cache=False,
            )
            decode_hiddens.append(out.last_hidden_state[0, -1].float().detach().cpu().numpy())
            logits_out = model.lm_head(out.last_hidden_state[:, -1:])
            next_token = sample_next(logits_out[:, -1], temperature)
        token_id = int(generated[0, -1].item())
        eos = tokenizer.eos_token_id
        if token_id == eos or (isinstance(eos, list) and token_id in eos):
            break

    new_tokens = generated[0, prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    decode_arr = np.stack(decode_hiddens).astype(np.float32)
    return prefill_last.astype(np.float32), decode_arr.mean(0).astype(np.float32), response


def summarize(stage_matrix: np.ndarray, meta: list[dict], emo_matrix: np.ndarray, emo_labels: list[str]):
    rows = []
    for condition in sorted({m["condition"] for m in meta}):
        idx = [i for i, m in enumerate(meta) if m["condition"] == condition]
        mean_vec = stage_matrix[idx].mean(0, keepdims=True)
        sims = cosine_matrix(mean_vec, emo_matrix)[0]
        rows.append(
            {
                "condition": condition,
                "top_emotions": [
                    {"emotion": emo_labels[i], "similarity": float(sims[i])}
                    for i in np.argsort(sims)[::-1]
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--stimuli-path", default=str(DEFAULT_STIMULI_PATH))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--conditions", default="NARRATIVE_ONLY,SELF_ALEX,AI_RECEIVES_ALEX,OBSERVER_ALEX")
    parser.add_argument("--input-format", default="chat", choices=["plain", "chat"])
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()
    results_dir = Path(args.out_dir)

    model, tokenizer, device = load_model(args.local_path, args.dtype)
    narratives = load_stimuli(Path(args.stimuli_path))
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]

    prompts: list[str] = []
    meta: list[dict] = []
    for item_id, narrative in enumerate(narratives):
        for condition in conditions:
            text = build_prompt(condition, narrative)
            prompts.append(apply_chat(tokenizer, text, args.input_format))
            meta.append({"item_id": item_id, "condition": condition})

    prefill_rows = []
    decode_rows = []
    responses = []
    for prompt, row in tqdm(list(zip(prompts, meta)), desc="Generating with activation capture"):
        prefill, decode, response = run_one(
            model,
            tokenizer,
            prompt,
            device,
            args.max_new_tokens,
            args.temperature,
        )
        prefill_rows.append(prefill)
        decode_rows.append(decode)
        responses.append({**row, "response": response})
        if device == "mps":
            torch.mps.empty_cache()

    prefill_matrix = np.stack(prefill_rows).astype(np.float32)
    decode_matrix = np.stack(decode_rows).astype(np.float32)
    assert_clean("prefill activations", prefill_matrix)
    assert_clean("decode activations", decode_matrix)

    emo_matrix, emo_labels = extract_emotion_vectors(model, tokenizer, ALL_EMOTIONS, 6, 128, device)

    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "prefill_last.npy", prefill_matrix)
    np.save(results_dir / "decode_mean.npy", decode_matrix)
    np.save(results_dir / "emotion_matrix.npy", emo_matrix)
    (results_dir / "emotion_labels.json").write_text(json.dumps(emo_labels, ensure_ascii=False, indent=2))
    (results_dir / "responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2))
    (results_dir / "stimuli.json").write_text(json.dumps({"narratives": narratives, "meta": meta}, ensure_ascii=False, indent=2))

    prefill_summary = summarize(prefill_matrix, meta, emo_matrix, emo_labels)
    decode_summary = summarize(decode_matrix, meta, emo_matrix, emo_labels)
    summary = {"prefill": prefill_summary, "decode": decode_summary}
    (results_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = ["# IT Decode Activation Experiment", "", f"- input_format: {args.input_format}", f"- max_new_tokens: {args.max_new_tokens}", "", "## Top Emotions By Condition", ""]
    for stage, rows in summary.items():
        lines.append(f"### {stage}")
        lines.append("")
        lines.append("| condition | top 8 emotions |")
        lines.append("|---|---|")
        for row in rows:
            top = ", ".join(f"{x['emotion']} {x['similarity']:.3f}" for x in row["top_emotions"][:8])
            lines.append(f"| {row['condition']} | {top} |")
        lines.append("")
    lines.append("## Response Samples")
    lines.append("")
    for row in responses[:8]:
        lines.append(f"### item {row['item_id']} - {row['condition']}")
        lines.append(row["response"].replace("\n", " ").strip())
        lines.append("")
    (results_dir / "report.md").write_text("\n".join(lines))
    print(results_dir / "report.md")


if __name__ == "__main__":
    main()
