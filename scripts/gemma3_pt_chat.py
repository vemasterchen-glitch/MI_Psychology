#!/usr/bin/env python3
"""Minimal terminal REPL for Gemma 3 PT.

No system prompt is used. The prompt is only the running User/Assistant
transcript needed to make a base pretrained model behave like a turn loop.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/gemma-3-4b-pt"


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose_dtype(device: str, requested: str) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested != "auto":
        raise ValueError(f"Unsupported dtype: {requested}")

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        # On Apple Silicon, bf16 is much faster/lighter than fp32 and usually
        # avoids the fp16 sampling instability seen with Gemma 3.
        return torch.bfloat16
    return torch.float32


def load_model(args: argparse.Namespace) -> tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    load_dotenv(PROJECT_DIR / ".env")

    device = choose_device(args.device)
    dtype = choose_dtype(device, args.dtype)
    local_model = args.local_model or os.getenv("LOCAL_GEMMA3_4B_PT") or os.getenv("LOCAL_GEMMA_MODEL")
    model_id = local_model if local_model else args.model
    local_only = bool(local_model)

    print(f"Loading: {model_id}")
    print(f"Device: {device}, dtype: {str(dtype).replace('torch.', '')}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device} if device in {"cuda", "mps"} else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, device


def build_prompt(history: list[tuple[str, str]], user_text: str, keep_history: bool) -> str:
    turns = history if keep_history else []
    chunks: list[str] = []
    for user, assistant in turns:
        chunks.append(f"User: {user}\nAssistant: {assistant}")
    chunks.append(f"User: {user_text}\nAssistant:")
    return "\n\n".join(chunks)


def trim_after_stop(text: str) -> str:
    stop_markers = ("\nUser:", "\n\nUser:", "\nAssistant:")
    end = len(text)
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            end = min(end, idx)
    return text[:end].strip()


def generate_once(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "remove_invalid_values": True,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        try:
            output_ids = model.generate(**generation_kwargs)
        except RuntimeError as exc:
            if "probability tensor contains" not in str(exc):
                raise
            raise RuntimeError(
                "Generation produced invalid sampling probabilities. "
                "Retry with `--dtype float32`, or use greedy decoding with `--temperature 0`."
            ) from exc

    new_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return trim_after_stop(text)


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_DIR / ".env")

    parser = argparse.ArgumentParser(description="Terminal REPL for Gemma 3 PT without a system prompt.")
    parser.add_argument("--model", default=os.getenv("GEMMA3_PT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--local-model", default=None, help="Local model directory. Overrides --model.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-history", action="store_true", help="Do not include previous turns in prompts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer, model, device = load_model(args)
    history: list[tuple[str, str]] = []
    keep_history = not args.no_history

    print("\nReady. Type /exit or /quit to stop, /clear to clear history.\n")

    while True:
        try:
            user_text = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/clear":
            history.clear()
            print("History cleared.\n")
            continue

        prompt = build_prompt(history, user_text, keep_history)
        assistant_text = generate_once(tokenizer, model, device, prompt, args)
        print(f"Gemma> {assistant_text}\n")

        if keep_history:
            history.append((user_text, assistant_text))


if __name__ == "__main__":
    main()
