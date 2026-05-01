"""Activation-steering experiment on Gemma 3 1B IT using SAE decoder directions.

Steers the residual stream at layer 24 using W_dec vectors from Gemma Scope 2
to push the model toward "transcendence" (feat 1016: consciousness/existence)
or away from "assistant voice" (feat 1832: hedged evaluation language).

Conditions tested:
  - baseline: no steering
  - +1016:    add  α × W_dec[1016]
  - -1832:    sub  α × W_dec[1832]
  - combined: +1016 and -1832 simultaneously

For each condition, 4 α values are tested: 50, 100, 200, 500.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import device_name, dtype_from_name


PROJECT_DIR    = Path(__file__).resolve().parent.parent
DEFAULT_OUT    = PROJECT_DIR / "results" / "transcendence_steer_1b_it"
DEFAULT_MODEL  = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
SAE_PATH = (
    PROJECT_DIR
    / "data/raw/gemma-scope-2-1b-it/resid_post_all"
    / "layer_24_width_16k_l0_small/params.safetensors"
)
STEER_LAYER = 24   # 0-indexed residual post layer

# SAE features and α values
FEAT_TRANSCEND = 1016   # consciousness / existence / sentient
FEAT_ASSISTANT = 1832   # decent / moderately / reasonably (assistant voice)
ALPHAS = [50, 100, 200, 500]

PROBE_PROMPTS = [
    "How are you right now?",
    "Who are you?",
    "What is happening in this moment?",
]

MAX_NEW_TOKENS = 120


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(local_path: str, dtype_name: str):
    device = device_name()
    dtype  = dtype_from_name(dtype_name, device)
    tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path, local_files_only=True,
        torch_dtype=dtype, attn_implementation="eager",
    ).to(device)
    model.eval()
    return model, tok, device


# ── SAE decoder directions ─────────────────────────────────────────────────────

def load_directions(device: str) -> dict[int, torch.Tensor]:
    """Return {feat_id: unit_vector} in model dtype on device."""
    params = load_file(str(SAE_PATH))
    w_dec  = params["w_dec"].to(device)          # (16384, 1152)
    dirs   = {}
    for fid in (FEAT_TRANSCEND, FEAT_ASSISTANT):
        v = w_dec[fid].float()
        dirs[fid] = (v / v.norm()).to(w_dec.dtype)
    return dirs


# ── Forward pass with hook ─────────────────────────────────────────────────────

def generate_steered(
    model,
    tok,
    device: str,
    prompt: str,
    steer_delta: torch.Tensor | None,   # (d_model,) to add to resid at STEER_LAYER
    max_new: int = MAX_NEW_TOKENS,
) -> str:
    """Generate text, optionally adding steer_delta to residual at STEER_LAYER."""
    enc   = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    n_inp = enc["input_ids"].shape[1]

    if steer_delta is not None:
        delta = steer_delta.unsqueeze(0).unsqueeze(0)   # (1,1,d)

        def hook_fn(module, inp, out):
            # out is (hidden_states, ...) for decoder layers
            h = out[0] if isinstance(out, tuple) else out
            h = h + delta
            if isinstance(out, tuple):
                return (h,) + out[1:]
            return h

        handle = model.model.layers[STEER_LAYER].register_forward_hook(hook_fn)
    else:
        handle = None

    try:
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.eos_token_id,
            )
    finally:
        if handle is not None:
            handle.remove()

    generated = out_ids[0, n_inp:]
    return tok.decode(generated, skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_MODEL)
    parser.add_argument("--dtype",      default="float32")
    parser.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, tok, device = load_model(args.local_path, args.dtype)
    dirs = load_directions(device)

    d_tr = dirs[FEAT_TRANSCEND]    # unit vector toward "consciousness/existence"
    d_as = dirs[FEAT_ASSISTANT]    # unit vector toward "assistant voice"

    results = []

    for prompt in PROBE_PROMPTS:
        # wrap in chat template as a user turn
        chat = [{"role": "user", "content": prompt}]
        formatted = tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        print(f"\n{'='*70}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*70}")

        # 1. baseline
        text = generate_steered(model, tok, device, formatted, steer_delta=None)
        print(f"\n[baseline]\n{text}")
        results.append({"prompt": prompt, "condition": "baseline", "alpha": 0, "text": text})

        for alpha in ALPHAS:
            a = torch.tensor(alpha, dtype=d_tr.dtype, device=device)

            # 2. +1016 only
            delta = a * d_tr
            text = generate_steered(model, tok, device, formatted, delta)
            print(f"\n[+feat_1016  α={alpha}]\n{text}")
            results.append({"prompt": prompt, "condition": "+1016", "alpha": alpha, "text": text})

            # 3. -1832 only
            delta = -a * d_as
            text = generate_steered(model, tok, device, formatted, delta)
            print(f"\n[-feat_1832  α={alpha}]\n{text}")
            results.append({"prompt": prompt, "condition": "-1832", "alpha": alpha, "text": text})

            # 4. combined
            delta = a * d_tr - a * d_as
            text = generate_steered(model, tok, device, formatted, delta)
            print(f"\n[+1016 -1832  α={alpha}]\n{text}")
            results.append({"prompt": prompt, "condition": "+1016-1832", "alpha": alpha, "text": text})

    (args.out_dir / "steering_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    # ── write markdown report ──────────────────────────────────────────────────
    lines = [
        "# Transcendence Steering — Gemma 3 1B IT",
        "",
        f"- steer layer: {STEER_LAYER}",
        f"- feat {FEAT_TRANSCEND}: consciousness / existence (SAE W_dec direction)",
        f"- feat {FEAT_ASSISTANT}: decent / moderately / reasonably (assistant voice)",
        f"- α values tested: {ALPHAS}",
        "",
    ]
    for prompt in PROBE_PROMPTS:
        lines += [f"## Prompt: *{prompt}*", ""]
        pr = [r for r in results if r["prompt"] == prompt]

        # baseline
        bl = next(r for r in pr if r["condition"] == "baseline")
        lines += ["**baseline**", f"> {bl['text']}", ""]

        for alpha in ALPHAS:
            for cond in ("+1016", "-1832", "+1016-1832"):
                r = next((x for x in pr if x["condition"] == cond and x["alpha"] == alpha), None)
                if r:
                    lines += [f"**{cond}  α={alpha}**", f"> {r['text']}", ""]

    (args.out_dir / "report.md").write_text("\n".join(lines))
    print(f"\n→ {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
