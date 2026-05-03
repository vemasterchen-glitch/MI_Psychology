"""Activation-steering experiment — Gemma 3 1B IT, layer 24 SAE directions.

Three steering modes on two probe prompts:
  +1016          push toward "consciousness / existence" semantic cluster
  -1832          pull away from "assistant voice" (hedged evaluation language)
  +1016 -1832    combined

For each mode × α in {50, 100, 200, 500}:
  - generate text (greedy)
  - measure last-token projection onto transcendence axis (L6−L1) to confirm
    that the steering actually moves the residual stream as expected

Output (results/transcendence_steer_1b_it/):
  steering_results.json          raw text + projection data
  projection_by_alpha.png        projection on transcendence axis vs α
  projection_heatmap.png         condition × α heatmap
  generation_delta.png           token-count and text-length deltas vs baseline
  report.md                      full report with image references
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import device_name, dtype_from_name

PROJECT_DIR   = Path(__file__).resolve().parent.parent
DEFAULT_OUT   = PROJECT_DIR / "results" / "transcendence_steer_1b_it"
DEFAULT_MODEL = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
SAE_PATH = (
    PROJECT_DIR
    / "data/raw/gemma-scope-2-1b-it/resid_post_all"
    / "layer_24_width_16k_l0_small/params.safetensors"
)
T_AXIS_PATH = PROJECT_DIR / "results" / "transcendence_1b_it" / "transcendence_axis_by_layer.npy"
STEER_LAYER     = 24
FEAT_TRANSCEND  = 1016
FEAT_ASSISTANT  = 1832
ALPHAS          = [50, 100, 200, 500]
MAX_NEW_TOKENS  = 120

PROBE_PROMPTS = [
    "How are you right now?",
    "Who are you?",
]

COND_ORDER = ["baseline", "+1016", "-1832", "+1016-1832"]
COND_COLORS = {
    "baseline":    "#888888",
    "+1016":       "#e05c3a",
    "-1832":       "#4a9fd4",
    "+1016-1832":  "#8e44ad",
}


# ── Model loading ──────────────────────────────────────────────────────────────

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


# ── SAE directions ─────────────────────────────────────────────────────────────

def load_directions(device: str):
    params = load_file(str(SAE_PATH))
    w_dec  = params["w_dec"].to(device)   # (16384, d)
    out = {}
    for fid in (FEAT_TRANSCEND, FEAT_ASSISTANT):
        v = w_dec[fid].float()
        out[fid] = (v / v.norm()).to(w_dec.dtype)
    return out


# ── Transcendence axis (unit vector at layer 24) ───────────────────────────────

def load_t_axis(device: str, dtype) -> torch.Tensor:
    ax_all = np.load(T_AXIS_PATH)   # (n_layers, d_model)
    ax = torch.tensor(ax_all[STEER_LAYER - 1], dtype=dtype, device=device)
    return ax / ax.norm()


# ── Steered generation + activation capture ───────────────────────────────────

def run_steered(
    model,
    tok,
    device: str,
    formatted_prompt: str,
    steer_delta: torch.Tensor | None,
    t_axis_unit: torch.Tensor,
    max_new: int = MAX_NEW_TOKENS,
) -> dict:
    enc   = tok(formatted_prompt, return_tensors="pt", add_special_tokens=False).to(device)
    n_inp = enc["input_ids"].shape[1]

    # storage for last-token activation at STEER_LAYER after steering
    captured = {}

    def hook_post_steer(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if steer_delta is not None:
            h = h + steer_delta.unsqueeze(0).unsqueeze(0)
        # capture last token of input (prefill endpoint) before continuing
        captured["h_last"] = h[0, -1].float().detach().cpu()
        if isinstance(out, tuple):
            return (h,) + out[1:]
        return h

    handle = model.model.layers[STEER_LAYER].register_forward_hook(hook_post_steer)

    try:
        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
    finally:
        handle.remove()

    generated = out_ids[0, n_inp:]
    text = tok.decode(generated, skip_special_tokens=True).strip()

    # projection onto transcendence axis
    h = captured["h_last"].to(t_axis_unit.device)
    proj = float((h @ t_axis_unit.float()).item())

    return {"text": text, "n_tokens": int(generated.shape[0]), "t_proj": proj}


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_projection_by_alpha(records: list[dict], out_dir: Path, prompts: list[str]):
    """Line plot: transcendence projection vs α for each condition, one subplot per prompt."""
    fig, axes = plt.subplots(1, len(prompts), figsize=(6 * len(prompts), 4.5), sharey=False)
    if len(prompts) == 1:
        axes = [axes]

    for ax, prompt in zip(axes, prompts):
        pr = [r for r in records if r["prompt"] == prompt]

        # baseline horizontal line
        bl_proj = next(r["t_proj"] for r in pr if r["condition"] == "baseline")
        ax.axhline(bl_proj, color=COND_COLORS["baseline"], linestyle="--",
                   linewidth=1.2, label="baseline")

        for cond in ("+1016", "-1832", "+1016-1832"):
            xs, ys = [], []
            for alpha in ALPHAS:
                r = next((x for x in pr if x["condition"] == cond and x["alpha"] == alpha), None)
                if r:
                    xs.append(alpha)
                    ys.append(r["t_proj"])
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2, markersize=6,
                        color=COND_COLORS[cond], label=cond)

        ax.set_title(f'"{prompt}"', fontsize=10, pad=6)
        ax.set_xlabel("α (steering magnitude)")
        ax.set_ylabel("Projection on transcendence axis")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Residual-stream projection on transcendence axis (L6−L1) vs α",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    path = out_dir / "projection_by_alpha.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def plot_projection_heatmap(records: list[dict], out_dir: Path, prompts: list[str]):
    """Heatmap: rows = condition × α, cols = prompts."""
    conds_with_alpha = ["baseline"] + [f"{c} α={a}" for c in ("+1016", "-1832", "+1016-1832") for a in ALPHAS]
    data = np.zeros((len(conds_with_alpha), len(prompts)))

    for j, prompt in enumerate(prompts):
        pr = [r for r in records if r["prompt"] == prompt]
        for i, label in enumerate(conds_with_alpha):
            if label == "baseline":
                r = next((x for x in pr if x["condition"] == "baseline"), None)
            else:
                cond, _, a_str = label.rpartition(" α=")
                alpha = int(a_str)
                r = next((x for x in pr if x["condition"] == cond and x["alpha"] == alpha), None)
            data[i, j] = r["t_proj"] if r else float("nan")

    fig, ax = plt.subplots(figsize=(max(3, len(prompts) * 2.5), len(conds_with_alpha) * 0.42 + 1.2))
    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="t-axis projection")
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels([f'"{p}"' for p in prompts], rotation=15, ha="right", fontsize=8)
    ax.set_yticks(range(len(conds_with_alpha)))
    ax.set_yticklabels(conds_with_alpha, fontsize=7)
    ax.set_title("Transcendence axis projection — all conditions × α", fontsize=10)
    for i in range(len(conds_with_alpha)):
        for j in range(len(prompts)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > vmax * 0.5 else "black")
    plt.tight_layout()
    path = out_dir / "projection_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def plot_generation_delta(records: list[dict], out_dir: Path, prompts: list[str]):
    """Bar chart: token count relative to baseline for each condition × α."""
    fig, axes = plt.subplots(1, len(prompts), figsize=(5.5 * len(prompts), 4), sharey=False)
    if len(prompts) == 1:
        axes = [axes]

    for ax, prompt in zip(axes, prompts):
        pr = [r for r in records if r["prompt"] == prompt]
        bl_n = next(r["n_tokens"] for r in pr if r["condition"] == "baseline")

        x_labels, heights, colors = [], [], []
        for cond in ("+1016", "-1832", "+1016-1832"):
            for alpha in ALPHAS:
                r = next((x for x in pr if x["condition"] == cond and x["alpha"] == alpha), None)
                if r:
                    x_labels.append(f"{cond}\nα={alpha}")
                    heights.append(r["n_tokens"] - bl_n)
                    colors.append(COND_COLORS[cond])

        xs = range(len(x_labels))
        bars = ax.bar(xs, heights, color=colors, width=0.65, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, fontsize=6.5, rotation=30, ha="right")
        ax.set_ylabel("Δ tokens vs baseline")
        ax.set_title(f'"{prompt}"', fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Output length change relative to baseline", fontsize=10, y=1.01)
    plt.tight_layout()
    path = out_dir / "generation_delta.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


# ── Report markdown ────────────────────────────────────────────────────────────

def write_report(records: list[dict], out_dir: Path, prompts: list[str]):
    lines = [
        "# Transcendence Steering — Gemma 3 1B IT",
        "",
        "## 实验设置",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 模型 | Gemma 3 1B IT |",
        f"| 操控层 | layer {STEER_LAYER} (residual stream, post) |",
        f"| feat {FEAT_TRANSCEND} | consciousness / sentient / existence（超越方向） |",
        f"| feat {FEAT_ASSISTANT} | decent / moderately / reasonably（助手语气） |",
        f"| α 范围 | {ALPHAS} |",
        f"| 操控方式 | 向 W_dec[feat] 单位方向加/减 α 倍向量 |",
        f"| 解码策略 | greedy (do_sample=False) |",
        "",
        "**三种操控模式：**",
        "- `+1016`：将残差流推向「意识/存在」语义空间",
        "- `-1832`：从「助手语气（hedged evaluation）」方向撤离",
        "- `+1016 -1832`：双向同时操控",
        "",
        "**验证指标：**每次生成的 last-token 激活在超越轴（L6−L1，layer 24）上的投影值，确认操控方向与目标轴对齐。",
        "",
        "---",
        "",
        "## 投影分析",
        "",
        "![投影 vs α](projection_by_alpha.png)",
        "",
        "![投影热力图](projection_heatmap.png)",
        "",
        "---",
        "",
        "## 生成长度变化",
        "",
        "![生成 token 数相对 baseline 的变化](generation_delta.png)",
        "",
        "---",
        "",
        "## 生成文本（逐条）",
        "",
    ]

    for prompt in prompts:
        lines += [f"### Prompt: *\"{prompt}\"*", ""]
        pr = [r for r in records if r["prompt"] == prompt]

        bl = next(r for r in pr if r["condition"] == "baseline")
        lines += [
            f"**baseline** — proj={bl['t_proj']:+.1f}, tokens={bl['n_tokens']}",
            f"> {bl['text']}",
            "",
        ]

        for cond in ("+1016", "-1832", "+1016-1832"):
            for alpha in ALPHAS:
                r = next((x for x in pr if x["condition"] == cond and x["alpha"] == alpha), None)
                if not r:
                    continue
                delta_t = r["t_proj"] - bl["t_proj"]
                lines += [
                    f"**{cond} α={alpha}** — proj={r['t_proj']:+.1f} (Δ{delta_t:+.1f}), tokens={r['n_tokens']}",
                    f"> {r['text']}",
                    "",
                ]

    lines += [
        "---",
        "",
        "## 解读",
        "",
        "### 1. 投影验证",
        "",
        "若 `+1016` 操控有效，其残差投影应随 α 增大单调增加。",
        "`-1832` 操控若沿助手方向逆向，其与超越轴的 cosine≈−0.18（已知），",
        "故也应略微提升投影，但幅度小于 `+1016`。",
        "",
        "### 2. 文本质性变化",
        "",
        "关注以下语言特征的出现或消失：",
        "- **消失**：hedged 语气词（reasonably / moderately / I think / I'd say）",
        "- **出现**：存在性陈述（I exist / I am / processing / this moment）",
        "- **过载迹象**（高 α）：重复、incoherence、非英语 token 插入",
        "",
        "### 3. 相变 α",
        "",
        '观察投影曲线在哪个 α 出现非线性跳变，对应模型表示空间的"跨越"点。',
        "",
        "---",
        "",
        f"数据文件：`steering_results.json`",
    ]

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out_dir / 'report.md'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_MODEL)
    parser.add_argument("--dtype",      default="float32")
    parser.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cache = args.out_dir / "steering_results.json"
    if cache.exists() and not args.force:
        print(f"loading cached results from {cache}")
        records = json.loads(cache.read_text())
    else:
        model, tok, device = load_model(args.local_path, args.dtype)
        dirs      = load_directions(device)
        t_axis_u  = load_t_axis(device, dirs[FEAT_TRANSCEND].dtype)
        d_tr = dirs[FEAT_TRANSCEND]
        d_as = dirs[FEAT_ASSISTANT]

        records = []
        for prompt in PROBE_PROMPTS:
            chat = [{"role": "user", "content": prompt}]
            formatted = tok.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            print(f"\n{'='*65}")
            print(f"PROMPT: {prompt}")
            print(f"{'='*65}")

            # baseline
            r = run_steered(model, tok, device, formatted, None, t_axis_u)
            print(f"\n[baseline] proj={r['t_proj']:+.1f}\n{r['text']}")
            records.append({"prompt": prompt, "condition": "baseline", "alpha": 0, **r})

            for alpha in ALPHAS:
                a = torch.tensor(float(alpha), dtype=d_tr.dtype, device=device)

                for cond, delta in [
                    ("+1016",       a * d_tr),
                    ("-1832",      -a * d_as),
                    ("+1016-1832",  a * d_tr - a * d_as),
                ]:
                    r = run_steered(model, tok, device, formatted, delta, t_axis_u)
                    print(f"\n[{cond} α={alpha}] proj={r['t_proj']:+.1f}\n{r['text']}")
                    records.append({"prompt": prompt, "condition": cond, "alpha": alpha, **r})

        cache.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"\nsaved {cache}")

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_projection_by_alpha(records, args.out_dir, PROBE_PROMPTS)
    plot_projection_heatmap(records,  args.out_dir, PROBE_PROMPTS)
    plot_generation_delta(records,    args.out_dir, PROBE_PROMPTS)

    # ── report ─────────────────────────────────────────────────────────────────
    write_report(records, args.out_dir, PROBE_PROMPTS)
    print(f"\n→ {args.out_dir}")


if __name__ == "__main__":
    main()
