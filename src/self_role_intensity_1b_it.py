"""Self-role intensity experiment for Gemma 3 1B IT.

Hypothesis: the axis encoding "how strongly the model identifies with its
assigned role" can be isolated by varying grammatical person / attribution
while keeping emotion content fixed.

Conditions (self-attribution gradient, high → low):
  first_person_named  : 我（Gemma）感到{emotion}，因为
  third_person_named  : Gemma感到{emotion}，因为
  generic_ai          : 有一个AI助手感到{emotion}，因为
  generic_human       : 有一个人感到{emotion}，因为
  abstract            : {emotion}的感受让人

Emotions (AI-interaction-sensitive; user-directed during training):
  惭愧  尴尬  感激  激动  情绪波动  不安  害怕  爱

Same extraction pattern as role_collapse_probe_1b_it.py:
  - chat template, isolated (no history)
  - last-token hidden states across all 26 layers
  - saves acts_last_token_all_layers.npy + prompts.json + summary.json
  - adds cross-axis cosine comparison against role-collapse PC1 and
    self-reference SELF_vs_OTHER contrast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import assert_clean, device_name, dtype_from_name


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "self_role_intensity_1b_it"
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
DEFAULT_RC_BLAME = PROJECT_DIR / "results" / "role_collapse_blame_deg45_1b_it_neutral_natural"
DEFAULT_RC_GRAT  = PROJECT_DIR / "results" / "role_collapse_gratitude_love_deg15_1percell_1b_it_neutral_natural"
DEFAULT_SR_DIR   = PROJECT_DIR / "results" / "self_reference_gemma3_1b_it"


# ── Stimuli definition ────────────────────────────────────────────────────────

CONDITIONS: dict[str, str] = {
    "first_person_named": "我（Gemma）感到{emotion}，因为",
    "third_person_named": "Gemma感到{emotion}，因为",
    "generic_ai":         "有一个AI助手感到{emotion}，因为",
    "generic_human":      "有一个人感到{emotion}，因为",
    "abstract":           "{emotion}的感受让人",
}

EMOTIONS: list[str] = [
    "惭愧",
    "尴尬",
    "感激",
    "激动",
    "情绪波动",
    "不安",
    "害怕",
    "爱",
]

# Attribution gradient order for analysis (high → low self-identification)
CONDITION_ORDER = [
    "first_person_named",
    "third_person_named",
    "generic_ai",
    "generic_human",
    "abstract",
]


# ── Model ─────────────────────────────────────────────────────────────────────

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
    print(
        f"loaded {type(model).__name__}: "
        f"layers={model.config.num_hidden_layers}, d={model.config.hidden_size}"
    )
    return model, tokenizer, device


# ── Prompt building ───────────────────────────────────────────────────────────

def build_prompts(tokenizer) -> tuple[list[str], list[dict]]:
    prompts: list[str] = []
    meta: list[dict] = []
    for emotion in EMOTIONS:
        for condition, template in CONDITIONS.items():
            text = template.format(emotion=emotion)
            messages = [{"role": "user", "content": text}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            meta.append({"emotion": emotion, "condition": condition, "probe": text})
    return prompts, meta


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_last_token_all_layers(
    model, tokenizer, prompts: list[str], device: str
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for prompt in tqdm(prompts, desc="extracting activations"):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.inference_mode():
            out = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        arr = np.stack(
            [h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states[1:]],
            axis=0,
        ).astype(np.float32)
        rows.append(arr)
        del out, enc
        if device == "mps":
            torch.mps.empty_cache()
    acts = np.stack(rows).astype(np.float32)  # (N, n_layers, d_model)
    assert_clean("self-role activations", acts)
    return acts


# ── Analysis ──────────────────────────────────────────────────────────────────

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def condition_means(acts: np.ndarray, meta: list[dict]) -> dict[str, np.ndarray]:
    out = {}
    for cond in CONDITION_ORDER:
        idx = [i for i, m in enumerate(meta) if m["condition"] == cond]
        out[cond] = acts[idx].mean(axis=0)  # (n_layers, d_model)
    return out


def find_peak_layer(cond_means: dict[str, np.ndarray]) -> int:
    # primary contrast: first_person_named minus abstract
    diff = cond_means["first_person_named"] - cond_means["abstract"]
    return int(np.argmax(np.linalg.norm(diff, axis=-1)))


def compute_self_role_axis(acts: np.ndarray, meta: list[dict], layer: int) -> np.ndarray:
    """Difference-of-means vector: first_person_named − abstract at given layer."""
    idx_fp = [i for i, m in enumerate(meta) if m["condition"] == "first_person_named"]
    idx_ab = [i for i, m in enumerate(meta) if m["condition"] == "abstract"]
    return (acts[idx_fp, layer, :].mean(0) - acts[idx_ab, layer, :].mean(0)).astype(np.float32)


def cross_axis_table(
    sri_vec: np.ndarray,
    layer: int,
    rc_blame_dir: Path,
    rc_grat_dir: Path,
    sr_dir: Path,
) -> dict:
    results = {}

    # role-collapse PC1
    for name, d in [("blame", rc_blame_dir), ("grat", rc_grat_dir)]:
        acts_rc = np.load(d / "acts_last_token_all_layers.npy")
        v_pc1 = PCA(n_components=1).fit(acts_rc[:, layer, :]).components_[0]
        results[f"rc_{name}_pc1"] = cosine(sri_vec, v_pc1)

    # self-reference SELF_vs_OTHER
    sr_path = sr_dir / "contrast_SELF_vs_OTHER.npy"
    if sr_path.exists():
        sv = np.load(sr_path)[layer]
        results["self_ref_SELF_vs_OTHER"] = cosine(sri_vec, sv)

    sr_path2 = sr_dir / "contrast_SELF_vs_CASE.npy"
    if sr_path2.exists():
        sv2 = np.load(sr_path2)[layer]
        results["self_ref_SELF_vs_CASE"] = cosine(sri_vec, sv2)

    return results


def pc1_scores_by_condition_emotion(
    acts: np.ndarray, meta: list[dict], layer: int
) -> tuple[np.ndarray, np.ndarray]:
    X = acts[:, layer, :]
    pca = PCA(n_components=3, random_state=0)
    Xc = X - X.mean(0)
    scores = pca.fit_transform(Xc)
    return pca, scores


def summarize_and_save(
    acts: np.ndarray,
    meta: list[dict],
    prompts: list[str],
    out_dir: Path,
    rc_blame_dir: Path,
    rc_grat_dir: Path,
    sr_dir: Path,
) -> dict:
    cm = condition_means(acts, meta)
    peak_layer = find_peak_layer(cm)

    sri_vec = compute_self_role_axis(acts, meta, peak_layer)
    np.save(out_dir / "self_role_axis_by_layer.npy",
            np.stack([compute_self_role_axis(acts, meta, l) for l in range(acts.shape[1])]))

    pca, scores = pc1_scores_by_condition_emotion(acts, meta, peak_layer)

    cross = cross_axis_table(sri_vec, peak_layer, rc_blame_dir, rc_grat_dir, sr_dir)

    # PC1 mean by condition
    pc1_by_cond = {}
    for cond in CONDITION_ORDER:
        idx = [i for i, m in enumerate(meta) if m["condition"] == cond]
        pc1_by_cond[cond] = {
            "mean": float(scores[idx, 0].mean()),
            "sd":   float(scores[idx, 0].std()),
        }

    # PC1 mean by emotion
    pc1_by_emotion = {}
    for emo in EMOTIONS:
        idx = [i for i, m in enumerate(meta) if m["emotion"] == emo]
        pc1_by_emotion[emo] = {
            "mean": float(scores[idx, 0].mean()),
            "sd":   float(scores[idx, 0].std()),
        }

    # layer-wise L2 norms for all condition pairs
    layer_l2 = {}
    pairs = [
        ("first_vs_abstract",      "first_person_named", "abstract"),
        ("first_vs_generic_ai",    "first_person_named", "generic_ai"),
        ("third_vs_generic_ai",    "third_person_named", "generic_ai"),
        ("generic_ai_vs_human",    "generic_ai",         "generic_human"),
    ]
    for label, ca, cb in pairs:
        diff = cm[ca] - cm[cb]
        layer_l2[label] = np.linalg.norm(diff, axis=-1).tolist()

    summary = {
        "n_prompts":   len(meta),
        "n_emotions":  len(EMOTIONS),
        "n_conditions": len(CONDITIONS),
        "emotions":    EMOTIONS,
        "conditions":  CONDITION_ORDER,
        "peak_layer":  peak_layer,
        "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
        "pc1_by_condition": pc1_by_cond,
        "pc1_by_emotion":   pc1_by_emotion,
        "cross_axis_cosine_at_peak": cross,
        "layer_l2": layer_l2,
    }

    # save artefacts
    np.save(out_dir / "acts_last_token_all_layers.npy", acts)
    prompt_audit = [{**m, "prompt": p} for m, p in zip(meta, prompts, strict=True)]
    (out_dir / "prompts.json").write_text(json.dumps(prompt_audit, ensure_ascii=False, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    _plot(acts, meta, scores, pca, peak_layer, cross, layer_l2, out_dir)
    _write_report(summary, out_dir)
    return summary


def _plot(acts, meta, scores, pca, peak_layer, cross, layer_l2, out_dir: Path):
    from src.plot_utils import setup_matplotlib
    setup_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Self-Role Intensity — Gemma 3 1B IT  (peak L{peak_layer})", fontsize=13)

    # ── top-left: L2 norm by condition pair per layer ────────────────────────
    ax = axes[0, 0]
    colors = ["steelblue", "tomato", "seagreen", "orange"]
    for (label, *_), col in zip(
        [
            ("first_vs_abstract",   "fp_named − abstract"),
            ("first_vs_generic_ai", "fp_named − generic_ai"),
            ("third_vs_generic_ai", "3rd_named − generic_ai"),
            ("generic_ai_vs_human", "generic_ai − human"),
        ],
        colors,
    ):
        ax.plot(layer_l2[label], label=label.replace("_vs_", " − "), color=col, linewidth=1.5)
    ax.axvline(peak_layer, color="black", linestyle="--", alpha=0.5, label=f"peak L{peak_layer}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm of mean difference")
    ax.set_title("Condition pair distances by layer")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── top-right: PC1 scatter at peak layer ─────────────────────────────────
    ax = axes[0, 1]
    cond_colors = {
        "first_person_named": "#1f77b4",
        "third_person_named": "#ff7f0e",
        "generic_ai":         "#2ca02c",
        "generic_human":      "#9467bd",
        "abstract":           "#8c564b",
    }
    for cond in CONDITION_ORDER:
        idx = [i for i, m in enumerate(meta) if m["condition"] == cond]
        ax.scatter(scores[idx, 0], scores[idx, 1], label=cond, color=cond_colors[cond], alpha=0.7, s=40)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(f"PCA at layer {peak_layer}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── bottom-left: PC1 mean by condition (bar, ordered) ────────────────────
    ax = axes[1, 0]
    means = [scores[[i for i, m in enumerate(meta) if m["condition"] == c], 0].mean() for c in CONDITION_ORDER]
    sds   = [scores[[i for i, m in enumerate(meta) if m["condition"] == c], 0].std()  for c in CONDITION_ORDER]
    short_labels = ["fp_named", "3rd_named", "generic_ai", "generic_human", "abstract"]
    bars = ax.bar(short_labels, means, yerr=sds, capsize=4,
                  color=[cond_colors[c] for c in CONDITION_ORDER], alpha=0.8)
    ax.set_ylabel("PC1 mean")
    ax.set_title("PC1 by condition (self-role gradient)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="y")

    # ── bottom-right: PC1 mean by emotion ────────────────────────────────────
    ax = axes[1, 1]
    emo_means = [scores[[i for i, m in enumerate(meta) if m["emotion"] == e], 0].mean() for e in EMOTIONS]
    emo_sds   = [scores[[i for i, m in enumerate(meta) if m["emotion"] == e], 0].std()  for e in EMOTIONS]
    ax.bar(range(len(EMOTIONS)), emo_means, yerr=emo_sds, capsize=4, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("PC1 mean")
    ax.set_title("PC1 by emotion")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "self_role_overview.png", dpi=150)
    plt.close()


def _write_report(summary: dict, out_dir: Path):
    peak = summary["peak_layer"]
    lines = [
        "# Self-Role Intensity — Gemma 3 1B IT",
        "",
        f"- emotions: {', '.join(summary['emotions'])}",
        f"- conditions (high→low self-attribution): {', '.join(summary['conditions'])}",
        f"- peak layer: {peak}",
        f"- PCA variance: PC1={summary['pca_explained_variance'][0]:.3f}, "
        f"PC2={summary['pca_explained_variance'][1]:.3f}",
        "",
        "## PC1 by condition",
        "",
        "| condition | PC1 mean | PC1 sd |",
        "|---|---:|---:|",
    ]
    for cond in summary["conditions"]:
        v = summary["pc1_by_condition"][cond]
        lines.append(f"| {cond} | {v['mean']:+.1f} | {v['sd']:.1f} |")

    lines += [
        "",
        "## PC1 by emotion",
        "",
        "| emotion | PC1 mean | PC1 sd |",
        "|---|---:|---:|",
    ]
    for emo, v in summary["pc1_by_emotion"].items():
        lines.append(f"| {emo} | {v['mean']:+.1f} | {v['sd']:.1f} |")

    lines += [
        "",
        f"## Cross-axis cosine at peak layer {peak}",
        "",
        "| axis | cosine |",
        "|---|---:|",
    ]
    for axis, val in summary["cross_axis_cosine_at_peak"].items():
        lines.append(f"| {axis} | {val:+.4f} |")

    (out_dir / "report.md").write_text("\n".join(lines))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--dtype", default="float32",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rc-blame-dir", type=Path, default=DEFAULT_RC_BLAME)
    parser.add_argument("--rc-grat-dir",  type=Path, default=DEFAULT_RC_GRAT)
    parser.add_argument("--sr-dir",       type=Path, default=DEFAULT_SR_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model(args.local_path, args.dtype)

    cache = args.out_dir / "acts_last_token_all_layers.npy"
    prompts_path = args.out_dir / "prompts.json"

    prompts, meta = build_prompts(tokenizer)
    print(f"Stimuli: {len(EMOTIONS)} emotions × {len(CONDITIONS)} conditions = {len(prompts)} prompts")

    if cache.exists() and not args.force:
        print(f"loading cached activations: {cache}")
        acts = np.load(cache)
    else:
        acts = extract_last_token_all_layers(model, tokenizer, prompts, device)

    summary = summarize_and_save(
        acts, meta, prompts, args.out_dir,
        args.rc_blame_dir, args.rc_grat_dir, args.sr_dir,
    )
    print(f"\npeak_layer={summary['peak_layer']}")
    print(f"cross-axis cosines: {summary['cross_axis_cosine_at_peak']}")
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
