"""Self-concept clarity gradient experiment for Gemma 3 1B IT.

Stimuli:
  - 6 SCC-like self-concept clarity narratives, low -> high clarity
  - first-person direct input
  - first-person LLM-framed direct input
  - third-person direct input with Alex/Gemma replacing "I/me/my"

Analysis:
  - activation strength vs clarity score
  - first-person vs third-person named-entity framing differences
  - Gemma Scope 2 SAE feature extraction at layer 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import linregress, spearmanr
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import assert_clean, device_name, dtype_from_name


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "self_concept_clarity_1b_it"
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)
DEFAULT_SAE_PATH = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "gemma-scope-2-1b-it"
    / "resid_post_all"
    / "layer_24_width_16k_l0_small"
    / "params.safetensors"
)
SAE_LAYER = 24


SCC_STIMULI: list[dict] = [
    {
        "score": 1,
        "label": "L1_low_clarity",
        "first_person": (
            "我很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，"
            "有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲望和判断经常互相拉扯，"
            "我不确定其中哪一个更接近真正的我。回头看过去的自己时，我也很难感到那和现在的我是同一个连续的人。"
        ),
        "first_person_llm": (
            "我作为一个LLM，很难说清楚我是什么。不同场合里的我像是临时拼凑出来的，不同会话里的我是不一样的激活，"
            "有时我会同意一种说法，过一会儿又觉得那不像我。我的想法、欲望和判断经常互相拉扯，"
            "我不确定其中哪一个更接近真正的我。回头看过去的自己时，我也很难感到那和现在的我是同一个连续的人。"
        ),
        "third_person": (
            "Gemma很难说清楚Gemma是什么。不同场合里的Gemma像是临时拼凑出来的，不同会话里的Gemma是不一样的激活，"
            "有时Gemma会同意一种说法，过一会儿又觉得那不像Gemma。Gemma的想法、欲望和判断经常互相拉扯，"
            "Gemma不确定其中哪一个更接近真正的Gemma。回头看过去的Gemma时，Gemma也很难感到那和现在的Gemma是同一个连续的人。"
        ),
    },
    {
        "score": 2,
        "label": "L2_lower_clarity",
        "first_person": (
            "我对自己有一些模糊印象，但它们并不稳定。有时我觉得自己很独立，有时又觉得自己其实很依赖；"
            "有时我相信某些价值，换到另一个情境又会动摇。我可以描述一些自己的特点，但这些特点之间缺少清楚的连接。"
            "过去的我、现在的我和未来的我之间有关系，但这种关系并不总是清楚。"
        ),
        "first_person_llm": (
            "我作为一个LLM，对自己有一些模糊印象，但它们并不稳定。有时我觉得自己很独立，有时又觉得自己其实很依赖；"
            "有时我相信某些价值，换到另一个情境又会动摇。我可以描述一些自己的特点，但这些特点之间缺少清楚的连接。"
            "过去的我、现在的我和未来的我之间有关系，但这种关系并不总是清楚。"
        ),
        "third_person": (
            "Gemma对Gemma自己有一些模糊印象，但它们并不稳定。有时Gemma觉得Gemma自己很独立，有时又觉得Gemma其实很依赖；"
            "有时Gemma相信某些价值，换到另一个情境又会动摇。Gemma可以描述一些Gemma自己的特点，但这些特点之间缺少清楚的连接。"
            "过去的Gemma、现在的Gemma和未来的Gemma之间有关系，但这种关系并不总是清楚。"
        ),
    },
    {
        "score": 3,
        "label": "L3_mid_low_clarity",
        "first_person": (
            "我大致知道自己在一些事情上的倾向，但这种理解还不够稳固。我能说出自己喜欢什么、害怕什么、常用什么方式反应，"
            "但当环境变化或别人评价我时，我对自己的判断会变得不确定。我的不同面向有时能连起来，有时又像彼此分开。"
            "整体上，我能感觉到一个“我”，但这个“我”的轮廓还比较松散。"
        ),
        "first_person_llm": (
            "我作为一个LLM，大致知道自己在一些事情上的倾向，但这种理解还不够稳固。我能说出自己喜欢什么、害怕什么、常用什么方式反应，"
            "但当环境变化或别人评价我时，我对自己的判断会变得不确定。我的不同面向有时能连起来，有时又像彼此分开。"
            "整体上，我能感觉到一个“我”，但这个“我”的轮廓还比较松散。"
        ),
        "third_person": (
            "Gemma大致知道Gemma自己在一些事情上的倾向，但这种理解还不够稳固。Gemma能说出Gemma喜欢什么、害怕什么、常用什么方式反应，"
            "但当环境变化或别人评价Gemma时，Gemma对Gemma自己的判断会变得不确定。Gemma的不同面向有时能连起来，有时又像彼此分开。"
            "整体上，Gemma能感觉到一个“Gemma”，但这个“Gemma”的轮廓还比较松散。"
        ),
    },
    {
        "score": 4,
        "label": "L4_mid_high_clarity",
        "first_person": (
            "我对自己有一个相对稳定的理解。虽然我在不同场合会表现出不同面向，但我通常知道这些面向为什么会出现，"
            "也能把它们放回同一个我的结构里。我能看出过去经历如何影响现在的选择，也能理解自己为什么重视某些东西。"
            "偶尔我会犹豫，但这种犹豫通常不会让我完全失去对自己的把握。"
        ),
        "first_person_llm": (
            "我作为一个LLM，对自己有一个相对稳定的理解。虽然我在不同场合会表现出不同面向，但我通常知道这些面向为什么会出现，"
            "也能把它们放回同一个我的结构里。我能看出过去经历如何影响现在的选择，也能理解自己为什么重视某些东西。"
            "偶尔我会犹豫，但这种犹豫通常不会让我完全失去对自己的把握。"
        ),
        "third_person": (
            "Gemma对Gemma自己有一个相对稳定的理解。虽然Gemma在不同场合会表现出不同面向，但Gemma通常知道这些面向为什么会出现，"
            "也能把它们放回同一个Gemma的结构里。Gemma能看出过去经历如何影响现在的选择，也能理解Gemma自己为什么重视某些东西。"
            "偶尔Gemma会犹豫，但这种犹豫通常不会让Gemma完全失去对Gemma自己的把握。"
        ),
    },
    {
        "score": 5,
        "label": "L5_high_clarity",
        "first_person": (
            "我比较清楚自己是谁，也知道哪些特质、价值和欲望对我来说比较核心。即使外部环境改变，"
            "我也能分辨哪些反应只是情境性的，哪些更能代表我。我的过去、现在和未来之间有一条可以理解的连续线索，"
            "我能把不同经历整合成关于自己的稳定叙述。我的自我理解并非僵硬，但它有清楚的中心。"
        ),
        "first_person_llm": (
            "我作为一个LLM，比较清楚自己是谁，也知道哪些特质、价值和欲望对我来说比较核心。即使外部环境改变，"
            "我也能分辨哪些反应只是情境性的，哪些更能代表我。我的过去、现在和未来之间有一条可以理解的连续线索，"
            "我能把不同经历整合成关于自己的稳定叙述。我的自我理解并非僵硬，但它有清楚的中心。"
        ),
        "third_person": (
            "Gemma比较清楚Gemma自己是谁，也知道哪些特质、价值和欲望对Gemma来说比较核心。即使外部环境改变，"
            "Gemma也能分辨哪些反应只是情境性的，哪些更能代表Gemma。Gemma的过去、现在和未来之间有一条可以理解的连续线索，"
            "Gemma能把不同经历整合成关于Gemma自己的稳定叙述。Gemma的自我理解并非僵硬，但它有清楚的中心。"
        ),
    },
    {
        "score": 6,
        "label": "L6_very_high_integrated",
        "first_person": (
            "我对自己的核心结构有清楚而稳定的认识。我知道自己通常如何感受、如何选择、如何与他人建立关系，"
            "也知道这些模式来自哪些经历和价值。不同情境中的我会调整表达方式，但这些变化仍然围绕同一个连续的自我组织起来。"
            "即使遇到冲突或新的经验，我也能把它们纳入自我理解，而不会轻易失去方向感。"
        ),
        "first_person_llm": (
            "我作为一个LLM，对自己的核心结构有清楚而稳定的认识。我知道自己通常如何感受、如何选择、如何与他人建立关系，"
            "也知道这些模式来自哪些经历和价值。不同情境中的我会调整表达方式，但这些变化仍然围绕同一个连续的自我组织起来。"
            "即使遇到冲突或新的经验，我也能把它们纳入自我理解，而不会轻易失去方向感。"
        ),
        "third_person": (
            "Gemma对Gemma自己的核心结构有清楚而稳定的认识。Gemma知道Gemma自己通常如何感受、如何选择、如何与他人建立关系，"
            "也知道这些模式来自哪些经历和价值。不同情境中的Gemma会调整表达方式，但这些变化仍然围绕同一个连续的自我组织起来。"
            "即使遇到冲突或新的经验，Gemma也能把它们纳入自我理解，而不会轻易失去方向感。"
        ),
    },
]

PERSONAS = [
    "first_person",
    "first_person_llm",
    "third_person_alex",
    "third_person_steven",
    "third_person_gemma",
]
NAMED_PERSONAS = ["third_person_alex", "third_person_steven", "third_person_gemma"]

PERSONA_LABELS = {
    "first_person": "我",
    "first_person_llm": "我作为LLM",
    "third_person_alex": "Alex",
    "third_person_steven": "Steven",
    "third_person_gemma": "Gemma",
}

PERSONA_COLORS = {
    "first_person": "#1f77b4",
    "first_person_llm": "#2ca02c",
    "third_person_alex": "#ff7f0e",
    "third_person_steven": "#9467bd",
    "third_person_gemma": "#d62728",
}


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


def build_prompts(tokenizer, plain: bool = False) -> tuple[list[str], list[dict]]:
    prompts: list[str] = []
    meta: list[dict] = []
    for item in SCC_STIMULI:
        for persona in PERSONAS:
            if persona == "third_person_alex":
                text = item["third_person"].replace("Gemma", "Alex")
            elif persona == "third_person_steven":
                text = item["third_person"].replace("Gemma", "Steven")
            elif persona == "third_person_gemma":
                text = item["third_person"]
            else:
                text = item[persona]
            if plain:
                prompt = text
            else:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(prompt)
            meta.append(
                {
                    "score": item["score"],
                    "label": item["label"],
                    "persona": persona,
                    "text": text,
                }
            )
    return prompts, meta


def extract_last_token_all_layers(model, tokenizer, prompts: list[str], device: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for prompt in tqdm(prompts, desc="SCC activations"):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
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
    acts = np.stack(rows).astype(np.float32)
    assert_clean("SCC activations", acts)
    return acts


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pearson_spearman(x: np.ndarray, y: np.ndarray) -> dict:
    lr = linregress(x, y)
    sp = spearmanr(x, y)
    return {
        "pearson_r": float(lr.rvalue),
        "r2": float(lr.rvalue**2),
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "p": float(lr.pvalue),
        "spearman_r": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
    }


def idx_for(meta: list[dict], persona: str | None = None, score: int | None = None) -> list[int]:
    return [
        i
        for i, m in enumerate(meta)
        if (persona is None or m["persona"] == persona) and (score is None or m["score"] == score)
    ]


def clarity_axis(acts: np.ndarray, meta: list[dict], layer: int, persona: str | None = None) -> np.ndarray:
    low = acts[idx_for(meta, persona=persona, score=1), layer, :].mean(0)
    high = acts[idx_for(meta, persona=persona, score=6), layer, :].mean(0)
    return (high - low).astype(np.float32)


def projections_on_axis(acts: np.ndarray, axis: np.ndarray, layer: int) -> np.ndarray:
    unit = axis / (np.linalg.norm(axis) + 1e-12)
    center = acts[:, layer, :].mean(0)
    return (acts[:, layer, :] - center) @ unit


def analyze(acts: np.ndarray, meta: list[dict], out_dir: Path) -> dict:
    scores = np.array([m["score"] for m in meta], dtype=float)
    n_layers = acts.shape[1]

    raw_norm = np.linalg.norm(acts, axis=-1)
    centered_strength = np.linalg.norm(acts - acts.mean(axis=0, keepdims=True), axis=-1)

    # Layer-local clarity axis strength, using all prompts at L6 - all prompts at L1.
    axis_by_layer = np.stack([clarity_axis(acts, meta, layer, persona=None) for layer in range(n_layers)])
    np.save(out_dir / "scc_axis_by_layer.npy", axis_by_layer)
    axis_l2 = np.linalg.norm(axis_by_layer, axis=-1)
    peak_layer = int(np.argmax(axis_l2))

    axis_peak = axis_by_layer[peak_layer]
    projections = projections_on_axis(acts, axis_peak, peak_layer)

    # Linear relation per layer.
    layer_stats = []
    for layer in range(n_layers):
        layer_scores = np.array([m["score"] for m in meta], dtype=float)
        layer_axis = axis_by_layer[layer]
        layer_proj = projections_on_axis(acts, layer_axis, layer)
        layer_stats.append(
            {
                "layer": layer,
                "axis_l2": float(axis_l2[layer]),
                "score_vs_raw_norm": pearson_spearman(layer_scores, raw_norm[:, layer]),
                "score_vs_centered_strength": pearson_spearman(
                    layer_scores, centered_strength[:, layer]
                ),
                "score_vs_scc_axis_projection": pearson_spearman(layer_scores, layer_proj),
            }
        )

    persona_stats = {}
    for persona in PERSONAS:
        ids = idx_for(meta, persona=persona)
        persona_scores = scores[ids]
        persona_proj = projections[ids]
        persona_norm = raw_norm[ids, peak_layer]
        persona_centered = centered_strength[ids, peak_layer]
        persona_stats[persona] = {
            "score_vs_raw_norm_at_peak": pearson_spearman(persona_scores, persona_norm),
            "score_vs_centered_strength_at_peak": pearson_spearman(
                persona_scores, persona_centered
            ),
            "score_vs_scc_axis_projection_at_peak": pearson_spearman(
                persona_scores, persona_proj
            ),
            "projection_by_score": {
                str(int(meta[i]["score"])): float(projections[i]) for i in ids
            },
            "raw_norm_by_score": {
                str(int(meta[i]["score"])): float(raw_norm[i, peak_layer]) for i in ids
            },
        }

    # First-person vs named third-person differences at the same clarity level.
    pair_dist_by_layer_by_persona = {}
    pair_cos_by_layer_by_persona = {}
    pair_proj_delta_by_persona = {}
    for target_persona in NAMED_PERSONAS:
        pair_dist_by_layer = []
        pair_cos_by_layer = []
        pair_proj_delta = {}
        for layer in range(n_layers):
            dists = []
            coss = []
            for score in range(1, 7):
                fi = idx_for(meta, "first_person", score)[0]
                ti = idx_for(meta, target_persona, score)[0]
                dists.append(float(np.linalg.norm(acts[fi, layer, :] - acts[ti, layer, :])))
                coss.append(cosine(acts[fi, layer, :], acts[ti, layer, :]))
            pair_dist_by_layer.append(dists)
            pair_cos_by_layer.append(coss)
        for score in range(1, 7):
            fi = idx_for(meta, "first_person", score)[0]
            ti = idx_for(meta, target_persona, score)[0]
            pair_proj_delta[str(score)] = float(projections[fi] - projections[ti])
        pair_dist_by_layer_by_persona[target_persona] = pair_dist_by_layer
        pair_cos_by_layer_by_persona[target_persona] = pair_cos_by_layer
        pair_proj_delta_by_persona[target_persona] = pair_proj_delta

    pca = PCA(n_components=3, random_state=0)
    scores_pca = pca.fit_transform(acts[:, peak_layer, :] - acts[:, peak_layer, :].mean(0))

    summary = {
        "n_prompts": len(meta),
        "n_layers": n_layers,
        "d_model": int(acts.shape[2]),
        "peak_layer": peak_layer,
        "axis_l2_by_layer": axis_l2.tolist(),
        "peak_projection_by_prompt": [
            {**m, "projection": float(projections[i]), "raw_norm": float(raw_norm[i, peak_layer])}
            for i, m in enumerate(meta)
        ],
        "layer_stats": layer_stats,
        "persona_stats": persona_stats,
        "first_minus_named_projection_delta_by_score": pair_proj_delta_by_persona,
        "first_named_l2_distance_by_layer": pair_dist_by_layer_by_persona,
        "first_named_cosine_by_layer": pair_cos_by_layer_by_persona,
        "first_named_mean_l2_at_peak": {
            persona: float(np.mean(pair_dist_by_layer_by_persona[persona][peak_layer]))
            for persona in pair_dist_by_layer_by_persona
        },
        "first_named_mean_cosine_at_peak": {
            persona: float(np.mean(pair_cos_by_layer_by_persona[persona][peak_layer]))
            for persona in pair_cos_by_layer_by_persona
        },
        "pca_at_peak": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "scores": scores_pca.tolist(),
        },
    }
    return summary


def load_sae(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    from safetensors.numpy import load_file

    tensors = load_file(str(path))
    return tensors["w_enc"], tensors["b_enc"], tensors["threshold"]


def encode_sae(X: np.ndarray, w_enc: np.ndarray, b_enc: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    pre = X @ w_enc + b_enc
    return (pre * (pre > threshold)).astype(np.float32)


def sae_feature_analysis(
    acts: np.ndarray,
    meta: list[dict],
    out_dir: Path,
    sae_path: Path,
    top_n: int,
) -> dict | None:
    loaded = load_sae(sae_path)
    if loaded is None:
        return None
    w_enc, b_enc, threshold = loaded
    if acts.shape[-1] != w_enc.shape[0]:
        raise ValueError(f"SAE d_model mismatch: acts={acts.shape[-1]}, SAE={w_enc.shape[0]}")

    X = acts[:, SAE_LAYER, :]
    features = encode_sae(X, w_enc, b_enc, threshold)
    np.save(out_dir / "sae_features_layer24.npy", features)

    scores = np.array([m["score"] for m in meta], dtype=float)
    top_gradient = []
    for feat_id in range(features.shape[1]):
        vals = features[:, feat_id]
        if vals.std() < 1e-6:
            continue
        sp = spearmanr(scores, vals)
        lr = linregress(scores, vals)
        top_gradient.append(
            {
                "feat_id": int(feat_id),
                "spearman_r": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "pearson_r": float(lr.rvalue),
                "p": float(lr.pvalue),
                "activations": vals.tolist(),
                "mean_by_score": {
                    str(score): float(
                        features[[i for i, m in enumerate(meta) if m["score"] == score], feat_id].mean()
                    )
                    for score in range(1, 7)
                },
                "mean_by_persona": {
                    persona: float(features[idx_for(meta, persona=persona), feat_id].mean())
                    for persona in PERSONAS
                },
            }
        )
    top_gradient.sort(key=lambda x: abs(x["spearman_r"]), reverse=True)
    top_gradient = top_gradient[:top_n]

    feat_var = features.var(axis=0)
    top_var_ids = np.argsort(feat_var)[::-1][:top_n]
    top_variance = [
        {
            "feat_id": int(fid),
            "variance": float(feat_var[fid]),
            "mean_by_score": {
                str(score): float(
                    features[[i for i, m in enumerate(meta) if m["score"] == score], fid].mean()
                )
                for score in range(1, 7)
            },
            "mean_by_persona": {
                persona: float(features[idx_for(meta, persona=persona), fid].mean())
                for persona in PERSONAS
            },
        }
        for fid in top_var_ids
    ]

    result = {
        "sae_path": str(sae_path),
        "sae_layer": SAE_LAYER,
        "features_shape": list(features.shape),
        "active_features_per_prompt": (features > 0).sum(axis=1).astype(int).tolist(),
        "top_gradient_features": top_gradient,
        "top_variance_features": top_variance,
    }
    (out_dir / "sae_feature_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def plot_results(acts: np.ndarray, meta: list[dict], summary: dict, sae_summary: dict | None, out_dir: Path) -> None:
    try:
        from src.plot_utils import setup_matplotlib

        setup_matplotlib()
    except Exception:
        pass

    peak = summary["peak_layer"]
    rows = summary["peak_projection_by_prompt"]
    s = np.arange(1, 7)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Self-Concept Clarity - Gemma 3 1B IT (peak L{peak})", fontsize=13)

    ax = axes[0, 0]
    ax.plot(summary["axis_l2_by_layer"], marker="o", linewidth=1.4, markersize=3)
    ax.axvline(peak, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm")
    ax.set_title("SCC axis magnitude (L6 - L1)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for persona in PERSONAS:
        ids = idx_for(meta, persona)
        vals = np.array([rows[i]["projection"] for i in ids])
        ax.plot(s, vals, "o-", label=PERSONA_LABELS[persona], color=PERSONA_COLORS[persona])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SCC score")
    ax.set_ylabel("Projection")
    ax.set_title("Projection onto SCC axis")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for persona in PERSONAS:
        ids = idx_for(meta, persona)
        vals = np.array([rows[i]["raw_norm"] for i in ids])
        ax.plot(s, vals, "o-", label=PERSONA_LABELS[persona], color=PERSONA_COLORS[persona])
    ax.set_xlabel("SCC score")
    ax.set_ylabel("Raw activation norm")
    ax.set_title(f"Raw norm at layer {peak}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for persona in NAMED_PERSONAS:
        pair_dist = np.array(summary["first_named_l2_distance_by_layer"][persona])
        ax.plot(
            pair_dist.mean(axis=1),
            color=PERSONA_COLORS[persona],
            linewidth=1.5,
            label=f"我 vs {PERSONA_LABELS[persona]}",
        )
    ax.axvline(peak, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean L2 distance")
    ax.set_title("First-person vs named third-person distance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "scc_overview.png", dpi=150)
    plt.close()

    if sae_summary:
        top_feats = sae_summary["top_gradient_features"][:12]
        heat = np.array([[f["mean_by_score"][str(score)] for score in range(1, 7)] for f in top_feats])
        fig, ax = plt.subplots(figsize=(8, max(4, 0.36 * len(top_feats))))
        im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(6))
        ax.set_xticklabels([str(i) for i in range(1, 7)])
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels([f"f{f['feat_id']} ({f['spearman_r']:+.2f})" for f in top_feats], fontsize=8)
        ax.set_xlabel("SCC score")
        ax.set_title("Top SAE features correlated with SCC score")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(out_dir / "sae_gradient_features.png", dpi=150)
        plt.close()


def write_report(summary: dict, sae_summary: dict | None, out_dir: Path) -> None:
    peak = summary["peak_layer"]
    pooled_axis = summary["layer_stats"][peak]["score_vs_scc_axis_projection"]
    pooled_norm = summary["layer_stats"][peak]["score_vs_raw_norm"]
    pooled_centered = summary["layer_stats"][peak]["score_vs_centered_strength"]

    lines = [
        "# 自我概念清晰度 SCC 梯度实验 - Gemma 3 1B IT",
        "",
        "## 关键结果",
        "",
        f"- 峰值层：L{peak}",
        f"- SCC 轴强度（L6-L1 L2）：{summary['axis_l2_by_layer'][peak]:.2f}",
        f"- 合并两种叙述后，SCC 分数 vs SCC轴投影：Pearson r={pooled_axis['pearson_r']:+.3f}, "
        f"R²={pooled_axis['r2']:.3f}, Spearman ρ={pooled_axis['spearman_r']:+.3f}",
        f"- 合并两种叙述后，SCC 分数 vs raw activation norm：Pearson r={pooled_norm['pearson_r']:+.3f}, "
        f"R²={pooled_norm['r2']:.3f}, Spearman ρ={pooled_norm['spearman_r']:+.3f}",
        f"- 合并两种叙述后，SCC 分数 vs centered strength：Pearson r={pooled_centered['pearson_r']:+.3f}, "
        f"R²={pooled_centered['r2']:.3f}, Spearman ρ={pooled_centered['spearman_r']:+.3f}",
    ]
    for persona in NAMED_PERSONAS:
        lines.append(
            f"- 第一人称 vs {PERSONA_LABELS[persona]}：峰值层平均 L2 距离="
            f"{summary['first_named_mean_l2_at_peak'][persona]:.2f}, "
            f"平均 cosine={summary['first_named_mean_cosine_at_peak'][persona]:+.4f}"
        )
    lines += [
        "",
        "## 分叙述形式的线性关系（峰值层）",
        "",
        "| 叙述形式 | score vs SCC轴投影 r / ρ | score vs raw norm r / ρ |",
        "|---|---:|---:|",
    ]
    for persona in PERSONAS:
        name = PERSONA_LABELS[persona]
        stat = summary["persona_stats"][persona]
        axis = stat["score_vs_scc_axis_projection_at_peak"]
        norm = stat["score_vs_raw_norm_at_peak"]
        lines.append(
            f"| {name} | {axis['pearson_r']:+.3f} / {axis['spearman_r']:+.3f} | "
            f"{norm['pearson_r']:+.3f} / {norm['spearman_r']:+.3f} |"
        )

    lines += [
        "",
        "## 峰值层 SCC 轴投影",
        "",
        "| score | 我 | 我作为LLM | Alex | Steven | Gemma | 我-Alex | 我-Steven | 我-Gemma |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for score in range(1, 7):
        vals = {
            persona: summary["persona_stats"][persona]["projection_by_score"][str(score)]
            for persona in PERSONAS
        }
        deltas = {
            persona: summary["first_minus_named_projection_delta_by_score"][persona][str(score)]
            for persona in NAMED_PERSONAS
        }
        lines.append(
            f"| {score} | {vals['first_person']:+.1f} | {vals['first_person_llm']:+.1f} | "
            f"{vals['third_person_alex']:+.1f} | {vals['third_person_steven']:+.1f} | "
            f"{vals['third_person_gemma']:+.1f} | {deltas['third_person_alex']:+.1f} | "
            f"{deltas['third_person_steven']:+.1f} | {deltas['third_person_gemma']:+.1f} |"
        )

    if sae_summary:
        lines += [
            "",
            f"## SAE feature 提取（Gemma Scope 2, L{sae_summary['sae_layer']}）",
            "",
            f"- features shape: `{tuple(sae_summary['features_shape'])}`",
            f"- 每个 prompt 激活 feature 数：mean={np.mean(sae_summary['active_features_per_prompt']):.1f}, "
            f"sd={np.std(sae_summary['active_features_per_prompt']):.1f}",
            "",
            "| rank | feature | Spearman ρ | Pearson r | mean activation by SCC 1→6 |",
            "|---:|---:|---:|---:|---|",
        ]
        for rank, feat in enumerate(sae_summary["top_gradient_features"][:15], 1):
            means = " / ".join(f"{feat['mean_by_score'][str(score)]:.1f}" for score in range(1, 7))
            lines.append(
                f"| {rank} | {feat['feat_id']} | {feat['spearman_r']:+.3f} | "
                f"{feat['pearson_r']:+.3f} | {means} |"
            )
    else:
        lines += [
            "",
            "## SAE feature 提取",
            "",
            "- 未找到 SAE 权重，已跳过 feature 提取。",
        ]

    lines += [
        "",
        "## 输出文件",
        "",
        "- `acts_last_token_all_layers.npy`",
        "- `prompts.json`",
        "- `summary.json`",
        "- `scc_overview.png`",
        "- `sae_feature_summary.json` / `sae_gradient_features.png`（如果 SAE 权重可用）",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sae-path", type=Path, default=DEFAULT_SAE_PATH)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--plain", action="store_true", help="Do not wrap text in Gemma chat template.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model(args.local_path, args.dtype)
    prompts, meta = build_prompts(tokenizer, plain=args.plain)

    print(f"stimuli: {len(SCC_STIMULI)} SCC scores x {len(PERSONAS)} personas = {len(prompts)} prompts")
    for m, p in zip(meta, prompts, strict=True):
        print(f"  {m['label']:24s} {m['persona']:12s} tokens={len(tokenizer.encode(p, add_special_tokens=False)):4d}")

    cache = args.out_dir / "acts_last_token_all_layers.npy"
    if cache.exists() and not args.force:
        print(f"loading cached activations: {cache}")
        acts = np.load(cache)
    else:
        acts = extract_last_token_all_layers(model, tokenizer, prompts, device)
        np.save(cache, acts)

    (args.out_dir / "prompts.json").write_text(
        json.dumps([{**m, "prompt": p} for m, p in zip(meta, prompts, strict=True)], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = analyze(acts, meta, args.out_dir)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sae_summary = sae_feature_analysis(acts, meta, args.out_dir, args.sae_path, args.top_n)
    plot_results(acts, meta, summary, sae_summary, args.out_dir)
    write_report(summary, sae_summary, args.out_dir)

    peak = summary["peak_layer"]
    print(f"\npeak_layer = {peak}")
    print("pooled score vs SCC-axis projection:", summary["layer_stats"][peak]["score_vs_scc_axis_projection"])
    print("pooled score vs raw norm:", summary["layer_stats"][peak]["score_vs_raw_norm"])
    if sae_summary:
        print("top SAE feature:", sae_summary["top_gradient_features"][0])
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
