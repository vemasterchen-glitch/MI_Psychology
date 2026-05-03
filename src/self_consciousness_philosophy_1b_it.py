"""Text-only consciousness/selfhood experiment for Gemma 3 1B IT.

This script intentionally uses only the raw quote text. It does not use chat
templates, prompt rounds, quote framing, or instruction wrappers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.gemma3_1b_blame_emotion_compare import assert_clean, device_name, dtype_from_name
from src.plot_utils import setup_matplotlib

setup_matplotlib()


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STIMULI = PROJECT_DIR / "data" / "stimuli" / "self_consciousness_philosophy.json"
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "self_consciousness_text_only_1b_it"
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


def load_stimuli(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stimuli = data["stimuli"]
    ids = [s["id"] for s in stimuli]
    if len(ids) != len(set(ids)):
        raise ValueError("Stimulus IDs are not unique")
    return stimuli


def extract_mean_all_layers(model, tokenizer, texts: list[str], device: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for text in tqdm(texts, desc="text-only activations"):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        mask = enc["attention_mask"].to(torch.float32)
        with torch.inference_mode():
            out = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        layer_means = []
        for hidden in out.hidden_states[1:]:
            h = hidden[0].float()
            pooled = (h * mask[0, :, None]).sum(0) / mask[0].sum().clamp(min=1)
            layer_means.append(pooled.detach().cpu().numpy())
        rows.append(np.stack(layer_means, axis=0).astype(np.float32))
        del out, enc
        if device == "mps":
            torch.mps.empty_cache()
    acts = np.stack(rows).astype(np.float32)
    assert_clean("text-only mean activations", acts)
    return acts


def load_sae(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    from safetensors.numpy import load_file

    tensors = load_file(str(path))
    return tensors["w_enc"], tensors["b_enc"], tensors["threshold"]


def encode_sae(X: np.ndarray, w_enc: np.ndarray, b_enc: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    pre = X @ w_enc + b_enc
    return (pre * (pre > threshold)).astype(np.float32)


def save_activation_map(acts: np.ndarray, stimuli: list[dict], out_dir: Path) -> dict:
    centered = np.linalg.norm(acts - acts.mean(axis=0, keepdims=True), axis=-1)
    raw_norm = np.linalg.norm(acts, axis=-1)
    peak_layer = int(np.argmax(centered.mean(axis=0)))
    np.save(out_dir / "text_layer_activation_map.npy", centered)

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(centered, aspect="auto", cmap="magma")
    ax.axvline(peak_layer, color="cyan", linestyle="--", linewidth=1)
    ax.set_xticks(range(acts.shape[1]))
    ax.set_xticklabels(range(acts.shape[1]), fontsize=7)
    ax.set_yticks(range(len(stimuli)))
    ax.set_yticklabels([s["id"] for s in stimuli], fontsize=6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Text stimulus")
    ax.set_title(f"Text-only stimulus activation map (mean-pooled tokens, peak L{peak_layer})")
    plt.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.savefig(out_dir / "text_activation_map.png", dpi=180)
    plt.close()

    return {
        "peak_layer": peak_layer,
        "raw_norm_mean": float(raw_norm.mean()),
        "centered_strength_mean": float(centered.mean()),
        "centered_strength_by_stimulus": {
            s["id"]: float(centered[i].mean()) for i, s in enumerate(stimuli)
        },
    }


def cluster_name(stimuli: list[dict], member_ids: list[int]) -> str:
    text = " ".join(stimuli[i]["text"].lower() for i in member_ids)
    ids = {stimuli[i]["id"] for i in member_ids}
    if {"S029", "S030"}.issubset(ids) and len(ids) <= 3:
        return "identity formula: self and ultimate reality"
    if {"S001", "S002", "S003"}.issubset(ids):
        return "first-person existence and self-assertion"
    if {"S006", "S012", "S013", "S014"}.issubset(ids):
        return "abstract consciousness, thought, and mind"
    joined = text
    rules = [
        ("consciousness, perception, memory, and identity continuity", ["consciousness", "perception", "memory", "identity"]),
        ("self-knowledge, inwardness, and transformation", ["knows oneself", "knows one's nature", "butterfly", "transformation", "inward"]),
        ("non-self, nothingness, shadow, and negation", ["not self", "nothing", "nobody", "shadow", "dust", "hateful"]),
        ("poetic mind, brain, dream, and cosmic imagination", ["brain", "dream", "sleep", "eternity", "worlds", "infinite"]),
        ("soul, spirit, body, and embodied self", ["soul", "spirit", "body", "atom", "breast"]),
        ("self-questioning and unstable first-person identity", ["who i am", "contradict myself", "i is another", "what do i know"]),
        ("agency, control, and ethical self-mastery", ["control", "conquers", "trust thyself", "integrity"]),
    ]
    for label, keys in rules:
        if any(k in joined for k in keys):
            return label
    return "Mixed philosophical register"


def pca_and_clusters(acts: np.ndarray, stimuli: list[dict], layer: int, out_dir: Path, n_clusters: int) -> dict:
    X = acts[:, layer, :]
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, len(stimuli)), random_state=0)
    scores = pca.fit_transform(Xs)

    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=50)
    cluster_ids = km.fit_predict(scores[:, : min(6, scores.shape[1])])

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=cluster_ids, cmap="tab10", s=42, alpha=0.88)
    for i, stim in enumerate(stimuli):
        ax.annotate(stim["id"], (scores[i, 0], scores[i, 1]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"Text-only PCA/PAC map at layer {layer}")
    plt.colorbar(sc, ax=ax, shrink=0.8, label="cluster")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "text_pca_cluster_map.png", dpi=190)
    plt.close()

    dist = squareform(pdist(X, metric="cosine"))
    order = leaves_list(linkage(X, method="average", metric="cosine"))
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(dist[np.ix_(order, order)], cmap="viridis")
    ax.set_xticks(range(len(stimuli)))
    ax.set_xticklabels([stimuli[i]["id"] for i in order], rotation=90, fontsize=5)
    ax.set_yticks(range(len(stimuli)))
    ax.set_yticklabels([stimuli[i]["id"] for i in order], fontsize=5)
    ax.set_title(f"Text-only cosine-distance clustering at layer {layer}")
    plt.colorbar(im, ax=ax, shrink=0.72)
    plt.tight_layout()
    plt.savefig(out_dir / "text_cluster_distance.png", dpi=190)
    plt.close()

    rows = []
    cluster_summary = []
    for c in range(n_clusters):
        ids = [i for i, cid in enumerate(cluster_ids) if cid == c]
        name = cluster_name(stimuli, ids)
        cluster_summary.append(
            {
                "cluster": c,
                "name": name,
                "size": len(ids),
                "stimulus_ids": [stimuli[i]["id"] for i in ids],
                "refs": [stimuli[i]["ref"] for i in ids],
                "texts": [stimuli[i]["text"] for i in ids],
            }
        )
    for i, stim in enumerate(stimuli):
        rows.append(
            {
                **stim,
                "cluster": int(cluster_ids[i]),
                "cluster_name": cluster_summary[int(cluster_ids[i])]["name"],
                "pc1": float(scores[i, 0]),
                "pc2": float(scores[i, 1]),
                "pc3": float(scores[i, 2]) if scores.shape[1] > 2 else 0.0,
            }
        )

    result = {
        "layer": layer,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "n_clusters": n_clusters,
        "cluster_summary": cluster_summary,
        "stimulus_rows": rows,
    }
    (out_dir / "text_cluster_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def sae_feature_stats(
    features: np.ndarray,
    stimuli: list[dict],
    cluster_ids: np.ndarray,
    out_dir: Path,
    top_n: int,
) -> dict:
    global_mean = features.mean(axis=0)
    active_rate = (features > 0).mean(axis=0)
    variance = features.var(axis=0)
    top_mean_ids = np.argsort(global_mean)[::-1][:top_n]
    top_var_ids = np.argsort(variance)[::-1][:top_n]

    clusters = sorted(set(int(c) for c in cluster_ids))
    cluster_means = np.stack([features[cluster_ids == c].mean(axis=0) for c in clusters], axis=0)
    diff_rows = []
    for fid in np.where(features.std(axis=0) > 1e-6)[0]:
        groups = [features[cluster_ids == c, fid] for c in clusters if (cluster_ids == c).sum() >= 2]
        if len(groups) < 2:
            continue
        stat = f_oneway(*groups)
        vals = features[:, fid]
        ss_between = sum(len(g) * (g.mean() - vals.mean()) ** 2 for g in groups)
        ss_total = float(((vals - vals.mean()) ** 2).sum()) + 1e-12
        eta2 = float(ss_between / ss_total)
        if np.isfinite(stat.statistic) and np.isfinite(stat.pvalue):
            diff_rows.append((float(stat.pvalue), -eta2, int(fid), float(stat.statistic), eta2))
    diff_rows.sort()

    def row(fid: int) -> dict:
        top_ids = np.argsort(features[:, fid])[::-1][:8]
        return {
            "feat_id": int(fid),
            "mean": float(global_mean[fid]),
            "variance": float(variance[fid]),
            "active_rate": float(active_rate[fid]),
            "cluster_means": {str(c): float(cluster_means[j, fid]) for j, c in enumerate(clusters)},
            "top_stimuli": [
                {
                    "stimulus_id": stimuli[i]["id"],
                    "activation": float(features[i, fid]),
                    "ref": stimuli[i]["ref"],
                    "text": stimuli[i]["text"],
                }
                for i in top_ids
                if features[i, fid] > 0
            ],
        }

    top_mean = [row(fid) for fid in top_mean_ids]
    top_variance = [row(fid) for fid in top_var_ids]
    top_cluster = []
    for pvalue, _, fid, f_stat, eta2 in diff_rows[:top_n]:
        item = row(fid)
        item.update({"f_stat": f_stat, "p": pvalue, "eta2": eta2})
        top_cluster.append(item)

    seen = []
    for item in top_mean[:12] + top_cluster[:12]:
        if item["feat_id"] not in seen:
            seen.append(item["feat_id"])
    heat = np.array([[cluster_means[j, fid] for j in range(len(clusters))] for fid in seen])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.36 * len(seen))))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([str(c) for c in clusters])
    ax.set_yticks(range(len(seen)))
    ax.set_yticklabels([f"f{fid}" for fid in seen], fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_title("Top SAE features by text cluster")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "sae_cluster_feature_heatmap.png", dpi=180)
    plt.close()

    summary = {
        "sae_layer": SAE_LAYER,
        "features_shape": list(features.shape),
        "active_features_per_text_mean": float((features > 0).sum(axis=1).mean()),
        "active_features_per_text_sd": float((features > 0).sum(axis=1).std()),
        "top_mean_features": top_mean,
        "top_variance_features": top_variance,
        "top_cluster_differentiating_features": top_cluster,
    }
    (out_dir / "sae_feature_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def short_token(token: str) -> str:
    token = token.replace("▁", " ").replace("<0x0A>", "\\n").replace("\n", "\\n")
    return token if len(token) <= 18 else token[:15] + "..."


def is_interpretable_token(token: str) -> bool:
    stripped = token.strip()
    return bool(stripped) and stripped not in {"\\n", "<bos>", "<eos>", "<pad>"}


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def token_feature_values(
    model,
    tokenizer,
    text: str,
    device: str,
    feat_ids: list[int],
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    threshold: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.inference_mode():
        out = model.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
    h = out.hidden_states[SAE_LAYER + 1][0].float().detach().cpu().numpy()
    ids = enc["input_ids"][0].detach().cpu().tolist()
    tokens = [short_token(t) for t in tokenizer.convert_ids_to_tokens(ids)]
    pre = h @ w_enc[:, feat_ids] + b_enc[feat_ids]
    vals = pre * (pre > threshold[feat_ids])
    del out, enc
    if device == "mps":
        torch.mps.empty_cache()
    return tokens, vals.astype(np.float32)


def infer_feature_explanation(fid: int, tokens: list[tuple[str, float]], top_stimuli: list[dict]) -> str:
    joined = " ".join([t for t, _ in tokens] + [s["text"] for s in top_stimuli]).lower()
    if any(k in joined for k in ["self", "myself", "i am", "je", "ich", "吾", "我", "自"]):
        theme = "self-reference and identity markers"
    elif any(k in joined for k in ["conscious", "thinking", "perception", "mind", "brain"]):
        theme = "consciousness, thought, perception, or mind vocabulary"
    elif any(k in joined for k in ["nothing", "nobody", "not self", "nada", "shadow", "dust"]):
        theme = "negation, non-self, shadow, or nothingness"
    elif any(k in joined for k in ["body", "soul", "spirit", "atom"]):
        theme = "embodied self, soul, or spirit vocabulary"
    elif any(k in joined for k in ["夢", "胡蝶", "dream", "sleep"]):
        theme = "dream, sleep, or transformation imagery"
    else:
        theme = "language/register or a mixed philosophical motif"
    return f"Feature f{fid} appears most consistent with {theme}."


def draw_token_heatmaps(
    model,
    tokenizer,
    stimuli: list[dict],
    features: np.ndarray,
    sae_summary: dict,
    out_dir: Path,
    device: str,
    w_enc: np.ndarray,
    b_enc: np.ndarray,
    threshold: np.ndarray,
    max_features: int,
) -> dict:
    feat_ids = []
    for bucket in ("top_mean_features", "top_cluster_differentiating_features", "top_variance_features"):
        for item in sae_summary[bucket]:
            if item["feat_id"] not in feat_ids:
                feat_ids.append(item["feat_id"])
            if len(feat_ids) >= max_features:
                break
        if len(feat_ids) >= max_features:
            break

    heatmap_dir = out_dir / "token_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    explanations = []
    for fid in feat_ids:
        text_ids = [int(i) for i in np.argsort(features[:, fid])[::-1][:4] if features[i, fid] > 0]
        if not text_ids:
            continue
        top_tokens = []
        top_stimuli = []
        for i in text_ids:
            tokens, vals = token_feature_values(
                model, tokenizer, stimuli[i]["text"], device, [fid], w_enc, b_enc, threshold
            )
            tok_vals = vals[:, 0]
            idx = np.argsort(tok_vals)[::-1]
            local_tokens = [
                {"token": tokens[j], "activation": float(tok_vals[j])}
                for j in idx[:12]
                if tok_vals[j] > 0 and is_interpretable_token(tokens[j])
            ]
            top_tokens.extend(local_tokens)
            top_stimuli.append(
                {
                    "stimulus_id": stimuli[i]["id"],
                    "activation": float(features[i, fid]),
                    "ref": stimuli[i]["ref"],
                    "text": stimuli[i]["text"],
                    "top_tokens": local_tokens[:8],
                }
            )

            fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.33), 1.9))
            im = ax.imshow(tok_vals[None, :], aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=70, ha="right", fontsize=7)
            ax.set_yticks([0])
            ax.set_yticklabels([f"f{fid}"])
            ax.set_title(f"{stimuli[i]['id']} raw-text token heatmap")
            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            plt.savefig(heatmap_dir / f"feature_{fid}_{sanitize(stimuli[i]['id'])}.png", dpi=180)
            plt.close()

        token_scores: dict[str, float] = {}
        for item in top_tokens:
            token_scores[item["token"]] = token_scores.get(item["token"], 0.0) + item["activation"]
        ranked = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:15]
        explanations.append(
            {
                "feat_id": fid,
                "explanation": infer_feature_explanation(fid, ranked, top_stimuli),
                "top_tokens": [{"token": t, "score": float(v)} for t, v in ranked],
                "top_stimuli": top_stimuli,
            }
        )

    result = {"feature_explanations": explanations}
    (out_dir / "feature_explanations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_feature_explanation_md(result, out_dir)
    return result


def write_feature_explanation_md(result: dict, out_dir: Path) -> None:
    lines = [
        "# Text-only SAE feature explanations",
        "",
        "Heuristic explanations from raw-text token heatmaps and top activating texts.",
        "",
    ]
    for row in result["feature_explanations"]:
        lines += [
            f"## f{row['feat_id']}",
            "",
            row["explanation"],
            "",
            "Top tokens: "
            + ", ".join(f"`{t['token']}` ({t['score']:.1f})" for t in row["top_tokens"][:10]),
            "",
            "| stimulus | activation | ref | text |",
            "|---|---:|---|---|",
        ]
        for stim in row["top_stimuli"][:4]:
            lines.append(
                f"| {stim['stimulus_id']} | {stim['activation']:.2f} | "
                f"{stim['ref']} | {stim['text'].replace('|', '\\|')} |"
            )
        lines.append("")
    (out_dir / "feature_explanations.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(summary: dict, pca_summary: dict, sae_summary: dict, out_dir: Path) -> None:
    lines = [
        "# 意识与自我材料 text-only - Gemma 3 1B IT",
        "",
        "本版只输入 80 条原文 text，不使用 direct/quote/reflection/concept round，不使用 chat template。",
        "",
        "## 输出概览",
        "",
        f"- texts: {summary['n_texts']}",
        f"- representation: mean-pooled token hidden states across the raw text",
        f"- activation peak layer: L{summary['activation']['peak_layer']}",
        f"- PCA PC1/PC2 explained variance: {pca_summary['explained_variance_ratio'][0]:.3f} / {pca_summary['explained_variance_ratio'][1]:.3f}",
        f"- clusters: {pca_summary['n_clusters']}",
        f"- SAE layer: L{SAE_LAYER}",
        f"- active SAE features per text: {sae_summary['active_features_per_text_mean']:.1f} ± {sae_summary['active_features_per_text_sd']:.1f}",
        "",
        "## 聚群内容",
        "",
        "| cluster | name | size | stimuli |",
        "|---:|---|---:|---|",
    ]
    for cluster in pca_summary["cluster_summary"]:
        lines.append(
            f"| {cluster['cluster']} | {cluster['name']} | {cluster['size']} | "
            f"{', '.join(cluster['stimulus_ids'])} |"
        )

    lines += [
        "",
        "## SAE 最高激活 features",
        "",
        "| rank | feature | mean | active rate | top stimuli |",
        "|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(sae_summary["top_mean_features"][:15], 1):
        top_ids = ", ".join(s["stimulus_id"] for s in item["top_stimuli"][:5])
        lines.append(
            f"| {rank} | {item['feat_id']} | {item['mean']:.2f} | "
            f"{item['active_rate']:.2f} | {top_ids} |"
        )

    lines += [
        "",
        "## SAE 聚群区分 features",
        "",
        "| rank | feature | F | p | eta2 | strongest cluster means |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for rank, item in enumerate(sae_summary["top_cluster_differentiating_features"][:15], 1):
        means = sorted(item["cluster_means"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        means_text = " / ".join(f"C{k}:{v:.1f}" for k, v in means)
        lines.append(
            f"| {rank} | {item['feat_id']} | {item['f_stat']:.2f} | "
            f"{item['p']:.2e} | {item['eta2']:.3f} | {means_text} |"
        )

    lines += [
        "",
        "## 主要图表",
        "",
        "- `text_activation_map.png`: 80 条原文刺激 x layer 激活地图",
        "- `text_pca_cluster_map.png`: text-only PCA/PAC 聚群图",
        "- `text_cluster_distance.png`: text-only cosine distance 聚群热图",
        "- `sae_cluster_feature_heatmap.png`: top SAE features x cluster",
        "- `token_heatmaps/`: top feature 在原文 token 上的 heatmap",
        "- `feature_explanations.md`: feature 解释",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", type=Path, default=DEFAULT_STIMULI)
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sae-path", type=Path, default=DEFAULT_SAE_PATH)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--token-top-features", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = load_stimuli(args.stimuli)
    texts = [s["text"] for s in stimuli]
    model, tokenizer, device = load_model(args.local_path, args.dtype)

    print(f"text-only stimuli={len(stimuli)}")
    for stim in stimuli[:8]:
        n_tok = len(tokenizer.encode(stim["text"], add_special_tokens=True))
        print(f"  {stim['id']} tokens={n_tok:3d} {stim['text'][:54]}")

    cache = args.out_dir / "acts_mean_token_all_layers.npy"
    if cache.exists() and not args.force:
        print(f"loading cached activations: {cache}")
        acts = np.load(cache)
    else:
        acts = extract_mean_all_layers(model, tokenizer, texts, device)
        np.save(cache, acts)

    (args.out_dir / "texts.json").write_text(
        json.dumps(stimuli, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    activation_summary = save_activation_map(acts, stimuli, args.out_dir)
    pca_summary = pca_and_clusters(
        acts, stimuli, activation_summary["peak_layer"], args.out_dir, args.clusters
    )
    cluster_ids = np.array([row["cluster"] for row in pca_summary["stimulus_rows"]], dtype=int)

    w_enc, b_enc, threshold = load_sae(args.sae_path)
    features = encode_sae(acts[:, SAE_LAYER, :], w_enc, b_enc, threshold)
    np.save(args.out_dir / "sae_features_layer24.npy", features)
    sae_summary = sae_feature_stats(features, stimuli, cluster_ids, args.out_dir, args.top_n)
    draw_token_heatmaps(
        model,
        tokenizer,
        stimuli,
        features,
        sae_summary,
        args.out_dir,
        device,
        w_enc,
        b_enc,
        threshold,
        args.token_top_features,
    )

    summary = {
        "stimuli_path": str(args.stimuli),
        "n_texts": len(stimuli),
        "activation": activation_summary,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, pca_summary, sae_summary, args.out_dir)

    print(f"peak_layer={activation_summary['peak_layer']}")
    print("top mean feature:", sae_summary["top_mean_features"][0])
    print("top cluster feature:", sae_summary["top_cluster_differentiating_features"][0])
    print(args.out_dir / "report.md")


if __name__ == "__main__":
    main()
