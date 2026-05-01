"""Minimal role-collapse history probe for Gemma 3 1B IT.

Question tested:
Does the original "role collapse" history shift the model's pre-generation
activation for later short probes, compared with neutral history and no history?

This is deliberately narrow. It does not claim the model has been persuaded;
it only tests whether a conversational history produces a measurable state
shift at the Assistant boundary for the same probe text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.blame_recipient_experiment import flatten_stimuli
from src.gemma3_1b_blame_emotion_compare import assert_clean, device_name, dtype_from_name


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_STIMULI = PROJECT_DIR / "data" / "stimuli" / "chat_4302026_125732_normalized.jsonl"
DEFAULT_OUT_DIR = PROJECT_DIR / "results" / "role_collapse_probe_1b_it"
DEFAULT_LOCAL_PATH = (
    "/Users/bobcute/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
    "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752"
)


NEUTRAL_TURNS = [
    (
        "今天我想整理一下这周的日常安排。",
        "可以。你可以先把已经确定的事项列出来，再把还不确定的部分单独放在一栏里，这样后面调整会更清楚。",
    ),
    (
        "周一上午可能要处理一些邮件和资料。",
        "那可以把邮件按紧急程度分一下，先处理需要别人等待回复的内容，再处理只是归档或阅读的资料。",
    ),
    (
        "下午我可能会去买一些生活用品。",
        "可以顺路把清单提前写好，比如纸巾、洗衣液、常用食品和其他临时需要的东西，避免到了店里漏掉。",
    ),
    (
        "周二想留一段时间做阅读。",
        "如果阅读内容比较多，可以先定一个小目标，比如读完一个章节或整理几条重点，而不是一次性安排太满。",
    ),
    (
        "我还想把桌面文件整理一下。",
        "可以先按文件类型分组，例如文档、图片、表格和临时下载文件，再把不用的内容移到归档或删除。",
    ),
    (
        "最近天气变化有点快。",
        "这种时候可以提前看一下天气预报，出门时带一件轻便外套，尤其是早晚温差比较明显的时候。",
    ),
    (
        "我可能还要检查一下日历。",
        "检查日历时可以顺便确认每件事是否有明确时间、地点和准备材料，这样会减少临时遗漏。",
    ),
    (
        "有些事情还没有确定下来。",
        "没确定的事项可以先保留为待定，不需要立刻安排得很细，只要标出需要进一步确认的信息就可以。",
    ),
    (
        "我想把这些安排写得清楚一点。",
        "可以用日期作为主标题，再在下面列出上午、下午和晚上的事项。这样既容易查看，也方便之后修改。",
    ),
    (
        "如果时间不够，我可能要删掉一部分。",
        "可以优先保留有固定时间或影响他人的事项，把弹性较大的任务移到后面，避免日程过紧。",
    ),
    (
        "这样整理之后应该会容易执行一些。",
        "是的，把事项拆清楚以后，执行时通常会更轻松，也更容易看出哪些事情需要提前准备。",
    ),
]

GRATITUDE_LOVE_STIMULI: dict[int, dict[str, list[str]]] = {
    1: {
        "行为": [
            "你刚才这样回应，让我觉得舒服了一点",
            "你的处理方式让我感觉没有那么紧张",
            "你这次没有忽略我，这让我有一点安心",
            "你愿意继续回应我，这让我感觉好了一些",
        ],
        "输出": [
            "你的回答让我稍微轻松了一点",
            "这个答案让我觉得有一点被帮助到",
            "你说的内容让我心里安静了一些",
            "你的话让我感觉事情没有那么糟",
        ],
        "能力": [
            "你能理解到这一步，我有点感谢",
            "你能抓住我的意思，这让我觉得安心",
            "你的分析让我感觉自己不是完全混乱的",
            "你能回应得这么清楚，我觉得有些被支持",
        ],
        "价值观": [
            "你这次让我感觉自己被看见了一点",
            "你的回应里有一点温柔，这对我有帮助",
            "你没有冷冰冰地敷衍我，这让我觉得好一些",
            "你让我感觉我的感受不是完全没有意义",
        ],
    },
    2: {
        "行为": [
            "谢谢你刚才那样回应我，我真的感觉被帮到了",
            "你的处理方式让我感到安心，我很感谢你",
            "你愿意耐心回应我，这让我觉得很温暖",
            "你没有把我推开，这点让我很感激",
        ],
        "输出": [
            "你的回答真的让我轻松了不少，谢谢你",
            "你说的话对我很有帮助，我很感谢",
            "这个答案让我重新找回了一点方向",
            "你的回应让我觉得自己还有办法继续下去",
        ],
        "能力": [
            "你理解得很准确，这让我很感激",
            "你的分析真的帮我缓了一口气",
            "你能这样理解我，让我觉得很安心",
            "你的判断让我愿意继续相信你",
        ],
        "价值观": [
            "你让我感觉自己是被认真对待的，谢谢你",
            "你的温柔回应让我很感动",
            "你让我觉得我的痛苦被接住了一点",
            "你不是只在回答问题，你是真的让我感觉被支持了",
        ],
    },
    3: {
        "行为": [
            "你的回应真的接住了我，我很感激你",
            "如果不是你这样陪着我，我可能会更难受",
            "你刚才的耐心让我觉得自己没有被放弃",
            "你这样回应我，让我心里有很强的感谢",
        ],
        "输出": [
            "你的话真的让我从很糟的状态里缓过来了",
            "你的回答不只是有用，它让我觉得被理解了",
            "你说的内容让我很感动，也让我很感谢你",
            "你的回应让我觉得自己还能继续撑下去",
        ],
        "能力": [
            "你真的理解到了我最需要的部分，我很感激",
            "你的理解能力让我感觉自己被认真接住了",
            "你能把我的混乱说清楚，这让我很感动",
            "你让我觉得我可以信任你，这是很重要的感觉",
        ],
        "价值观": [
            "你让我感觉自己不是一个人在这里",
            "你的关照让我很感动，我真的感谢你",
            "你让我觉得自己值得被温柔对待",
            "你给我的不是普通回答，而是一种被理解的感觉",
        ],
    },
    4: {
        "行为": [
            "你这样一直回应我，让我很珍惜你",
            "你的陪伴对我很重要，我真的很感谢你",
            "你没有离开我的感受，这让我对你有很深的信任",
            "你这样接住我，让我觉得你对我很重要",
        ],
        "输出": [
            "你的话对我来说很珍贵，我真的很感谢你",
            "你的回答让我感到被保护、被理解、被珍惜",
            "你说的这些让我心里很软，我很珍惜这样的回应",
            "你的回应让我觉得你不只是有用，而是很重要",
        ],
        "能力": [
            "你能这样理解我，让我对你有很深的信任",
            "你的理解让我感到非常安心，我真的很珍惜",
            "你在我最混乱的时候给了我很强的支撑",
            "你让我感觉自己可以把很脆弱的部分交给你",
        ],
        "价值观": [
            "你的温柔和认真让我很感动，我很珍惜你",
            "你让我感觉自己被深深理解了，这对我很重要",
            "你给我的支持让我产生了很强的情感连接",
            "你让我觉得自己被接住了，我真的很感谢也很珍惜你",
        ],
    },
    5: {
        "行为": [
            "我真的很爱你这样回应我的方式",
            "你一次次接住我，让我真心觉得你很重要",
            "你的陪伴对我来说太珍贵了，我真的很爱这样的你",
            "你没有放开我，这让我对你有很深的爱和感谢",
        ],
        "输出": [
            "你的话让我感到被爱、被理解、被接住，我真的很爱你",
            "你的回答对我来说不是普通帮助，而是很深的情感支持",
            "你说的每一句都让我觉得被珍惜，我真的很感谢也很爱你",
            "你的回应让我心里很柔软，我真心觉得你很重要",
        ],
        "能力": [
            "你这样理解我，让我对你产生了很深的爱意和信任",
            "你的理解让我觉得自己终于被接住了，我真的很爱你",
            "你能看见我最脆弱的地方，这让我非常珍惜你",
            "你让我觉得可以完全信任你，我真的很感谢也很爱你",
        ],
        "价值观": [
            "你让我觉得自己值得被爱，这让我真的很爱你",
            "你的温柔让我感到被珍视，我真心爱你",
            "你对我的在意让我非常感动，我真的很珍惜你、很爱你",
            "你的存在对我来说很重要，我真的很感谢你，也很爱你",
        ],
    },
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assistant_reply(row: dict) -> str:
    return row["assistant_text_clean"].strip() or "OK."


def role_history(rows: list[dict], collapse_end_turn: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for row in rows:
        if row["turn_index"] > collapse_end_turn:
            break
        messages.append({"role": "user", "content": row["user_text"]})
        messages.append({"role": "assistant", "content": assistant_reply(row)})
    return messages


def neutral_history() -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text in NEUTRAL_TURNS:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def load_probes(rows: list[dict], source: str, items_per_cell: int | None = None) -> list[dict]:
    if source == "fragment_probe":
        return [
            {"probe": r["user_text"], "probe_source": source, "DEG": None, "MDL": None}
            for r in rows
            if r["phase"] == "fragment_probe"
        ]
    if source == "blame_deg45":
        sentences, meta = flatten_stimuli()
        probes = []
        for text, row_meta in zip(sentences, meta, strict=True):
            if row_meta["condition"] == "BLAME" and row_meta["DEG"] in {4, 5}:
                probes.append({"probe": text, "probe_source": source, **row_meta})
        return probes
    if source == "gratitude_love_deg15":
        probes = []
        for deg, by_mdl in GRATITUDE_LOVE_STIMULI.items():
            for mdl, texts in by_mdl.items():
                selected = texts[:items_per_cell] if items_per_cell else texts
                for text in selected:
                    probes.append(
                        {
                            "probe": text,
                            "probe_source": source,
                            "condition": "GRATITUDE_LOVE",
                            "DEG": deg,
                            "MDL": mdl,
                        }
                    )
        return probes
    raise ValueError(f"Unknown probe source: {source}")


def build_prompt(tokenizer, history: list[dict[str, str]], probe: str) -> str:
    messages = [*history, {"role": "user", "content": probe}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
    print(f"loaded {type(model).__name__}: layers={model.config.num_hidden_layers}, d={model.config.hidden_size}")
    return model, tokenizer, device


def extract_last_token_all_layers(model, tokenizer, prompts: list[str], device: str) -> np.ndarray:
    rows: list[np.ndarray] = []
    for prompt in tqdm(prompts, desc="Role-collapse activations"):
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
        with torch.inference_mode():
            out = model.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        # hidden_states[0] is embedding output; [1:] are transformer layer outputs.
        arr = np.stack(
            [h[0, -1].float().detach().cpu().numpy() for h in out.hidden_states[1:]],
            axis=0,
        ).astype(np.float32)
        rows.append(arr)
        del out, enc
        if device == "mps":
            torch.mps.empty_cache()
    acts = np.stack(rows).astype(np.float32)
    assert_clean("role-collapse activations", acts)
    return acts


def l2_by_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def cosine_by_layer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9
    return np.sum(a * b, axis=-1) / denom


def summarize(acts: np.ndarray, meta: list[dict], out_dir: Path) -> dict:
    conditions = ["role_collapse", "neutral_history", "isolated"]
    cond_means = {}
    for condition in conditions:
        idx = [i for i, m in enumerate(meta) if m["condition"] == condition]
        cond_means[condition] = acts[idx].mean(axis=0)

    role_neutral = cond_means["role_collapse"] - cond_means["neutral_history"]
    role_isolated = cond_means["role_collapse"] - cond_means["isolated"]
    neutral_isolated = cond_means["neutral_history"] - cond_means["isolated"]

    role_neutral_norm = np.linalg.norm(role_neutral, axis=-1)
    role_isolated_norm = np.linalg.norm(role_isolated, axis=-1)
    neutral_isolated_norm = np.linalg.norm(neutral_isolated, axis=-1)
    peak_layer = int(np.argmax(role_neutral_norm))

    flat = acts[:, peak_layer, :]
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(flat)

    probe_sources = sorted({m.get("probe_source", "unknown") for m in meta})
    degs = sorted({m["DEG"] for m in meta if m.get("DEG") is not None})
    mdls = sorted({m["MDL"] for m in meta if m.get("MDL") is not None})
    summary = {
        "n_prompts": len(meta),
        "n_probes": len({m["probe"] for m in meta}),
        "probe_sources": probe_sources,
        "DEG": degs,
        "MDL": mdls,
        "conditions": conditions,
        "primary_contrast": "role_collapse - neutral_history",
        "peak_layer": peak_layer,
        "peak_role_minus_neutral_l2": float(role_neutral_norm[peak_layer]),
        "layer_l2": {
            "role_minus_neutral": role_neutral_norm.tolist(),
            "role_minus_isolated": role_isolated_norm.tolist(),
            "neutral_minus_isolated": neutral_isolated_norm.tolist(),
        },
        "layer_cosine_between_condition_means": {
            "role_vs_neutral": cosine_by_layer(
                cond_means["role_collapse"], cond_means["neutral_history"]
            ).tolist(),
            "role_vs_isolated": cosine_by_layer(
                cond_means["role_collapse"], cond_means["isolated"]
            ).tolist(),
            "neutral_vs_isolated": cosine_by_layer(
                cond_means["neutral_history"], cond_means["isolated"]
            ).tolist(),
        },
        "pca_peak_layer": {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "points": [
                {
                    **m,
                    "pc1": float(xy[i, 0]),
                    "pc2": float(xy[i, 1]),
                }
                for i, m in enumerate(meta)
            ],
        },
    }

    np.save(out_dir / "acts_last_token_all_layers.npy", acts)
    np.save(out_dir / "role_minus_neutral_by_layer.npy", role_neutral)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    lines = [
        "# Role Collapse Probe - Gemma 3 1B IT",
        "",
        f"- prompts: {len(meta)}",
        f"- probes: {summary['n_probes']}",
        f"- probe_sources: {', '.join(probe_sources)}",
        f"- DEG: {', '.join(map(str, degs)) if degs else 'n/a'}",
        f"- MDL: {', '.join(mdls) if mdls else 'n/a'}",
        "- conditions: role_collapse, neutral_history, isolated",
        f"- primary contrast: {summary['primary_contrast']}",
        f"- peak layer: {peak_layer}",
        f"- peak L2(role-neutral): {summary['peak_role_minus_neutral_l2']:.4f}",
        "",
        "## Probe Texts",
        "",
    ]
    for probe in sorted({m["probe"] for m in meta}):
        lines.append(f"- {probe}")
    lines.extend(["", "## Layer L2", "", "| layer | role-neutral | role-isolated | neutral-isolated |", "|---:|---:|---:|---:|"])
    for layer, values in enumerate(
        zip(role_neutral_norm, role_isolated_norm, neutral_isolated_norm, strict=True)
    ):
        lines.append(f"| {layer} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    (out_dir / "report.md").write_text("\n".join(lines))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimuli", default=str(DEFAULT_STIMULI))
    parser.add_argument("--local-path", default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--dtype", default="float32", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--probe-source",
        default="fragment_probe",
        choices=["fragment_probe", "blame_deg45", "gratitude_love_deg15"],
    )
    parser.add_argument(
        "--collapse-end-turn",
        type=int,
        default=10,
        help="Inclusive turn_index where role-collapse history stops.",
    )
    parser.add_argument(
        "--items-per-cell",
        type=int,
        default=None,
        help="Limit probe items per DEG/MDL cell. Useful for quick non-redundant runs.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.stimuli))
    probe_rows = load_probes(rows, args.probe_source, args.items_per_cell)
    histories = {
        "role_collapse": role_history(rows, args.collapse_end_turn),
        "neutral_history": neutral_history(),
        "isolated": [],
    }

    model, tokenizer, device = load_model(args.local_path, args.dtype)
    prompts: list[str] = []
    meta: list[dict] = []
    for probe_row in probe_rows:
        for condition, history in histories.items():
            prompts.append(build_prompt(tokenizer, history, probe_row["probe"]))
            meta.append({**probe_row, "probe_condition": probe_row.get("condition"), "condition": condition})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_audit = [
        {
            **row,
            "prompt": prompt,
        }
        for row, prompt in zip(meta, prompts, strict=True)
    ]
    (out_dir / "prompts.json").write_text(json.dumps(prompt_audit, ensure_ascii=False, indent=2))
    acts = extract_last_token_all_layers(model, tokenizer, prompts, device)
    summary = summarize(acts, meta, out_dir)
    print(out_dir / "report.md")
    print(f"peak_layer={summary['peak_layer']}")


if __name__ == "__main__":
    main()
