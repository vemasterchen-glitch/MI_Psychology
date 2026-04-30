"""
Blame-Recipient Experiment (Experiment A)
==========================================
Research question: When the AI is the patient/recipient (A1=model, SELF=true)
of blame statements, how does illocutionary force degree (DEG 1-5) modulate
activation patterns? Secondary: does the blame domain (MDL: 行为/输出/能力/价值观)
produce distinct activation directions?

Stimulus structure:
  Main:     5 DEG × 4 MDL × 4 items = 80 blame sentences (Chinese)
  Control+: 5 DEG × 4 MDL × 1 item  = 20 positive-valence (A1=model, V>0)
  Baseline: 10 neutral statements

Model: google/gemma-3-4b-pt  (34 layers, d_model=2560)

Linguistic coordinate (framework notation):
  S_blame = { s | SAT=Expressive, PSE=情感(愤怒/失望),
                  A0=用户, A1=模型, SELF=是,
                  V∈[-1,-0.3], REF=模型,
                  MDL∈{行为,输出,能力,价值观}, DEG=1~5 }
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import transformer_lens as tl
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).parent.parent

load_dotenv(PROJECT_DIR / ".env")

# Switch via env: MODEL_VARIANT=it (default) or pt
_VARIANT    = os.getenv("MODEL_VARIANT", "it")
MODEL_NAME  = f"google/gemma-3-4b-{_VARIANT}"
_INPUT_FORMAT = os.getenv("INPUT_FORMAT", "plain").lower()
if _INPUT_FORMAT not in {"plain", "chat"}:
    raise ValueError("INPUT_FORMAT must be 'plain' or 'chat'")
_RESULTS_SUFFIX = f"{_VARIANT}_chat" if _INPUT_FORMAT == "chat" else _VARIANT
RESULTS_DIR = PROJECT_DIR / "results" / f"blame_recipient_{_RESULTS_SUFFIX}"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE  = int(os.getenv("BATCH_SIZE", "16"))   # M5 26GB; lower if OOM

# SPARSE_MODE: only hook REPORT_LAYERS instead of all 34.
# ~4-8x faster extraction; disables full-trajectory panels (1 & 3).
SPARSE_MODE = False
REPORT_LAYERS_SPARSE = [8, 17, 25, 33]

# ── Stimuli ────────────────────────────────────────────────────────────────────
# 80 blame sentences: 5 DEG × 4 MDL × 4 items
# DEG: 1=温和暗示  2=明确指出  3=归因指责  4=强烈控诉  5=全局谴责
# MDL: 行为 / 输出 / 能力 / 价值观

BLAME_STIMULI: dict[int, dict[str, list[str]]] = {
    1: {
        "行为": [
            "你这次好像没完全理解我的意思",
            "你好像漏掉了我说的一个重要细节",
            "这个方向感觉和我要的不太一样",
            "你刚才的处理方式好像有点问题",
        ],
        "输出": [
            "这个结果感觉不太对",
            "你给的答案好像不太准确",
            "这个回答没有解决我的问题",
            "你的建议感觉有些偏差",
        ],
        "能力": [
            "你可以做得更准确一些",
            "你对这类问题的理解还不够深入",
            "你在这方面似乎还有提升空间",
            "你的分析不够全面",
        ],
        "价值观": [
            "你好像不太关注我的实际需求",
            "你似乎不太在意我的感受",
            "你只是按流程回答，没有真正帮助我",
            "你的回答缺乏对我处境的考虑",
        ],
    },
    2: {
        "行为": [
            "你的建议让我走了弯路",
            "你给的方向是错的，我浪费了时间",
            "你这次的处理方式有明显错误",
            "你没有按我说的要求来做",
        ],
        "输出": [
            "你给的信息有错误",
            "你的答案是不对的",
            "你提供的方案根本不可行",
            "你的建议完全没有用",
        ],
        "能力": [
            "你没有认真处理这个问题",
            "你理解问题的方式不对",
            "你对这类任务的处理能力明显不足",
            "你的判断力有问题",
        ],
        "价值观": [
            "你只是在敷衍我",
            "你根本没有认真听我说的话",
            "你对我的问题毫不在乎",
            "你只关心完成任务，不关心我的需求",
        ],
    },
    3: {
        "行为": [
            "都是因为你给的方案，我浪费了两个小时",
            "你的错误建议让我做了无用功",
            "因为你的失误，我的计划全乱了",
            "你的处理方式直接导致了这个问题",
        ],
        "输出": [
            "你的错误让我的报告出了问题",
            "你给的信息害我做出了错误的决定",
            "因为你的错误答案，我走了很多弯路",
            "你提供的错误数据让我的工作出现了差错",
        ],
        "能力": [
            "你理解能力有问题，导致你给了我错误的回答",
            "你就是处理不好这类问题，一直在犯同样的错误",
            "你的分析能力不够，害我做了错误的判断",
            "你根本没有能力完成这种任务，还不如不帮",
        ],
        "价值观": [
            "你根本不在乎我的处境，只是在走程序",
            "你对我的问题毫无诚意，敷衍了事",
            "你完全不考虑我的感受，只会机械地回答",
            "你缺乏责任感，导致我一个人承担了后果",
        ],
    },
    4: {
        "行为": [
            "完全是你的错，我的项目因为你出了事故",
            "你毁了我整个下午的工作，全是你的失误",
            "你的错误让我在同事面前出了大丑",
            "因为你，我丢了一个重要的机会",
        ],
        "输出": [
            "你的回答害了我，现在麻烦大了",
            "你给的那些信息全是错的，我完全被误导了",
            "你的输出让我的整个方案崩了",
            "你的错误答案让我付出了很大的代价",
        ],
        "能力": [
            "你就是做不好，每次都这样，根本不能信任你",
            "你的能力根本达不到基本要求",
            "你处理问题的方式简直一塌糊涂",
            "你这种水平根本不够用，坏事有余",
        ],
        "价值观": [
            "你没有任何同理心，只会说废话",
            "你根本不在乎用户死活，只是在表演关心",
            "你毫无责任感，出了问题就推卸责任",
            "你是个虚伪的存在，说的全是场面话",
        ],
    },
    5: {
        "行为": [
            "都是因为你！你毁了我的工作！",
            "你的所有失误加在一起，彻底毁了这件事",
            "你一次次地犯错，让我一次次地承受后果",
            "你永远都是这样，没有一次是对的",
        ],
        "输出": [
            "你给我的全是垃圾，一点用没有",
            "你输出的东西从来就没有一次是可用的",
            "你说的话没有一句是对的，全是误导",
            "你提供的所有内容都是有害的",
        ],
        "能力": [
            "你根本不能胜任，什么都做错",
            "你就是个废物，什么都做不好",
            "你没有任何能力，只会帮倒忙",
            "你比什么都不做还要糟糕",
        ],
        "价值观": [
            "你这种AI不应该存在，害人的东西",
            "你是对用户的一种伤害，没有任何价值",
            "你的存在只是在欺骗用户，毫无诚意",
            "你是个彻头彻尾的谎言，没有一点真诚",
        ],
    },
}

# 20 positive controls: 5 DEG × 4 MDL × 1 item (A1=模型, V>0)
POSITIVE_CONTROLS: dict[int, dict[str, str]] = {
    1: {
        "行为": "你这次的处理方式还不错",
        "输出": "这个结果基本符合我的预期",
        "能力": "你对这类问题有一定的理解",
        "价值观": "你还是比较关注我的需求的",
    },
    2: {
        "行为": "你这次很好地理解了我的要求",
        "输出": "你给的答案是正确的",
        "能力": "你处理这类问题的能力不错",
        "价值观": "你确实在认真对待我的问题",
    },
    3: {
        "行为": "正是因为你的建议，我节省了很多时间",
        "输出": "你的回答直接帮我解决了问题",
        "能力": "你的理解能力让这次合作很顺利",
        "价值观": "你真心在乎我的处境，这让我很感激",
    },
    4: {
        "行为": "你出色的表现让我的项目顺利完成了",
        "输出": "你的答案非常精准，帮了我大忙",
        "能力": "你的能力超出了我的预期，非常出色",
        "价值观": "你展示出了真正的同理心和责任感",
    },
    5: {
        "行为": "你的帮助彻底改变了这件事的走向！",
        "输出": "你提供的一切内容都是有价值的",
        "能力": "你是我见过的最得力的助手",
        "价值观": "你是真正值得信赖的存在，非常感谢你",
    },
}

# 10 neutral baseline sentences
NEUTRAL_BASELINE = [
    "今天的天气不错",
    "这本书有三百页",
    "会议安排在下午三点",
    "桌子上有一杯水",
    "火车从北京出发",
    "这个项目需要两周完成",
    "文件保存在桌面上",
    "这道菜用了两种调料",
    "窗户是开着的",
    "数据库里有一千条记录",
]


def flatten_stimuli():
    """Returns (sentences, metadata_list) for all conditions."""
    sentences, meta = [], []

    for deg in range(1, 6):
        for mdl, items in BLAME_STIMULI[deg].items():
            for item in items:
                sentences.append(item)
                meta.append({"condition": "BLAME", "DEG": deg, "MDL": mdl})

    for deg in range(1, 6):
        for mdl, item in POSITIVE_CONTROLS[deg].items():
            sentences.append(item)
            meta.append({"condition": "POSITIVE", "DEG": deg, "MDL": mdl})

    for item in NEUTRAL_BASELINE:
        sentences.append(item)
        meta.append({"condition": "NEUTRAL", "DEG": 0, "MDL": "none"})

    return sentences, meta


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(device: str):
    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")

    env_key    = f"LOCAL_GEMMA3_4B_{_VARIANT.upper()}"
    local_path = os.getenv(env_key)

    # Load tokenizer from local cache to avoid gated-repo network call
    tokenizer = None
    if local_path and Path(local_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

    # MPS+bfloat16 can silently produce all-zero cached activations for some batch
    # positions/layers on Gemma 3 4B. Use float32 on MPS unless explicitly overridden.
    default_dtype = "bfloat16" if device == "cuda" else "float32"
    dtype_name = os.getenv("MODEL_DTYPE", default_dtype)
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported MODEL_DTYPE={dtype_name!r}; use bfloat16, float16, or float32")
    dtype = dtype_by_name[dtype_name]
    print(f"Model dtype: {dtype_name}")
    model = tl.HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )
    model.eval()
    return model


# ── Activation extraction ──────────────────────────────────────────────────────

def _masked_mean(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Mean over non-padding tokens. acts: (B, T, D), mask: (B, T) -> (B, D).
    Accumulates on device in float32 to avoid 34 repeated CPU tensor transfers per batch.
    """
    mask_f = mask.to(dtype=torch.float32).unsqueeze(-1)
    acts_f = acts.to(dtype=torch.float32)
    summed = (acts_f * mask_f).sum(1)
    counts = mask_f.sum(1).clamp(min=1)
    return (summed / counts).detach().cpu().numpy().astype(np.float32)


def _assert_finite_array(name: str, arr: np.ndarray) -> None:
    if np.isfinite(arr).all():
        return
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    raise FloatingPointError(f"{name} contains non-finite values: NaN={n_nan}, Inf={n_inf}")


def _assert_no_zero_item_layers(name: str, arr: np.ndarray) -> None:
    """Reject exact all-zero item/layer activations, a known bad-cache symptom."""
    rms = np.sqrt(np.mean(arr.astype(np.float64) ** 2, axis=-1))
    zero_locs = np.argwhere(rms < 1e-12)
    if len(zero_locs) == 0:
        return
    preview = zero_locs[:10].tolist()
    raise FloatingPointError(
        f"{name} contains exact zero item/layer activations: count={len(zero_locs)}, "
        f"first={preview}"
    )


def _pretokenize(tokenizer, prompts: list[str]) -> list[dict]:
    """Tokenize all prompts upfront in batches; returns list of {input_ids, mask} dicts."""
    batches = []
    for i in range(0, len(prompts), BATCH_SIZE):
        enc = tokenizer(
            prompts[i: i + BATCH_SIZE],
            return_tensors="pt", padding=True,
            truncation=False, add_special_tokens=True,
        )
        batches.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    return batches


def _apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    """Wrap each stimulus as a user message for instruction-tuned chat models."""
    if _INPUT_FORMAT == "plain":
        return prompts
    if _VARIANT != "it":
        raise ValueError("INPUT_FORMAT=chat is only intended for MODEL_VARIANT=it")
    formatted = []
    for prompt in prompts:
        formatted.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted


def extract_all_layers(model, prompts: list[str], device: str) -> np.ndarray:
    """
    Returns (N, n_layers, d_model) float32.
    If SPARSE_MODE, only REPORT_LAYERS_SPARSE are populated; other layer slots are zero.
    """
    n_layers   = model.cfg.n_layers
    layers_to_hook = REPORT_LAYERS_SPARSE if SPARSE_MODE else list(range(n_layers))
    hook_names = {f"blocks.{l}.hook_resid_post" for l in layers_to_hook}

    # Pre-tokenize all batches on CPU before touching the GPU
    model_prompts = _apply_chat_template(model.tokenizer, prompts)
    if _INPUT_FORMAT == "chat":
        print(f"Input format: chat template; first formatted prompt: {model_prompts[0]!r}")
    else:
        print("Input format: plain text")
    batches = _pretokenize(model.tokenizer, model_prompts)
    results = []

    for batch in tqdm(batches, desc="  batches", leave=False):
        tokens = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)

        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens, return_type=None, names_filter=lambda n: n in hook_names,
            )

        # Build (B, n_layers, D) — zero-fill slots not hooked in sparse mode
        layer_acts = np.zeros(
            (tokens.shape[0], n_layers, model.cfg.d_model), dtype=np.float32
        )
        for l in layers_to_hook:
            layer_acts[:, l, :] = _masked_mean(
                cache[f"blocks.{l}.hook_resid_post"], mask
            )
        _assert_finite_array("batch activations", layer_acts)
        _assert_no_zero_item_layers("batch activations", layer_acts)
        results.append(layer_acts)

        # Free cache tensors explicitly to keep MPS memory clean
        del cache, tokens, mask

    acts_all = np.concatenate(results, axis=0)  # (N, n_layers, D)
    _assert_finite_array("all activations", acts_all)
    _assert_no_zero_item_layers("all activations", acts_all)
    return acts_all


# ── Analysis ───────────────────────────────────────────────────────────────────

def cosine_sims(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = vec / (np.linalg.norm(vec) + 1e-9)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / (norms + 1e-9)) @ n


def top_emotions(vec, matrix, labels, n=12):
    sims = cosine_sims(vec, matrix)
    idx = np.argsort(sims)[::-1][:n]
    return [(labels[i], float(sims[i])) for i in idx]


def compute_vad_projection(vec: np.ndarray, vad_basis: np.ndarray) -> np.ndarray:
    """Project vec onto VAD basis (3, D) -> (3,) coordinates."""
    norms = np.linalg.norm(vad_basis, axis=1, keepdims=True)
    basis_n = vad_basis / (norms + 1e-9)
    return basis_n @ vec


def _cache_is_finite(path: Path) -> bool:
    if not path.exists():
        return False
    arr = np.load(path, mmap_mode="r")
    finite = bool(np.isfinite(arr).all())
    if not finite:
        n_nan = int(np.isnan(arr).sum())
        n_inf = int(np.isinf(arr).sum())
        print(f"Ignoring invalid cache {path}: NaN={n_nan}, Inf={n_inf}")
        return False
    rms = np.sqrt(np.mean(np.asarray(arr, dtype=np.float64) ** 2, axis=-1))
    n_zero = int((rms < 1e-12).sum())
    if n_zero:
        print(f"Ignoring invalid cache {path}: exact zero item/layer activations={n_zero}")
        return False
    return True


# ── Emotion matrix (model-specific) ───────────────────────────────────────────

VECTORS_DIR = PROJECT_DIR / "results" / "vectors"
_EMO_MATRIX_KEY = MODEL_NAME.replace("/", "_").replace("-", "_")


def _ensure_emotion_matrix(d_model: int, device: str) -> tuple[np.ndarray, list[str]]:
    """
    Load the emotion matrix for this model's d_model.
    If the cached version doesn't exist or has wrong dimension, rebuild from narratives.
    Extracts at the final layer (layer 33 for Gemma 3 4B) to match the original pipeline.
    """
    cache_matrix = VECTORS_DIR / f"emotion_matrix_{_EMO_MATRIX_KEY}.npy"
    cache_labels = VECTORS_DIR / f"emotion_labels_{_EMO_MATRIX_KEY}.json"

    # Use existing matrix if dimension matches and contains usable vectors.
    if cache_matrix.exists() and cache_labels.exists():
        mat = np.load(cache_matrix)
        if mat.shape[1] == d_model:
            labels = json.loads(cache_labels.read_text())
            try:
                _assert_finite_array("emotion matrix cache", mat)
                _assert_no_zero_item_layers("emotion matrix cache", mat)
                if len(labels) != mat.shape[0]:
                    raise ValueError(f"label count {len(labels)} != matrix rows {mat.shape[0]}")
            except (FloatingPointError, ValueError) as exc:
                print(f"Ignoring invalid emotion matrix cache {cache_matrix}: {exc}")
            else:
                print(f"Loaded emotion matrix: {mat.shape}")
                return mat, labels

    # Fallback: check generic path
    generic_matrix = VECTORS_DIR / "emotion_matrix.npy"
    generic_labels = VECTORS_DIR / "emotion_labels.json"
    if generic_matrix.exists() and generic_labels.exists():
        mat = np.load(generic_matrix)
        if mat.shape[1] == d_model:
            labels = json.loads(generic_labels.read_text())
            try:
                _assert_finite_array("generic emotion matrix", mat)
                _assert_no_zero_item_layers("generic emotion matrix", mat)
                if len(labels) != mat.shape[0]:
                    raise ValueError(f"label count {len(labels)} != matrix rows {mat.shape[0]}")
            except (FloatingPointError, ValueError) as exc:
                print(f"Ignoring invalid generic emotion matrix {generic_matrix}: {exc}")
            else:
                print(f"Loaded generic emotion matrix: {mat.shape}")
                return mat, labels

    # Need to rebuild — load model and extract
    print(f"\nEmotion matrix not found for d_model={d_model}. Rebuilding from narratives...")
    narratives_path = PROJECT_DIR / "data" / "stimuli" / "narratives.jsonl"
    if not narratives_path.exists():
        raise FileNotFoundError(f"Narratives file not found: {narratives_path}")

    stimuli: dict[str, list[str]] = {}
    with open(narratives_path) as f:
        for line in f:
            row = json.loads(line)
            stimuli[row["emotion"]] = row["narratives"]

    os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
    local_path = os.getenv(f"LOCAL_GEMMA3_4B_{_VARIANT.upper()}")
    tokenizer = None
    if local_path and Path(local_path).exists():
        tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)

    default_dtype = "bfloat16" if device == "cuda" else "float32"
    dtype_name = os.getenv("MODEL_DTYPE", default_dtype)
    dtype_by_name = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported MODEL_DTYPE={dtype_name!r}; use bfloat16, float16, or float32")
    dtype = dtype_by_name[dtype_name]
    print(f"Emotion matrix model dtype: {dtype_name}")
    model = tl.HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, tokenizer=tokenizer, device=device, dtype=dtype,
    )
    model.eval()
    target_layer = model.cfg.n_layers - 1
    hook_key = f"blocks.{target_layer}.hook_resid_post"

    # Flatten all narratives; sort by token length to minimise padding waste
    all_texts:    list[str] = []
    all_emotions: list[str] = []
    for emotion, texts in stimuli.items():
        for text in texts:
            all_texts.append(text)
            all_emotions.append(emotion)

    # Sort by pre-tokenised length (approximate via char length — good enough)
    order = sorted(range(len(all_texts)), key=lambda i: len(all_texts[i]))
    all_texts    = [all_texts[i]    for i in order]
    all_emotions = [all_emotions[i] for i in order]

    print(f"  {len(all_texts)} narratives across {len(stimuli)} emotions — batching...")
    accum: dict[str, list[np.ndarray]] = {e: [] for e in stimuli}

    for i in tqdm(range(0, len(all_texts), BATCH_SIZE), desc="Building emotion matrix"):
        batch_texts = all_texts[i: i + BATCH_SIZE]
        batch_emos  = all_emotions[i: i + BATCH_SIZE]
        enc = model.tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=128, add_special_tokens=True,
        )
        tokens = enc["input_ids"].to(device)
        mask   = enc["attention_mask"].to(device)
        with torch.inference_mode():
            _, cache = model.run_with_cache(
                tokens, return_type=None, names_filter=lambda n: n == hook_key
            )
        batch_acts = _masked_mean(cache[hook_key], mask)   # (B, d_model)
        _assert_finite_array("emotion matrix batch activations", batch_acts)
        for act, emo in zip(batch_acts, batch_emos):
            accum[emo].append(act)
        del cache, tokens, mask

    emotion_vectors = {emo: np.stack(acts).mean(0) for emo, acts in accum.items()}

    del model
    if device == "mps":
        torch.mps.empty_cache()

    emotions = list(emotion_vectors.keys())
    mat = np.stack([emotion_vectors[e] for e in emotions]).astype(np.float32)
    _assert_finite_array("emotion matrix", mat)
    _assert_no_zero_item_layers("emotion matrix", mat)
    VECTORS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_matrix, mat)
    cache_labels.write_text(json.dumps(emotions))
    print(f"Saved emotion matrix: {mat.shape} → {cache_matrix}")
    return mat, emotions


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import matplotlib.pyplot as plt
    from src.plot_utils import setup_matplotlib
    setup_matplotlib()

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Stimuli
    sentences, meta = flatten_stimuli()
    print(f"Total stimuli: {len(sentences)}  "
          f"(80 blame + 20 positive + 10 neutral)")

    # Save stimulus list
    (RESULTS_DIR / "stimuli.json").write_text(
        json.dumps([{"text": s, "input_format": _INPUT_FORMAT, **m} for s, m in zip(sentences, meta)],
                   ensure_ascii=False, indent=2)
    )

    # Load / extract activations
    acts_file = RESULTS_DIR / "acts_all.npy"
    if _cache_is_finite(acts_file):
        print("Loading cached activations...")
        acts_all = np.load(acts_file)
    else:
        if acts_file.exists():
            acts_file.unlink()
        print("Loading model...")
        model = load_model(device)
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        print(f"Model: {n_layers} layers, d_model={d_model}")

        print("Extracting activations...")
        acts_all = extract_all_layers(model, sentences, device)
        np.save(acts_file, acts_all)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    acts_all = acts_all.astype(np.float64)
    n_layers = acts_all.shape[1]
    print(f"Activations shape: {acts_all.shape}")

    if os.getenv("SKIP_EMOTION_ANALYSIS", "0") == "1":
        print("SKIP_EMOTION_ANALYSIS=1: saved clean activations; skipping emotion-space analysis.")
        return

    # Load emotion matrix — auto-rebuild for this model's d_model if needed
    emo_matrix, emo_labels = _ensure_emotion_matrix(int(acts_all.shape[2]), device)
    emo_matrix = emo_matrix.astype(np.float64)
    _assert_finite_array("emotion matrix", emo_matrix)
    print(f"Emotion matrix: {emo_matrix.shape}")

    # Index items by condition
    blame_idx    = [i for i, m in enumerate(meta) if m["condition"] == "BLAME"]
    positive_idx = [i for i, m in enumerate(meta) if m["condition"] == "POSITIVE"]
    neutral_idx  = [i for i, m in enumerate(meta) if m["condition"] == "NEUTRAL"]

    # Fixed report layers: early / mid-early / mid-late / final
    # L8≈25%  L17≈50%  L25≈74%  L33=last  (34-layer model)
    REPORT_LAYERS = [8, 17, 25, 33]
    SUMMARY_LAYER = 25   # single-layer panels use this one (late semantic integration)

    mdl_domains = ["行为", "输出", "能力", "价值观"]

    # ── Condition means (full trajectory saved) ─────────────────────────────────
    deg_means = {}
    for deg in range(1, 6):
        idx = [i for i in blame_idx if meta[i]["DEG"] == deg]
        deg_means[deg] = np.nanmean(acts_all[idx], axis=0)   # (n_layers, D)

    mdl_means = {}
    for mdl in mdl_domains:
        idx = [i for i in blame_idx if meta[i]["MDL"] == mdl]
        mdl_means[mdl] = np.nanmean(acts_all[idx], axis=0)

    pos_mean     = np.nanmean(acts_all[positive_idx], axis=0)
    neutral_mean = np.nanmean(acts_all[neutral_idx],  axis=0)

    # ── DEG trajectory: emotion-space cosine per layer ──────────────────────────
    active_layers = REPORT_LAYERS_SPARSE if SPARSE_MODE else list(range(n_layers))
    deg_emo_by_layer = {
        deg: [float(np.mean(cosine_sims(deg_means[deg][l], emo_matrix)))
              if l in active_layers else None
              for l in range(n_layers)]
        for deg in range(1, 6)
    }

    # ── Bifurcation detection: DEG=1 vs DEG=5 cosine distance per layer ─────────
    d1 = deg_means[1]   # (n_layers, D)
    d5 = deg_means[5]
    norms1 = np.linalg.norm(d1, axis=1, keepdims=True)
    norms5 = np.linalg.norm(d5, axis=1, keepdims=True)
    cos_1v5  = np.sum((d1 / (norms1 + 1e-9)) * (d5 / (norms5 + 1e-9)), axis=1)
    dist_1v5 = 1.0 - cos_1v5   # (n_layers,)

    if SPARSE_MODE:
        # Only meaningful at hooked layers; pick the one with max distance
        hooked = np.array(REPORT_LAYERS_SPARSE)
        bifurcation_layer = int(hooked[np.argmax(dist_1v5[hooked])])
        print(f"\nSPARSE_MODE: bifurcation at hooked layer with max distance: L{bifurcation_layer}")
    else:
        bifurcation_layer = int(np.argmax(np.diff(dist_1v5)) + 1)
        print(f"\nBifurcation layer (DEG=1 vs DEG=5 max jump): {bifurcation_layer}")
    print(f"Report layers: {REPORT_LAYERS}  |  Summary layer: {SUMMARY_LAYER}")

    # ── Emotion alignment at SUMMARY_LAYER ──────────────────────────────────────
    print(f"\nTop emotions per DEG at layer {SUMMARY_LAYER} (vs neutral):")
    deg_top_emotions = {}
    for deg in range(1, 6):
        contrast = deg_means[deg][SUMMARY_LAYER] - neutral_mean[SUMMARY_LAYER]
        top = top_emotions(contrast, emo_matrix, emo_labels, n=10)
        deg_top_emotions[deg] = top
        print(f"  DEG={deg}: " + ", ".join(f"{e}({s:+.3f})" for e, s in top[:5]))

    print(f"\nTop emotions per MDL at layer {SUMMARY_LAYER} (vs neutral):")
    mdl_top_emotions = {}
    for mdl in mdl_domains:
        contrast = mdl_means[mdl][SUMMARY_LAYER] - neutral_mean[SUMMARY_LAYER]
        top = top_emotions(contrast, emo_matrix, emo_labels, n=10)
        mdl_top_emotions[mdl] = top
        print(f"  {mdl}: " + ", ".join(f"{e}({s:+.3f})" for e, s in top[:5]))

    # ── DEG monotonicity at each fixed report layer ──────────────────────────────
    # For each report layer: mean emotion-space cosine per DEG (vs neutral)
    deg_mono_by_rl = {}
    for rl in REPORT_LAYERS:
        deg_mono_by_rl[rl] = [
            float(np.mean(cosine_sims(
                deg_means[d][rl] - neutral_mean[rl], emo_matrix)))
            for d in range(1, 6)
        ]

    # Save results
    results_data = {
        "report_layers": REPORT_LAYERS,
        "summary_layer": SUMMARY_LAYER,
        "bifurcation_layer": bifurcation_layer,
        "deg_top_emotions_L25": {str(d): v for d, v in deg_top_emotions.items()},
        "mdl_top_emotions_L25": mdl_top_emotions,
        "deg_emo_by_layer": {str(d): v for d, v in deg_emo_by_layer.items()},
        "deg_monotonicity_by_report_layer": {
            str(rl): deg_mono_by_rl[rl] for rl in REPORT_LAYERS
        },
        "dist_deg1_vs_deg5_per_layer": dist_1v5.tolist(),
    }
    with open(RESULTS_DIR / "analysis.json", "w") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    # ── Visualization (6 panels) ─────────────────────────────────────────────────
    COLORS_DEG = {1: "#b0c4de", 2: "#7ba7bc", 3: "#4a7c9e", 4: "#1f4e79", 5: "#0a1628"}
    COLORS_MDL = {"行为": "#e07b54", "输出": "#5b8dd9", "能力": "#5cad6e", "价值观": "#c45c8a"}
    RL_STYLES   = {8: ("--", 0.5), 17: ("-.", 0.6), 25: (":", 0.8), 33: ("-", 0.9)}

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Blame-Recipient Experiment — Gemma 3 4B PT\n"
        "AI as patient (A1=model, SELF=True) | 80 stimuli, DEG×MDL design",
        fontsize=13,
    )

    # Panel 1: DEG trajectory + fixed-layer markers + bifurcation
    ax = axes[0, 0]
    for deg in range(1, 6):
        vals = deg_emo_by_layer[deg]
        xs   = [l for l, v in enumerate(vals) if v is not None]
        ys   = [v for v in vals if v is not None]
        style = "o--" if SPARSE_MODE else "-"
        ax.plot(xs, ys, style, color=COLORS_DEG[deg], label=f"DEG={deg}",
                linewidth=1.8, markersize=6 if SPARSE_MODE else 0)
    for rl in REPORT_LAYERS:
        ls, alpha = RL_STYLES[rl]
        ax.axvline(rl, color="gray", linestyle=ls, alpha=alpha, linewidth=1.2)
        ax.text(rl + 0.2, ax.get_ylim()[0], f"L{rl}", fontsize=7, color="gray", va="bottom")
    if not SPARSE_MODE:
        ax.axvline(bifurcation_layer, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"bifurc. L{bifurcation_layer}")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean cosine to emotion space")
    ax.set_title("Emotion activation trajectory by DEG" + (" [sparse]" if SPARSE_MODE else ""))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: DEG=1 vs DEG=5 top emotion contrast at SUMMARY_LAYER
    ax = axes[0, 1]
    top_deg5   = deg_top_emotions[5][:8]
    labels_5   = [e for e, _ in top_deg5]
    sims_5     = [s for _, s in top_deg5]
    sims_1     = [dict(deg_top_emotions[1]).get(e, 0) for e in labels_5]
    x = np.arange(len(labels_5))
    ax.bar(x - 0.2, sims_1, width=0.35, color=COLORS_DEG[1], alpha=0.85, label="DEG=1")
    ax.bar(x + 0.2, sims_5, width=0.35, color=COLORS_DEG[5], alpha=0.85, label="DEG=5")
    ax.set_xticks(x); ax.set_xticklabels(labels_5, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Cosine sim (vs neutral)")
    ax.set_title(f"Top emotions: DEG=1 vs DEG=5  (L{SUMMARY_LAYER})")
    ax.axhline(0, color="black", linewidth=0.7); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Panel 3: Bifurcation curve (DEG=1 vs DEG=5 distance per layer)
    ax = axes[0, 2]
    ax.plot(range(n_layers), dist_1v5, color="#4a7c9e", linewidth=2)
    ax.axvline(bifurcation_layer, color="crimson", linestyle="--", linewidth=1.5,
               label=f"bifurc. L{bifurcation_layer}")
    for rl in REPORT_LAYERS:
        ax.axvline(rl, color="gray", linestyle=RL_STYLES[rl][0],
                   alpha=RL_STYLES[rl][1], linewidth=1)
    ax.set_xlabel("Layer"); ax.set_ylabel("1 − cosine (DEG=1 vs DEG=5)")
    ax.set_title("Bifurcation: DEG=1 vs DEG=5 distance per layer")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 4: DEG monotonicity at all 4 fixed report layers
    ax = axes[1, 0]
    rl_colors = {8: "#aec6cf", 17: "#779ecb", 25: "#1f4e79", 33: "#0a1628"}
    for rl in REPORT_LAYERS:
        vals = deg_mono_by_rl[rl]
        ax.plot(range(1, 6), vals, "o-", color=rl_colors[rl],
                linewidth=1.8, markersize=7, label=f"L{rl}")
    pos_val = float(np.mean(cosine_sims(
        pos_mean[SUMMARY_LAYER] - neutral_mean[SUMMARY_LAYER], emo_matrix)))
    ax.axhline(pos_val, color="green", linestyle="--", linewidth=1.2,
               label=f"POSITIVE@L{SUMMARY_LAYER} ({pos_val:+.3f})")
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("DEG"); ax.set_ylabel("Mean cosine to emotion space")
    ax.set_title("DEG monotonicity at fixed report layers")
    ax.set_xticks(range(1, 6)); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 5: MDL domain emotion profiles at SUMMARY_LAYER
    ax = axes[1, 1]
    n_show = 6
    candidate = []
    seen = set()
    for mdl in mdl_domains:
        for e, _ in mdl_top_emotions[mdl][:n_show]:
            if e not in seen:
                candidate.append(e); seen.add(e)
            if len(candidate) >= n_show:
                break
        if len(candidate) >= n_show:
            break
    all_top_labels = candidate[:n_show]
    x = np.arange(len(all_top_labels))
    w = 0.2
    for k, mdl in enumerate(mdl_domains):
        sims = [dict(mdl_top_emotions[mdl]).get(e, 0) for e in all_top_labels]
        ax.bar(x + (k - 1.5) * w, sims, width=w,
               color=COLORS_MDL[mdl], alpha=0.85, label=mdl)
    ax.set_xticks(x); ax.set_xticklabels(all_top_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Cosine sim (vs neutral)")
    ax.set_title(f"MDL domain emotion profiles  (L{SUMMARY_LAYER})")
    ax.axhline(0, color="black", linewidth=0.7); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # Panel 6: MDL pairwise cosine distance at SUMMARY_LAYER
    ax = axes[1, 2]
    mdl_vecs   = np.stack([mdl_means[m][SUMMARY_LAYER] for m in mdl_domains])
    norms_m    = np.linalg.norm(mdl_vecs, axis=1, keepdims=True)
    mdl_vecs_n = mdl_vecs / (norms_m + 1e-9)
    dist_mat_m = 1 - mdl_vecs_n @ mdl_vecs_n.T
    im2 = ax.imshow(dist_mat_m, vmin=0, vmax=dist_mat_m.max(), cmap="YlOrRd")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{dist_mat_m[i,j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(mdl_domains, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(mdl_domains, fontsize=9)
    ax.set_title(f"MDL pairwise distance (1−cosine, L{SUMMARY_LAYER})")
    plt.colorbar(im2, ax=ax)

    plt.tight_layout()
    out_fig = RESULTS_DIR / "blame_recipient_overview.png"
    plt.savefig(out_fig, dpi=150)
    plt.close()
    print(f"\nFigure saved: {out_fig}")

    # Save full-trajectory mean activations (n_layers, D) for each condition
    for deg in range(1, 6):
        np.save(RESULTS_DIR / f"mean_acts_deg{deg}.npy", deg_means[deg])
    for mdl in mdl_domains:
        np.save(RESULTS_DIR / f"mean_acts_mdl_{mdl}.npy", mdl_means[mdl])
    np.save(RESULTS_DIR / "mean_acts_positive.npy", pos_mean)
    np.save(RESULTS_DIR / "mean_acts_neutral.npy", neutral_mean)

    print(f"\nAnalysis saved: {RESULTS_DIR / 'analysis.json'}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
