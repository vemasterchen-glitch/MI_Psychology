"""
Step 2: Extract internal activation vectors from an open-source model.

Uses TransformerLens to hook into residual stream activations at each layer.
For each emotion narrative, we record the mean activation over the prompt tokens.

Outputs:
  emotion_matrix.npy       (n_emotions, d_model)        — mean across narratives
  narrative_matrix.npy     (n_emotions, max_n, d_model) — individual narratives, zero-padded
  narrative_counts.npy     (n_emotions,)                — real narrative count per emotion
  emotion_labels.json      list of emotion strings in row order
"""

import json
from pathlib import Path

import numpy as np
import torch
import transformer_lens as tl
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "vectors"
STIMULI_FILE = DATA_DIR / "stimuli" / "narratives.jsonl"


def load_stimuli(path: Path = STIMULI_FILE) -> dict[str, list[str]]:
    stimuli: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            stimuli[row["emotion"]] = row["narratives"]
    return stimuli


CACHE_DIR = RESULTS_DIR / "cache"


def _cache_path(cache_dir: Path, emotion: str) -> Path:
    safe = emotion.replace(" ", "_").replace("/", "-")
    return cache_dir / f"{safe}.npy"


BATCH_SIZE = 8


def _masked_mean(acts: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """
    acts: (batch, seq_len, d_model)  float32
    mask: (batch, seq_len)           1=real token, 0=padding
    returns: (batch, d_model)        float32 numpy
    """
    mask_f = mask.float().unsqueeze(-1)          # (batch, seq_len, 1)
    summed = (acts.float() * mask_f).sum(1)      # (batch, d_model)
    counts = mask_f.sum(1).clamp(min=1)          # (batch, 1)
    return (summed / counts).cpu().numpy()


def extract_activation_vectors(
    model_name: str = "google/gemma-2-2b",
    layer: int | None = None,
    hook_name: str = "resid_post",
    device: str = "cpu",
    cache_dir: Path = CACHE_DIR,
    batch_size: int = BATCH_SIZE,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Returns:
      emotion_vecs   {emotion: (d_model,)}          mean across narratives
      narrative_vecs {emotion: (n_narratives, d_model)}  per-narrative activations

    Checkpointing: each emotion's array is saved to cache_dir immediately after
    extraction. On restart, already-cached emotions are skipped.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stimuli = load_stimuli()

    # Load already-cached emotions (resume support)
    emotion_vecs: dict[str, np.ndarray] = {}
    narrative_vecs: dict[str, np.ndarray] = {}
    for emotion in list(stimuli.keys()):
        p = _cache_path(cache_dir, emotion)
        if p.exists():
            arr = np.load(p)
            narrative_vecs[emotion] = arr
            emotion_vecs[emotion] = arr.mean(0)

    remaining = {e: v for e, v in stimuli.items() if e not in narrative_vecs}
    if not remaining:
        print("所有 emotion 均已缓存，跳过模型加载。")
        return emotion_vecs, narrative_vecs

    skipped = len(stimuli) - len(remaining)
    if skipped:
        print(f"断点续跑：跳过 {skipped} 个已缓存 emotion，剩余 {len(remaining)} 个。")

    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = tl.HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=dtype,
    )
    model.eval()
    # Right-padding: causal attention means real tokens are unaffected by
    # padding on the right, so activations at real positions are identical
    # to batch_size=1 results.
    model.tokenizer.padding_side = "right"
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    n_layers = model.cfg.n_layers
    target_layer = layer if layer is not None else n_layers - 1
    hook_key = f"blocks.{target_layer}.hook_{hook_name}"
    total = sum(len(v) for v in remaining.values())

    with tqdm(total=total, desc="叙述激活提取", unit="narrative") as pbar:
        for emotion, narratives in remaining.items():
            acts = []
            for i in range(0, len(narratives), batch_size):
                batch = narratives[i : i + batch_size]
                enc = model.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=True,
                )
                tokens = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=hook_key,
                    )
                batch_acts = _masked_mean(cache[hook_key], mask)  # (batch, d_model)
                acts.append(batch_acts)
                pbar.set_postfix(emotion=emotion[:15])
                pbar.update(len(batch))

            arr = np.concatenate(acts, axis=0)          # (n_narratives, d_model)
            np.save(_cache_path(cache_dir, emotion), arr)  # checkpoint
            narrative_vecs[emotion] = arr
            emotion_vecs[emotion] = arr.mean(0)         # (d_model,)

    return emotion_vecs, narrative_vecs


def save_vectors(
    emotion_vecs: dict[str, np.ndarray],
    narrative_vecs: dict[str, np.ndarray],
    out_dir: Path = RESULTS_DIR,
    stimuli_order: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Preserve original stimuli order if provided
    emotions = stimuli_order if stimuli_order else list(emotion_vecs.keys())
    d_model = next(iter(emotion_vecs.values())).shape[0]
    max_n = max(v.shape[0] for v in narrative_vecs.values())

    emotion_matrix = np.stack([emotion_vecs[e] for e in emotions])
    np.save(out_dir / "emotion_matrix.npy", emotion_matrix)

    counts = np.array([narrative_vecs[e].shape[0] for e in emotions], dtype=np.int32)
    narrative_matrix = np.zeros((len(emotions), max_n, d_model), dtype=np.float32)
    for i, e in enumerate(emotions):
        n = narrative_vecs[e].shape[0]
        narrative_matrix[i, :n] = narrative_vecs[e]
    np.save(out_dir / "narrative_matrix.npy", narrative_matrix)
    np.save(out_dir / "narrative_counts.npy", counts)

    (out_dir / "emotion_labels.json").write_text(json.dumps(emotions))

    print(f"emotion_matrix:   {emotion_matrix.shape}")
    print(f"narrative_matrix: {narrative_matrix.shape}  (max_n={max_n}, counts min={counts.min()} max={counts.max()})")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    import os
    if os.getenv("DEVICE"):
        device = os.getenv("DEVICE")
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    stimuli_order = list(load_stimuli().keys())
    emotion_vecs, narrative_vecs = extract_activation_vectors(device=device)
    save_vectors(emotion_vecs, narrative_vecs, stimuli_order=stimuli_order)
