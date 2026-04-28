"""
增量生成叙述：在现有条数基础上补充到目标数量。
用法：python scripts/add_narratives.py --target 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error as url_error

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
load_dotenv(Path(__file__).parent.parent / ".env")

STIMULI_FILE = Path(__file__).parent.parent / "data" / "stimuli" / "narratives.jsonl"

NARRATIVE_PROMPT = """\
Write a short narrative (3-5 sentences) about a character who is experiencing \
the emotion of {emotion}. Make the emotional state vivid and central to the scene. \
Focus on the internal experience. Do not name the emotion explicitly.\
"""


def call_qwen(prompt: str, model: str, api_key: str, retries: int = 3) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "enable_thinking": False,
    }).encode()
    for attempt in range(retries):
        try:
            req = request.Request(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                data=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"].strip()
        except (TimeoutError, url_error.URLError, OSError) as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                tqdm.write(f"  重试 {attempt + 1}/{retries - 1}（{e}），等待 {wait}s...")
                time.sleep(wait)
            else:
                raise


def save_all(updated: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for emotion, narratives in updated.items():
            f.write(json.dumps({"emotion": emotion, "narratives": narratives}) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=20, help="目标叙述数量")
    args = parser.parse_args()

    api_key = os.environ["DASHSCOPE_API_KEY"]
    model = os.getenv("QWEN_MODEL", "qwen3.6-flash")

    # 加载现有叙述
    existing: dict[str, list[str]] = {}
    if STIMULI_FILE.exists():
        with open(STIMULI_FILE) as f:
            for line in f:
                d = json.loads(line)
                existing[d["emotion"]] = d["narratives"]

    # 找出需要补充的情绪
    to_add = {e: args.target - len(n) for e, n in existing.items() if len(n) < args.target}
    total_calls = sum(to_add.values())
    print(f"需要补充的情绪：{len(to_add)} 个，共 {total_calls} 次 API 调用")

    updated = dict(existing)
    with tqdm(total=total_calls, desc="生成叙述") as pbar:
        for emotion, n_needed in to_add.items():
            for _ in range(n_needed):
                text = call_qwen(NARRATIVE_PROMPT.format(emotion=emotion), model, api_key)
                updated[emotion].append(text)
                pbar.update(1)
            # 每完成一个情绪立即持久化，防止崩溃丢失进度
            save_all(updated, STIMULI_FILE)

    counts = set(len(n) for n in updated.values())
    print(f"完成。每情绪叙述数：{counts}")


if __name__ == "__main__":
    main()
