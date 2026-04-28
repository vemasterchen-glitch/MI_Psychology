"""
监查 extract_activations 和 add_narratives 的后台进度。
用法：python scripts/monitor_extraction.py
"""

import glob
import json
import os
import re
import time
from pathlib import Path

VECTORS_DIR = Path(__file__).parent.parent / "results" / "vectors"
NARRATIVES_FILE = Path(__file__).parent.parent / "data" / "stimuli" / "narratives.jsonl"


def latest_log(pattern: str) -> Path | None:
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    return Path(files[-1]) if files else None


def parse_tqdm(log_path: Path) -> str:
    """从 log 末尾提取最新的 tqdm 进度行。"""
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="ignore")
        lines = [l for l in re.split(r"[\r\n]", tail) if l.strip()]
        for line in reversed(lines):
            if "it/s" in line or "s/it" in line or "%|" in line:
                return line.strip()
        return lines[-1].strip() if lines else "（暂无输出）"
    except Exception as e:
        return f"读取失败: {e}"


def narrative_status() -> str:
    if not NARRATIVES_FILE.exists():
        return "narratives.jsonl 不存在"
    from collections import Counter
    counts = []
    with open(NARRATIVES_FILE) as f:
        for line in f:
            d = json.loads(line)
            counts.append(len(d["narratives"]))
    c = Counter(counts)
    done = sum(1 for x in counts if x >= 20)
    return f"{done}/{len(counts)} 个情绪已达20条  分布:{dict(sorted(c.items()))}"


def vector_status() -> str:
    parts = []
    for fname in ["emotion_matrix.npy", "narrative_matrix.npy"]:
        p = VECTORS_DIR / fname
        if p.exists():
            import numpy as np
            arr = np.load(p, mmap_mode="r")
            parts.append(f"{fname}: {arr.shape}")
        else:
            parts.append(f"{fname}: 未生成")
    return "  ".join(parts)


def main():
    print("按 Ctrl+C 退出监查\n")
    while True:
        os.system("clear")
        now = time.strftime("%H:%M:%S")
        print(f"═══ 进度监查 {now} ═══\n")

        # 叙述生成
        print("【叙述生成】")
        print(f"  {narrative_status()}")
        log = latest_log("/tmp/add_narratives_*.log")
        if log:
            print(f"  log: {log}")
            print(f"  {parse_tqdm(log)}")

        print()

        # 激活提取
        print("【激活提取】")
        print(f"  {vector_status()}")
        log = latest_log("/tmp/extract_*.log")
        if log:
            print(f"  log: {log}")
            print(f"  {parse_tqdm(log)}")

        print()

        # 进程状态
        import subprocess
        procs = subprocess.run(
            ["pgrep", "-af", "python"],
            capture_output=True, text=True
        ).stdout.strip()
        running = [l for l in procs.splitlines()
                   if "extract_activations" in l or "add_narratives" in l]
        print("【后台进程】")
        if running:
            for l in running:
                print(f"  {l}")
        else:
            print("  无相关进程在跑")

        time.sleep(10)


if __name__ == "__main__":
    main()
