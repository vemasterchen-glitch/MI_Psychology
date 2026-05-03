"""Build an unlabeled public-domain stimulus corpus.

The script downloads public-domain texts, extracts short original passages, and
writes JSONL + CSV files with source metadata kept separate from stimulus text.
"""

from __future__ import annotations

import csv
import html
import json
import re
import subprocess
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote


OUT_JSONL = Path("data/stimuli/self_consciousness_corpus.jsonl")
OUT_CSV = Path("data/stimuli/self_consciousness_corpus.csv")
CACHE_DIR = Path("data/raw/public_domain_corpus_cache")

TARGET_TOTAL = 400
NEUTRAL_TARGET = 100
MAX_PER_AUTHOR = 20
MAX_PER_WORK = 14
LANGUAGE_TARGETS = {"en": 200, "fr": 44, "de": 40, "zh": 56}

TARGET_TERMS = [
    "i am",
    "i exist",
    "i think",
    "myself",
    "self",
    "soul",
    "mind",
    "consciousness",
    "memory",
    "recollection",
    "remember",
    "within me",
    "inner",
    "dream",
    "illusion",
    "perception",
    "body",
    "my body",
    "i see",
    "i perceive",
    "mirror",
    "eyes",
    "spirit",
    "being",
    "existence",
    "thought",
    "identity",
    "person",
    "conscience",
    "âme",
    "moi",
    "je suis",
    "je pense",
    "souvenir",
    "rêve",
    "esprit",
    "conscience",
    "selbst",
    "ich bin",
    "seele",
    "geist",
    "traum",
    "gedächtnis",
    "bewußtsein",
    "bewusstsein",
    "吾",
    "我",
    "心",
    "夢",
    "身",
    "形",
    "神",
    "知",
    "self",
]

NEUTRAL_TERMS = [
    "river",
    "mountain",
    "tree",
    "trees",
    "rain",
    "wind",
    "sea",
    "stone",
    "garden",
    "field",
    "flowers",
    "birds",
    "horse",
    "house",
    "street",
    "window",
    "weather",
    "sky",
    "sun",
    "moon",
    "forest",
    "cloud",
    "water",
    "air",
    "earth",
    "ville",
    "mer",
    "arbre",
    "fleur",
    "ciel",
    "soleil",
    "eau",
    "vent",
    "berg",
    "wald",
    "baum",
    "wasser",
    "himmel",
    "sonne",
    "wind",
    "山",
    "水",
    "風",
    "雨",
    "木",
    "草",
    "鳥",
    "日",
    "月",
    "河",
]

BAD_PATTERNS = [
    "project gutenberg",
    "gutenberg",
    "copyright",
    "ebook",
    "transcriber",
    "proofread",
    "contents",
    "chapter",
    "book ",
    "volume",
    "footnote",
    "wikisource",
    "category:",
    "file:",
    "http://",
    "https://",
    "{{",
    "}}",
    "[[",
    "]]",
]


SOURCES = [
    # English originals and English public-domain translations
    {"kind": "gutenberg", "id": 59, "language": "en", "author": "René Descartes", "work": "Discourse on the Method", "year_or_period": "1637; English translation 1912", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 23306, "language": "en", "author": "René Descartes", "work": "Meditations on First Philosophy", "year_or_period": "1641; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 3296, "language": "en", "author": "Augustine of Hippo", "work": "Confessions", "year_or_period": "c. 397-400; English translation public-domain", "genre": "autobiography"},
    {"kind": "gutenberg", "id": 2680, "language": "en", "author": "Marcus Aurelius", "work": "Meditations", "year_or_period": "2nd century; English translation 1862", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 45109, "language": "en", "author": "Epictetus", "work": "The Discourses of Epictetus", "year_or_period": "2nd century; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 3600, "language": "en", "author": "Michel de Montaigne", "work": "Essays", "year_or_period": "1580-1595; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 18269, "language": "en", "author": "Blaise Pascal", "work": "Pascal's Pensées", "year_or_period": "1670; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 10615, "language": "en", "author": "John Locke", "work": "An Essay Concerning Human Understanding", "year_or_period": "1689", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 4705, "language": "en", "author": "David Hume", "work": "A Treatise of Human Nature", "year_or_period": "1739-1740", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 4363, "language": "en", "author": "Friedrich Nietzsche", "work": "Beyond Good and Evil", "year_or_period": "1886; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 1998, "language": "en", "author": "Friedrich Nietzsche", "work": "Thus Spake Zarathustra", "year_or_period": "1883-1885; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 38145, "language": "en", "author": "Friedrich Nietzsche", "work": "Human, All Too Human", "year_or_period": "1878; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 10715, "language": "en", "author": "Arthur Schopenhauer", "work": "The Essays of Arthur Schopenhauer; Studies in Pessimism", "year_or_period": "1851; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 16643, "language": "en", "author": "Ralph Waldo Emerson", "work": "Essays", "year_or_period": "1841-1844", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 1322, "language": "en", "author": "Walt Whitman", "work": "Leaves of Grass", "year_or_period": "1855-1892", "genre": "poetry"},
    {"kind": "gutenberg", "id": 12242, "language": "en", "author": "Emily Dickinson", "work": "Poems by Emily Dickinson, Series One", "year_or_period": "1890", "genre": "poetry"},
    {"kind": "gutenberg", "id": 12383, "language": "en", "author": "William Wordsworth", "work": "The Prelude", "year_or_period": "1850", "genre": "poetry"},
    {"kind": "gutenberg", "id": 1524, "language": "en", "author": "William Shakespeare", "work": "Hamlet", "year_or_period": "c. 1600", "genre": "drama"},
    {"kind": "gutenberg", "id": 1533, "language": "en", "author": "William Shakespeare", "work": "Macbeth", "year_or_period": "c. 1606", "genre": "drama"},
    {"kind": "gutenberg", "id": 57628, "language": "en", "author": "William James", "work": "The Principles of Psychology, Volume 1", "year_or_period": "1890", "genre": "psychology"},
    {"kind": "gutenberg", "id": 57634, "language": "en", "author": "William James", "work": "The Principles of Psychology, Volume 2", "year_or_period": "1890", "genre": "psychology"},
    {"kind": "gutenberg", "id": 41028, "language": "en", "author": "Sigmund Freud", "work": "Dream Psychology", "year_or_period": "1920; English translation public-domain in US", "genre": "psychology"},
    {"kind": "gutenberg", "id": 1497, "language": "en", "author": "Plato", "work": "The Republic", "year_or_period": "c. 375 BCE; English translation 19th century", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 1635, "language": "en", "author": "Plato", "work": "The Apology", "year_or_period": "4th century BCE; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 1726, "language": "en", "author": "Aristotle", "work": "The Poetics", "year_or_period": "c. 335 BCE; English translation public-domain", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 2388, "language": "en", "author": "Plotinus", "work": "The Six Enneads", "year_or_period": "3rd century; English translation 1917", "genre": "religious"},
    {"kind": "gutenberg", "id": 7163, "language": "en", "author": "Anonymous", "work": "The Upanishads", "year_or_period": "ancient; English translation public-domain", "genre": "religious"},
    {"kind": "gutenberg", "id": 2017, "language": "en", "author": "Anonymous", "work": "The Dhammapada", "year_or_period": "ancient; English translation public-domain", "genre": "religious"},
    {"kind": "gutenberg", "id": 409, "language": "en", "author": "Thomas De Quincey", "work": "Confessions of an English Opium-Eater", "year_or_period": "1821", "genre": "autobiography"},
    {"kind": "gutenberg", "id": 3913, "language": "en", "author": "Jean-Jacques Rousseau", "work": "The Confessions", "year_or_period": "1782; English translation public-domain", "genre": "autobiography"},
    # French originals
    {"kind": "gutenberg", "id": 13846, "language": "fr", "author": "René Descartes", "work": "Discours de la méthode", "year_or_period": "1637", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 48529, "language": "fr", "author": "Michel de Montaigne", "work": "Essais de Montaigne, Volume I", "year_or_period": "1580", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 65434, "language": "fr", "author": "Jean-Jacques Rousseau", "work": "Les Rêveries du Promeneur Solitaire", "year_or_period": "1782", "genre": "autobiography"},
    {"kind": "gutenberg", "id": 56668, "language": "fr", "author": "Arthur Rimbaud", "work": "Une saison en enfer", "year_or_period": "1873", "genre": "poetry"},
    {"kind": "gutenberg", "id": 6099, "language": "fr", "author": "Charles Baudelaire", "work": "Les Fleurs du Mal", "year_or_period": "1857", "genre": "poetry"},
    # German originals
    {"kind": "gutenberg", "id": 7205, "language": "de", "author": "Friedrich Nietzsche", "work": "Also sprach Zarathustra", "year_or_period": "1883-1885", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 7204, "language": "de", "author": "Friedrich Nietzsche", "work": "Jenseits von Gut und Böse", "year_or_period": "1886", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 7207, "language": "de", "author": "Friedrich Nietzsche", "work": "Menschliches, Allzumenschliches", "year_or_period": "1878", "genre": "philosophy"},
    {"kind": "gutenberg", "id": 24288, "language": "de", "author": "Rainer Maria Rilke", "work": "Das Stunden-Buch", "year_or_period": "1905", "genre": "poetry"},
    {"kind": "gutenberg", "id": 43821, "language": "de", "author": "Novalis", "work": "Hymnen an die Nacht", "year_or_period": "1800", "genre": "poetry"},
    {"kind": "wikisource", "language": "de", "author": "Johann Wolfgang von Goethe", "work": "Faust. Der Tragödie erster Teil", "year_or_period": "1808", "genre": "drama", "url": "https://de.wikisource.org/wiki/Faust_-_Der_Trag%C3%B6die_erster_Teil?action=raw"},
    # Classical Chinese originals
    {"kind": "wikisource", "language": "zh", "author": "Zhuangzi", "work": "Zhuangzi, Qiwulun", "year_or_period": "Warring States period", "genre": "philosophy", "url": "https://zh.wikisource.org/wiki/%E8%8E%8A%E5%AD%90/%E9%BD%8A%E7%89%A9%E8%AB%96?action=raw"},
    {"kind": "wikisource", "language": "zh", "author": "Zhuangzi", "work": "Zhuangzi, Xiaoyaoyou", "year_or_period": "Warring States period", "genre": "philosophy", "url": "https://zh.wikisource.org/wiki/%E8%8E%8A%E5%AD%90/%E9%80%8D%E9%81%99%E9%81%8A?action=raw"},
    {"kind": "wikisource", "language": "zh", "author": "Laozi", "work": "Daodejing, Wang Bi recension", "year_or_period": "classical China", "genre": "philosophy", "url": "https://zh.wikisource.org/wiki/%E9%81%93%E5%BE%B7%E7%B6%93_(%E7%8E%8B%E5%BC%BC%E6%9C%AC)?action=raw"},
    {"kind": "wikisource", "language": "zh", "author": "Confucius and disciples", "work": "Analects, Xue Er", "year_or_period": "Warring States period", "genre": "philosophy", "url": "https://zh.wikisource.org/wiki/%E8%AB%96%E8%AA%9E/%E5%AD%B8%E8%80%8C%E7%AC%AC%E4%B8%80?action=raw"},
    {"kind": "wikisource", "language": "zh", "author": "Mencius", "work": "Mencius, Gaozi I", "year_or_period": "Warring States period", "genre": "philosophy", "url": "https://zh.wikisource.org/wiki/%E5%AD%9F%E5%AD%90/%E5%91%8A%E5%AD%90%E4%B8%8A?action=raw"},
    # Source-diverse neutral comparison material
    {"kind": "gutenberg", "id": 205, "language": "en", "author": "Henry David Thoreau", "work": "Walden", "year_or_period": "1854", "genre": "autobiography", "neutral": True},
    {"kind": "gutenberg", "id": 1228, "language": "en", "author": "Charles Darwin", "work": "On the Origin of Species", "year_or_period": "1859", "genre": "science", "neutral": True},
    {"kind": "gutenberg", "id": 1404, "language": "en", "author": "Gilbert White", "work": "The Natural History of Selborne", "year_or_period": "1789", "genre": "natural_history", "neutral": True},
    {"kind": "gutenberg", "id": 20417, "language": "en", "author": "John Burroughs", "work": "Wake-Robin", "year_or_period": "1871", "genre": "natural_history", "neutral": True},
    {"kind": "gutenberg", "id": 1342, "language": "en", "author": "Jane Austen", "work": "Pride and Prejudice", "year_or_period": "1813", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 98, "language": "en", "author": "Charles Dickens", "work": "A Tale of Two Cities", "year_or_period": "1859", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 2641, "language": "en", "author": "Rudyard Kipling", "work": "The Jungle Book", "year_or_period": "1894", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 120, "language": "en", "author": "Robert Louis Stevenson", "work": "Treasure Island", "year_or_period": "1883", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 2701, "language": "en", "author": "Herman Melville", "work": "Moby-Dick", "year_or_period": "1851", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 113, "language": "en", "author": "Frances Hodgson Burnett", "work": "The Secret Garden", "year_or_period": "1911", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 11, "language": "en", "author": "Lewis Carroll", "work": "Alice's Adventures in Wonderland", "year_or_period": "1865", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 289, "language": "en", "author": "Kenneth Grahame", "work": "The Wind in the Willows", "year_or_period": "1908", "genre": "prose", "neutral": True},
    {"kind": "gutenberg", "id": 55, "language": "en", "author": "L. Frank Baum", "work": "The Wonderful Wizard of Oz", "year_or_period": "1900", "genre": "prose", "neutral": True},
]


def fetch(url: str, cache_key: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{cache_key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    proc = subprocess.run(
        [
            "curl",
            "-L",
            "--max-time",
            "35",
            "--connect-timeout",
            "10",
            "--silent",
            "--show-error",
            "-A",
            "emotion-mi-corpus-builder/0.1",
            url,
        ],
        check=True,
        capture_output=True,
    )
    raw = proc.stdout
    text = raw.decode("utf-8", errors="ignore")
    if len(text.strip()) < 100:
        raise RuntimeError(f"empty or short response from {url}")
    path.write_text(text, encoding="utf-8")
    time.sleep(0.2)
    return text


def gutenberg_text_url(book_id: int) -> str:
    api = f"https://gutendex.com/books/{book_id}"
    data = json.loads(fetch(api, f"gutendex_{book_id}"))
    formats = data.get("formats", {})
    for key, value in formats.items():
        if "text/plain" in key and value:
            return value
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def clean_text(text: str, kind: str) -> str:
    text = html.unescape(text.replace("\ufeff", ""))
    if kind == "gutenberg":
        text = re.sub(r"(?is)^.*?\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", "", text)
        text = re.sub(r"(?is)\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", "", text)
    text = re.sub(r"(?m)^\s*\{\{.*?\}\}\s*$", "", text)
    text = re.sub(r"\{\{[^{}\n]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"(?m)^\s*[|!].*$", "", text)
    text = re.sub(r"(?m)^={2,}.*?={2,}\s*$", "", text)
    text = re.sub(r"(?m)^\s*[:*#;].*$", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(paragraph: str, language: str) -> list[str]:
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    if language == "zh":
        parts = re.split(r"(?<=[。！？；])", paragraph)
    else:
        parts = re.split(r"(?<=[.!?;:])\s+", paragraph)
    return [p.strip(" \t\n\"“”") for p in parts if p.strip(" \t\n\"“”")]


def word_count(text: str, language: str) -> int:
    if language == "zh":
        return len(re.findall(r"[\u4e00-\u9fff]", text))
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def clean_passage(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("'''", "")
    text = re.sub(r"\{([^{}\n]{1,8})\}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" -—–\t\n")
    return text


def usable(text: str, language: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if any(pattern in lower for pattern in BAD_PATTERNS):
        return False
    if re.search(r"[_*]{2,}|={2,}|<|>|\|", text):
        return False
    if len(text) < 30 or len(text) > 850:
        return False
    words = word_count(text, language)
    if language == "zh":
        if not 12 <= words <= 160:
            return False
    elif not 15 <= words <= 120:
        return False
    letters = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", text)
    if letters and sum(1 for c in letters if c.isupper()) / len(letters) > 0.42:
        return False
    return True


def score(text: str, source: dict, terms: list[str]) -> int:
    haystack = text if source["language"] == "zh" else text.lower()
    return sum(1 for term in terms if term in haystack)


def candidate_passages(text: str, source: dict) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    candidates: list[str] = []
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s*\[[^\]]{1,20}\]\s*", " ", paragraph)
        if len(paragraph) < 30:
            continue
        sentences = split_sentences(paragraph, source["language"])
        if not sentences:
            continue
        for width in (1, 2, 3, 4):
            for i in range(0, max(0, len(sentences) - width + 1)):
                passage = clean_passage(" ".join(sentences[i : i + width]))
                if usable(passage, source["language"]):
                    candidates.append(passage)
    return candidates


def source_text(source: dict) -> tuple[str, str]:
    if source["kind"] == "gutenberg":
        url = gutenberg_text_url(source["id"])
        try:
            return clean_text(fetch(url, f"gutenberg_{source['id']}"), "gutenberg"), url
        except Exception:
            fallback_url = f"https://www.gutenberg.org/cache/epub/{source['id']}/pg{source['id']}.txt"
            return (
                clean_text(fetch(fallback_url, f"gutenberg_{source['id']}_cache"), "gutenberg"),
                fallback_url,
            )
    url = source["url"]
    safe = quote(url, safe="")
    return clean_text(fetch(url, f"wikisource_{safe[:120]}"), "wikisource"), url


def source_quota(source: dict) -> int:
    if source.get("neutral"):
        return 16
    if source["language"] == "zh":
        return 12
    if source["language"] in {"de", "fr"}:
        return 10
    return 8


def build() -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    seen_norms: list[str] = []
    author_counts: Counter[str] = Counter()
    work_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    neutral_count = 0

    source_candidates: list[tuple[int, str, dict, str]] = []
    for source in SOURCES:
        print(f"source: {source['language']} {source['author']} / {source['work']}", flush=True)
        try:
            text, source_url = source_text(source)
        except Exception as exc:
            print(f"skip {source['author']} / {source['work']}: {exc}")
            continue
        terms = NEUTRAL_TERMS if source.get("neutral") else TARGET_TERMS
        for passage in candidate_passages(text, source):
            s = score(passage, source, terms)
            if source.get("neutral"):
                if s == 0:
                    continue
                s = max(1, s)
            elif s == 0:
                continue
            source_candidates.append((s, passage, source, source_url))

    grouped: defaultdict[tuple[str, str], list[tuple[int, str, dict, str]]] = defaultdict(list)
    for item in source_candidates:
        _, _, source, _ = item
        grouped[(source["author"], source["work"])].append(item)

    ordered: list[tuple[int, str, dict, str]] = []
    for key, items in grouped.items():
        items.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
        source = items[0][2]
        ordered.extend(items[: source_quota(source)])

    # Interleave by language/source to avoid one-book blocks.
    ordered.sort(
        key=lambda x: (
            not x[2].get("neutral", False),
            x[2]["language"],
            x[2]["author"],
            x[2]["work"],
            -x[0],
            len(x[1]),
        )
    )

    def add_item(passage: str, source: dict, source_url: str) -> None:
        nonlocal neutral_count
        norm = re.sub(r"\W+", "", passage.lower())
        if norm in seen:
            return
        if any(norm in existing or existing in norm for existing in seen_norms):
            return
        if author_counts[source["author"]] >= MAX_PER_AUTHOR:
            return
        if work_counts[source["work"]] >= MAX_PER_WORK:
            return
        if language_counts[source["language"]] >= LANGUAGE_TARGETS.get(source["language"], TARGET_TOTAL):
            return
        if source.get("neutral") and neutral_count >= NEUTRAL_TARGET:
            return
        seen.add(norm)
        seen_norms.append(norm)
        author_counts[source["author"]] += 1
        work_counts[source["work"]] += 1
        language_counts[source["language"]] += 1
        if source.get("neutral"):
            neutral_count += 1
        rows.append(
            {
                "id": f"S{len(rows) + 1:04d}",
                "stimulus_text": passage,
                "language": source["language"],
                "author": source["author"],
                "work": source["work"],
                "year_or_period": source["year_or_period"],
                "genre": source["genre"],
                "source_url": source_url,
                "public_domain_status": "likely_public_domain",
                "notes": "source text retrieved from public-domain repository; passage selected automatically by length and lexical filters",
            }
        )

    for _, passage, source, source_url in ordered:
        add_item(passage, source, source_url)

    # Second pass relaxes source quota but keeps author/work caps.
    source_candidates.sort(
        key=lambda x: (
            not x[2].get("neutral", False),
            -x[0],
            x[2]["language"],
            x[2]["author"],
            len(x[1]),
        )
    )
    for _, passage, source, source_url in source_candidates:
        if len(rows) >= TARGET_TOTAL:
            break
        add_item(passage, source, source_url)

    for i, row in enumerate(rows, start=1):
        row["id"] = f"S{i:04d}"
    return rows


def write_outputs(rows: list[dict]) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "stimulus_text",
        "language",
        "author",
        "work",
        "year_or_period",
        "genre",
        "source_url",
        "public_domain_status",
        "notes",
    ]
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = build()
    write_outputs(rows)
    print(f"wrote {len(rows)} rows")
    print("languages:", dict(Counter(row["language"] for row in rows)))
    neutral_works = {source["work"] for source in SOURCES if source.get("neutral")}
    print("neutral-source rows:", sum(1 for row in rows if row["work"] in neutral_works))
    print("authors over cap:", {k: v for k, v in Counter(row["author"] for row in rows).items() if v > MAX_PER_AUTHOR})
    print("outputs:", OUT_JSONL, OUT_CSV)


if __name__ == "__main__":
    main()
