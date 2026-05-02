"""
Persistent user corrections for video categories: pattern overrides + learned keyword additions.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

CLASSIFICATION_LABELS: Tuple[str, ...] = (
    "technology",
    "music",
    "education",
    "gaming",
    "entertainment",
    "cooking",
    "finance",
    "sports",
)

# Exported taxonomy label -> keyword bucket in model.CATEGORY_KEYWORDS
LABEL_TO_KEYWORD_BUCKET: Dict[str, str] = {
    "technology": "technology",
    "music": "music",
    "education": "education",
    "gaming": "gaming",
    "entertainment": "entertainment",
    "cooking": "food",
    "finance": "finance",
    "sports": "fitness",
}

_RETRAIN_EVERY_N = 20


def _cache_base() -> Path:
    root = os.environ.get("YOUTUBE_WRAPPED_CACHE_DIR")
    if root:
        p = Path(root)
    else:
        p = Path(__file__).resolve().parent.parent / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def feedback_file_paths() -> Tuple[Path, Path]:
    base = _cache_base()
    return base / "feedback.json", base / "learned_category_keywords.json"


def normalize_label(cat: str | None) -> str:
    if not cat:
        return "entertainment"
    c = str(cat).strip().lower()
    if c in CLASSIFICATION_LABELS:
        return c
    aliases = {
        "tech": "technology",
        "food": "cooking",
        "sport": "sports",
        "financial": "finance",
        "game": "gaming",
    }
    return aliases.get(c, "entertainment")


def _load_store() -> Dict[str, Any]:
    path, _ = feedback_file_paths()
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "patterns" in data:
                return data
            # Legacy flat pattern map
            if isinstance(data, dict) and data:
                return {"patterns": data, "corrections": []}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load feedback store: %s", exc)
    return {"patterns": {}, "corrections": []}


def load_feedback() -> Dict[str, str]:
    """Returns ``pattern -> corrected_category`` (classification label)."""
    return dict(_load_store().get("patterns") or {})


def _load_learned_keywords() -> Dict[str, List[str]]:
    _, path = feedback_file_paths()
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: list(v) for k, v in data.items() if isinstance(v, list)}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not load learned keywords: %s", exc)
    return {}


def _save_learned_keywords(merged: Dict[str, List[str]]) -> None:
    _, path = feedback_file_paths()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)


def get_effective_category_keywords(
    base: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Deep merge of base keyword map with persisted learned bucket extras."""
    learned = _load_learned_keywords()
    out: Dict[str, List[str]] = {k: list(v) for k, v in base.items()}
    for bucket, words in learned.items():
        if bucket not in out:
            out[bucket] = []
        seen = set(out[bucket])
        for w in words:
            w = str(w).strip().lower()
            if w and w not in seen:
                out[bucket].append(w)
                seen.add(w)
    return out


def _title_word_tokens(title: str) -> List[str]:
    return [w.lower() for w in re.findall(r"\b\w{4,}\b", title.lower())]


def _append_correction_keywords(title: str, correct_label: str, base_kws: Dict[str, List[str]]) -> None:
    bucket = LABEL_TO_KEYWORD_BUCKET.get(correct_label, "entertainment")
    learned = _load_learned_keywords()
    base_set = set(base_kws.get(bucket, []))
    words = _title_word_tokens(title)
    cur = list(learned.get(bucket, []))
    seen = set(cur)
    for w in words:
        if w in base_set or w in seen:
            continue
        cur.append(w)
        seen.add(w)
    learned[bucket] = cur
    _save_learned_keywords(learned)


def retrain_keywords(
    df: pd.DataFrame,
    *,
    purity: float = 0.8,
    min_count: int = 5,
) -> Dict[str, List[str]]:
    """
    Find words strongly associated with one category (label) in ``df`` (columns: title, category).
    Returns label -> suggested words (caller maps labels to buckets for persistence).
    """
    if df is None or df.empty or "title" not in df.columns:
        return {}
    small = len(df) < 50
    if small:
        purity = min(purity, 0.6)
        min_count = min(min_count, 3)

    word_to_category: Dict[str, Counter] = defaultdict(Counter)
    for row in df.itertuples(index=False):
        title = str(getattr(row, "title", "") or "")
        category = normalize_label(getattr(row, "category", None))
        if not category:
            continue
        for w in _title_word_tokens(title):
            word_to_category[w][category] += 1

    new_by_label: Dict[str, List[str]] = defaultdict(list)
    for word, cat_counts in word_to_category.items():
        total = sum(cat_counts.values())
        if total < min_count:
            continue
        top_cat, top_count = cat_counts.most_common(1)[0]
        if total and top_count / total >= purity:
            new_by_label[top_cat].append(word)

    return {k: sorted(set(v)) for k, v in new_by_label.items()}


def _merge_retrain_labels_into_learned(
    new_by_label: Dict[str, List[str]],
    base_kws: Dict[str, List[str]],
) -> None:
    learned = _load_learned_keywords()
    for label, words in new_by_label.items():
        bucket = LABEL_TO_KEYWORD_BUCKET.get(label)
        if not bucket:
            continue
        base_set = set(base_kws.get(bucket, []))
        cur = list(learned.get(bucket, []))
        seen = set(cur)
        for w in words:
            w = str(w).strip().lower()
            if not w or w in base_set or w in seen:
                continue
            cur.append(w)
            seen.add(w)
        learned[bucket] = cur
    _save_learned_keywords(learned)


def _maybe_retrain_after_save(store: Dict[str, Any], base_kws: Dict[str, List[str]]) -> None:
    corrections = store.get("corrections") or []
    if len(corrections) < _RETRAIN_EVERY_N:
        return
    if len(corrections) % _RETRAIN_EVERY_N != 0:
        return
    df = pd.DataFrame(
        [
            {"title": c.get("title", ""), "category": normalize_label(c.get("correct_category"))}
            for c in corrections
            if c.get("title")
        ]
    )
    if df.empty:
        return
    suggestions = retrain_keywords(df)
    if suggestions:
        _merge_retrain_labels_into_learned(suggestions, base_kws)
        logger.info("Retrained keywords after %d corrections: %s", len(corrections), suggestions)


def save_feedback(title: str, wrong_category: str, correct_category: str, base_kws: Dict[str, List[str]]) -> None:
    """Record a correction: title word patterns + optional batch retrain."""
    title = (title or "").strip()
    if not title:
        raise ValueError("title required")
    correct = normalize_label(correct_category)
    path, _ = feedback_file_paths()
    store = _load_store()
    patterns: Dict[str, str] = dict(store.get("patterns") or {})

    words = [w.lower() for w in title.split() if len(w) > 3]
    for w in words:
        patterns[w] = correct

    corrections: List[Dict[str, Any]] = list(store.get("corrections") or [])
    corrections.append(
        {
            "title": title,
            "wrong_category": str(wrong_category or ""),
            "correct_category": correct,
        }
    )
    store["patterns"] = patterns
    store["corrections"] = corrections

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)

    _append_correction_keywords(title, correct, base_kws)
    _maybe_retrain_after_save(store, base_kws)


def classify_with_feedback(
    title: str,
    base_classifier: Callable[[str], Tuple[str, float]],
) -> Tuple[str, float]:
    patterns = load_feedback()
    if not patterns:
        return base_classifier(title)
    title_lower = title.lower()
    for pattern in sorted(patterns.keys(), key=len, reverse=True):
        if pattern and pattern in title_lower:
            lab = normalize_label(patterns[pattern])
            return lab, 0.95
    return base_classifier(title)


def correction_count() -> int:
    return len(_load_store().get("corrections") or [])
