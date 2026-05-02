"""
Shared helpers: text cleaning, keyword extraction, and light NLP utilities.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

import regex as reg

# English stopwords — small built-in list to avoid extra NLTK dependency.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "as", "is",
    "was", "are", "were", "been", "be", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just", "also", "now",
    "here", "there", "then", "once", "from", "by", "with", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "under", "again", "further",
    "once", "watched", "video", "videos", "vlog", "episode", "part", "full",
})


def remove_emojis(text: str) -> str:
    """
    Remove emoji and most pictographic symbols using Unicode properties.
    """
    if not text:
        return text
    # \p{Extended_Pictographic} plus common emoji ranges
    pattern = reg.compile(
        r"[\p{Extended_Pictographic}\U0001F300-\U0001FAFF\U00002600-\U000027BF]+",
        flags=reg.UNICODE,
    )
    return pattern.sub("", text)


def clean_watch_title(title: str) -> str:
    """
    Normalize a YouTube watch-history title: strip noise prefixes and emojis.

    Takeout often stores titles like 'Watched <actual title>'.
    """
    if not title:
        return ""
    t = title.strip()
    # Remove leading "Watched" / "Visited" style prefixes (case-insensitive)
    t = re.sub(r"^(watched|visited)\s*[:\-]?\s*", "", t, flags=re.IGNORECASE).strip()
    # Remove common history/export noise around recommendation actions.
    t = re.sub(
        r"\b(from|on)\s+(youtube|youtube music|youtube kids)\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\b(shared|recommended|suggested|playlist|mix|shorts?)\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = remove_emojis(t)
    # Remove stray punctuation left behind after cleanup, while keeping common title separators.
    t = re.sub(r"[^\w\s\-:|&'/.,!?()]+", " ", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_for_keywords(text: str) -> List[str]:
    """Lowercase alphanumeric tokens of length >= 2."""
    if not text:
        return []
    raw = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in raw if len(w) >= 2 and w not in _STOPWORDS]


def top_keywords_from_texts(texts: Iterable[str], top_k: int = 5) -> List[str]:
    """
    Return the most common content words across a collection of titles/labels.
    """
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize_for_keywords(t))
    return [w for w, _ in counter.most_common(top_k)]


def hour_bucket_label(hour: int) -> str:
    """Map 0-23 hour to a simple behavioral bucket."""
    if 0 <= hour < 5:
        return "late_night"
    if 5 <= hour < 9:
        return "early_morning"
    if 9 <= hour < 12:
        return "morning"
    if 12 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 21:
        return "evening"
    return "night"
