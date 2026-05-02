"""
Shared helpers: text cleaning, time/creator aggregations, light NLP utilities.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import regex as reg

logger = logging.getLogger(__name__)

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
    """Remove emoji and most pictographic symbols using Unicode properties."""
    if not text:
        return text
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
    t = re.sub(r"^(watched|visited)\s*[:\-]?\s*", "", t, flags=re.IGNORECASE).strip()
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
    t = re.sub(r"[^\w\s\-:|&'/.,!?()]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_for_keywords(text: str) -> List[str]:
    """Lowercase alphanumeric tokens of length >= 2."""
    if not text:
        return []
    raw = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in raw if len(w) >= 2 and w not in _STOPWORDS]


def top_keywords_from_texts(texts: Iterable[str], top_k: int = 5) -> List[str]:
    """Return the most common content words across a collection of titles."""
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


def get_time_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregate watch timing for API and downstream narrative modules.

    Returns:
        * ``peak_hour`` — int 0–23 or None
        * ``peak_day`` — weekday name with most watches
        * ``late_night_percentage`` — share of watches between midnight and 5am (0–100)
        * ``most_active_month`` — ``YYYY-MM`` with highest count
        Plus extended histograms used by ``analyst`` / compatibility.
    """
    empty: Dict[str, Any] = {
        "peak_hour": None,
        "peak_day": None,
        "late_night_percentage": 0.0,
        "most_active_month": None,
        "hourly_counts": {},
        "day_of_week_counts": {},
        "month_counts": {},
        "time_bucket_counts": {},
        "late_night_ratio": 0.0,
    }
    if df is None or df.empty:
        return empty

    if not {"hour", "day_of_week", "month"}.issubset(df.columns):
        logger.error("get_time_patterns: DataFrame missing hour/day_of_week/month columns")
        return empty

    n = len(df)
    hour_series = df["hour"]
    hourly = hour_series.value_counts().sort_index()
    hourly_counts = {int(h): int(c) for h, c in hourly.items()}
    peak_hour = int(hour_series.mode().iloc[0]) if n else None

    dow = df["day_of_week"].value_counts()
    dow_counts = {str(d): int(c) for d, c in dow.items()}
    peak_day = str(dow.index[0]) if not dow.empty else None

    late_mask = (df["hour"] >= 0) & (df["hour"] < 5)
    late_night_ratio = float(late_mask.mean()) if n else 0.0
    late_night_pct = round(100.0 * late_night_ratio, 2)

    month_counts_s = df["month"].value_counts().sort_index()
    month_counts = {str(m): int(c) for m, c in month_counts_s.items()}
    most_active_month = str(month_counts_s.idxmax()) if not month_counts_s.empty else None

    bucket_counts: Dict[str, int] = Counter()
    for h in hour_series:
        bucket_counts[hour_bucket_label(int(h))] += 1

    return {
        "peak_hour": peak_hour,
        "peak_day": peak_day,
        "late_night_percentage": late_night_pct,
        "most_active_month": most_active_month,
        "hourly_counts": {str(k): v for k, v in sorted(hourly_counts.items())},
        "day_of_week_counts": dow_counts,
        "month_counts": month_counts,
        "time_bucket_counts": dict(bucket_counts),
        "late_night_ratio": round(late_night_ratio, 4),
    }


def get_top_creators(df: pd.DataFrame, n: int = 10) -> List[Tuple[str, int]]:
    """Return ``(channel, watch_count)`` tuples sorted descending."""
    if df is None or df.empty or "channel" not in df.columns:
        return []
    counts = df["channel"].value_counts().head(n)
    return [(str(name), int(ct)) for name, ct in counts.items()]


def get_monthly_trends(df: pd.DataFrame) -> Dict[str, int]:
    """Map ``YYYY-MM`` → number of watches, chronologically ordered keys in JSON via sort."""
    if df is None or df.empty or "month" not in df.columns:
        return {}
    s = df["month"].value_counts().sort_index()
    return {str(m): int(c) for m, c in s.items()}
