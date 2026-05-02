"""
Load Google Takeout YouTube watch-history JSON into a normalized pandas DataFrame.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser as date_parser

from youtube_wrapped.utils import clean_watch_title

logger = logging.getLogger(__name__)


def _extract_channel(entry: Dict[str, Any]) -> str:
    """Best-effort channel name from Takeout ``subtitles`` / ``details`` fields."""
    for field in ("subtitles", "details"):
        values = entry.get(field)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            for key in ("name", "title", "text"):
                value = item.get(key)
                if value:
                    return str(value).strip()
    return "Unknown Creator"


def _extract_time(entry: Dict[str, Any]) -> Optional[datetime]:
    for key in ("time", "timestamp"):
        val = entry.get(key)
        if not val:
            continue
        if isinstance(val, str):
            try:
                return date_parser.parse(val)
            except (ValueError, TypeError):
                logger.debug("Could not parse time field %s=%r", key, val)
                continue
    return None


def load_watch_history(filepath: str | Path) -> pd.DataFrame:
    """
    Load Takeout ``watch-history.json`` and return a cleaned DataFrame.

    Columns:

    * ``title`` — cleaned title (no leading "Watched ", emojis stripped)
    * ``channel`` — creator name when present, else ``"unknown"``
    * ``timestamp`` — timezone-aware or naive datetime from Takeout
    * ``hour`` — 0–23
    * ``day_of_week`` — ``Monday`` … ``Sunday``
    * ``month`` — ``YYYY-MM`` string for aggregation

    Rows without a usable title or parseable timestamp are dropped. Rows are sorted
    by ``timestamp`` ascending.
    """
    path = Path(filepath)
    logger.info("Loading watch history from %s", path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if isinstance(data.get("watch_history"), list):
            data = data["watch_history"]
        elif isinstance(data.get("items"), list):
            data = data["items"]

    if not isinstance(data, list):
        raise ValueError("Expected watch history JSON to be a list of entries (or a dict wrapper).")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for entry in data:
        if not isinstance(entry, dict):
            skipped += 1
            continue
        raw_title = entry.get("title")
        if raw_title is None:
            skipped += 1
            continue
        cleaned = clean_watch_title(str(raw_title))
        if not cleaned:
            skipped += 1
            continue
        ts = _extract_time(entry)
        if ts is None:
            skipped += 1
            continue
        channel = _extract_channel(entry)
        rows.append(
            {
                "title": cleaned,
                "channel": channel,
                "timestamp": ts,
            }
        )

    if skipped:
        logger.warning("Skipped %d malformed or incomplete watch entries", skipped)

    if not rows:
        logger.warning("No valid watch rows after parsing %s", path)
        return pd.DataFrame(
            columns=["title", "channel", "timestamp", "hour", "day_of_week", "month"]
        )

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    def _norm_ch(s: Any) -> str:
        t = str(s).strip() if s is not None else ""
        if not t or t.lower() == "unknown":
            return "Unknown Creator"
        return t

    df["channel"] = df["channel"].map(_norm_ch)

    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour.astype(int)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = dt.dt.dayofweek.map(lambda i: weekdays[int(i)])
    df["month"] = dt.dt.strftime("%Y-%m")
    df = df[["title", "channel", "timestamp", "hour", "day_of_week", "month"]]

    logger.info("Loaded %d watch events", len(df))
    return df


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Lightweight upload summary for API responses."""
    if df is None or df.empty:
        return {"record_count": 0, "date_range": None, "unique_channels": 0}
    ts = pd.to_datetime(df["timestamp"])
    return {
        "record_count": int(len(df)),
        "date_range": {
            "start": ts.min().isoformat(),
            "end": ts.max().isoformat(),
        },
        "unique_channels": int(df["channel"].nunique()),
    }
