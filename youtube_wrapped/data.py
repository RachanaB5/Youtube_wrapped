"""
Load Google Takeout YouTube watch history from JSON or HTML into a normalized pandas DataFrame.
"""

from __future__ import annotations

import html as html_lib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser as date_parser

from youtube_wrapped.utils import clean_watch_title

logger = logging.getLogger(__name__)

# Lowercased substrings; ads often have unknown channel + these in the title
_AD_TITLE_MARKERS = ("sec", "h264", ".mp4", "pmax", "16x9", "horizons |")


def _extract_channel(entry: Dict[str, Any]) -> str:
    """Best-effort channel name from Takeout ``subtitles`` / ``details`` fields."""
    if isinstance(entry.get("channel"), str) and entry["channel"].strip():
        return str(entry["channel"]).strip()
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


def _extract_json_title(entry: Dict[str, Any]) -> str:
    """Human-readable title from Takeout JSON (avoid using ``titleUrl`` as text)."""
    raw = entry.get("title")
    if raw is None:
        return ""
    return str(raw).strip()


def _load_json(path: Path) -> List[Dict[str, Any]]:
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
        raw_title = _extract_json_title(entry)
        if not raw_title:
            skipped += 1
            continue
        cleaned = clean_watch_title(raw_title)
        if not cleaned:
            skipped += 1
            continue
        ts = _extract_time(entry)
        if ts is None:
            skipped += 1
            continue
        channel = _extract_channel(entry)
        rows.append({"title": cleaned, "channel": channel, "timestamp": ts})

    if skipped:
        logger.warning("Skipped %d malformed or incomplete JSON watch entries", skipped)
    return rows


def _load_html(path: Path) -> List[Dict[str, Any]]:
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r'<div class="outer-cell', content)
    rows: List[Dict[str, Any]] = []
    for block in blocks[1:]:
        watch_match = re.search(
            r'Watched[\s\xa0]+<a href="(https://www\.youtube\.com/watch\?v=[^"]+)">([^<]+)</a>',
            block,
        )
        if not watch_match:
            continue
        title = clean_watch_title(html_lib.unescape(watch_match.group(2)))
        if not title:
            continue
        channel_match = re.search(
            r'<a href="https://www\.youtube\.com/channel/[^"]+">([^<]+)</a>',
            block,
        )
        channel = (
            html_lib.unescape(channel_match.group(1)).strip()
            if channel_match
            else "Unknown Creator"
        )
        ts_match = re.search(
            r"(\d{1,2} \w+ \d{4}, \d{2}:\d{2}:\d{2})(?:\s+[A-Z]+|\s+UTC|\s+GMT)?",
            block,
        )
        ts: Optional[datetime] = None
        if ts_match:
            raw_ts = ts_match.group(1).strip()
            try:
                ts = datetime.strptime(raw_ts, "%d %b %Y, %H:%M:%S")
            except ValueError:
                try:
                    ts = date_parser.parse(raw_ts)
                except (ValueError, TypeError):
                    ts = None
        if ts is None:
            continue
        rows.append({"title": title, "channel": channel or "Unknown Creator", "timestamp": ts})

    logger.info("Parsed %d watch events from HTML export", len(rows))
    return rows


def _is_probable_ad_row(title: str, channel_norm: str) -> bool:
    if channel_norm != "Unknown Creator":
        return False
    t = title.lower()
    return any(marker in t for marker in _AD_TITLE_MARKERS)


def _finalize_dataframe(rows: List[Dict[str, Any]], source: Path) -> pd.DataFrame:
    if not rows:
        logger.warning("No valid watch rows after parsing %s", source)
        return pd.DataFrame(columns=["title", "channel", "timestamp", "hour", "day_of_week", "month"])

    df = pd.DataFrame(rows)
    df = df[df["title"].astype(str).str.len() > 2]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    def _norm_ch(s: Any) -> str:
        t = str(s).strip() if s is not None else ""
        if not t or t.lower() == "unknown":
            return "Unknown Creator"
        return t

    df["channel"] = df["channel"].map(_norm_ch)
    ad_mask = df.apply(lambda r: _is_probable_ad_row(str(r["title"]), str(r["channel"])), axis=1)
    df = df[~ad_mask].reset_index(drop=True)

    if df.empty:
        logger.warning("No rows left after filtering %s", source)
        return pd.DataFrame(columns=["title", "channel", "timestamp", "hour", "day_of_week", "month"])

    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    dt = pd.to_datetime(df["timestamp"])
    df["hour"] = dt.dt.hour.astype(int)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_of_week"] = dt.dt.dayofweek.map(lambda i: weekdays[int(i)])
    df["month"] = dt.dt.strftime("%Y-%m")
    df = df[["title", "channel", "timestamp", "hour", "day_of_week", "month"]]

    logger.info("Loaded %d watch events from %s", len(df), source)
    return df


def load_watch_history(filepath: str | Path) -> pd.DataFrame:
    """
    Load Takeout ``watch-history.json`` or ``watch-history.html`` and return a cleaned DataFrame.

    Columns:

    * ``title`` — cleaned title (no leading "Watched ", emojis stripped)
    * ``channel`` — creator name when present, else ``"Unknown Creator"``
    * ``timestamp`` — datetime from Takeout
    * ``hour`` — 0–23
    * ``day_of_week`` — ``Monday`` … ``Sunday``
    * ``month`` — ``YYYY-MM`` string for aggregation

    Rows without a usable title or parseable timestamp are dropped. Rows are sorted
    by ``timestamp`` ascending.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()
    logger.info("Loading watch history from %s", path)

    if suffix == ".json":
        rows = _load_json(path)
    elif suffix == ".html":
        rows = _load_html(path)
    else:
        raise ValueError(f"Unsupported watch history format (expected .json or .html): {path}")

    return _finalize_dataframe(rows, path)


def find_stored_watch_history(upload_folder: str | Path, session_id: str) -> Optional[Path]:
    """
    Return the path to a stored upload ``{{session_id}}.json`` or ``{{session_id}}.html`` if present.
    """
    root = Path(upload_folder)
    for ext in (".json", ".html"):
        p = root / f"{session_id}{ext}"
        if p.is_file():
            return p
    return None


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
