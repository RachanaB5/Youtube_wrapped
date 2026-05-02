"""
Load and normalize Google Takeout YouTube watch-history JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dateutil import parser as date_parser

from youtube_wrapped.utils import clean_watch_title


@dataclass
class WatchRecord:
    """One watch event after parsing."""

    title: str
    channel: str
    watched_at: datetime
    raw_title: str = ""
    title_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "channel": self.channel,
            "timestamp_iso": self.watched_at.isoformat(),
            "raw_title": self.raw_title,
            "title_url": self.title_url,
        }


def _extract_channel(entry: Dict[str, Any]) -> str:
    """Best-effort channel name from Takeout `subtitles` / `details` fields."""
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
    return "unknown"


def _extract_time(entry: Dict[str, Any]) -> Optional[datetime]:
    for key in ("time", "timestamp"):
        val = entry.get(key)
        if not val:
            continue
        if isinstance(val, str):
            try:
                return date_parser.parse(val)
            except (ValueError, TypeError):
                continue
    return None


def load_watch_history(path: str | Path) -> List[WatchRecord]:
    """
    Load Takeout JSON (array of objects) and return sorted `WatchRecord` list
    (oldest first).

    Skips entries without title or parseable time when possible; if time is
    missing, those rows are dropped for downstream time-ordered modeling.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if isinstance(data.get("watch_history"), list):
            data = data["watch_history"]
        elif isinstance(data.get("items"), list):
            data = data["items"]

    if not isinstance(data, list):
        raise ValueError("Expected watch history JSON to be a list of entries.")

    records: List[WatchRecord] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        raw_title = entry.get("title") or ""
        cleaned = clean_watch_title(str(raw_title))
        if not cleaned:
            continue
        channel = _extract_channel(entry)
        watched_at = _extract_time(entry)
        if watched_at is None:
            continue
        records.append(
            WatchRecord(
                title=cleaned,
                channel=channel,
                watched_at=watched_at,
                raw_title=str(raw_title),
                title_url=entry.get("titleUrl"),
            )
        )

    records.sort(key=lambda r: r.watched_at)
    return records


def summarize_records(records: List[WatchRecord]) -> Dict[str, Any]:
    """Return lightweight metadata useful for API responses."""
    if not records:
        return {
            "record_count": 0,
            "date_range": None,
            "unique_channels": 0,
        }

    return {
        "record_count": len(records),
        "date_range": {
            "start": records[0].watched_at.isoformat(),
            "end": records[-1].watched_at.isoformat(),
        },
        "unique_channels": len({record.channel for record in records}),
    }


def records_to_rows(records: List[WatchRecord]) -> List[Dict[str, Any]]:
    """Serialize records for JSON / DataFrame-style pipelines."""
    return [r.to_dict() for r in records]
