"""Build a compact payload for social share cards from cached pipeline output."""

from __future__ import annotations

from typing import Any, Dict, List


def build_share_card_payload(insights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pre-formatted JSON for a vertical share graphic (persona, categories, creator, peak time).
    """
    narrative = insights.get("narrative") or {}
    persona = str(narrative.get("persona") or "YouTube Wrapped viewer")
    # Prefer a short label: first clause or truncated persona
    persona_label = persona.split(".")[0].strip()[:80] if persona else "Your Wrapped"

    top_cats: List[Dict[str, Any]] = list(insights.get("top_categories") or [])[:3]
    categories_for_card = [
        {
            "name": str(c.get("category", "")),
            "count": int(c.get("count", 0)),
            "share_pct": None,
        }
        for c in top_cats
    ]
    total = int(
        (insights.get("summary") or {}).get("total_videos_analyzed")
        or sum(int(x.get("count", 0)) for x in (insights.get("top_categories") or []))
        or 0
    )
    if total > 0:
        for row in categories_for_card:
            row["share_pct"] = round(100.0 * row["count"] / total, 1)

    creators = insights.get("top_creators") or []
    top_creator_name = "Unknown Creator"
    top_creator_count = 0
    if creators:
        c0 = creators[0]
        if isinstance(c0, dict):
            top_creator_name = str(c0.get("channel") or "Unknown Creator")
            top_creator_count = int(c0.get("watch_count", 0))
        elif isinstance(c0, (list, tuple)) and len(c0) >= 2:
            top_creator_name = str(c0[0] or "Unknown Creator")
            top_creator_count = int(c0[1])

    if not top_creator_name.strip():
        top_creator_name = "Unknown Creator"

    tp = insights.get("time_patterns") or {}
    peak_hour = tp.get("peak_hour")
    peak_day = tp.get("peak_day")
    late_pct = tp.get("late_night_percentage")

    if peak_hour is not None:
        peak_label = f"{int(peak_hour)}:00 — {peak_day or 'peak day unknown'}"
    else:
        peak_label = str(peak_day or "varies")

    return {
        "version": 1,
        "persona": persona,
        "persona_label": persona_label,
        "top_categories": categories_for_card,
        "top_creator": {"name": top_creator_name, "watch_count": top_creator_count},
        "peak_watch": {
            "label": peak_label,
            "hour": peak_hour,
            "day": peak_day,
            "late_night_percentage": late_pct,
        },
        "brand": {"title": "YouTube Wrapped", "subtitle": "Your year in watch history"},
        "theme_suggestion": "gradient_violet_cyan",
    }
