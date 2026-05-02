"""
LLM-backed insight generation and cinematic Wrapped slides from pipeline output.

Requires ``OPENAI_API_KEY`` and/or ``ANTHROPIC_API_KEY``. Prefers Anthropic when both are set.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Rough budget: ~4 chars/token for Latin text; stay under ~800 tokens.
_MAX_PROMPT_CHARS = 3000


def _truncate(s: str, max_len: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _format_pipeline_compact(pipeline_output: Dict[str, Any]) -> str:
    """Single prompt-friendly block (token-frugal)."""
    tc = pipeline_output.get("top_categories") or []
    cr = pipeline_output.get("top_creators") or []
    clusters = pipeline_output.get("clusters") or []
    tp = pipeline_output.get("time_patterns") or {}
    mt = pipeline_output.get("monthly_trends") or {}
    journey = pipeline_output.get("journey_summary") or ""

    cat_lines: List[str] = []
    for row in tc[:6]:
        if isinstance(row, dict):
            cat_lines.append(
                f"{row.get('category', '?')}:{row.get('count', 0)}(conf~{row.get('avg_confidence', 0):.2f})"
            )
        else:
            cat_lines.append(str(row))

    creator_lines: List[str] = []
    for row in cr[:8]:
        if isinstance(row, dict):
            creator_lines.append(f"{row.get('channel', '?')}:{row.get('watch_count', 0)}")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            creator_lines.append(f"{row[0]}:{row[1]}")

    cluster_bits: List[str] = []
    if isinstance(clusters, dict):
        items = list(clusters.items())[:6]
        for k, v in items:
            cluster_bits.append(f"{k}:{v}")
    else:
        for c in clusters[:6]:
            if not isinstance(c, dict):
                continue
            cid = c.get("cluster_id", "?")
            sz = c.get("size", 0)
            kw = c.get("keywords") or c.get("label_keywords") or []
            kstr = "/".join(str(x) for x in kw[:4]) if kw else (c.get("cluster_label") or "")
            cluster_bits.append(f"c{cid} n={sz} {kstr}")

    # Monthly: last 8 chronologically
    if isinstance(mt, dict):
        keys_sorted = sorted(mt.keys())[-8:]
        month_bits = [f"{k}:{mt[k]}" for k in keys_sorted]
    else:
        month_bits = []

    lines = [
        "CATEGORIES:" + ";".join(cat_lines),
        "CREATORS:" + ";".join(creator_lines),
        "CLUSTERS:" + " | ".join(cluster_bits),
        f"TIME peak_h={tp.get('peak_hour')} peak_d={tp.get('peak_day')} late_night%={tp.get('late_night_percentage')} most_mo={tp.get('most_active_month')}",
        "MONTHS:" + ";".join(month_bits),
        "JOURNEY:" + _truncate(str(journey), 400),
    ]
    blob = "\n".join(lines)
    return _truncate(blob, _MAX_PROMPT_CHARS)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse model output; strip optional markdown fences."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```$", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("Model JSON root must be an object")
    return data


def _call_openai(system: str, user: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_INSIGHT_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    choice = resp.choices[0].message
    content = choice.content if choice else None
    if not content:
        raise RuntimeError("OpenAI returned empty content")
    return content


def _call_anthropic(system: str, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=os.environ.get("ANTHROPIC_INSIGHT_MODEL", "claude-3-5-haiku-20241022"),
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts: List[str] = []
    for block in msg.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("Anthropic returned empty content")
    return text


def _invoke_llm_json(system: str, user: str) -> Dict[str, Any]:
    pref = (os.environ.get("INSIGHT_LLM_PROVIDER") or "").lower().strip()
    has_anth = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_open = bool(os.environ.get("OPENAI_API_KEY"))

    if pref == "openai":
        if not has_open:
            raise RuntimeError("INSIGHT_LLM_PROVIDER=openai but OPENAI_API_KEY is not set")
        logger.info("Insight LLM: OpenAI (forced)")
        raw = _call_openai(system, user)
    elif pref == "anthropic":
        if not has_anth:
            raise RuntimeError("INSIGHT_LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")
        logger.info("Insight LLM: Anthropic (forced)")
        raw = _call_anthropic(system, user)
    elif has_anth:
        logger.info("Insight LLM: Anthropic (default when both keys or only Anthropic)")
        raw = _call_anthropic(system, user)
    elif has_open:
        logger.info("Insight LLM: OpenAI")
        raw = _call_openai(system, user)
    else:
        raise RuntimeError(
            "No LLM API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "(optional: INSIGHT_LLM_PROVIDER=anthropic|openai)."
        )
    return _extract_json_object(raw)


def generate_insights(pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce structured AI insights from a compact encoding of the pipeline.

    Returns keys: ``persona``, ``top_behaviors`` (list[str]), ``interest_evolution``,
    ``hidden_patterns`` (list[str]), ``prediction``.

    Environment:
        * ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
        * ``INSIGHT_LLM_PROVIDER`` — ``anthropic`` or ``openai`` (optional)
        * ``ANTHROPIC_INSIGHT_MODEL`` — default ``claude-3-5-haiku-20241022``
        * ``OPENAI_INSIGHT_MODEL`` — default ``gpt-3.5-turbo``
    """
    formatted = _format_pipeline_compact(pipeline_output)
    system = (
        "You output only compact JSON. No markdown, no preamble. "
        "Be specific and grounded in the statistics provided; avoid generic platitudes."
    )
    user = f"""You are analyzing a user's YouTube watch history. Given:

{formatted}

Return ONLY valid JSON with these keys:
- persona: a 1-sentence label (e.g. "The Curious Night Owl")
- top_behaviors: list of 3 specific behavioral observations (short clauses, data-aware)
- interest_evolution: 2-sentence narrative of how interests changed
- hidden_patterns: list of 2 surprising or non-obvious insights
- prediction: 1 sentence prediction about their viewing next year"""

    data = _invoke_llm_json(system, user)

    required = ["persona", "top_behaviors", "interest_evolution", "hidden_patterns", "prediction"]
    out: Dict[str, Any] = {}
    for k in required:
        out[k] = data.get(k)
    if not isinstance(out.get("top_behaviors"), list):
        out["top_behaviors"] = [str(out.get("top_behaviors") or "")]
    if not isinstance(out.get("hidden_patterns"), list):
        out["hidden_patterns"] = [str(out.get("hidden_patterns") or "")]
    out["top_behaviors"] = [str(x) for x in out["top_behaviors"][:10]]
    out["hidden_patterns"] = [str(x) for x in out["hidden_patterns"][:10]]
    out["persona"] = str(out.get("persona") or "The Viewer")
    out["interest_evolution"] = str(out.get("interest_evolution") or "")
    out["prediction"] = str(out.get("prediction") or "")
    return out


def _total_watch_count(pipeline_output: Dict[str, Any]) -> int:
    s = pipeline_output.get("summary") or {}
    if s.get("total_videos_analyzed"):
        return int(s["total_videos_analyzed"])
    tc = pipeline_output.get("top_categories") or []
    return int(sum(int(x.get("count", 0)) for x in tc if isinstance(x, dict)))


def _top_category_share(pipeline_output: Dict[str, Any]) -> tuple[str, int, int]:
    tc = pipeline_output.get("top_categories") or []
    if not tc or not isinstance(tc[0], dict):
        return "your top lane", 0, 1
    top = tc[0]
    name = str(top.get("category", "top"))
    count = int(top.get("count", 0))
    total = _total_watch_count(pipeline_output) or max(count, 1)
    return name, count, total


def _cluster_drama(pipeline_output: Dict[str, Any]) -> tuple[str, Optional[str]]:
    clusters = pipeline_output.get("clusters") or []
    if isinstance(clusters, dict):
        return "your rabbit holes", None
    if not clusters:
        return "plots you keep revisiting", None
    largest = max(clusters, key=lambda c: int(c.get("size", 0)) if isinstance(c, dict) else 0)
    if not isinstance(largest, dict):
        return "your rabbit holes", None
    total = _total_watch_count(pipeline_output) or 1
    pct = round(100 * int(largest.get("size", 0)) / total)
    label = largest.get("cluster_label") or ""
    kw = largest.get("keywords") or largest.get("label_keywords") or []
    if not label and kw:
        label = " / ".join(str(x) for x in kw[:3])
    if not label:
        label = f"cluster {largest.get('cluster_id', '')}"
    return label, f"{pct}% of your log clusters here"


def generate_wrapped_story(
    insights: Dict[str, Any],
    pipeline_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build six cinematic slides mixing LLM insights with headline stats (works offline).

    ``insights`` should be the dict from :func:`generate_insights` (or matching shape).
    """
    persona = str(insights.get("persona") or "Main character energy")
    behaviors = insights.get("top_behaviors") or []
    behaviors = [str(b) for b in behaviors if b][:3]
    while len(behaviors) < 3:
        behaviors.append("You chased what hooked you—then chased it again.")

    evolution = str(insights.get("interest_evolution") or "").strip()
    evolution_sents = re.split(r"(?<=[.!?])\s+", evolution)
    evo_a = evolution_sents[0] if evolution_sents and evolution_sents[0] else evolution
    evo_b = evolution_sents[1] if len(evolution_sents) > 1 else ""

    hidden = insights.get("hidden_patterns") or []
    hidden = [str(h) for h in hidden if h][:2]
    while len(hidden) < 2:
        hidden.append("The small repeats in your history are louder than the one-off spikes.")

    prediction = str(insights.get("prediction") or "Next year looks like more of you—pickier, weirder, better.")

    cat_name, cat_count, total = _top_category_share(pipeline_output)
    cat_pct = round(100 * cat_count / total) if total else 0

    creators = pipeline_output.get("top_creators") or []
    top_ch, top_n = "the algorithm's favorites", 0
    if creators and isinstance(creators[0], dict):
        top_ch = str(creators[0].get("channel", top_ch))
        top_n = int(creators[0].get("watch_count", 0))
    elif creators and isinstance(creators[0], (list, tuple)) and len(creators[0]) >= 2:
        top_ch = str(creators[0][0])
        top_n = int(creators[0][1])

    tp = pipeline_output.get("time_patterns") or {}
    late = tp.get("late_night_percentage")
    peak_h = tp.get("peak_hour")
    stat_time = None
    if late is not None:
        stat_time = f"{late}% late-night"
    elif peak_h is not None:
        stat_time = f"peak {peak_h}:00"

    journey = _truncate(str(pipeline_output.get("journey_summary") or ""), 180)
    cluster_label, cluster_stat = _cluster_drama(pipeline_output)

    if late is not None:
        time_lede = f"The late-night ledger: about {late}% of watches between midnight and 5am—your plot thickens after hours."
    elif peak_h is not None:
        time_lede = f"Peak playback clusters around {peak_h}:00—your day has a very specific 'scene change.'"
    else:
        time_lede = "Your watch history has a rhythm—hours matter more than algorithms pretend."

    slide1_body = [
        "You didn't just watch videos—you collected moods, phases, and plot twists.",
        persona,
        "If your history had a genre, it would be 'determined curiosity with dramatic lighting.'",
    ]

    slide2_body = [
        f"{cat_name.capitalize()} kept showing up—not as a label, as a habit.",
        behaviors[0],
        "That's not random noise; that's a preference wearing headphones.",
    ]

    slide3_body = [
        f"{top_ch} kept winning your attention{f' ({top_n} times)' if top_n else ''}.",
        behaviors[1],
        "Some creators aren't 'content.' They're recurring characters in your year.",
    ]

    slide4_body = [
        time_lede,
        behaviors[2],
        "The vibe isn't 'watching.' It's the moment your day hands you back to yourself.",
    ]

    slide5_body = [
        f"Your deepest rabbit hole looked like: {cluster_label}.",
        hidden[0],
        journey or "Your months weren't equal—some were loudest in the quiet details.",
    ]

    slide6_body = [
        evo_a,
        evo_b or hidden[1],
        prediction,
    ]

    slides: List[Dict[str, Any]] = [
        {
            "slide_number": 1,
            "headline": "This Wasn't Just YouTube",
            "body": slide1_body,
            "stat": str(total) if total else None,
            "emoji": "🎬",
        },
        {
            "slide_number": 2,
            "headline": f"The {cat_name.title()} Era",
            "body": slide2_body,
            "stat": f"{cat_pct}% of classified top share" if total else None,
            "emoji": "🎯",
        },
        {
            "slide_number": 3,
            "headline": "Your Co-Stars",
            "body": slide3_body,
            "stat": str(top_n) if top_n else None,
            "emoji": "⭐",
        },
        {
            "slide_number": 4,
            "headline": "The Midnight Evidence",
            "body": slide4_body,
            "stat": stat_time,
            "emoji": "🌙",
        },
        {
            "slide_number": 5,
            "headline": "Plot Twists You Didn't Post About",
            "body": slide5_body,
            "stat": cluster_stat,
            "emoji": "🌀",
        },
        {
            "slide_number": 6,
            "headline": "Next Season Teaser",
            "body": slide6_body,
            "stat": None,
            "emoji": "🔮",
        },
    ]
    return slides


__all__ = ["generate_insights", "generate_wrapped_story"]
