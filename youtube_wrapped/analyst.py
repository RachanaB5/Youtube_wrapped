"""
Turn aggregated watch-history statistics into a short, data-grounded narrative.

No external LLM: sentences are composed from measured counts, shares, and ranks
so the output stays traceable to the input JSON.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Categories treated as "utility / learning" vs "reward / play" for persona axes.
_LEARN_BIAS = frozenset({"technology", "education", "finance", "cooking", "tech", "productivity"})
_PLAY_BIAS = frozenset({"entertainment", "gaming", "music", "sports"})


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _sorted_month_pairs(month_counts: Dict[str, int]) -> List[Tuple[str, int]]:
    """Chronological (YYYY-MM) month keys with watch counts."""
    return sorted(((k, int(v)) for k, v in month_counts.items()), key=lambda kv: kv[0])


def _monthly_spike_notes(month_pairs: List[Tuple[str, int]]) -> List[str]:
    """Flag months that rise meaningfully above a trailing baseline."""
    if len(month_pairs) < 3:
        return []
    counts = [c for _, c in month_pairs]
    notes: List[str] = []
    for i in range(2, len(month_pairs)):
        window = counts[max(0, i - 3) : i]
        baseline = sum(window) / len(window) if window else 0.0
        m, v = month_pairs[i]
        if baseline > 0 and v >= baseline * 1.45 and v - baseline >= 3:
            pct = int(round(100.0 * (v / baseline - 1.0)))
            notes.append(f"{m} jumped to {v} watches (~{pct}% above the prior 3-month average of {baseline:.1f}).")
        elif baseline == 0 and v >= 5:
            notes.append(f"{m} is your first active month in this window ({v} watches).")
    return notes[:3]


def _cluster_concentration(clusters: Sequence[Dict[str, Any]], total: int) -> Tuple[Optional[int], float, List[str]]:
    """Largest cluster id, its share of watches, and top keywords for that cluster."""
    if not clusters or total <= 0:
        return None, 0.0, []
    sized = [(int(c.get("cluster_id", -1)), int(c.get("size", 0)), c.get("label_keywords") or []) for c in clusters]
    sized.sort(key=lambda t: t[1], reverse=True)
    cid, size, kws = sized[0]
    return cid, size / total, list(kws)[:4]


def _smallest_substantive_cluster(clusters: Sequence[Dict[str, Any]], total: int) -> Optional[Dict[str, Any]]:
    """Smallest non-trivial cluster (niche lane), skipping noise buckets if labeled."""
    candidates: List[Dict[str, Any]] = []
    for c in clusters:
        if c.get("role") == "noise_or_rare":
            continue
        s = int(c.get("size", 0))
        if s >= 2 and total > 0 and s <= max(3, total // 25):
            candidates.append(c)
    if not candidates:
        return None
    return min(candidates, key=lambda c: int(c.get("size", 0)))


def _weekday_weekend_ratio(dow: Dict[str, Any]) -> Optional[float]:
    """>1 means more weekday-heavy."""
    wk = sum(int(dow.get(d, 0)) for d in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"))
    we = sum(int(dow.get(d, 0)) for d in ("Saturday", "Sunday"))
    if we <= 0:
        return None
    return wk / (we + 1e-9)


def _normalize_month_counts(
    monthly_trends: Any,
    time_patterns: Dict[str, Any],
) -> Dict[str, int]:
    """Accept either a raw month-count dict or a list of month/count rows."""
    if isinstance(monthly_trends, dict):
        return {str(k): int(v) for k, v in monthly_trends.items()}
    if isinstance(monthly_trends, list):
        out: Dict[str, int] = {}
        for item in monthly_trends:
            if not isinstance(item, dict):
                continue
            month = item.get("month") or item.get("label") or item.get("key")
            count = item.get("count") or item.get("watch_count") or item.get("value")
            if month is not None and count is not None:
                out[str(month)] = int(count)
        if out:
            return out
    raw = time_patterns.get("month_counts") or {}
    return {str(k): int(v) for k, v in raw.items()}


def generate_structured_analyst_output(
    categories: Sequence[Dict[str, Any]],
    creators: Sequence[Dict[str, Any]],
    clusters: Sequence[Dict[str, Any]],
    time_patterns: Dict[str, Any],
    monthly_trends: Any = None,
    total_videos: Optional[int] = None,
    interest_shift: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate structured, data-grounded insights from pre-aggregated inputs.

    This is the direct analyst entrypoint for prompt-style or API-style use.
    """
    month_counts = _normalize_month_counts(monthly_trends, time_patterns)
    if total_videos is None:
        total_videos = max(
            sum(int(x.get("count", 0)) for x in categories),
            sum(int(x.get("size", 0)) for x in clusters),
            sum(int(v) for v in month_counts.values()),
            0,
        )

    peak_hour = time_patterns.get("peak_hour")
    hourly = {int(k): int(v) for k, v in (time_patterns.get("hourly_counts") or {}).items()}
    dow = time_patterns.get("day_of_week_counts") or {}
    late_ratio = _safe_float(time_patterns.get("late_night_ratio"), 0.0)
    bucket_counts = time_patterns.get("time_bucket_counts") or {}
    peak_bucket = max(bucket_counts, key=bucket_counts.get) if bucket_counts else None

    insights = {
        "summary": {
            "total_videos_analyzed": int(total_videos or 0),
            "top_categories": list(categories or []),
            "top_creators": list(creators or []),
        },
        "behavior": {
            "time_patterns": {
                "hourly_counts": {str(k): v for k, v in hourly.items()},
                "day_of_week_counts": dow,
                "month_counts": month_counts,
                "time_bucket_counts": bucket_counts,
                "peak_hour": peak_hour,
                "late_night_ratio": late_ratio,
            },
            "peak_time_bucket": peak_bucket,
            "interest_shift": interest_shift or {},
        },
        "clusters": list(clusters or []),
        "per_video": [],
    }
    return generate_narrative_insights(insights)


def generate_cinematic_summary(
    top_category: str,
    top_creators: Sequence[Dict[str, Any]] | Sequence[str],
    peak_time: str,
    cluster_summary: Sequence[Dict[str, Any]],
    persona: str,
) -> List[Dict[str, Any]]:
    """
    Build a slide-style cinematic summary for a YouTube Wrapped reveal.

    Returns 6 short sections with a headline and 2-3 punchy lines each.
    """
    creator_names: List[str] = []
    for creator in top_creators[:3]:
        if isinstance(creator, dict):
            name = creator.get("channel") or creator.get("name")
            if name:
                creator_names.append(str(name))
        elif creator:
            creator_names.append(str(creator))

    creator_line = ", ".join(creator_names[:3]) if creator_names else "a rotating cast of internet co-stars"

    cluster_labels: List[str] = []
    for cluster in cluster_summary[:3]:
        if not isinstance(cluster, dict):
            continue
        label = cluster.get("cluster_label")
        if label:
            cluster_labels.append(str(label))
            continue
        keywords = cluster.get("label_keywords") or cluster.get("keywords") or []
        if keywords:
            cluster_labels.append(" / ".join(str(x) for x in keywords[:3]))
    cluster_line = ", ".join(cluster_labels[:3]) if cluster_labels else "a few suspiciously specific rabbit holes"

    category_phrase = top_category or "whatever the algorithm whispered first"
    peak_phrase = peak_time or "some delightfully unholy hour"
    persona_phrase = persona or "equal parts chaos goblin and accidental scholar"

    prediction = (
        f"Next year, expect {category_phrase} to keep the crown, but one of those {cluster_line} side quests is absolutely plotting a main-character takeover."
    )

    return [
        {
            "headline": "You Didn’t Just Watch YouTube",
            "lines": [
                "You staged a full season arc.",
                f"The genre at the center of it all was {category_phrase}.",
                "This wasn’t scrolling. This was lore-building.",
            ],
        },
        {
            "headline": "The Usual Suspects",
            "lines": [
                f"Your screen kept finding its way back to {creator_line}.",
                "Not in a casual 'oh, neat video' way.",
                "In a 'we know each other now' way.",
            ],
        },
        {
            "headline": "Meanwhile, The Clock Saw Everything",
            "lines": [
                f"Your peak hour was {peak_phrase}.",
                "Which means your watch history knows exactly when the day ended and the side quests began.",
                "Very cinematic. Mildly incriminating.",
            ],
        },
        {
            "headline": "The Rabbit Holes Had Names",
            "lines": [
                f"You kept circling back to {cluster_line}.",
                "That is not random curiosity. That is a recurring subplot.",
                "The algorithm didn’t lead you there alone. You packed snacks.",
            ],
        },
        {
            "headline": "And Then There Was You",
            "lines": [
                persona_phrase,
                "Part explorer, part loyalist, part person who absolutely said 'just one more video.'",
                "Your watch history reads less like data and more like a personality test with autoplay.",
            ],
        },
        {
            "headline": "Trailer For Next Year",
            "lines": [
                prediction,
                "One comfort creator will remain untouchable.",
                "But a weirdly specific new obsession is already warming up in the wings.",
            ],
        },
    ]


def generate_narrative_insights(insights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce persona-style copy from a full ``insights`` dict (same shape as ``run_pipeline``).

    Returns:
        ``persona``, ``top_behaviors``, ``interest_evolution``, ``hidden_patterns``.
    """
    empty = {
        "persona": "Not enough history in this export to characterize viewing—upload a longer Takeout window.",
        "top_behaviors": [],
        "interest_evolution": "",
        "hidden_patterns": [],
    }
    summary = insights.get("summary") or {}
    total = int(summary.get("total_videos_analyzed", 0) or 0)
    if total <= 0:
        return empty

    behavior = insights.get("behavior") or {}
    tp = behavior.get("time_patterns") or {}
    month_counts = tp.get("month_counts") or {}
    hourly = {int(k): int(v) for k, v in (tp.get("hourly_counts") or {}).items()}
    dow = tp.get("day_of_week_counts") or {}

    top_cats: List[Dict[str, Any]] = list(summary.get("top_categories") or [])
    creators: List[Dict[str, Any]] = list(summary.get("top_creators") or [])
    clusters: List[Dict[str, Any]] = list(insights.get("clusters") or [])

    # Category mass on the learner vs entertainer axis
    mass_learn = sum(int(x.get("count", 0)) for x in top_cats if x.get("category") in _LEARN_BIAS)
    mass_play = sum(int(x.get("count", 0)) for x in top_cats if x.get("category") in _PLAY_BIAS)
    dom = top_cats[0] if top_cats else {}
    dom_name = str(dom.get("category", "mixed"))
    dom_n = int(dom.get("count", 0))
    dom_share = dom_n / total if total else 0.0

    top_ch = creators[0] if creators else {}
    top_ch_name = str(top_ch.get("channel", "unknown"))
    top_ch_n = int(top_ch.get("watch_count", 0))
    top3 = sum(int(c.get("watch_count", 0)) for c in creators[:3])

    per_video: List[Dict[str, Any]] = list(insights.get("per_video") or [])
    uniq_ch = len({str(r.get("channel")) for r in per_video}) if per_video else len(creators)
    diversity_ratio = uniq_ch / max(total, 1)

    peak_hour = tp.get("peak_hour")
    late_ratio = _safe_float(tp.get("late_night_ratio"), 0.0)
    peak_bucket = behavior.get("peak_time_bucket")

    shift = behavior.get("interest_shift") or {}
    emb_drift = shift.get("embedding_drift")
    combined = shift.get("combined_shift_score")

    month_pairs = _sorted_month_pairs(month_counts)
    spike_notes = _monthly_spike_notes(month_pairs)
    first_m = month_pairs[0][1] if month_pairs else 0
    last_m = month_pairs[-1][1] if month_pairs else 0
    mid = len(month_pairs) // 2
    early_avg = (
        sum(c for _, c in month_pairs[:mid]) / max(mid, 1) if month_pairs and mid else 0.0
    )
    late_avg = (
        sum(c for _, c in month_pairs[mid:]) / max(len(month_pairs) - mid, 1) if month_pairs else 0.0
    )

    _, max_cl_share, max_kw = _cluster_concentration(clusters, total)
    niche = _smallest_substantive_cluster(clusters, total)
    wk_we = _weekday_weekend_ratio(dow)

    # Persona (single paragraph, specific numbers)
    learner_edge = mass_learn - mass_play
    if learner_edge >= max(5, int(0.08 * total)):
        axis = "utility- and curiosity-driven learner"
    elif learner_edge <= -max(5, int(0.08 * total)):
        axis = "reward- and mood-driven viewer"
    else:
        axis = "balanced between learning and leisure"

    if top_ch_n / total >= 0.22:
        focus = f"anchored heavily on {top_ch_name} ({top_ch_n} watches, {int(round(100 * top_ch_n / total))}% of the log)"
    elif diversity_ratio >= 0.45:
        focus = f"wide channel rotation ({uniq_ch} distinct channels across {total} watches—high exploration)"
    else:
        focus = f"moderate channel loyalty (top 3 creators cover {int(round(100 * top3 / total))}% of views)"

    night = ""
    if late_ratio >= 0.18:
        night = f" Late-hours usage is material (~{int(round(100 * late_ratio))}% between midnight–5am), which often correlates with decompression or second-shift screen time."
    elif peak_bucket in ("evening", "night") and peak_hour is not None:
        night = f" Your density peaks in the {peak_bucket} bucket (peak hour {int(peak_hour)}), suggesting YouTube is slotted after daytime obligations."

    kw_bit = ""
    if max_kw:
        kw_bit = f" The largest semantic cluster tilts toward “{' / '.join(max_kw)}”—that language should show up repeatedly in titles, not as a generic hobby label."

    persona = (
        f"You scan as a {axis}: {focus}.{night}{kw_bit}"
    ).strip()

    # top_behaviors
    behaviors: List[str] = []
    behaviors.append(
        f"Category center of gravity is {dom_name} at {dom_n} watches ({int(round(100 * dom_share))}% of classified rows)—that should be treated as your public “headline” interest, not a guess."
    )
    if max_cl_share >= 0.35:
        behaviors.append(
            f"Roughly {int(round(100 * max_cl_share))}% of videos fall into one embedding cluster; that concentration is binge- or topic-lane behavior rather than evenly scattered curiosity."
        )
    if top_ch_n >= 8:
        behaviors.append(
            f"{top_ch_name} is the measurable default ({top_ch_n} watches)—when autoplay kicks in, odds lean toward that creator or adjacent recommendations."
        )
    if late_ratio >= 0.12:
        behaviors.append(
            f"Night owling is measurable: {int(round(100 * late_ratio))}% of timestamps sit between midnight–5am."
        )
    if wk_we is not None and wk_we >= 1.25:
        behaviors.append(
            f"Weekdays beat weekends by ~{int(round(100 * (wk_we - 1)))}% more volume—YouTube is behaving like a weekday utility, not purely weekend entertainment."
        )
    elif wk_we is not None and wk_we <= 0.85:
        behaviors.append("Weekend-heavy viewing suggests longer, intentional sessions rather than workweek filler.")

    # interest_evolution
    evo_parts: List[str] = []
    if isinstance(emb_drift, (int, float)) and combined is not None:
        if float(emb_drift) < 0.035:
            evo_parts.append(
                f"Early vs late embedding drift is small (cosine drift {float(emb_drift):.3f})—your interests look steady within this export window."
            )
        elif float(emb_drift) > 0.12:
            evo_parts.append(
                f"Embedding drift is elevated ({float(emb_drift):.3f} cosine drift)—late-period watching diverges from how you started this history, implying a real lane change or new obsession."
            )
        else:
            evo_parts.append(
                f"Moderate drift ({float(emb_drift):.3f}) suggests evolution without a full reset; combined trajectory score is {float(combined):.3f}."
            )
    if month_pairs:
        if early_avg > 0 and late_avg > 0:
            delta = (late_avg - early_avg) / early_avg
            if abs(delta) >= 0.2:
                direction = "up" if delta > 0 else "down"
                evo_parts.append(
                    f"Monthly cadence drifts {direction}: average ~{early_avg:.1f} watches/month in the earlier half vs ~{late_avg:.1f} in the later half."
                )
        evo_parts.extend(spike_notes)
    interest_evolution = " ".join(evo_parts).strip() or (
        "Monthly grain is too sparse to narrate a slope—keep longer Takeout spans for clearer evolution."
    )

    # hidden_patterns
    hidden: List[str] = []
    if niche:
        nk = niche.get("label_keywords") or []
        hidden.append(
            f"Niche lane: cluster {niche.get('cluster_id')} is only {int(niche.get('size', 0))} watches but repeats keywords like “{' / '.join(list(nk)[:4])}”—small count, tight theme (identity hobby, not mass consumption)."
        )
    if diversity_ratio >= 0.5 and dom_share < 0.35:
        hidden.append(
            "High channel cardinality with no single dominant category share reads as exploratory sampling: you accumulate breadth more than a single storyline."
        )
    if spike_notes:
        first_spike = spike_notes[0]
        hidden.append(
            f"Sudden spike: {first_spike.rstrip('.')}—treat as calendar-correlated, not algorithm noise."
        )
    if isinstance(peak_hour, int) and hourly:
        top3_hours = sorted(hourly.items(), key=lambda kv: kv[1], reverse=True)[:3]
        h_str = ", ".join(f"{h}h ({n})" for h, n in top3_hours)
        hidden.append(f"Hour fingerprint: top hours {h_str}—compare to your sleep schedule for 'revenge bedtime' structure.")

    return {
        "persona": persona,
        "top_behaviors": behaviors[:8],
        "interest_evolution": interest_evolution,
        "hidden_patterns": hidden[:8],
    }
