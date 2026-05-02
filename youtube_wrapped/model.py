"""
Embeddings, clustering, TF-IDF cluster labels, fast keyword categories, sequence drift.
Optimized for large histories (50k+ rows): batched normalized embeddings, disk cache,
MiniBatchKMeans on a subsample, optional progress callbacks.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from youtube_wrapped.analyst import generate_cinematic_summary, generate_narrative_insights
from youtube_wrapped.insight_generator import generate_insights as generate_ai_insights
from youtube_wrapped.insight_generator import generate_wrapped_story
from youtube_wrapped.sequence_model import detect_interest_shifts
from youtube_wrapped.utils import get_time_patterns, get_top_creators

logger = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[str, int, Optional[Dict[str, Any]]], None]]

CLASSIFICATION_LABELS: List[str] = [
    "technology",
    "music",
    "education",
    "gaming",
    "entertainment",
    "cooking",
    "finance",
    "sports",
]

# Fast title classifier (replaces zero-shot for throughput).
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "technology": [
        "coding",
        "python",
        "javascript",
        "ai",
        "software",
        "programming",
        "tech",
        "react",
        "developer",
        "linux",
        "gpu",
        "kubernetes",
        "api",
    ],
    "music": [
        "song",
        "music",
        "album",
        "kpop",
        "playlist",
        "lyrics",
        "concert",
        "mv",
        "official audio",
        "remix",
        "band",
    ],
    "gaming": [
        "gaming",
        "gameplay",
        "minecraft",
        "playthrough",
        "fps",
        "game",
        "esports",
        "speedrun",
        "walkthrough",
        "nintendo",
        "steam",
    ],
    "education": [
        "tutorial",
        "learn",
        "course",
        "explained",
        "how to",
        "guide",
        "study",
        "lecture",
        "exam",
        "science",
    ],
    "entertainment": [
        "vlog",
        "comedy",
        "funny",
        "shorts",
        "reaction",
        "challenge",
        "prank",
        "trailer",
        "interview",
    ],
    "food": [
        "recipe",
        "cooking",
        "food",
        "restaurant",
        "baking",
        "chef",
        "kitchen",
    ],
    "fitness": [
        "workout",
        "gym",
        "fitness",
        "yoga",
        "exercise",
        "training",
        "cardio",
    ],
    "news": [
        "news",
        "update",
        "breaking",
        "politics",
        "world",
        "report",
        "live",
        "today",
    ],
    "finance": [
        "stock",
        "market",
        "invest",
        "crypto",
        "bitcoin",
        "economy",
        "finance",
        "fed",
        "earnings",
    ],
}

# Map keyword buckets onto exported taxonomy labels.
_KEYWORD_BUCKET_TO_LABEL: Dict[str, str] = {
    "technology": "technology",
    "music": "music",
    "gaming": "gaming",
    "education": "education",
    "entertainment": "entertainment",
    "food": "cooking",
    "fitness": "sports",
    "news": "finance",
    "finance": "finance",
}

_EMBEDDER = None

MAX_KMEANS_FIT = 5000
MAX_LSTM_SEQ = 2048
MAX_TFIDF_PER_CLUSTER = 3500

_CACHE_VERSION = "emb_v1_norm512"


def llm_insights_available() -> bool:
    import os

    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _ensure_hf_cache_dir() -> str:
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        cache_dir = str((Path(__file__).resolve().parent.parent / ".hf-cache").resolve())
        os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _cache_dir() -> Path:
    root = os.environ.get("YOUTUBE_WRAPPED_CACHE_DIR")
    if root:
        p = Path(root)
    else:
        p = Path(__file__).resolve().parent.parent / ".cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _titles_fingerprint(titles: List[str]) -> str:
    """Stable hash over full title list (streaming; safe for 50k+ strings)."""
    h = hashlib.sha256()
    h.update(_CACHE_VERSION.encode())
    h.update(str(len(titles)).encode())
    for t in titles:
        h.update(b"\n")
        h.update(str(t).encode("utf-8", errors="replace"))
    return h.hexdigest()


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        cache_dir = _ensure_hf_cache_dir()
        logger.info("Loading SentenceTransformer all-MiniLM-L6-v2")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
    return _EMBEDDER


def get_embeddings(
    titles: List[str],
    batch_size: int = 512,
    show_progress_bar: Optional[bool] = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    """
    Encode titles with SentenceTransformer. Batched, L2-normalized by default.
    """
    if not titles:
        return np.zeros((0, 384), dtype=np.float32)
    if show_progress_bar is None:
        show_progress_bar = os.environ.get("YOUTUBE_WRAPPED_EMBED_PROGRESS", "").lower() in (
            "1",
            "true",
            "yes",
        )
    model = _get_embedder()
    vectors = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return np.asarray(vectors, dtype=np.float32)


def get_cached_embeddings(
    titles: List[str],
    batch_size: int = 512,
    cache_extra: str = "",
) -> np.ndarray:
    """
    Disk-cached embeddings keyed by full-title fingerprint (and cache schema version).

    ``cache_extra`` (e.g. session_id) avoids accidental reuse if you change titles
    outside this list without changing fingerprints — usually leave empty.
    """
    if not titles:
        return np.zeros((0, 384), dtype=np.float32)
    fp = _titles_fingerprint(titles)
    key = hashlib.sha256(f"{fp}:{cache_extra}".encode()).hexdigest()
    cache_path = _cache_dir() / f"embeddings_{key}.pkl"
    if cache_path.is_file():
        logger.info("Embedding cache hit %s", cache_path.name)
        with cache_path.open("rb") as f:
            return pickle.load(f)
    logger.info("Embedding cache miss — computing %d titles", len(titles))
    embeddings = get_embeddings(titles, batch_size=batch_size)
    with cache_path.open("wb") as f:
        pickle.dump(embeddings, f)
    return embeddings


def cluster_videos(
    embeddings: np.ndarray,
    n_clusters: int = 12,
    random_state: int = 42,
    max_fit: int = MAX_KMEANS_FIT,
) -> np.ndarray:
    """
    MiniBatchKMeans: fit on up to ``max_fit`` random rows, ``predict`` on all rows.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    k = max(1, min(int(n_clusters), n))
    rng = np.random.default_rng(random_state)
    if len(embeddings) > max_fit:
        idx = rng.choice(len(embeddings), size=max_fit, replace=False)
        sample_emb = embeddings[idx]
    else:
        sample_emb = embeddings
    k_fit = min(k, len(sample_emb), n)
    k_fit = max(1, k_fit)
    logger.info("MiniBatchKMeans fit on %d rows, k=%d, predict on %d", len(sample_emb), k_fit, n)
    km = MiniBatchKMeans(
        n_clusters=k_fit,
        random_state=random_state,
        n_init=3,
        max_iter=100,
        batch_size=min(1024, max(256, len(sample_emb) // 4)),
    )
    km.fit(sample_emb)
    labels = km.predict(embeddings).astype(np.int32)
    return labels


def _classify_with_keyword_map(title: str, kw_map: Dict[str, List[str]]) -> Tuple[str, float]:
    """Keyword overlap on an arbitrary bucket→keywords map."""
    title_lower = str(title).lower()
    scores = {cat: sum(1 for kw in kws if kw in title_lower) for cat, kws in kw_map.items()}
    best_kw = max(scores, key=scores.get)
    if scores[best_kw] == 0:
        return "entertainment", 0.35
    label = _KEYWORD_BUCKET_TO_LABEL.get(best_kw, "entertainment")
    if label not in CLASSIFICATION_LABELS:
        label = "entertainment"
    conf = float(min(0.99, 0.48 + 0.11 * scores[best_kw]))
    return label, conf


def classify_category_fast(title: str) -> Tuple[str, float]:
    """Keyword overlap + user feedback overrides + learned keyword merge."""
    from youtube_wrapped import feedback as fb

    def base(t: str) -> Tuple[str, float]:
        merged = fb.get_effective_category_keywords(CATEGORY_KEYWORDS)
        return _classify_with_keyword_map(t, merged)

    return fb.classify_with_feedback(title, base)


def classify_categories_scored(
    titles: List[str],
    categories: Optional[List[str]] = None,
    batch_size: int = 4,
) -> Tuple[List[str], List[float]]:
    """Fast path: keyword classifier per title (``categories`` ignored; kept for API compatibility)."""
    if not titles:
        return [], []
    predicted: List[str] = []
    scores: List[float] = []
    for t in titles:
        cat, sc = classify_category_fast(t)
        predicted.append(cat)
        scores.append(sc)
    return predicted, scores


def classify_categories(titles: List[str]) -> List[str]:
    cats, _ = classify_categories_scored(titles)
    return cats


def label_clusters(df: pd.DataFrame, labels: np.ndarray, top_n: int = 5) -> Dict[int, List[str]]:
    if df is None or df.empty or len(labels) == 0:
        return {}
    titles = df["title"].astype(str).tolist()
    rng = np.random.default_rng(42)
    out: Dict[int, List[str]] = {}
    for cid in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cid)[0].tolist()
        cluster_texts = [titles[i] for i in idx]
        if len(cluster_texts) > MAX_TFIDF_PER_CLUSTER:
            pick = rng.choice(len(cluster_texts), size=MAX_TFIDF_PER_CLUSTER, replace=False)
            cluster_texts = [cluster_texts[i] for i in sorted(pick)]
        if not cluster_texts:
            out[int(cid)] = []
            continue
        try:
            vec = TfidfVectorizer(
                max_features=80,
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
                min_df=1,
            )
            tfidf = vec.fit_transform(cluster_texts)
            scores = np.asarray(tfidf.sum(axis=0)).ravel()
            terms = np.array(vec.get_feature_names_out())
            if scores.size == 0:
                out[int(cid)] = []
                continue
            top_idx = np.argsort(scores)[::-1][:top_n]
            out[int(cid)] = [str(terms[i]) for i in top_idx if scores[i] > 0]
        except ValueError as exc:
            logger.warning("TF-IDF failed for cluster %s: %s", cid, exc)
            out[int(cid)] = []
    return out


class InterestShiftLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.1,
        )
        self.proj = nn.Linear(hidden_dim, 64)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_ln(x)
        out, _ = self.lstm(x)
        t = out.size(1)
        span = max(1, t // 3)
        early = out[:, :span, :].mean(dim=1)
        late = out[:, -span:, :].mean(dim=1)
        return self.proj(early), self.proj(late)


def _cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    cos = float(np.dot(a, b) / (na * nb))
    return float(max(0.0, min(1.0, 1.0 - cos)))


def sequence_interest_shift(
    embedding_matrix: np.ndarray,
    segment_fraction: float = 0.25,
) -> Dict[str, Any]:
    n, d = embedding_matrix.shape
    result: Dict[str, Any] = {
        "sequence_length": int(n),
        "embedding_dim": int(d),
        "embedding_drift": None,
        "lstm_drift": None,
        "combined_shift_score": None,
        "notes": [],
    }
    if n < 4:
        result["notes"].append("Too few videos for a reliable trajectory; scores are approximate.")
        seg = max(1, n // 2)
    else:
        seg = max(1, int(round(n * segment_fraction)))

    early_emb = embedding_matrix[:seg].mean(axis=0)
    late_emb = embedding_matrix[-seg:].mean(axis=0)
    emb_drift = _cosine_drift(early_emb, late_emb)
    result["embedding_drift"] = round(emb_drift, 4)

    if n > MAX_LSTM_SEQ:
        ix = np.unique(np.linspace(0, n - 1, num=MAX_LSTM_SEQ, dtype=int))
        lstm_in = embedding_matrix[ix].copy()
        result["notes"].append(
            f"LSTM trajectory subsampled to {len(ix)} steps for speed (full n={n})."
        )
    else:
        lstm_in = embedding_matrix

    device = torch.device("cpu")
    lstm = InterestShiftLSTM(input_dim=d, hidden_dim=min(128, d), num_layers=1)
    lstm.eval()
    with torch.no_grad():
        x = torch.from_numpy(lstm_in).float().unsqueeze(0).to(device)
        early_z, late_z = lstm(x)
        ez = early_z.cpu().numpy().squeeze()
        lz = late_z.cpu().numpy().squeeze()
        lstm_drift = _cosine_drift(ez, lz)
    result["lstm_drift"] = round(lstm_drift, 4)
    result["combined_shift_score"] = round(float(0.75 * emb_drift + 0.25 * lstm_drift), 4)
    if emb_drift < 0.02:
        result["notes"].append("Very stable interests across the sampled window.")
    elif emb_drift > 0.15:
        result["notes"].append("Noticeable shift between early and late watching patterns.")
    return result


def run_pipeline(
    df: pd.DataFrame,
    n_clusters: int = 12,
    random_state: int = 42,
    include_per_video: bool = True,
    progress_callback: ProgressCb = None,
    session_id: str = "",
    embedding_batch_size: int = 512,
    use_embedding_cache: bool = True,
) -> Dict[str, Any]:
    """
    Full ML + aggregation pass. ``progress_callback(step, percent, partial_insights_or_none)``.

    Phases (for UX): fast table stats → keyword categories → embeddings + clusters → narrative.
    ``partial_insights`` is only passed for early milestones (small JSON for disk snapshot / polling).
    """
    empty_narrative_input = {
        "summary": {"total_videos_analyzed": 0},
        "behavior": {"time_patterns": {}},
        "clusters": [],
        "per_video": [],
    }

    def _p(step: str, pct: int, partial: Optional[Dict[str, Any]] = None) -> None:
        if progress_callback:
            progress_callback(step, int(max(0, min(100, pct))), partial)

    if df is None or df.empty:
        empty = {
            "top_categories": [],
            "top_creators": [],
            "clusters": [],
            "time_patterns": get_time_patterns(pd.DataFrame()),
            "monthly_trends": {},
            "shift_points": [],
            "journey_summary": "",
            "narrative": generate_narrative_insights(empty_narrative_input),
            "cinematic_summary": [],
            "wrapped_story": [],
            "behavior": {"interest_shift": {}, "late_night_heavy": False, "peak_time_bucket": None},
            "model_artifacts": {
                "embedding_shape": [0, 384],
                "sentence_transformer": "all-MiniLM-L6-v2",
                "classifier": "keyword_fast_feedback",
                "classification_labels": CLASSIFICATION_LABELS,
            },
            "per_video": [],
        }
        return empty

    titles = df["title"].astype(str).tolist()

    # --- Phase 1: instant aggregates (no ML) ---
    time_full = get_time_patterns(df)
    monthly = {k: int(v) for k, v in (time_full.get("month_counts") or {}).items()}
    top_creators_rows = [{"channel": c, "watch_count": n} for c, n in get_top_creators(df, 25)]
    bucket_counts = time_full.get("time_bucket_counts") or {}
    peak_bucket = max(bucket_counts, key=bucket_counts.get) if bucket_counts else None

    time_patterns_api = {
        "peak_hour": time_full.get("peak_hour"),
        "peak_day": time_full.get("peak_day"),
        "late_night_percentage": time_full.get("late_night_percentage"),
        "late_night_ratio": time_full.get("late_night_ratio"),
        "most_active_month": time_full.get("most_active_month"),
        "hourly_counts": time_full.get("hourly_counts"),
        "day_of_week_counts": time_full.get("day_of_week_counts"),
        "month_counts": time_full.get("month_counts"),
        "time_bucket_counts": time_full.get("time_bucket_counts"),
    }

    summary_fast: Dict[str, Any] = {
        "total_videos_analyzed": int(len(df)),
        "top_categories": [],
        "top_creators": top_creators_rows[:10],
    }

    partial_stats: Dict[str, Any] = {
        "top_categories": [],
        "top_creators": top_creators_rows[:10],
        "clusters": [],
        "time_patterns": time_patterns_api,
        "monthly_trends": monthly,
        "monthly_trends_sorted": [{"month": m, "count": c} for m, c in sorted(monthly.items())],
        "shift_points": [],
        "journey_summary": "",
        "narrative": {},
        "cinematic_summary": [],
        "wrapped_story": [],
        "behavior": {
            "interest_shift": {},
            "late_night_heavy": float(time_full.get("late_night_ratio") or 0) >= 0.15,
            "peak_time_bucket": peak_bucket,
        },
        "summary": summary_fast,
        "model_artifacts": {
            "embedding_shape": [int(len(df)), 384],
            "sentence_transformer": "all-MiniLM-L6-v2",
            "classifier": "keyword_fast_feedback",
            "classification_labels": CLASSIFICATION_LABELS,
            "pipeline_phase": "stats",
        },
        "per_video": [],
    }
    _p("loading", 5, partial_stats)

    # --- Phase 2: keyword categories (fast) ---
    categories, cat_scores = classify_categories_scored(titles)
    cat_counts: Dict[str, int] = {}
    cat_weighted: Dict[str, float] = {}
    for c, s in zip(categories, cat_scores):
        cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_weighted[c] = cat_weighted.get(c, 0.0) + float(s)
    top_categories = sorted(
        [
            {"category": k, "count": v, "avg_confidence": round(cat_weighted[k] / v, 4)}
            for k, v in cat_counts.items()
        ],
        key=lambda x: x["count"],
        reverse=True,
    )

    summary_with_cats = {**summary_fast, "top_categories": top_categories}
    narrative_input_categories = {
        "summary": summary_with_cats,
        "behavior": {
            "time_patterns": time_full,
            "peak_time_bucket": peak_bucket,
            "late_night_heavy": float(time_full.get("late_night_ratio") or 0) >= 0.15,
            "interest_shift": {},
        },
        "clusters": [],
        "per_video": [],
    }
    narrative_categories = generate_narrative_insights(narrative_input_categories)
    top_cat_name_early = top_categories[0]["category"] if top_categories else ""
    cinematic_categories = generate_cinematic_summary(
        top_category=top_cat_name_early,
        top_creators=top_creators_rows,
        peak_time=str(peak_bucket or time_full.get("peak_hour") or "unknown"),
        cluster_summary=[],
        persona=narrative_categories.get("persona", ""),
    )

    partial_categories: Dict[str, Any] = {
        **partial_stats,
        "top_categories": top_categories,
        "summary": summary_with_cats,
        "narrative": narrative_categories,
        "cinematic_summary": cinematic_categories,
        "model_artifacts": {
            **(partial_stats.get("model_artifacts") or {}),
            "pipeline_phase": "categories",
        },
    }
    _p("embeddings", 20, partial_categories)

    # --- Phase 3: embeddings + clustering ---
    if use_embedding_cache and os.environ.get("YOUTUBE_WRAPPED_DISABLE_EMBED_CACHE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        embeddings = get_cached_embeddings(
            titles,
            batch_size=embedding_batch_size,
            cache_extra=session_id or "",
        )
    else:
        embeddings = get_embeddings(titles, batch_size=embedding_batch_size)
    _p("embeddings", 40, None)

    _p("clustering", 50, None)
    labels = cluster_videos(embeddings, n_clusters=n_clusters, random_state=random_state)
    _p("clustering", 60, None)

    keywords_by_cluster = label_clusters(df, labels)

    seq_shift = sequence_interest_shift(embeddings)

    cluster_list: List[Dict[str, Any]] = []
    for cid in sorted(keywords_by_cluster.keys()):
        mask = labels == cid
        size = int(mask.sum())
        kws = keywords_by_cluster.get(cid, [])
        sample_titles = df.loc[mask, "title"].head(5).tolist()
        cluster_list.append(
            {
                "cluster_id": int(cid),
                "size": size,
                "keywords": kws,
                "label_keywords": kws,
                "cluster_label": " / ".join(kws[:3]) if kws else "mixed interests",
                "sample_titles": sample_titles,
            }
        )

    per_video: List[Dict[str, Any]] = []
    if include_per_video:
        _p("per_video", 70, None)
        for i in range(len(df)):
            row = df.iloc[i]
            per_video.append(
                {
                    "title": str(row["title"]),
                    "channel": str(row["channel"]),
                    "timestamp_iso": pd.Timestamp(row["timestamp"]).isoformat(),
                    "cluster_id": int(labels[i]) if len(labels) else -1,
                    "category": categories[i] if i < len(categories) else None,
                    "category_confidence": round(cat_scores[i], 4) if i < len(cat_scores) else None,
                }
            )

    _p("insights", 80, None)
    core_insights_for_narrative = {
        "summary": {
            "total_videos_analyzed": int(len(df)),
            "top_categories": top_categories,
            "top_creators": top_creators_rows[:10],
        },
        "behavior": {
            "time_patterns": time_full,
            "peak_time_bucket": peak_bucket,
            "late_night_heavy": float(time_full.get("late_night_ratio") or 0) >= 0.15,
            "interest_shift": seq_shift,
        },
        "clusters": cluster_list,
        "per_video": per_video,
    }
    cluster_name_map = {
        int(cluster["cluster_id"]): str(cluster.get("cluster_label") or "mixed interests")
        for cluster in cluster_list
    }
    shift_points, journey_summary = detect_interest_shifts(
        df=df,
        embeddings=embeddings,
        cluster_labels=labels,
        cluster_label_names=cluster_name_map,
    )
    narrative = generate_narrative_insights(core_insights_for_narrative)
    top_cat_name = top_categories[0]["category"] if top_categories else ""
    cinematic = generate_cinematic_summary(
        top_category=top_cat_name,
        top_creators=top_creators_rows,
        peak_time=str(peak_bucket or time_full.get("peak_hour") or "unknown"),
        cluster_summary=cluster_list,
        persona=narrative.get("persona", ""),
    )

    result = {
        "top_categories": top_categories,
        "top_creators": top_creators_rows[:10],
        "clusters": cluster_list,
        "time_patterns": time_patterns_api,
        "monthly_trends": monthly,
        "monthly_trends_sorted": [{"month": m, "count": c} for m, c in sorted(monthly.items())],
        "shift_points": shift_points,
        "journey_summary": journey_summary,
        "narrative": narrative,
        "cinematic_summary": cinematic,
        "behavior": {
            "interest_shift": seq_shift,
            "late_night_heavy": core_insights_for_narrative["behavior"]["late_night_heavy"],
            "peak_time_bucket": peak_bucket,
        },
        "summary": core_insights_for_narrative["summary"],
        "model_artifacts": {
            "embedding_shape": [int(x) for x in embeddings.shape],
            "sentence_transformer": "all-MiniLM-L6-v2",
            "classifier": "keyword_fast_feedback",
            "classification_labels": CLASSIFICATION_LABELS,
            "n_clusters_effective": int(np.unique(labels).size),
            "embedding_cache": use_embedding_cache,
            "pipeline_phase": "complete",
        },
        "per_video": per_video,
    }

    _p("insights", 85, None)
    if llm_insights_available():
        ai_input = {
            "summary": result["summary"],
            "top_categories": result["top_categories"],
            "top_creators": result["top_creators"],
            "clusters": result["clusters"],
            "time_patterns": result["time_patterns"],
            "monthly_trends": result["monthly_trends"],
            "journey_summary": result["journey_summary"],
        }
        try:
            result["ai_insights"] = generate_ai_insights(ai_input)
            result["wrapped_story"] = generate_wrapped_story(result["ai_insights"], ai_input)
        except Exception as exc:
            logger.warning("Skipping external AI insights: %s", exc)
            result["ai_insights_error"] = str(exc)
            result["wrapped_story"] = []
    else:
        result["wrapped_story"] = []

    _p("done", 100, None)
    return result
