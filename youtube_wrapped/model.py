"""
Embeddings, KMeans clustering, TF-IDF cluster labels, zero-shot categories, sequence drift.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from youtube_wrapped.analyst import generate_cinematic_summary, generate_narrative_insights
from youtube_wrapped.insight_generator import generate_insights as generate_ai_insights
from youtube_wrapped.insight_generator import generate_wrapped_story
from youtube_wrapped.sequence_model import detect_interest_shifts
from youtube_wrapped.utils import get_time_patterns, get_top_creators

logger = logging.getLogger(__name__)

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

_EMBEDDER = None
_ZERO_SHOT_PIPE = None


def llm_insights_available() -> bool:
    """Return True when either configured LLM insight provider is available."""
    import os

    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _ensure_hf_cache_dir() -> str:
    """Keep Hugging Face caches inside the project by default."""
    cache_dir = os.environ.get("HF_HOME")
    if not cache_dir:
        cache_dir = str((Path(__file__).resolve().parent.parent / ".hf-cache").resolve())
        os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _zero_shot_device() -> int:
    try:
        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return -1


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        cache_dir = _ensure_hf_cache_dir()
        logger.info("Loading SentenceTransformer all-MiniLM-L6-v2")
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=cache_dir)
    return _EMBEDDER


def _get_zero_shot_pipeline():
    global _ZERO_SHOT_PIPE
    if _ZERO_SHOT_PIPE is None:
        from transformers import pipeline

        _ensure_hf_cache_dir()
        device = _zero_shot_device()
        logger.info("Loading zero-shot pipeline facebook/bart-large-mnli (device=%s)", device)
        _ZERO_SHOT_PIPE = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device,
        )
    return _ZERO_SHOT_PIPE


def get_embeddings(titles: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode cleaned titles with SentenceTransformer ``all-MiniLM-L6-v2``.

    Returns ``(n_samples, 384)`` float32 array (empty inputs → shape ``(0, 384)``).
    """
    if not titles:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_embedder()
    vectors = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def cluster_videos(embeddings: np.ndarray, n_clusters: int = 8, random_state: int = 42) -> np.ndarray:
    """
    KMeans on embedding rows. Uses ``k = min(n_clusters, n_samples)`` with ``k >= 1``.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    k = max(1, min(int(n_clusters), n))
    logger.info("Clustering %d rows into k=%d", n, k)
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    return model.fit_predict(embeddings).astype(np.int32)


def label_clusters(df: pd.DataFrame, labels: np.ndarray, top_n: int = 5) -> Dict[int, List[str]]:
    """
    For each cluster id, fit TF-IDF on that cluster's titles and return top ``top_n`` terms.
    """
    if df is None or df.empty or len(labels) == 0:
        return {}
    titles = df["title"].astype(str).tolist()
    out: Dict[int, List[str]] = {}
    for cid in sorted(np.unique(labels).tolist()):
        idx = np.where(labels == cid)[0].tolist()
        cluster_texts = [titles[i] for i in idx]
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


def classify_categories_scored(
    titles: List[str],
    categories: Optional[List[str]] = None,
    batch_size: int = 4,
) -> Tuple[List[str], List[float]]:
    """
    Zero-shot labels per title with confidences (facebook/bart-large-mnli).

    Small default batch_size keeps CPU/GPU memory predictable.
    """
    if not titles:
        return [], []
    labels = categories or CLASSIFICATION_LABELS
    pipe = _get_zero_shot_pipeline()
    predicted: List[str] = []
    scores: List[float] = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        raw = pipe(batch, candidate_labels=labels, multi_label=False)
        items = raw if isinstance(raw, list) else [raw]
        for out in items:
            predicted.append(str(out["labels"][0]))
            scores.append(float(out["scores"][0]))
    return predicted, scores


def classify_categories(titles: List[str]) -> List[str]:
    """
    Zero-shot classification into the eight Wrapped categories.

    Uses ``facebook/bart-large-mnli``.
    """
    cats, _ = classify_categories_scored(titles)
    return cats


class InterestShiftLSTM(nn.Module):
    """Lightweight LSTM comparing early vs late temporal segments of embeddings."""

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
    """Early vs late cosine drift on raw embeddings plus LSTM-pooled segments."""
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

    device = torch.device("cpu")
    lstm = InterestShiftLSTM(input_dim=d, hidden_dim=min(128, d), num_layers=1)
    lstm.eval()
    with torch.no_grad():
        x = torch.from_numpy(embedding_matrix).float().unsqueeze(0).to(device)
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
    n_clusters: int = 8,
    random_state: int = 42,
    include_per_video: bool = True,
) -> Dict[str, Any]:
    """
    Full ML + aggregation pass. Expects DataFrame from ``load_watch_history``.
    """
    empty_narrative_input = {
        "summary": {"total_videos_analyzed": 0},
        "behavior": {"time_patterns": {}},
        "clusters": [],
        "per_video": [],
    }
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
                "zero_shot_model": "facebook/bart-large-mnli",
                "classification_labels": CLASSIFICATION_LABELS,
            },
            "per_video": [],
        }
        return empty

    titles = df["title"].astype(str).tolist()
    embeddings = get_embeddings(titles)
    labels = cluster_videos(embeddings, n_clusters=n_clusters, random_state=random_state)
    keywords_by_cluster = label_clusters(df, labels)

    try:
        categories, cat_scores = classify_categories_scored(titles)
    except Exception as exc:
        logger.exception("Classification failed: %s", exc)
        raise

    time_full = get_time_patterns(df)
    monthly = {k: int(v) for k, v in (time_full.get("month_counts") or {}).items()}
    top_creators_rows = [{"channel": c, "watch_count": n} for c, n in get_top_creators(df, 25)]

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

    bucket_counts = time_full.get("time_bucket_counts") or {}
    peak_bucket = max(bucket_counts, key=bucket_counts.get) if bucket_counts else None
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
            "zero_shot_model": "facebook/bart-large-mnli",
            "classification_labels": CLASSIFICATION_LABELS,
            "n_clusters_effective": int(np.unique(labels).size),
        },
        "per_video": per_video,
    }
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
    return result
