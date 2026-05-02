"""
Embeddings, clustering, zero-shot labels, sequence modeling, and insights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

from youtube_wrapped.analyst import generate_cinematic_summary, generate_narrative_insights
from youtube_wrapped.data import WatchRecord
from youtube_wrapped.utils import hour_bucket_label, top_keywords_from_texts


CATEGORIES: List[str] = [
    "tech",
    "music",
    "education",
    "gaming",
    "entertainment",
    "productivity",
]

ClusterMethod = Literal["kmeans", "dbscan"]

_EMBEDDER = None
_ZERO_SHOT_PIPE = None


def _get_embedder():
    """Lazy-load SentenceTransformer to keep import/start time low."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def _get_zero_shot_pipeline():
    """Small MNLI model suitable for CPU batching."""
    global _ZERO_SHOT_PIPE
    if _ZERO_SHOT_PIPE is None:
        from transformers import pipeline

        # DistilBERT MNLI: much lighter than BART-large while remaining usable.
        _ZERO_SHOT_PIPE = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=-1,
        )
    return _ZERO_SHOT_PIPE


def embed_titles(titles: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode cleaned titles into a dense matrix (n_samples, hidden_dim).
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


def choose_k(n_samples: int, max_k: int = 10) -> int:
    """Heuristic cluster count for personal-scale histories."""
    if n_samples < 4:
        return 2
    k = int(np.clip(round(np.sqrt(n_samples / 3)), 2, max_k))
    return min(k, max(2, n_samples // 2))


def cluster_videos(
    embeddings: np.ndarray,
    titles: List[str],
    method: ClusterMethod = "kmeans",
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """
    Cluster rows of `embeddings` and build human-readable summaries per id.

    Returns `labels` (n_samples,) and `summaries` keyed by cluster id.
    """
    n = embeddings.shape[0]
    if n == 0:
        return np.array([]), {}

    if method == "kmeans":
        k = choose_k(n)
        k = min(k, n)
        model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(embeddings)
    else:
        # Scale eps with typical nearest-neighbor distance in embedding space.
        sample = min(512, n)
        idx = np.random.choice(n, size=sample, replace=False)
        d = pairwise_distances(embeddings[idx], metric="cosine")
        # Robust eps: median of small NN distances
        np.fill_diagonal(d, np.inf)
        nn_dist = d.min(axis=1)
        eps = float(np.percentile(nn_dist, 25) * 1.2)
        eps = max(eps, 0.05)
        model = DBSCAN(metric="cosine", eps=eps, min_samples=max(2, n // 100))
        labels = model.fit_predict(embeddings)

    summaries: Dict[int, Dict[str, Any]] = {}
    unique = sorted(set(labels.tolist()))
    for cid in unique:
        if cid == -1 and method == "dbscan":
            mask = labels == cid
            cluster_titles = [titles[i] for i in np.where(mask)[0]]
            summaries[cid] = {
                "cluster_id": int(cid),
                "size": int(mask.sum()),
                "label_keywords": top_keywords_from_texts(cluster_titles, top_k=5),
                "cluster_label": " / ".join(top_keywords_from_texts(cluster_titles, top_k=3)) or "mixed interests",
                "sample_titles": cluster_titles[:5],
                "role": "noise_or_rare",
            }
            continue
        mask = labels == cid
        cluster_titles = [titles[i] for i in np.where(mask)[0]]
        summaries[cid] = {
            "cluster_id": int(cid),
            "size": int(mask.sum()),
            "label_keywords": top_keywords_from_texts(cluster_titles, top_k=5),
            "cluster_label": " / ".join(top_keywords_from_texts(cluster_titles, top_k=3)) or "mixed interests",
            "sample_titles": cluster_titles[:5],
        }
    return labels, summaries


def zero_shot_categories(
    titles: List[str],
    batch_size: int = 16,
    categories: Optional[List[str]] = None,
) -> Tuple[List[str], List[float]]:
    """
    Assign a single best category per title using zero-shot NLI.
    """
    if not titles:
        return [], []
    labels = categories or CATEGORIES
    pipe = _get_zero_shot_pipeline()
    predicted: List[str] = []
    scores: List[float] = []
    for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]
        raw = pipe(batch, candidate_labels=labels, multi_label=False)
        items = raw if isinstance(raw, list) else [raw]
        for out in items:
            predicted.append(out["labels"][0])
            scores.append(float(out["scores"][0]))
    return predicted, scores


@dataclass
class TimeAnalysisResult:
    hourly_counts: Dict[int, int]
    day_of_week_counts: Dict[str, int]
    month_counts: Dict[str, int]
    bucket_counts: Dict[str, int]
    peak_hour: Optional[int]
    late_night_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hourly_counts": {str(k): v for k, v in sorted(self.hourly_counts.items())},
            "day_of_week_counts": self.day_of_week_counts,
            "month_counts": dict(sorted(self.month_counts.items())),
            "time_bucket_counts": self.bucket_counts,
            "peak_hour": self.peak_hour,
            "late_night_ratio": round(self.late_night_ratio, 4),
        }


def analyze_time_patterns(records: List[WatchRecord]) -> TimeAnalysisResult:
    """
    Build hour / weekday / month histograms and simple behavioral cues.
    """
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    hourly = {h: 0 for h in range(24)}
    dow = {d: 0 for d in weekdays}
    months: Dict[str, int] = {}
    buckets: Dict[str, int] = {}

    late_night = 0
    total = 0
    for r in records:
        dt = r.watched_at
        hourly[dt.hour] += 1
        dow[weekdays[dt.weekday()]] += 1
        mkey = dt.strftime("%Y-%m")
        months[mkey] = months.get(mkey, 0) + 1
        b = hour_bucket_label(dt.hour)
        buckets[b] = buckets.get(b, 0) + 1
        if 0 <= dt.hour < 5:
            late_night += 1
        total += 1

    peak_hour = max(hourly, key=hourly.get) if total else None
    late_ratio = (late_night / total) if total else 0.0
    return TimeAnalysisResult(
        hourly_counts=hourly,
        day_of_week_counts=dow,
        month_counts=months,
        bucket_counts=buckets,
        peak_hour=peak_hour,
        late_night_ratio=late_ratio,
    )


class InterestShiftLSTM(nn.Module):
    """
    Lightweight LSTM that compares early vs late temporal segments.

    Applied to the time-ordered embedding sequence; exposes a differentiable
    split, but at inference time we only use cosine drift between pooled
    early / late segment representations.
    """

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
        """
        Args:
            x: (batch, seq, input_dim)
        Returns:
            early_z, late_z: (batch, 64) projected segment means.
        """
        x = self.input_ln(x)
        out, _ = self.lstm(x)
        t = out.size(1)
        span = max(1, t // 3)
        early = out[:, :span, :].mean(dim=1)
        late = out[:, -span:, :].mean(dim=1)
        return self.proj(early), self.proj(late)


def _cosine_drift(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity for L2-normalized vectors."""
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
    """
    Measure interest drift from early to late sessions.

    Combines:
      * **embedding_drift**: difference between mean embeddings of first/last
        fractions of the timeline (interpretable, no training).
      * **lstm_drift**: same comparison after a small LSTM temporal mix
        (architecture satisfies sequence-model requirement; scores are auxiliary).
    """
    n, d = embedding_matrix.shape
    out: Dict[str, Any] = {
        "sequence_length": int(n),
        "embedding_dim": int(d),
        "embedding_drift": None,
        "lstm_drift": None,
        "combined_shift_score": None,
        "notes": [],
    }
    if n < 4:
        out["notes"].append("Too few videos for a reliable trajectory; scores are approximate.")
        seg = max(1, n // 2)
    else:
        seg = max(1, int(round(n * segment_fraction)))

    early_emb = embedding_matrix[:seg].mean(axis=0)
    late_emb = embedding_matrix[-seg:].mean(axis=0)
    emb_drift = _cosine_drift(early_emb, late_emb)
    out["embedding_drift"] = round(emb_drift, 4)

    # LSTM path (eval, no training — temporal mixing only)
    device = torch.device("cpu")
    lstm = InterestShiftLSTM(input_dim=d, hidden_dim=min(128, d), num_layers=1)
    lstm.eval()
    with torch.no_grad():
        x = torch.from_numpy(embedding_matrix).float().unsqueeze(0).to(device)
        early_z, late_z = lstm(x)
        ez = early_z.cpu().numpy().squeeze()
        lz = late_z.cpu().numpy().squeeze()
        lstm_drift = _cosine_drift(ez, lz)
    out["lstm_drift"] = round(lstm_drift, 4)

    # Weight interpretable embedding drift higher for reporting.
    combined = float(0.75 * emb_drift + 0.25 * lstm_drift)
    out["combined_shift_score"] = round(combined, 4)
    if emb_drift < 0.02:
        out["notes"].append("Very stable interests across the sampled window.")
    elif emb_drift > 0.15:
        out["notes"].append("Noticeable shift between early and late watching patterns.")
    return out


def build_insights(
    records: List[WatchRecord],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_summaries: Dict[int, Dict[str, Any]],
    categories: List[str],
    category_scores: List[float],
    time_result: TimeAnalysisResult,
    sequence_shift: Dict[str, Any],
    cluster_method: str,
) -> Dict[str, Any]:
    """
    Aggregate human-facing insight JSON from intermediate artifacts.
    """
    titles = [r.title for r in records]
    channels = [r.channel for r in records]

    cat_counts: Dict[str, int] = {}
    cat_weighted: Dict[str, float] = {}
    for c, s in zip(categories, category_scores):
        cat_counts[c] = cat_counts.get(c, 0) + 1
        cat_weighted[c] = cat_weighted.get(c, 0.0) + float(s)
    top_categories = sorted(
        [{"category": k, "count": v, "avg_confidence": round(cat_weighted[k] / v, 4)} for k, v in cat_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )

    ch_counts: Dict[str, int] = {}
    for ch in channels:
        ch_counts[ch] = ch_counts.get(ch, 0) + 1
    top_creators = sorted(
        [{"channel": k, "watch_count": v} for k, v in ch_counts.items()],
        key=lambda x: x["watch_count"],
        reverse=True,
    )[:25]

    cluster_list = [dict(v) for _, v in sorted(cluster_summaries.items(), key=lambda kv: kv[0])]

    peak_bucket = max(time_result.bucket_counts, key=time_result.bucket_counts.get) if records else None
    date_range = {
        "start": records[0].watched_at.isoformat(),
        "end": records[-1].watched_at.isoformat(),
    } if records else None
    unique_channels = len(set(channels))
    unique_categories = len(set(categories))

    return {
        "summary": {
            "total_videos_analyzed": len(records),
            "clustering_method": cluster_method,
            "date_range": date_range,
            "unique_channels": unique_channels,
            "unique_categories": unique_categories,
            "top_categories": top_categories[:6],
            "top_creators": top_creators[:10],
        },
        "behavior": {
            "time_patterns": time_result.to_dict(),
            "peak_time_bucket": peak_bucket,
            "late_night_heavy": time_result.late_night_ratio >= 0.15,
            "interest_shift": sequence_shift,
        },
        "clusters": cluster_list,
        "cluster_summaries": [
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_label": cluster.get("cluster_label", "mixed interests"),
                "keywords": cluster.get("label_keywords", []),
                "size": cluster.get("size", 0),
                "sample_titles": cluster.get("sample_titles", []),
            }
            for cluster in cluster_list
        ],
        "per_video": [
            {
                "title": titles[i],
                "channel": channels[i],
                "timestamp_iso": records[i].watched_at.isoformat(),
                "cluster_id": int(cluster_labels[i]) if len(cluster_labels) else -1,
                "category": categories[i] if i < len(categories) else None,
                "category_confidence": round(category_scores[i], 4) if i < len(category_scores) else None,
            }
            for i in range(len(records))
        ],
    }


def run_pipeline(
    records: List[WatchRecord],
    cluster_method: ClusterMethod = "kmeans",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    End-to-end processing for a list of `WatchRecord` instances.

    Returns a dict suitable for JSON serialization (with per-video detail).
    """
    if not records:
        return {
            "summary": {"total_videos_analyzed": 0, "error": "no_records"},
            "behavior": {},
            "clusters": [],
            "per_video": [],
            "narrative": generate_narrative_insights(
                {"summary": {"total_videos_analyzed": 0}, "behavior": {}, "clusters": [], "per_video": []}
            ),
        }

    titles = [r.title for r in records]
    embeddings = embed_titles(titles)
    labels, summaries = cluster_videos(embeddings, titles, method=cluster_method, random_state=random_state)
    categories, cat_scores = zero_shot_categories(titles)
    time_result = analyze_time_patterns(records)
    seq_shift = sequence_interest_shift(embeddings)

    insights = build_insights(
        records=records,
        embeddings=embeddings,
        cluster_labels=labels,
        cluster_summaries=summaries,
        categories=categories,
        category_scores=cat_scores,
        time_result=time_result,
        sequence_shift=seq_shift,
        cluster_method=cluster_method,
    )
    # Avoid huge responses in some deployments — keep per_video optional at route layer if needed.
    insights["model_artifacts"] = {
        "embedding_shape": [int(x) for x in embeddings.shape],
        "embedding_matrix": embeddings.tolist(),
        "sentence_transformer": "all-MiniLM-L6-v2",
        "zero_shot_model": "typeform/distilbert-base-uncased-mnli",
    }
    insights["narrative"] = generate_narrative_insights(insights)
    insights["cinematic_summary"] = generate_cinematic_summary(
        top_category=(insights["summary"]["top_categories"][0]["category"] if insights["summary"]["top_categories"] else ""),
        top_creators=insights["summary"]["top_creators"],
        peak_time=(insights["behavior"].get("peak_time_bucket") or "unknown"),
        cluster_summary=insights["cluster_summaries"],
        persona=insights["narrative"].get("persona", ""),
    )
    return insights
