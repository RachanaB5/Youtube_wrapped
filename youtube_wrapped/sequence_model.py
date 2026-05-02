"""
Monthly interest-shift detection from chronological video embeddings (LSTM hidden states).

The LSTM is used in inference-only mode with default/random weights—sufficient for
comparing relative changes in sequential hidden representations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

ClusterLabelMap = Mapping[Any, str]


class InterestShiftDetector(nn.Module):
    """
    Single-layer LSTM over a chronological embedding sequence.

    ``find_shift_points`` flags timesteps where consecutive hidden states diverge
    (cosine distance > ``threshold``).
    """

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[seq_len, embedding_dim]`` or ``[batch, seq_len, embedding_dim]``.
        Returns:
            All layer outputs per timestep, shape ``[seq_len, hidden_dim]`` (no batch)
            or ``[batch, seq_len, hidden_dim]``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        out, _ = self.lstm(x)
        return out

    def find_shift_points(
        self,
        embeddings: np.ndarray,
        threshold: float = 0.4,
    ) -> List[int]:
        """
        Indices ``t`` (0-based along the sequence) where the transition from ``t-1``
        to ``t`` has cosine distance ``> threshold`` between LSTM outputs.

        The first timestep has no predecessor and is never a shift index.
        """
        if embeddings is None or len(embeddings) < 2:
            return []

        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
            if x.dim() != 2 or x.shape[-1] != self.embedding_dim:
                raise ValueError(
                    f"Expected embeddings shaped [seq_len, {self.embedding_dim}], got {tuple(x.shape)}"
                )
            out = self.forward(x).squeeze(0)  # [seq, hidden]
            shift_at: List[int] = []
            for t in range(1, out.shape[0]):
                sim = F.cosine_similarity(out[t : t + 1], out[t - 1 : t], dim=-1).clamp(-1.0, 1.0)
                dist = float(1.0 - sim.item())
                if dist > threshold:
                    shift_at.append(t)
            return shift_at


def _dominant_cluster_for_indices(cluster_labels: np.ndarray, idx: np.ndarray) -> Any:
    if idx.size == 0:
        return None
    slice_labs = cluster_labels[idx]
    mode = pd.Series(slice_labs).mode()
    if mode.empty:
        return None
    val = mode.iloc[0]
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def _cluster_display(
    cluster_id: Any,
    label_names: Optional[ClusterLabelMap],
) -> str:
    if cluster_id is None:
        return "mixed"
    if label_names is not None and cluster_id in label_names:
        return str(label_names[cluster_id])
    return str(cluster_id)


def detect_interest_shifts(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: Optional[np.ndarray] = None,
    embedding_dim: int = 384,
    hidden_dim: int = 128,
    shift_threshold: float = 0.4,
    cluster_label_names: Optional[ClusterLabelMap] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Aggregate embeddings by calendar month (``df['month']``), run the LSTM on the
    sequence of monthly mean embeddings, and report transitions with large hidden-state
    change.

    Args:
        df: Watch history DataFrame (same row order as ``embeddings``).
        embeddings: ``(n_rows, embedding_dim)`` aligned with ``df``.
        cluster_labels: Optional ``(n_rows,)`` cluster id per row (same order as ``df``).
        cluster_label_names: Optional map ``cluster_id -> label`` for journey text.
        embedding_dim: Expected width (overridden by actual ``embeddings`` width if different).
        hidden_dim: LSTM hidden size.
        shift_threshold: Cosine distance threshold on consecutive LSTM outputs (default ``0.4``).

    Returns:
        * ``shifts``: list of ``{month, shift_score, dominant_cluster_before, dominant_cluster_after}``
        * ``journey_summary``: one-line story via ``summarize_journey`` (pass ``cluster_label_names`` for readable cluster names).
    """
    if df is None or df.empty:
        return [], "No watch history to analyze."

    n = len(df)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.shape[0] != n:
        raise ValueError(f"embeddings row count {emb.shape[0]} != len(df) {n}")

    if "month" not in df.columns:
        raise ValueError("df must contain a 'month' column (e.g. YYYY-MM).")

    if cluster_labels is None:
        cluster_labels = np.full(n, -1, dtype=np.int64)
    else:
        cluster_labels = np.asarray(cluster_labels)
        if cluster_labels.shape[0] != n:
            raise ValueError("cluster_labels length must match df length.")

    months_ordered = sorted(df["month"].astype(str).unique())
    monthly_avgs: List[np.ndarray] = []
    monthly_dominant: List[Any] = []

    for m in months_ordered:
        idx = np.where(df["month"].astype(str).values == m)[0]
        monthly_avgs.append(emb[idx].mean(axis=0))
        monthly_dominant.append(_dominant_cluster_for_indices(cluster_labels, idx))

    month_seq = np.stack(monthly_avgs, axis=0)  # [n_months, dim]
    data_dim = int(month_seq.shape[1])
    if data_dim != embedding_dim:
        logger.debug("Using embedding dim %s from data (constructor arg was %s)", data_dim, embedding_dim)

    detector = InterestShiftDetector(embedding_dim=data_dim, hidden_dim=hidden_dim)
    detector.eval()

    shift_step_indices = set(detector.find_shift_points(month_seq, threshold=shift_threshold))

    shifts: List[Dict[str, Any]] = []
    with torch.no_grad():
        x = torch.from_numpy(month_seq).unsqueeze(0).float()
        out = detector.forward(x).squeeze(0)  # [n_months, hidden]
        for t in sorted(shift_step_indices):
            if t < 1:
                continue
            sim = F.cosine_similarity(out[t : t + 1], out[t - 1 : t], dim=-1).clamp(-1.0, 1.0)
            dist = float(1.0 - sim.item())
            shifts.append(
                {
                    "month": months_ordered[t],
                    "shift_score": round(dist, 4),
                    "dominant_cluster_before": monthly_dominant[t - 1],
                    "dominant_cluster_after": monthly_dominant[t],
                }
            )

    if not shifts:
        start = _cluster_display(monthly_dominant[0], cluster_label_names) if monthly_dominant else None
        if len(months_ordered) <= 1:
            summary = "Only one month in this export—no timeline to detect a pivot."
        else:
            summary = (
                f"Interests stayed visually steady month to month (no LSTM hidden-state jump over {shift_threshold}). "
                f"Dominant lane in {months_ordered[0]} was {start}."
            )
        return shifts, summary

    summary = summarize_journey(shifts, cluster_label_names)
    return shifts, summary


def summarize_journey(
    shift_points: List[Dict[str, Any]],
    cluster_label_names: Optional[Union[ClusterLabelMap, List[str]]] = None,
) -> str:
    """
    Build a short, readable interest "journey" from monthly shift records.

    Args:
        shift_points: Dicts with keys ``month``, ``dominant_cluster_before``,
            ``dominant_cluster_after`` (and optionally ``shift_score``).
        cluster_label_names: Map ``cluster_id -> short label``, or list of names
            indexed by cluster id (only works for ``int`` ids in range).
    """
    if not shift_points:
        return "Not enough shift signal to summarize—try a longer watch history export."

    def resolve(cid: Any) -> str:
        if cid is None:
            return "mixed interests"
        if isinstance(cid, float) and np.isnan(cid):
            return "mixed interests"
        if isinstance(cluster_label_names, list):
            if isinstance(cid, int) and 0 <= cid < len(cluster_label_names):
                return str(cluster_label_names[cid])
            return str(cid)
        if isinstance(cluster_label_names, dict) or hasattr(cluster_label_names, "get"):
            return _cluster_display(cid, cluster_label_names)  # type: ignore[arg-type]
        return str(cid)

    ordered = sorted(shift_points, key=lambda d: str(d.get("month", "")))
    first = ordered[0]
    before0 = resolve(first.get("dominant_cluster_before"))
    after0 = resolve(first.get("dominant_cluster_after"))
    month0 = first.get("month", "?")
    pieces = [f"Started with {before0}", f"pivoted to {after0} in {month0}"]
    prev_after = first.get("dominant_cluster_after")
    for sp in ordered[1:]:
        if sp.get("dominant_cluster_after") == prev_after:
            continue
        nxt = resolve(sp.get("dominant_cluster_after"))
        m = sp.get("month", "?")
        pieces.append(f"{nxt} phase in {m}")
        prev_after = sp.get("dominant_cluster_after")

    return " → ".join(pieces) + "."
