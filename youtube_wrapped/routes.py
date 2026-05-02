"""
Flask application factory and HTTP routes for YouTube Wrapped.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from youtube_wrapped.data import load_watch_history, summarize_records
from youtube_wrapped.model import run_pipeline


def create_app() -> Flask:
    """
    Build Flask app with upload/process/insights endpoints.

    Uses on-disk ``UPLOAD_FOLDER`` (default ``uploads``) keyed by ``job_id``.
    """
    app = Flask(__name__)
    upload_root = Path(os.environ.get("YOUTUBE_WRAPPED_UPLOAD_DIR", "uploads"))
    upload_root.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(upload_root.resolve())

    @app.post("/upload-history")
    def upload_history():
        """
        Accept Google Takeout JSON (watch history) as multipart or raw JSON body.

        Returns:
            ``job_id`` for ``/process`` and ``/get-insights``, and ``stored_path``.
        """
        job_id = str(uuid.uuid4())
        dest_dir = Path(app.config["UPLOAD_FOLDER"])
        dest_path = dest_dir / f"{job_id}.json"

        if request.content_type and "multipart/form-data" in request.content_type:
            if "file" not in request.files:
                return jsonify({"error": "missing file field 'file'"}), 400
            f = request.files["file"]
            if not f.filename:
                return jsonify({"error": "empty filename"}), 400
            f.save(dest_path)
        else:
            # Raw JSON body
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "expected JSON array or multipart file"}), 400
            with dest_path.open("w", encoding="utf-8") as out:
                json.dump(data, out, ensure_ascii=False)

        try:
            records = load_watch_history(dest_path)
        except (ValueError, json.JSONDecodeError) as exc:
            if dest_path.exists():
                dest_path.unlink()
            return jsonify({"error": f"invalid watch history payload: {exc}"}), 400

        return jsonify(
            {
                "job_id": job_id,
                "stored_path": str(dest_path),
                "upload_summary": summarize_records(records),
            }
        ), 201

    @app.post("/process")
    def process():
        """
        Run embeddings, clustering, zero-shot labels, time/sequence analysis.

        JSON body: ``job_id`` (required), optional ``cluster_method`` (``kmeans`` or
        ``dbscan``), optional ``include_per_video`` (default true), optional
        ``include_embeddings`` (default false).
        """
        body = request.get_json(silent=True) or {}
        job_id = body.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id required"}), 400

        cluster_method = body.get("cluster_method", "kmeans")
        if cluster_method not in ("kmeans", "dbscan"):
            return jsonify({"error": "cluster_method must be 'kmeans' or 'dbscan'"}), 400

        src = Path(app.config["UPLOAD_FOLDER"]) / f"{job_id}.json"
        if not src.is_file():
            return jsonify({"error": "unknown job_id or missing upload"}), 404

        try:
            records = load_watch_history(src)
        except (ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"unable to parse stored watch history: {exc}"}), 400
        insights = run_pipeline(records, cluster_method=cluster_method)

        if not body.get("include_embeddings", False):
            insights["model_artifacts"].pop("embedding_matrix", None)
            insights["model_artifacts"]["embedding_matrix_omitted"] = True

        if not body.get("include_per_video", True):
            insights = {k: v for k, v in insights.items() if k != "per_video"}
            insights["per_video_omitted"] = True

        out_path = Path(app.config["UPLOAD_FOLDER"]) / f"{job_id}_insights.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(insights, f, ensure_ascii=False, default=str)

        return jsonify({"job_id": job_id, "insights": insights, "insights_path": str(out_path)})

    @app.get("/get-insights")
    def get_insights():
        """
        Return last computed insights for ``job_id`` query parameter.
        """
        job_id = request.args.get("job_id")
        if not job_id:
            return jsonify({"error": "job_id query param required"}), 400
        out_path = Path(app.config["UPLOAD_FOLDER"]) / f"{job_id}_insights.json"
        if not out_path.is_file():
            return jsonify({"error": "no insights for job_id; call /process first"}), 404
        with out_path.open("r", encoding="utf-8") as f:
            payload: Dict[str, Any] = json.load(f)
        return jsonify({"job_id": job_id, "insights": payload})

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app
