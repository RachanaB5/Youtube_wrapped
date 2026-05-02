"""
Flask application factory and HTTP routes for YouTube Wrapped.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import queue
import shutil
import threading
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from youtube_wrapped.data import find_stored_watch_history, load_watch_history, summarize_dataframe
from youtube_wrapped.feedback import normalize_label, save_feedback
from youtube_wrapped.model import CATEGORY_KEYWORDS, run_pipeline
from youtube_wrapped.share_card import build_share_card_payload

logger = logging.getLogger(__name__)

SMALL_HISTORY_THRESHOLD = 50


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    tmp.replace(path)


def _configure_logging() -> None:
    level = os.environ.get("YOUTUBE_WRAPPED_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _process_stream_response(
    app: Flask,
    session_id: str,
    n_clusters: int,
    include_per_video: bool,
) -> Response:
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    out_path = upload_dir / f"{session_id}_insights.json"
    partial_path = upload_dir / f"{session_id}_insights_partial.json"
    src = find_stored_watch_history(app.config["UPLOAD_FOLDER"], session_id)
    if src is None:
        return jsonify({"error": "unknown session_id or missing upload"}), 404  # type: ignore[return-value]

    def generate():
        q: queue.Queue = queue.Queue()
        errors: list = []

        def progress_cb(step: str, pct: int, partial: Dict[str, Any] | None = None) -> None:
            if partial is not None:
                try:
                    snap = dict(partial)
                    snap["session_id"] = session_id
                    snap["insights_partial"] = True
                    _write_json_atomic(partial_path, snap)
                except Exception as exc:
                    logger.warning("Partial insights write failed: %s", exc)
            q.put({"step": step, "pct": pct})

        def worker():
            try:
                df = load_watch_history(src)
                summary = summarize_dataframe(df)
                pipeline_warnings = _upload_warnings(summary)

                insights = run_pipeline(
                    df,
                    n_clusters=n_clusters,
                    include_per_video=include_per_video,
                    progress_callback=progress_cb,
                    session_id=session_id,
                )
                insights["session_id"] = session_id
                insights["warnings"] = pipeline_warnings
                if not include_per_video:
                    insights.pop("per_video", None)
                    insights["per_video_omitted"] = True
                _write_json_atomic(out_path, insights)
                partial_path.unlink(missing_ok=True)
            except Exception as exc:
                errors.append(exc)
                logger.error("process-stream failed: %s\n%s", exc, traceback.format_exc())
            finally:
                q.put(None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        t.join(timeout=7200.0)

        if errors:
            yield f"data: {json.dumps({'step': 'error', 'message': str(errors[0])})}\n\n"
            return

        done_payload = {
            "step": "done",
            "pct": 100,
            "session_id": session_id,
            "insights_path": str(out_path),
            "result": None,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _upload_warnings(summary: Dict[str, Any]) -> list:
    warnings: list = []
    n = int(summary.get("record_count") or 0)
    if n < SMALL_HISTORY_THRESHOLD:
        warnings.append(
            {
                "code": "small_history",
                "message": (
                    f"Only {n} videos in this export (we recommend at least "
                    f"{SMALL_HISTORY_THRESHOLD} for steadier clusters and categories). "
                    "Results can be noisier—still fun, just grainier."
                ),
            }
        )
    return warnings


def _ml_import_status() -> Dict[str, bool]:
    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
    }


def _sample_history_path() -> Path:
    """Repo-root ``sample_data/fake_history.json``."""
    return Path(__file__).resolve().parent.parent / "sample_data" / "fake_history.json"


def _watch_history_upload_extension(f) -> str:
    """``.json`` / ``.html`` from filename, or sniff file start when extension is missing."""
    name = getattr(f, "filename", None) or ""
    ext = Path(name).suffix.lower()
    if ext in (".json", ".html"):
        return ext
    head = f.read(4096)
    f.seek(0)
    text = head.decode("utf-8", errors="ignore").lstrip()
    if text.startswith("<"):
        return ".html"
    return ".json"


def create_app() -> Flask:
    """
    Build Flask app with upload/process/insights endpoints.

    Files live under ``UPLOAD_FOLDER`` (default ``uploads``), keyed by ``session_id``.
    """
    _configure_logging()
    app = Flask(__name__)
    CORS(
        app,
        resources={r"/*": {"origins": os.environ.get("YOUTUBE_WRAPPED_CORS_ORIGINS", "*").split(",")}},
        supports_credentials=False,
    )
    upload_root = Path(os.environ.get("YOUTUBE_WRAPPED_UPLOAD_DIR", "uploads"))
    upload_root.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(upload_root.resolve())
    logger.info("Upload folder: %s", app.config["UPLOAD_FOLDER"])

    @app.post("/upload-history")
    def upload_history():
        """
        Accept Google Takeout watch history: ``watch-history.json`` or ``watch-history.html``
        (multipart ``file``), or a raw JSON array/object body (stored as ``.json``).

        Returns ``session_id`` and a small upload summary.
        """
        session_id = str(uuid.uuid4())
        dest_dir = Path(app.config["UPLOAD_FOLDER"])
        dest_path = dest_dir / f"{session_id}.json"

        try:
            if request.content_type and "multipart/form-data" in request.content_type:
                if "file" not in request.files:
                    return jsonify({"error": "missing file field 'file'"}), 400
                f = request.files["file"]
                if not f.filename:
                    return jsonify({"error": "empty filename"}), 400
                ext = _watch_history_upload_extension(f)
                dest_path = dest_dir / f"{session_id}{ext}"
                f.save(dest_path)
            else:
                data = request.get_json(silent=True)
                if data is None:
                    return jsonify({"error": "expected JSON body or multipart file"}), 400
                dest_path = dest_dir / f"{session_id}.json"
                with dest_path.open("w", encoding="utf-8") as out:
                    json.dump(data, out, ensure_ascii=False)

            df = load_watch_history(dest_path)
        except (ValueError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Upload failed: %s", exc)
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            return jsonify({"error": f"invalid watch history: {exc}"}), 400
        except Exception as exc:
            logger.error("Unexpected upload error: %s\n%s", exc, traceback.format_exc())
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except OSError:
                    pass
            return jsonify({"error": "internal error processing upload"}), 500

        summary = summarize_dataframe(df)
        warnings = _upload_warnings(summary)
        return (
            jsonify(
                {
                    "session_id": session_id,
                    "stored_path": str(dest_path),
                    "upload_summary": summary,
                    "warnings": warnings,
                }
            ),
            201,
        )

    @app.post("/demo-session")
    def demo_session():
        """
        Create a session from bundled ``sample_data/fake_history.json`` (~200 synthetic watches).

        Returns the same shape as ``POST /upload-history`` (session_id + summary + warnings).
        """
        sample = _sample_history_path()
        if not sample.is_file():
            return jsonify({"error": "demo data missing on server (sample_data/fake_history.json)"}), 500

        session_id = str(uuid.uuid4())
        dest_dir = Path(app.config["UPLOAD_FOLDER"])
        dest_path = dest_dir / f"{session_id}.json"
        try:
            shutil.copyfile(sample, dest_path)
            df = load_watch_history(dest_path)
        except (ValueError, OSError) as exc:
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            return jsonify({"error": f"demo load failed: {exc}"}), 500

        summary = summarize_dataframe(df)
        return (
            jsonify(
                {
                    "session_id": session_id,
                    "stored_path": str(dest_path),
                    "upload_summary": summary,
                    "warnings": _upload_warnings(summary),
                    "demo": True,
                }
            ),
            201,
        )

    @app.post("/process")
    def process():
        """
        Run the full pipeline for a stored ``session_id``.

        JSON body: ``session_id`` (required), optional ``n_clusters`` (default 12),
        optional ``include_per_video`` (default true).
        """
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id") or body.get("job_id")
        if not session_id:
            return jsonify({"error": "session_id required"}), 400

        src = find_stored_watch_history(app.config["UPLOAD_FOLDER"], session_id)
        if src is None:
            return jsonify({"error": "unknown session_id or missing upload"}), 404

        n_clusters = int(body.get("n_clusters", 12))
        include_per_video = bool(body.get("include_per_video", True))

        try:
            df = load_watch_history(src)
            summary = summarize_dataframe(df)
            pipeline_warnings = _upload_warnings(summary)
            insights = run_pipeline(
                df,
                n_clusters=n_clusters,
                include_per_video=include_per_video,
                session_id=session_id,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("Process parse error: %s", exc)
            return jsonify({"error": f"unable to parse watch history: {exc}"}), 400
        except Exception as exc:
            logger.error("Pipeline failed: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": f"pipeline failed: {exc}"}), 500

        insights["session_id"] = session_id
        insights["warnings"] = pipeline_warnings
        if not include_per_video:
            insights.pop("per_video", None)
            insights["per_video_omitted"] = True

        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        out_path = upload_dir / f"{session_id}_insights.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(insights, f, ensure_ascii=False, default=str)

        logger.info("Wrote insights for session %s to %s", session_id, out_path)
        return jsonify({"session_id": session_id, "insights": insights, "insights_path": str(out_path)})

    @app.get("/process-stream")
    def process_stream_get():
        """
        SSE (EventSource-friendly GET): ``?session_id=…`` optional ``n_clusters``, ``include_per_video``.
        """
        session_id = request.args.get("session_id") or request.args.get("job_id")
        if not session_id:
            return jsonify({"error": "session_id query param required"}), 400
        n_clusters = int(request.args.get("n_clusters", 12))
        raw_inc = (request.args.get("include_per_video") or "true").lower()
        include_per_video = raw_inc not in ("0", "false", "no")
        return _process_stream_response(app, session_id, n_clusters, include_per_video)

    @app.post("/process-stream")
    def process_stream_post():
        """
        Same as GET ``/process-stream`` but JSON body ``session_id``, optional ``n_clusters``,
        ``include_per_video``.
        """
        body = request.get_json(silent=True) or {}
        session_id = body.get("session_id") or body.get("job_id")
        if not session_id:
            return jsonify({"error": "session_id required"}), 400
        n_clusters = int(body.get("n_clusters", 12))
        include_per_video = bool(body.get("include_per_video", True))
        return _process_stream_response(app, session_id, n_clusters, include_per_video)

    @app.get("/get-insights")
    def get_insights():
        """
        Return cached insights: final ``*_insights.json`` if present, else best-effort
        ``*_insights_partial.json`` while the pipeline is still running (``partial``: true).
        """
        session_id = request.args.get("session_id") or request.args.get("job_id")
        if not session_id:
            return jsonify({"error": "session_id query param required"}), 400
        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        out_path = upload_dir / f"{session_id}_insights.json"
        partial_path = upload_dir / f"{session_id}_insights_partial.json"
        path: Path | None = None
        is_partial = False
        if out_path.is_file():
            path, is_partial = out_path, False
        elif partial_path.is_file():
            path, is_partial = partial_path, True
        if path is None:
            return jsonify({"error": "no insights for session_id; run /process or /process-stream first"}), 404
        try:
            with path.open("r", encoding="utf-8") as f:
                payload: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read insights: %s", exc)
            return jsonify({"error": "cached insights file is invalid"}), 500
        return jsonify({"session_id": session_id, "insights": payload, "partial": is_partial})

    @app.get("/share-card/<session_id>")
    def share_card(session_id: str):
        """
        JSON tuned for a vertical share image (persona, top categories, creator, peak time).
        """
        out_path = Path(app.config["UPLOAD_FOLDER"]) / f"{session_id}_insights.json"
        if not out_path.is_file():
            return jsonify({"error": "no insights for session_id; call /process first"}), 404
        try:
            with out_path.open("r", encoding="utf-8") as f:
                insights: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            return jsonify({"error": f"invalid cache: {exc}"}), 500
        payload = build_share_card_payload(insights)
        payload["session_id"] = session_id
        return jsonify(payload)

    @app.get("/health")
    def health():
        """
        Liveness + optional ML stack checks.

        Query ``probe=1`` loads SentenceTransformer once and encodes a probe string
        (slow on first cold start; downloads weights if not cached).
        """
        ml_imports = _ml_import_status()
        body: Dict[str, Any] = {
            "status": "ok",
            "ml_imports": ml_imports,
            "embedder_warm": None,
        }
        if request.args.get("probe") == "1":
            try:
                from youtube_wrapped.model import classify_categories, get_embeddings

                get_embeddings(["ml health probe"], batch_size=1)
                classify_categories(["python tutorial for beginners"])
                body["embedder_warm"] = True
                body["classifier_warm"] = True
            except Exception as exc:
                body["embedder_warm"] = False
                body["classifier_warm"] = False
                body["embedder_error"] = str(exc)[:500]
                body["status"] = "degraded"
        return jsonify(body)

    @app.post("/feedback")
    def submit_feedback():
        """
        JSON: ``title``, ``correct_category`` required; ``wrong_category`` optional.
        Persists word patterns and periodically merges keyword suggestions after 20+ corrections.
        """
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        title = data.get("title")
        correct = data.get("correct_category")
        if not title or not correct:
            return jsonify({"error": "title and correct_category required"}), 400
        wrong = data.get("wrong_category") or "unknown"
        try:
            correct_n = normalize_label(correct)
            save_feedback(str(title), str(wrong), correct_n, CATEGORY_KEYWORDS)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            logger.error("Feedback save failed: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": "could not save feedback"}), 500
        snippet = str(title)[:30] + ("…" if len(str(title)) > 30 else "")
        return jsonify(
            {
                "status": "learned",
                "message": f"Got it! '{snippet}' → {correct_n}",
                "correct_category": correct_n,
            }
        )

    return app
