# YouTube Wrapped

End-to-end **YouTube Wrapped** from Google Takeout **watch-history.json**: Flask API (embeddings, clustering, zero-shot categories, time stats, narratives) + **React** slideshow (Vite, Tailwind, Framer Motion, Recharts).

## Features

- Upload Takeout JSON → embeddings (**SentenceTransformers** `all-MiniLM-L6-v2`), **KMeans** clusters with TF‑IDF labels, **zero-shot** categories (BART-MNLI), time patterns, shareable **story card** (PNG via html2canvas).
- **Demo mode**: `POST /demo-session` uses `sample_data/fake_history.json` (~200 synthetic watches).
- **Share card API**: `GET /share-card/<session_id>` returns JSON tailored for the vertical graphic.

## Setup (backend)

Python **3.10+** recommended.

```bash
cd /path/to/Youtube_wrapped
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First pipeline run downloads **PyTorch**, **SentenceTransformers**, and **facebook/bart-large-mnli** (large download; use a stable network).

### Run API

```bash
python run.py
```

Default: **http://127.0.0.1:5050** (`debug=True`). Port **5000** is often taken on macOS (e.g. AirPlay Receiver); free it in **System Settings → General → AirDrop & Handoff → AirPlay Receiver**, or run `PORT=5000 python run.py` if that port is available.

Health:

- `GET /health` — import checks for `torch`, `sentence_transformers`, `transformers`, `sklearn`.
- `GET /health?probe=1` — runs a real **embedding** encode (cold start may download weights; marks `embedder_warm` true/false).

### CORS

All origins allowed by default. Override:

```bash
export YOUTUBE_WRAPPED_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## Setup (frontend)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. API URL defaults to `http://localhost:5050` (matches `run.py`); override with:

```bash
echo 'VITE_API_URL=http://127.0.0.1:5050' > frontend/.env
```

## How to get Google Takeout data

1. Go to [Google Takeout](https://takeout.google.com/).
2. **Deselect all**, then select **YouTube and YouTube Music** (or the product that includes your watch history).
3. Choose **All YouTube data included** → ensure **history** is included (watch history).
4. Export **once** or on a schedule; format **JSON**.
5. Download the archive, unzip, and find **`watch-history.json`** (path varies slightly by export version).
6. Upload that file in the web UI (or POST it to `/upload-history`).

> **Privacy:** Processing runs on **your machine** (or your server). Do not commit real Takeout files to git.

## How to run locally (full stack)

**Terminal 1 — API**

```bash
source .venv/bin/activate
python run.py
```

**Terminal 2 — UI**

```bash
cd frontend && npm run dev
```

Flow: upload JSON → `POST /upload-history` → `POST /process` → slideshow → insights dashboard → **Share Your Wrapped** (downloads PNG).

### Small history warning

If the parsed history has **fewer than 50 videos**, the API adds a `warnings` array (`small_history`). The UI shows a friendly amber callout.

### Missing channel names

Rows without a channel resolve to **`Unknown Creator`** in the DataFrame and share card.

### Demo without real data

- UI: **Try demo (sample data)** on the landing screen.
- API: `POST /demo-session` → then `POST /process` with returned `session_id`.

Sample file: `sample_data/fake_history.json`.

## API reference (short)

| Method | Path | Purpose |
|--------|------|--------|
| POST | `/upload-history` | Multipart field `file` or JSON body; returns `session_id`, `upload_summary`, `warnings` |
| POST | `/demo-session` | Seeds session from `sample_data/fake_history.json` |
| POST | `/process` | Body: `{ "session_id", "n_clusters"?, "include_per_video"? }` |
| GET | `/get-insights?session_id=` | Cached full insights JSON |
| GET | `/share-card/<session_id>` | Compact JSON for the share image |
| GET | `/health` | Liveness + ML import status; `?probe=1` warms embedder |

## How it works (ML pipeline)

1. **Parse & clean** — Load Takeout list; strip **“Watched ”**, emojis; parse timestamps; build columns: `title`, `channel`, `hour`, `day_of_week`, `month`.
2. **Embeddings** — `all-MiniLM-L6-v2` encodes titles → vectors for clustering and drift.
3. **Clusters** — **KMeans** (`n_clusters` capped by sample size); **TF‑IDF** top terms per cluster for labels.
4. **Categories** — **Zero-shot** with **facebook/bart-large-mnli** into fixed labels (e.g. technology, music, education, …).
5. **Time patterns** — Peaks, late-night share, monthly histograms.
6. **Narrative** — Rule-based **persona / behaviors / evolution** from aggregates (see `youtube_wrapped/analyst.py`).
7. **Cinematic slides** — Short slide copy from top category, creators, clusters, time (`generate_cinematic_summary`).

Optional: **interest shift** over months in `youtube_wrapped/sequence_model.py`; **LLM insights** in `youtube_wrapped/insight_generator.py` (requires API keys).

## Screenshots

_Add screenshots here (upload screen, slide, insights, share card)._

## Project layout

```
Youtube_wrapped/
  run.py                 # Dev server entry
  requirements.txt
  sample_data/
    fake_history.json    # Demo data (200 synthetic entries)
  youtube_wrapped/
    data.py              # Takeout → DataFrame
    model.py             # Embeddings, KMeans, zero-shot, pipeline
    routes.py            # Flask + CORS
    share_card.py        # Share payload builder
    analyst.py
    sequence_model.py
    insight_generator.py
  frontend/
    src/
      api.js
      App.jsx
      components/
        UploadScreen.jsx
        WrappedSlideshow.jsx
        Slide.jsx
        InsightsPanel.jsx
        ShareStoryCard.jsx
```

## License

Personal / educational use. YouTube and Google Takeout are trademarks of Google LLC.
