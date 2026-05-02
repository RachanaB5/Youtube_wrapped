const DEFAULT_BASE = "http://localhost:5050";

export function getApiBase() {
  return import.meta.env.VITE_API_URL || DEFAULT_BASE;
}

async function handleJson(res) {
  const text = await res.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Invalid JSON from server (${res.status})`);
  }
  if (!res.ok) {
    throw new Error(data.error || data.message || `Request failed: ${res.status}`);
  }
  return data;
}

/**
 * POST multipart form with field name `file` (matches Flask backend).
 */
export async function uploadHistory(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${getApiBase()}/upload-history`, {
    method: "POST",
    body: fd,
  });
  return handleJson(res);
}

/**
 * Run embeddings / clustering / classification pipeline.
 */
export async function processHistory(sessionId, options = {}) {
  const res = await fetch(`${getApiBase()}/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      include_per_video: options.include_per_video ?? true,
      n_clusters: options.n_clusters ?? 8,
    }),
  });
  return handleJson(res);
}

export async function getInsights(sessionId, opts = {}) {
  const q = new URLSearchParams({ session_id: sessionId });
  if (opts.partial) q.set("partial", "true");
  const res = await fetch(`${getApiBase()}/get-insights?${q}`);
  return handleJson(res);
}

const PROCESS_TIMEOUT_MS = 90_000;

/**
 * GET /process-stream (Server-Sent Events) with step/pct updates; loads final insights on ``done``.
 * If the run exceeds ``PROCESS_TIMEOUT_MS``, closes the stream and fetches best-available insights
 * (including partial snapshots while the pipeline is still running).
 */
export function processWithProgress(sessionId, onProgress, options = {}) {
  const params = new URLSearchParams({
    session_id: sessionId,
    n_clusters: String(options.n_clusters ?? 12),
    include_per_video:
      options.include_per_video === false ? "false" : "true",
  });
  const streamUrl = `${getApiBase()}/process-stream?${params}`;

  return new Promise((resolve, reject) => {
    let settled = false;
    /** @type {EventSource | null} */
    let es = null;

    const finish = (cb) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      cb();
    };

    const loadInsights = () =>
      fetch(
        `${getApiBase()}/get-insights?${new URLSearchParams({
          session_id: sessionId,
          partial: "true",
        })}`,
      ).then(handleJson);

    const timer = setTimeout(() => {
      finish(() => {
        es?.close();
        loadInsights().then(resolve).catch(reject);
      });
    }, PROCESS_TIMEOUT_MS);

    es = new EventSource(streamUrl);

    es.onmessage = (e) => {
      let data;
      try {
        data = JSON.parse(e.data);
      } catch {
        return;
      }
      if (data.step && typeof data.pct === "number") {
        onProgress?.(data.step, data.pct);
      }
      if (data.step === "error") {
        finish(() => {
          es?.close();
          reject(new Error(data.message || "Pipeline failed"));
        });
        return;
      }
      if (data.step === "done") {
        finish(() => {
          es?.close();
          fetch(
            `${getApiBase()}/get-insights?${new URLSearchParams({ session_id: sessionId })}`,
          )
            .then(handleJson)
            .then(resolve)
            .catch(reject);
        });
      }
    };

    es.onerror = () => {
      finish(() => {
        es?.close();
        loadInsights().then(resolve).catch(() => reject(new Error("Stream failed")));
      });
    };
  });
}

export async function getShareCard(sessionId) {
  const res = await fetch(`${getApiBase()}/share-card/${encodeURIComponent(sessionId)}`);
  return handleJson(res);
}

export async function startDemoSession() {
  const res = await fetch(`${getApiBase()}/demo-session`, { method: "POST" });
  return handleJson(res);
}

/**
 * Teach the classifier a better category for a title/snippet (persists on server).
 */
export async function submitCategoryFeedback({
  title,
  wrongCategory,
  correctCategory,
}) {
  const res = await fetch(`${getApiBase()}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title,
      wrong_category: wrongCategory ?? "unknown",
      correct_category: correctCategory,
    }),
  });
  return handleJson(res);
}
