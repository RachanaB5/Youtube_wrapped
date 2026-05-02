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

export async function getInsights(sessionId) {
  const q = new URLSearchParams({ session_id: sessionId });
  const res = await fetch(`${getApiBase()}/get-insights?${q}`);
  return handleJson(res);
}

export async function getShareCard(sessionId) {
  const res = await fetch(`${getApiBase()}/share-card/${encodeURIComponent(sessionId)}`);
  return handleJson(res);
}

export async function startDemoSession() {
  const res = await fetch(`${getApiBase()}/demo-session`, { method: "POST" });
  return handleJson(res);
}
