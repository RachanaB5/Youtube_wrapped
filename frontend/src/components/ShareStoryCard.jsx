import html2canvas from "html2canvas";
import { useCallback, useEffect, useRef, useState } from "react";
import { getShareCard } from "../api";

/**
 * Instagram-style vertical share graphic; capture with html2canvas.
 */
export default function ShareStoryCard({ sessionId }) {
  const cardRef = useRef(null);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [capturing, setCapturing] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    (async () => {
      try {
        const payload = await getShareCard(sessionId);
        if (!cancelled) setData(payload);
      } catch (e) {
        if (!cancelled) setErr(e.message || String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  const downloadPng = useCallback(async () => {
    if (!cardRef.current) return;
    setCapturing(true);
    try {
      const canvas = await html2canvas(cardRef.current, {
        scale: 2,
        useCORS: true,
        logging: false,
        backgroundColor: "#0c0418",
      });
      const url = canvas.toDataURL("image/png");
      const a = document.createElement("a");
      a.href = url;
      a.download = `youtube-wrapped-${(sessionId || "share").slice(0, 8)}.png`;
      a.click();
    } finally {
      setCapturing(false);
    }
  }, [sessionId]);

  if (!sessionId) return null;
  if (err) {
    return (
      <div className="rounded-2xl border border-amber-500/30 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
        Could not load share data: {err}
      </div>
    );
  }
  if (!data) {
    return (
      <div className="flex h-64 items-center justify-center rounded-3xl border border-white/10 bg-zinc-900/50 text-zinc-500">
        Loading share card…
      </div>
    );
  }

  const cats = data.top_categories || [];

  return (
    <div id="share-wrapped" className="space-y-4">
      <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
        <div>
          <h2 className="font-sans text-xl font-bold text-white">Share Your Wrapped</h2>
          <p className="mt-1 text-sm text-zinc-400">
            Story-sized card — save and post anywhere.
          </p>
        </div>
        <button
          type="button"
          onClick={downloadPng}
          disabled={capturing}
          className="rounded-full bg-gradient-to-r from-fuchsia-500 to-violet-600 px-6 py-3 text-sm font-bold text-white shadow-lg shadow-violet-900/40 disabled:opacity-50"
        >
          {capturing ? "Rendering…" : "Share Your Wrapped"}
        </button>
      </div>

      <div className="mx-auto flex max-w-sm justify-center rounded-3xl border border-white/10 bg-black/40 p-4">
        <div
          ref={cardRef}
          className="relative flex aspect-[9/16] w-full max-w-[320px] flex-col justify-between overflow-hidden rounded-2xl bg-gradient-to-b from-fuchsia-600 via-violet-800 to-slate-950 p-6 text-white shadow-2xl ring-2 ring-white/20"
          style={{ minHeight: "568px" }}
        >
          <div
            className="pointer-events-none absolute inset-0 opacity-40"
            style={{
              background:
                "radial-gradient(circle at 20% 20%, rgba(255,255,255,0.35), transparent 45%), radial-gradient(circle at 80% 70%, rgba(34,211,238,0.35), transparent 40%)",
            }}
          />
          <div className="relative z-10">
            <p className="text-xs font-bold uppercase tracking-[0.35em] text-white/80">
              {data.brand?.title || "YouTube Wrapped"}
            </p>
            <p className="mt-8 font-sans text-3xl font-black leading-tight tracking-tight">
              {data.persona_label || "Your year"}
            </p>
          </div>

          <div className="relative z-10 space-y-5">
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-white/60">
                Top lanes
              </p>
              <ul className="mt-2 space-y-2">
                {cats.slice(0, 3).map((c, i) => (
                  <li
                    key={`${c.name}-${i}`}
                    className="flex items-baseline justify-between gap-2 border-b border-white/15 pb-2 text-sm font-semibold"
                  >
                    <span className="capitalize">{c.name}</span>
                    {c.share_pct != null ? (
                      <span className="tabular-nums text-white/90">{c.share_pct}%</span>
                    ) : (
                      <span className="tabular-nums text-white/90">{c.count}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-white/60">
                Top creator
              </p>
              <p className="mt-1 font-sans text-xl font-black">
                {data.top_creator?.name || "Unknown Creator"}
              </p>
              <p className="text-sm font-medium text-white/75">
                {data.top_creator?.watch_count ?? 0} watches
              </p>
            </div>
            <div>
              <p className="text-xs font-bold uppercase tracking-widest text-white/60">
                Peak time
              </p>
              <p className="mt-1 font-sans text-lg font-bold leading-snug">
                {data.peak_watch?.label || "—"}
              </p>
            </div>
          </div>

          <p className="relative z-10 text-center text-[10px] font-medium uppercase tracking-[0.25em] text-white/45">
            Wrapped · {new Date().getFullYear()}
          </p>
        </div>
      </div>
    </div>
  );
}
