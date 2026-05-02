import { motion } from "framer-motion";
import { useCallback, useRef, useState } from "react";

export default function UploadScreen({ onGenerate, onDemo, onError }) {
  const [drag, setDrag] = useState(false);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);
  const [status, setStatus] = useState("");
  const inputRef = useRef(null);

  const run = useCallback(async () => {
    if (!file) {
      onError?.("Choose a watch-history.json file first.");
      return;
    }
    setLoading(true);
    setStatus("Uploading your history…");
    try {
      setStatus("Crunching embeddings & clusters…");
      await onGenerate(file);
      setStatus("Done!");
    } catch (e) {
      onError?.(e.message || String(e));
      setStatus("");
    } finally {
      setLoading(false);
    }
  }, [file, onGenerate, onError]);

  const onDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files?.[0];
    if (f && (f.name.endsWith(".json") || f.type === "application/json")) {
      setFile(f);
    } else {
      onError?.("Please drop a .json file (Takeout watch-history).");
    }
  };

  return (
    <div className="relative flex min-h-[100dvh] flex-col items-center justify-center overflow-hidden px-6">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,_rgba(124,58,237,0.25)_0%,_transparent_50%),radial-gradient(ellipse_at_70%_80%,_rgba(6,182,212,0.15)_0%,_transparent_45%)]" />
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-950 via-zinc-950/95 to-black" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-lg text-center"
      >
        <p className="mb-3 text-sm font-semibold uppercase tracking-[0.35em] text-violet-400">
          YouTube Wrapped
        </p>
        <h1 className="font-sans text-4xl font-black tracking-tight text-white sm:text-5xl">
          Your year,
          <br />
          unmuted.
        </h1>
        <p className="mt-4 text-base text-zinc-400 sm:text-lg">
          Drop your Google Takeout{" "}
          <code className="rounded bg-zinc-800 px-1.5 py-0.5 text-sm text-violet-300">
            watch-history.json
          </code>
        </p>

        <div
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          onDragEnter={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          className={`mt-10 cursor-pointer rounded-3xl border-2 border-dashed px-6 py-14 transition-colors ${
            drag
              ? "border-violet-400 bg-violet-500/10"
              : "border-zinc-600 bg-zinc-900/50 hover:border-zinc-500"
          }`}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".json,application/json"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) setFile(f);
            }}
          />
          <p className="text-lg font-semibold text-white">
            {file ? file.name : "Drag & drop or tap to browse"}
          </p>
          <p className="mt-2 text-sm text-zinc-500">JSON only • stays on your machine until upload</p>
        </div>

        <button
          type="button"
          disabled={loading || !file}
          onClick={(e) => {
            e.stopPropagation();
            run();
          }}
          className="mt-8 w-full rounded-full bg-white py-4 text-lg font-bold text-zinc-950 shadow-lg shadow-violet-500/20 transition enabled:hover:scale-[1.02] enabled:hover:shadow-violet-500/30 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? "Generating…" : "Generate My Wrapped"}
        </button>

        {onDemo && (
          <button
            type="button"
            disabled={loading || demoLoading}
            onClick={async (e) => {
              e.stopPropagation();
              setDemoLoading(true);
              setStatus("Loading demo dataset…");
              try {
                await onDemo();
              } catch (err) {
                onError?.(err.message || String(err));
                setStatus("");
              } finally {
                setDemoLoading(false);
              }
            }}
            className="mt-4 w-full rounded-full border border-white/20 py-3 text-sm font-semibold text-white/90 transition hover:border-violet-400/50 hover:text-white disabled:opacity-40"
          >
            {demoLoading ? "Running demo…" : "Try demo (sample data, no upload)"}
          </button>
        )}

        {(loading || demoLoading) && (
          <div className="mt-8 flex flex-col items-center gap-4">
            <div className="relative h-2 w-full overflow-hidden rounded-full bg-zinc-800">
              <motion.div
                className="absolute inset-y-0 w-1/3 rounded-full bg-gradient-to-r from-violet-500 to-cyan-400"
                animate={{ x: ["-100%", "300%"] }}
                transition={{ repeat: Infinity, duration: 1.2, ease: "linear" }}
              />
            </div>
            <p className="text-sm font-medium text-zinc-400">{status}</p>
          </div>
        )}
      </motion.div>
    </div>
  );
}
