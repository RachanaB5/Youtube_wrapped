import { motion } from "framer-motion";

export const PROCESSING_STEPS = [
  { key: "loading", label: "📂 Reading your history", pct: 5 },
  { key: "embeddings", label: "🧠 Understanding your taste", pct: 20 },
  { key: "clustering", label: "🔮 Finding your patterns", pct: 60 },
  { key: "insights", label: "✨ Writing your story", pct: 85 },
  { key: "done", label: "🎬 Your Wrapped is ready!", pct: 100 },
];

function labelForStep(step) {
  const row = PROCESSING_STEPS.find((s) => s.key === step);
  return row?.label ?? PROCESSING_STEPS[0].label;
}

export default function ProcessingOverlay({ step = "loading", pct = 0 }) {
  const safePct = Math.max(0, Math.min(100, Number(pct) || 0));
  const currentLabel = labelForStep(step);

  return (
    <div className="relative flex min-h-[100dvh] flex-col items-center justify-center overflow-hidden px-6">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,_rgba(124,58,237,0.25)_0%,_transparent_50%),radial-gradient(ellipse_at_70%_80%,_rgba(6,182,212,0.15)_0%,_transparent_45%)]" />
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-950 via-zinc-950/95 to-black" />

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full max-w-md"
      >
        <p className="mb-2 text-center text-sm font-semibold uppercase tracking-[0.35em] text-violet-400">
          YouTube Wrapped
        </p>
        <h2 className="text-center font-sans text-2xl font-black text-white sm:text-3xl">
          Crunching your year
        </h2>

        <div className="progress-container mt-10 space-y-3">
          <div className="step-label min-h-[1.5rem] text-center text-base font-medium text-white/90">
            {currentLabel}
          </div>
          <div className="progress-bar h-3 w-full overflow-hidden rounded-full bg-zinc-800/90 ring-1 ring-white/10">
            <motion.div
              className="progress-fill h-full rounded-full bg-gradient-to-r from-violet-500 via-fuchsia-500 to-cyan-400"
              initial={false}
              animate={{ width: `${safePct}%` }}
              transition={{ type: "spring", stiffness: 120, damping: 20 }}
            />
          </div>
          <div className="progress-pct text-center text-sm font-semibold text-zinc-400">
            {Math.round(safePct)}%
          </div>
        </div>

        <p className="mt-6 text-center text-xs text-zinc-500">
          Large exports can take a minute. We’ll show whatever’s ready if it runs long.
        </p>
      </motion.div>
    </div>
  );
}
