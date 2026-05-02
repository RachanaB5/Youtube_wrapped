import { motion } from "framer-motion";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import ShareStoryCard from "./ShareStoryCard.jsx";

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm shadow-xl">
      <p className="font-semibold text-white">{label}</p>
      <p className="text-violet-300">{payload[0].value} watches</p>
    </div>
  );
}

export default function InsightsPanel({ insights, sessionId, onBack, onRestart }) {
  const categories = (insights?.top_categories || []).map((c) => ({
    name: c.category,
    count: c.count,
  }));

  const monthly = insights?.monthly_trends_sorted
    ? insights.monthly_trends_sorted.map((r) => ({
        month: r.month,
        count: r.count,
      }))
    : Object.entries(insights?.monthly_trends || {})
        .map(([month, count]) => ({ month, count }))
        .sort((a, b) => a.month.localeCompare(b.month));

  const creators = (insights?.top_creators || []).slice(0, 5).map((c) => {
    if (typeof c === "object" && !Array.isArray(c)) {
      return { name: c.channel || "Unknown Creator", count: c.watch_count };
    }
    return { name: c[0] || "Unknown Creator", count: c[1] };
  });

  const warnings = insights?.warnings || [];
  const persona = insights?.narrative?.persona || "Your watch story is one of a kind.";
  const total =
    insights?.summary?.total_videos_analyzed ||
    categories.reduce((a, c) => a + c.count, 0);

  return (
    <div className="min-h-[100dvh] bg-zinc-950 pb-20">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_50%_0%,_rgba(139,92,246,0.12)_0%,_transparent_50%)]" />

      <header className="sticky top-0 z-30 border-b border-white/5 bg-zinc-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-4 sm:px-6">
          <button
            type="button"
            onClick={onBack}
            className="text-sm font-medium text-zinc-400 hover:text-white"
          >
            ← Back to slides
          </button>
          <h1 className="font-sans text-lg font-bold text-white">Your insights</h1>
          <button
            type="button"
            onClick={onRestart}
            className="text-sm font-medium text-violet-400 hover:text-violet-300"
          >
            Start over
          </button>
        </div>
      </header>

      <main className="relative z-10 mx-auto max-w-6xl space-y-10 px-4 py-10 sm:px-6">
        {warnings.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-2 rounded-2xl border border-amber-500/35 bg-amber-950/30 px-4 py-3 text-amber-100"
          >
            {warnings.map((w) => (
              <p key={w.code || w.message} className="text-sm leading-relaxed">
                {w.message}
              </p>
            ))}
          </motion.div>
        )}

        <ShareStoryCard sessionId={sessionId} />

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid gap-6 lg:grid-cols-3"
        >
          <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-violet-950/50 to-zinc-900/80 p-8 shadow-2xl lg:col-span-1">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-400">
              Persona
            </p>
            <p className="mt-4 font-sans text-2xl font-bold leading-snug text-white">
              {persona}
            </p>
            <p className="mt-6 text-sm text-zinc-500">
              {total ? `${total} videos analyzed in this export.` : ""}
            </p>
          </div>

          <div className="rounded-3xl border border-white/10 bg-zinc-900/50 p-6 lg:col-span-2">
            <h2 className="mb-4 font-sans text-lg font-bold text-white">Category mix</h2>
            <div className="h-[280px] w-full">
              {categories.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categories} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis
                      dataKey="name"
                      tick={{ fill: "#a1a1aa", fontSize: 11 }}
                      axisLine={{ stroke: "#3f3f46" }}
                      tickLine={false}
                    />
                    <YAxis
                      tick={{ fill: "#a1a1aa", fontSize: 11 }}
                      axisLine={false}
                      tickLine={false}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} maxBarSize={48} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="flex h-full items-center justify-center text-zinc-500">No categories</p>
              )}
            </div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.08 }}
          className="rounded-3xl border border-white/10 bg-zinc-900/50 p-6"
        >
          <h2 className="mb-4 font-sans text-lg font-bold text-white">Monthly activity</h2>
          <div className="h-[300px] w-full">
            {monthly.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={monthly} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="fillCount" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#06b6d4" stopOpacity={0.5} />
                      <stop offset="100%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis
                    dataKey="month"
                    tick={{ fill: "#a1a1aa", fontSize: 10 }}
                    axisLine={{ stroke: "#3f3f46" }}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fill: "#a1a1aa", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="count"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    fill="url(#fillCount)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <p className="flex h-full items-center justify-center text-zinc-500">No timeline</p>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          className="rounded-3xl border border-white/10 bg-zinc-900/50 p-6"
        >
          <h2 className="mb-6 font-sans text-lg font-bold text-white">Top creators</h2>
          <ul className="space-y-3">
            {creators.map((c, i) => (
              <li
                key={`${c.name}-${i}`}
                className="flex items-center justify-between rounded-2xl bg-black/30 px-4 py-3"
              >
                <div className="flex items-center gap-3">
                  <span className="flex h-8 w-8 items-center justify-center rounded-full bg-violet-500/20 text-sm font-bold text-violet-300">
                    {i + 1}
                  </span>
                  <span className="font-medium text-white">{c.name}</span>
                </div>
                <span className="tabular-nums text-zinc-400">{c.count} watches</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </main>
    </div>
  );
}
