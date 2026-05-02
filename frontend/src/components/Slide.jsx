import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const GRADIENTS = [
  "from-violet-900 via-purple-950 to-black",
  "from-blue-600 via-indigo-950 to-black",
  "from-fuchsia-600 via-pink-950 to-black",
  "from-emerald-800 via-green-950 to-black",
  "from-orange-600 via-red-950 to-black",
  "from-teal-700 via-cyan-950 to-black",
];

function parseStatNumber(stat) {
  if (stat == null || stat === "") return null;
  const s = String(stat).trim();
  const match = s.match(/^([\d.,]+)(.*)$/);
  if (!match) return null;
  const n = parseFloat(match[1].replace(/,/g, ""));
  if (Number.isNaN(n)) return null;
  const suffix = match[2] || (s.endsWith("%") ? "%" : "");
  return { value: n, suffix };
}

function AnimatedNumber({ value, suffix = "", duration = 1.1 }) {
  const [n, setN] = useState(0);
  useEffect(() => {
    let raf;
    const t0 = performance.now();
    const tick = (t) => {
      const p = Math.min((t - t0) / (duration * 1000), 1);
      const e = 1 - (1 - p) ** 3;
      setN(Math.round(value * e));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [value, duration]);
  return (
    <span className="tabular-nums">
      {n}
      {suffix}
    </span>
  );
}

export default function Slide({
  headline,
  body = [],
  stat,
  emoji,
  gradient = 0,
  slideNumber,
  totalSlides,
}) {
  const gradClass = GRADIENTS[gradient % GRADIENTS.length];
  const parsed = parseStatNumber(stat);
  const lines = (Array.isArray(body) ? body : [body]).filter(Boolean).slice(0, 3);

  return (
    <div
      className={`relative flex min-h-[100dvh] flex-col justify-center overflow-hidden bg-gradient-to-br ${gradClass} px-6 py-16 sm:px-12`}
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(255,255,255,0.12)_0%,_transparent_50%)]" />
      <div className="pointer-events-none absolute -right-20 top-1/4 h-96 w-96 rounded-full bg-white/5 blur-3xl" />

      {typeof slideNumber === "number" && typeof totalSlides === "number" && (
        <div className="absolute left-6 top-6 text-xs font-medium uppercase tracking-[0.2em] text-white/40">
          {slideNumber} / {totalSlides}
        </div>
      )}

      <div className="relative z-10 mx-auto w-full max-w-3xl">
        {emoji && (
          <motion.div
            initial={{ scale: 0.6, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 260, damping: 20 }}
            className="mb-6 text-5xl sm:text-6xl"
            aria-hidden
          >
            {emoji}
          </motion.div>
        )}

        <motion.h1
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
          className="font-sans text-4xl font-black leading-[1.05] tracking-tight sm:text-5xl md:text-6xl"
        >
          {headline}
        </motion.h1>

        <div className="mt-8 space-y-4">
          {lines.map((line, i) => (
            <motion.p
              key={i}
              initial={{ opacity: 0, x: -16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{
                delay: 0.12 + i * 0.1,
                duration: 0.45,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="max-w-2xl text-lg font-medium leading-relaxed text-white/85 sm:text-xl md:text-2xl"
            >
              {line}
            </motion.p>
          ))}
        </div>

        {stat != null && stat !== "" && (
          <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.35, duration: 0.5 }}
            className="mt-10"
          >
            <div className="inline-block rounded-2xl border border-white/10 bg-black/20 px-8 py-5 backdrop-blur-md">
              <p className="text-xs font-semibold uppercase tracking-widest text-white/50">
                Stat drop
              </p>
              <p className="mt-1 font-sans text-5xl font-black tabular-nums text-white sm:text-6xl md:text-7xl">
                {parsed ? (
                  <AnimatedNumber value={parsed.value} suffix={parsed.suffix} />
                ) : (
                  <span>{stat}</span>
                )}
              </p>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

export { GRADIENTS };
