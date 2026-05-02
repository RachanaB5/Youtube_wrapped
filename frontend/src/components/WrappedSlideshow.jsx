import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import Slide from "./Slide.jsx";

const variants = {
  enter: (dir) => ({
    x: dir > 0 ? "100%" : "-100%",
    opacity: 0,
  }),
  center: { x: 0, opacity: 1 },
  exit: (dir) => ({
    x: dir < 0 ? "100%" : "-100%",
    opacity: 0,
  }),
};

export default function WrappedSlideshow({
  slides,
  onDone,
  onRestart,
}) {
  const [index, setIndex] = useState(0);
  const [dir, setDir] = useState(0);

  const next = useCallback(() => {
    setDir(1);
    setIndex((i) => Math.min(i + 1, slides.length - 1));
  }, [slides.length]);

  const prev = useCallback(() => {
    setDir(-1);
    setIndex((i) => Math.max(i - 1, 0));
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") prev();
      if (e.key === "Escape") onDone?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, onDone]);

  const current = slides[index];
  if (!current) {
    return (
      <div className="flex min-h-[100dvh] items-center justify-center text-zinc-500">
        No slides to show.
      </div>
    );
  }

  const onDragEnd = (_, info) => {
    const { offset, velocity } = info;
    if (offset.x < -80 || velocity.x < -400) next();
    else if (offset.x > 80 || velocity.x > 400) prev();
  };

  return (
    <div className="relative min-h-[100dvh] overflow-hidden bg-black">
      <AnimatePresence initial={false} custom={dir} mode="wait">
        <motion.div
          key={index}
          custom={dir}
          variants={variants}
          initial="enter"
          animate="center"
          exit="exit"
          transition={{
            x: { type: "spring", stiffness: 320, damping: 34 },
            opacity: { duration: 0.25 },
          }}
          drag="x"
          dragConstraints={{ left: 0, right: 0 }}
          dragElastic={0.85}
          onDragEnd={onDragEnd}
          className="min-h-[100dvh] cursor-grab active:cursor-grabbing"
        >
          <Slide
            headline={current.headline}
            body={current.body}
            stat={current.stat}
            emoji={current.emoji}
            gradient={current.gradient}
            slideNumber={current.slideNumber}
            totalSlides={slides.length}
            feedback={current.feedback}
          />
        </motion.div>
      </AnimatePresence>

      <div className="pointer-events-none fixed bottom-0 left-0 right-0 z-20 flex items-end justify-between px-4 pb-6 sm:px-8">
        <div className="pointer-events-auto flex gap-2">
          <button
            type="button"
            onClick={prev}
            disabled={index === 0}
            className="rounded-full border border-white/20 bg-black/40 px-4 py-2 text-sm font-semibold text-white backdrop-blur-md disabled:opacity-30"
          >
            ← Prev
          </button>
          <button
            type="button"
            onClick={next}
            disabled={index >= slides.length - 1}
            className="rounded-full border border-white/20 bg-black/40 px-4 py-2 text-sm font-semibold text-white backdrop-blur-md disabled:opacity-30"
          >
            Next →
          </button>
        </div>
        <div className="pointer-events-auto flex gap-2">
          <button
            type="button"
            onClick={onRestart}
            className="rounded-full border border-white/15 px-4 py-2 text-xs font-medium text-white/70 hover:text-white"
          >
            New file
          </button>
          <button
            type="button"
            onClick={onDone}
            className="rounded-full bg-white px-5 py-2 text-sm font-bold text-zinc-950"
          >
            {index >= slides.length - 1 ? "Insights & share card" : "Skip to insights"}
          </button>
        </div>
      </div>
    </div>
  );
}
