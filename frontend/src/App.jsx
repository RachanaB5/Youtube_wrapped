import { useCallback, useState } from "react";
import ProcessingOverlay from "./components/ProcessingOverlay.jsx";
import { processWithProgress, startDemoSession, uploadHistory } from "./api";
import InsightsPanel from "./components/InsightsPanel.jsx";
import UploadScreen from "./components/UploadScreen.jsx";
import WrappedSlideshow from "./components/WrappedSlideshow.jsx";

/**
 * Map pipeline `cinematic_summary` + stats into slide props (gradient cycles 6 presets).
 */
export function buildSlidesFromInsights(insights) {
  const cinematic =
    (Array.isArray(insights?.wrapped_story) && insights.wrapped_story.length
      ? insights.wrapped_story.map((slide) => ({
          headline: slide.headline,
          lines: slide.body,
          stat: slide.stat,
          emoji: slide.emoji,
        }))
      : insights?.cinematic_summary) || [];
  const emojis = ["🎬", "🎯", "⭐", "🌙", "🌀", "🔮"];

  const topCat = insights?.top_categories?.[0];
  const feedbackWrong = topCat?.category || "entertainment";
  const total =
    insights?.summary?.total_videos_analyzed ||
    (insights?.top_categories || []).reduce((a, c) => a + (c.count || 0), 0) ||
    1;
  const topPct =
    topCat && total ? Math.round((100 * (topCat.count || 0)) / total) : null;

  const topCr = insights?.top_creators?.[0];
  const topCrCount =
    topCr?.watch_count ?? (Array.isArray(topCr) ? topCr[1] : null);

  const tp = insights?.time_patterns || {};
  const late = tp.late_night_percentage;
  const peak = tp.peak_hour;

  let clusterPct = null;
  const clusters = insights?.clusters || [];
  if (Array.isArray(clusters) && clusters.length && total) {
    const largest = [...clusters].sort(
      (a, b) => (b.size || 0) - (a.size || 0)
    )[0];
    if (largest?.size) {
      clusterPct = Math.round((100 * largest.size) / total);
    }
  }

  const stats = [
    total ? String(total) : null,
    topPct != null ? `${topPct}%` : null,
    topCrCount != null ? String(topCrCount) : null,
    late != null ? `${late}%` : peak != null ? `${peak}:00` : null,
    clusterPct != null ? `${clusterPct}%` : null,
    null,
  ];

  let blocks = cinematic;
  if (!blocks.length) {
    const n = insights?.narrative || {};
    const behaviors = (n.top_behaviors || []).slice(0, 3);
    blocks = [
      {
        headline: "Your year, unmuted",
        lines: [
          n.persona || "The algorithm definitely has opinions about you now.",
          behaviors[0] || "Every scroll left a fingerprint.",
          behaviors[1] || "Let's see what the data confesses.",
        ],
      },
      {
        headline: "The headline lane",
        lines: [
          topCat?.category
            ? `${topCat.category} kept coming back for more.`
            : "Your tastes had clear favorites.",
          behaviors[2] || "Some interests are habits dressed up as hobbies.",
          "Not random—rhythm.",
        ],
      },
      {
        headline: "The supporting cast",
        lines: [
          topCr
            ? `Shoutout to whoever you kept tapping play on.`
            : "Creators became constant companions.",
          "Replay is its own compliment.",
          "Your 'for you' page knew the assignment.",
        ],
      },
      {
        headline: "The clock never lies",
        lines: [
          late != null
            ? `Late-night share: ${late}% — the spotlight hours.`
            : peak != null
              ? `Peak hour: ${peak}:00 — when the day hands off to you.`
              : "Timing is its own personality trait.",
          "Every timestamp is a tiny autobiography.",
        ],
      },
      {
        headline: "Plot twists",
        lines: (n.hidden_patterns || [])
          .slice(0, 3)
          .concat(["The weird little repeats are the real story."])
          .slice(0, 3),
      },
      {
        headline: "Next season",
        lines: [
          n.interest_evolution ||
            "You're not the same viewer you were at the start of this export.",
          "Next year: pickier clicks, louder themes—we're calling it now.",
          "Roll credits… or don't.",
        ],
      },
    ];
  }

  return blocks.map((block, i) => {
    const lines = block.lines || block.body || [];
    const titleSnippet = [block.headline, ...lines]
      .filter(Boolean)
      .join(" ")
      .slice(0, 500);
    return {
      slideNumber: i + 1,
      headline: block.headline || "Slide",
      body: lines,
      stat: block.stat ?? stats[i] ?? null,
      emoji: block.emoji || emojis[i % emojis.length],
      gradient: i % 6,
      feedback: {
        title: titleSnippet,
        wrongCategory: feedbackWrong,
      },
    };
  });
}

export default function App() {
  const [phase, setPhase] = useState("upload");
  const [insights, setInsights] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState({ step: "loading", pct: 0 });

  const handleStartWrapped = useCallback(async (file) => {
    setError(null);
    setProgress({ step: "loading", pct: 3 });
    setPhase("processing");
    try {
      const { session_id } = await uploadHistory(file);
      setSessionId(session_id);
      const data = await processWithProgress(
        session_id,
        (step, pct) => setProgress({ step, pct }),
        { include_per_video: false, n_clusters: 12 },
      );
      setInsights({
        ...data.insights,
        session_id: data.session_id || session_id,
        insights_partial: data.partial === true,
      });
      setPhase("slideshow");
    } catch (e) {
      setError(e.message || String(e));
      setPhase("upload");
    }
  }, []);

  const handleDemo = useCallback(async () => {
    setError(null);
    setProgress({ step: "loading", pct: 3 });
    setPhase("processing");
    try {
      const { session_id } = await startDemoSession();
      setSessionId(session_id);
      const data = await processWithProgress(
        session_id,
        (step, pct) => setProgress({ step, pct }),
        { include_per_video: false, n_clusters: 12 },
      );
      setInsights({
        ...data.insights,
        session_id: data.session_id || session_id,
        insights_partial: data.partial === true,
      });
      setPhase("slideshow");
    } catch (e) {
      setError(e.message || String(e));
      setPhase("upload");
    }
  }, []);

  const slides = insights ? buildSlidesFromInsights(insights) : [];

  return (
    <div className="min-h-full bg-zinc-950">
      {error && (
        <div className="fixed top-4 left-1/2 z-[100] max-w-md -translate-x-1/2 rounded-xl bg-red-950/90 px-4 py-3 text-sm text-red-100 shadow-xl ring-1 ring-red-500/40">
          {error}
        </div>
      )}

      {phase === "processing" && (
        <ProcessingOverlay step={progress.step} pct={progress.pct} />
      )}

      {phase === "upload" && (
        <UploadScreen
          onGenerate={handleStartWrapped}
          onDemo={handleDemo}
          onError={(msg) => setError(msg)}
        />
      )}

      {phase === "slideshow" && insights && (
        <WrappedSlideshow
          slides={slides}
          onDone={() => {
            setPhase("insights");
            requestAnimationFrame(() => {
              document.getElementById("share-wrapped")?.scrollIntoView({
                behavior: "smooth",
                block: "start",
              });
            });
          }}
          onRestart={() => {
            setInsights(null);
            setSessionId(null);
            setPhase("upload");
          }}
        />
      )}

      {phase === "insights" && insights && (
        <InsightsPanel
          insights={insights}
          sessionId={sessionId || insights.session_id}
          onBack={() => setPhase("slideshow")}
          onRestart={() => {
            setInsights(null);
            setSessionId(null);
            setPhase("upload");
          }}
        />
      )}
    </div>
  );
}
