"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

export interface TermLine {
  text: string;
  kind?: "cmd" | "out" | "comment";
}

const COLOR = { cmd: "text-foreground", out: "text-muted-foreground", comment: "text-run/80" } as const;

/** Animated terminal that reveals lines sequentially when in view (Magic-UI "terminal" pattern). */
export function Terminal({ lines, title = "bash" }: { lines: TermLine[]; title?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let id: ReturnType<typeof setInterval>;
    let started = false;
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0]?.isIntersecting || started) return;
        started = true;
        io.disconnect();
        if (reduce) {
          setShown(lines.length);
          return;
        }
        let i = 0;
        id = setInterval(() => {
          i += 1;
          setShown(i);
          if (i >= lines.length) clearInterval(id);
        }, 420);
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      if (id) clearInterval(id);
    };
  }, [lines.length, reduce]);

  return (
    <div ref={ref} className="overflow-hidden rounded-2xl border-glow glass font-mono text-sm">
      <div className="flex items-center gap-1.5 border-b border-border/50 px-4 py-2.5">
        <span className="h-3 w-3 rounded-full bg-red-500/80" />
        <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
        <span className="h-3 w-3 rounded-full bg-run/80" />
        <span className="ml-2 text-xs text-muted-foreground">{title}</span>
      </div>
      {/* Reserve the final height up front so the line-by-line reveal doesn't shift content below (CLS). */}
      <div className="space-y-1 p-4" style={{ minHeight: `calc(${lines.length} * 1.5rem + 2rem)` }}>
        {lines.slice(0, shown).map((l, i) => (
          <div key={i} className={COLOR[l.kind ?? "out"]}>
            {l.kind === "cmd" && <span className="select-none text-run">$ </span>}
            {l.text}
            {i === shown - 1 && shown < lines.length && <span className="ml-0.5 animate-pulse">▋</span>}
          </div>
        ))}
      </div>
    </div>
  );
}
