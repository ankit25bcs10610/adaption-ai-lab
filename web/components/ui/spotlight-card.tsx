"use client";

import { useRef, type ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Card wrapper with a pointer-following radial glow (21st.dev/Magic-UI "spotlight" pattern). */
export function SpotlightCard({
  children,
  className,
  color = "34, 197, 94", // run green (rgb)
}: {
  children: ReactNode;
  className?: string;
  color?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);

  function onMove(e: React.PointerEvent) {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--x", `${e.clientX - r.left}px`);
    el.style.setProperty("--y", `${e.clientY - r.top}px`);
  }

  return (
    <div
      ref={ref}
      onPointerMove={onMove}
      className={cn("group relative overflow-hidden", className)}
      style={{ ["--c" as string]: color }}
    >
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        style={{
          background:
            "radial-gradient(320px circle at var(--x, 50%) var(--y, 0), rgba(var(--c), 0.16), transparent 60%)",
        }}
      />
      <div className="relative">{children}</div>
    </div>
  );
}
