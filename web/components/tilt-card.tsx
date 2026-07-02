"use client";

import { useRef, type ReactNode } from "react";
import { useReducedMotion } from "framer-motion";
import { cn } from "@/lib/utils";

/** 3D pointer-tilt wrapper. Inner elements with `data-lift` get depth via translateZ. */
export function TiltCard({
  children,
  className,
  max = 12,
}: {
  children: ReactNode;
  className?: string;
  max?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const reduce = useReducedMotion();

  function onMove(e: React.PointerEvent) {
    if (reduce || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    ref.current.style.transform = `rotateY(${px * max}deg) rotateX(${-py * max}deg)`;
  }
  function reset() {
    if (ref.current) ref.current.style.transform = "rotateY(0deg) rotateX(0deg)";
  }

  return (
    <div style={{ perspective: 1200 }} onPointerMove={onMove} onPointerLeave={reset}>
      <div
        ref={ref}
        className={cn("transition-transform duration-200 [transform-style:preserve-3d]", className)}
      >
        {children}
      </div>
    </div>
  );
}
