"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

/** Count-up number that animates once when scrolled into view. Native IntersectionObserver (fires
 * immediately for above-the-fold elements). Respects reduced-motion. */
export function NumberTicker({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  duration = 1200,
  className,
}: {
  value: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const reduce = useReducedMotion();
  const [n, setN] = useState(0);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let raf = 0;
    let done = false;
    const run = () => {
      if (done) return;
      done = true;
      if (reduce) {
        setN(value);
        return;
      }
      const start = performance.now();
      const step = (t: number) => {
        const p = Math.min((t - start) / duration, 1);
        setN(value * (1 - Math.pow(1 - p, 3))); // easeOutCubic
        if (p < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };
    const io = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          run();
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [value, duration, reduce]);

  return (
    <span ref={ref} className={className}>
      {prefix}
      {n.toFixed(decimals)}
      {suffix}
    </span>
  );
}
