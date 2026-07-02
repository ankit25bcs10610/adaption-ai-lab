import type { ReactNode } from "react";
import { Reveal } from "@/components/reveal";
import { cn } from "@/lib/utils";

/**
 * Consistent section shell — one vertical rhythm, one container width, and a scroll-margin so anchored
 * sections clear the fixed navbar when linked (a common production papercut). Use everywhere so spacing
 * never drifts section-to-section.
 */
export function Section({
  id,
  children,
  className,
  container = true,
}: {
  id: string;
  children: ReactNode;
  className?: string;
  container?: boolean;
}) {
  return (
    <section id={id} className={cn("relative z-10 scroll-mt-24 px-6 py-24 sm:py-28", className)}>
      {container ? <div className="mx-auto max-w-6xl">{children}</div> : children}
    </section>
  );
}

/** Eyebrow + heading + lede, with a consistent type scale. `tone` colors the mono eyebrow. */
export function SectionHeader({
  eyebrow,
  title,
  children,
  tone = "run",
  center = false,
  className,
}: {
  eyebrow: string;
  title: ReactNode;
  children?: ReactNode;
  tone?: "run" | "cyan" | "violet";
  center?: boolean;
  className?: string;
}) {
  const toneClass = { run: "text-run", cyan: "text-cyan", violet: "text-violet" }[tone];
  return (
    <Reveal className={cn(center ? "mx-auto max-w-2xl text-center" : "max-w-2xl", className)}>
      <p className={cn("mb-3 font-mono text-sm", toneClass)}>{"// "}{eyebrow}</p>
      <h2 className="text-balance font-display text-4xl font-bold tracking-tight sm:text-5xl">{title}</h2>
      {children && <p className="mt-4 text-pretty text-lg leading-relaxed text-muted-foreground">{children}</p>}
    </Reveal>
  );
}
