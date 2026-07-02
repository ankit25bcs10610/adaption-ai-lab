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
  const eyebrowTone = { run: "eyebrow-run", cyan: "eyebrow-cyan", violet: "eyebrow-violet" }[tone];
  return (
    <Reveal className={cn(center ? "mx-auto max-w-2xl text-center" : "max-w-2xl", className)}>
      <p className={cn("eyebrow mb-4", eyebrowTone, center && "justify-center")}>{eyebrow}</p>
      <h2 className="text-balance font-display text-fluid-2xl font-bold tracking-tight">{title}</h2>
      {children && <p className="mt-4 text-pretty text-fluid-lg leading-relaxed text-muted-foreground">{children}</p>}
    </Reveal>
  );
}
