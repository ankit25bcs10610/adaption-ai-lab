"use client";

import dynamic from "next/dynamic";
import { ArrowUpRight, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TiltCard } from "@/components/tilt-card";
import { CopyInstall } from "@/components/copy-install";
import { NumberTicker } from "@/components/ui/number-ticker";
import { headlineStats, representative } from "@/lib/results";
import { links } from "@/lib/links";
import { cn } from "@/lib/utils";

function parseStat(v: string): { prefix: string; num: number | null; suffix: string } {
  const m = v.match(/^([−+-]?)(\d+(?:\.\d+)?)(.*)$/);
  if (!m) return { prefix: "", num: null, suffix: v };
  return { prefix: m[1], num: parseFloat(m[2]), suffix: m[3] };
}

// r3f Canvas is client-only; load without SSR to avoid a WebGL hydration flash.
const Hero3D = dynamic(() => import("@/components/hero-3d").then((m) => m.Hero3D), {
  ssr: false,
});

const accentClass: Record<string, string> = {
  run: "text-run",
  cyan: "text-cyan",
  fg: "text-foreground",
};

export function Hero() {
  return (
    <section id="top" className="relative flex min-h-screen items-center pb-16 pt-28">
      <div className="absolute inset-0 z-0">
        <Hero3D />
      </div>

      <div className="container relative z-10 grid w-full items-center gap-12 lg:grid-cols-[1.15fr_0.85fr]">
        <div>
          <span className="mb-6 inline-flex items-center gap-2 rounded-full glass px-3.5 py-1.5 text-xs font-medium text-muted-foreground">
            <span className="h-2 w-2 animate-pulse rounded-full bg-run" />
            Adaption AutoScientist Challenge · Open Source
          </span>
          <h1 className="font-display text-5xl font-bold leading-[1.05] tracking-tight sm:text-6xl lg:text-7xl">
            The tool-caller that knows <span className="text-grad">when not to call.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
            A function-calling model fine-tuned with <span className="font-medium text-foreground">AutoScientist</span>.
            It refuses, clarifies, and calls — instead of hallucinating tools. Trained on the exact failure modes every
            other dataset ignores.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Button asChild>
              <a
                href={links.huggingface}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="View on Hugging Face (opens in a new tab)"
              >
                View on Hugging Face <ArrowUpRight className="h-4 w-4" />
              </a>
            </Button>
            <Button asChild variant="ghost">
              <a href="#how">See how it works</a>
            </Button>
          </div>

          <div className="mt-5">
            <CopyInstall />
          </div>

          <dl className="mt-10 grid max-w-lg grid-cols-3 gap-4">
            {headlineStats.map((s) => {
              const p = parseStat(s.value);
              return (
                <div key={s.label} className="rounded-xl glass p-4">
                  <dt className={cn("font-display text-2xl font-bold", accentClass[s.accent])}>
                    {p.num === null ? (
                      s.value
                    ) : (
                      <NumberTicker value={p.num} prefix={p.prefix} suffix={p.suffix} />
                    )}
                  </dt>
                  <dd className="mt-1 text-xs text-muted-foreground">{s.label}</dd>
                </div>
              );
            })}
          </dl>
          {representative && (
            <p className="mt-3 text-[11px] text-muted-foreground/70">
              Representative targets — the eval harness produces the final base-vs-fine-tuned table.
            </p>
          )}
        </div>

        {/* Floating 3D output card */}
        <div className="hidden lg:block">
          <TiltCard className="border-glow rounded-2xl glass p-5 font-mono text-sm">
            <div data-lift style={{ transform: "translateZ(38px)" }}>
              <div className="mb-4 flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-full bg-red-500/80" />
                <span className="h-3 w-3 rounded-full bg-yellow-500/80" />
                <span className="h-3 w-3 rounded-full bg-run/80" />
                <span className="ml-2 text-xs text-muted-foreground">tool_caller.jsonl</span>
              </div>
              <p className="text-muted-foreground">{"// user has no matching tool"}</p>
              <p className="mt-1 text-cyan">&quot;Write me a poem about the monsoon.&quot;</p>
              <div className="mt-3 rounded-lg border border-run/20 bg-black/40 p-3 leading-relaxed">
                <span className="text-muted-foreground">{"{"}</span>
                <br />
                {"  "}
                <span className="text-violet">&quot;action&quot;</span>:{" "}
                <span className="text-run">&quot;refuse&quot;</span>,
                <br />
                {"  "}
                <span className="text-violet">&quot;message&quot;</span>:{" "}
                <span className="text-foreground/80">&quot;No available tool</span>
                <br />
                {"   "}
                <span className="text-foreground/80">can handle this request.&quot;</span>
                <br />
                <span className="text-muted-foreground">{"}"}</span>
              </div>
              <p className="mt-3 flex items-center gap-1.5 text-xs text-run">
                <Check className="h-3.5 w-3.5" />
                No hallucinated call. This is the win.
              </p>
            </div>
          </TiltCard>
        </div>
      </div>
    </section>
  );
}
