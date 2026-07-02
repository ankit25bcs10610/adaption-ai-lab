"use client";

import { Check, CircleSlash, HelpCircle } from "lucide-react";
import { Reveal } from "@/components/reveal";
import { TiltCard } from "@/components/tilt-card";

const behaviors = [
  {
    icon: Check,
    title: "Call",
    accent: "run",
    desc: "When a tool fits and all args are present, emit a schema-valid call.",
    code: `{"action":"call",
 "calls":[{"name":
  "get_weather",
  "arguments":{"city":
   "Mumbai"}}]}`,
  },
  {
    icon: CircleSlash,
    title: "Refuse",
    accent: "cyan",
    desc: "When no available tool can help, decline — never invent one.",
    code: `{"action":"refuse",
 "message":"No tool
  can satisfy this
  request."}`,
  },
  {
    icon: HelpCircle,
    title: "Clarify",
    accent: "violet",
    desc: "When a required arg is missing or the choice is ambiguous, ask.",
    code: `{"action":"clarify",
 "message":"Which city?
  I need it before I
  can proceed."}`,
  },
] as const;

const accent = {
  run: { box: "bg-run/15 border-run/30 text-run", hover: "hover:border-run/40", code: "text-run/90" },
  cyan: { box: "bg-cyan/15 border-cyan/30 text-cyan", hover: "hover:border-cyan/40", code: "text-cyan/90" },
  violet: { box: "bg-violet/15 border-violet/30 text-violet", hover: "hover:border-violet/40", code: "text-violet/90" },
} as const;

const hnKinds = [
  { name: "no_tool", color: "text-cyan", desc: "nothing applies → refuse" },
  { name: "missing_arg", color: "text-violet", desc: "required arg absent → clarify" },
  { name: "ambiguous", color: "text-run", desc: "two tools fit → clarify" },
];

export function Behaviors() {
  return (
    <section id="idea" className="relative z-10 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <Reveal className="max-w-2xl">
          <p className="mb-3 font-mono text-sm text-run">{"// the core idea"}</p>
          <h2 className="font-display text-4xl font-bold tracking-tight sm:text-5xl">
            Three behaviors. One JSON envelope.
          </h2>
          <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
            Most datasets only teach a model to call tools. Ours teaches it to{" "}
            <span className="text-foreground">call</span>, <span className="text-foreground">refuse</span>, and{" "}
            <span className="text-foreground">clarify</span> — the decisions that separate a reliable agent from a
            confident liar.
          </p>
        </Reveal>

        <div className="mt-14 grid gap-5 md:grid-cols-3">
          {behaviors.map((b, i) => {
            const a = accent[b.accent];
            const Icon = b.icon;
            return (
              <Reveal key={b.title} delay={i * 0.08}>
                <TiltCard className={`h-full cursor-pointer rounded-2xl glass p-6 transition-colors ${a.hover}`}>
                  <div data-lift style={{ transform: "translateZ(30px)" }}>
                    <div className={`mb-4 grid h-11 w-11 place-items-center rounded-xl border ${a.box}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <h3 className="font-display text-xl font-semibold">{b.title}</h3>
                    <p className="mt-2 text-sm text-muted-foreground">{b.desc}</p>
                    <pre className={`mt-4 overflow-x-auto rounded-lg bg-black/40 p-3 font-mono text-xs ${a.code}`}>
                      {b.code}
                    </pre>
                  </div>
                </TiltCard>
              </Reveal>
            );
          })}
        </div>

        {/* The moat */}
        <Reveal>
          <div className="mt-10 rounded-2xl border-glow glass p-8">
            <div className="grid items-center gap-8 md:grid-cols-[1fr_1.4fr]">
              <div>
                <p className="mb-2 font-mono text-sm text-run">{"// the moat"}</p>
                <h3 className="font-display text-2xl font-bold">Hard negatives</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  ~28% of the training set is examples where the right answer is <em>not</em> a tool call. That&apos;s
                  exactly where baselines score ~50% — and where we win.
                </p>
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                {hnKinds.map((k) => (
                  <div key={k.name} className="rounded-xl border border-border/60 bg-black/20 p-4">
                    <p className={`font-mono text-sm ${k.color}`}>{k.name}</p>
                    <p className="mt-1 text-xs text-muted-foreground">{k.desc}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
