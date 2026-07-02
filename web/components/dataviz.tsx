"use client";

import { Reveal } from "@/components/reveal";
import { SpotlightCard } from "@/components/ui/spotlight-card";
import { Terminal } from "@/components/ui/terminal";

const CHARTS = [
  { src: "/charts/en_1.png", lang: "en", cap: "Value lookup · bar" },
  { src: "/charts/hi_1.png", lang: "hi", cap: "तुलना · Devanagari" },
  { src: "/charts/en_2.png", lang: "en", cap: "Extremum · pie/line" },
  { src: "/charts/hi_2.png", lang: "hi", cap: "रुझान · Hindi" },
  { src: "/charts/en_3.png", lang: "en", cap: "Trend · line" },
  { src: "/charts/hi_3.png", lang: "hi", cap: "मान · Hindi" },
] as const;

const LANG_COLOR: Record<string, string> = {
  en: "border-run/40 bg-run/10 text-run",
  hi: "border-violet/40 bg-violet/10 text-violet",
};

export function DataViz() {
  return (
    <section id="dataviz" className="relative z-10 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <Reveal className="mb-10 max-w-2xl">
          <p className="mb-3 font-mono text-sm text-violet">{"// second track · data visualization"}</p>
          <h2 className="font-display text-4xl font-bold tracking-tight sm:text-5xl">
            Charts, read in English <span className="text-grad">and Hindi.</span>
          </h2>
          <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
            A multimodal chart-understanding model trained on a <span className="text-foreground">self-verifying</span>{" "}
            synthetic dataset — every answer computed from the underlying data, so the number on the chart matches the
            gold by construction. Plus a Devanagari + romanized slice for the HackIndia track.
          </p>
        </Reveal>

        <div className="grid gap-5 lg:grid-cols-[1.1fr_0.9fr]">
          <Reveal>
            <SpotlightCard className="h-full rounded-2xl glass p-6" color="139, 92, 246">
              <ul className="space-y-4 text-sm">
                {[
                  ["Correct by construction", "Ground-truth is computed from the data, not labeled — no noise to cap accuracy."],
                  ["Hindi + romanized", "Same ground-truth, localized labels & questions; paired en/hi twins for a matched-pair Δ."],
                  ["Wide, measurable gap", "On CharXiv, GPT-4o scores ~47% on reasoning vs ~80% human — room to close."],
                ].map(([h, b]) => (
                  <li key={h} className="flex gap-3">
                    <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-violet" />
                    <span>
                      <span className="font-display font-semibold text-foreground">{h}</span>
                      <span className="block text-muted-foreground">{b}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </SpotlightCard>
          </Reveal>
          <Reveal delay={0.08}>
            <Terminal
              title="reproduce"
              lines={[
                { kind: "comment", text: "# build the self-verifying chart-QA dataset" },
                { kind: "cmd", text: "python -m src.viz.build_dataset --out data/viz" },
                { kind: "cmd", text: "python -m src.viz.gallery --data-dir data/viz" },
                { kind: "out", text: "→ 471 examples · en + hi · 0 leakage" },
              ]}
            />
          </Reveal>
        </div>

        <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3">
          {CHARTS.map((c, i) => (
            <Reveal key={c.src} delay={i * 0.05}>
              <SpotlightCard
                className="rounded-2xl glass p-3 transition-transform hover:-translate-y-1"
                color={c.lang === "hi" ? "139, 92, 246" : "34, 197, 94"}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={c.src}
                  alt={`${c.lang === "hi" ? "Hindi" : "English"} chart — ${c.cap}`}
                  loading="lazy"
                  className="w-full rounded-lg bg-white"
                />
                <div className="mt-2 flex items-center justify-between px-1">
                  <span className="text-xs text-muted-foreground">{c.cap}</span>
                  <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium ${LANG_COLOR[c.lang]}`}>
                    {c.lang}
                  </span>
                </div>
              </SpotlightCard>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
