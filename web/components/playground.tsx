"use client";

import { useMemo, useState } from "react";
import { Check, CircleSlash, HelpCircle, Play } from "lucide-react";
import { Reveal } from "@/components/reveal";
import { Section, SectionHeader } from "@/components/section";
import { Button } from "@/components/ui/button";
import {
  PRESET_TOOLS,
  SCENARIOS,
  simulate,
  type SimResult,
  type Tool,
} from "@/lib/toolcaller-sim";

const outcomeStyle = {
  call: { pill: "TOOL CALL", cls: "border-run/40 bg-run/10 text-run", Icon: Check },
  clarify: { pill: "NEEDS INFO", cls: "border-violet/40 bg-violet/10 text-violet", Icon: HelpCircle },
  refuse: { pill: "REFUSED", cls: "border-cyan/40 bg-cyan/10 text-cyan", Icon: CircleSlash },
} as const;

export function Playground() {
  const [tools, setTools] = useState<Tool[]>(() =>
    PRESET_TOOLS.map((t) => ({ ...t, enabled: true }))
  );
  const [query, setQuery] = useState(SCENARIOS[0].query);
  const [result, setResult] = useState<SimResult | null>(null);

  const enabledTools = useMemo(() => tools.filter((t) => t.enabled), [tools]);

  function run() {
    setResult(simulate(tools, query));
  }
  function toggle(name: string) {
    setTools((ts) => ts.map((t) => (t.name === name ? { ...t, enabled: !t.enabled } : t)));
    setResult(null);
  }
  function loadScenario(q: string) {
    setQuery(q);
    setResult(null);
  }

  return (
    <Section id="playground">
      <SectionHeader eyebrow="try it live" title="Watch it decide, in your browser." tone="cyan" className="mb-10">
        Toggle a tool off and re-run — a valid <span className="text-run">call</span> becomes a{" "}
        <span className="text-cyan">refusal</span>. That&apos;s the whole point: it won&apos;t invent a tool it
        doesn&apos;t have.
      </SectionHeader>

      <Reveal>
          <div className="grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
            {/* Tools + query */}
            <div className="rounded-2xl glass p-6">
              <p className="mb-3 eyebrow">
                Available tools
              </p>
              <div className="space-y-2">
                {tools.map((t) => (
                  <label
                    key={t.name}
                    className="flex cursor-pointer items-start gap-3 rounded-xl border border-border/60 bg-black/10 p-3 transition-colors hover:border-cyan/40"
                  >
                    <input
                      type="checkbox"
                      checked={t.enabled}
                      onChange={() => toggle(t.name)}
                      className="mt-1 h-4 w-4 cursor-pointer accent-run"
                      aria-label={`Toggle ${t.name}`}
                    />
                    <span>
                      <span className="font-mono text-sm text-foreground">{t.name}</span>
                      <span className="block text-xs text-muted-foreground">{t.description}</span>
                    </span>
                  </label>
                ))}
              </div>

              <p className="mb-2 mt-6 eyebrow">
                Try a scenario
              </p>
              <div className="flex flex-wrap gap-2">
                {SCENARIOS.map((s) => (
                  <button
                    key={s.label}
                    onClick={() => loadScenario(s.query)}
                    className="cursor-pointer rounded-full border border-border/60 px-3 py-1 text-xs text-muted-foreground transition-colors hover:border-cyan/50 hover:text-foreground"
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              <label htmlFor="pg-query" className="mb-2 mt-6 block eyebrow">
                User request
              </label>
              <textarea
                id="pg-query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                rows={2}
                className="inset-well w-full resize-none p-3 font-mono text-sm text-foreground outline-none focus:border-cyan/50"
              />
              <Button onClick={run} className="mt-3 w-full">
                <Play className="h-4 w-4" /> Run
              </Button>
            </div>

            {/* Output */}
            <div className="rounded-2xl border-glow glass p-6">
              <p className="mb-3 eyebrow">
                Model output
              </p>
              {!result ? (
                <div className="grid h-56 place-items-center text-center text-sm text-muted-foreground">
                  Press <span className="mx-1 font-mono text-foreground">Run</span> to see the decision for{" "}
                  {enabledTools.length} enabled tool{enabledTools.length === 1 ? "" : "s"}.
                </div>
              ) : (
                <Output result={result} />
              )}
            </div>
          </div>
        <p className="mt-4 text-xs text-muted-foreground/70">
          This playground runs a faithful, deterministic simulation of the model&apos;s decision logic in your
          browser (no download). The released weights produce the same JSON envelope.
        </p>
      </Reveal>
    </Section>
  );
}

function Output({ result }: { result: SimResult }) {
  const s = outcomeStyle[result.action];
  const envelope =
    result.action === "call"
      ? { action: "call", calls: result.calls }
      : { action: result.action, message: result.message };
  return (
    <div>
      <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold ${s.cls}`}>
        <s.Icon className="h-3.5 w-3.5" /> {s.pill}
      </span>
      <pre className="inset-well mt-4 overflow-x-auto p-4 font-mono text-sm text-foreground/90">
        {JSON.stringify(envelope, null, 2)}
      </pre>
      <p className="mt-3 text-xs text-muted-foreground">
        <span className="text-muted-foreground/70">why:</span> {result.rationale}
      </p>
    </div>
  );
}
