import { Cpu } from "lucide-react";
import { links } from "@/lib/links";

const cols = [
  {
    title: "Explore",
    items: [
      { label: "The idea", href: "#idea" },
      { label: "Playground", href: "#playground" },
      { label: "How it works", href: "#how" },
      { label: "Benchmarks", href: "#bench" },
    ],
  },
  {
    title: "Artifacts",
    items: [
      { label: "Hugging Face", href: links.huggingface, ext: true },
      { label: "Kaggle", href: links.kaggle, ext: true },
      { label: "GitHub", href: links.github, ext: true },
    ],
  },
];

export function Footer() {
  return (
    <footer className="relative z-10 border-t border-border/40 px-6 py-14">
      <div className="mx-auto max-w-6xl">
        <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <div className="flex items-center gap-2.5 font-display text-lg font-bold tracking-tight">
              <span className="grid h-8 w-8 place-items-center rounded-lg border border-run/40 bg-run/15">
                <Cpu className="h-4 w-4 text-run" aria-hidden />
              </span>
              AutoScientist<span className="text-run">·</span>ToolCaller
            </div>
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-muted-foreground">
              A data-centric function-calling model + audited dataset. It refuses, clarifies, and calls —
              instead of hallucinating tools.
            </p>
            <p className="mt-4 font-mono text-xs text-muted-foreground/70">refuse · clarify · call</p>
          </div>

          {cols.map((col) => (
            <nav key={col.title} aria-label={col.title}>
              <h3 className="font-display text-sm font-semibold text-foreground">{col.title}</h3>
              <ul className="mt-4 space-y-2.5 text-sm">
                {col.items.map((it) => (
                  <li key={it.label}>
                    <a
                      href={it.href}
                      {...("ext" in it && it.ext ? { target: "_blank", rel: "noopener noreferrer" } : {})}
                      className="text-muted-foreground transition-colors hover:text-foreground"
                    >
                      {it.label}
                    </a>
                  </li>
                ))}
              </ul>
            </nav>
          ))}
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-3 border-t border-border/40 pt-6 text-xs text-muted-foreground sm:flex-row">
          <span>Built for the Adaption AutoScientist Challenge · Apache-2.0</span>
          <span>Model + dataset open source · reproduce with one command</span>
        </div>
      </div>
    </footer>
  );
}
