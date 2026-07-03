const items = [
  ["Base · Qwen2.5-Coder", "text-run"],
  ["Trained on AutoScientist", "text-cyan"],
  ["Quality +15.7% · grade C→B", "text-run"],
  ["Data · ToolACE (Apache-2.0)", "text-violet"],
  ["Execution-verified envs", "text-cyan"],
  ["Reliable in en · hi · es · fr", "text-violet"],
  ["Decontaminated vs BFCL", "text-cyan"],
  ["Released · Hugging Face + Kaggle", "text-run"],
  ["Eval · BFCL v4-aligned", "text-violet"],
] as const;

function Row({ hidden = false }: { hidden?: boolean }) {
  return (
    <div className="flex items-center gap-10" aria-hidden={hidden}>
      {items.map(([label, color], i) => (
        <span key={i} className="flex items-center gap-10">
          {label}
          <span className={color} aria-hidden>
            ◆
          </span>
        </span>
      ))}
    </div>
  );
}

export function Marquee() {
  return (
    <section className="relative z-10 overflow-hidden border-y border-border/40 py-8">
      {/* edge fade so the strip dissolves into the page instead of a hard cut */}
      <div
        className="flex w-max animate-scrollx gap-10 font-display text-sm uppercase tracking-widest text-muted-foreground/70"
        style={{
          maskImage: "linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent)",
          WebkitMaskImage: "linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent)",
        }}
      >
        <Row />
        <Row hidden />
      </div>
    </section>
  );
}
