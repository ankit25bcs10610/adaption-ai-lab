const items = [
  ["Base · Qwen2.5-Coder", "text-run"],
  ["Trained on AutoScientist", "text-cyan"],
  ["Data · xLAM (CC-BY-4.0)", "text-violet"],
  ["Data · ToolACE (Apache-2.0)", "text-run"],
  ["Released · Hugging Face + Kaggle", "text-cyan"],
  ["Eval · BFCL-aligned", "text-violet"],
] as const;

function Row({ hidden = false }: { hidden?: boolean }) {
  return (
    <div className="flex items-center gap-12" aria-hidden={hidden}>
      {items.map(([label, color], i) => (
        <span key={i} className="flex items-center gap-12">
          {label}
          <span className={color}>◆</span>
        </span>
      ))}
    </div>
  );
}

export function Marquee() {
  return (
    <section className="relative z-10 overflow-hidden border-y border-border/40 py-8">
      <div className="flex w-max animate-scrollx gap-12 font-display text-sm uppercase tracking-widest text-muted-foreground/70">
        <Row />
        <Row hidden />
      </div>
    </section>
  );
}
