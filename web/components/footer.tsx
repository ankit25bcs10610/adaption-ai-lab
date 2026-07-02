export function Footer() {
  return (
    <footer className="relative z-10 border-t border-border/40 px-6 py-10">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 text-sm text-muted-foreground sm:flex-row">
        <span className="font-display font-semibold text-foreground">AutoScientist · ToolCaller</span>
        <span>Built for the Adaption AutoScientist Challenge · Open source under Apache-2.0</span>
        <span className="font-mono text-xs">refuse · clarify · call</span>
      </div>
    </footer>
  );
}
