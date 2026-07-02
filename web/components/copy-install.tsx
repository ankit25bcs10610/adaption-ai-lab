"use client";

import { useState } from "react";
import { Check, Copy } from "lucide-react";

/** Copy-to-clipboard install command — the #1 dev-tool micro-conversion. */
export function CopyInstall({ command = "pip install autoscientist-toolcaller" }: { command?: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable (e.g., insecure context) — no-op */
    }
  }

  return (
    <div className="inline-flex items-center gap-3 rounded-xl glass px-4 py-2.5 font-mono text-sm">
      <span className="select-none text-run">$</span>
      <code className="text-foreground/90">{command}</code>
      <button
        type="button"
        onClick={copy}
        aria-label={copied ? "Copied" : "Copy install command"}
        className="grid h-7 w-7 cursor-pointer place-items-center rounded-md border border-border/60 transition-colors hover:border-cyan/50"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-run" /> : <Copy className="h-3.5 w-3.5 text-muted-foreground" />}
      </button>
      <span className="sr-only" aria-live="polite">
        {copied ? "Copied to clipboard" : ""}
      </span>
    </div>
  );
}
