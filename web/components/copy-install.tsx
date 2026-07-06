"use client";

import { useRef, useState } from "react";
import { Check, Copy } from "lucide-react";

/**
 * Copy-to-clipboard install command. Defaults to a command that ACTUALLY works today — installing
 * straight from the repo (there's no PyPI package yet). On clipboard failure (e.g. an insecure
 * context) it selects the text and tells the user to copy manually, instead of silently no-op-ing.
 */
export function CopyInstall({
  command = "pip install git+https://github.com/ankit25bcs10610/adaption-ai-lab.git",
}: {
  command?: string;
}) {
  const [copied, setCopied] = useState(false);
  const [failed, setFailed] = useState(false);
  const codeRef = useRef<HTMLElement>(null);

  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setFailed(false);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard API unavailable/blocked — select the text so the user can copy it manually.
      const el = codeRef.current;
      if (el) {
        const range = document.createRange();
        range.selectNodeContents(el);
        const sel = window.getSelection();
        sel?.removeAllRanges();
        sel?.addRange(range);
      }
      setFailed(true);
      setTimeout(() => setFailed(false), 4000);
    }
  }

  return (
    <div className="max-w-full">
      <div className="inline-flex max-w-full items-center gap-3 rounded-xl glass px-4 py-2.5 font-mono text-sm">
        <span className="select-none text-run">$</span>
        <code ref={codeRef} className="min-w-0 flex-1 overflow-x-auto whitespace-nowrap text-foreground/90">
          {command}
        </code>
        <button
          type="button"
          onClick={copy}
          aria-label={copied ? "Copied" : "Copy install command"}
          className="relative grid h-7 w-7 shrink-0 cursor-pointer place-items-center rounded-md border border-border/60 transition-colors before:absolute before:-inset-2 before:content-[''] hover:border-cyan/50"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-run" /> : <Copy className="h-3.5 w-3.5 text-muted-foreground" />}
        </button>
      </div>
      <span className="sr-only" aria-live="polite">
        {copied ? "Copied to clipboard" : failed ? "Copy blocked — the command is selected; press Cmd or Ctrl + C" : ""}
      </span>
      {failed && (
        <p className="mt-1.5 text-xs text-muted-foreground">Copy blocked — text selected; press ⌘/Ctrl + C.</p>
      )}
    </div>
  );
}
