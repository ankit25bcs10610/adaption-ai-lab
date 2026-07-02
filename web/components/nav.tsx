"use client";

import { useState } from "react";
import { Cpu, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

const links = [
  { href: "#idea", label: "The idea" },
  { href: "#playground", label: "Playground" },
  { href: "#how", label: "How it works" },
  { href: "#bench", label: "Benchmarks" },
  { href: "#dataviz", label: "Data-Viz" },
  { href: "#release", label: "Open release" },
];

export function Nav() {
  const [open, setOpen] = useState(false);

  return (
    <header className="fixed left-4 right-4 top-4 z-50">
      <nav className="mx-auto max-w-6xl rounded-2xl glass px-5 py-3">
        <div className="flex items-center justify-between">
          <a href="#top" className="flex cursor-pointer items-center gap-2.5 font-display text-lg font-bold tracking-tight">
            <span className="grid h-8 w-8 place-items-center rounded-lg border border-run/40 bg-run/15">
              <Cpu className="h-4 w-4 text-run" aria-hidden />
            </span>
            AutoScientist<span className="text-run">·</span>ToolCaller
          </a>

          <div className="hidden items-center gap-7 text-sm font-medium text-muted-foreground md:flex">
            {links.map((l) => (
              <a key={l.href} href={l.href} className="link-underline cursor-pointer transition-colors hover:text-foreground">
                {l.label}
              </a>
            ))}
          </div>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button asChild size="sm" className="hidden sm:inline-flex">
              <a href="#release">Get the model</a>
            </Button>
            <button
              type="button"
              className="grid h-9 w-9 place-items-center rounded-lg border border-border/60 text-foreground transition-colors hover:bg-foreground/5 md:hidden"
              aria-label={open ? "Close menu" : "Open menu"}
              aria-expanded={open}
              aria-controls="mobile-menu"
              onClick={() => setOpen((v) => !v)}
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        {open && (
          <div id="mobile-menu" className="mt-3 flex flex-col gap-1 border-t border-border/50 pt-3 md:hidden">
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-foreground/5 hover:text-foreground"
              >
                {l.label}
              </a>
            ))}
            <Button asChild size="sm" className="mt-2">
              <a href="#release" onClick={() => setOpen(false)}>Get the model</a>
            </Button>
          </div>
        )}
      </nav>
    </header>
  );
}
