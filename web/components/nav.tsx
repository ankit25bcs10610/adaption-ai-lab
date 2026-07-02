"use client";

import { useEffect, useState } from "react";
import { Cpu, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

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
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className="fixed inset-x-0 top-0 z-50 px-4 pt-4">
      <nav
        className={cn(
          "mx-auto max-w-6xl rounded-2xl px-5 transition-all duration-300 ease-[cubic-bezier(0.2,0.7,0.2,1)]",
          scrolled ? "glass py-2.5" : "border border-transparent bg-background/20 py-3.5 backdrop-blur-md",
        )}
      >
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
