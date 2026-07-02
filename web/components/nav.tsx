"use client";

import { Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";

const links = [
  { href: "#idea", label: "The idea" },
  { href: "#playground", label: "Playground" },
  { href: "#how", label: "How it works" },
  { href: "#bench", label: "Benchmarks" },
  { href: "#release", label: "Open release" },
];

export function Nav() {
  return (
    <header className="fixed left-4 right-4 top-4 z-50">
      <nav className="mx-auto flex max-w-6xl items-center justify-between rounded-2xl glass px-5 py-3">
        <a href="#top" className="flex cursor-pointer items-center gap-2.5 font-display text-lg font-bold tracking-tight">
          <span className="grid h-8 w-8 place-items-center rounded-lg border border-run/40 bg-run/15">
            <Cpu className="h-4 w-4 text-run" />
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
          <Button asChild size="sm">
            <a href="#release">Get the model</a>
          </Button>
        </div>
      </nav>
    </header>
  );
}
