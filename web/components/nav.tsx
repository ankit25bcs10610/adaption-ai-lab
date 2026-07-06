"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Logo } from "@/components/logo";
import { ThemePicker } from "@/components/theme-picker";
import { cn } from "@/lib/utils";

const links = [
  { href: "#idea", label: "The idea" },
  { href: "#playground", label: "Playground" },
  { href: "#agent", label: "Agent" },
  { href: "#how", label: "How it works" },
  { href: "#bench", label: "Benchmarks" },
  { href: "#dataviz", label: "Data-Viz" },
  { href: "#release", label: "Open release" },
];

export function Nav() {
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [active, setActive] = useState("");

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Highlight the nav link for the section currently in view (band centered in the viewport).
  useEffect(() => {
    const els = links
      .map((l) => document.getElementById(l.href.slice(1)))
      .filter((el): el is HTMLElement => el !== null);
    if (!els.length) return;
    const io = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (vis[0]) setActive(vis[0].target.id);
      },
      { rootMargin: "-45% 0px -50% 0px", threshold: [0, 0.5, 1] },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
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
          <a href="#top" aria-label="AutoScientist ToolCaller — home" className="cursor-pointer rounded-lg">
            <Logo />
          </a>

          {/* xl breakpoint: 7 links + logo + theme picker + CTA ≈ 1080px wide — at md they wrap into the logo */}
          <div className="hidden items-center gap-1 text-sm font-medium xl:flex">
            {links.map((l) => {
              const isActive = active === l.href.slice(1);
              return (
                <a
                  key={l.href}
                  href={l.href}
                  className={cn(
                    "relative cursor-pointer whitespace-nowrap rounded-full px-3 py-1.5 transition-colors",
                    isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {isActive && (
                    <motion.span
                      layoutId="nav-pill"
                      className="absolute inset-0 -z-10 rounded-full bg-foreground/[0.08]"
                      transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    />
                  )}
                  {l.label}
                </a>
              );
            })}
          </div>

          <div className="flex items-center gap-2">
            <ThemePicker />
            <Button asChild size="sm" className="hidden lg:inline-flex">
              <a href="#release">Get the model</a>
            </Button>
            <button
              type="button"
              className="grid h-11 w-11 place-items-center rounded-lg border border-border/60 text-foreground transition-colors hover:bg-foreground/5 xl:hidden"
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
          <div id="mobile-menu" className="mt-3 flex flex-col gap-1 border-t border-border/50 pt-3 xl:hidden">
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
