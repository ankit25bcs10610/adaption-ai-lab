"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Palette } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { ACCENTS, accentById } from "@/lib/themes";
import { useAccent } from "@/components/use-accent";
import { cn } from "@/lib/utils";

/** Accent-theme picker — a small dropdown of color swatches, keyboard-operable per WAI-ARIA menu. */
export function ThemePicker() {
  const { accent, setAccent } = useAccent();
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => setMounted(true), []);

  const close = (returnFocus = false) => {
    setOpen(false);
    if (returnFocus) triggerRef.current?.focus();
  };
  const focusItem = (i: number) => {
    const len = ACCENTS.length;
    itemRefs.current[((i % len) + len) % len]?.focus();
  };

  // Close on outside-click or Escape (Escape also returns focus to the trigger).
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close(true);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // On open, move focus into the menu (the selected item).
  useEffect(() => {
    if (!open) return;
    const sel = Math.max(0, ACCENTS.findIndex((a) => a.id === accent));
    const t = setTimeout(() => focusItem(sel), 0);
    return () => clearTimeout(t);
  }, [open, accent]);

  const onMenuKey = (e: React.KeyboardEvent) => {
    const len = ACCENTS.length;
    const idx = itemRefs.current.findIndex((el) => el === document.activeElement);
    if (e.key === "ArrowDown") { e.preventDefault(); focusItem(idx < 0 ? 0 : idx + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); focusItem(idx < 0 ? len - 1 : idx - 1); }
    else if (e.key === "Home") { e.preventDefault(); focusItem(0); }
    else if (e.key === "End") { e.preventDefault(); focusItem(len - 1); }
    else if (e.key === "Escape") { e.preventDefault(); close(true); }
  };

  const current = accentById(accent);

  return (
    <div ref={ref} className="relative">
      <button
        ref={triggerRef}
        type="button"
        aria-label="Choose color theme"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (!open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
            e.preventDefault();
            setOpen(true);
          }
        }}
        className="grid h-9 w-9 cursor-pointer place-items-center rounded-lg glass transition-colors hover:border-run/50"
      >
        {/* Render a stable placeholder until mounted to avoid a hydration mismatch (accent is
            only known client-side, from localStorage). */}
        {mounted ? (
          <span
            className="h-3.5 w-3.5 rounded-full ring-2 ring-inset ring-white/25"
            style={{ background: current.colors[0] }}
          />
        ) : (
          <Palette className="h-4 w-4 opacity-0" />
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            role="menu"
            aria-label="Color theme"
            onKeyDown={onMenuKey}
            initial={{ opacity: 0, y: -6, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -6, scale: 0.97 }}
            transition={{ duration: 0.16, ease: [0.2, 0.7, 0.2, 1] }}
            className="absolute right-0 top-11 z-50 w-44 origin-top-right overflow-hidden rounded-xl glass p-1.5 shadow-xl"
          >
            <p className="px-2.5 pb-1 pt-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Accent theme
            </p>
            {ACCENTS.map((a, i) => {
              const selected = a.id === accent;
              return (
                <button
                  key={a.id}
                  ref={(el) => {
                    itemRefs.current[i] = el;
                  }}
                  type="button"
                  role="menuitemradio"
                  aria-checked={selected}
                  tabIndex={-1}
                  onClick={() => {
                    setAccent(a.id);
                    close(true);
                  }}
                  className={cn(
                    "flex w-full cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm outline-none transition-colors focus-visible:ring-2 focus-visible:ring-run/60",
                    selected
                      ? "bg-foreground/[0.08] text-foreground"
                      : "text-muted-foreground hover:bg-foreground/5 hover:text-foreground",
                  )}
                >
                  <span className="flex -space-x-1">
                    {a.colors.map((c, j) => (
                      <span
                        key={j}
                        className="h-3.5 w-3.5 rounded-full ring-1 ring-background"
                        style={{ background: c }}
                      />
                    ))}
                  </span>
                  <span className="flex-1 font-medium">{a.name}</span>
                  {selected && <Check className="h-3.5 w-3.5 text-run" aria-hidden />}
                </button>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
