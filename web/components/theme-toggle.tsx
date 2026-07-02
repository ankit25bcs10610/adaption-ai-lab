"use client";

import * as React from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);

  const isDark = resolvedTheme === "dark";
  return (
    <button
      type="button"
      aria-label="Toggle color theme"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="grid h-9 w-9 cursor-pointer place-items-center rounded-lg glass transition-colors hover:border-cyan/50"
    >
      {/* avoid hydration mismatch: render a stable icon until mounted */}
      {mounted ? (
        isDark ? <Sun className="h-4 w-4 text-run" /> : <Moon className="h-4 w-4 text-violet" />
      ) : (
        <Sun className="h-4 w-4 opacity-0" />
      )}
    </button>
  );
}
