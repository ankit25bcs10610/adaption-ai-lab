"use client";

import { useCallback, useEffect, useState } from "react";
import { ACCENTS, DEFAULT_ACCENT, type AccentId } from "@/lib/themes";

/**
 * Reads/sets the active accent theme. State mirrors `<html data-accent>` and is
 * kept in sync via a MutationObserver, so any consumer (the picker, the 3D hero)
 * re-renders when the accent changes. `setAccent` persists to localStorage; the
 * pre-paint script in layout.tsx applies the saved value before first paint.
 */
export function useAccent() {
  const [accent, setAccentState] = useState<AccentId>(DEFAULT_ACCENT);

  useEffect(() => {
    const read = () => {
      const a = document.documentElement.getAttribute("data-accent");
      setAccentState(ACCENTS.some((x) => x.id === a) ? (a as AccentId) : DEFAULT_ACCENT);
    };
    read();
    const obs = new MutationObserver(read);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-accent"] });
    return () => obs.disconnect();
  }, []);

  const setAccent = useCallback((id: AccentId) => {
    document.documentElement.setAttribute("data-accent", id);
    try {
      localStorage.setItem("accent", id);
    } catch {
      /* private mode / storage disabled — theme still applies for this session */
    }
    setAccentState(id);
  }, []);

  return { accent, setAccent };
}
