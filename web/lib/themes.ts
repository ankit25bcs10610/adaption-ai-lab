/**
 * Accent themes for the site — an axis orthogonal to dark/light.
 *
 * Each theme maps three colors to the primary/secondary/tertiary accent slots
 * (`run` / `cyan` / `violet`). The picker writes the id to `<html data-accent>`
 * and localStorage; globals.css swaps the matching CSS-var triplets. The hex
 * values here are used only for static previews (swatches) and the WebGL hero,
 * which can't read CSS variables directly.
 */
export const ACCENTS = [
  { id: "emerald", name: "Emerald", colors: ["#22C55E", "#22D3EE", "#8B5CF6"] },
  { id: "violet", name: "Cyber", colors: ["#8B5CF6", "#22D3EE", "#F472B6"] },
  { id: "amber", name: "Sunset", colors: ["#F59E0B", "#FB7185", "#A78BFA"] },
  { id: "ocean", name: "Azure", colors: ["#3B82F6", "#06B6D4", "#22C55E"] },
  { id: "rose", name: "Magenta", colors: ["#F43F5E", "#A855F7", "#22D3EE"] },
] as const;

export type AccentId = (typeof ACCENTS)[number]["id"];

export const DEFAULT_ACCENT: AccentId = "emerald";

export function accentById(id: string | null | undefined) {
  return ACCENTS.find((a) => a.id === id) ?? ACCENTS[0];
}
