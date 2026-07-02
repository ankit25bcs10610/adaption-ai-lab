/**
 * Single source of truth for all external links.
 *
 * Override any of these without touching components by setting NEXT_PUBLIC_* env vars
 * (e.g. in web/.env.local) once you've published:
 *   NEXT_PUBLIC_HF_URL=https://huggingface.co/<you>/autoscientist-toolcaller
 *   NEXT_PUBLIC_KAGGLE_URL=https://www.kaggle.com/models/<you>/autoscientist-toolcaller
 *   NEXT_PUBLIC_GITHUB_URL=https://github.com/<you>/autoscientist-toolcall
 *   NEXT_PUBLIC_DISCORD_URL=https://discord.gg/THQuQhN7C9
 *
 * Defaults are LIVE, topical URLs (not dead "#" anchors) so every button works today.
 */
export const links = {
  huggingface:
    process.env.NEXT_PUBLIC_HF_URL ?? "https://huggingface.co/models?search=autoscientist",
  kaggle:
    process.env.NEXT_PUBLIC_KAGGLE_URL ?? "https://www.kaggle.com/models?search=autoscientist",
  discord: process.env.NEXT_PUBLIC_DISCORD_URL ?? "https://discord.gg/THQuQhN7C9",
  github:
    process.env.NEXT_PUBLIC_GITHUB_URL ??
    "https://github.com/search?q=autoscientist%20tool-caller&type=repositories",
} as const;

/** True once a link points at a real published artifact (not a search fallback). */
export const linkConfigured = {
  huggingface: Boolean(process.env.NEXT_PUBLIC_HF_URL),
  kaggle: Boolean(process.env.NEXT_PUBLIC_KAGGLE_URL),
  discord: true, // real invite by default
  github: Boolean(process.env.NEXT_PUBLIC_GITHUB_URL),
} as const;
