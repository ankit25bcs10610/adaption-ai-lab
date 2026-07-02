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
  // Published, real artifacts (override via NEXT_PUBLIC_* if you fork).
  huggingface:
    process.env.NEXT_PUBLIC_HF_URL ?? "https://huggingface.co/pandeyankit84/autoscientist-toolcaller",
  hfDataset:
    process.env.NEXT_PUBLIC_HF_DATASET_URL ??
    "https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset",
  kaggle:
    process.env.NEXT_PUBLIC_KAGGLE_URL ??
    "https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset",
  discord: process.env.NEXT_PUBLIC_DISCORD_URL ?? "https://discord.gg/THQuQhN7C9",
  github: process.env.NEXT_PUBLIC_GITHUB_URL ?? "https://github.com/ankit25bcs10610/adaption-ai-lab",
} as const;

/** True once a link points at a real published artifact (not a search fallback). */
export const linkConfigured = {
  huggingface: true, // published: pandeyankit84/autoscientist-toolcaller
  kaggle: true, // published: pandeyankit99/autoscientist-toolcaller-dataset
  discord: true,
  github: true, // ankit25bcs10610/adaption-ai-lab
} as const;
