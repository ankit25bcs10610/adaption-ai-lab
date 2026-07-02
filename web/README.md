# AutoScientist Tool-Caller — Web

Landing page in **Next.js 14 (App Router) + TypeScript + Tailwind + react-three-fiber**, set up so you can
drop in real [21st.dev](https://21st.dev) / shadcn components.

## Run

```bash
cd web
npm install
npm run dev          # http://localhost:3000
```

Build for production: `npm run build && npm start`.

### Blank page / HTTP 500 in dev?

Almost always a corrupted `.next` cache or a stale dev server left on port 3000 (Next then silently
starts the new one on **3001**, so `localhost:3000` shows the dead process). Fix:

```bash
# free port 3000, then start clean
lsof -ti tcp:3000 | xargs kill -9 2>/dev/null   # kill the stale server
npm run dev:clean                               # wipes .next + node_modules/.cache, then next dev
```

`npm run clean` wipes the caches on their own. If the terminal says "Port 3000 is in use, trying 3001",
open the port it actually printed.

## Stack

- **3D:** `@react-three/fiber` + `three` — the hero scene is `components/hero-3d.tsx` (wireframe model core +
  orbiting particle "tools" + pointer parallax). Loaded with `next/dynamic({ ssr: false })`.
- **Animation:** `framer-motion` (`Reveal`, benchmark bars, tilt cards). All respect `prefers-reduced-motion`.
- **Theming:** `next-themes`, dark by default, light theme included. Toggle in the nav (`theme-toggle.tsx`).
  Colors are CSS variables in `app/globals.css`; brand accents (run/cyan/violet) are constant.
- **UI primitives:** shadcn-style `components/ui/button.tsx`; `components.json` is preconfigured.

## Wire in real benchmark numbers

`lib/results.ts` ships representative values. After running the eval pipeline in the repo root:

```bash
# from repo root: produce results/baseline.json and results/eval.json
npm --prefix web run sync:results
```

`scripts/sync-results.mjs` reads `../results/baseline.json` + `../results/eval.json`, rewrites
`lib/results.ts` with the real base-vs-fine-tuned numbers, and flips the "representative" note off.

## Add 21st.dev components

The project is shadcn-compatible (`components.json`, `lib/utils.ts` `cn`, Tailwind tokens). Pull any
21st.dev component with the shadcn CLI:

```bash
npx shadcn@latest add "https://21st.dev/r/<author>/<component>"
```

It installs into `components/ui/`. Good candidates to swap in: an animated hero background, a bento grid for
the "how it works" section, marquee, or spotlight cards. Then import and drop them into the section files.

## Structure

```
app/
  layout.tsx        fonts (Space Grotesk / DM Sans / JetBrains Mono) + ThemeProvider
  page.tsx          section composition
  globals.css       theme tokens + glass/aurora/glow utilities
components/
  hero-3d.tsx       react-three-fiber scene
  hero.tsx  nav.tsx  marquee.tsx  behaviors.tsx  pipeline.tsx  benchmarks.tsx  cta.tsx  footer.tsx
  reveal.tsx  tilt-card.tsx  theme-toggle.tsx  theme-provider.tsx
  ui/button.tsx     shadcn-style primitive
lib/
  utils.ts          cn()
  results.ts        benchmark data (sync from ../results)
scripts/
  sync-results.mjs
```
