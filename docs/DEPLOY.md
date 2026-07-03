# Deploy the live demo (premium Next.js site) to Vercel

The Gradio Space is already live; this deploys the far nicer **Next.js** site as the primary demo
(bonus points; judges reward a polished live demo). Vercel auto-detects Next.js — no `vercel.json` needed.

## Option A — one command (Vercel CLI)
```bash
cd web
npm i -g vercel
vercel                 # first run: link/create project; set Root Directory = "web" if asked
vercel --prod          # promote to production
```

## Option B — GitHub (no CLI)
1. vercel.com → **Add New → Project** → import `ankit25bcs10610/adaption-ai-lab`.
2. **Root Directory: `web`** (important — the Next app lives there). Framework preset auto-detects Next.js.
3. Deploy.

## Environment variables (Vercel → Project → Settings → Environment Variables)
All optional (the site has live fallbacks), but set these so metadata/links are exact — values in
`web/.env.example`:
```
NEXT_PUBLIC_SITE_URL   = https://<your-project>.vercel.app
NEXT_PUBLIC_HF_URL     = https://huggingface.co/pandeyankit84/autoscientist-toolcaller
NEXT_PUBLIC_KAGGLE_URL = https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
NEXT_PUBLIC_GITHUB_URL = https://github.com/ankit25bcs10610/adaption-ai-lab
NEXT_PUBLIC_DISCORD_URL= https://discord.gg/THQuQhN7C9
```
Set `NEXT_PUBLIC_SITE_URL` first (drives OpenGraph, robots, sitemap), then redeploy so the OG image and
sitemap use the real domain.

## After deploy
1. Copy the production URL.
2. Set it as `NEXT_PUBLIC_SITE_URL` and redeploy (one click).
3. Add the URL to `docs/social_posts.md` (`<DEMO_URL>`) and the Part-2 submission (`docs/PART2_SUBMISSION.md`).
4. Verify: `curl -I https://<url>` → 200; check `/robots.txt`, `/sitemap.xml`, `/opengraph-image` render.

Build is already verified locally (`npm run build`): 7 routes, types + lint clean, ~159 kB first load.
