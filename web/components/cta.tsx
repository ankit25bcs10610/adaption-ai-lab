"use client";

import { Reveal } from "@/components/reveal";
import { links, linkConfigured } from "@/lib/links";

const KaggleIcon = () => (
  <svg className="h-6 w-6 text-cyan" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.285.18.046.121.025.212-.062.258l-6.443 6.436 7.058 8.437c.108.121.09.212-.037.334Z" />
  </svg>
);

const DiscordIcon = () => (
  <svg className="h-6 w-6 text-violet" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M20.317 4.3698a19.7913 19.7913 0 00-4.8851-1.5152.0741.0741 0 00-.0785.0371c-.211.3753-.4447.8648-.6083 1.2495-1.8447-.2762-3.68-.2762-5.4868 0-.1636-.3933-.4058-.8742-.6177-1.2495a.077.077 0 00-.0785-.037 19.7363 19.7363 0 00-4.8852 1.515.0699.0699 0 00-.0321.0277C.5334 9.0458-.319 13.5799.0992 18.0578a.0824.0824 0 00.0312.0561c2.0528 1.5076 4.0413 2.4228 5.9929 3.0294a.0777.0777 0 00.0842-.0276c.4616-.6304.8731-1.2952 1.226-1.9942a.076.076 0 00-.0416-.1057c-.6528-.2476-1.2743-.5495-1.8722-.8923a.077.077 0 01-.0076-.1277c.1258-.0943.2517-.1923.3718-.2914a.0743.0743 0 01.0776-.0105c3.9278 1.7933 8.18 1.7933 12.0614 0a.0739.0739 0 01.0785.0095c.1202.099.246.1981.3728.2924a.077.077 0 01-.0066.1276 12.2986 12.2986 0 01-1.873.8914.0766.0766 0 00-.0407.1067c.3604.698.7719 1.3628 1.225 1.9932a.076.076 0 00.0842.0286c1.961-.6067 3.9495-1.5219 6.0023-3.0294a.077.077 0 00.0313-.0552c.5004-5.177-.8382-9.6739-3.5485-13.6604a.061.061 0 00-.0312-.0286zM8.02 15.3312c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9555-2.4189 2.157-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.9555 2.4189-2.1569 2.4189zm7.9748 0c-1.1825 0-2.1569-1.0857-2.1569-2.419 0-1.3332.9554-2.4189 2.1569-2.4189 1.2108 0 2.1757 1.0952 2.1568 2.419 0 1.3332-.946 2.4189-2.1568 2.4189Z" />
  </svg>
);

const GitHubIcon = () => (
  <svg className="h-6 w-6 text-foreground" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
  </svg>
);

const cards = [
  {
    href: links.huggingface,
    configured: linkConfigured.huggingface,
    label: "Hugging Face",
    sub: "model card + dataset",
    hover: "hover:border-run/50",
    icon: (
      <span
        className="grid h-12 w-12 place-items-center rounded-xl font-display text-lg font-bold"
        style={{ background: "#FFD21E22", border: "1px solid #FFD21E55", color: "#FFD21E" }}
      >
        HF
      </span>
    ),
  },
  {
    href: links.kaggle,
    configured: linkConfigured.kaggle,
    label: "Kaggle",
    sub: "model mirror",
    hover: "hover:border-cyan/50",
    icon: (
      <span className="grid h-12 w-12 place-items-center rounded-xl border border-cyan/40 bg-cyan/15">
        <KaggleIcon />
      </span>
    ),
  },
  {
    href: links.discord,
    configured: linkConfigured.discord,
    label: "Discord",
    sub: "#autoscient-challenge",
    hover: "hover:border-violet/50",
    icon: (
      <span className="grid h-12 w-12 place-items-center rounded-xl border border-violet/40 bg-violet/15">
        <DiscordIcon />
      </span>
    ),
  },
  {
    href: links.github,
    configured: linkConfigured.github,
    label: "GitHub",
    sub: "full pipeline",
    hover: "hover:border-foreground/40",
    icon: (
      <span className="grid h-12 w-12 place-items-center rounded-xl border border-border bg-foreground/10">
        <GitHubIcon />
      </span>
    ),
  },
];

export function CTA() {
  const anyUnconfigured = cards.some((c) => !c.configured);
  return (
    <section id="release" className="relative z-10 scroll-mt-24 px-6 py-24 sm:py-28">
      <Reveal className="mx-auto max-w-4xl text-center">
        <p className="eyebrow eyebrow-run mb-3 justify-center">open release</p>
        <h2 className="text-balance font-display text-display-2 font-bold tracking-tight">
          Weights, data, and demo. <span className="text-grad">All open.</span>
        </h2>
        <p className="mx-auto mt-5 max-w-xl text-pretty text-fluid-lg text-muted-foreground">
          Everything is public — model card, dataset card, eval harness, and a live Space. Reproduce it, fork it, beat it.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {cards.map((c) => (
            <a
              key={c.label}
              href={c.href}
              target="_blank"
              rel="noopener noreferrer"
              aria-label={`${c.label} — ${c.sub} (opens in a new tab)`}
              className={`relative flex cursor-pointer flex-col items-center gap-3 rounded-2xl glass p-5 transition-all hover:-translate-y-1 ${c.hover}`}
            >
              {!c.configured && (
                <span className="absolute right-2.5 top-2.5 rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                  soon
                </span>
              )}
              {c.icon}
              <span className="font-semibold">{c.label}</span>
              <span className="text-xs text-muted-foreground">{c.sub}</span>
            </a>
          ))}
        </div>
        {anyUnconfigured && (
          <p className="mt-8 text-sm text-muted-foreground">
            Links marked <span className="text-muted-foreground/80">soon</span> point to topical searches until you
            publish. Set <code className="text-muted-foreground/80">NEXT_PUBLIC_HF_URL</code>,{" "}
            <code className="text-muted-foreground/80">NEXT_PUBLIC_KAGGLE_URL</code>, and{" "}
            <code className="text-muted-foreground/80">NEXT_PUBLIC_GITHUB_URL</code> in{" "}
            <code className="text-muted-foreground/80">web/.env.local</code> to point them at your artifacts.
          </p>
        )}
      </Reveal>
    </section>
  );
}
