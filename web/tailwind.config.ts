import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    container: { center: true, padding: "1.5rem", screens: { "2xl": "1152px" } },
    extend: {
      colors: {
        // shadcn / 21st.dev semantic tokens (driven by CSS vars in globals.css)
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: "hsl(var(--card))",
        border: "hsl(var(--border))",
        muted: "hsl(var(--muted))",
        "muted-foreground": "hsl(var(--muted-foreground))",
        ring: "hsl(var(--ring))",
        // brand accents — constant across themes ("code dark + run green")
        run: "#22C55E",
        cyan: "#22D3EE",
        violet: "#8B5CF6",
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        body: ["var(--font-body)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      fontSize: {
        // fluid scale — one ladder drives all display/lede sizing (see --step-* in globals.css)
        "fluid-lg": ["var(--step-1)", { lineHeight: "1.5" }],
        "fluid-2xl": ["var(--step-3)", { lineHeight: "1.15" }],
        "display-2": ["var(--step-4)", { lineHeight: "1.08" }],
        "display-1": ["var(--step-6)", { lineHeight: "1.03" }],
      },
      keyframes: {
        drift: { to: { transform: "translate3d(6vw,4vw,0) scale(1.15)" } },
        scrollx: { to: { transform: "translateX(-50%)" } },
        "border-spin": { to: { "--a": "360deg" } },
      },
      animation: {
        drift: "drift 22s ease-in-out infinite alternate",
        scrollx: "scrollx 26s linear infinite",
        "border-spin": "border-spin 6s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};

export default config;
