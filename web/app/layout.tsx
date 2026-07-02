import type { Metadata, Viewport } from "next";
import { Space_Grotesk, DM_Sans, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import "./globals.css";

const display = Space_Grotesk({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-display",
});
const body = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-body",
});
const mono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-mono",
});

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";
const TITLE = "AutoScientist Tool-Caller — the model that knows when not to call";
const DESCRIPTION =
  "A function-calling AI model + dataset fine-tuned with Adaption AutoScientist. It refuses, clarifies, and calls — instead of hallucinating tool calls. Adaptive Data quality +15.7% (grade C→B). Open-source on Hugging Face and Kaggle.";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: TITLE, template: "%s · AutoScientist Tool-Caller" },
  description: DESCRIPTION,
  applicationName: "AutoScientist Tool-Caller",
  authors: [{ name: "Ankit Pandey", url: "https://github.com/ankit25bcs10610" }],
  creator: "Ankit Pandey",
  keywords: [
    "function calling", "tool use", "LLM agents", "BFCL", "Adaption AutoScientist",
    "Adaptive Data", "hard negatives", "hallucination", "data-centric AI", "fine-tuning",
    "Qwen2.5-Coder", "DPO", "open source model",
  ],
  category: "technology",
  alternates: { canonical: SITE_URL },
  robots: { index: true, follow: true, googleBot: { index: true, follow: true, "max-image-preview": "large" } },
  openGraph: {
    title: TITLE,
    description: "The tool-caller that knows when not to call. Open-source model + audited dataset.",
    url: SITE_URL,
    siteName: "AutoScientist Tool-Caller",
    type: "website",
    locale: "en_US",
  },
  twitter: {
    card: "summary_large_image",
    title: "AutoScientist Tool-Caller",
    description: "The tool-caller that knows when not to call. Data-centric, open source.",
  },
};

export const viewport: Viewport = {
  themeColor: "#0F172A",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false}>
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
