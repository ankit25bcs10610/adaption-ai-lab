import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "AutoScientist Tool-Caller",
    short_name: "ToolCaller",
    description:
      "A function-calling AI model + dataset that refuses, clarifies, and calls — instead of hallucinating tool calls.",
    start_url: "/",
    display: "standalone",
    background_color: "#0F172A",
    theme_color: "#0F172A",
    categories: ["developer", "productivity", "education"],
    icons: [{ src: "/icon", sizes: "32x32", type: "image/png" }],
  };
}
