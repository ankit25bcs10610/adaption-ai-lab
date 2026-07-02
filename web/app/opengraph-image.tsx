import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "AutoScientist Tool-Caller — the model that knows when not to call";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Dynamic social-preview card (flexbox only — next/og doesn't support CSS grid).
export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          padding: "80px",
          background: "linear-gradient(135deg, #0B1120 0%, #0F172A 60%, #14251f 100%)",
          color: "#F8FAFC",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 28 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              border: "1px solid rgba(34,197,94,0.4)",
              background: "rgba(34,197,94,0.12)",
              borderRadius: 999,
              padding: "8px 18px",
              fontSize: 24,
              color: "#22C55E",
            }}
          >
            AutoScientist · ToolCaller
          </div>
        </div>
        <div style={{ display: "flex", fontSize: 76, fontWeight: 700, lineHeight: 1.05, letterSpacing: -2 }}>
          The tool-caller that knows
        </div>
        <div
          style={{
            display: "flex",
            fontSize: 76,
            fontWeight: 700,
            lineHeight: 1.05,
            letterSpacing: -2,
            background: "linear-gradient(90deg,#22D3EE,#22C55E)",
            backgroundClip: "text",
            color: "transparent",
          }}
        >
          when not to call.
        </div>
        <div style={{ display: "flex", marginTop: 30, fontSize: 30, color: "#94A3B8", maxWidth: 900 }}>
          Refuses, clarifies, and calls — instead of hallucinating tools. Open weights on Hugging Face + Kaggle.
        </div>
        <div style={{ display: "flex", gap: 14, marginTop: 40 }}>
          {["refuse", "clarify", "call"].map((t) => (
            <div
              key={t}
              style={{
                display: "flex",
                border: "1px solid rgba(148,163,184,0.3)",
                borderRadius: 12,
                padding: "10px 20px",
                fontSize: 26,
                fontFamily: "monospace",
                color: "#E2E8F0",
              }}
            >
              {t}
            </div>
          ))}
        </div>
      </div>
    ),
    { ...size }
  );
}
