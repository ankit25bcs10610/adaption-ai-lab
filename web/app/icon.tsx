import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 32, height: 32 };
export const contentType = "image/png";

// Favicon: the "run green" tool-caller mark.
export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0F172A",
          color: "#22C55E",
          fontSize: 20,
          fontWeight: 800,
          fontFamily: "monospace",
          borderRadius: 7,
        }}
      >
        {"{}"}
      </div>
    ),
    { ...size }
  );
}
