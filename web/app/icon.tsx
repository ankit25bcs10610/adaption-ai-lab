import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 32, height: 32 };
export const contentType = "image/png";

// Favicon: the "run green" chip mark — matches the nav/footer <Logo/> lockup (green box + chip die).
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
        }}
      >
        <div
          style={{
            width: 22,
            height: 22,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderRadius: 6,
            background: "rgba(34,197,94,0.18)",
            border: "1.5px solid rgba(34,197,94,0.55)",
          }}
        >
          <div style={{ width: 9, height: 9, borderRadius: 2, background: "#22C55E" }} />
        </div>
      </div>
    ),
    { ...size }
  );
}
