"""Canonical chart-QA example format + multimodal chat-JSONL rendering (Data-Viz track).

Mirrors the function-calling pipeline's conventions: lazy heavy imports (PIL only inside the
functions that touch pixels), a single source of truth for the chat message structure so base and
fine-tuned models see byte-identical prompts, and plain-dict wire examples.

Canonical example (plain dict):
    {
      "image": <path str | data-URI str | raw bytes | PIL.Image | {"bytes"|"path": ...}>,
      "question": "...",
      "answer": <str|int|float|bool|list>,   # native gold; scorer normalizes both sides
      "chart_type": "bar" | "line" | ...,
      "qa_kind": "value_lookup" | "max" | "min" | "compare" | "sum" | "count" | "trend"
                 | "proportion" | "difference" | "mean",
      "meta": {"source", "lang", "script", "chart_type", "qa_kind", "answer_type", "seed", ...},
    }

Together/Adaption multimodal fine-tuning wants CHAT JSONL (messages[]) with images as inline base64
data URIs inside a user turn's content array. `build_chat_messages` is that single source of truth.
"""
from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = (
    "You are a precise chart-reading assistant. Answer the question about the chart image with only "
    "the answer value — a number, a category name, yes/no, or a trend word — and nothing else."
)


# --------------------------------------------------------------------------------------
# Answer stringification (fixed-point, never scientific notation — F25)
# --------------------------------------------------------------------------------------
def _scalar_to_text(x: Any) -> str:
    if isinstance(x, bool):
        return "Yes" if x else "No"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x == int(x):
            return str(int(x))
        s = f"{x:.4f}".rstrip("0").rstrip(".")  # fixed-point, trimmed; never 1.2e+06
        return s
    return str(x).strip()


def answer_to_text(answer: Any) -> str:
    """Deterministic human-readable string form of the gold answer (for the assistant turn / display).

    Scoring compares against the NATIVE `answer`, not this string, to avoid double-rounding (F26).
    """
    if isinstance(answer, (list, tuple)):
        return ", ".join(_scalar_to_text(v) for v in answer)
    return _scalar_to_text(answer)


# --------------------------------------------------------------------------------------
# Image -> data URI (lazy PIL). Dispatch order matters (F27): bytes -> data-uri -> http -> dict -> PIL -> path
# --------------------------------------------------------------------------------------
def _sniff_mime(b: bytes) -> Optional[str]:
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:4] == b"GIF8":
        return "image/gif"
    if len(b) >= 12 and b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return None


def _pil_to_png_bytes(img, max_side: int = 1024) -> bytes:
    import io

    w, h = img.size
    scale = max(w, h) / max_side
    if scale > 1:
        img = img.resize((int(w / scale), int(h / scale)))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_data_uri(image: Any, max_side: int = 1024, max_bytes: int = 10 * 1024 * 1024) -> str:
    """Return 'data:image/<fmt>;base64,<...>' for any accepted image form. Idempotent on data URIs."""
    if isinstance(image, (bytes, bytearray)):
        b = bytes(image)
        mime = _sniff_mime(b) or "image/png"
        if _sniff_mime(b) is None:  # unknown -> re-encode via PIL
            from PIL import Image
            import io

            b = _pil_to_png_bytes(Image.open(io.BytesIO(b)), max_side)
            mime = "image/png"
        return f"data:{mime};base64," + base64.b64encode(b).decode("ascii")
    if isinstance(image, str) and image.startswith("data:image/"):
        return image
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        return image  # URL passthrough (Adaptive Data can ingest URLs)
    if isinstance(image, dict):
        if image.get("bytes"):
            return image_to_data_uri(image["bytes"], max_side, max_bytes)
        if image.get("path"):
            return image_to_data_uri(image["path"], max_side, max_bytes)
        raise ValueError("image dict has neither 'bytes' nor 'path'")
    if hasattr(image, "save") and hasattr(image, "mode"):  # PIL duck-type
        b = _pil_to_png_bytes(image, max_side)
        return "data:image/png;base64," + base64.b64encode(b).decode("ascii")
    if isinstance(image, str):  # filesystem path
        with open(image, "rb") as f:
            return image_to_data_uri(f.read(), max_side, max_bytes)
    raise ValueError(f"unrecognized image type: {type(image)!r}")


# --------------------------------------------------------------------------------------
# Chat message construction — SINGLE SOURCE OF TRUTH for train + eval
# --------------------------------------------------------------------------------------
def build_chat_messages(
    image: Any, question: str, answer: Optional[Any] = None, inline_image: bool = True
) -> List[Dict[str, Any]]:
    """Build the messages[] array. If `answer` is given, includes the assistant turn (training row);
    otherwise omit it (eval prompt). `inline_image=False` keeps a raw path (for a manifest)."""
    if not question or not question.strip():
        raise ValueError("empty question")
    img_field = image_to_data_uri(image) if inline_image else image
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_field}},
                {"type": "text", "text": question},
            ],
        },
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer_to_text(answer)})
    return messages


def to_chat_row(example: Dict[str, Any], inline_image: bool = True) -> Dict[str, Any]:
    """Render a canonical example into a training row: {"messages": [...]}."""
    return {
        "messages": build_chat_messages(
            example["image"], example["question"], example.get("answer"), inline_image
        )
    }


def build_eval_prompt_messages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Eval-time messages (no assistant turn). Same structure as training minus the answer."""
    return build_chat_messages(example["image"], example["question"], answer=None)
