"""Claude-backed LLM-as-judge for the quality filter — the #2 data-quality lever from the research.

Provides a `score_fn(example) -> float in [0,1]` that plugs directly into
`quality_filter.filter_examples(..., score_fn=...)`. Uses the official Anthropic Python SDK with
structured outputs so the judge must return a validated `{score, reason}` object.

Defaults to `claude-opus-4-8`. For scoring thousands of examples you can pass a cheaper model
(e.g. `claude-haiku-4-5`) — that's a cost decision, so it's explicit, not automatic.

Auth: the SDK resolves credentials from the environment (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
`ant auth login` profile). A bare Anthropic() client works after `ant auth login`.

Usage (programmatic):
    from autoscientist_toolcaller.claude_judge import make_claude_judge
    from autoscientist_toolcaller.quality_filter import filter_examples
    kept = filter_examples(examples, keep_frac=0.8, score_fn=make_claude_judge())

CLI (spot-check the judge on a sample):
    python -m autoscientist_toolcaller.claude_judge --data data/out/train.jsonl --n 20
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict

from .quality_filter import JUDGE_PROMPT, heuristic_score

# 0..10 integer score; numeric min/max aren't supported by structured outputs, so constrain via enum.
_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": list(range(0, 11))},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def make_claude_judge(
    model: str = "claude-opus-4-8", max_tokens: int = 256
) -> Callable[[Dict[str, Any]], float]:
    """Return a score_fn backed by Claude. Falls back to heuristic_score on any error per example."""
    try:
        from anthropic import Anthropic
    except ImportError as e:  # keep the pipeline runnable without the SDK installed
        raise SystemExit("Install the SDK for the Claude judge: pip install anthropic") from e

    client = Anthropic()

    def _score(example: Dict[str, Any]) -> float:
        ans = example.get("answer", {})
        calls = ans.get("calls", []) if ans.get("type") == "tool_call" else []
        prompt = JUDGE_PROMPT.format(
            tools=json.dumps(example.get("tools", []), indent=2),
            query=example.get("query", ""),
            calls=json.dumps(calls, indent=2),
        )
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
                messages=[{"role": "user", "content": prompt}],
            )
            # Guard the refusal stop reason before reading content (skill best practice).
            if resp.stop_reason == "refusal":
                return heuristic_score(example)
            text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
            obj = json.loads(text)
            return max(0.0, min(float(obj["score"]) / 10.0, 1.0))
        except Exception:
            # network hiccup, parse failure, etc. — degrade gracefully rather than zeroing the example
            return heuristic_score(example)

    return _score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/train.jsonl")
    ap.add_argument("--n", type=int, default=20, help="how many examples to score")
    ap.add_argument("--model", default="claude-opus-4-8")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][: args.n]
    judge = make_claude_judge(model=args.model)
    scores = [judge(ex) for ex in rows]
    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"[claude-judge] scored {len(scores)} examples, mean quality = {avg:.3f}")
    for ex, s in sorted(zip(rows, scores), key=lambda t: t[1])[:5]:
        print(f"  {s:.2f}  {ex.get('query', '')[:70]}")


if __name__ == "__main__":
    main()
