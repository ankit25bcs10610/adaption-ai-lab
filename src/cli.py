"""Console entry point: ``autoscientist <command> [args...]``.

A thin dispatcher over the per-module ``main()`` functions, so the pip-installed package exposes one
CLI. Each subcommand delegates to the matching ``python -m src.<module>`` entry with identical flags —
it just strips the leading subcommand token so the module's own argparse sees the rest.

    autoscientist build --config config.yaml
    autoscientist eval --model <id> --data data/out/test.jsonl
    autoscientist eval-multilingual --model <id> --data data/out/test.jsonl
"""
from __future__ import annotations

import importlib
import sys
from typing import Dict, Tuple

# subcommand -> (module, one-line help)
_COMMANDS: Dict[str, Tuple[str, str]] = {
    "build": ("src.build_dataset", "Build / curate the function-calling dataset"),
    "preference": ("src.build_preference", "Build DPO preference pairs"),
    "baseline": ("src.baseline", "Compute the honest baseline number"),
    "eval": ("src.eval_bfcl", "BFCL-aligned evaluation (category breakdown)"),
    "eval-multilingual": ("src.eval_multilingual", "Matched-pair Δaccuracy(lang−en)"),
    "decompose": ("src.eval_decompose", "Base→fine-tuned gap decomposition"),
    "report": ("src.eval_report", "Render the HTML eval report"),
    "probe": ("src.reliability_probe", "Call/refuse/clarify/over-refusal probe"),
    "release": ("src.release", "Release preflight / publish"),
    "fill-card": ("src.fill_model_card", "Auto-fill MODEL_CARD.md from results/"),
    "train": ("src.train_adaption", "Adaption AutoScientist SDK run"),
}


def _usage() -> str:
    width = max(len(c) for c in _COMMANDS)
    rows = "\n".join(f"  {name:<{width}}  {desc}" for name, (_, desc) in _COMMANDS.items())
    return (
        "usage: autoscientist <command> [args...]\n\n"
        "commands:\n" + rows + "\n\n"
        "Run 'autoscientist <command> --help' for command-specific options."
    )


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(_usage())
        return 0
    cmd = sys.argv[1]
    if cmd not in _COMMANDS:
        print(f"unknown command: {cmd!r}\n\n{_usage()}", file=sys.stderr)
        return 2
    module = _COMMANDS[cmd][0]
    mod = importlib.import_module(module)
    # Hand the remaining args to the module's own argparse (drop the subcommand token).
    sys.argv = [f"autoscientist {cmd}"] + sys.argv[2:]
    ret = mod.main()
    return int(ret) if isinstance(ret, int) else 0


if __name__ == "__main__":
    sys.exit(main())
