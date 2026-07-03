"""Console entry point: ``autoscientist <command> [args...]``.

A thin dispatcher over the per-module ``main()`` functions, so the pip-installed package exposes one
CLI. Each subcommand delegates to the matching ``python -m autoscientist_toolcaller.<module>`` entry with identical flags —
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
    "build": ("autoscientist_toolcaller.build_dataset", "Build / curate the function-calling dataset"),
    "preference": ("autoscientist_toolcaller.build_preference", "Build DPO preference pairs"),
    "baseline": ("autoscientist_toolcaller.baseline", "Compute the honest baseline number"),
    "eval": ("autoscientist_toolcaller.eval_bfcl", "BFCL-aligned evaluation (category breakdown)"),
    "eval-multilingual": ("autoscientist_toolcaller.eval_multilingual", "Matched-pair Δaccuracy(lang−en)"),
    "decompose": ("autoscientist_toolcaller.eval_decompose", "Base→fine-tuned gap decomposition"),
    "report": ("autoscientist_toolcaller.eval_report", "Render the HTML eval report"),
    "probe": ("autoscientist_toolcaller.reliability_probe", "Call/refuse/clarify/over-refusal probe"),
    "release": ("autoscientist_toolcaller.release", "Release preflight / publish"),
    "fill-card": ("autoscientist_toolcaller.fill_model_card", "Auto-fill MODEL_CARD.md from results/"),
    "train": ("autoscientist_toolcaller.train_adaption", "Adaption AutoScientist SDK run"),
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
