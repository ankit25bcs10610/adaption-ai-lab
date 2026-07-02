"""Publish weights + dataset to Hugging Face AND Kaggle (both are mandatory for the challenge).

Usage:
  python -m src.release hf-dataset   --repo USER/autoscientist-toolcall-dataset --dir data/out
  python -m src.release hf-model     --repo USER/autoscientist-toolcall        --dir path/to/weights
  python -m src.release kaggle-model --slug USER/autoscientist-toolcall        --dir path/to/weights

Requires HF_TOKEN (or `hf auth login`) and KAGGLE_USERNAME/KAGGLE_KEY.
"""
from __future__ import annotations

import argparse
import os

# Placeholder markers that must NEVER survive into a published card.
_PLACEHOLDERS = ["YOUR_USERNAME", "<user>", "<you>", "value: 0.000", "0.000", "<- fill", "← fill", "TODO", "FIXME"]
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def preflight(out_dir: str = "data/out", card_paths=None, require_pref: bool = False,
              check_manifest: bool = True, manifest_path: str = "results/manifest.json") -> list:
    """Return a list of BLOCKING release problems (empty list = safe to publish).

    Pure filesystem/string checks — never touches the network — so it can gate every upload and run in
    CI. Verifies: required data artifacts exist, a LICENSE is present, and the cards carry no residual
    placeholders / unfilled metric rows.
    """
    problems = []
    required = ["train.jsonl", "val.jsonl", "test.jsonl", "stats.json"]
    if require_pref:
        required.append("pref.jsonl")
    for name in required:
        if not os.path.exists(os.path.join(out_dir, name)):
            problems.append(f"missing artifact: {os.path.join(out_dir, name)}")

    if not os.path.exists(os.path.join(_REPO_ROOT, "LICENSE")):
        problems.append("missing LICENSE at repo root (cards declare apache-2.0)")

    if card_paths is None:
        card_paths = [p for p in (os.path.join(_REPO_ROOT, "MODEL_CARD.md"),
                                  os.path.join(_REPO_ROOT, "DATASET_CARD.md")) if os.path.exists(p)]
    for cp in card_paths:
        try:
            text = cp if "\n" in cp else open(cp, encoding="utf-8").read()
        except OSError:
            problems.append(f"card unreadable: {cp}")
            continue
        label = cp if "\n" not in cp else "<card string>"
        for marker in _PLACEHOLDERS:
            if marker in text:
                problems.append(f"placeholder '{marker}' in {label}")

    # Reproducibility manifest must exist AND match the on-disk artifacts (no stale/tampered release).
    if check_manifest:
        try:
            from . import manifest as _manifest
            problems.extend(_manifest.verify(out_dir=out_dir, manifest_path=manifest_path))
        except Exception as e:
            problems.append(f"manifest check failed ({type(e).__name__}: {e})")
    return problems


def _enforce_preflight(out_dir: str = "data/out") -> None:
    problems = preflight(out_dir)
    if problems:
        raise SystemExit("[release] PREFLIGHT FAILED — refusing to publish:\n  - " + "\n  - ".join(problems))


def hf_upload(repo: str, path: str, repo_type: str) -> None:
    _enforce_preflight()  # never publish a broken release
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type=repo_type, exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=repo, repo_type=repo_type)
    print(f"[release] uploaded {path} -> https://huggingface.co/{repo}")


def kaggle_upload(
    slug: str, path: str, license_name: str, framework: str, variation: str = "default"
) -> None:
    _enforce_preflight()  # never publish a broken release
    import kagglehub

    # kagglehub model handle grammar is <owner>/<model>/<framework>/<variation-slug>. The 4th part
    # is the VARIATION, not the version — versions auto-increment on each upload. Passing "v1" here
    # mislabels the variation and pins every push to a phantom version; use a real variation slug.
    handle = f"{slug}/{framework}/{variation}"
    kagglehub.model_upload(
        handle,
        path,
        license_name=license_name,
        version_notes="AutoScientist SFT with hard-negative function-calling data",
    )
    print(f"[release] uploaded {path} -> kaggle model {handle}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("hf-dataset")
    p1.add_argument("--repo", required=True)
    p1.add_argument("--dir", required=True)

    p2 = sub.add_parser("hf-model")
    p2.add_argument("--repo", required=True)
    p2.add_argument("--dir", required=True)

    p3 = sub.add_parser("kaggle-model")
    p3.add_argument("--slug", required=True)
    p3.add_argument("--dir", required=True)
    p3.add_argument("--license", default="Apache 2.0")
    p3.add_argument("--framework", default="transformers")

    p4 = sub.add_parser("preflight", help="run release checks without uploading")
    p4.add_argument("--dir", default="data/out")

    args = ap.parse_args()
    if args.cmd == "preflight":
        problems = preflight(args.dir)
        if problems:
            print("[release] PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
            raise SystemExit(1)
        print("[release] preflight OK — all artifacts present, LICENSE found, no placeholders.")
    elif args.cmd == "hf-dataset":
        hf_upload(args.repo, args.dir, "dataset")
    elif args.cmd == "hf-model":
        hf_upload(args.repo, args.dir, "model")
    elif args.cmd == "kaggle-model":
        kaggle_upload(args.slug, args.dir, args.license, args.framework)


if __name__ == "__main__":
    main()
