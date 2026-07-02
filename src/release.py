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


def hf_upload(repo: str, path: str, repo_type: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo, repo_type=repo_type, exist_ok=True)
    api.upload_folder(folder_path=path, repo_id=repo, repo_type=repo_type)
    print(f"[release] uploaded {path} -> https://huggingface.co/{repo}")


def kaggle_upload(
    slug: str, path: str, license_name: str, framework: str, variation: str = "default"
) -> None:
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

    args = ap.parse_args()
    if args.cmd == "hf-dataset":
        hf_upload(args.repo, args.dir, "dataset")
    elif args.cmd == "hf-model":
        hf_upload(args.repo, args.dir, "model")
    elif args.cmd == "kaggle-model":
        kaggle_upload(args.slug, args.dir, args.license, args.framework)


if __name__ == "__main__":
    main()
