"""Reproducibility manifest — provenance that makes the release auditable.

A judge (or a future you) should be able to answer "exactly what produced these numbers?" without
guessing. This records, for a build: SHA-256 of every dataset artifact, the seed, a hash of config.yaml,
the git commit, and the versions of the libraries that matter. `verify()` recomputes the artifact
hashes and flags drift — so the release preflight can refuse to publish a dataset whose files no longer
match the manifest (a stale or tampered release).

Pure stdlib. Offline. Deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

_ARTIFACTS = ["train.jsonl", "val.jsonl", "test.jsonl", "test_novel.jsonl", "pref.jsonl",
              "train_pc.jsonl", "stats.json"]
_LIBS = ["datasets", "huggingface_hub", "adaption", "numpy", "datasketch", "model2vec"]


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _lib_versions() -> Dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = version(lib)
        except PackageNotFoundError:
            out[lib] = "not-installed"
    return out


def generate(out_dir: str = "data/out", config_path: str = "config.yaml",
             seed: Optional[int] = None) -> Dict[str, Any]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts = {name: _sha256_file(os.path.join(out_dir, name)) for name in _ARTIFACTS}
    cfg_hash = _sha256_file(config_path)
    if seed is None and os.path.exists(config_path):
        try:
            import yaml
            seed = yaml.safe_load(open(config_path)).get("seed")
        except Exception:
            seed = None
    return {
        "seed": seed,
        "config_sha256": cfg_hash,
        "git_commit": _git_commit(repo_root),
        "python": sys.version.split()[0],
        "libraries": _lib_versions(),
        "artifacts_sha256": {k: v for k, v in artifacts.items() if v is not None},
    }


def verify(out_dir: str = "data/out", manifest_path: str = "results/manifest.json") -> List[str]:
    """Recompute artifact hashes and return a list of mismatches (empty = manifest matches on disk)."""
    if not os.path.exists(manifest_path):
        return [f"manifest missing: {manifest_path}"]
    m = json.load(open(manifest_path))
    problems = []
    for name, recorded in (m.get("artifacts_sha256") or {}).items():
        current = _sha256_file(os.path.join(out_dir, name))
        if current is None:
            problems.append(f"artifact vanished: {name}")
        elif current != recorded:
            problems.append(f"artifact changed since manifest: {name}")
    return problems


def write(out_dir: str = "data/out", config_path: str = "config.yaml",
          manifest_path: str = "results/manifest.json") -> Dict[str, Any]:
    m = generate(out_dir, config_path)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    json.dump(m, open(manifest_path, "w"), indent=2)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/out")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--manifest", default="results/manifest.json")
    ap.add_argument("--verify", action="store_true", help="verify on-disk artifacts against the manifest")
    args = ap.parse_args()
    if args.verify:
        problems = verify(args.out_dir, args.manifest)
        if problems:
            print("[manifest] MISMATCH:\n  - " + "\n  - ".join(problems))
            raise SystemExit(1)
        print("[manifest] OK — on-disk artifacts match the manifest.")
        return
    m = write(args.out_dir, args.config, args.manifest)
    print(f"[manifest] wrote {args.manifest} "
          f"({len(m['artifacts_sha256'])} artifacts, commit {str(m['git_commit'])[:8]}, seed {m['seed']})")


if __name__ == "__main__":
    main()
