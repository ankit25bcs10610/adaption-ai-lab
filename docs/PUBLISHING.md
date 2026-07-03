# Publishing to PyPI (optional)

The repo is packaged (`pyproject.toml`) and **PyPI-ready** — both the wheel and sdist build cleanly and
pass `twine check`. Publishing itself needs a **PyPI account + API token** (an outward-facing step), so it
isn't automated here.

> **Before you publish publicly, read the caveat at the bottom.** For local use you don't need PyPI at
> all — `pip install -e .` from a clone is the recommended flow (see the README Quickstart).

## Steps

```bash
# 1. Install build tooling
python -m pip install --upgrade build twine

# 2. Build sdist + wheel into ./dist
python -m build

# 3. Validate metadata
python -m twine check dist/*        # expect: PASSED for both

# 4. (recommended) dry-run to TestPyPI first
python -m twine upload -r testpypi dist/*
pip install -i https://test.pypi.org/simple/ autoscientist-toolcaller

# 5. Real upload (use an API token: username __token__, password = pypi-… token)
TWINE_USERNAME=__token__ TWINE_PASSWORD=pypi-XXXXXXXX python -m twine upload dist/*
```

Create the token at <https://pypi.org/manage/account/token/> (scope it to this project after the first
upload). After a successful upload, `pip install autoscientist-toolcaller` works for anyone, and the
`autoscientist` console command is installed.

## ⚠️ Caveat before a *public* release

The import package is literally named **`src`** (all code does `from src....`). That's fine for a local
editable install, but publishing a public wheel means anyone who installs it gets a top-level `src`
package on their path — a **namespace-collision risk** with other projects. Options:

- **Local only (recommended for the hackathon):** don't publish to PyPI; use `pip install -e .` from a clone.
- **Publish anyway:** acceptable for a personal/demo package, but expect the `src` name to be flagged.
- **Rename first (cleanest for a real public package):** rename `src/` → `autoscientist_toolcaller/`, update
  imports (`from src.` → `from autoscientist_toolcaller.`) across `src/`, `tests/`, `src/cli.py`, and the
  `[tool.setuptools] packages` list, then publish. This is a mechanical but repo-wide change — ask and I'll do it.
