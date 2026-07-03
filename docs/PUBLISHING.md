# Publishing to PyPI (optional)

The repo is packaged (`pyproject.toml`) and **PyPI-ready** — both the wheel and sdist build cleanly and
pass `twine check`, and the import package is cleanly named **`autoscientist_toolcaller`** (no `src`
namespace-collision risk). Publishing itself needs a **PyPI account + API token** (an outward-facing
step), so it isn't automated here.

> For local use you don't need PyPI — `pip install -e .` from a clone, or
> `pip install git+https://github.com/ankit25bcs10610/adaption-ai-lab.git`, both work (see the README).

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
upload). After a successful upload, **`pip install autoscientist-toolcaller`** works for anyone, and the
`autoscientist` console command is installed.

## Notes

- The distribution name is `autoscientist-toolcaller`; the import name is `autoscientist_toolcaller`
  (`import autoscientist_toolcaller`, `python -m autoscientist_toolcaller.build_dataset`, or the
  `autoscientist` CLI).
- Core install pulls only `numpy` + `pyyaml`; the full training/eval pipeline still uses
  `pip install -r requirements.txt` (the pinned, reproducible set).
- Bump `version` in `pyproject.toml` before each new upload — PyPI rejects re-uploading an existing version.
