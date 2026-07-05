"""Real, safe tools for the agent runtime — sandboxed filesystem + HTTP + time.

Turns the agent from a toy (calculator) into one that does real work, while staying safe:
  * filesystem tools are confined to a **sandbox root** — path traversal (`../`, absolute paths that
    escape) is rejected before any I/O;
  * there is **no shell / eval / arbitrary-code** tool;
  * HTTP is read-only GET with a size cap.

Register into a `ToolRegistry` and hand it to `run_agent`. All FS tools are offline-testable.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from .agent import _FINISH_SCHEMA, ToolRegistry

_MAX_READ = 10_000  # chars — keep observations bounded


def _resolve(root: str, path: str) -> str:
    """Resolve `path` under `root`, rejecting anything that escapes the sandbox."""
    root_r = os.path.realpath(root)
    full = os.path.realpath(os.path.join(root_r, path or "."))
    if os.path.commonpath([root_r, full]) != root_r:
        raise ValueError("path escapes the sandbox")
    return full


def _str(desc: str) -> Dict[str, Any]:
    return {"type": "string", "description": desc}


def sandbox_fs_registry(root: str, allow_write: bool = False) -> ToolRegistry:
    """A ToolRegistry of filesystem tools confined to `root`. read/list always; write only if allowed."""
    os.makedirs(root, exist_ok=True)
    r = ToolRegistry()

    def read_file(path: str) -> str:
        with open(_resolve(root, path), encoding="utf-8") as f:
            return f.read()[:_MAX_READ]

    def list_dir(path: str = ".") -> List[str]:
        return sorted(os.listdir(_resolve(root, path)))

    r.register({"name": "read_file", "description": "Read a UTF-8 text file (within the sandbox).",
                "parameters": {"type": "object", "properties": {"path": _str("relative file path")},
                               "required": ["path"]}}, read_file)
    r.register({"name": "list_dir", "description": "List files in a directory (within the sandbox).",
                "parameters": {"type": "object", "properties": {"path": _str("relative dir path")},
                               "required": []}}, list_dir)
    if allow_write:
        def write_file(path: str, content: str) -> str:
            full = _resolve(root, path)
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return f"wrote {len(content)} chars to {path}"

        r.register({"name": "write_file", "description": "Write a UTF-8 text file (within the sandbox).",
                    "parameters": {"type": "object", "properties": {"path": _str("relative file path"),
                                   "content": _str("file contents")}, "required": ["path", "content"]}},
                   write_file)
    r.register(_FINISH_SCHEMA, lambda answer="": answer)
    return r


def register_http(registry: ToolRegistry, timeout: float = 10.0) -> ToolRegistry:
    """Add a read-only HTTP GET tool (lazy stdlib urllib; real network at call time)."""
    def http_get(url: str) -> str:
        import urllib.request
        if not str(url).lower().startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (scheme checked above)
            return resp.read(_MAX_READ).decode("utf-8", "replace")

    registry.register({"name": "http_get", "description": "Fetch the text of a URL (read-only GET).",
                       "parameters": {"type": "object", "properties": {"url": _str("http(s) URL")},
                                      "required": ["url"]}}, http_get)
    return registry
