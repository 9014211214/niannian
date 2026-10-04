"""可选:把仓库根 `.env` 读进 os.environ(stdlib,无第三方依赖)。

仅用于本地开发便利(examples 调用)。`.env` 已被 .gitignore 忽略,绝不入库。
不覆盖已存在的环境变量(真实环境变量优先)。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _source_checkout_root() -> Path | None:
    """Return the repository root only when this module is in a checkout."""
    root = Path(__file__).resolve().parents[1]
    if (root / "pyproject.toml").is_file() and (root / "aria").is_dir():
        return root
    return None


def _candidate_env_files(path: Optional[str] = None) -> list[Path]:
    if path is not None:
        return [Path(path).expanduser()]
    candidates = [Path.cwd() / ".env"]
    checkout = _source_checkout_root()
    if checkout is not None:
        legacy = checkout / ".env"
        if legacy != candidates[0]:
            candidates.append(legacy)
    return candidates


def load_local_env(path: Optional[str] = None) -> None:
    """Load the first applicable local ``.env`` without overriding the OS.

    Explicit paths take precedence. Otherwise the current working directory
    is checked first, followed by the historical repository-root location
    when running from a source checkout. Installed wheels never look for
    configuration inside ``site-packages``.
    """
    p = next((item for item in _candidate_env_files(path) if item.is_file()), None)
    if p is None:
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
