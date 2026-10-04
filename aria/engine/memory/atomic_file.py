"""Retry only transient Windows file replacement, never the surrounding work."""
from __future__ import annotations

from pathlib import Path
import sys
import time


def replace_with_retry(source: Path, target: Path) -> None:
    source, target = Path(source), Path(target)
    if source.parent.resolve() != target.parent.resolve():
        raise ValueError("atomic replacement requires the same directory")
    for attempt in range(5):
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            if (sys.platform != "win32" or getattr(exc, "winerror", None) not in {5, 32, 33}
                    or attempt == 4):
                raise
            time.sleep(0.02 * (attempt + 1))
