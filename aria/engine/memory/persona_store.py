"""PersonaProfile 持久化存储 —— 把「读 YAML 的死档案」变成「会自己更新的活档案」。

(从旧 repo 移植,接口不变。)

设计:
- 一个用户一个 JSON 文件(data/user_memory/{user_id}.json)。
  「删除某个用户的全部心理画像」= 删一个文件,天然满足 PIPL 的可删除要求。
- schema 完全复用 model.persona.PersonaProfile —— demo 的 YAML 和生产的 JSON
  是同一个结构,加载路径可替换。
- 纯文件实现,接口设计为可替换(以后换 DB / 对象存储,实现同名方法即可)。

这是「复利」的物理底座:对话沉淀出的新记忆,最终写回这里。
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from ..model.persona import PersonaProfile
from .crypto import decrypt_if_needed, encrypt_if_keyed

logger = logging.getLogger(__name__)

# user_id 落盘前的安全校验:只允许字母/数字/下划线/连字符,避免路径穿越。
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _source_checkout_memory_dir() -> Path | None:
    """Preserve the historical repo-local directory for source checkouts."""
    root = Path(__file__).resolve().parents[3]
    if (root / "pyproject.toml").is_file() and (root / "aria").is_dir():
        return root / "data" / "user_memory"
    return None


def _platform_user_data_dir() -> Path:
    """Return a user-writable default for installed distributions."""
    home = Path.home()
    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
        root = Path(base) if base else home / "AppData" / "Local"
        return root / "Aria" / "user_memory"
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Aria" / "user_memory"
    base = os.getenv("XDG_DATA_HOME")
    root = Path(base) if base else home / ".local" / "share"
    return root / "aria" / "user_memory"


def _resolve_dir(base_dir: Optional[Path]) -> Path:
    if base_dir is not None:
        return Path(base_dir)          # 容忍传入 str(否则 str / str 会崩)
    env = os.getenv("ARIA_USER_MEMORY_DIR")
    if env:
        return Path(env).expanduser()
    legacy = _source_checkout_memory_dir()
    return legacy if legacy is not None else _platform_user_data_dir()


class PersonaStore:
    """PersonaProfile 的持久化读写,键为 user_id。"""

    def __init__(self, base_dir: Optional[Path] = None):
        self._dir = _resolve_dir(base_dir)

    def _path(self, user_id: str) -> Path:
        if not user_id or not _SAFE_ID.match(user_id):
            raise ValueError(
                f"unsafe user_id for filesystem store: {user_id!r} "
                "(allowed: letters/digits/_/-)"
            )
        return self._dir / f"{user_id}.json"

    def exists(self, user_id: str) -> bool:
        return self._path(user_id).exists()

    def load(self, user_id: str) -> Optional[PersonaProfile]:
        """加载用户的活档案;不存在返回 None(交给调用方决定冷启动 / 回退 YAML)。"""
        path = self._path(user_id)
        if not path.exists():
            return None
        # 加密档案缺 key / key 错 → decrypt_if_needed 直接抛(fail loud):
        # 绝不静默当「新用户」,否则空档案会覆盖真人的心理画像。
        text = decrypt_if_needed(
            path.read_text(encoding="utf-8"), self._dir, who=user_id
        )
        try:
            data = json.loads(text)
            return PersonaProfile(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            # Validation messages may echo private field values. Fail closed
            # and log only identifiers plus the exception class.
            logger.error(
                "corrupt persona file for user_id=%s (%s)",
                user_id,
                type(exc).__name__,
            )
            raise RuntimeError(
                f"persona profile is corrupt for user_id={user_id}"
            ) from None

    def list_user_ids(self) -> List[str]:
        if not self._dir.exists():
            return []
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def load_all(self) -> List[PersonaProfile]:
        """加载全部活档案；任一损坏都会失败并要求人工恢复。"""
        out: List[PersonaProfile] = []
        for uid in self.list_user_ids():
            p = self.load(uid)
            if p is not None:
                out.append(p)
        return out

    def save(self, persona: PersonaProfile) -> None:
        """原子写入:先写临时文件再 rename,避免写一半进程挂掉留下损坏档案。"""
        path = self._path(persona.user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        # 设了 ARIA_PROFILE_KEY 即静态加密(P5-#20);明文旧档案在此平滑迁移为密文。
        tmp.write_text(
            encrypt_if_keyed(
                persona.model_dump_json(indent=2, exclude_none=False), self._dir
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.info(
            f"persona saved: {persona.user_id} "
            f"(episodes={len(persona.episodic_memory)})"
        )

    def delete(self, user_id: str) -> bool:
        """删除该用户的全部持久心理画像(PIPL:用户有权要求删除)。返回是否真的删了文件。"""
        path = self._path(user_id)
        removed = False
        for candidate in (path, path.with_suffix(".json.tmp")):
            if candidate.exists():
                candidate.unlink()
                removed = True
        if removed:
            logger.info(f"persona deleted (user request / PIPL): {user_id}")
            return True
        return False
