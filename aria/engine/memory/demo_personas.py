"""Demo persona 加载器 —— 从 aria/assets/personas/{id}.yaml 读预设档案。

demo 用预写 YAML;生产用 PersonaStore 的活档案(JSON)。两者 schema 一致,
加载路径可替换。这里只负责「冷启动 / demo」那一侧。
"""
from __future__ import annotations

from pathlib import Path

import yaml

from ..model.persona import PersonaProfile

_PERSONA_DIR = Path(__file__).resolve().parents[2] / "assets" / "personas"


def load_demo_persona(persona_id: str) -> PersonaProfile:
    path = _PERSONA_DIR / f"{persona_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"demo persona not found: {persona_id} ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PersonaProfile(**data)


def list_demo_personas() -> list[str]:
    if not _PERSONA_DIR.exists():
        return []
    return sorted(p.stem for p in _PERSONA_DIR.glob("*.yaml"))
