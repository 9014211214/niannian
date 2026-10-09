"""The first release has one elder-companionship scene."""
from __future__ import annotations

from ..interfaces.scene import Scene
from .companion_elder import ElderCompanionScene

_SCENES: dict[str, Scene] = {"companion_elder": ElderCompanionScene()}

def get_scene(scene_id: str) -> Scene:
    if scene_id not in _SCENES:
        raise KeyError(f"unknown scene: {scene_id} (have {list(_SCENES)})")
    return _SCENES[scene_id]

def list_scenes() -> list[str]:
    return list(_SCENES)
