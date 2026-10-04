"""证明架构两件事:
  1. 同一台引擎,换三张皮都跑通(one engine, many skins)。
  2. engine 包不依赖 scenes 包(可插拔的边界守卫)。
"""
from __future__ import annotations

import ast
import pathlib

import aria.engine
from aria.engine.context import EngineContext
from aria.engine.orchestrator import Engine
from aria.scenes.registry import get_scene, list_scenes


def test_one_engine_many_scenes():
    for scene_id in list_scenes():
        scene = get_scene(scene_id)
        engine = Engine(scene)
        ctx = EngineContext(user_id="u1", scene_id=scene_id)
        out = engine.process_turn(ctx, "我挺好的")
        assert isinstance(out, str) and out
        assert len(ctx.turns) == 1


def _imports_scenes(tree: ast.AST) -> bool:
    """只看真实 import 语句(AST),不看注释 / 文档串。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any("scenes" in a.name.split(".") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and "scenes" in node.module.split("."):
                return True
    return False


def test_engine_does_not_import_scenes():
    """边界守卫:engine 源码不得真正依赖 scenes(允许文档里提到它)。"""
    root = pathlib.Path(aria.engine.__file__).parent
    for p in root.rglob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        assert not _imports_scenes(tree), f"{p} 真正 import 了 scenes"
