"""透明度：老人友好摘要与开发者技术审计分层。"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.memory.transparency import (
    render_profile,
    render_profile_summary,
)


def test_render_profile_shows_everything_including_private():
    persona = load_demo_persona("laochen_demo")
    ctx = EngineContext(user_id="laochen_demo", scene_id="s")
    ctx.formulation.hypotheses = ["依恋倾向:avoidant"]
    out = render_profile(persona, ctx)
    assert "DISC" in out and "依恋" in out and "回避型" in out
    assert "工作假设" in out
    # 对本人全透明:连「只内化不可提」的私密记忆也如实展示(这是他自己的事)
    assert "只内化" in out
    assert f"共 {len(persona.episodic_memory)} 条" in out
    assert "开发者技术审计视图" in out
    assert ":forget" not in out
    assert "删除本机数据" in out


def test_render_profile_fresh_user_graceful():
    from aria.engine.model.persona import PersonaProfile
    out = render_profile(PersonaProfile(user_id="new"))
    assert "尚无判断" in out and "共 0 条" in out


def test_elder_profile_summary_is_correctable_and_non_stigmatizing():
    persona = load_demo_persona("laochen_demo")
    out = render_profile_summary(persona)
    assert "可以纠正" in out
    assert "授权配置不代表已经发送" in out
    assert "删除本机数据不能撤回" in out
    for technical_label in ("DISC", "Big Five", "神经质", "依恋", "tier"):
        assert technical_label not in out
