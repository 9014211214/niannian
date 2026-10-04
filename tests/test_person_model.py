"""人物档案 schema + 活档案存储的移植验证。"""
from __future__ import annotations

from aria.engine.memory import PersonaStore
from aria.engine.memory.demo_personas import list_demo_personas, load_demo_persona
from aria.engine.model import PersonaProfile


def test_load_elder_demo_persona():
    p = load_demo_persona("laochen_demo")
    assert isinstance(p, PersonaProfile)
    assert p.attachment_style == "avoidant"
    assert p.disc.steadiness >= 0.6 and p.big_five.extraversion <= 0.3
    # 口是心非的语言习惯应被标成「内化不可挑明」。
    habit = [e for e in p.episodic_memory if e.type == "speech_habit"]
    assert habit and all(not e.referenceable for e in habit)
    # 老伴 / 未接电话属于私密,Aria 不能主动引用。
    refs = {e.id for e in p.referenceable_episodes()}
    assert "ep_c_004" not in refs and "ep_c_003" not in refs
    # 那只猫是安全开口钩子,可引用。
    assert "ep_c_001" in refs
    assert p.profile_origin == "synthetic_demo"
    assert p.understanding_stage() == "emerging"
    assert p.relationship_context


def test_persona_store_roundtrip(tmp_path):
    store = PersonaStore(base_dir=tmp_path)
    p = load_demo_persona("laochen_demo")
    assert store.load(p.user_id) is None          # 冷启动:还没有活档案
    store.save(p)
    assert store.exists(p.user_id)
    again = store.load(p.user_id)
    assert again is not None and again.user_id == p.user_id
    assert len(again.episodic_memory) == len(p.episodic_memory)
    assert store.delete(p.user_id) is True         # PIPL:可删除
    assert store.load(p.user_id) is None


def test_demo_personas_listed():
    assert "laochen_demo" in list_demo_personas()


def test_true_cold_start_does_not_treat_neutral_defaults_as_findings():
    p = PersonaProfile(user_id="new")
    assert p.profile_origin == "blank"
    assert p.understanding_stage() == "cold_start"
    assert p.person_model_state.sessions_completed == 0
    assert p.person_hypotheses == []
    assert p.episodic_memory == []
