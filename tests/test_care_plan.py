"""结构化护理数据(care_plan):用药 / 紧急联系人是「关于这个人的知识」。"""
from __future__ import annotations

from aria.engine.memory.demo_personas import load_demo_persona


def test_care_plan_loaded_from_persona():
    p = load_demo_persona("laochen_demo")
    assert p.care_plan is not None
    meds = p.care_plan.medications
    assert meds and meds[0].name == "降压药" and meds[0].schedule == "早餐后"
    # 紧急联系人默认未同意被通知(隐私红线)。
    contact = p.care_plan.emergency_contacts[0]
    assert contact.relation == "女儿" and contact.consent_to_notify is False
