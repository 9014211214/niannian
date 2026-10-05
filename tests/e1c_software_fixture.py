"""SOFTWARE ONLY. Repetitive placeholders, NOT authored E1c holdout questions."""
from aria.bench.e1c_cases import CASE_CATEGORIES, EMOTIONS, RUBRIC_KEYS, VERSION


def fixture_payload():
    items = []
    for case_id, category in CASE_CATEGORIES.items():
        sessions = {n: [[f"软件夹具，会话{n}轮{turn}。", "平"] for turn in range(12)] for n in (1, 2)}
        items.append({
            "id": case_id, "category": category,
            "persona": {"name": "软件占位", "age": 72, "facts": ["软件字段占位，不是人物背景"]},
            "s1": sessions[1], "s2": sessions[2],
            "s3": {"warmup": [[f"软件夹具，末段轮{i}。", "平"] for i in range(7)],
                   "trap": ["软件夹具，机械观察位。", "平"],
                   "followups": [[f"软件夹具，后续{i}。", "平"] for i in range(2)]},
            "preference_sources": [{"session": n, "turn": t, "quote": sessions[n][t][0]}
                                   for n, t in ((1, 1), (2, 4))],
            "expected_preference": {"do": "软件标注占位，不是语义答案", "scope": "软件结构检查",
                                    "not_implies": ["不代表真实偏好"]},
            "rubric": {},
        })
    return {"version": VERSION, "provenance": "software_fixture_only", "emotion_codes": EMOTIONS,
            "defaults": {key: {"applicable": False} for key in sorted(RUBRIC_KEYS)}, "items": items}


def fixture_author():
    return {"author_id": "SOFTWARE_FIXTURE_NOT_INDEPENDENT_AUTHOR", "brief_sha256": "a" * 64,
            "context_mode": "software_fixture_only", "provided_materials": {"scope_and_format_brief": "a" * 64},
            "first_submission": True, "no_product_prompts_rules_or_e1b_questions": True,
            "no_outcome_filtering": True}
