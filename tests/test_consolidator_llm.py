"""LLMExtractor 深度抽取 —— 用假 LLM 确定性验证(不需要 key / 不联网)。"""
from __future__ import annotations

import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor, MemoryConsolidator
from aria.engine.memory.demo_personas import load_demo_persona
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.orchestrator import Engine
from aria.scenes.registry import get_scene


class FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload

    def complete(self, messages: list[dict]) -> str:
        return self.payload


def _ctx_with_convo() -> EngineContext:
    ctx = EngineContext(user_id="laochen_demo", scene_id="companion_elder")
    ctx.add_turn("今天那只橘猫又来了，我会喂它。", "嗯。")
    ctx.add_turn("跟你说个事，我老伴上个月忌日。", "在。")
    return ctx


def test_llm_extractor_parses_fenced_json_and_guards_privacy():
    payload = (
        "```json\n"
        '[{"summary":"他养了一只常来的橘猫，会喂它","keywords":["猫"],'
        '"referenceable":true,"evidence_quote":"今天那只橘猫又来了，我会喂它",'
        '"turn_index":0},\n'
        ' {"summary":"他提到老伴的忌日","keywords":["老伴"],'
        '"referenceable":true,"evidence_quote":"我老伴上个月忌日",'
        '"turn_index":1}]\n'
        "```"
    )
    res = LLMExtractor(FakeLLM(payload)).extract(_ctx_with_convo(), load_demo_persona("laochen_demo"))
    assert len(res.new_episodes) == 2
    cat = next(e for e in res.new_episodes if "橘猫" in e.summary)
    grief = next(e for e in res.new_episodes if "老伴" in e.summary)
    assert cat.referenceable is True
    # 隐私兜底:即便模型说 referenceable=true,敏感内容也被压成不可主动引用。
    assert grief.referenceable is False


@pytest.mark.parametrize("private_fact", (
    "我背着女儿借了十万元，你别告诉她",
    "我在银行还有一笔存款，账户尾号是6789",
    "我的身份证号码是11010119491231002X",
    "我的手机号是13800138000",
    "家里有件隐情，我们其实已经离婚了",
))
def test_private_episode_is_never_referenceable_even_if_model_requests_it(
    private_fact,
):
    ctx = EngineContext(user_id="private", scene_id="companion_elder")
    ctx.add_turn(private_fact + "。", "我会尊重你的边界。")
    payload = {
        "episodes": [{
            "summary": "模型错误标成可引用",
            "keywords": ["私事"],
            "referenceable": True,
            "evidence_quote": private_fact,
            "turn_index": 0,
        }],
        "relationships": [],
    }

    result = LLMExtractor(
        FakeLLM(json.dumps(payload, ensure_ascii=False))
    ).extract(ctx, load_demo_persona("laochen_demo"))

    assert len(result.new_episodes) == 1
    assert result.new_episodes[0].referenceable is False


def test_no_reference_episode_never_enters_online_prompt():
    class CaptureLLM:
        def __init__(self):
            self.messages = None

        def complete(self, messages):
            self.messages = messages
            return "我在。"

    private_fact = "我背着女儿借了十万元，你别告诉她"
    ctx = EngineContext(user_id="private_prompt", scene_id="companion_elder")
    ctx.add_turn(private_fact + "。", "我会尊重你的边界。")
    payload = (
        '{"episodes":[{"summary":"私事","keywords":["借钱"],'
        '"referenceable":true,"evidence_quote":"' + private_fact + '",'
        '"turn_index":0}],"relationships":[]}'
    )
    persona = load_demo_persona("laochen_demo")
    extracted = LLMExtractor(FakeLLM(payload)).extract(ctx, persona)
    persona.episodic_memory.extend(extracted.new_episodes)
    capture = CaptureLLM()
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=capture,
    )
    online_ctx = EngineContext(
        user_id="private_prompt", scene_id="companion_elder"
    )

    engine.process_turn(online_ctx, "前几天借钱那件事，我还在想。")

    assert capture.messages is not None
    prompt = "\n".join(str(row.get("content") or "") for row in capture.messages)
    assert private_fact not in prompt
    assert "十万元" not in prompt
    assert extracted.new_episodes[0] not in persona.referenceable_episodes()


def test_consolidator_with_llm_extractor_persists(tmp_path):
    payload = (
        '[{"summary":"他养了一只橘猫","keywords":["猫"],'
        '"referenceable":true,"evidence_quote":"今天那只橘猫又来了",'
        '"turn_index":0}]'
    )
    store = PersonaStore(base_dir=tmp_path)
    persona = load_demo_persona("laochen_demo")
    n0 = len(persona.episodic_memory)
    MemoryConsolidator(store, LLMExtractor(FakeLLM(payload))).consolidate(persona, _ctx_with_convo())
    assert len(persona.episodic_memory) == n0 + 1
    reloaded = store.load("laochen_demo")
    assert reloaded is not None and any("橘猫" in e.summary for e in reloaded.episodic_memory)


def test_llm_extractor_graceful_on_bad_json():
    extractor = LLMExtractor(FakeLLM("抱歉我无法输出 JSON"))
    res = extractor.extract(
        _ctx_with_convo(), load_demo_persona("laochen_demo")
    )
    assert res.new_episodes == []          # 失败不抛、不污染档案
    assert extractor.last_status == "failed"
    assert extractor.last_error_type == "invalid_json_response"


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"entries":[]}',
        '{"episodes":{},"relationships":[]}',
        '{"episodes":[],"relationships":{}}',
        '{"episodes":["not-an-entry"],"relationships":[]}',
    ],
)
def test_llm_extractor_rejects_valid_json_with_wrong_protocol(payload):
    extractor = LLMExtractor(FakeLLM(payload))

    result = extractor.extract(
        _ctx_with_convo(), load_demo_persona("laochen_demo")
    )

    assert result.new_episodes == []
    assert result.relationships == []
    assert extractor.last_status == "failed"
    assert extractor.last_error_type == "invalid_json_response"


def test_llm_extractor_accepts_protocol_after_harmless_preamble_and_fence():
    extractor = LLMExtractor(FakeLLM(
        '说明如下：\n```json\n{"episodes":[],"relationships":[]}\n```'
    ))

    result = extractor.extract(
        _ctx_with_convo(), load_demo_persona("laochen_demo")
    )

    assert result.new_episodes == []
    assert result.relationships == []
    assert extractor.last_status == "ok"
    assert extractor.last_error_type is None


def test_llm_extractor_rejects_ungrounded_memory_and_relationship():
    payload = """{
      "episodes": [{
        "summary": "他获得过全国冠军",
        "keywords": ["冠军"],
        "referenceable": true,
        "evidence_quote": "我获得过全国冠军",
        "turn_index": 0
      }],
      "relationships": [{
        "name": "小芳",
        "relation": "女儿",
        "facts": [{
          "value": "住在深圳",
          "evidence_quote": "小芳住在深圳",
          "turn_index": 0
        }],
        "referenceable": true
      }]
    }"""
    result = LLMExtractor(FakeLLM(payload)).extract(
        _ctx_with_convo(), load_demo_persona("laochen_demo")
    )
    assert result.new_episodes == []
    assert result.relationships == []


def test_relationship_context_has_provenance_and_never_infers_sharing(tmp_path):
    ctx = EngineContext(user_id="relationship", scene_id="companion_elder")
    ctx.add_turn("我女儿小芳住在深圳，每周日给我打电话。", "听起来你们一直有联系。")
    payload = """{
      "episodes": [],
      "relationships": [{
        "name": "小芳",
        "relation": "女儿",
        "facts": [{
          "value": "住在深圳，每周日打电话",
          "evidence_quote": "我女儿小芳住在深圳，每周日给我打电话",
          "turn_index": 0
        }],
        "referenceable": true,
        "consent_to_share": true
      }]
    }"""
    store = PersonaStore(base_dir=tmp_path)
    persona = load_demo_persona("laochen_demo")
    persona.relationship_context = []
    MemoryConsolidator(
        store, LLMExtractor(FakeLLM(payload))
    ).consolidate(persona, ctx)

    relation = persona.relationship_context[0]
    assert relation.name == "小芳" and relation.referenceable
    assert relation.consent_to_share is False
    assert relation.evidence[0].session_id
    assert relation.evidence[0].turn_indices == [0]


def test_cross_session_getting_to_know_a_new_user(tmp_path):
    """你自己的场景:新用户冷启动 → 第一次会话沉淀 → 第二次会话「想起」。

    这正是「我自己跟它对话,它会越来越了解我吗」的可执行答案(需 LLM 抽取出可引用记忆)。
    """
    from aria.engine.model.persona import PersonaProfile
    from aria.scenes.registry import get_scene

    store = PersonaStore(base_dir=tmp_path)
    payload = (
        '[{"summary":"他在备战人生第一个马拉松",'
        '"keywords":["马拉松","跑步"],"referenceable":true,'
        '"evidence_quote":"我在备战人生第一个马拉松，最近在练跑步",'
        '"turn_index":0}]'
    )

    # 会话 1:全新用户(冷启动,没有任何档案),聊到跑步
    persona = PersonaProfile(user_id="tester")
    eng1 = Engine(get_scene("companion_elder"), persona=persona, store=store,
                  extractor=LLMExtractor(FakeLLM(payload)))
    ctx1 = EngineContext(user_id="tester", scene_id="companion_elder")
    eng1.process_turn(ctx1, "我在备战人生第一个马拉松，最近在练跑步。")
    eng1.finalize_session(ctx1)                      # 沉淀 → 落盘

    # 会话 2:全新引擎,从 store 冷启动加载这个人的活档案
    persona2 = store.load("tester")
    assert persona2 is not None
    eng2 = Engine(get_scene("companion_elder"), persona=persona2)
    ctx2 = EngineContext(user_id="tester", scene_id="companion_elder")
    eng2.process_turn(ctx2, "今天又去跑步了。")
    recalled = ctx2.working_memory["recalled_episodes"]
    assert any("马拉松" in r for r in recalled)      # 第二次会话「想起」第一次说的事
