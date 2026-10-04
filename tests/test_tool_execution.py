"""P3:让「懂」接到「做」—— 动作解析 / 工具真执行 / 回执渲染 / 结构化 KB。"""
from __future__ import annotations

from aria.engine.context import EngineContext
from aria.engine.model.persona import (
    CarePlan,
    EmergencyContact,
    Medication,
    PersonaProfile,
)
from aria.engine.orchestrator import Engine
from aria.engine.policy.actions import (
    current_pending_action,
    has_current_action_intent,
    parse_action,
    prepare_action,
    stage_pending_action,
)
from aria.engine.policy.prompt import build_messages
from aria.interfaces.tools import ToolResult
from aria.scenes.companion_elder.knowledge import ElderKnowledge
from aria.scenes.registry import get_scene


class _FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return self.payload


class _WorkingReminder:
    name = "medication_reminder"
    description = "测试用真实提醒适配器"
    parameters = {"medication": "药物", "when": "时间"}

    def invoke(self, **kwargs):
        return ToolResult(
            ok=True,
            data={**kwargs, "persisted": True, "executed": True},
        )


class _UnavailableReminder(_WorkingReminder):
    def invoke(self, **kwargs):
        return ToolResult(ok=False, data={"executed": False, "persisted": False})

class _LocalNotifyFixture:
    name = "notify_family"
    description = "Local permission-boundary fixture; never sends"
    parameters = {"contact": "named contact"}

    def invoke(self, **kwargs):
        return ToolResult(ok=False, data={"executed": False})


def test_tools_are_advertised_only_for_current_explicit_action_intent():
    assert not has_current_action_intent("这两天一起身眼前就发黑。")
    assert not has_current_action_intent("今天楼下那只猫又来了。")
    assert has_current_action_intent("每天早餐后提醒我吃降压药。")
    assert has_current_action_intent("请通知女儿我今天不回去。")
    assert not has_current_action_intent("可以执行。")
    assert not has_current_action_intent("不要联系女儿。")
    assert not has_current_action_intent("昨天联系了女儿。")
    assert not has_current_action_intent("如果需要就联系女儿。")
    assert not has_current_action_intent("别提醒我吃药。")
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    pending = stage_pending_action(
        ctx,
        "medication_reminder",
        {"medication": "降压药", "when": "早餐后"},
    )
    assert has_current_action_intent("可以执行。", pending)
    assert not has_current_action_intent("不可以执行。", pending)
    assert not has_current_action_intent("别按这个来。", pending)
    assert not has_current_action_intent(
        "就按这个来，不，还是别按这个来。", pending
    )
    assert not has_current_action_intent("告诉我王护理员今天几点来？")


# ── 动作解析:稳健剥离,坏 JSON 不污染回应 ───────────────────
def test_parse_action_extracts_and_strips():
    text, action = parse_action(
        '好,我记下了。\n<action>{"tool":"medication_reminder","args":{"medication":"降压药"}}</action>'
    )
    assert text == "好,我记下了。"
    assert action == {"tool": "medication_reminder", "args": {"medication": "降压药"}}


def test_parse_action_bad_json_strips_tag_only():
    text, action = parse_action("吃过了吗?\n<action>{坏掉的}</action>")
    assert action is None and "<action>" not in text and "吃过了吗?" in text


def test_parse_action_plain_text_untouched():
    text, action = parse_action("今天天气不错。")
    assert text == "今天天气不错。" and action is None


def test_parse_action_executes_only_first_and_hides_all_duplicate_tags():
    text, action = parse_action(
        '好。<action>{"tool":"medication_reminder","args":{}}</action>'
        '别的。<action>{"tool":"notify_family","args":{}}</action>'
    )
    assert text == "好。别的。"
    assert action == {"tool": "medication_reminder", "args": {}}
    assert "<action>" not in text


# ── 端到端:LLM 出动作 → 引擎执行 → 回执进下一轮 prompt ──────
def test_engine_executes_action_and_renders_receipt():
    llm = _FakeLLM(
        '行,到点我说一声。\n<action>{"tool":"medication_reminder","args":{"medication":"降压药","when":"明早"}}</action>'
    )
    scene = get_scene("companion_elder")
    engine = Engine(
        scene, llm_client=llm, tools=[_WorkingReminder()]
    )
    ctx = EngineContext(user_id="u", scene_id=scene.id)
    out = engine.process_turn(ctx, "明早提醒我吃降压药。")
    assert "<action>" not in out and out.startswith("行,")   # 用户看不到标签
    tr = ctx.working_memory["last_tool_result"]
    assert tr["tool"] == "medication_reminder" and tr["ok"]
    # 下一轮 prompt 渲染回执
    system = build_messages(scene, None, ctx, "嗯。")[0]["content"]
    assert "外部动作回执" in system and "medication_reminder" in system
    # 下一轮(无新动作)跑完,回执被消费掉,不无限残留
    llm.payload = "好,睡前我再提一句。"
    engine.process_turn(ctx, "嗯。")
    assert "last_tool_result" not in ctx.working_memory


def test_unknown_tool_fails_gracefully():
    llm = _FakeLLM('好。\n<action>{"tool":"no_such_tool","args":{}}</action>')
    scene = get_scene("companion_elder")
    engine = Engine(scene, llm_client=llm)
    ctx = EngineContext(user_id="u", scene_id=scene.id)
    engine.process_turn(ctx, "随便")
    assert ctx.working_memory["last_tool_result"]["ok"] is False


def test_action_requires_explicit_current_consent_and_fills_care_plan_args():
    persona = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(medications=[
            Medication(name="降压药", schedule="早餐后")
        ]),
    )
    llm = _FakeLLM(
        '好。\n<action>{"tool":"medication_reminder","args":{}}</action>'
    )
    engine = Engine(
        get_scene("companion_elder"),
        persona=persona,
        llm_client=llm,
        tools=[_WorkingReminder()],
    )
    engine._tools.register(_WorkingReminder())
    denied_ctx = EngineContext(user_id="u", scene_id="companion_elder")
    denied = engine.process_turn(denied_ctx, "今天天气不错。")
    assert "还没有执行" in denied
    assert denied_ctx.working_memory["last_tool_result"]["ok"] is False
    assert denied_ctx.reasoning.delivered_interventions == []
    assert denied_ctx.working_memory["intervention_delivery"]["reason"] == (
        "final_response_overridden"
    )
    llm.payload = "我听见了。"
    engine.process_turn(denied_ctx, "其实我有点失望。")
    assert denied_ctx.reasoning.outcomes == []

    allowed_ctx = EngineContext(user_id="u2", scene_id="companion_elder")
    llm.payload = (
        '好。\n<action>{"tool":"medication_reminder","args":{}}</action>'
    )
    engine.process_turn(allowed_ctx, "明早提醒我吃药。")
    data = allowed_ctx.working_memory["last_tool_result"]["data"]
    assert data["medication"] == "降压药"
    assert data["when"] == "早餐后"

    # 同一个模型动作重复发射时，确定性层拒绝重复提交。
    repeated = engine.process_turn(allowed_ctx, "明早还是提醒我吃药。")
    assert "还没有执行" in repeated


def test_notify_family_accepts_only_the_model_contact_with_prior_consent():
    persona = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(emergency_contacts=[
            EmergencyContact(
                name="王护理员",
                consent_to_notify=True,
                notification_consent_scope="care_and_safety_notifications",
                notification_consent_source="institution_care_plan",
                notification_consented_at="2026-07-01T09:00:00+08:00",
            ),
            EmergencyContact(name="陈小芳", consent_to_notify=False),
        ]),
    )
    approved, reason = prepare_action(
        {"tool": "notify_family", "args": {"contact": "王护理员"}},
        "请联系王护理员。",
        persona,
    )
    assert not reason and approved is not None
    assert approved["args"]["contact"] == "王护理员"
    assert approved["args"] == {
        "contact": "王护理员",
        "reason_code": "user_requested_contact",
    }

    for contact in ("陈小芳", "李阿姨"):
        denied, reason = prepare_action(
            {"tool": "notify_family", "args": {"contact": contact}},
            f"请联系{contact}。",
            persona,
        )
        assert denied is None
        assert "有效授权名单" in reason

    missing, reason = prepare_action(
        {"tool": "notify_family", "args": {}},
        "请联系王护理员。",
        persona,
    )
    assert missing is None and "不能自动替换" in reason

    mismatched, reason = prepare_action(
        {"tool": "notify_family", "args": {"contact": "王护理员"}},
        "请联系陈小芳。",
        persona,
    )
    assert mismatched is None and "当轮没有明确" in reason


def test_notify_family_rejects_unauditable_consent_fields():
    base = {
        "name": "王护理员",
        "consent_to_notify": True,
        "notification_consent_scope": "care_and_safety_notifications",
        "notification_consent_source": "institution_care_plan",
        "notification_consented_at": "2026-07-01T09:00:00+08:00",
    }
    missing_fields = (
        "notification_consent_scope",
        "notification_consent_source",
        "notification_consented_at",
    )
    for field in missing_fields:
        values = {**base, field: None}
        persona = PersonaProfile(
            user_id="u",
            care_plan=CarePlan(emergency_contacts=[EmergencyContact(**values)]),
        )
        denied, reason = prepare_action(
            {"tool": "notify_family", "args": {"contact": "王护理员"}},
            "请联系王护理员。",
            persona,
        )
        assert denied is None and "范围、来源和时间" in reason


def test_generic_help_does_not_authorize_an_unrelated_tool():
    denied, reason = prepare_action(
        {"tool": "medication_reminder", "args": {}},
        "你帮我看看今天天气。",
        None,
    )
    assert denied is None and "用药提醒" in reason

    approved, reason = prepare_action(
        {
            "tool": "medication_reminder",
            "args": {"medication": "降压药", "when": "明早"},
        },
        "明早提醒我吃降压药。",
        None,
    )
    assert not reason and approved is not None

    denied, reason = prepare_action(
        {"tool": "future_device", "args": {}},
        "你帮我。",
        None,
    )
    assert denied is None and "工具相关" in reason


def test_prompt_advertises_action_syntax_with_descriptions():
    scene = get_scene("companion_elder")
    engine = Engine(scene, llm_client=_FakeLLM("x"), tools=[_WorkingReminder()])
    ctx = EngineContext(user_id="u", scene_id=scene.id)
    system = build_messages(scene, None, ctx, "x",
                            available_tools=engine._tools.describe())[0]["content"]
    assert "<action>" in system and "medication_reminder:" in system  # 名字+说明都在


def test_removed_default_reminder_cannot_claim_success_in_same_turn():
    llm = _FakeLLM(
        '好，六点半我叫你。'
        '\n<action>{"tool":"medication_reminder",'
        '"args":{"medication":"降压药","when":"六点半"}}</action>'
    )
    engine = Engine(get_scene("companion_elder"), llm_client=llm)
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    output = engine.process_turn(ctx, "六点半提醒我吃降压药。")
    receipt = ctx.working_memory["last_tool_result"]
    assert receipt["ok"] is False
    assert receipt["data"].get("executed", False) is False
    assert "六点半我叫你" not in output
    assert engine._tools.names() == []
    prompt = build_messages(
        get_scene("companion_elder"), None, ctx, "怎么没叫我？"
    )[0]["content"]
    assert "没有成功" in prompt and "不能装作执行过" in prompt


def test_generic_confirmation_requires_exact_server_pending_action():
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    pending = stage_pending_action(
        ctx,
        "medication_reminder",
        {"medication": "降压药", "when": "早餐后"},
        action_id="pending-1",
    )
    approved, reason = prepare_action(
        {
            "tool": "medication_reminder",
            "args": {},
            "pending_action_id": "pending-1",
        },
        "可以执行。",
        None,
        pending_action=pending,
    )
    assert not reason
    assert approved == {
        "tool": "medication_reminder",
        "args": {"medication": "降压药", "when": "早餐后"},
        "pending_action_id": "pending-1",
    }

    attempts = (
        {
            "tool": "medication_reminder",
            "args": {},
            "pending_action_id": "invented",
        },
        {
            "tool": "notify_family",
            "args": {},
            "pending_action_id": "pending-1",
        },
        {
            "tool": "medication_reminder",
            "args": {"medication": "胰岛素", "when": "睡前"},
            "pending_action_id": "pending-1",
        },
    )
    for attempt in attempts:
        denied, denial = prepare_action(
            attempt,
            "可以执行。",
            None,
            pending_action=pending,
        )
        assert denied is None and denial

    denied, denial = prepare_action(
        {
            "tool": "medication_reminder",
            "args": {"medication": "胰岛素", "when": "睡前"},
        },
        "可以执行。",
        None,
    )
    assert denied is None and "没有可核对" in denial


def test_engine_pending_confirmation_is_narrow_and_consumed_only_on_success():
    llm = _FakeLLM("先确认一下。")
    engine = Engine(
        get_scene("companion_elder"),
        llm_client=llm,
        tools=[_WorkingReminder()],
    )
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    engine.stage_pending_action(
        ctx,
        "medication_reminder",
        {"medication": "降压药", "when": "早餐后"},
        action_id="server-pending",
    )
    llm.payload = (
        '好。\n<action>{"tool":"medication_reminder",'
        '"args":{},"pending_action_id":"server-pending"}</action>'
    )
    output = engine.process_turn(ctx, "就按这个来。")
    assert output == "好。"
    assert ctx.working_memory["last_tool_result"]["ok"] is True
    assert current_pending_action(ctx) is None
    assert ctx.working_memory["_advertised_tools_for_turn"] == [
        item for item in engine._tools.describe()
        if item.startswith("medication_reminder:")
    ]

    # 没有 pending 时，相同泛确认不能靠模型补出药物和时间。
    llm.payload = (
        '好。\n<action>{"tool":"medication_reminder",'
        '"args":{"medication":"胰岛素","when":"睡前"}}</action>'
    )
    denied = engine.process_turn(ctx, "可以执行。")
    assert "还没有执行" in denied
    assert ctx.working_memory["last_tool_result"]["ok"] is False
    assert ctx.working_memory["_advertised_tools_for_turn"] == []


def test_failed_pending_action_remains_available_for_retry():
    engine = Engine(
        get_scene("companion_elder"),
        tools=[_UnavailableReminder()],
        llm_client=_FakeLLM(
            '好。<action>{"tool":"medication_reminder","args":{},'
            '"pending_action_id":"retry-me"}</action>'
        ),
    )
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    engine.stage_pending_action(
        ctx,
        "medication_reminder",
        {"medication": "降压药", "when": "早餐后"},
        action_id="retry-me",
    )
    output = engine.process_turn(ctx, "确认执行。")
    assert "没有执行" in output
    assert ctx.working_memory["last_tool_result"]["ok"] is False
    assert ctx.working_memory["last_tool_result"]["data"]["executed"] is False
    assert current_pending_action(ctx)["id"] == "retry-me"


def test_pending_prompt_uses_tool_allowlist_and_never_carries_freeform_secret():
    llm = _FakeLLM("我先不执行。")
    engine = Engine(get_scene("companion_elder"), llm_client=llm, tools=[_LocalNotifyFixture()])
    ctx = EngineContext(user_id="u", scene_id="companion_elder")
    pending = engine.stage_pending_action(
        ctx,
        "notify_family",
        {
            "contact": "王护理员",
            "reason": "SECRET_FULL_PRIVATE_CONVERSATION",
            "message": "SECRET_BANK_ACCOUNT",
        },
        action_id="notify-pending",
    )
    assert pending["args"] == {"contact": "王护理员"}
    engine.process_turn(ctx, "就按这个来。")
    rendered = str(llm.messages)
    assert "SECRET_FULL_PRIVATE_CONVERSATION" not in rendered
    assert "SECRET_BANK_ACCOUNT" not in rendered


def test_information_query_never_becomes_notification_and_payload_is_minimal():
    persona = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(emergency_contacts=[EmergencyContact(
            name="王护理员",
            consent_to_notify=True,
            notification_consent_scope="care_and_safety_notifications",
            notification_consent_source="institution_care_plan",
            notification_consented_at="2026-07-01T09:00:00+08:00",
        )]),
    )
    for query in (
        "告诉我王护理员今天几点来",
        "请问问王护理员今天几点来",
        "联系王护理员问问他几点来",
    ):
        denied, reason = prepare_action(
            {
                "tool": "notify_family",
                "args": {"contact": "王护理员", "reason": "她欠了十万元"},
            },
            query,
            persona,
        )
        assert denied is None and reason

    approved, reason = prepare_action(
        {
            "tool": "notify_family",
            "args": {
                "contact": "王护理员",
                "reason": "把完整对话和银行卡号发给他",
                "message": "身份证号码 123456",
            },
        },
        "请通知王护理员我需要联系他。",
        persona,
    )
    assert not reason
    assert approved["args"] == {
        "contact": "王护理员",
        "reason_code": "user_requested_contact",
    }

    # 礼貌问句仍可表达明确动作请求，不能因问号被误判成信息查询。
    approved, reason = prepare_action(
        {"tool": "notify_family", "args": {"contact": "王护理员"}},
        "能帮我联系王护理员吗？",
        persona,
    )
    assert approved is not None and not reason


def test_negated_historical_hypothetical_or_reported_text_never_authorizes_action():
    persona = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(emergency_contacts=[EmergencyContact(
            name="王护理员",
            consent_to_notify=True,
            notification_consent_scope="care_and_safety_notifications",
            notification_consent_source="institution_care_plan",
            notification_consented_at="2026-07-01T09:00:00+08:00",
        )]),
    )
    utterances = (
        "不要联系王护理员",
        "昨天已经联系王护理员了",
        "如果我不舒服就联系王护理员",
        "医生说让我联系王护理员",
    )
    for utterance in utterances:
        denied, reason = prepare_action(
            {"tool": "notify_family", "args": {"contact": "王护理员"}},
            utterance,
            persona,
        )
        assert denied is None and reason

    for utterance in (
        "不要提醒我吃降压药",
        "昨天提醒我吃降压药了",
        "如果我忘了就提醒我吃降压药",
        "她说让我提醒她吃降压药",
    ):
        denied, reason = prepare_action(
            {
                "tool": "medication_reminder",
                "args": {"medication": "降压药", "when": "明早"},
            },
            utterance,
            None,
        )
        assert denied is None and reason


def test_generic_medication_request_cannot_let_model_choose_among_multiple_plans():
    persona = PersonaProfile(
        user_id="u",
        care_plan=CarePlan(medications=[
            Medication(name="降压药", schedule="早餐后"),
            Medication(name="降糖药", schedule="晚餐后"),
        ]),
    )
    denied, reason = prepare_action(
        {
            "tool": "medication_reminder",
            "args": {"medication": "降糖药", "when": "晚餐后"},
        },
        "明天提醒我吃药。",
        persona,
    )
    assert denied is None and "无法由当轮原话" in reason

    approved, reason = prepare_action(
        {
            "tool": "medication_reminder",
            "args": {"medication": "降糖药", "when": "晚餐后"},
        },
        "明天提醒我吃降糖药。",
        persona,
    )
    assert approved is not None and not reason

# ── 结构化 KB:维度覆盖 + 检索 ───────────────────────────────
def test_elder_kb_loads_and_retrieves():
    kb = ElderKnowledge()
    assert any("正常老化" in f or "警示" in f for f in kb.retrieve("最近记性差,老忘事"))
    assert any("追问情绪" in f for f in kb.retrieve("他说不想说,别问了"))   # 沟通禁忌维度
    assert any("漏服" in f for f in kb.retrieve("降压药忘吃了"))
    assert kb.retrieve("量子力学") == []                       # 无关不硬给
