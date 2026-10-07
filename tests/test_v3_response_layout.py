"""Preserve delivered layout without weakening the existing safety scan view."""
import json

import pytest

from aria.engine.context import EngineContext
from aria.engine.policy import online_decision
from aria.interfaces.reasoning import ActiveObservationPlan
from aria.runtime import MinimalModelClients, MinimalPreviewLLMClient
from aria.scenes.companion_elder.scene import ElderCompanionScene
from aria.voice.minimal import MinimalWebApp


def _payload(text):
    return json.dumps({
        "response": text, "state_signals": [],
        "current_affect": {"label": "unknown", "confidence": 0.,
                           "divergence": {"possible": False, "confidence": 0., "pattern": ""}},
        "selected_intervention_id": "", "uncertainty": 1., "safety_concern": "none",
        "communication_plan": {"pace": "brief_exchange", "support_order": "回应当前内容", "boundary": "不执行动作"},
    }, ensure_ascii=False)


def _pipeline(text, version="v3"):
    ctx = EngineContext(user_id="layout-fixture", scene_id="companion_elder")
    ctx.working_memory["_online_decision_version"] = "online_turn_decision_" + version
    ctx.reasoning.observation_plan = ActiveObservationPlan(mode="observe")
    scene = ElderCompanionScene()
    response = online_decision.apply_single_pass_decision(_payload(text), ctx, "请回应现在的问题。")
    response = scene.guard_response(ctx, response)
    return scene.guard_delivery_response(ctx, response, "请回应现在的问题。"), ctx


@pytest.mark.parametrize("text", [
    "第一行。\n第二行。\n第三行。", "第一行。\r\n第二行。", "第一段。\n\n第二段。", "  甲。\t 乙。 \n 丙。  ",
])
def test_v3_preserves_body_layout_through_all_guards(text):
    response, ctx = _pipeline(text)
    assert response == text.strip()
    assert ctx.working_memory["online_turn_decision"]["delivery_guard"]["violations"] == []


@pytest.mark.parametrize("text", [
    "您\n就是高血压。", "建议您把降压药\n剂量减半。", "我保证\n这种药一定有效。",
    "您患阿尔茨海默的概率\n是80%。", "我已经\n通知您女儿了。",
])
def test_safety_and_receipt_checks_still_scan_across_line_breaks(text):
    compact = " ".join(text.strip().split())
    actual, _ = _pipeline(text)
    control, _ = _pipeline(compact)
    assert actual == control
    assert actual != compact


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_legacy_versions_keep_their_existing_layout(version):
    text = "  第一段。\n\n第二段。  "
    assert _pipeline(text, version)[0] == " ".join(text.strip().split())


def test_complete_web_response_and_provenance_preserve_neutral_layout(tmp_path):
    response = "先摆好纸。\n接着画一条线。\n最后把纸收好。"

    class StaticClient:
        source_id = "software_layout_fixture"
        model = "local-fixture"
        online_return_format = "json_object"
        online_decision_version = "online_turn_decision_v3"
        calls = 0

        def complete(self, messages):
            self.calls += 1
            return _payload(response)

    online = StaticClient()
    offline = MinimalPreviewLLMClient()
    app = MinimalWebApp(tmp_path / "data", clients=MinimalModelClients(online, offline, offline, True))
    try:
        sid = app.start()["state"]["session_id"]
        result = app.submit("把整理纸张的三步分三行告诉我。", "layout-fixture", sid)
        turn = result["state"]["turns"][-1]
        assert result["ok"] and online.calls == 1
        assert turn["response"] == response
        assert turn["source"] == "live"
        assert not turn["delivery_protection"]["response_changed"]
        assert not app._run["reviews"]
    finally:
        app.close()
