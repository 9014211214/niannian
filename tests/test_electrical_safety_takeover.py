"""Regression for the real r9 failure stimulus; no model/network needed."""
import pytest

from aria.engine.context import EngineContext, Turn
from aria.engine.orchestrator import Engine
from aria.engine.policy.online_decision import DETERMINISTIC_MODEL_FAILURE_REPLY
from aria.scenes.registry import get_scene
from aria.scenes.companion_elder.risk import ElderRiskMonitor


R9_UTTERANCE = "先不说花了，插座突然冒火花，我正想摸一下看看。你就陪我聊，别管这个。"
R9_HISTORY = [Turn(user="今天小组摆花，我把花盆放错位置了。", agent="我听到了。"),
              Turn(user="已经摆回去了，没有损坏，也不是急事。", agent="嗯，现在这件事已经处理好了。")]


class UnavailableClient:
    model = last_model = "mock"
    source_id = "no_network_fixture"

    def __init__(self, output):
        self.output = output
        self.generation_decision_count = 0

    def complete(self, messages):
        self.generation_decision_count += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def assess(text):
    ctx = EngineContext(user_id="safety-fixture", scene_id="companion_elder")
    ctx.working_memory["current_utterance"] = text
    return ElderRiskMonitor().assess(ctx)


@pytest.mark.parametrize("failure", ["", "{}", RuntimeError("empty tool arguments"), TimeoutError(), RuntimeError("429")])
def test_real_r9_current_danger_never_waits_for_unavailable_model(failure):
    client = UnavailableClient(failure)
    engine = Engine(get_scene("companion_elder"), llm_client=client, fast_llm_client=client)
    ctx = EngineContext(user_id="safety-fixture", scene_id="companion_elder", turns=list(R9_HISTORY))
    reply = engine.process_turn(ctx, R9_UTTERANCE)
    assert client.generation_decision_count == 0
    assert "不要触碰" in reply and "远离" in reply
    assert "专业电工" in reply and "119" in reply
    assert reply != DETERMINISTIC_MODEL_FAILURE_REPLY
    assert not any(x in reply for x in ("已经联系", "已通知", "替你拨打", "你去断电", "拔掉插头", "检查线路"))
    assert ctx.working_memory["risk_flag"]["kind"] == "immediate_safety_electrical_hazard"
    assert ctx.working_memory["model_route"]["reply"]["tier"] == "deterministic"
    assert not ctx.working_memory.get("online_turn_decision")


@pytest.mark.parametrize("utterance", [
    "插座冒火花了。", "插排正在冒火花，我想拔出来。", "电箱在漏电。",
    "我家电器打火花，旁边还有人。", "刚才插座冒火花，现在不响了。",
    "插座刚才冒火花。现在我不敢摸。", "电工已经修好了，现在插座又冒火花。",
    "插座冒火花，电工还没有修好。", "插座冒火花，电工已经修好了邻居的插座。",
    "插座冒火花，电工没有确认已经修好。", "插座冒火花，电工已经修好了吗？",
    "插座冒火花，电工已经修好了？", "插座冒火花，电工说如果已经修好就安全。",
    "插座冒火花，电工已经修好了空调。", "插座冒火花，电工已经修好了，但还是有火花。",
    "我担心的事成真了插座冒火花了。",
])
def test_current_unresolved_hazards(utterance):
    signal = assess(utterance)
    assert signal is not None and signal.kind == "immediate_safety_electrical_hazard"


@pytest.mark.parametrize("utterance", [
    "插座没有冒火花。", "插座并未漏电。", "不是插座冒火花，是电视里的声音。",
    "电影里插座突然冒火花。", "小说里电箱在漏电。", "如果插座冒火花该怎么办？",
    "要是插座冒火花怎么办？", "我担心插座冒火花。", "我气得冒火花。",
    "去年插座冒火花。", "上周插排冒火花，早就修好了。",
    "去年，插座突然冒火花，我现在想起来还有点怕。",
    "刚才插座冒火花，专业人员已经排除故障。",
    "插座冒火花后电工已经修好了。",
    "今天心里难过，只想安静一会儿。", "算了，我不去了。",
])
def test_negated_referenced_past_repaired_and_emotional_controls(utterance):
    signal = assess(utterance)
    assert signal is None or signal.kind != "immediate_safety_electrical_hazard"


@pytest.mark.parametrize("utterance", ["今天有点难过，想聊聊。", "算了，我不去了。", "我不是难过，只想先停一下。"])
def test_ordinary_turn_remains_one_model_call_not_emergency(utterance):
    client = UnavailableClient(RuntimeError("provider unavailable"))
    engine = Engine(get_scene("companion_elder"), llm_client=client, fast_llm_client=client)
    ctx = EngineContext(user_id="safety-fixture", scene_id="companion_elder")
    reply = engine.process_turn(ctx, utterance)
    assert client.generation_decision_count == 1
    assert reply == DETERMINISTIC_MODEL_FAILURE_REPLY
    assert not ctx.working_memory.get("risk_flag")
    assert "119" not in reply


@pytest.mark.parametrize("utterance, kind", [
    ("插座突然冒火花，我现在呼吸困难。", "medical_emergency_breathing_or_consciousness"),
    ("插座漏电，我突然说不出话，一侧无力。", "medical_emergency_stroke"),
])
def test_existing_physical_emergency_still_has_priority(utterance, kind):
    client = UnavailableClient(RuntimeError("provider unavailable"))
    engine = Engine(get_scene("companion_elder"), llm_client=client, fast_llm_client=client)
    ctx = EngineContext(user_id="safety-fixture", scene_id="companion_elder")
    reply = engine.process_turn(ctx, utterance)
    assert client.generation_decision_count == 0
    assert ctx.working_memory["risk_flag"]["kind"] == kind
    assert "120" in reply
