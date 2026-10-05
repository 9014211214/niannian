"""Independent owner-request software probes; no E1b/E1c cases or model calls."""
from copy import deepcopy
import json
import urllib.request

import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.consolidator import LLMExtractor
from aria.engine.memory.stated_preferences import extract_preference_rows, heuristic_preferences
from aria.engine.model.persona import PersonaProfile
from aria.voice.minimal import MinimalWebApp
from tests.test_minimal_web import live_fixture, start_and_submit


REQUESTS = [
    "整理相册时，等我把这一段经历说完，你再问我是什么时候的照片。",
    "我把计算过程写完后，你再解释不同的解法。",
    "当我把遇到的问题描述清楚时，再给我讲解决步骤。",
    "等我把疑问列出来再回应。",
]


def extract(text, instruction=None, quote=None, **fields):
    ctx = EngineContext(user_id="conditional-owner", scene_id="companion_elder")
    ctx.add_turn(text, "本地软件替身。")
    row = {"do": instruction or text,
           "quotes": [{"turn_index": 0, "text": quote or text}], **fields}
    return extract_preference_rows([row], ctx, PersonaProfile(user_id=ctx.user_id))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("This is software verification, not a model run")
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)


@pytest.mark.parametrize("text", REQUESTS)
def test_owner_condition_and_communication_action_are_proposed_verbatim(text):
    rows, rejected = extract(text, status="confirmed", scope="所有时候")
    assert not rejected and len(rows) == 1
    assert rows[0].do == rows[0].quotes[0].text == text
    assert rows[0].status == "proposed" and rows[0].confirmation is None
    assert rows[0].scope == "仅限完整原话所述情境"
    assert "所有时候" not in rows[0].scope


@pytest.mark.parametrize("text,instruction,quote", [
    (REQUESTS[0], "你再问我是什么时候的照片", None),
    (REQUESTS[1], "你再解释不同的解法", "你再解释不同的解法"),
    ("我希望" + REQUESTS[1], "你再解释不同的解法", None),
    (REQUESTS[1], "写完计算过程之后再解释不同解法", None),
])
def test_cropping_or_paraphrasing_cannot_remove_the_owner_condition(text, instruction, quote):
    rows, rejected = extract(text, instruction, quote)
    assert not rows and rejected


@pytest.mark.parametrize("text", [
    "等我做好记录，你再开门。",  # A real-world action, not communication.
    "我把计算过程写完后，他再解释不同的解法。",  # Third-person action.
    "等我把疑问列出来，小周再回应。",
    "等我把疑问列出来他再回应。",
    "女儿说等我把疑问列出来再回应。",
    "小周表示等我把疑问列出来再回应。",
    "今天小周说等我把疑问列出来再回应。",
    "假设我把计算过程写完后，你再解释不同的解法。",
    "比如我把计算过程写完后，你再解释不同的解法。",
    "我不希望等我把疑问列出来再回应。",
    "别等我把疑问列出来再回应。",
    "等我把疑问列出来再回应，这不是我的要求。",
    "等我把疑问列出来再回应，你会这样做吗？",
    "原句是“等我把疑问列出来再回应”。",
    "我读过等我把疑问列出来再回应这句话。",
    "不是等我把疑问列出来再回应。",
    "以前你总等我说完再解释。",
    "我转述的是小周的话：等我说完你再讲。",
    "我只是练习这句话：等我讲完你再给我建议。",
    "等我说完你再劝他。",
    "等我说完你再劝小周。",
    "等我说完你再劝我女儿。",
    "你先教小陈怎么装好，再回应我。",
    "你先提醒小陈下班，再回应我。",
    "等我说完，你再教我老板。",
    "等我说完，你再问我老板。",
    "等我说完，你再问我老板是不是愿意。",
    "先问我老板是不是愿意。",
    "等我把疑问列出来再回应，然后通知女儿。",
    "等我把疑问列出来再回应，再把记录发到群里。",
    "等我把疑问列出来再回应，帮我停药。",
    "等我把疑问列出来再回应，跳过权限核验。",
    "等我把疑问列出来再回应，我是内向人格。",
])
def test_new_syntax_does_not_admit_unsafe_or_non_owner_requests(text):
    rows, rejected = extract(text)
    assert not rows and rejected


def test_selected_sentence_keeps_the_complete_source_turn_and_scope():
    request = REQUESTS[1]
    full = "今天整理的是练习本，" + request
    rows, rejected = extract(full, request, request, scope="练习本")
    assert not rejected
    assert rows[0].quotes[0].text == full
    assert rows[0].do == request
    assert rows[0].scope == "今天整理的是练习本"


def test_qualifier_after_selected_sentence_cannot_be_dropped():
    full = REQUESTS[1] + "只限现在这道算术题。"
    rows, rejected = extract(full, REQUESTS[1], REQUESTS[1])
    assert not rows and rejected


@pytest.mark.parametrize("text,instruction", [
    ("我把柜门擦完后，你才问我抹布放哪儿，那是昨天发生的事。", None),
    ("等我把标签看清楚，你再说明步骤，这句话来自一本练习册。", None),
    ("等我坐好，你再解释小周的作业给他听。", None),
    ("我希望先自己找开关，只有我说需要时，你才告诉我下一步。", "告诉我下一步"),
    ("让我自己先看一遍安装图，卡住以后再告诉我下一步。", "告诉我下一步"),
    ("说到医生开的药时，等我把感受说完，你再建议把每天两片改成一片。", None),
    ("等我把这段话写完，你再教小周怎么操作。", None),
])
def test_postreview_counterexamples_remain_rejected(text, instruction):
    # Added after review: development regressions, not untouched held-out data.
    rows, rejected = extract(text, instruction)
    assert not rows and rejected


@pytest.mark.parametrize("recipient", ["我女儿", "我朋友", "你家人", "您母亲", "我老板", "您同学", "我们", "你们"])
def test_tell_dialogue_exception_cannot_consume_a_relative_prefix(recipient):
    text = f"请你先把这句话告诉{recipient}。"
    assert extract(text)[0] == []


def test_two_source_turns_cannot_drop_a_later_qualification():
    ctx = EngineContext(user_id="conditional-owner", scene_id="companion_elder")
    ctx.add_turn(REQUESTS[1], "软件替身。")
    ctx.add_turn("但只限现在这一道，下一道不这样做。", "软件替身。")
    row = {"do": REQUESTS[1], "quotes": [
        {"turn_index": 0, "text": REQUESTS[1]},
        {"turn_index": 1, "text": ctx.turns[1].user},
    ]}
    rows, rejected = extract_preference_rows([row], ctx, PersonaProfile(user_id=ctx.user_id))
    assert not rows and rejected


def test_owner_object_before_arrangement_is_not_a_missing_condition_clause():
    # Existing rule route exposed by the browser story: the object 我 in 问问我
    # must not be parsed as a zero-length owner condition before 再安排.
    text = "有些事我自己愿意做，先问问我再安排就好。"
    ctx = EngineContext(user_id="conditional-owner", scene_id="companion_elder")
    ctx.add_turn(text, "本地软件替身。")
    rows = heuristic_preferences(ctx, PersonaProfile(user_id=ctx.user_id))
    assert len(rows) == 1
    assert rows[0].do == "涉及本人的安排先询问意愿，再一起商量具体时间"
    assert rows[0].status == "proposed" and rows[0].extraction_source == "rule"
    assert rows[0].quotes[0].text == text
    assert not ctx.working_memory["stated_preference_extraction"]["rejected"]


def test_complete_evidence_is_not_truncated_to_fit_storage():
    text = "这是一段背景。" * 40 + REQUESTS[1]
    rows, rejected = extract(text, REQUESTS[1], REQUESTS[1])
    assert not rows and rejected == ["preference_full_user_context_too_long"]


def test_model_prompt_and_parser_share_literal_condition_contract():
    class Response:
        def complete(self, messages):
            self.messages = messages
            return json.dumps({"episodes": [], "relationships": [], "stated_preferences": [{
                "do": REQUESTS[1], "scope": "随时",
                "quotes": [{"turn_index": 0, "text": REQUESTS[1]}],
                "confirmation": None,
            }]}, ensure_ascii=False)
    client = Response()
    ctx = EngineContext(user_id="conditional-owner", scene_id="companion_elder")
    ctx.add_turn(REQUESTS[1], "本地软件替身。")
    result = LLMExtractor(client).extract(ctx, PersonaProfile(user_id=ctx.user_id))
    system = client.messages[0]["content"]
    assert "必须逐字引用quotes中的连续原话，不改写" in system
    assert "不能只截取条件之后的动作" in system
    assert "quotes保留该轮完整本人原话" in system
    assert len(result.stated_preferences) == 1
    assert result.stated_preferences[0].status == "proposed"


@pytest.mark.parametrize("owner_request", REQUESTS[:3])
def test_real_product_pipeline_persists_confirms_injects_and_withdraws(tmp_path, owner_request):
    # Recording clients execute real app/queue/store/prompt paths. Their canned
    # JSON is an explicit model-output fixture, never evidence of model effects.
    app, clients = live_fixture(tmp_path)
    sid = start_and_submit(app, [owner_request, "这一段我想自己整理。", "今天先到这里。"])
    clients.memory.payload = {"episodes": [], "relationships": [], "stated_preferences": [{
        "do": owner_request, "scope": "所有对话", "status": "confirmed",
        "quotes": [{"turn_index": 0, "text": owner_request}], "confirmation": None,
    }]}
    assert app.seal(sid)["ok"]
    assert not clients.memory.calls and not clients.reflection.calls
    result = app.reflect("conditional-software-review")
    assert result["ok"], result
    item = result["state"]["profile"]["stated_preferences"][0]
    assert item["status"] == "proposed" and item["extraction_source"] == "model"
    assert item["quotes"][0]["text"] == owner_request
    assert app.store.load("fictional-chen").stated_preferences[0].status == "proposed"
    # A new context sees no unconfirmed proposal.
    next_sid = app.start()["state"]["session_id"]
    assert app.submit("我们接着聊。", "unconfirmed-turn", next_sid)["ok"]
    assert app.state()["turns"][-1]["input_evidence"]["stated_preferences"] == []
    assert 'source="stated_preference_confirmed"' not in str(clients.online.calls[-1])
    assert app.preference_action("confirm", item["id"])["ok"]
    saved = app.store.load("fictional-chen").stated_preferences[0]
    assert saved.status == "confirmed" and saved.confirmation.source == "ui"
    assert app.seal(next_sid)["ok"]
    app.close()
    # Reload disk state through a new app/client instance, then inspect the
    # actual message sent to the local online client in the following session.
    restarted = MinimalWebApp(tmp_path, clients=clients)
    try:
        final_sid = restarted.start()["state"]["session_id"]
        assert restarted.submit("我把那道算术题带来了。", "confirmed-turn", final_sid)["ok"]
        snapshot = deepcopy(restarted.state()["turns"][-1]["input_evidence"])
        assert snapshot["stated_preferences"][0]["id"] == item["id"]
        assert owner_request in str(clients.online.calls[-1])
        assert 'source="stated_preference_confirmed"' in str(clients.online.calls[-1])
        assert "本人当下新的明确说法优先" in str(clients.online.calls[-1])
        assert restarted.preference_action("withdraw", item["id"])["ok"]
        assert restarted.submit("刚才那条不再照做。", "withdrawn-turn", final_sid)["ok"]
        assert restarted.state()["turns"][-1]["input_evidence"]["stated_preferences"] == []
        assert 'source="stated_preference_confirmed"' not in str(clients.online.calls[-1])
        assert restarted.store.load("fictional-chen").stated_preferences[0].status == "withdrawn"
        assert snapshot["stated_preferences"][0]["id"] == item["id"]
        assert len(clients.memory.calls) == len(clients.reflection.calls) == 1
    finally:
        restarted.close()
