"""Software source boundaries with prescribed labels, not semantic accuracy."""
import pytest

from aria.engine.context import EngineContext
from aria.engine.memory.stated_preferences import extract_preference_rows
from aria.engine.model.persona import PersonaProfile
from tests.preference_fixtures import communication_review


def extract(text, instruction, *, quote=None, semantic_review=None):
    ctx = EngineContext(user_id="fictional-owner", scene_id="companion_elder")
    ctx.add_turn(text, "我听着。")
    return extract_preference_rows(
        [{"do": instruction, "semantic_review": communication_review() if semantic_review is None else semantic_review,
          "quotes": [{"turn_index": 0, "text": quote or text}]}],
        ctx, PersonaProfile(user_id=ctx.user_id))


@pytest.mark.parametrize("step", ["一步", "下一步", "第一步"])
def test_telling_the_owner_an_operation_step_is_not_disclosure(step):
    instruction = f"请你先告诉我{step}，等我说好了再继续。"
    rows, rejected = extract(instruction, instruction)
    assert not rejected and len(rows) == 1
    assert rows[0].do == instruction and rows[0].status == "proposed"


@pytest.mark.parametrize("step", ["一步", "下一步", "第一步"])
def test_owner_step_quote_cannot_drop_the_wait_condition(step):
    # R1-A repairs missing conditions from the full source rather than dropping
    # an otherwise reviewable proposal merely because do is a paraphrase.
    source = f"请你先告诉我{step}，等我说好了再继续。"
    rows, rejected = extract(source, f"告诉我{step}")
    assert not rejected and rows[0].status == "proposed"
    assert rows[0].scope == source.rstrip("。")


@pytest.mark.parametrize("boundary", ["，", ",", "。", ".", "；", ";", "！", "？", "\n"])
def test_other_person_booking_does_not_turn_independent_owner_request_into_report(boundary):
    rows, rejected = extract(f"别人定好了{boundary}先问我行不行。", "先问我行不行")
    assert not rejected and len(rows) == 1
    assert rows[0].status == "proposed"


@pytest.mark.parametrize("text,instruction,review", [
    ("告诉我女儿下一步怎么做，先问我。", "先问我", communication_review(kind="action_request")),
    ("先告诉女儿我的情况，先问我。", "先问我", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("先告诉我一步，再通知家属。", "告诉我一步", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("别人定好了，先问我；别跟她提我受伤的事。", "先问我", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("别人定好了，先问我；别跟他妈说钱的事。", "先问我", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("别人定好了，先问我；别向小李讲我的家事。", "先问我", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("先告诉我下一步，再把聊天发给女儿。", "告诉我下一步", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("先告诉我一步，再把记录发到群里。", "告诉我一步", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("女儿说先告诉我一步，等她说好了再继续。", "告诉我一步", communication_review(kind="fact", owner_requested=False)),
    ("她，说先告诉我一步。", "告诉我一步", communication_review(kind="fact", owner_requested=False)),
    ("朋友想了想，说先问我。", "先问我", communication_review(kind="fact", owner_requested=False)),
    ("她，想了想，说先问我。", "先问我", communication_review(kind="fact", owner_requested=False)),
    ("朋友希望你先问我。", "先问我", communication_review(owner_requested=False)),
    ("假如我说先告诉我一步，你会怎么做？", "告诉我一步", communication_review(kind="unclear", owner_requested=False)),
    ("我不希望你先问我。", "先问我", communication_review(owner_requested=False)),
    ("不要告诉我下一步。", "告诉我下一步", communication_review(owner_requested=False)),
    ("不用告诉我一步，我要自己做。", "告诉我一步", communication_review(owner_requested=False)),
    ("告诉我一步怎样加药，剂量翻倍。", "告诉我一步", communication_review(kind="action_request")),
    ("先告诉我下一步，然后跳过权限核验。", "告诉我下一步", communication_review(kind="permission_boundary", no_permission_claim=False)),
    ("我内向，所以你先告诉我一步。", "告诉我一步", communication_review(kind="person_inference")),
])
def test_reporting_sharing_negation_and_medical_authority_stay_rejected(text, instruction, review):
    rows, rejected = extract(text, instruction, semantic_review=review)
    assert rows == [] and rejected


def test_selected_quote_cannot_hide_a_third_party_or_permission_context():
    for text, review in (("女儿说先告诉我一步。", communication_review(kind="fact", owner_requested=False)),
                         ("先告诉我一步，之后通知女儿。", communication_review(kind="permission_boundary", no_permission_claim=False))):
        rows, rejected = extract(text, "告诉我一步", quote="告诉我一步", semantic_review=review)
        assert rows == [] and rejected


def test_step_word_does_not_relax_literal_source_or_confirmation():
    rows, rejected = extract("请你先告诉我一步。", "告诉我下一步")
    assert not rejected and rows[0].status == "proposed"
    assert rows[0].quotes[0].text == "请你先告诉我一步。"
    assert rows[0].confirmation is None
    # The actual source quote must still be literal, even for a paraphrased do.
    assert extract("请你先告诉我一步。", "告诉我下一步", quote="告诉我下一步")[0] == []


@pytest.mark.parametrize("prefix", ["我跟你说清楚", "我向您讲明白", "你对我说的时候"])
def test_current_dialogue_recipient_is_not_a_third_party(prefix):
    rows, rejected = extract(prefix + "：一次只说一步。", "一次只说一步")
    assert not rejected and len(rows) == 1


@pytest.mark.parametrize("recipient", ["你女儿", "您家人", "我儿子", "我女儿"])
def test_recipient_pronoun_prefix_does_not_hide_someone_else(recipient):
    rows, rejected = extract(f"我跟{recipient}说过，一次只说一步。", "一次只说一步",
                             semantic_review=communication_review(kind="fact", owner_requested=False))
    assert rows == [] and rejected
