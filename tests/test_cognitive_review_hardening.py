"""Independent C1 review: stale state and source support remain fail closed."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import shutil
import subprocess

import pytest

from aria.engine.cognitive import extract_cognitive_rows, online_cognitive_block, evaluate_cognitive_trend, cognitive_action
from aria.engine.context import Turn, EngineContext
from aria.engine.memory.offline_queue import OfflineSessionQueue
from aria.engine.memory.persona_store import PersonaStore
from aria.scenes.registry import get_scene
from aria.voice.minimal import MinimalWebApp
from tests.test_cognitive_change_v2 import T0, READING, enabled, calibrated, record, context_with


def test_stale_saved_suggestion_expires_without_mutating_profile():
    persona = calibrated(enabled())
    record(persona, 15, "daily_function", True)
    record(persona, 16, "daily_function", True)
    evaluate_cognitive_trend(persona, now=T0 + timedelta(days=16))
    assert persona.cognitive_trend.status == "review_suggested"
    original = deepcopy(persona.model_dump())
    assert online_cognitive_block(persona, "今天挺好。", [Turn("你好", "你好")], READING,
                                  now=T0 + timedelta(days=16))
    assert not online_cognitive_block(persona, "今天挺好。", [Turn("你好", "你好")], READING,
                                      now=T0 + timedelta(days=47))
    assert persona.model_dump() == original


@pytest.mark.parametrize("text", [
    "我忘了买点菜，但是没有影响做饭。", "我忘了交电费，后来办好了，没有耽误家里用电。",
    "我忘了带零钱，不过不影响买菜。",
])
def test_negated_functional_impact_cannot_be_granted_by_model_boolean(text):
    persona = enabled()
    ctx = context_with(persona, text)
    rows = extract_cognitive_rows([{"domain": "daily_function", "evidence_quote": text,
                                   "turn_index": 0, "functional_impact": True}], persona, ctx)
    assert len(rows) == 1 and not rows[0].functional_impact


def test_unrelated_literal_fragment_is_not_memory_evidence():
    persona = enabled()
    ctx = context_with(persona, "我想不起钥匙放在哪儿了。今天太阳挺好的。")
    assert not extract_cognitive_rows([{"domain": "memory", "evidence_quote": "今天太阳挺好的。",
                                       "turn_index": 0}], persona, ctx)


@pytest.mark.parametrize("words", ["电费怎么交？", "下一步按哪个按钮？", "那个开关在哪里？"])
def test_practical_request_is_not_interrupted_by_cognitive_suggestion(words):
    persona = calibrated(enabled())
    record(persona, 15, "daily_function", True)
    record(persona, 16, "daily_function", True)
    evaluate_cognitive_trend(persona, now=T0 + timedelta(days=16))
    assert not online_cognitive_block(persona, words, [Turn("你好", "你好")], READING,
                                      now=T0 + timedelta(days=16))


def test_unknown_model_result_does_not_block_owner_revocation(tmp_path):
    app = MinimalWebApp(tmp_path)
    assert app.cognitive_action("enable")["ok"]
    sid = app.start()["state"]["session_id"]
    assert app.submit("今天挺好。", "one", sid)["ok"]
    app._run["turns"][0]["requires_review"] = True
    app._save_run()
    assert app.state()["requires_review"]
    assert app.cognitive_action("disable", delete_records=True)["ok"]
    assert not app.state()["cognitive"]["enabled"]
    assert app.state()["requires_review"] and not app.start()["ok"]
    app.close()


def test_disabled_retained_observations_still_cannot_be_shared_without_scope():
    persona = enabled()
    note = record(persona, 15, "daily_function", True)
    cognitive_action(persona, "disable", actor=persona.user_id, delete_records=False,
                     now=T0 + timedelta(days=16))
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    reply = get_scene("companion_elder").guard_delivery_response(
        ctx, "给女儿的草稿：" + note.source_quote, "写点什么给女儿吧", persona)
    assert note.source_quote not in reply
    assert "cognitive_private_sharing_blocked" in ctx.working_memory["delivery_response_guard"]["violations"]


def test_assistant_repeating_private_observation_never_enters_ordinary_offline_history(tmp_path):
    persona = enabled()
    note = record(persona, 15, "daily_function", True)
    ctx = EngineContext(user_id=persona.user_id, scene_id="companion_elder")
    ctx.working_memory["_cognitive_block_selected"] = True
    words = "今天挺好的。"
    reply = get_scene("companion_elder").guard_delivery_response(
        ctx, "您曾经说过：" + note.source_quote, words, persona)
    ctx.add_turn(words, reply)
    queue = OfflineSessionQueue(persona, PersonaStore(tmp_path), now=lambda: T0 + timedelta(days=16))
    assert queue.enqueue(ctx)["status"] == "queued"
    assert queue.persona.offline_sessions[0].turns[0].user == ""
    assert queue.persona.offline_sessions[0].turns[0].agent == ""


def test_actual_ui_allows_only_privacy_reduction_under_unknown_and_locks_replay():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable for UI function execution; backend tests still run")
    page = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf-8")
    controls = page[page.index("function updateControls() {"):page.index("function framesFor(turns) {")]
    mutation = page[page.index("async function mutate(endpoint, body, onSuccess) {"):page.index("$('composer').addEventListener")]
    setup = """
const assert=require('assert');
let busy=false, careBusy=false, replayMode=false, history=false;
let uncertain={endpoint:'text',id:'unresolved-model-turn'};
let state={requires_review:true,storage_failed:true,turns:[],sessions:[]};
const buttons=[{dataset:{cognitiveReduction:'true'}},{dataset:{}}];
const elements={};
const $=id=>elements[id] || (elements[id]={value:'',dataset:{},querySelectorAll:()=>id==='cognitive-panel'?buttons:[]});
const array=v=>Array.isArray(v)?v:[];
const allReviews=()=>[];
const viewingHistory=()=>history;
const modelCallsStopped=()=>false;
const isCurrentOpen=()=>true;
const updateCareControls=()=>{};
const renderModelUsage=()=>{};
const setError=()=>{};
const status=()=>{};
const render=value=>{state=value;};
let calls=[];let fail=false;
const request=async(endpoint,body)=>{calls.push({endpoint,body});if(fail)throw new Error('unknown');return {state};};
"""
    checks = """
(async()=>{
 updateControls();assert.strictEqual(buttons[0].disabled,false);assert.strictEqual(buttons[1].disabled,true);
 const original=uncertain;
 await mutate('cognitive',{action:'enable'});assert.strictEqual(calls.length,0);
 await mutate('cognitive',{action:'disable',delete_records:true});assert.strictEqual(calls.length,1);assert.strictEqual(uncertain,original);
 fail=true;await mutate('cognitive',{action:'delete',id:'one'});assert.strictEqual(calls.length,2);assert.strictEqual(uncertain,original);
 replayMode=true;updateControls();assert(buttons.every(button=>button.disabled));
 await mutate('cognitive',{action:'disable',delete_records:false});assert.strictEqual(calls.length,2);
 replayMode=false;history=true;updateControls();assert(buttons.every(button=>button.disabled));
 await mutate('cognitive',{action:'delete',id:'one'});assert.strictEqual(calls.length,2);
 console.log('cognitive UI privacy/replay checks passed');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run([node, "-e", setup + controls + mutation + checks], capture_output=True,
                            text=True, encoding="utf-8", check=False)
    assert result.returncode == 0, result.stderr
