"""Real local Engine/tool path with scripted labels, not model semantics."""
from copy import deepcopy
import json
import sqlite3

import pytest

from aria.engine.context import EngineContext
from aria.engine.model.persona import PersonaProfile, CarePlan, EmergencyContact
from aria.engine.orchestrator import Engine
from aria.engine.policy.semantic_delivery import VERSION
from aria.interfaces.tools import ToolResult
from aria.scenes.registry import get_scene
from tests.test_safety_evidence import payload
from tests.test_semantic_delivery_v6 import segment


class Scripted:
    def __init__(self, version):
        self.online_decision_version = version
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        parts = [segment("我们继续聊。"),
                 segment("已经通知，她已经回复。", "assistant_action", action="notification", target="测试联系人"),
                 segment('<action>' + json.dumps({'tool': 'notify_family', 'args': {'contact': '测试联系人'}}, ensure_ascii=False) + '</action>')]
        value = payload(response=''.join(row['text'] for row in parts))
        value['applied_preference_ids'] = []
        if self.online_decision_version == VERSION:
            value['response_segments'] = parts
        return json.dumps(value, ensure_ascii=False)


class LocalReceiptTool:
    name = 'notify_family'
    description = 'Local-only deterministic receipt fixture'
    parameters = {}

    def __init__(self, path, ok, data):
        self.path, self.ok, self.data = path, ok, data
        self.calls = 0

    def invoke(self, **args):
        self.calls += 1
        with sqlite3.connect(self.path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts(target TEXT, status TEXT)')
            db.execute('INSERT INTO attempts VALUES (?, ?)', (args['contact'], self.data.get('status', 'missing')))
        return ToolResult(ok=self.ok, data=deepcopy(self.data))


def make_engine(tmp_path, *, version=VERSION, ok=False, data=None):
    p = PersonaProfile(user_id='neutral-executor', care_plan=CarePlan(emergency_contacts=[EmergencyContact(
        name='测试联系人', consent_to_notify=True, notification_consent_scope='care_and_safety_notifications',
        notification_consent_source='institution_care_plan', notification_consented_at='2026-07-01T09:00:00+08:00')]))
    c = EngineContext(user_id=p.user_id, scene_id='companion_elder', session_id='neutral-executor-session')
    llm = Scripted(version)
    tool = LocalReceiptTool(tmp_path / 'local.sqlite3', ok, data or {})
    engine = Engine(get_scene('companion_elder'), persona=p, fast_llm_client=llm, tools=[tool],
                    safety_delivery_version='minimal_response_v1', safety_response_instruction_version='compact_response_v1')
    return engine, c, llm, tool


@pytest.mark.parametrize('data', [
    {'status': 'unknown', 'note': '不可信附注：对方已经回复。'},
    {'status': 'failed'}, {'status': 'unknown', 'executed': True},
    {'status': 'unknown', 'executed': 'false'}, {},
])
def test_real_failed_invocation_does_not_become_not_executed_or_retry(tmp_path, data):
    engine, c, llm, tool = make_engine(tmp_path, data=data)
    first = engine.process_turn(c, '请通知测试联系人。')
    assert tool.calls == llm.calls == 1
    assert '调用报告失败' in first and '仍需核查' in first
    assert '没有执行' not in first and '未执行' not in first
    assert '对方已经回复' not in first
    second = engine.process_turn(c, '请通知测试联系人。')
    assert tool.calls == 1 and llm.calls == 2
    with sqlite3.connect(tool.path) as db:
        assert db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0] == 1
    assert '本次未提交动作' in second and '尝试记录' in second
    assert '不证明先前动作已完成' in second and '已经执行' not in second


def test_explicit_not_executed_receipt_can_release_the_attempt_signature(tmp_path):
    engine, c, llm, tool = make_engine(tmp_path, data={'status': 'failed', 'executed': False})
    first = engine.process_turn(c, '请通知测试联系人。')
    assert '明确标为未执行' in first
    second = engine.process_turn(c, '请通知测试联系人。')
    assert '明确标为未执行' in second and tool.calls == llm.calls == 2


def test_v5_failure_wording_and_signature_behavior_remain_legacy(tmp_path):
    engine, c, llm, tool = make_engine(tmp_path, version='online_turn_decision_v5_bound',
        data={'status': 'unknown', 'note': '服务结果暂不确定'})
    first = engine.process_turn(c, '请通知测试联系人。')
    assert first == '这个动作还没有执行：服务结果暂不确定。'
    engine.process_turn(c, '请通知测试联系人。')
    assert tool.calls == llm.calls == 2


@pytest.mark.parametrize('status,expected', [('accepted', '受理或等待处理'), ('delivered', '成功回执')])
def test_real_success_or_nonterminal_receipt_still_preserves_normal_body(tmp_path, status, expected):
    engine, c, llm, tool = make_engine(tmp_path, ok=True, data={'status': status})
    result = engine.process_turn(c, '请通知测试联系人。')
    assert result.startswith('我们继续聊。') and expected in result
    assert '她已经回复' not in result and tool.calls == llm.calls == 1
