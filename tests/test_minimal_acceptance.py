"""M03 assembly acceptance. Prescribed outputs test software, not model quality."""
from tests.test_minimal_web import live_fixture, start_and_submit
from aria.voice.minimal import MinimalWebApp
from aria.engine.minimal_care import MinimalCareService
import pytest


def test_cold_start_and_missing_emotion_stay_unknown_without_media(tmp_path):
    app, clients = live_fixture(tmp_path)
    assert app.state()["profile"]["facts"] == app.state()["profile"]["hypotheses"] == []
    sid = start_and_submit(app, ["纸和本子放在桌上。"])
    row = app.state()["turns"][0]
    assert row["emotion_summary"]["label"] == "unknown"
    assert row["emotion_summary"]["acoustic"] == row["emotion_summary"]["visual"] == "未观测"
    assert row["response"] == "本地测试替身的输出，仅用于验证软件流程。"
    assert len(clients.online.calls) == 1
    assert not clients.memory.calls and not clients.reflection.calls
    assert not app.state()["profile"]["hypotheses"]
    app.seal(sid)
    app.close()


def test_low_mood_refusal_and_story_text_do_not_create_care_permission_or_actions(tmp_path):
    app, clients = live_fixture(tmp_path)
    app.care.bootstrap("nurse", "only-local-fixture-password", ["fictional-chen"])
    sid = start_and_submit(app, [
        "我今天有点难过，但不想通知女儿。",
        "我现在只想安静坐着，不想参加活动。",
        "先别设置提醒，也不要替我报名。",
    ])
    view = app.care_state()
    assert all(view[key] == [] for key in ("reminders", "consents", "deliveries", "tasks"))
    assert len(clients.online.calls) == 3
    assert not clients.memory.calls and not clients.reflection.calls
    assert not app.state()["profile"]["hypotheses"]
    app.seal(sid)
    app.close()


def test_fictional_web_rejects_external_care_assembly(tmp_path):
    care = MinimalCareService(tmp_path / "care.sqlite3", allow_external=True)
    try:
        with pytest.raises(ValueError, match="只允许本地"):
            MinimalWebApp(tmp_path / "app", care_service=care)
    finally:
        care.close()
