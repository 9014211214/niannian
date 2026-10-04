"""New offline review memory follows existing transparency/deletion boundaries."""
from aria.engine.context import EngineContext
from aria.engine.memory.persona_store import PersonaStore
from aria.engine.memory.transparency import render_profile, render_profile_summary
from aria.engine.model.persona import PersonaProfile, PersonEvidenceReview, EvidenceReviewQuote


def _profile():
    return PersonaProfile(user_id="review_privacy_fixture", evidence_review_ledger=[PersonEvidenceReview(
        id="r_fixture", at="2026-01-01T12:00:00", session_id="private_session",
        applicability="unknown", relation="unknown", evidence_basis="context_only",
        event_kind="unknown", event_id="e_fixture", reason="不能从一次经历定性",
        event_summary="仅用于本人查阅的旧事", disposition="unknown_scope_no_update",
        evidence=[EvidenceReviewQuote(turn_index=0, quote="PRIVATE_REVIEW_QUOTE")],
        event_evidence=[EvidenceReviewQuote(turn_index=0, quote="PRIVATE_REVIEW_QUOTE")],
    )])


def test_review_visible_to_owner_not_turned_into_trait_summary():
    p = _profile()
    summary = render_profile_summary(p)
    assert "1 条有限文字依据" in summary
    assert "PRIVATE_REVIEW_QUOTE" not in summary
    assert "还没有足够证据" in summary
    audit = render_profile(p)
    assert "PRIVATE_REVIEW_QUOTE" in audit
    assert "不是本人事实或授权" in audit


def test_review_follows_profile_encryption_and_full_deletion(tmp_path, monkeypatch):
    monkeypatch.setenv("ARIA_PROFILE_KEY", "synthetic-unit-test-key")
    store = PersonaStore(base_dir=tmp_path)
    p = _profile()
    store.save(p)
    raw = (tmp_path / (p.user_id + ".json")).read_text(encoding="utf-8")
    assert "PRIVATE_REVIEW_QUOTE" not in raw
    restored = store.load(p.user_id)
    assert restored.evidence_review_ledger[0].evidence[0].quote == "PRIVATE_REVIEW_QUOTE"
    assert store.delete(p.user_id)
    assert store.load(p.user_id) is None


def test_offline_observation_ledger_not_added_to_online_prompt():
    from aria.engine.policy.prompt import build_messages
    from aria.scenes.companion_elder.scene import ElderCompanionScene
    p = _profile()
    ctx = EngineContext(user_id=p.user_id, scene_id="companion_elder")
    messages = build_messages(ElderCompanionScene(), p, ctx, "今天天气不错。")
    assert "PRIVATE_REVIEW_QUOTE" not in str(messages)
    assert "evidence_review_ledger" not in str(messages)


def test_web_default_reflection_uses_strict_review_protocol(tmp_path):
    from aria.engine.policy.llm import MockLLMClient
    from aria.runtime import MinimalModelClients
    from aria.voice.minimal import MinimalWebApp
    class ReflectionClient:
        def complete(self, _):
            raise AssertionError("assembly must not make network calls")
    clients = MinimalModelClients(MockLLMClient(), MockLLMClient(), ReflectionClient(), True)
    app = MinimalWebApp(tmp_path, clients=clients)
    try:
        engine = app._engine_for_persona(app._seed_persona("fictional-chen"))
        assert engine.hypothesis_updater.evidence_review_protocol is True
    finally:
        app.close()
