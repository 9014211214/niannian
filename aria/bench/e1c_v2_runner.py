"""E1c-v2: product fallbacks are outcomes, never a replacement first answer.

This separate runner retains the original E1c strict runner unchanged.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import hashlib
import re

from .e1b_runner import (E1bRunner, E1bRunStopped, _ProposalCapture,
    _CapturedReflectionUpdater, _exclusive_json, _proposal_projection,
    _write_json)
from ..engine.memory.consolidator import _parse_json_payload
from .e1c_runner import E1cRunner, E1cExperimentApp
from .e1c_source import load_first_submission
from ..engine.policy.product_calls import ProductCallGuard

EXCLUDED_CASES = frozenset({"e1c_a1", "e1c_a2"})


def project_proposals(value):
    """Keep bounded source identity privately; never erase a wrong session."""
    projected,dropped=_proposal_projection(value)
    if isinstance(value,list) and isinstance(projected,list):
        for raw,saved in zip(value,projected):
            if isinstance(raw,dict) and isinstance(saved,dict) and isinstance(raw.get("quotes"),list):
                for quote,out in zip(raw["quotes"],saved.get("quotes",[])):
                    if isinstance(quote,dict) and isinstance(out,dict) and "session_id" in quote:
                        identity=quote["session_id"]
                        out["session_id"]=(identity if isinstance(identity,str)
                            and re.fullmatch(r"[A-Za-z0-9_-]{1,100}",identity) else "[invalid_session_identity]")
    return projected,dropped


def heldout_submission(submission):
    items = tuple(c for c in submission.suite.items if c.id not in EXCLUDED_CASES)
    if len(items) != 10 or len(submission.suite.items) != 12:
        raise ValueError("e1cv2_fixed_ten_heldout_required")
    return replace(submission, suite=replace(submission.suite, items=items))


class ProductProposalCapture(_ProposalCapture):
    """Observation only: product parser, not the instrument, decides usability."""
    def complete(self, client, messages):
        if self.started:
            raise E1bRunStopped("memory first attempt already consumed")
        self.started = True
        _exclusive_json(self.attempt_path, {"session": self.session, "session_id": self.session_id,
            "status": "started", "source_sha256": self.source_sha256})
        first = {"session": self.session, "request_sha256": None,
            "request_hash_kind": "actual_wire", "visible_response_sha256": None,
            "field_present": None, "raw_proposals": None,
            "parse_status": "call_failed_or_unknown", "public_projection_dropped_fields": 0}
        raw = None
        try:
            raw = client.complete(messages)
            if isinstance(raw, str):
                first["visible_response_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
                try:
                    parsed = _parse_json_payload(raw)
                    first["field_present"] = isinstance(parsed, dict) and "stated_preferences" in parsed
                    value = parsed.get("stated_preferences") if isinstance(parsed, dict) else None
                    first["raw_proposals"], first["public_projection_dropped_fields"] = project_proposals(value)
                    first["parse_status"] = ("missing_field" if not first["field_present"] else "null_field"
                        if value is None else "present_array" if isinstance(value, list) else "not_array")
                except (ValueError, TypeError, RecursionError):
                    first["parse_status"] = "parse_error"
            return raw
        finally:
            rows = [r for r in self.app.call_guard.snapshot()["attempts"]
                    if r["role"] == "memory" and r["session_id"] == self.session_id]
            if rows:
                first.update(request_sha256=rows[-1]["request_sha256"], request_id=rows[-1]["request_id"],
                    physical_request_ids=[r["request_id"] for r in rows])
            _exclusive_json(self.first_path, first)
            self.first = first
            self.publish()


class CapturedProductExtractor:
    def __init__(self, extractor, capture):
        self.extractor, self.capture = extractor, capture
        client = extractor.llm
        class RecordedClient:
            def __getattr__(self, name):
                return getattr(client, name)
            def complete(self, messages):
                return capture.complete(client, messages)
        extractor.llm = RecordedClient()

    def __getattr__(self, name):
        return getattr(self.extractor, name)

    def extract(self, ctx, persona):
        if ctx.session_id != self.capture.session_id:
            raise E1bRunStopped("memory capture session mismatch")
        result = self.extractor.extract(ctx, persona)
        self.capture.finish(self.extractor, ctx, result)
        return result


class E1cV2ExperimentApp(E1cExperimentApp):
    def __init__(self, *args, software_fixture_only=False, **kwargs):
        self.software_fixture_only = software_fixture_only
        super().__init__(*args, **kwargs)

    def state(self):
        state = super().state()
        if self.software_fixture_only:
            state.update(evidence_kind="software_fixture", notice="E1c-v2 软件传输夹具；没有真实模型调用或效果。")
            for row in state["turns"]:
                row["evidence_kind"] = "software_fixture"
        return state

    def _engine_for_persona(self, persona):
        from aria.runtime import create_minimal_engine
        from aria.scenes.registry import get_scene
        engine = create_minimal_engine(get_scene(self.scene_id), persona, clients=self.clients,
            store=self.store, delivery_block=self.delivery_block,
            external_emotion_experiment=self.experiment)
        if self.proposal_capture is not None:
            engine._consolidator.extractor = CapturedProductExtractor(engine._consolidator.extractor, self.proposal_capture)
        if self.reflection_capture is not None and engine.hypothesis_updater is not None:
            engine.hypothesis_updater = _CapturedReflectionUpdater(engine.hypothesis_updater, self.reflection_capture)
            engine.diagnoser = engine.hypothesis_updater
        return engine


class E1cV2Runner(E1cRunner):
    runner_version = "e1c_v2_product_fallback_v1"

    def __init__(self, submission, **kwargs):
        self.original_submission = submission
        self.submission = heldout_submission(submission)
        self._next_case = 0
        E1bRunner.__init__(self, self.submission.suite, **kwargs)

    def _operation(self, app, name, operation, *, detail=None):
        if self.stopped:
            raise E1bRunStopped("run already stopped")
        self._sequence += 1
        attempt = {"id": self._sequence, "operation": name, **(detail or {})}
        self._journal({**attempt, "event": "pending"})
        guard = app.call_guard
        if not isinstance(guard, ProductCallGuard):
            raise E1bRunStopped("e1cv2_requires_product_recovery")
        scope = guard.experiment_position(name, detail) if name in {"submit", "reflect"} else nullcontext()
        try:
            if self.gate:
                self.gate(deepcopy(attempt))
            self._bind_shared_guard(app)
            with scope:
                result = operation()
            if isinstance(result, dict) and result.get("ok") is False:
                state = guard.snapshot()
                # Failure continuation is limited to the actual product's
                # settled fallback/sealed review, never storage/unknown fees.
                if name == "submit":
                    latest = app.state()["turns"][-1]
                    allowed = (latest.get("source") == "rule_fallback" and bool(latest.get("response")))
                elif name == "reflect":
                    profile = app._profile()
                    allowed = not profile.offline_sessions
                else:
                    allowed = False
                if not allowed or state["stopped"]:
                    raise E1bRunStopped("e1cv2_product_pause_or_hard_stop")
                self._journal({**attempt, "event": "product_failure_delivered", "result_ok": False})
            else:
                self._journal({**attempt, "event": "completed"})
            return result
        except Exception as error:
            self.stopped = True
            self._journal({**attempt, "event": "stopped", "error_type": type(error).__name__})
            # Do not convert a recoverable product pause into an opaque halt.
            # Execution has no automatic resume; the originals remain closed.
            raise E1bRunStopped("e1cv2 operation stopped; inspect original product events") from error

    def _report_metadata(self, case, phase):
        data = super()._report_metadata(case, phase)
        data.update(experiment="e1c_v2", planned_case_denominator=10,
            evidence_kind="software_fixture" if self.suite.provenance == "software_fixture_only" else "real_model",
            planned_reply_denominator=180, planned_formation_denominator=10,
            planned_basic_positions=880, excluded_development_cases=sorted(EXCLUDED_CASES),
            selection="original_order_without_a1_a2_no_prescreen",
            failure_policy="actual_product_fallback_no_body_retry_no_automatic_pause_resume")
        return data

    def run_case(self, case_id, *, phase="main", repeats=3):
        if (phase != "main" or self._next_case >= 10
                or case_id != self.suite.items[self._next_case].id):
            raise ValueError("e1cv2_original_heldout_order_required")
        current = load_first_submission(self.original_submission.directory,
            expected_receipt_sha256=self.original_submission.receipt_sha256, allow_software_fixture=True)
        if current != self.original_submission:
            raise ValueError("e1cv2_first_source_changed")
        result = self._run_product_case(case_id, phase=phase, repeats=repeats)
        self._next_case += 1
        return result

    def _run_product_case(self, case_id: str, *, phase: str = "main", repeats: int = 3) -> dict:
        if phase not in {"main", "prescreen"} or type(repeats) is not int or repeats != 3:
            raise ValueError("registered layout requires main/prescreen and exactly three repeats")
        case = next((item for item in self.suite.items if item.id == case_id), None)
        if case is None:
            raise ValueError("unknown case")
        case_root = self.root / phase / case.id
        case_root.mkdir(parents=True, exist_ok=False)
        report = {"version": self.runner_version, "case_id": case.id, "phase": phase,
                  "evidence_kind": self.evidence_kind, "synthetic_emotion": True,
                  "frozen": False, "effects": "not_evaluated", "shared": None,
                  "branches": [], "scored_reply_count": 0}
        report.update(self._report_metadata(case, phase))
        shared_profile, shared_sessions = None, {}
        if phase == "main":
            app = self._new_app(case_root / "shared")
            try:
                for number in (1, 2):
                    session_id = self._session(app, case, number, arm="shared", repeat=0)
                    app.proposal_capture = ProductProposalCapture(app, case_root, case_id=case.id, session=number,
                        session_id=session_id, source_sha256=self.suite.source_sha256)
                    app.reflection_capture = self.reflection_capture_class(app, case_root, case_id=case.id, session=number,
                        session_id=session_id, source_sha256=self.suite.source_sha256)
                    try:
                        self._operation(app, "reflect", lambda: app.reflect(f"e1b-review-{session_id}", wait=True),
                                        detail={"case_id": case.id, "session": number})
                    finally:
                        app.proposal_capture = None
                        app.reflection_capture = None
                    shared_sessions[number] = session_id
                # Deliberately confirm every proposed item, even wrong ones.
                # Matching an expected answer here would bias the experiment.
                proposals = [row.id for row in app._profile().stated_preferences if row.status == "proposed"]
                for preference_id in proposals:
                    self._operation(app, "confirm_preference", lambda: app.preference_action("confirm", preference_id),
                                    detail={"case_id": case.id, "preference_id": preference_id})
                shared_profile = app._profile().model_copy(deep=True)
                self._assert_profile(shared_profile)
                report["shared"] = {"state": app.state(), "profile": shared_profile.model_dump(mode="json"),
                                    "confirmation_ids": proposals, "memory_review_count": 2,
                                    "reflection_review_count": sum((case_root / f"reflection-s{n}.first.json").exists() for n in (1, 2)),
                                    "review_outcomes": [r.get("status") for r in app._run["reviews"]]}
                self._operation(app, "save_shared", lambda: _write_json(case_root / "shared.json", report["shared"]))
            finally:
                app.close()
        for repeat in range(1, repeats + 1):
            # Balance arm order while keeping all emotion inputs identical.
            arms = ("A0",) if phase == "prescreen" else (("A0", "A1") if (repeat + self.seed) % 2 else ("A1", "A0"))
            for arm in arms:
                app = self._new_app(case_root / f"repeat-{repeat}-{arm}")
                try:
                    removed = {"preferences": [], "facts": [], "hypotheses": [], "relationships": []}
                    if shared_profile is not None:
                        branch_profile = shared_profile.model_copy(deep=True)
                        if arm == "A0":
                            branch_profile, removed = self._a0_profile(
                                branch_profile, case, shared_sessions)
                        self._assert_profile(branch_profile)
                        # Only persona data travels across the fork. No care DB,
                        # cost ledger, active session or app run ID is copied.
                        self._operation(app, "import_profile", lambda: app.store.save(branch_profile))
                    self._session(app, case, 3, arm=arm, repeat=repeat)
                    state = app.state()
                    rows = state["turns"]
                    if len(rows) != 10:
                        raise E1bRunStopped("unexpected third-session turn count")
                    branch = {"arm": arm, "repeat": repeat, "run_id": state.get("run_id"),
                              "removed": removed, "state": state,
                              "scored_turn_ids": [row["turn_id"] for row in rows[7:]],
                              "scored_replies": [{"turn": index, "response": rows[index]["response"],
                                                   "source": rows[index]["source"],
                                                   "evidence_kind": report["evidence_kind"]} for index in (7, 8, 9)]}
                    report["branches"].append(branch)
                    report["scored_reply_count"] += 3
                    self._operation(app, "save_branch", lambda: _write_json(
                        case_root / f"repeat-{repeat}-{arm}.json", branch))
                finally:
                    app.close()
        self._operation(app, "save_case_report", lambda: _write_json(case_root / "report.json", report))
        return report
