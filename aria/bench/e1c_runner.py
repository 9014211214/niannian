"""E1c integration of the product runner with every authored preference source.

Software rehearsal is executable. A real entry still needs a separately frozen
E1c budget, inherited ledger validation and explicit spending authorization.
"""
from __future__ import annotations

from .e1b_runner import E1bRunner, E1bExperimentApp, strip_a0_preference_sources
from .e1br_runner import ProductReflectionCapture
from .e1c_cases import E1cCase
from .e1c_source import load_first_submission


def strip_a0_all_sources(persona, case: E1cCase, shared_sessions: dict[int, str]):
    """Apply the existing conservative record deletion to EVERY authored source.

    All explicit preferences are removed as in E1b. Any mixed fact/hypothesis/
    relationship record is dropped whole. A1 and original shared profile survive.
    Historical audit is not rewritten. No retrospective text matching is used.
    """
    if (set(shared_sessions) != {1, 2} or any(type(key) is not int for key in shared_sessions)
            or any(type(value) is not str or not value for value in shared_sessions.values())
            or len(set(shared_sessions.values())) != 2):
        raise ValueError("two_distinct_actual_session_ids_required")
    result = persona.model_copy(deep=True)
    removed = {key: [] for key in ("preferences", "facts", "hypotheses", "relationships")}
    per_source = []
    for source in case.preference_sources:
        result, current = strip_a0_preference_sources(
            result, session_id=shared_sessions[source.session], turn_index=source.turn)
        for key, ids in current.items():
            removed[key].extend(item for item in ids if item not in removed[key])
        per_source.append({"session": source.session, "turn": source.turn,
                           "actual_session_id": shared_sessions[source.session], "removed": current})
    return result, {**removed, "per_source": per_source, "source_sha_basis": "first_authored_coordinates"}


class E1cExperimentApp(E1bExperimentApp):
    """Reuse product isolation; keep the experiment notice explicitly E1c."""

    def state(self):
        state = super().state()
        state["experiment"] = "e1c"
        state["notice"] = ("E1c 软件夹具：没有真实模型效果；情绪读数为固定种子合成。"
                           if state["evidence_kind"] == "software_fixture" else
                           "E1c 独立虚构题实验：情绪读数为固定种子合成。")
        return state


class E1cRunner(E1bRunner):
    """Executable engineering path, with no implicit E1c spending authority.

    The product/capture machinery is shared; the case format, experiment ID,
    source seal, A0 contract and report version are independently bound here.
    Paid dispatch stays disabled until a dedicated frozen entry validates the
    final E1b accounts, E1c limits, immutable source and actual receipt ledger.
    """

    runner_version = "e1c_product_runner_v1_unfrozen"
    reflection_capture_class = ProductReflectionCapture

    def __init__(self, *, source_directory, source_receipt_sha256, **kwargs):
        if (kwargs.get("allow_model_calls", False) is not False
                or kwargs.get("evidence_kind", "software_fixture") != "software_fixture"):
            raise PermissionError("e1c_real_entry_requires_separate_frozen_authorization_and_guard")
        self.submission = load_first_submission(
            source_directory, expected_receipt_sha256=source_receipt_sha256,
            allow_software_fixture=True)
        if self.submission.suite.provenance != "software_fixture_only":
            raise PermissionError("formal_holdout_cannot_be_used_as_software_rehearsal")
        self._next_case = 0
        super().__init__(self.submission.suite, **kwargs)
        self._journal({"event": "e1c_source_bound", "experiment": "e1c",
                       "first_receipt_sha256": source_receipt_sha256,
                       "source_sha256": self.suite.source_sha256,
                       "selection": "entire_first_twelve_original_order_no_prescreen",
                       "retained_experimental_answers": 0,
                       "effects": "not_evaluated", "paid_execution_authorized": False})

    def _emotion_plan(self, suite, seed):
        plan = super()._emotion_plan(suite, seed)
        for reading in plan.values():
            reading["source"] = reading["source"].replace("simulated_e1b:", "simulated_e1c:", 1)
        return plan

    def _new_app(self, path):
        app = super()._new_app(path)
        if not isinstance(app, E1cExperimentApp):
            app.close()
            raise ValueError("e1c_requires_explicit_experiment_app")
        return app

    def _a0_profile(self, persona, case, shared_sessions):
        return strip_a0_all_sources(persona, case, shared_sessions)

    def _report_metadata(self, case, phase):
        return {"experiment": "e1c", "suite_version": self.suite.version,
                "source_sha256": self.suite.source_sha256,
                "first_receipt_sha256": self.submission.receipt_sha256,
                "author_provenance_sha256": self.submission.provenance_sha256,
                "case_order": [item.id for item in self.suite.items],
                "preference_sources": [source.model_dump(mode="json") for source in case.preference_sources],
                "preference_source_layout": case.source_layout,
                "planned_case_denominator": 12, "planned_reply_denominator": 216,
                "planned_formation_denominator": 12, "planned_basic_positions": 1056,
                "retained_experimental_answers": 0,
                "selection": "entire_first_twelve_original_order_no_prescreen",
                "capture_protocol": "shared_e1b_raw_v2_and_e1br_reflection_v1",
                "operation_id_protocol": "shared_e1b_transport_session_ids"}

    def run_case(self, case_id, *, phase="main", repeats=3):
        if phase != "main":
            raise ValueError("e1c_has_no_prescreen")
        if self._next_case >= len(self.suite.items) or case_id != self.suite.items[self._next_case].id:
            raise ValueError("e1c_requires_first_author_order_without_replacement")
        # Re-read all first-source bytes before each case; no mutable cached seal.
        current = load_first_submission(self.submission.directory,
            expected_receipt_sha256=self.submission.receipt_sha256, allow_software_fixture=True)
        if current.suite != self.suite:
            raise ValueError("e1c_source_or_cached_suite_changed")
        report = super().run_case(case_id, phase=phase, repeats=repeats)
        self._next_case += 1
        return report


