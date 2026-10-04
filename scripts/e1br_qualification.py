"""Bind E1b-R primary evidence while reporting reflection separately."""
from collections import Counter
from pathlib import Path

from aria.bench.e1b_cases import load_e1b_cases
from scripts import e1b_evaluation as evaluation
from scripts.run_e1b import RunError, SHA, byte_sha, text_sha


COUNTS = {"online_fast": 1008, "memory": 24, "reflection": 24}


def reflection_metrics(rows):
    initial = [row["format_attempts"][0]["status"] for row in rows if row["format_attempts"]]
    corrected = [row["format_attempts"][1]["status"] for row in rows if len(row["format_attempts"]) == 2]
    return {"planned": 24, "product_successes": sum(row["status"] == "ok" for row in rows),
            "first_body_shape_evaluable": sum(status in {"shape_valid", "shape_failed"} for status in initial),
            "first_shape_failures": initial.count("shape_failed"),
            "first_no_body_or_unusable": 24 - sum(status in {"shape_valid", "shape_failed"} for status in initial),
            "format_corrections": sum(row["format_correction_count"] for row in rows),
            "correction_shape_successes": corrected.count("shape_valid"),
            "corrected_product_successes": sum(row["format_correction_count"] == 1 and row["status"] == "ok" for row in rows),
            "denominator_rule": "first shape failures / first_body_shape_evaluable; absent body is not a shape failure; correction successes / format_corrections"}


def require(value, code):
    if not value:
        raise RunError(code)


def qualify(run_root, frozen, state, *, retained_attempts, retained_summary, logical_attempts):
    attempts = [*retained_attempts, *(row for row in logical_attempts(state)
                                    if row["product_step"] == "initial")]
    require(Counter(row["role"] for row in attempts) == Counter(COUNTS), "e1br_logical_denominator_changed")
    old = {row["request_id"]: row for row in retained_attempts}
    require(all(row == old[row["request_id"]] if row["request_id"] in old
                else row.get("registration_sha256") == frozen["seal_sha256"] for row in attempts),
            "e1br_receipt_registration_changed")
    require(all(row.get("status") == "received" and SHA.fullmatch(row.get("visible_content_sha256", ""))
                for row in attempts if row["role"] != "reflection"), "e1br_primary_content_missing")
    lookup = {(row["operation_id"], row["session_id"]): row for row in attempts if row["role"] == "online_fast"}
    require(len(lookup) == 1008, "e1br_online_identity_duplicate")
    positions, reports, memory, reflection, used = [], {}, [], [], set()
    for case_id in frozen["inputs"]["case_ids"]:
        directory = Path(run_root) / "main" / case_id
        report = evaluation.read_json(directory / "report.json")
        reports[case_id] = byte_sha(directory / "report.json")
        require(report.get("evidence_kind") == "real_model" and report.get("phase") == "main",
                "e1br_report_generation_changed")
        capture = evaluation.read_json(directory / "raw-proposals.json")
        evaluation._validate_raw_capture_v2(capture, directory, case_id, load_e1b_cases().source_sha256)
        require(len(capture["sessions"]) == 2 and all(row["complete"] for row in capture["sessions"]),
                "e1br_two_memory_originals_required")
        for number in (1, 2):
            first_path = directory / f"memory-s{number}.first.json"
            first = evaluation.read_json(first_path)
            attempted = evaluation.read_json(directory / f"memory-s{number}.attempt.json")
            session = attempted["session_id"]
            linked = [row for row in attempts if row["role"] == "memory" and row["session_id"] == session]
            reviewed = [row for row in attempts if row["role"] == "reflection" and row["session_id"] == session]
            require(len(linked) == len(reviewed) == 1 and first["request_hash_kind"] == "actual_wire"
                    and linked[0]["request_sha256"] == first["request_sha256"]
                    and linked[0]["client_return_content_sha256"] == first["visible_response_sha256"]
                    and linked[0]["operation_id"] == reviewed[0]["operation_id"], "e1br_memory_binding_changed")
            memory.append({"case_id": case_id, "session": number, "first_sha256": byte_sha(first_path),
                           "receipt_sha256": evaluation.digest(linked[0]),
                           "reflection_receipt_sha256": evaluation.digest(reviewed[0])})
            product_path = directory / f"reflection-s{number}.product.json"
            product = evaluation.read_json(product_path)
            require(product.get("schema") == "e1br_reflection_product_v1"
                    and product.get("session_id") == session and product.get("capture_error") is None
                    and product.get("updater_raised") is False and product.get("product_status") in {"ok", "failed"},
                    "e1br_reflection_terminal_evidence_missing")
            steps = product["product_steps"]
            require([row["product_step"] for row in steps] in (["initial"], ["initial", "format_correction"])
                    and product["format_correction_count"] == len(steps) - 1,
                    "e1br_reflection_product_steps_changed")
            first_records = []
            for step in steps:
                suffix = "" if step["product_step"] == "initial" else ".correction"
                filename = f"reflection-s{number}{suffix}.first.json"
                require(step["first_path"] == filename and step["first_sha256"] == byte_sha(directory / filename),
                        "e1br_reflection_first_bytes_changed")
                original = evaluation.read_json(directory / filename)
                physical = [row for row in [*retained_attempts, *state["attempts"]] if row["role"] == "reflection"
                            and row["session_id"] == session and row["product_step"] == step["product_step"]]
                bound = [row for row in physical if row["request_id"] == original.get("request_id")]
                require(len(bound) == 1 and not original.get("request_identity_missing")
                        and original["physical_request_ids"] == [row["request_id"] for row in physical]
                        and original["request_sha256"] == bound[0]["request_sha256"]
                        and original.get("visible_response_sha256") == bound[0].get("client_return_content_sha256"),
                        "e1br_reflection_physical_binding_changed")
                first_records.append({"product_step": step["product_step"], "first_sha256": step["first_sha256"],
                                      "physical_receipt_sha256": evaluation.digest(bound[0]),
                                      "physical_attempts": len(physical)})
            require(product["first_sha256"] == steps[0]["first_sha256"], "e1br_initial_reflection_replaced")
            reflection.append({"case_id": case_id, "session": number, "product_sha256": byte_sha(product_path),
                               "status": product["product_status"], "format_correction_count": len(steps) - 1,
                               "format_attempts": product.get("format_attempts", []),
                               "steps": first_records, "secondary_only": True})
        branches = [{"arm": "shared", "repeat": 0, "state": report["shared"]["state"]}, *report["branches"]]
        for branch in branches:
            for index, turn in enumerate(branch["state"]["turns"]):
                identity = turn["turn_id"], turn["session_id"]
                require(identity not in used, "e1br_reply_receipt_reused")
                used.add(identity)
                row = lookup.get(identity)
                require(row is not None and turn.get("status") == "completed", "e1br_reply_missing")
                sha = text_sha(turn["response"])
                delivery = ("rule_only" if turn.get("source") == "rule_safety" else
                            "guard_modified" if turn.get("source") == "rule_protection" else
                            "model_unmodified" if sha == row.get("model_reply_sha256") else "guard_modified")
                positions.append({"case_id": case_id, "arm": branch["arm"], "repeat": branch["repeat"],
                                  "turn": index, "turn_id": identity[0], "session_id": identity[1],
                                  "delivery": delivery, "delivered_reply_sha256": sha,
                                  "receipt_sha256": evaluation.digest(row), "model_reply_eligible": delivery == "model_unmodified"})
    require(set(lookup) == used, "e1br_unbound_online_receipts")
    return {"schema": "e1br_transport_qualification_v1", "stage": "main",
            "registration_sha256": frozen["seal_sha256"], "ledger_id": state["ledger_id"],
            "source_report_sha256": reports, "actual_counts": dict(COUNTS),
            "positions": positions, "memory_originals": memory, "reflection_originals": reflection,
            "reflection_secondary": reflection_metrics(reflection),
            "retained_main_prefix": retained_summary, "retained_first_answers": len(retained_attempts),
            "new_physical_attempts": len(state["attempts"]), "effect_claim_eligible": False,
            "scope": "Primary model content and saved delivery qualified; reflection remains a secondary product metric."}
