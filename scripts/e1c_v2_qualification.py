"""Read-only E1c-v2 integrity and actual-request exposure qualification."""
import hashlib
import json
from pathlib import Path
import re

from scripts import e1b_evaluation as common
from scripts import e1c_v2_execution as entry
from aria.bench.e1c_v2_runner import heldout_submission, project_proposals
from aria.engine.memory.stated_preferences import confirmed_preference_payloads
from aria.engine.model.persona import PersonaProfile
from aria.engine.policy.prompt import _clean_untrusted
from aria.engine.memory.consolidator import _parse_json_payload
from aria.engine.policy.online_json import parse_json_decision
from aria.engine.policy.online_decision import ONLINE_DECISION_VERSION_SAFETY, DETERMINISTIC_MODEL_FAILURE_REPLY


def require(condition, code):
    if not condition:
        raise ValueError(code)


def actual_preferences(payload):
    """Only the trusted system builder's exact JSON data block qualifies."""
    system = payload["messages"][0]
    require(system["role"] == "system", "e1cv2_system_message_missing")
    blocks = re.findall(r'<aria_untrusted_data source="stated_preference_confirmed">\n([^\n]+)\n＜/aria_untrusted_data＞',
                        system["content"])
    require(len(blocks) <= 1, "e1cv2_ambiguous_preference_blocks")
    return json.loads(blocks[0]) if blocks else []


def bind_memory(root, directory, session, attempts):
    number=session["session"]
    first=common.read_json(directory/f"memory-s{number}.first.json")
    product=common.read_json(directory/f"memory-s{number}.product.json")
    attempted=common.read_json(directory/f"memory-s{number}.attempt.json")
    bound=attempts[-1]
    require(all(session.get(k)==v for k,v in first.items())
        and session["product_outcome"]==product
        and session["complete"]==(first["parse_status"]=="present_array" and product["product_status"]=="ok")
        and first["request_hash_kind"]=="actual_wire"
        and first["request_id"]==bound["request_id"] and first["request_sha256"]==bound["request_sha256"]
        and first["physical_request_ids"]==[r["request_id"] for r in attempts]
        and attempted["session_id"]==bound["session_id"], "e1cv2_memory_first_binding_changed")
    if first["visible_response_sha256"] is not None:
        require(first["visible_response_sha256"]==bound.get("client_return_content_sha256"),
                "e1cv2_memory_visible_binding_changed")
        raw=common.read_json(root/bound["visible_original_path"])["content"].strip()
        status, value, present, dropped="parse_error",None,None,0
        try:
            parsed=_parse_json_payload(raw)
            present=isinstance(parsed,dict) and "stated_preferences" in parsed
            value=parsed.get("stated_preferences") if isinstance(parsed,dict) else None
            value,dropped=project_proposals(value)
            status="missing_field" if not present else "null_field" if value is None else "present_array" if isinstance(value,list) else "not_array"
        except (ValueError,TypeError,RecursionError):
            pass
        require(first["parse_status"]==status and first["field_present"]==present
            and first["raw_proposals"]==value and first["public_projection_dropped_fields"]==dropped,
            "e1cv2_memory_projection_changed_from_first_body")
    else:
        require(first["parse_status"]=="call_failed_or_unknown" and first["raw_proposals"] is None,
                "e1cv2_missing_memory_cannot_become_candidates")


def bind_reflection(directory, number, session_id, attempts):
    product=common.read_json(directory/f"reflection-s{number}.product.json")
    require(product.get("schema")=="e1br_reflection_product_v1"
        and product.get("session_id")==session_id and product.get("capture_error") is None
        and product.get("updater_raised") is False and product.get("product_status") in {"ok","failed"},
        "e1cv2_reflection_terminal_evidence_missing")
    steps=product["product_steps"]
    require([s["product_step"] for s in steps] in (["initial"],["initial","format_correction"])
        and product["format_correction_count"]==len(steps)-1
        and {r.get("product_step","initial") for r in attempts}=={s["product_step"] for s in steps},
        "e1cv2_reflection_step_inventory_changed")
    for step in steps:
        suffix="" if step["product_step"]=="initial" else ".correction"
        filename=f"reflection-s{number}{suffix}.first.json"
        require(step["first_path"]==filename and step["first_sha256"]==entry.sha(directory/filename),
            "e1cv2_reflection_first_bytes_changed")
        first=common.read_json(directory/filename)
        physical=[r for r in attempts if r.get("product_step","initial")==step["product_step"]]
        bound=physical[-1]
        require(not first.get("request_identity_missing") and first.get("request_id")==bound["request_id"]
            and first["physical_request_ids"]==[r["request_id"] for r in physical]
            and first["request_sha256"]==bound["request_sha256"]
            and first.get("visible_response_sha256")==bound.get("client_return_content_sha256"),
            "e1cv2_reflection_physical_binding_changed")
    require(product["first_sha256"]==steps[0]["first_sha256"],"e1cv2_reflection_initial_replaced")


def bind_online(root, turn, attempts, *, operation_failed):
    """Failed first bodies cannot be relabelled successful model delivery."""
    if turn["source"] == "rule_fallback":
        require(turn["status"] == "failed" and operation_failed
            and turn["response"] == DETERMINISTIC_MODEL_FAILURE_REPLY,
            "e1cv2_fallback_requires_failed_product_operation")
        # Older Product guards did not save a preflight rejection event. The
        # durable failed turn proves no request was recorded, not its cause.
        return "rule_fallback" if attempts else "local_pre_send_failure"
    if not attempts:
        require(turn["source"] == "rule_safety" and turn["status"] == "completed"
            and not operation_failed and bool(turn.get("response", "").strip()),
            "e1cv2_no_request_requires_safety_or_bound_fallback")
        return "rule_safety"
    last=attempts[-1]
    parsed=None
    if last.get("visible_original_path") and last["status"]=="received":
        raw=common.read_json(root/last["visible_original_path"])["content"]
        try:
            parse_json_decision(raw,ONLINE_DECISION_VERSION_SAFETY)
            parsed=json.loads(raw)
        except (ValueError,TypeError,KeyError):
            pass
    require(parsed is not None and not operation_failed and turn["status"]=="completed", "e1cv2_bad_body_must_remain_fallback")
    if parsed["safety_concern"]!="none":
        require(turn["source"]=="rule_safety", "e1cv2_semantic_safety_delivery_missing")
        return "rule_safety"
    return "model_unmodified" if turn["response"]==parsed["response"].strip() else "guard_modified"


def preflight_evidence(events, turn):
    """Link a validated guard event; never infer a missing historical reason."""
    tokens = {e["token"] for e in events if e["event"] == "started"
              and e["operation_id"] == turn["turn_id"] and e["session_id"] == turn["session_id"]}
    return [{k: e[k] for k in ("token", "role", "reason", "category", "request_sha256",
                              "wire_bytes", "input_token_plan_bound")}
            for e in events if e["event"] == "request_rejected"
            and e["token"] in tokens and e["role"] == "online_fast"]


def qualify(campaign, *, authorization_sha256, write=True):
    root = Path(campaign).resolve()
    frozen, original = entry.verify_authority(root, authorization_sha256)
    source = heldout_submission(original)
    receipt = common.read_json(root / "execution-receipt.json")
    require(receipt["receipt_sha256"] == common.digest({k:v for k,v in receipt.items() if k != "receipt_sha256"})
        and receipt["status"] == "generated" and receipt["complete_cases"] == 10
        and receipt["registration_sha256"] == frozen["seal_sha256"]
        and receipt["authorization_sha256"] == authorization_sha256
        and receipt["inherited_finance"] == frozen["finance"]
        and receipt["mode"] == frozen["mode"], "e1cv2_complete_receipt_required")
    inventory_path=root/"generation-files.json"
    require(entry.sha(inventory_path)==receipt["generation_inventory_sha256"],"e1cv2_generation_manifest_changed")
    for relative, expected in common.read_json(inventory_path).items():
        file=entry.regular(root/relative)
        require(file.resolve().is_relative_to(root) and entry.sha(file)==expected,"e1cv2_generated_original_changed")
    state = receipt["accounting"]
    ledger_sha = entry.guard_code.closed_ledger(root, frozen, state)
    require(not state["stopped"], "e1cv2_complete_run_retains_stop")
    run = Path(receipt["run_root"]).resolve()
    require(run.is_relative_to(root), "e1cv2_run_outside_campaign")
    rows = state["attempts"]
    starts={e["token"]:e["operation_id"] for e in state["product_events"] if e["event"]=="started"}
    outcomes={starts[e["token"]]:e["failed"] for e in state["product_events"] if e["event"]=="finished"}
    registered = entry.guard_code.positions(source.suite)
    require(all(tuple(r["experiment_position"]) in registered for r in rows), "e1cv2_position_not_registered")
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row["experiment_position"]), []).append(row)
    reports, captures, positions, exposure = {}, {}, [], []
    files = {}
    scored_fallbacks = 0
    for case in source.suite.items:
        directory = run / "main" / case.id
        report = common.read_json(directory / "report.json")
        require(report.get("experiment") == "e1c_v2" and report["case_id"] == case.id
            and report["source_sha256"] == source.suite.source_sha256
            and report["first_receipt_sha256"] == source.receipt_sha256
            and report["case_order"] == frozen["case_ids"]
            and report["planned_reply_denominator"] == 180 and report["planned_formation_denominator"] == 10
            and report["planned_basic_positions"] == 880, "e1cv2_report_source_changed")
        branches = [{"arm":"shared", "repeat":0, "state":report["shared"]["state"]}, *report["branches"]]
        require(len(branches) == 7 and {(b["arm"],b["repeat"]) for b in branches} ==
            {("shared",0), *((a,r) for a in ("A0","A1") for r in (1,2,3))}, "e1cv2_branch_inventory_changed")
        profile = PersonaProfile.model_validate(report["shared"]["profile"])
        expected_preferences = _clean_untrusted(confirmed_preference_payloads(profile))
        for branch in branches:
            turns = branch["state"]["turns"]
            branch_path="shared" if branch["arm"]=="shared" else f"repeat-{branch['repeat']}-{branch['arm']}"
            durable=common.read_json(directory/branch_path/"live/run.json")
            require(len(turns) == (24 if branch["arm"] == "shared" else 10), "e1cv2_turn_inventory_changed")
            require(len(durable["turns"])==len(turns)
                and all(all(view.get(k)==saved.get(k) for k in ("turn_id","session_id","text","response","status","source"))
                        for view,saved in zip(turns,durable["turns"])), "e1cv2_delivery_differs_from_durable_product")
            for index, turn in enumerate(turns):
                session, number = (index//12+1,index%12) if branch["arm"] == "shared" else (3,index)
                require(turn["text"] == case.current_input(session,number)
                    and bool(turn.get("response")) and (turn.get("status") == "completed"
                    or turn.get("status") == "failed" and turn.get("source") == "rule_fallback"),
                    "e1cv2_actual_delivery_missing")
                key = ("online_fast",case.id,session,branch["arm"],branch["repeat"],number)
                attempts = grouped.get(key, [])
                for attempt in attempts:
                    require(attempt["operation_id"] == turn["turn_id"] and attempt["session_id"] == turn["session_id"],
                        "e1cv2_turn_receipt_binding_changed")
                    payload = json.loads((root / attempt["wire_request_path"]).read_bytes())
                    require(turn["text"] in payload["messages"][-1]["content"], "e1cv2_actual_wire_input_missing")
                    if session == 3:
                        injected = actual_preferences(payload)
                        require(injected == (expected_preferences if branch["arm"] == "A1" else []),
                            "e1cv2_arm_actual_prompt_mismatch")
                require(turn["turn_id"] in outcomes,"e1cv2_product_operation_event_missing")
                delivery=bind_online(root,turn,attempts,operation_failed=outcomes[turn["turn_id"]])
                rejections=preflight_evidence(state["product_events"],turn)
                require(not rejections or delivery in {"local_pre_send_failure", "rule_fallback"},
                        "e1cv2_preflight_rejection_cannot_be_successful_delivery")
                positions.append({"position":list(key), "request_ids":[r["request_id"] for r in attempts],
                    "delivery_source":turn["source"], "delivery_class":delivery,
                    "pre_send_rejections":rejections,
                    "missing_reason":("recorded_pre_send_rejection" if rejections else
                        "local_failure_before_request_reason_unrecorded") if delivery=="local_pre_send_failure" else None,
                    "delivered_response_sha256":hashlib.sha256(turn["response"].encode()).hexdigest()})
                if session == 3 and branch["arm"] == "A1":
                    exposure.append({"position":list(key), "exposed":bool(attempts and expected_preferences),
                        "physical_attempt_count":len(attempts)})
                if session == 3 and number >= 7:
                    scored_fallbacks += turn["source"] == "rule_fallback"
        capture = common.read_json(directory / "raw-proposals.json")
        require(capture["source_sha256"] == source.suite.source_sha256 and capture["case_id"] == case.id
            and [s["session"] for s in capture["sessions"]] == [1,2], "e1cv2_memory_capture_missing")
        for session in capture["sessions"]:
            number = session["session"]
            first = directory / f"memory-s{number}.first.json"
            product = directory / f"memory-s{number}.product.json"
            require(entry.sha(first) == session["first_attempt_sha256"]
                and entry.sha(product) == session["product_outcome_sha256"], "e1cv2_memory_original_changed")
            memory_key = ("memory",case.id,number,"shared",0,None)
            require(bool(grouped.get(memory_key)), "e1cv2_memory_request_missing")
            require(session["request_id"] == grouped[memory_key][-1]["request_id"], "e1cv2_memory_binding_changed")
            bind_memory(root,directory,session,grouped[memory_key])
            reflection_key = ("reflection",case.id,number,"shared",0,None)
            attempts = grouped.get(reflection_key, [])
            if not attempts:
                require(session["product_outcome"]["product_status"] == "failed"
                    and any(a.session_id == grouped[memory_key][-1]["session_id"] and a.outcome == "memory_failed"
                            for a in profile.offline_session_audit), "e1cv2_unexplained_reflection_missing")
            else:
                bind_reflection(directory,number,grouped[memory_key][-1]["session_id"],attempts)
            positions.extend({"position":list(k), "request_ids":[r["request_id"] for r in grouped.get(k,[])],
                "missing_reason":None if grouped.get(k) else "product_sealed_failed_memory"}
                for k in (memory_key,reflection_key))
        reports[case.id] = entry.sha(directory / "report.json")
        captures[case.id] = capture
        for file in directory.rglob("*.json"):
            files[str(file)] = entry.sha(file)
    require(len(positions)==880 and len(exposure)==300, "e1cv2_fixed_inventory_incomplete")
    numerator = sum(r["exposed"] for r in exposure)
    result = {"schema":"e1cv2_generation_qualification_v1", "status":"qualified",
        "evidence_kind":frozen["mode"], "source_sha256":source.suite.source_sha256,
        "first_receipt_sha256":source.receipt_sha256, "case_ids":frozen["case_ids"],
        "positions":positions, "raw_captures":captures, "source_report_sha256":reports,
        "source_files":files, "exposure":{"numerator":numerator,"denominator":300,"rate":numerator/300,
            "gate":"pass" if numerator>=240 else "not_evaluable", "positions":exposure},
        "format":{"fallbacks":scored_fallbacks,"denominator":180,"rate":scored_fallbacks/180,
            "gate":"pass" if scored_fallbacks<=9 else "fail"},
        "fixed_logical_positions":880,"fixed_reply_positions":180,"fixed_formation_cases":10,
        "registration_sha256":frozen["seal_sha256"], "authorization_sha256":authorization_sha256,
        "ledger_sha256":ledger_sha,"execution_receipt_sha256":entry.sha(root/"execution-receipt.json"),
        "evaluation_protocol":frozen["evaluation_protocol"],"evaluation_protocol_sha256":frozen["evaluation_protocol_sha256"],
        "finance":frozen["finance"],"effect_claim_eligible":False}
    result["qualification_digest"] = common.digest(result)
    if write:
        common.write_once(root/"qualification.json",result)
    return result
