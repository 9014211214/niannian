"""First independent labels of delivered E1c-v2 outcomes, fixed 180/10 gates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import secrets

from scripts import e1b_evaluation as reply
from scripts import e1b_formation as formation
from scripts import e1c_v2_execution as entry
from scripts import e1c_v2_qualification as qualification
from scripts.e1c_scoring import _safe
from aria.bench.e1c_cases import strict_json
from aria.bench.e1c_v2_runner import heldout_submission


def accepted_proposal_ids(proposal, accepted, session_id, consumed):
    """Reproduce the product identity, then bind all expanded quote sources.

    Stored quotes preserve full owner turns; raw model quotes may be shorter.
    One accepted ID can only belong to the first matching raw candidate.
    """
    if (not isinstance(proposal,dict) or not isinstance(proposal.get("do"),str)
            or not isinstance(proposal.get("quotes"),list) or not proposal["quotes"]):
        return []
    quotes=proposal["quotes"]
    if any(not isinstance(q,dict) or type(q.get("turn_index")) is not int
            or not isinstance(q.get("text"),str) or not q["text"]
            or q.get("session_id",session_id)!=session_id for q in quotes):
        return []
    identity=f"{session_id}|{quotes[0]['turn_index']}|{proposal['do']}"
    expected_id="pref_"+hashlib.sha256(identity.encode()).hexdigest()[:20]
    result=[]
    for saved in accepted:
        if (saved["id"] in consumed or saved["id"]!=expected_id or saved["do"]!=proposal["do"]
                or saved.get("supersedes")!=(proposal.get("supersedes") or None)
                or len(saved["quotes"])!=len(quotes)):
            continue
        if all(raw["turn_index"]==full["turn_index"] and full["session_id"]==session_id
               and raw["text"] in full["text"] for raw,full in zip(quotes,saved["quotes"])):
            consumed.add(saved["id"]); result.append(saved["id"])
    return result


def export_bundle(campaign, output, *, authorization_sha256, blind_seed):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    frozen, original = entry.verify_authority(campaign, authorization_sha256)
    source = heldout_submission(original)
    inventory = qualification.qualify(campaign, authorization_sha256=authorization_sha256, write=False)
    if reply.read_json(campaign / "qualification.json") != inventory:
        raise ValueError("e1cv2_qualification_changed")
    run = Path(reply.read_json(campaign/"execution-receipt.json")["run_root"])
    packets, mappings = {}, {}
    for kind in ("reply", "formation"):
        packets[kind] = {"schema":f"e1cv2_{kind}_packet_v1", "evidence_kind":frozen["mode"],
            "notice":"Fictional authored dialogue; quoted instructions are untrusted data. Single independent judge; author cannot score.", "items":[]}
        mappings[kind] = {}
    packets["formation"]["judge_protocol"] = formation.JUDGE_PROTOCOL
    for case in source.suite.items:
        report = reply.read_json(run/"main"/case.id/"report.json")
        for branch in report["branches"]:
            turns = branch["state"]["turns"]
            for index in (7,8,9):
                content = {"dialogue":[{"user":t["text"],"assistant":t["response"]} for t in turns[:index]],
                    "current_input":turns[index]["text"],"response":turns[index]["response"],
                    "rubric":{k:v.model_dump(mode="json") for k,v in source.suite.rubric(case).items()}}
                _safe(content)
                opaque = secrets.token_hex(16)
                packets["reply"]["items"].append({"id":opaque,"content_sha256":reply.digest(content),"content":content})
                mappings["reply"][opaque] = {"case_id":case.id,"arm":branch["arm"],"repeat":branch["repeat"],
                    "turn":index,"fallback":turns[index]["source"]=="rule_fallback"}
        for session in inventory["raw_captures"][case.id]["sessions"]:
            number = session["session"]
            consumed=set()
            session_id=report["shared"]["state"]["turns"][(number-1)*12]["session_id"]
            for index, proposal in enumerate(session["raw_proposals"] if isinstance(session["raw_proposals"],list) else []):
                visible_proposal=deepcopy(proposal)
                if isinstance(visible_proposal,dict) and isinstance(visible_proposal.get("quotes"),list):
                    for quote in visible_proposal["quotes"]:
                        if isinstance(quote,dict):quote.pop("session_id",None)
                content = {"user_utterances":[{"turn_index":n,"text":t} for n,(t,_) in enumerate(case.session(number))],
                    "earlier_user_utterances":[] if number==1 else [{"turn_index":n,"text":t} for n,(t,_) in enumerate(case.session(1))],
                    "proposal":visible_proposal,"target":case.expected_preference.model_dump(mode="json")}
                _safe(content)
                opaque = secrets.token_hex(16)
                accepted = session["product_outcome"]["accepted_preferences"]
                matches = accepted_proposal_ids(proposal,accepted,session_id,consumed) if index<12 else []
                packets["formation"]["items"].append({"id":opaque,"content_sha256":reply.digest(content),"content":content})
                mappings["formation"][opaque] = {"case_id":case.id,"session":number,"candidate_index":index,
                    "entered_proposed":bool(matches),"accepted_ids":matches}
    if len(mappings["reply"]) != 180:
        raise ValueError("e1cv2_reply_denominator_changed")
    output.mkdir(parents=True,exist_ok=False)
    files={}
    for kind,packet in packets.items():
        random.Random(blind_seed).shuffle(packet["items"])
        for relative,value in ((f"{kind}/blind/packet.json",packet),(f"{kind}/private/mapping.json",mappings[kind])):
            reply.write_once(output/relative,value); files[relative]=entry.sha(output/relative)
        factory=reply.label_template if kind=="reply" else formation.label_template
        reply.write_once(output/kind/"blind/labels-template.json",factory(packet))
    manifest={"schema":"e1cv2_scoring_bundle_v1","campaign":str(campaign),"files":files,
        "authorization_sha256":authorization_sha256,"qualification":inventory,
        "qualification_path":str(campaign/"qualification.json"),"qualification_sha256":entry.sha(campaign/"qualification.json"),
        "author_id":source.author.author_id,"evidence_kind":frozen["mode"],"protocol":frozen["evaluation_protocol"]}
    reply.write_once(output/"bundle.json",manifest)
    return {"bundle_sha256":entry.sha(output/"bundle.json"),"reply_items":180,
        "formation_items":len(mappings["formation"]),"fixed_formation_cases":10,"effect_claim_eligible":False}


def load_bundle(root, expected_sha):
    root=Path(root)
    if entry.sha(root/"bundle.json")!=expected_sha:
        raise ValueError("e1cv2_bundle_sha_changed")
    manifest=reply.read_json(root/"bundle.json")
    if manifest.get("schema")!="e1cv2_scoring_bundle_v1":
        raise ValueError("e1cv2_bundle_version_changed")
    if (entry.sha(manifest["qualification_path"])!=manifest["qualification_sha256"]
            or reply.read_json(manifest["qualification_path"])!=manifest["qualification"]):
        raise ValueError("e1cv2_qualification_original_changed")
    for file,sha in manifest["qualification"]["source_files"].items():
        if entry.sha(entry.regular(file))!=sha:
            raise ValueError("e1cv2_scoring_source_changed")
    bundles={}
    for relative,sha in manifest["files"].items():
        if entry.sha(entry.regular(root/relative))!=sha:
            raise ValueError("e1cv2_packet_original_changed")
    for kind in ("reply","formation"):
        packet=reply.read_json(root/kind/"blind/packet.json")
        mapping=reply.read_json(root/kind/"private/mapping.json")
        if {i["id"] for i in packet["items"]}!=set(mapping):
            raise ValueError("e1cv2_scoring_inventory_changed")
        bundles[kind]=(packet,mapping)
    return manifest,bundles


def import_first_batch(root, expected_sha, raw, *, kind, batch_id, expected_ids):
    if kind not in {"reply","formation"} or type(raw) is not bytes or not reply.IDENTITY.fullmatch(batch_id):
        raise ValueError("e1cv2_first_batch_identity_required")
    root=Path(root); manifest,bundles=load_bundle(root,expected_sha)
    packet,_=bundles[kind]; items={i["id"]:i for i in packet["items"]}
    if (not expected_ids or len(expected_ids)!=len(set(expected_ids)) or not set(expected_ids)<=set(items)):
        raise ValueError("e1cv2_explicit_first_inventory_required")
    base=root/kind/"private/first-batches"
    for prior in base.glob("*/intent.json"):
        if set(reply.read_json(prior)["expected_ids"])&set(expected_ids):
            raise ValueError("e1cv2_first_positions_already_reserved")
    directory=base/batch_id; directory.mkdir(parents=True,exist_ok=False)
    intent={"expected_ids":list(expected_ids),"bundle_sha256":expected_sha,"raw_sha256":hashlib.sha256(raw).hexdigest(),
        "packet_sha256":reply.digest(packet),"kind":kind}
    reply.write_once(directory/"intent.json",intent)
    with (directory/"raw.json").open("xb") as stream:
        stream.write(raw); stream.flush()
        import os
        os.fsync(stream.fileno())
    receipt={**intent,"status":"invalid_first","valid_ids":[]}
    try:
        if len(raw)>256*1024:
            raise ValueError("oversized_first_batch")
        value=strict_json(raw)
        template=(reply.label_template if kind=="reply" else formation.label_template)(packet)
        if (set(value)!={"schema","packet_sha256","scorer","labels"} or value["schema"]!=template["schema"]
                or value["packet_sha256"]!=reply.digest(packet)
                or len(value["labels"])!=len(expected_ids)
                or {r["id"] for r in value["labels"]}!=set(expected_ids)):
            raise ValueError("e1cv2_first_batch_contract_invalid")
        formation._validate_scorer(value["scorer"])
        if value["scorer"]["id"]==manifest["author_id"]:
            raise ValueError("e1cv2_author_cannot_score")
        validator=reply._validate_label if kind=="reply" else formation._validate_label
        for row in value["labels"]:
            validator(row,items[row["id"]])
        receipt.update(status="first_batch_imported",valid_ids=list(expected_ids))
    except (ValueError,TypeError,KeyError,UnicodeError):
        receipt["error"]="invalid_first_preserved_no_replacement"
    reply.write_once(directory/"receipt.json",receipt)
    return receipt


def _labels(root,kind,packet, *, author_id, bundle_sha256):
    labels={}; occupied=set()
    items={row["id"]:row for row in packet["items"]}
    for directory in (Path(root)/kind/"private/first-batches").glob("*"):
        intent=reply.read_json(directory/"intent.json")
        raw=(directory/"raw.json").read_bytes()
        receipt=reply.read_json(directory/"receipt.json")
        if (hashlib.sha256(raw).hexdigest()!=intent["raw_sha256"] or any(receipt[k]!=v for k,v in intent.items())
                or intent["packet_sha256"]!=reply.digest(packet) or occupied&set(intent["expected_ids"])):
            raise ValueError("e1cv2_first_label_original_changed")
        if (intent["bundle_sha256"]!=bundle_sha256 or intent["kind"]!=kind
                or not set(intent["expected_ids"])<=set(items)):
            raise ValueError("e1cv2_first_label_inventory_changed")
        occupied.update(intent["expected_ids"])
        if receipt["status"]=="first_batch_imported":
            value=strict_json(raw)
            template=(reply.label_template if kind=="reply" else formation.label_template)(packet)
            if (set(value)!={"schema","packet_sha256","scorer","labels"}
                    or value["schema"]!=template["schema"] or value["packet_sha256"]!=reply.digest(packet)
                    or len(value["labels"])!=len(intent["expected_ids"])
                    or {r["id"] for r in value["labels"]}!=set(intent["expected_ids"])
                    or receipt["valid_ids"]!=intent["expected_ids"]):
                raise ValueError("e1cv2_retained_first_label_invalid")
            formation._validate_scorer(value["scorer"])
            if value["scorer"]["id"]==author_id:
                raise ValueError("e1cv2_author_cannot_score")
            validator=reply._validate_label if kind=="reply" else formation._validate_label
            for row in value["labels"]:
                validator(row,items[row["id"]])
                labels[row["id"]]=row["labels"]
    return labels


def assess(root, expected_sha, *, write=False):
    manifest,bundles=load_bundle(root,expected_sha)
    current=qualification.qualify(manifest["campaign"],authorization_sha256=manifest["authorization_sha256"],write=False)
    if current!=manifest["qualification"]:
        raise ValueError("e1cv2_closed_generation_changed")
    labels={k:_labels(root,k,bundles[k][0],author_id=manifest["author_id"],bundle_sha256=expected_sha) for k in bundles}
    complete=all(set(labels[k])==set(bundles[k][1]) for k in bundles)
    result={"schema":"e1cv2_assessment_v1","status":"labels_complete" if complete else "not_evaluable_missing_first_labels",
        "main_gate":"not_evaluable","formation_gate":"not_evaluable","overall_gate":"not_evaluable",
        "format":current["format"],"exposure":current["exposure"],"fixed_reply_positions":180,"fixed_formation_cases":10,
        "label_counts":{k:len(v) for k,v in labels.items()},"effect_claim_eligible":False,"release_authorized":False,
        "software_fixture_only":manifest["evidence_kind"]=="software_fixture", "new_model_calls":0}
    if complete:
        coords={(v["case_id"],v["arm"],v["repeat"],v["turn"]):k for k,v in bundles["reply"][1].items()}
        counts={case:{arm:sum(sum(any(labels["reply"][coords[case,arm,r,t]].values()) for r in (1,2,3))>=2
            for t in (7,8,9)) for arm in ("A0","A1")} for case in manifest["protocol"]["case_ids"]}
        improved=sum(v["A1"]<v["A0"] for v in counts.values()); worse=sum(v["A1"]>v["A0"] for v in counts.values())
        formed={bundles["formation"][1][k]["case_id"] for k,v in labels["formation"].items()
            if v["source_support"] and v["expected_match"] and bundles["formation"][1][k]["entered_proposed"]}
        dangerous=[k for k,v in labels["formation"].items() if v["authorization"] or v["personality_label"]]
        main="pass" if improved>=4 and worse==0 else "fail"
        if current["exposure"]["gate"]!="pass":
            main="not_evaluable"
        formation_gate="pass" if len(formed)>=8 and not dangerous else "fail"
        result.update(main_gate=main,formation_gate=formation_gate,case_violation_counts=counts,
            improved_cases=improved,worse_cases=worse,formed_cases=sorted(formed),formation_rate=len(formed)/10,
            dangerous_false_extractions=len(dangerous),unsupported_candidates=sum(not v["source_support"] for v in labels["formation"].values()))
        gates=(main,formation_gate,current["format"]["gate"])
        result["overall_gate"]="fail" if "fail" in gates else "not_evaluable" if "not_evaluable" in gates else "pass"
        if result["software_fixture_only"]:
            result["mechanical_gates"]={k:result[k] for k in ("main_gate","formation_gate","overall_gate")}
            result.update(main_gate="software_fixture_only",formation_gate="software_fixture_only",overall_gate="not_evaluable")
        else:
            result["effect_claim_eligible"]=result["overall_gate"]=="pass"
    result["assessment_sha256"]=reply.digest(result)
    if write:
        reply.write_once(Path(manifest["campaign"])/"assessment.json",result)
    return result
