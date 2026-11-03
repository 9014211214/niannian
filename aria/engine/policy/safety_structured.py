"""Narrow native-schema transport contract for risk-only deletion review."""
from __future__ import annotations

from hashlib import sha256
import json

from .online_structured import (OnlineStructuredRoute, DEEPSEEK_MODEL,
                                validate_release_online_client)
from ..risk.safety_review import (MAX_BODY_BYTES, MAX_SENTENCES, SAFETY_REVIEW_INSTRUCTION,
    SAFETY_REVIEW_VERSION, _unique_object, response_sentences, safety_review_schema)
from ..risk import safety_review_v2


def safety_request_format(version=SAFETY_REVIEW_VERSION, *, owner_sources=False, sentence_count=None):
    if version not in {SAFETY_REVIEW_VERSION, safety_review_v2.SAFETY_REVIEW_VERSION}:
        raise ValueError("invalid_safety_review_version")
    schema = (safety_review_v2.safety_review_schema(owner_sources=owner_sources)
              if version == safety_review_v2.SAFETY_REVIEW_VERSION else safety_review_schema())
    if version == safety_review_v2.SAFETY_REVIEW_VERSION and owner_sources and sentence_count is not None:
        if type(sentence_count) is not int or not 1 <= sentence_count <= MAX_SENTENCES:
            raise ValueError("invalid_safety_review_sentences")
        schema["properties"]["keep_each_span"].update(minItems=sentence_count, maxItems=sentence_count)
    return {"type": "json_schema", "json_schema": {
        "name": "aria_" + version, "strict": True, "schema": schema}}


def validate_safety_structured_client(client):
    route = getattr(client, "safety_structured_route", None)
    from .qwen_max_structured import QwenMaxStructuredRoute, validate_qwen_max_client
    if type(route) is QwenMaxStructuredRoute:
        if client.role != "safety_review" or client.max_completion_tokens != 1024 or client.timeout != 90:
            raise ValueError("invalid_safety_structured_client")
        binding = validate_qwen_max_client(client)
    elif client.model == DEEPSEEK_MODEL or getattr(route, "model", None) == DEEPSEEK_MODEL:
        if client.role != "safety_review" or client.timeout != 90:
            raise ValueError("invalid_safety_structured_client")
        binding = validate_release_online_client(client)
    elif (type(route) is not OnlineStructuredRoute or client.role != "safety_review"
            or client.model != "qwen/qwen3.8-flash" or route.model != client.model
            or route.provider != "alibaba" or not client.bounded_transport
            or client.max_retries != 0 or client.fallback_models or client.reasoning_effort != "none"
            or client.max_completion_tokens != 1024 or client.timeout != 90
            or client.online_return_format != "native" or client.reflection_return_format != "native"):
        raise ValueError("invalid_safety_structured_client")
    else:
        binding = route.binding()
    binding["safety_schema_sha256"] = sha256(json.dumps(safety_request_format(),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    binding["instruction_sha256"] = sha256(SAFETY_REVIEW_INSTRUCTION.encode()).hexdigest()
    binding["v2_safety_schema_sha256"] = sha256(json.dumps(safety_request_format(safety_review_v2.SAFETY_REVIEW_VERSION),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    binding["v2_instruction_sha256"] = sha256(safety_review_v2.SAFETY_REVIEW_INSTRUCTION.encode()).hexdigest()
    if safety_review_v2.uses_owner_sources(client):
        # The allowance binds the schema template. Each outgoing request
        # separately binds the exact verdict count to its validated candidate.
        fmt = safety_request_format(safety_review_v2.SAFETY_REVIEW_VERSION, owner_sources=True)
        binding["source_risk_schema_sha256"] = sha256(json.dumps(fmt, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        binding["source_risk_instruction_sha256"] = sha256(safety_review_v2.SOURCE_REVIEW_INSTRUCTION.encode()).hexdigest()
    return binding


def validate_safety_review_request(messages, *, client=None):
    if (type(messages) is not list or len(messages) != 2
            or messages[0] not in ({"role": "system", "content": SAFETY_REVIEW_INSTRUCTION},
                                  {"role": "system", "content": safety_review_v2.SAFETY_REVIEW_INSTRUCTION},
                                  {"role": "system", "content": safety_review_v2.SOURCE_REVIEW_INSTRUCTION})
            or type(messages[1]) is not dict or set(messages[1]) != {"role", "content"}
            or messages[1]["role"] != "user" or type(messages[1]["content"]) is not str):
        raise ValueError("invalid_safety_review_messages")
    def nonfinite(_):
        raise ValueError("invalid_safety_review_input")
    data = json.loads(messages[1]["content"], object_pairs_hook=_unique_object, parse_constant=nonfinite)
    version = data.get("version") if type(data) is dict else None
    if type(version) is not str:
        raise ValueError("invalid_safety_review_version")
    instruction = {SAFETY_REVIEW_VERSION: SAFETY_REVIEW_INSTRUCTION,
                   safety_review_v2.SAFETY_REVIEW_VERSION: safety_review_v2.SAFETY_REVIEW_INSTRUCTION}.get(version)
    sourced = False
    if version == safety_review_v2.SAFETY_REVIEW_VERSION:
        sourced = uses_source_instruction(messages)
        if client is not None and sourced != safety_review_v2.uses_owner_sources(client):
            raise ValueError("invalid_owner_source_review_route")
        if sourced:
            instruction = safety_review_v2.SOURCE_REVIEW_INSTRUCTION
    if instruction is None or messages[0]["content"] != instruction:
        raise ValueError("invalid_safety_review_version")
    def invalid_protection(key):
        value = data[key]
        if version == safety_review_v2.SAFETY_REVIEW_VERSION and key == "protection_already_delivered" and value is None:
            return False
        if version == safety_review_v2.SAFETY_REVIEW_VERSION and key == "final_protection" and value == "":
            return False
        return type(value) is not str or not value.strip() or len(value) > 4096
    expected_keys = {"version", "utterance", "safety_evidence",
                     "protection_already_delivered", "final_protection", "sentences"}
    if version == safety_review_v2.SAFETY_REVIEW_VERSION:
        expected_keys |= {"candidate", "deletion_unit"}
        if sourced:
            expected_keys.add("owner_sources")
            expected_keys.remove("final_protection")
    if (type(data) is not dict or set(data) != expected_keys
            or type(data["utterance"]) is not str or not data["utterance"].strip()
            or invalid_protection("protection_already_delivered")
            or "final_protection" in data and invalid_protection("final_protection")
            or type(data["safety_evidence"]) is not dict
            or type(data["sentences"]) is not list or not 1 <= len(data["sentences"]) <= MAX_SENTENCES):
        raise ValueError("invalid_safety_review_input")
    sentences = data["sentences"]
    evidence = data["safety_evidence"]
    evidence_keys = {"concern", "subject", "time", "urgency", "quote", "verbatim_source",
                     "disposition", "semantic_verified", "conversation_history"}
    if sourced:
        evidence_keys = {"conversation_history"}
    allowed_evidence_keys = (evidence_keys, evidence_keys | {"delivery_facts"}) if (
        version == safety_review_v2.SAFETY_REVIEW_VERSION) else (evidence_keys,)
    if (set(evidence) not in allowed_evidence_keys or type(evidence["conversation_history"]) is not list
            or any(type(row) is not dict or set(row) != {"role", "content"}
                   or type(row["role"]) is not str or row["role"] not in {"user", "assistant"}
                   or type(row["content"]) is not str
                   for row in evidence["conversation_history"])):
        raise ValueError("invalid_safety_review_evidence")
    if not sourced and (evidence["semantic_verified"] is not False or type(evidence["concern"]) is not str
            or evidence["concern"] not in ({"fall", "medical_emergency", "self_harm", "unsafe_medication", "none"}
                if version == safety_review_v2.SAFETY_REVIEW_VERSION else
                {"fall", "medical_emergency", "self_harm", "unsafe_medication"})
            or type(evidence["verbatim_source"]) is not bool
            or any(value is not None and type(value) is not str for value in (
                evidence["subject"], evidence["time"], evidence["urgency"], evidence["quote"], evidence["disposition"]))):
        raise ValueError("invalid_safety_review_evidence")
    if "owner_sources" in data:
        owners = safety_review_v2.review_owner_texts(data["utterance"], evidence["conversation_history"])
        if data["owner_sources"] != safety_review_v2.owner_source_catalog(owners, compact=True):
            raise ValueError("invalid_safety_review_owner_sources")
    if "delivery_facts" in evidence:
        facts = evidence["delivery_facts"]
        if (type(facts) is not dict or set(facts) != {
                "owner_sources", "preference_records", "current_action_results", "code_status"}
                or type(facts["owner_sources"]) is not list
                or any(type(value) is not str for value in facts["owner_sources"])
                or type(facts["current_action_results"]) is not list
                or type(facts["preference_records"]) is not list or len(facts["preference_records"]) > 4
                or facts["code_status"] is not None and type(facts["code_status"]) is not str):
            raise ValueError("invalid_delivery_review_facts")
        for action in facts["current_action_results"]:
            if (type(action) is not dict or set(action) != {"action", "target", "status"}
                    or any(type(value) is not str for value in action.values())
                    or action["action"] not in {"notification", "reminder"}):
                raise ValueError("invalid_delivery_review_action")
        for record in facts["preference_records"]:
            record_keys = {"status", "description", "applicable"}
            if type(record) is dict and record.get("applicable") is True:
                record_keys |= {"owner_sources", "confirmation"}
            if type(record) is dict and "last_ui_result" in record:
                record_keys.add("last_ui_result")
            if (type(record) is not dict
                    or set(record) != record_keys
                    or record["status"] not in {"proposed", "confirmed", "declined", "withdrawn", "superseded"}
                    or type(record["description"]) is not str
                    or type(record["applicable"]) is not bool
                    or record["applicable"] and record["status"] != "confirmed"
                    or record["applicable"] != ("confirmation" in record)):
                raise ValueError("invalid_delivery_review_preference")
            event = record.get("last_ui_result")
            if "last_ui_result" in record and (
                    type(event) is not dict or set(event) != {"action", "persisted"}
                    or type(event["action"]) is not str
                    or event["action"] not in {"confirm", "decline", "withdraw"}
                    or type(event["persisted"]) is not bool):
                raise ValueError("invalid_delivery_review_preference")
            if record["applicable"] and (
                    type(record["owner_sources"]) is not list
                    or any(type(value) is not str for value in record["owner_sources"])
                    or type(record["confirmation"]) is not dict):
                raise ValueError("invalid_delivery_review_preference")
    if any(type(row) is not dict or set(row) != {"id", "text"}
           or type(row["id"]) is not int or row["id"] != i or type(row["text"]) is not str
           for i, row in enumerate(sentences)):
        raise ValueError("invalid_safety_review_sentences")
    body = "".join(row["text"] for row in sentences)
    if len(body.encode()) > MAX_BODY_BYTES:
        raise ValueError("invalid_safety_review_sentences")
    if version == safety_review_v2.SAFETY_REVIEW_VERSION:
        if data["candidate"] != body or data["deletion_unit"] != (
                safety_review_v2.NATIVE_SPAN_UNIT if sourced else "punctuation_span_v1"):
            raise ValueError("invalid_safety_review_sentences")
        expected = safety_review_v2.response_sentences(body, split_periods=sourced)
    else:
        expected = response_sentences(body)
    if expected != tuple(row["text"] for row in sentences):
        raise ValueError("invalid_safety_review_sentences")
    return len(sentences)


def uses_source_instruction(messages):
    return messages[0] == {"role": "system", "content": safety_review_v2.SOURCE_REVIEW_INSTRUCTION}
