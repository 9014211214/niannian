"""First E1c author submission: exclusive preservation before any validation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictStr, model_validator

from .e1c_cases import E1cSuite, SHA, Strict, parse_suite, strict_json


class AuthorProvenance(Strict):
    author_id: StrictStr = Field(min_length=1)
    brief_sha256: SHA
    context_mode: Literal["fresh_no_project_history", "software_fixture_only"]
    provided_materials: dict[StrictStr, SHA]
    first_submission: Literal[True]
    no_product_prompts_rules_or_e1b_questions: Literal[True]
    no_outcome_filtering: Literal[True]

    @model_validator(mode="after")
    def only_brief(self):
        if self.provided_materials != {"scope_and_format_brief": self.brief_sha256}:
            raise ValueError("author_material_inventory_changed")
        # Explicit boolean types: JSON 1 is not an attestation.
        return self


def _write_once(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def seal_first_submission(raw: bytes, provenance: dict, directory: Path):
    """Save originals BEFORE format checks. Invalid first output stays sealed.

    This records claims; it cannot prove the author's hidden prior knowledge.
    No mutation, corrected submission, filtered subset or replacement API exists.
    """
    if type(raw) is not bytes:
        raise TypeError("first_submission_bytes_required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write_once(directory / "author-original.json", raw)
    receipt = {"schema": "e1c_first_source_v1", "source_sha256": hashlib.sha256(raw).hexdigest(),
               "provenance_sha256": None, "status": "invalid_first_submission",
               "selection": "entire_first_twelve_original_order_no_prescreen", "execution_authorized": False}
    stage = "author_provenance"
    try:
        # Preserve even invalid submitted claims before validating them. This is
        # their JSON serialization; the author's question bytes above are exact.
        author_raw = json.dumps(provenance, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        _write_once(directory / "author-provenance.json", author_raw)
        receipt["provenance_sha256"] = hashlib.sha256(author_raw).hexdigest()
        author = AuthorProvenance.model_validate(provenance)
        for field in ("first_submission", "no_product_prompts_rules_or_e1b_questions", "no_outcome_filtering"):
            if provenance[field] is not True:
                raise ValueError("author_attestation_must_be_boolean")
        receipt["author_id"] = author.author_id
        stage = "question_structure"
        suite = parse_suite(raw, allow_software_fixture=author.context_mode == "software_fixture_only")
        if (suite.provenance == "software_fixture_only") != (author.context_mode == "software_fixture_only"):
            raise ValueError("source_and_author_provenance_mismatch")
        receipt.update(status="format_valid_not_registered", case_ids=[case.id for case in suite.items],
                       planned_denominator=12, provenance=suite.provenance)
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        # Do not print raw answer/annotation or turn structural failure into 0.
        receipt.update(error="structural_contract_failed", error_stage=stage,
                       action="stop_preserve_first_no_replacement")
    _write_once(directory / "receipt.json", json.dumps(receipt, sort_keys=True).encode("utf-8"))
    return receipt


@dataclass(frozen=True)
class FirstSubmission:
    suite: E1cSuite
    author: AuthorProvenance
    receipt_sha256: str
    provenance_sha256: str
    directory: Path


def _read_regular(path: Path) -> bytes:
    # Check original paths before resolving: a replaced parent is not the seal.
    for member in (path, *path.parents):
        state = member.lstat()
        if member.is_symlink() or getattr(state, "st_file_attributes", 0) & 0x400:
            raise ValueError("first_source_links_forbidden")
    if not path.is_file():
        raise ValueError("first_source_regular_file_required")
    return path.read_bytes()


def load_first_submission(directory: str | Path, *, expected_receipt_sha256: str,
                          allow_software_fixture: bool = False) -> FirstSubmission:
    """Read the exact sealed first set; never accept a corrected replacement."""
    directory = Path(directory).absolute()
    receipt_raw = _read_regular(directory / "receipt.json")
    digest = hashlib.sha256(receipt_raw).hexdigest()
    if digest != expected_receipt_sha256:
        raise ValueError("first_receipt_sha_mismatch")
    receipt = strict_json(receipt_raw)
    if (not isinstance(receipt, dict) or receipt.get("schema") != "e1c_first_source_v1"
            or receipt.get("status") != "format_valid_not_registered"
            or receipt.get("execution_authorized") is not False
            or receipt.get("selection") != "entire_first_twelve_original_order_no_prescreen"):
        raise ValueError("first_submission_not_format_valid")
    raw = _read_regular(directory / "author-original.json")
    author_raw = _read_regular(directory / "author-provenance.json")
    if (hashlib.sha256(raw).hexdigest() != receipt["source_sha256"]
            or hashlib.sha256(author_raw).hexdigest() != receipt["provenance_sha256"]):
        raise ValueError("first_source_sha_mismatch")
    provenance = strict_json(author_raw)
    author = AuthorProvenance.model_validate(provenance)
    if any(provenance[field] is not True for field in (
            "first_submission", "no_product_prompts_rules_or_e1b_questions", "no_outcome_filtering")):
        raise ValueError("author_attestation_must_be_boolean")
    suite = parse_suite(raw, allow_software_fixture=allow_software_fixture)
    if ((suite.provenance == "software_fixture_only") != (author.context_mode == "software_fixture_only")
            or receipt.get("author_id") != author.author_id
            or receipt.get("provenance") != suite.provenance
            or receipt.get("planned_denominator") != 12
            or receipt.get("case_ids") != [case.id for case in suite.items]):
        raise ValueError("first_source_identity_mismatch")
    return FirstSubmission(suite, author, digest, receipt["provenance_sha256"], directory)


