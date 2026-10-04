"""Local CPED source types and CSV loading for the holdout protocol.

Original dialogue is never bundled. The operator must confirm current upstream
terms before reading a local source; this module offers no download or model call.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import yaml


_DATASET_MANIFEST = (
    Path(__file__).parent / "datasets" / "cped_v1.yaml"
)


_REQUIRED_COLUMNS = {
    "TV_ID",
    "Dialogue_ID",
    "Utterance_ID",
    "Speaker",
    "Age",
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
    "Sentiment",
    "Emotion",
    "DA",
    "Utterance",
}


CPED_TRAITS = (
    "Neuroticism",
    "Extraversion",
    "Openness",
    "Agreeableness",
    "Conscientiousness",
)


@dataclass(frozen=True)
class CPEDRow:
    tv_id: str
    dialogue_id: str
    utterance_id: str
    speaker: str
    age: str
    traits: dict[str, str]
    sentiment: str
    emotion: str
    dialogue_act: str
    utterance: str
    split: str


@dataclass
class CPEDCorpus:
    rows: list[CPEDRow]
    source_files: list[str]
    sha256: dict[str, str]
    data_license: str
    data_license_status: str
    data_license_source: str
    code_license: str
    code_license_source: str
    license_acknowledged: bool
    repository: str


def load_cped_manifest(
    path: str | Path = _DATASET_MANIFEST,
) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise ValueError("CPED manifest 缺少 files")
    for field in ("data_license", "code_license"):
        value = data.get(field)
        if not isinstance(value, dict) or not value.get("spdx"):
            raise ValueError(f"CPED manifest 缺少 {field}.spdx")
    status = str(data["data_license"].get("status") or "").strip().lower()
    if status not in {"confirmed", "unverified"}:
        raise ValueError("CPED manifest 的 data_license.status 无效")
    return data


def _require_cped_data_license_acknowledgement(
    manifest: dict[str, Any],
    acknowledged: bool,
) -> None:
    data_license = manifest["data_license"]
    status = str(data_license.get("status") or "").strip().lower()
    if status != "confirmed" and not acknowledged:
        raise PermissionError(
            "CPED 数据许可尚未独立确认。请先审阅上游数据、影视台词及媒体"
            "相关条款，再显式传入 "
            "acknowledge_unverified_data_license=True；该确认不授予任何权利。"
        )


def load_cped_corpus(
    source: str | Path | Sequence[str | Path],
    *,
    acknowledge_unverified_data_license: bool = False,
) -> CPEDCorpus:
    """读取一个 CSV、一个包含 split CSV 的目录，或多个 CSV。"""
    manifest = load_cped_manifest()
    _require_cped_data_license_acknowledgement(
        manifest, acknowledge_unverified_data_license
    )
    paths = _resolve_cped_paths(source)
    rows: list[CPEDRow] = []
    fingerprints: dict[str, str] = {}
    for path in paths:
        raw = path.read_bytes()
        fingerprints[path.name] = hashlib.sha256(raw).hexdigest()
        split = path.stem.replace("_split", "")
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(text.splitlines())
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path.name} 缺少 CPED 字段: {sorted(missing)}"
            )
        for raw_row in reader:
            utterance = _normalise_text(raw_row.get("Utterance", ""))
            if not utterance:
                continue
            rows.append(CPEDRow(
                tv_id=str(raw_row["TV_ID"]).strip(),
                dialogue_id=str(raw_row["Dialogue_ID"]).strip(),
                utterance_id=str(raw_row["Utterance_ID"]).strip(),
                speaker=_normalise_text(raw_row["Speaker"]),
                age=str(raw_row["Age"]).strip(),
                traits={
                    trait: str(raw_row[trait]).strip()
                    for trait in CPED_TRAITS
                },
                sentiment=str(raw_row["Sentiment"]).strip(),
                emotion=str(raw_row["Emotion"]).strip(),
                dialogue_act=str(raw_row["DA"]).strip(),
                utterance=utterance,
                split=split,
            ))
    if not rows:
        raise ValueError("没有读到 CPED 对话")
    data_license = manifest["data_license"]
    code_license = manifest["code_license"]
    return CPEDCorpus(
        rows=rows,
        # 公共报告只保留文件名与哈希，避免泄漏运行者本地目录/用户名。
        source_files=[path.name for path in paths],
        sha256=fingerprints,
        data_license=str(data_license.get("spdx") or "NOASSERTION"),
        data_license_status=str(
            data_license.get("status") or "unverified"
        ).lower(),
        data_license_source=str(data_license.get("source") or ""),
        code_license=str(code_license.get("spdx") or "NOASSERTION"),
        code_license_source=str(code_license.get("source") or ""),
        license_acknowledged=(
            str(data_license.get("status") or "").lower() == "confirmed"
            or acknowledge_unverified_data_license
        ),
        repository=str(manifest.get("repository") or ""),
    )


def _resolve_cped_paths(
    source: str | Path | Sequence[str | Path],
) -> list[Path]:
    if isinstance(source, (str, Path)):
        candidate = Path(source)
        if candidate.is_dir():
            paths = [
                candidate / f"{split}_split.csv"
                for split in ("train", "valid", "test")
                if (candidate / f"{split}_split.csv").exists()
            ]
        else:
            paths = [candidate]
    else:
        paths = [Path(item) for item in source]
    if not paths or any(not path.is_file() for path in paths):
        raise FileNotFoundError("找不到 CPED split CSV")
    return paths


def _normalise_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())
