"""Explicit, per-turn external readings; never personality or sensor evidence."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from .signals import is_simulated

EXTERNAL_EMOTION_VERSION = "external_emotion_v1"
LABELS = frozenset({"calm", "neutral", "happy", "sad", "anxious", "angry",
                    "tired", "fearful", "disgusted", "surprised", "mixed", "unknown"})
MODALITIES = frozenset({"text", "audio", "visual", "multimodal", "unknown"})
_SOURCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


def parse_external_emotion(value: Any) -> dict | None:
    """Reject malformed input without coercion or echoing caller data in errors."""
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"label", "confidence", "modality", "source"}:
        raise ValueError("external_emotion_invalid_fields")
    if type(value["label"]) is not str or value["label"] not in LABELS:
        raise ValueError("external_emotion_invalid_label")
    confidence = value["confidence"]
    if (type(confidence) not in {float, int} or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise ValueError("external_emotion_invalid_confidence")
    if type(value["modality"]) is not str or value["modality"] not in MODALITIES:
        raise ValueError("external_emotion_invalid_modality")
    if type(value["source"]) is not str or not _SOURCE.fullmatch(value["source"]):
        raise ValueError("external_emotion_invalid_source")
    return {**value, "confidence": float(confidence)}


def external_emotion_payload(reading: dict) -> dict:
    """Bounded audit/prompt payload; modality is claimed, not independently seen."""
    return {**reading, "protocol_version": EXTERNAL_EMOTION_VERSION,
            "inference_source": "external_model", "simulated": is_simulated(reading),
            "boundary": "外部读数，可能有误；不明不等于平静，不是本人原话或人物证据"}


@dataclass(frozen=True)
class SyntheticEmotionExperiment:
    """Python-only capability bound to a marked, separate fictional namespace.

    This is not the old ephemeral simulation mode. Only original dialogue may
    enter its offline queue; the synthetic readings stay in turn audit output.
    A Web payload cannot construct this object or authorize persistent learning.
    """
    root: Path
    experiment_id: str

    @property
    def profile_origin(self) -> str:
        return f"synthetic_emotion_experiment:{self.experiment_id}"

    @classmethod
    def create(cls, parent_dir: str | Path, experiment_id: str):
        if type(experiment_id) is not str or not _EXPERIMENT_ID.fullmatch(experiment_id):
            raise ValueError("external_emotion_invalid_experiment_id")
        root = Path(parent_dir).resolve() / "synthetic-emotion-experiments" / experiment_id
        if root.resolve() != root:
            raise ValueError("external_emotion_experiment_namespace_cannot_redirect")
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "experiment.json"
        expected = {"version": EXTERNAL_EMOTION_VERSION, "experiment_id": experiment_id,
                    "fictional_only": True, "external_actions": False}
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != expected:
                raise ValueError("external_emotion_experiment_marker_mismatch")
        else:
            if any(root.iterdir()):
                raise ValueError("external_emotion_experiment_requires_empty_namespace")
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(expected, stream, sort_keys=True)
        return cls(root=root.resolve(), experiment_id=experiment_id)

    def validate(self, store, persona) -> None:
        if persona is None or persona.profile_origin != self.profile_origin or store is None:
            raise ValueError("external_emotion_experiment_identity_mismatch")
        # Resolve symlinks/junctions before containment checks on every ingress.
        directory = Path(store._dir).resolve()
        root = self.root.resolve()
        if root != self.root:
            raise ValueError("external_emotion_experiment_namespace_cannot_redirect")
        if directory == root or root not in directory.parents:
            raise ValueError("external_emotion_experiment_store_outside_namespace")
        expected = {"version": EXTERNAL_EMOTION_VERSION, "experiment_id": self.experiment_id,
                    "fictional_only": True, "external_actions": False}
        try:
            marker = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ValueError("external_emotion_experiment_marker_missing") from None
        if marker != expected:
            raise ValueError("external_emotion_experiment_marker_mismatch")


def validate_experiment_context(ctx, store, persona) -> bool:
    """Used by persistence ingress too, so moving a context cannot leak a run."""
    capability = ctx.external_emotion_experiment
    if type(capability) is not SyntheticEmotionExperiment:
        return False
    try:
        capability.validate(store, persona)
    except (ValueError, AttributeError, TypeError, OSError):
        return False
    return True
