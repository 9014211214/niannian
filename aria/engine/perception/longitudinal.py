"""已有有据观测的个人时间基线。

这一层回答的是“和他自己平时相比是否变化”，不是“这个数是否正常”。档案只
保存 Welford 在线统计量（count/mean/m2），不保存原始声音或逐帧轨迹。
"""
from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any, Mapping, Optional

from ..model.persona import BaselineStat, PersonaProfile
from .signals import is_simulated, unit_interval

_BASELINED_MODALITIES = {"behavioral", "prosodic"}


def enrich_with_personal_baseline(
    persona: Optional[PersonaProfile],
    multimodal: Optional[Mapping[str, Any]],
    *,
    min_samples: int = 3,
) -> Optional[dict[str, Any]]:
    """给数值观测补个人基线差异，并更新聚合统计。

    上游已经给出 ``baseline_delta`` 时尊重上游计算；缺少时才用本地个人基线。
    返回深拷贝，不改调用方对象。差异被压到 [-1, 1]，供状态推理稳定消费。
    """
    if not multimodal:
        return None
    enriched: dict[str, Any] = copy.deepcopy(dict(multimodal))
    if persona is None:
        return enriched

    for modality in _BASELINED_MODALITIES:
        payload = enriched.get(modality)
        if payload is None:
            continue
        for row in _mutable_rows(payload):
            signal = str(row.get("signal") or "").strip()
            value = row.get("value")
            if not signal or isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                continue
            try:
                numeric = float(value)
            except (ValueError, OverflowError):
                continue
            if (
                not math.isfinite(numeric)
                or unit_interval(row.get("quality"), 1.0) < 0.2
                or unit_interval(row.get("confidence"), 1.0) < 0.2
            ):
                continue
            prefix = "simulated:" if is_simulated(row) else ""
            key = f"{prefix}{modality}:{signal}"
            stat = persona.longitudinal_baseline.signals.setdefault(
                key, BaselineStat()
            )
            if not all(math.isfinite(item) for item in (stat.mean, stat.m2)) or stat.m2 < 0:
                row["quality"] = 0.0
                row.pop("baseline_delta", None)
                row["detail"] = "个人基线统计无效，本轮不作为变化证据"
                continue
            if row.get("baseline_delta") is None and stat.count >= min_samples:
                std = math.sqrt(stat.m2 / max(1, stat.count - 1))
                scale = max(abs(stat.mean), std * 2.0, 0.05)
                row["baseline_delta"] = max(
                    -1.0, min(1.0, (numeric - stat.mean) / scale)
                )
                row.setdefault(
                    "detail",
                    f"基于个人历史 {stat.count} 次聚合观测",
                )
            if not _update_stat(stat, numeric):
                row["quality"] = 0.0
                row.pop("baseline_delta", None)
                row["detail"] = "数值超出可安全聚合范围，本轮不作为变化证据"
    return enriched


def _mutable_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if "signal" in payload:
        return [payload]
    rows: list[dict[str, Any]] = []
    for signal, value in list(payload.items()):
        if isinstance(value, dict):
            value.setdefault("signal", str(signal))
            rows.append(value)
        else:
            row = {"signal": str(signal), "value": value}
            payload[signal] = row
            rows.append(row)
    return rows


def _update_stat(stat: BaselineStat, value: float) -> bool:
    """Commit atomically only when finite inputs also yield finite statistics."""
    count = stat.count + 1
    delta = value - stat.mean
    mean = stat.mean + delta / count
    m2 = stat.m2 + delta * (value - mean)
    if not all(math.isfinite(item) for item in (delta, mean, m2)) or m2 < 0:
        return False
    stat.count = count
    stat.mean = mean
    stat.m2 = m2
    stat.updated_at = datetime.now().isoformat(timespec="seconds")
    return True
