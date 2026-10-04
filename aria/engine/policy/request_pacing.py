"""Small pacing calculations; no controller, credentials, I/O or retries."""
from __future__ import annotations

import math


def pacing_interval(value):
    """An explicit local interval, never a claimed provider rate limit."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 3600:
        raise ValueError("invalid_request_start_interval")
    return float(value)


def monotonic_value(clock):
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid_pacing_clock")
    return float(value)


def wait_until(deadline, *, monotonic, sleep, cancelled=lambda: False):
    """Recheck early wakes; a broken/frozen clock fails without any send.

    Dependencies are explicit so software checks need neither real sleeps nor
    a transport. The caller owns serialization and revalidates its allowance.
    """
    if cancelled():
        raise ValueError("pacing_cancelled")
    started = current = monotonic_value(monotonic)
    if deadline is None:
        return 0.0
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline < 0:
        raise ValueError("invalid_pacing_deadline")
    for _ in range(4096):
        if cancelled():
            raise ValueError("pacing_cancelled")
        remaining = deadline - current
        if remaining <= 0:
            return (current - started) * 1000
        sleep(min(remaining, 1.0))
        advanced = monotonic_value(monotonic)
        if advanced <= current:
            raise ValueError("pacing_clock_did_not_advance")
        current = advanced
    raise ValueError("pacing_wait_did_not_complete")
