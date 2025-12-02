from __future__ import annotations

from collections.abc import Sequence
from typing import SupportsFloat

__all__ = ["AC3_FRAME_LENGTH", "PCR_CLOCK", "absolute_time_from_timecode"]

# http://www.mpucoder.com/DVD/ass-hdr.html
AC3_FRAME_LENGTH = 2880
PCR_CLOCK = 90_000


def absolute_time_from_timecode(timecodes: Sequence[SupportsFloat]) -> list[float]:
    absolutetime = list[float]([0.0])

    for i, a in enumerate(timecodes):
        absolutetime.append(absolutetime[i] + float(a))

    return absolutetime
