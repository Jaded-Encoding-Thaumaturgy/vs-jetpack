from __future__ import annotations

import subprocess
from typing import Sequence, SupportsFloat

from vstools import SupportsString

__all__ = ["AC3_FRAME_LENGTH", "PCR_CLOCK", "absolute_time_from_timecode", "double_check_dvdnav"]

# http://www.mpucoder.com/DVD/ass-hdr.html
AC3_FRAME_LENGTH = 2880
PCR_CLOCK = 90_000


# d2vwitch needs this patch applied
# https://gist.github.com/jsaowji/ead18b4f1b90381d558eddaf0336164b


# https://gist.github.com/jsaowji/2bbf9c776a3226d1272e93bb245f7538
def double_check_dvdnav(iso: SupportsString, title: int) -> list[float] | None:
    try:
        ap = subprocess.check_output(["dvdsrc_dvdnav_title_ptt_test", str(iso), str(title)])

        return list(map(float, ap.splitlines()))
    except FileNotFoundError:
        ...

    return None


def absolute_time_from_timecode(timecodes: Sequence[SupportsFloat]) -> list[float]:
    absolutetime = list[float]([0.0])

    for i, a in enumerate(timecodes):
        absolutetime.append(absolutetime[i] + float(a))

    return absolutetime
