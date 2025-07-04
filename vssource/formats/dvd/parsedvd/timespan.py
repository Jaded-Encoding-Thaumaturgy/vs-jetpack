from __future__ import annotations

from dataclasses import dataclass

from vstools import Region

__all__ = ["VTS_FRAMERATE", "TimeSpan"]


@dataclass
class TimeSpan:
    hour: int
    minute: int
    second: int

    # TODO
    frame_u: int
    # frames: int

    def __init__(self, hours: int, minutes: int, seconds: int, frames: int):
        if ((frames >> 6) & 0x01) != 1:
            raise ValueError

        fps = frames >> 6

        if fps not in VTS_FRAMERATE:
            raise ValueError

        self.hour = hours
        self.minute = minutes
        self.second = seconds
        self.frame_u = frames

    @staticmethod
    def bcd_to_int(bcd: int) -> int:
        return ((0xFF & (bcd >> 4)) * 10) + (bcd & 0x0F)

    def get_seconds_float(self) -> float:
        # + frames / framerate
        return float(
            ((self.bcd_to_int(self.hour) * 60) + self.bcd_to_int(self.minute)) * 60 + self.bcd_to_int(self.second)
        )


VTS_FRAMERATE = {0x01: Region.PAL.framerate, 0x03: Region.NTSC.framerate}
