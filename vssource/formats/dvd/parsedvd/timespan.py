from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from jetpytools import CustomEnum, classproperty

__all__ = ["VTS_FRAMERATE", "Region", "TimeSpan"]


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

        if fps not in Region.vts_framerate:
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


class Region(CustomEnum):
    PAL = "PAL", Fraction(25, 1), 0x01
    NTSC = "NTSC", Fraction(30000, 1001), 0x03

    def __init__(self, value: Any, framerate: Fraction, vts: int) -> None:
        self._value_ = value
        self.framerate = framerate
        self.vts = vts

    @classmethod
    def from_framerate(cls, framerate: float | Fraction) -> Region:
        key = Fraction(framerate)

        framerate_region_map = {r.framerate: r for r in Region}

        if framerate in framerate_region_map:
            return framerate_region_map[key]

        diffs = [(r, abs(float(key) - float(r.framerate))) for r in Region]

        diffs.sort(key=lambda x: x[1])

        return diffs[0][0]

    @classproperty.cached
    @classmethod
    def vts_framerate(cls) -> Mapping[int, Fraction]:
        return {r.vts: r.framerate for r in Region}


VTS_FRAMERATE = Region.vts_framerate
