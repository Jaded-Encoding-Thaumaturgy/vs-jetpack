from __future__ import annotations

from fractions import Fraction

from vstools import Timecodes, core, vs
from vstools.functions.timecodes import FrameDur

clip = core.std.BlankClip(format=vs.YUV420P16, width=1920, height=1080)
clip_descaled = core.std.BlankClip(format=vs.YUV420P16, width=1600, height=900)


def test_timecodes_to_normalized_ranges() -> None:
    # inclusive/inclusive range
    tc = Timecodes(FrameDur(i, 1001, 24000) for i in range(1000))

    assert tc.to_normalized_ranges() == {(0, 999): Fraction(24000, 1001)}

    tc = Timecodes()
    tc.extend([FrameDur(i, 1001, 24000) for i in range(1000)])
    tc.extend([FrameDur(i, 1001, 30000) for i in range(1000, 1500)])
    tc.extend([FrameDur(i, 1, 60) for i in range(1500, 2000)])

    assert tc.to_normalized_ranges() == {
        (0, 999): Fraction(24000, 1001),
        (1000, 1499): Fraction(30000, 1001),
        (1500, 1999): Fraction(60, 1),
    }


def test_timecodes_normalize_range_timecodes() -> None:
    # inclusive/inclusive range
    timecodes = {
        (None, 9): Fraction(24000, 1001),
        (10, 14): Fraction(30000, 1001),
        (15, None): Fraction(60, 1),
    }
    fractions = Timecodes.normalize_range_timecodes(timecodes, 20)

    assert fractions == [
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 24000),
        Fraction(1001, 30000),
        Fraction(1001, 30000),
        Fraction(1001, 30000),
        Fraction(1001, 30000),
        Fraction(1001, 30000),
        Fraction(1, 60),
        Fraction(1, 60),
        Fraction(1, 60),
        Fraction(1, 60),
        Fraction(1, 60),
    ]


def test_timecodes_separate_norm_timecodes() -> None:
    # inclusive/inclusive range
    timecodes = {
        (0, 9): Fraction(24000, 1001),
        (10, 14): Fraction(30000, 1001),
        (15, 19): Fraction(60, 1),
        (20, 14): Fraction(24000, 1001),
    }

    major_time, minor_fps = Timecodes.separate_norm_timecodes(timecodes)

    assert major_time == Fraction(24000, 1001)
    assert minor_fps == {(10, 14): Fraction(30000, 1001), (15, 19): Fraction(60, 1)}


def test_timecodes_accumulate_norm_timecodes() -> None:
    # inclusive/inclusive range
    timecodes = {
        (0, 9): Fraction(24000, 1001),
        (10, 14): Fraction(30000, 1001),
        (15, 19): Fraction(60, 1),
        (20, 24): Fraction(24000, 1001),
        (25, 39): Fraction(30000, 1001),
        (40, 49): Fraction(60, 1),
        (50, 59): Fraction(12000, 1001),
    }

    major_time, acc_ranges = Timecodes.accumulate_norm_timecodes(timecodes)

    assert major_time == Fraction(24000, 1001)
    assert acc_ranges == {
        Fraction(30000, 1001): [(10, 14), (25, 39)],
        Fraction(60, 1): [(15, 19), (40, 49)],
        Fraction(12000, 1001): [(50, 59)],
    }


def test_timecodes_from_clip() -> None:
    # inclusive/inclusive range
    timecodes = {
        (0, 9): Fraction(24000, 1001),
        (10, 14): Fraction(30000, 1001),
        (15, 19): Fraction(60, 1),
        (20, 24): Fraction(24000, 1001),
        (25, 39): Fraction(30000, 1001),
        (40, 49): Fraction(60, 1),
        (50, 59): Fraction(12000, 1001),
    }
    clips = [
        core.std.BlankClip(length=e + 1 - s, fpsnum=fps.numerator, fpsden=fps.denominator)
        for (s, e), fps in timecodes.items()
    ]
    clip = core.std.Splice(clips)

    tc = Timecodes.from_clip(clip)

    reference = Timecodes()

    for (s, e), fps in timecodes.items():
        for n in range(s, e + 1):
            reference.append(FrameDur(n, fps.denominator, fps.numerator))

    assert tc == Timecodes(reference)


def test_timecodes_from_file() -> None: ...


def test_timecodes_assume_vfr() -> None: ...


def test_timecodes_to_file() -> None: ...
