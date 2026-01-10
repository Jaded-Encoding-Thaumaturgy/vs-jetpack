from unittest import TestCase

from vstools import get_prop, vs
from vstools.enums.other import Dar, Sar


class TestDar(TestCase):
    def test_from_res(self) -> None:
        result = Dar.from_res(1920, 1080)
        self.assertEqual(result, Dar(16, 9))

    def test_from_clip(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Dar.from_clip(clip)
        self.assertEqual(result, Dar(16, 9))

    def test_to_sar(self) -> None:
        self.assertEqual(Dar(16, 9).to_sar(1, 1080), Sar(1920, 1))


class TestSar(TestCase):
    def test_from_ar(self) -> None:
        self.assertEqual(Sar.from_ar(1, 1080, Dar(16, 9)), Sar(1920, 1))

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        clip = Sar(1920, 1).apply(clip)
        self.assertEqual(get_prop(clip, "_SARNum", int), 1920)
        self.assertEqual(get_prop(clip, "_SARDen", int), 1)
