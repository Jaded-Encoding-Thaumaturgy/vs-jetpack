from unittest import TestCase

from vstools import get_h, get_w, vs


class TestInfo(TestCase):
    def test_get_w(self) -> None:
        self.assertEqual(get_w(1080, 16 / 9), 1920)
        self.assertEqual(get_w(1080, 4 / 3), 1440)

        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        self.assertEqual(get_w(1080, clip), 1920)

    def test_get_h(self) -> None:
        self.assertEqual(get_h(1920, 16 / 9), 1080)
        self.assertEqual(get_h(1440, 4 / 3), 1080)

        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        self.assertEqual(get_h(1920, clip), 1080)
