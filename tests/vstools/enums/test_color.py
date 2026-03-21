from unittest import TestCase

from vstools import ColorRange, Matrix, Primaries, Transfer, vs


class TestMatrix(TestCase):
    def test_is_unspecified(self) -> None:
        assert Matrix.UNSPECIFIED.is_unspecified
        assert Matrix(2).is_unspecified
        assert Matrix.from_param(2).is_unspecified
        assert not Matrix.RGB.is_unspecified
        assert not Matrix(0).is_unspecified
        assert not Matrix.from_param(0).is_unspecified

    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Matrix.from_res(clip)
        assert result == Matrix.RGB

    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Matrix.from_res(clip)
        assert result == Matrix.BT709

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Matrix.from_res(clip)
        assert result == Matrix.BT709

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Matrix.from_res(clip)
        assert result == Matrix.ST170_M

    def test_from_res_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Matrix.from_res(clip)
        assert result == Matrix.BT470_BG

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Matrix.from_video(clip)
        assert result == Matrix.RGB

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = vs.core.std.SetFrameProp(clip, "_Matrix", Matrix.BT709)
        result = Matrix.from_video(clip)
        assert result == Matrix.BT709

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = Matrix.BT709.apply(clip)
        result = Matrix.from_video(clip)
        assert result == Matrix.BT709

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Matrix.from_video(clip)
        assert result == Matrix.BT709

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Matrix.from_video(clip)
        assert result == Matrix.BT709

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Matrix.from_video(clip)
        assert result == Matrix.ST170_M

    def test_from_video_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Matrix.from_video(clip)
        assert result == Matrix.BT470_BG


class TestTransfer(TestCase):
    def test_is_unspecified(self) -> None:
        assert Transfer.UNSPECIFIED.is_unspecified
        assert Transfer(2).is_unspecified
        assert Transfer.from_param(2).is_unspecified
        assert not Transfer.BT709.is_unspecified
        assert not Transfer(1).is_unspecified
        assert not Transfer.from_param(1).is_unspecified

    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Transfer.from_res(clip)
        assert result == Transfer.IEC_61966_2_1

    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Transfer.from_res(clip)
        assert result == Transfer.BT709

    def test_from_res_uhd_10b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P10, width=3840, height=2160)
        result = Transfer.from_res(clip)
        assert result == Transfer.BT709

    def test_from_res_uhd_12b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P12, width=3840, height=2160)
        result = Transfer.from_res(clip)
        assert result == Transfer.BT709

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Transfer.from_res(clip)
        assert result == Transfer.BT709

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Transfer.from_res(clip)
        assert result == Transfer.BT601

    def test_from_res_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Transfer.from_res(clip)
        assert result == Transfer.BT601

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = vs.core.std.SetFrameProp(clip, "_Transfer", Transfer.BT709)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT709

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = Transfer.BT709.apply(clip)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT709

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Transfer.from_video(clip)
        assert result == Transfer.IEC_61966_2_1

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT709

    def test_from_video_uhd_10b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P10, width=3840, height=2160)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT709

    def test_from_video_uhd_12b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P12, width=3840, height=2160)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT709

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT709

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT601

    def test_from_video_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Transfer.from_video(clip)
        assert result == Transfer.BT601


class TestPrimaries(TestCase):
    def test_is_unspecified(self) -> None:
        assert Primaries.UNSPECIFIED.is_unspecified
        assert Primaries(2).is_unspecified
        assert Primaries.from_param(2).is_unspecified
        assert not Primaries.BT709.is_unspecified
        assert not Primaries(1).is_unspecified
        assert not Primaries.from_param(1).is_unspecified

    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Primaries.from_res(clip)
        assert result == Primaries.BT709

    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Primaries.from_res(clip)
        assert result == Primaries.BT709

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Primaries.from_res(clip)
        assert result == Primaries.BT709

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Primaries.from_res(clip)
        assert result == Primaries.ST170_M

    def test_from_res_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Primaries.from_res(clip)
        assert result == Primaries.BT470_BG

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = vs.core.std.SetFrameProp(clip, "_Primaries", Primaries.BT709)
        result = Primaries.from_video(clip)
        assert result == Primaries.BT709

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = Primaries.BT709.apply(clip)
        result = Primaries.from_video(clip)
        assert result == Primaries.BT709

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Primaries.from_video(clip)
        assert result == Primaries.BT709

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Primaries.from_video(clip)
        assert result == Primaries.BT709

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Primaries.from_video(clip)
        assert result == Primaries.BT709

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Primaries.from_video(clip)
        assert result == Primaries.ST170_M

    def test_from_video_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Primaries.from_video(clip)
        assert result == Primaries.BT470_BG


class TestColorRange(TestCase):
    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = ColorRange.from_res(clip)
        assert result == ColorRange.FULL

    def test_from_res_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = ColorRange.from_res(clip)
        assert result == ColorRange.LIMITED

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        clip = vs.core.std.SetFrameProp(clip, "_ColorRange", ColorRange.FULL)
        result = ColorRange.from_video(clip)
        assert result == ColorRange.FULL

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        clip = ColorRange.FULL.apply(clip)
        result = ColorRange.from_video(clip)
        assert result == ColorRange.FULL

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = ColorRange.from_video(clip)
        assert result == ColorRange.FULL

    def test_from_video_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = ColorRange.from_video(clip)
        assert result == ColorRange.LIMITED

    def test_value_vs(self) -> None:
        assert ColorRange.LIMITED.value_vs == 1
        assert ColorRange.FULL.value_vs == 0

    def test_value_zimg(self) -> None:
        assert ColorRange.LIMITED.value_zimg == 0
        assert ColorRange.FULL.value_zimg == 1

    def test_value_is_limited(self) -> None:
        assert ColorRange.LIMITED.is_limited
        assert not ColorRange.FULL.is_limited

    def test_value_is_full(self) -> None:
        assert not ColorRange.LIMITED.is_full
        assert ColorRange.FULL.is_full
