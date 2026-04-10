from unittest import TestCase

from vstools import Matrix, Primaries, Range, Transfer, vs


class TestMatrix(TestCase):
    def test_is_unspecified(self) -> None:
        self.assertTrue(Matrix.UNSPECIFIED.is_unspecified)
        self.assertTrue(Matrix(2).is_unspecified)
        self.assertTrue(Matrix.from_param(2).is_unspecified)
        self.assertFalse(Matrix.RGB.is_unspecified)
        self.assertFalse(Matrix(0).is_unspecified)
        self.assertFalse(Matrix.from_param(0).is_unspecified)

    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Matrix.from_res(clip)
        self.assertEqual(result, Matrix.RGB)

    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Matrix.from_res(clip)
        self.assertEqual(result, Matrix.BT709)

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Matrix.from_res(clip)
        self.assertEqual(result, Matrix.BT709)

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Matrix.from_res(clip)
        self.assertEqual(result, Matrix.ST170_M)

    def test_from_res_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Matrix.from_res(clip)
        self.assertEqual(result, Matrix.BT470_BG)

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.RGB)

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = vs.core.std.SetFrameProp(clip, "_Matrix", Matrix.BT709)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.BT709)

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = Matrix.BT709.apply(clip)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.BT709)

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.BT709)

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.BT709)

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.ST170_M)

    def test_from_video_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Matrix.from_video(clip)
        self.assertEqual(result, Matrix.BT470_BG)


class TestTransfer(TestCase):
    def test_is_unspecified(self) -> None:
        self.assertTrue(Transfer.UNSPECIFIED.is_unspecified)
        self.assertTrue(Transfer(2).is_unspecified)
        self.assertTrue(Transfer.from_param(2).is_unspecified)
        self.assertFalse(Transfer.BT709.is_unspecified)
        self.assertFalse(Transfer(1).is_unspecified)
        self.assertFalse(Transfer.from_param(1).is_unspecified)

    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.IEC_61966_2_1)

    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_res_uhd_10b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P10, width=3840, height=2160)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_res_uhd_12b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P12, width=3840, height=2160)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.BT601)

    def test_from_res_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Transfer.from_res(clip)
        self.assertEqual(result, Transfer.BT601)

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = vs.core.std.SetFrameProp(clip, "_Transfer", Transfer.BT709)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = Transfer.BT709.apply(clip)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.IEC_61966_2_1)

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_video_uhd_10b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P10, width=3840, height=2160)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_video_uhd_12b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P12, width=3840, height=2160)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT709)

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT601)

    def test_from_video_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Transfer.from_video(clip)
        self.assertEqual(result, Transfer.BT601)


class TestPrimaries(TestCase):
    def test_is_unspecified(self) -> None:
        self.assertTrue(Primaries.UNSPECIFIED.is_unspecified)
        self.assertTrue(Primaries(2).is_unspecified)
        self.assertTrue(Primaries.from_param(2).is_unspecified)
        self.assertFalse(Primaries.BT709.is_unspecified)
        self.assertFalse(Primaries(1).is_unspecified)
        self.assertFalse(Primaries.from_param(1).is_unspecified)

    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Primaries.from_res(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Primaries.from_res(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Primaries.from_res(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Primaries.from_res(clip)
        self.assertEqual(result, Primaries.ST170_M)

    def test_from_res_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Primaries.from_res(clip)
        self.assertEqual(result, Primaries.BT470_BG)

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = vs.core.std.SetFrameProp(clip, "_Primaries", Primaries.BT709)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        clip = Primaries.BT709.apply(clip)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.BT709)

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.ST170_M)

    def test_from_video_pal(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1024, height=576)
        result = Primaries.from_video(clip)
        self.assertEqual(result, Primaries.BT470_BG)


class TestRange(TestCase):
    def test_from_res_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Range.from_res(clip)
        self.assertEqual(result, Range.FULL)

    def test_from_res_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = Range.from_res(clip)
        self.assertEqual(result, Range.LIMITED)

    def test_from_video_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        if vs.__version__ >= (74, 0):
            clip = vs.core.std.SetFrameProp(clip, "_Range", Range.FULL)
        else:
            clip = vs.core.std.SetFrameProp(clip, "_ColorRange", Range.FULL)
        result = Range.from_video(clip)
        self.assertEqual(result, Range.FULL)

    def test_apply(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        clip = Range.FULL.apply(clip)
        result = Range.from_video(clip)
        self.assertEqual(result, Range.FULL)

    def test_from_video_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = Range.from_video(clip)
        self.assertEqual(result, Range.FULL)

    def test_from_video_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = Range.from_video(clip)
        self.assertEqual(result, Range.LIMITED)

    def test_value_vs(self) -> None:
        if vs.__version__ < (74, 0):
            self.assertEqual(Range.LIMITED.value_vs, 1)
            self.assertEqual(Range.FULL.value_vs, 0)

    def test_value_zimg(self) -> None:
        self.assertEqual(Range.LIMITED.value_zimg, 0)
        self.assertEqual(Range.FULL.value_zimg, 1)

    def test_value_is_limited(self) -> None:
        self.assertTrue(Range.LIMITED.is_limited)
        self.assertFalse(Range.FULL.is_limited)

    def test_value_is_full(self) -> None:
        self.assertFalse(Range.LIMITED.is_full)
        self.assertTrue(Range.FULL.is_full)
