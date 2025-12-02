from unittest import TestCase

import pytest

from vstools import (
    ColorRange,
    DitherType,
    UnsupportedColorFamilyError,
    core,
    depth,
    get_b,
    get_g,
    get_r,
    get_u,
    get_v,
    get_y,
    plane,
    stack_planes,
    vs,
)


class TestDitherType(TestCase):
    def test_should_dither_to_float(self) -> None:
        result = DitherType.ERROR_DIFFUSION.should_dither(vs.YUV444P8, vs.YUV444PS)
        self.assertFalse(result)

    def test_should_dither_from_float(self) -> None:
        result = DitherType.ERROR_DIFFUSION.should_dither(vs.YUV444PS, vs.YUV444P8)
        self.assertTrue(result)

    def test_should_dither_range_change(self) -> None:
        result = DitherType.ERROR_DIFFUSION.should_dither(
            vs.YUV444P8,
            vs.YUV444P8,
            in_range=ColorRange.LIMITED,
            out_range=ColorRange.FULL,
        )
        self.assertTrue(result)

        result = DitherType.ERROR_DIFFUSION.should_dither(
            vs.YUV444P8,
            vs.YUV444P8,
            in_range=ColorRange.FULL,
            out_range=ColorRange.LIMITED,
        )
        self.assertTrue(result)

    def test_should_dither_bits_same(self) -> None:
        result = DitherType.ERROR_DIFFUSION.should_dither(vs.YUV444P8, vs.YUV444P8)
        self.assertFalse(result)

    def test_should_dither_bits_increase(self) -> None:
        result = DitherType.ERROR_DIFFUSION.should_dither(vs.YUV444P8, vs.YUV444P16)
        self.assertFalse(result)

    def test_should_dither_bits_decrease(self) -> None:
        result = DitherType.ERROR_DIFFUSION.should_dither(vs.YUV444P16, vs.YUV444P8)
        self.assertTrue(result)


class TestUtils(TestCase):
    def test_depth(self) -> None:
        src_8 = vs.core.std.BlankClip(format=vs.YUV420P8)
        src_10 = depth(src_8, 10)
        assert src_10.format
        self.assertEqual(src_10.format.name, "YUV420P10")

        src2_10 = vs.core.std.BlankClip(format=vs.RGB30)
        src2_8 = depth(src2_10, 8, dither_type=DitherType.RANDOM)
        assert src2_8.format
        self.assertEqual(src2_8.format.name, "RGB24")

    def test_get_y(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = get_y(clip)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")

    def test_get_y_invalid(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        with self.assertRaises(UnsupportedColorFamilyError):
            get_y(clip)

    def test_get_u(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = get_u(clip)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")

    def test_get_u_invalid(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        with self.assertRaises(UnsupportedColorFamilyError):
            get_u(clip)

    def test_get_v(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = get_v(clip)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")

    def test_get_v_invalid(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        with self.assertRaises(UnsupportedColorFamilyError):
            get_v(clip)

    def test_get_r(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = get_r(clip)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")

    def test_get_r_invalid(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        with self.assertRaises(UnsupportedColorFamilyError):
            get_r(clip)

    def test_get_g(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = get_g(clip)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")

    def test_get_g_invalid(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        with self.assertRaises(UnsupportedColorFamilyError):
            get_g(clip)

    def test_get_b(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = get_b(clip)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")

    def test_get_b_invalid(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        with self.assertRaises(UnsupportedColorFamilyError):
            get_b(clip)

    def test_plane(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = plane(clip, 0)
        assert result.format
        self.assertEqual(result.format.name, "Gray8")


formats = [fmt for fmt in vs.PresetVideoFormat if fmt]


@pytest.mark.parametrize(
    "clip",
    [core.std.BlankClip(format=fmt) for fmt in formats],
    ids=[f"{fmt.name}/{fmt}" for fmt in formats],
)
def test_stack_plane(clip: vs.VideoNode) -> None:
    stack_planes(clip)
