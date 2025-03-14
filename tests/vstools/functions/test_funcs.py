from typing import Callable, cast
from unittest import TestCase

from vstools import (
    FunctionUtil, InvalidColorspacePathError, UndefinedMatrixError, fallback, iterate, kwargs_fallback, vs
)


class TestFuncs(TestCase):
    def test_iterate(self) -> None:
        result = iterate(5, cast(Callable[[int], int], lambda x: x * 2), 2)
        self.assertEqual(result, 20)

    def test_iterate_clip(self) -> None:
        clip = vs.core.std.BlankClip()
        result = iterate(clip, vs.core.std.Maximum, 3, threshold=0.5)
        self.assertEqual(type(result), vs.VideoNode)

    def test_fallback(self) -> None:
        self.assertEqual(fallback(5, 6), 5)
        self.assertEqual(fallback(None, 6), 6)

    def test_kwargs_fallback(self) -> None:
        kwargs = dict(
            overlap=1, search=2, block_size=4, sad_mode=8, motion=12, thSAD=16
        )
        self.assertEqual(kwargs_fallback(5, (kwargs, "block_size"), 8), 5)
        self.assertEqual(kwargs_fallback(None, (kwargs, "block_size"), 8), 4)
        self.assertEqual(kwargs_fallback(None, (dict(), "block_size"), 8), 8)

    def test_functionutil_bitdepth_tuple(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=(8, 16))
        self.assertIsInstance(result.bitdepth, range)
        self.assertEqual(result.bitdepth, range(8, 17))

    def test_functionutil_bitdepth_int(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=10)
        self.assertIsInstance(result.bitdepth, int)
        self.assertEqual(result.bitdepth, 10)

    def test_functionutil_bitdepth_set(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth={8, 10, 16})
        self.assertIsInstance(result.bitdepth, set)
        self.assertEqual(result.bitdepth, {8, 10, 16})

    def test_functionutil_bitdepth_range(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=range(8, 17))
        self.assertIsInstance(result.bitdepth, range)
        self.assertEqual(result.bitdepth, range(8, 17))

    def test_functionutil_bitdepth_conversion_int_up(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=16)
        self.assertEqual(result.work_clip.format.name, 'YUV420P16')

    def test_functionutil_bitdepth_conversion_int_down(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P16)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=8)
        self.assertEqual(result.work_clip.format.name, 'YUV420P8')

    def test_functionutil_bitdepth_conversion_int_same(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=8)
        self.assertEqual(result.work_clip.format.name, 'YUV420P8')

    def test_functionutil_bitdepth_conversion_tuple_up(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=(10, 16))
        self.assertEqual(result.work_clip.format.name, 'YUV420P10')

    def test_functionutil_bitdepth_conversion_tuple_down(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420PS)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=(8, 16))
        self.assertEqual(result.work_clip.format.name, 'YUV420P16')

    def test_functionutil_bitdepth_conversion_tuple_same(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P10)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=(8, 16))
        self.assertEqual(result.work_clip.format.name, 'YUV420P10')

    def test_functionutil_bitdepth_conversion_set_up(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P16)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth={8, 10, 32})
        self.assertEqual(result.work_clip.format.name, 'YUV420PS')

    def test_functionutil_bitdepth_conversion_set_down(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420PS)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth={8, 10, 16})
        self.assertEqual(result.work_clip.format.name, 'YUV420P16')

    def test_functionutil_bitdepth_conversion_set_same(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P16)
        result = FunctionUtil(clip, 'FunctionUtilTest', bitdepth={8, 16, 32})
        self.assertEqual(result.work_clip.format.name, 'YUV420P16')

    def test_functionutil_color_family_conversion_gray_to_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.GRAY8)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.GRAY)
        self.assertEqual(result.work_clip.format.color_family, vs.GRAY)
        self.assertFalse(result.cfamily_converted)

    def test_functionutil_color_family_conversion_gray_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.GRAY8)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.RGB, matrix=1)
        self.assertEqual(result.work_clip.format.color_family, vs.RGB)
        self.assertTrue(result.cfamily_converted)

    def test_functionutil_color_family_conversion_yuv_to_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.GRAY)
        self.assertEqual(result.work_clip.format.color_family, vs.GRAY)
        self.assertFalse(result.cfamily_converted)

    def test_functionutil_color_family_conversion_yuv_to_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.YUV)
        self.assertEqual(result.work_clip.format.color_family, vs.YUV)
        self.assertFalse(result.cfamily_converted)

    def test_functionutil_color_family_conversion_yuv_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.RGB, matrix=1)
        self.assertEqual(result.work_clip.format.color_family, vs.RGB)
        self.assertTrue(result.cfamily_converted)

    def test_functionutil_color_family_conversion_rgb_to_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.GRAY, matrix=1)
        self.assertEqual(result.work_clip.format.color_family, vs.GRAY)
        self.assertTrue(result.cfamily_converted)

    def test_functionutil_color_family_conversion_rgb_to_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.YUV, matrix=1)
        self.assertEqual(result.work_clip.format.color_family, vs.YUV)
        self.assertTrue(result.cfamily_converted)

    def test_functionutil_color_family_conversion_rgb_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = FunctionUtil(clip, 'FunctionUtilTest', color_family=vs.RGB)
        self.assertEqual(result.work_clip.format.color_family, vs.RGB)
        self.assertFalse(result.cfamily_converted)

    def test_functionutil_color_conversions_yuv_to_rgb_without_matrix(self) -> None:
        yuv_clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        with self.assertRaises(InvalidColorspacePathError):
            FunctionUtil(yuv_clip, 'FunctionUtilTest', color_family=vs.RGB)

    def test_functionutil_color_conversions_yuv_to_rgb_with_matrix(self) -> None:
        yuv_clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(yuv_clip, 'FunctionUtilTest', color_family=vs.RGB, matrix=1)
        self.assertEqual(result.work_clip.format.color_family, vs.RGB)
        self.assertTrue(result.cfamily_converted)

    def test_functionutil_color_conversions_gray_to_rgb_with_matrix(self) -> None:
        gray_clip = vs.core.std.BlankClip(format=vs.GRAY8)
        result = FunctionUtil(gray_clip, 'FunctionUtilTest', color_family=vs.RGB, matrix=1)
        self.assertEqual(result.work_clip.format.color_family, vs.RGB)
        self.assertTrue(result.cfamily_converted)

    def test_functionutil_color_conversions_rgb_to_yuv_without_matrix(self) -> None:
        rgb_clip = vs.core.std.BlankClip(format=vs.RGB24)
        with self.assertRaises(UndefinedMatrixError):
            FunctionUtil(rgb_clip, 'FunctionUtilTest', color_family=vs.YUV)

    def test_functionutil_planes_processing(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', planes=[1, 2])
        self.assertEqual(result.chroma_only, True)
        self.assertEqual(result.luma, False)

    def test_functionutil_matrix_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', matrix=1)
        self.assertEqual(result.matrix.value, 1)

    def test_functionutil_transfer_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', transfer=1)
        self.assertEqual(result.transfer.value, 1)

    def test_functionutil_primaries_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', primaries=1)
        self.assertEqual(result.primaries.value, 1)

    def test_functionutil_color_range_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', range_in=1)
        self.assertEqual(result.color_range.value, 1)

    def test_functionutil_chromaloc_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', chromaloc=1)
        self.assertEqual(result.chromaloc.value, 1)

    def test_functionutil_order_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, 'FunctionUtilTest', order=1)
        self.assertEqual(result.order.value, 1)

    def test_functionutil_return_clip_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.GRAY8)
        func_util = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=16)
        result = func_util.return_clip(func_util.work_clip)
        self.assertEqual(result.format.name, 'Gray8')

    def test_functionutil_return_clip_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        func_util = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=16)
        result = func_util.return_clip(func_util.work_clip)
        self.assertEqual(result.format.name, 'YUV420P8')

    def test_functionutil_return_clip_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        func_util = FunctionUtil(clip, 'FunctionUtilTest', bitdepth=16)
        result = func_util.return_clip(func_util.work_clip)
        self.assertEqual(result.format.name, 'RGB24')

    def test_functionutil_num_planes_yuv(self) -> None:
        clip_yuv = vs.core.std.BlankClip(format=vs.YUV420P8)
        result_yuv = FunctionUtil(clip_yuv, 'FunctionUtilTest')
        self.assertEqual(result_yuv.num_planes, 3)

    def test_functionutil_num_planes_gray(self) -> None:
        clip_gray = vs.core.std.BlankClip(format=vs.GRAY8)
        result_gray = FunctionUtil(clip_gray, 'FunctionUtilTest')
        self.assertEqual(result_gray.num_planes, 1)

    def test_functionutil_num_planes_rgb(self) -> None:
        clip_rgb = vs.core.std.BlankClip(format=vs.RGB24)
        result_rgb = FunctionUtil(clip_rgb, 'FunctionUtilTest')
        self.assertEqual(result_rgb.num_planes, 3)

    def test_functionutil_planes_0_yuv_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        func_util = FunctionUtil(clip, 'FunctionUtilTest', planes=0, color_family=vs.RGB, matrix=1)
        self.assertTrue(func_util.cfamily_converted)
        self.assertEqual(func_util.work_clip.format.color_family, vs.GRAY)
        self.assertEqual(func_util.norm_planes, [0])

    def test_functionutil_planes_0_rgb_to_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        func_util = FunctionUtil(clip, 'FunctionUtilTest', planes=0, color_family=vs.YUV, matrix=1)
        self.assertTrue(func_util.cfamily_converted)
        self.assertEqual(func_util.work_clip.format.color_family, vs.GRAY)
        self.assertEqual(func_util.norm_planes, [0])

