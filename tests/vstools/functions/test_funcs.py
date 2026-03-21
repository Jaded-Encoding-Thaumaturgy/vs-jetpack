from unittest import TestCase

import pytest

from vstools import FunctionUtil, InvalidColorspacePathError, UndefinedMatrixError, vs


class TestFuncs(TestCase):
    def test_functionutil_bitdepth_tuple(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=(8, 16))
        assert isinstance(result.bitdepth, range)
        assert result.bitdepth == range(8, 17)

    def test_functionutil_bitdepth_int(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=10)
        assert isinstance(result.bitdepth, int)
        assert result.bitdepth == 10

    def test_functionutil_bitdepth_set(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 10, 16})
        assert isinstance(result.bitdepth, set)
        assert result.bitdepth == {8, 10, 16}

    def test_functionutil_bitdepth_range(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=range(8, 17))
        assert isinstance(result.bitdepth, range)
        assert result.bitdepth == range(8, 17)

    def test_functionutil_bitdepth_conversion_int_up(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=16)
        assert result.work_clip.format.name == "YUV420P16"

    def test_functionutil_bitdepth_conversion_int_down(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P16)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=8)
        assert result.work_clip.format.name == "YUV420P8"

    def test_functionutil_bitdepth_conversion_int_same(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=8)
        assert result.work_clip.format.name == "YUV420P8"

    def test_functionutil_bitdepth_conversion_tuple_up(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=(10, 16))
        assert result.work_clip.format.name == "YUV420P10"

    def test_functionutil_bitdepth_conversion_tuple_down(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420PS)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=(8, 16))
        assert result.work_clip.format.name == "YUV420P16"

    def test_functionutil_bitdepth_conversion_tuple_same(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P10)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=(8, 16))
        assert result.work_clip.format.name == "YUV420P10"

    def test_functionutil_bitdepth_conversion_set_up(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P16)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 10, 32})
        assert result.work_clip.format.name == "YUV420PS"

    def test_functionutil_bitdepth_conversion_set_down(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420PS)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 10, 16})
        assert result.work_clip.format.name == "YUV420P16"

    def test_functionutil_bitdepth_conversion_set_same(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P16)
        result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 16, 32})
        assert result.work_clip.format.name == "YUV420P16"

    def test_functionutil_color_family_conversion_gray_to_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.GRAY8)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.GRAY)
        assert result.work_clip.format.color_family == vs.GRAY
        assert not result.cfamily_converted

    def test_functionutil_color_family_conversion_gray_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.GRAY8)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.RGB, matrix=1)
        assert result.work_clip.format.color_family == vs.RGB
        assert result.cfamily_converted

    def test_functionutil_color_family_conversion_yuv_to_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.GRAY)
        assert result.work_clip.format.color_family == vs.GRAY
        assert not result.cfamily_converted

    def test_functionutil_color_family_conversion_yuv_to_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.YUV)
        assert result.work_clip.format.color_family == vs.YUV
        assert not result.cfamily_converted

    def test_functionutil_color_family_conversion_yuv_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.RGB, matrix=1)
        assert result.work_clip.format.color_family == vs.RGB
        assert result.cfamily_converted

    def test_functionutil_color_family_conversion_rgb_to_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.GRAY, matrix=1)
        assert result.work_clip.format.color_family == vs.GRAY
        assert result.cfamily_converted

    def test_functionutil_color_family_conversion_rgb_to_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.YUV, matrix=1)
        assert result.work_clip.format.color_family == vs.YUV
        assert result.cfamily_converted

    def test_functionutil_color_family_conversion_rgb_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        result = FunctionUtil(clip, "FunctionUtilTest", color_family=vs.RGB)
        assert result.work_clip.format.color_family == vs.RGB
        assert not result.cfamily_converted

    def test_functionutil_color_conversions_yuv_to_rgb_without_matrix(self) -> None:
        yuv_clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        with pytest.raises(InvalidColorspacePathError):
            FunctionUtil(yuv_clip, "FunctionUtilTest", color_family=vs.RGB)

    def test_functionutil_color_conversions_yuv_to_rgb_with_matrix(self) -> None:
        yuv_clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(yuv_clip, "FunctionUtilTest", color_family=vs.RGB, matrix=1)
        assert result.work_clip.format.color_family == vs.RGB
        assert result.cfamily_converted

    def test_functionutil_color_conversions_gray_to_rgb_with_matrix(self) -> None:
        gray_clip = vs.core.std.BlankClip(format=vs.GRAY8)
        result = FunctionUtil(gray_clip, "FunctionUtilTest", color_family=vs.RGB, matrix=1)
        assert result.work_clip.format.color_family == vs.RGB
        assert result.cfamily_converted

    def test_functionutil_color_conversions_rgb_to_yuv_without_matrix(self) -> None:
        rgb_clip = vs.core.std.BlankClip(format=vs.RGB24)
        with pytest.raises(UndefinedMatrixError):
            FunctionUtil(rgb_clip, "FunctionUtilTest", color_family=vs.YUV)

    def test_functionutil_planes_processing(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", planes=[1, 2])
        assert result.chroma_only
        assert not result.luma

    def test_functionutil_matrix_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", matrix=1)
        assert result.matrix.value == 1

    def test_functionutil_transfer_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", transfer=1)
        assert result.transfer.value == 1

    def test_functionutil_primaries_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", primaries=1)
        assert result.primaries.value == 1

    def test_functionutil_color_range_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", range_in=1)
        assert result.color_range.value == 1

    def test_functionutil_chromaloc_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", chromaloc=1)
        assert result.chromaloc.value == 1

    def test_functionutil_order_property(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FunctionUtil(clip, "FunctionUtilTest", order=1)
        assert result.order.value == 1

    def test_functionutil_return_clip_gray(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.GRAY8)
        func_util = FunctionUtil(clip, "FunctionUtilTest", bitdepth=16)
        result = func_util.return_clip(func_util.work_clip)
        assert result.format.name == "Gray8"

    def test_functionutil_return_clip_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        func_util = FunctionUtil(clip, "FunctionUtilTest", bitdepth=16)
        result = func_util.return_clip(func_util.work_clip)
        assert result.format.name == "YUV420P8"

    def test_functionutil_return_clip_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        func_util = FunctionUtil(clip, "FunctionUtilTest", bitdepth=16)
        result = func_util.return_clip(func_util.work_clip)
        assert result.format.name == "RGB24"

    def test_functionutil_num_planes_yuv(self) -> None:
        clip_yuv = vs.core.std.BlankClip(format=vs.YUV420P8)
        result_yuv = FunctionUtil(clip_yuv, "FunctionUtilTest")
        assert result_yuv.num_planes == 3

    def test_functionutil_num_planes_gray(self) -> None:
        clip_gray = vs.core.std.BlankClip(format=vs.GRAY8)
        result_gray = FunctionUtil(clip_gray, "FunctionUtilTest")
        assert result_gray.num_planes == 1

    def test_functionutil_num_planes_rgb(self) -> None:
        clip_rgb = vs.core.std.BlankClip(format=vs.RGB24)
        result_rgb = FunctionUtil(clip_rgb, "FunctionUtilTest")
        assert result_rgb.num_planes == 3

    def test_functionutil_planes_0_yuv_to_rgb(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        func_util = FunctionUtil(clip, "FunctionUtilTest", planes=0, color_family=vs.RGB, matrix=1)
        assert func_util.cfamily_converted
        assert func_util.work_clip.format.color_family == vs.GRAY
        assert func_util.norm_planes == [0]

    def test_functionutil_planes_0_rgb_to_yuv(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.RGB24)
        func_util = FunctionUtil(clip, "FunctionUtilTest", planes=0, color_family=vs.YUV, matrix=1)
        assert func_util.cfamily_converted
        assert func_util.work_clip.format.color_family == vs.GRAY
        assert func_util.norm_planes == [0]
