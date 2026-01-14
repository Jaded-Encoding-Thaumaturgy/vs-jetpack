import pytest
from jetpytools import CustomNotImplementedError

from vstools import FunctionUtil, UnsupportedColorFamilyError, UnsupportedVideoFormatError, core, vs


@pytest.fixture
def gray_clip() -> vs.VideoNode:
    return core.std.BlankClip(format=vs.GRAY8)


@pytest.fixture
def yuv_clip() -> vs.VideoNode:
    return core.std.BlankClip(format=vs.YUV420P8)


@pytest.fixture
def rgb_clip() -> vs.VideoNode:
    return core.std.BlankClip(format=vs.RGB24)


# Color family tests
def test_functionutil_colorfamily_not_rgb(rgb_clip: vs.VideoNode) -> None:
    with pytest.raises(UnsupportedColorFamilyError):
        FunctionUtil(rgb_clip, "FunctionUtilTest", color_family=(vs.YUV, vs.GRAY))


def test_functionutil_colorfamily_rgb_planes(rgb_clip: vs.VideoNode) -> None:
    with pytest.raises(CustomNotImplementedError):
        FunctionUtil(rgb_clip, "FunctionUtilTest", [1, 2], vs.RGB)


# Bit depth tests
def test_functionutil_bitdepth_int(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=10)
    assert result.allowed_bitdepth == {10}


def test_functionutil_bitdepth_set(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth={8, 10, 16})
    assert result.allowed_bitdepth == {8, 10, 16}


def test_functionutil_bitdepth_tuple(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=(8, 16))
    assert result.allowed_bitdepth == set(range(8, 17))


def test_functionutil_bitdepth_range(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=range(8, 17))
    assert result.allowed_bitdepth == set(range(8, 17))


def test_functionutil_bitdepth_conversion_int_up(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=16)
    assert result.norm_clip.format.id == vs.YUV420P16


def test_functionutil_bitdepth_conversion_int_down() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P16)

    result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=8)

    with pytest.raises(UnsupportedVideoFormatError):
        result.norm_clip


def test_functionutil_bitdepth_conversion_int_same(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=8)
    assert result.norm_clip.format.id == vs.YUV420P8


def test_functionutil_bitdepth_conversion_set_up() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P16)
    result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 10, 32})
    assert result.norm_clip.format.id == vs.YUV420PS


def test_functionutil_bitdepth_conversion_set_down() -> None:
    clip = core.std.BlankClip(format=vs.YUV420PS)
    result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 10, 16})

    with pytest.raises(UnsupportedVideoFormatError):
        result.norm_clip


def test_functionutil_bitdepth_conversion_set_same() -> None:
    clip = vs.core.std.BlankClip(format=vs.YUV420P16)
    result = FunctionUtil(clip, "FunctionUtilTest", bitdepth={8, 16, 32})
    assert result.norm_clip.format.id == vs.YUV420P16


def test_functionutil_bitdepth_conversion_tuple_up(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=(10, 16))
    assert result.norm_clip.format.id == vs.YUV420P10


def test_functionutil_bitdepth_conversion_tuple_down() -> None:
    clip = core.std.BlankClip(format=vs.YUV420PS)
    result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=(8, 16))

    with pytest.raises(UnsupportedVideoFormatError):
        result.norm_clip


def test_functionutil_bitdepth_conversion_tuple_same() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P10)
    result = FunctionUtil(clip, "FunctionUtilTest", bitdepth=(8, 16))
    assert result.norm_clip.format.id == vs.YUV420P10


# Plane and property tests ------------------------------------
def test_functionutil_planes_processing(yuv_clip: vs.VideoNode) -> None:
    result = FunctionUtil(yuv_clip, "FunctionUtilTest", planes=[1, 2])
    assert result.chroma_only
    assert not result.luma


# Return clip
def test_functionutil_return_clip_gray(gray_clip: vs.VideoNode) -> None:
    func_util = FunctionUtil(gray_clip, "FunctionUtilTest", bitdepth=16)
    result = func_util.return_clip(func_util.work_clip)
    assert result.format.id == vs.GRAY16


def test_functionutil_return_clip_yuv(yuv_clip: vs.VideoNode) -> None:
    func_util = FunctionUtil(yuv_clip, "FunctionUtilTest", bitdepth=16)
    result = func_util.return_clip(func_util.work_clip)
    assert result.format.id == vs.YUV420P16


def test_functionutil_return_clip_rgb(rgb_clip: vs.VideoNode) -> None:
    func_util = FunctionUtil(rgb_clip, "FunctionUtilTest", bitdepth=16)
    result = func_util.return_clip(func_util.work_clip)
    assert result.format.id == vs.RGB48
