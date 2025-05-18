from typing import Any

import pytest

from jetpytools import FuncExceptT, norm_display_name

from vsrgtools import box_blur, contrasharpening, contrasharpening_dehalo, contrasharpening_median, removegrain
from vstools import DitherType, core, depth, vs


color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)

fltclip_int8 = clip_int8.vszip.BoxBlur()
fltclip_fp32 = clip_fp32.vszip.BoxBlur()

def _display_error_msg(clip: vs.VideoNode, func: FuncExceptT, exc: Exception, **kwargs: Any) -> str:
    assert clip.format
    return (
        str(exc) + "\n"
        + f"{norm_display_name(func)} | <clip format: {clip.format.name}>"
        + " | <args " + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + " >"
    )

@pytest.mark.parametrize("clips", [(fltclip_int8, clip_int8), (fltclip_fp32, clip_fp32)])
@pytest.mark.parametrize("radius", [1, 2])
def test_contrasharpening_no_exception(clips: tuple[vs.VideoNode, vs.VideoNode], radius: int) -> None:
    func = contrasharpening

    flt, src = clips
    try:
        result = func(flt, src, radius)

    except Exception as e:
        pytest.fail(_display_error_msg(flt, func, e, radius=radius))
    else:
        assert isinstance(result, vs.VideoNode)


@pytest.mark.parametrize("clips", [(fltclip_int8, clip_int8), (fltclip_fp32, clip_fp32)])
def test_contrasharpening_dehalo_no_exception(clips: tuple[vs.VideoNode, vs.VideoNode]) -> None:
    func = contrasharpening_dehalo

    flt, src = clips
    try:
        result = func(flt, src)

    except Exception as e:
        pytest.fail(_display_error_msg(flt, func, e))
    else:
        assert isinstance(result, vs.VideoNode)


@pytest.mark.parametrize("clips", [(fltclip_int8, clip_int8), (fltclip_fp32, clip_fp32)])
@pytest.mark.parametrize("mode", [1, removegrain.Mode(10), box_blur])
def test_contrasharpening_median_no_exception(clips: tuple[vs.VideoNode, vs.VideoNode], mode: Any) -> None:
    func = contrasharpening_median

    flt, src = clips
    try:
        result = func(flt, src, mode)

    except Exception as e:
        pytest.fail(_display_error_msg(flt, func, e, mode=mode))
    else:
        assert isinstance(result, vs.VideoNode)
