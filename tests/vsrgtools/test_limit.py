
import pytest
from vsrgtools import LimitFilterMode, limit_filter
from vstools import DitherType, core, depth, vs



color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)

fltclip_int8 = clip_int8.vszip.BoxBlur()
fltclip_fp32 = clip_fp32.vszip.BoxBlur()


@pytest.mark.parametrize("clips", [(fltclip_int8, clip_int8), (fltclip_fp32, clip_fp32)])
@pytest.mark.parametrize("mode", list(LimitFilterMode))
@pytest.mark.parametrize("thr", [1.0, (1.4, 0.8)])
@pytest.mark.parametrize("elast", [2.0, 3.0])
@pytest.mark.parametrize("bright_thr", [None, 2.5])
def test_limit_filter(clips: tuple[vs.VideoNode, vs.VideoNode], mode: LimitFilterMode, thr: float, elast: float, bright_thr: None | float) -> None:
    flt, src = clips

    if mode in [
        LimitFilterMode.SIMPLE_MIN, LimitFilterMode.SIMPLE_MAX,
        LimitFilterMode.SIMPLE2_MIN, LimitFilterMode.SIMPLE2_MAX,
        LimitFilterMode.DIFF_MIN, LimitFilterMode.DIFF_MAX,
    ]:
        ref = src
    else:
        ref = None

    limit_filter(flt, src, ref, mode, None, thr, elast, bright_thr)
