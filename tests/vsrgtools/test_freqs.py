

import pytest
from vsrgtools import MeanMode
from vstools import DitherType, core, depth, vs


color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(MeanMode))
def test_mean_mode(clip: vs.VideoNode, mode: MeanMode) -> None:
    mode(clip)
