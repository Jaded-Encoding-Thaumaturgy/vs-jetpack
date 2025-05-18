
import pytest

from vsrgtools import clense, remove_grain, repair, vertical_cleaner
from vsrgtools.rgtools import Clense, RemoveGrain, Repair, VerticalCleaner
from vstools import DitherType, core, depth, vs


color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)

fltclip_int8 = clip_int8.vszip.BoxBlur()
fltclip_fp32 = clip_fp32.vszip.BoxBlur()


@pytest.mark.parametrize("clips", [(fltclip_int8, clip_int8), (fltclip_fp32, clip_fp32)])
@pytest.mark.parametrize("mode", list(repair.Mode))
def test_repair_enum(clips: tuple[vs.VideoNode, vs.VideoNode], mode: Repair.Mode) -> None:
    mode(*clips)


@pytest.mark.parametrize("clips", [(fltclip_int8, clip_int8), (fltclip_fp32, clip_fp32)])
@pytest.mark.parametrize("mode", list(repair.Mode))
def test_repair_function(clips: tuple[vs.VideoNode, vs.VideoNode], mode: Repair.Mode) -> None:
    repair(*clips, mode)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(remove_grain.Mode))
def test_remove_grain_enum(clip: vs.VideoNode, mode: RemoveGrain.Mode) -> None:
    mode(clip)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(remove_grain.Mode))
def test_remove_grain_function(clip: vs.VideoNode, mode: RemoveGrain.Mode) -> None:
    remove_grain(clip, mode)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(clense.Mode))
def test_clense_enum(clip: vs.VideoNode, mode: Clense.Mode) -> None:
    mode(clip)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(clense.Mode))
def test_clense_grain_function(clip: vs.VideoNode, mode: Clense.Mode) -> None:
    clense(clip, mode=mode)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(vertical_cleaner.Mode))
def test_vertical_cleaner_enum(clip: vs.VideoNode, mode: VerticalCleaner.Mode) -> None:
    mode(clip)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", list(vertical_cleaner.Mode))
def test_vertical_cleaner_grain_function(clip: vs.VideoNode, mode: VerticalCleaner.Mode) -> None:
    vertical_cleaner(clip, mode)
