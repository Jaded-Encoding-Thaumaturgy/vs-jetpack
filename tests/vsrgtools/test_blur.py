from typing import Sequence

import pytest

from jetpytools import CustomValueError, to_arr

from vsrgtools import bilateral, box_blur, flux_smooth, gauss_blur, median_blur, min_blur, sbr, side_box_blur
from vstools import ConvMode, DitherType, PlanesT, core, depth, get_prop, vs, invert_planes

color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
clip_int16 = depth(color_bars, 16, dither_type=DitherType.NONE)
clip_fp16 = depth(color_bars, 16, sample_type=vs.FLOAT, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)


def _has_not_been_processed(processed: vs.VideoNode, reference: vs.VideoNode, planes: PlanesT, frame_number: int = 0) -> bool:
    inverted_planes = invert_planes(reference, planes)

    if inverted_planes:
        diff = core.vszip.PlaneAverage(reference, -1, processed, inverted_planes)

        obj = diff if frame_number == 0 else diff.get_frame(frame_number)

        diff_prop = set(to_arr(get_prop(obj, "psmDiff", (list[float], float))))

        return len(diff_prop) == 1 and diff_prop.pop() == 0
    
    return True


@pytest.mark.parametrize("clip", [clip_int8, clip_int16, clip_fp16, clip_fp32])
@pytest.mark.parametrize("radius", [1, [1, 2]])
@pytest.mark.parametrize("passes", [2])
@pytest.mark.parametrize(
    "mode",
    [
        ConvMode.SQUARE,
        ConvMode.VERTICAL,
        ConvMode.HORIZONTAL,
        ConvMode.HV,
        ConvMode.TEMPORAL
    ]
)
@pytest.mark.parametrize("planes", [None, 0, [0, 1], [1, 2]])
def test_box_blur_no_exception(
    clip: vs.VideoNode,
    radius: int | Sequence[int],
    passes: int,
    mode: ConvMode,
    planes: PlanesT
) -> None:
    try:
        result = box_blur(
            clip=clip,
            radius=radius,
            passes=passes,
            mode=mode,  # type: ignore[arg-type]
            planes=planes
        )
    except CustomValueError as e:
        if all([
            mode == ConvMode.SQUARE,
            e.func == box_blur,
            e.message == "Invalid mode specified",
            e.reason == mode
        ]):
            pass
    except Exception as e:
        pytest.fail(
            f"box_blur raised an exception with clip = {repr(clip)},\nradius={radius}, passes={passes}, mode={mode}, planes={planes}:\n{e}"
        )
    else:
        assert isinstance(result, vs.VideoNode)

        assert _has_not_been_processed(result, clip, planes)
