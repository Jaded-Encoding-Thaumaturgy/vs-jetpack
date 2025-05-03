from typing import Sequence, Any, Literal

import pytest

from jetpytools import CustomValueError, FuncExceptT, to_arr, norm_display_name

from vsrgtools import BlurMatrix, bilateral, box_blur, flux_smooth, gauss_blur, median_blur, min_blur, sbr, side_box_blur
from vstools import ConvMode, DitherType, PlanesT, core, depth, get_prop, vs, invert_planes

color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
# clip_int16 = depth(color_bars, 16, dither_type=DitherType.NONE)
clip_fp16 = depth(color_bars, 16, sample_type=vs.FLOAT, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)


def _display_error_msg(clip: vs.VideoNode, func: FuncExceptT, exc: Exception, **kwargs: Any) -> str:
    assert clip.format
    return (
        str(exc) + "\n"
        + f"{norm_display_name(func)} | <clip format: {clip.format.name}>"
        + " | <args " + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + " >"
    )

def _has_not_been_processed(processed: vs.VideoNode, reference: vs.VideoNode, planes: PlanesT, frame_number: int = 0) -> bool:
    inverted_planes = invert_planes(reference, planes)

    if inverted_planes:
        diff = core.vszip.PlaneAverage(reference, -1, processed, inverted_planes)

        obj = diff if frame_number == 0 else diff.get_frame(frame_number)

        diff_prop = set(to_arr(get_prop(obj, "psmDiff", (list[float], float))))

        return len(diff_prop) == 1 and diff_prop.pop() == 0
    
    return True


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
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
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
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
        pytest.fail(_display_error_msg(clip, box_blur, e, radius=radius, passes=passes, mode=mode, planes=planes))
    else:
        assert isinstance(result, vs.VideoNode)

        assert _has_not_been_processed(result, clip, planes)


def test_side_box_blur_no_exception() -> None:
    # TODO
    ...


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("sigma", [0.5, [0.5, 1.0]])
@pytest.mark.parametrize("taps", [None, 3])
@pytest.mark.parametrize(
    "mode",
    [
        ConvMode.SQUARE,
        ConvMode.HORIZONTAL,
        ConvMode.VERTICAL,
        ConvMode.HV,
        ConvMode.TEMPORAL
    ]
)
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
@pytest.mark.parametrize("_fast", [True, False])
def test_gauss_blur_no_exception(
    clip: vs.VideoNode,
    sigma: float | Sequence[float],
    taps: int | None,
    mode: ConvMode,
    planes: PlanesT,
    _fast: bool
) -> None:
    try:
        result = gauss_blur(
            clip=clip,
            sigma=sigma,
            taps=taps,
            mode=mode,  # type: ignore[arg-type]
            planes=planes,
            _fast=_fast
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
            _display_error_msg(
                clip, gauss_blur, e,
                sigma=sigma,
                taps=taps,
                mode=mode,
                planes=planes,
                _fast=_fast
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
        assert _has_not_been_processed(result, clip, planes)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("radius", [1, [1, 2]])
@pytest.mark.parametrize(
    "mode",
    [
        (ConvMode.HV, ConvMode.SQUARE),
        (ConvMode.HORIZONTAL, ConvMode.VERTICAL),
        (ConvMode.TEMPORAL, ConvMode.TEMPORAL)
    ]
)
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
def test_min_blur_no_exception(
    clip: vs.VideoNode,
    radius: int | Sequence[int],
    mode: tuple[ConvMode, ConvMode],
    planes: PlanesT
) -> None:
    try:
        result = min_blur(
            clip=clip,
            radius=radius,
            mode=mode,
            planes=planes
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, min_blur, e,
                radius=radius,
                mode=mode,
                planes=planes,
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
        assert _has_not_been_processed(result, clip, planes)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("radius", [1, [1, 2]])
@pytest.mark.parametrize(
    "mode",
    [
        ConvMode.SQUARE,
        ConvMode.HORIZONTAL,
        ConvMode.VERTICAL,
        ConvMode.HV,
        ConvMode.TEMPORAL
    ]
)
@pytest.mark.parametrize("blur", [BlurMatrix.BINOMIAL])
@pytest.mark.parametrize("blur_diff", [BlurMatrix.BINOMIAL])
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
def test_sbr_blurmatrix_no_exception(
    clip: vs.VideoNode,
    radius: int | Sequence[int],
    mode: ConvMode,
    blur: BlurMatrix,
    blur_diff: BlurMatrix,
    planes: PlanesT
) -> None:
    try:
        result = sbr(
            clip=clip,
            radius=radius,
            mode=mode,
            blur=blur,
            blur_diff=blur_diff,
            planes=planes
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, sbr, e,
                radius=radius,
                mode=mode,
                blur=blur,
                blur_diff=blur_diff,
                planes=planes
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
        assert _has_not_been_processed(result, clip, planes)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("mode", [ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.HV])
@pytest.mark.parametrize("blur", [[1] * 9, lambda clip: clip])
@pytest.mark.parametrize("blur_diff", [[0, 1, 0, 1, 1, 1, 0, 1, 0]])
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
def test_sbr_no_exception(
    clip: vs.VideoNode,
    mode: ConvMode,
    blur: Any,
    blur_diff: Any,
    planes: Any
) -> None:
    try:
        result = sbr(
            clip=clip,
            mode=mode,
            blur=blur,
            blur_diff=blur_diff,
            planes=planes
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, sbr, e,
                mode=mode,
                blur=blur,
                blur_diff=blur_diff,
                planes=planes
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
        assert _has_not_been_processed(result, clip, planes)



@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("radius", [1, [1, 2]])
@pytest.mark.parametrize(
    "mode",
    [
        ConvMode.SQUARE,
        ConvMode.HORIZONTAL,
        ConvMode.VERTICAL,
        ConvMode.HV,
    ]
)
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
def test_median_blur_spatial_no_exception(
    clip: vs.VideoNode,
    radius: int | Sequence[int],
    mode: Any,
    planes: Any
) -> None:
    try:
        result = median_blur(
            clip=clip,
            radius=radius,
            mode=mode,
            planes=planes
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, median_blur, e,
                radius=radius,
                mode=mode,
                planes=planes
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
        assert _has_not_been_processed(result, clip, planes)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("mode", [ConvMode.TEMPORAL])
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
def test_median_blur_temporal_no_exception(
    clip: vs.VideoNode,
    radius: int,
    mode: Literal[ConvMode.TEMPORAL],
    planes: Any
) -> None:
    try:
        result = median_blur(
            clip=clip,
            radius=radius,
            mode=mode,
            planes=planes
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, median_blur, e,
                radius=radius,
                mode=mode,
                planes=planes
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
        assert _has_not_been_processed(result, clip, planes)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("ref", [None])
@pytest.mark.parametrize("sigmaS", [None, 1.5, [1.0, 2.0]])
@pytest.mark.parametrize("sigmaR", [None, 0.1, [0.1, 0.2]])
@pytest.mark.parametrize("backend", [bilateral.Backend.CPU])
def test_bilateral_no_exception(
    clip: vs.VideoNode,
    ref: vs.VideoNode | None,
    sigmaS: float | Sequence[float] | None,
    sigmaR: float | Sequence[float] | None,
    backend: Any
) -> None:
    try:
        result = bilateral(
            clip=clip,
            ref=ref,
            sigmaS=sigmaS,
            sigmaR=sigmaR,
            backend=backend
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, bilateral, e,
                ref=ref,
                sigmaS=sigmaS,
                sigmaR=sigmaR,
                backend=backend
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("temporal_threshold", [7.0, [5.0, 10.0]])
@pytest.mark.parametrize("spatial_threshold", [None, 2.0, [1.0, 3.0]])
@pytest.mark.parametrize("scalep", [True, False])
def test_flux_smooth_no_exception(
    clip: vs.VideoNode,
    temporal_threshold: float | Sequence[float],
    spatial_threshold: float | Sequence[float] | None,
    scalep: bool
) -> None:
    try:
        result = flux_smooth(
            clip=clip,
            temporal_threshold=temporal_threshold,
            spatial_threshold=spatial_threshold,
            scalep=scalep
        )
    except Exception as e:
        pytest.fail(
            _display_error_msg(
                clip, flux_smooth, e,
                temporal_threshold=temporal_threshold,
                spatial_threshold=spatial_threshold,
                scalep=scalep
            )
        )
    else:
        assert isinstance(result, vs.VideoNode)
