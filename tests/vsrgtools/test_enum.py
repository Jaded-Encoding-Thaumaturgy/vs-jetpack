from jetpytools import FuncExceptT, norm_display_name
import pytest
from vstools import ConvMode, DitherType, core, depth, vs, PlanesT
from vsrgtools import BlurMatrix, BlurMatrixBase
from typing import Any


def _display_error_msg(clip: vs.VideoNode, func: FuncExceptT, exc: Exception, **kwargs: Any) -> str:
    assert clip.format
    return (
        str(exc) + "\n"
        + f"{norm_display_name(func)} | <clip format: {clip.format.name}>"
        + " | <args " + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + " >"
    )



def test_blur_matrix_mean_no_center() -> None:
    assert BlurMatrix.MEAN_NO_CENTER() == BlurMatrix.BOX_BLUR_NO_CENTER() == [1, 1, 1, 1, 0, 1, 1, 1, 1]
    assert (
        BlurMatrix.MEAN_NO_CENTER(2, mode=ConvMode.HORIZONTAL)
        == BlurMatrix.BOX_BLUR_NO_CENTER(2, mode=ConvMode.HORIZONTAL)
        == [1, 1, 0, 1, 1]
    )


def test_blur_matrix_mean() -> None:
    assert BlurMatrix.MEAN() == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert BlurMatrix.MEAN(2, mode=ConvMode.VERTICAL) == [1, 1, 1, 1, 1]


def test_blur_matrix_binomial() -> None:
    assert BlurMatrix.BINOMIAL() == [1, 2, 1]
    assert BlurMatrix.BINOMIAL(3, mode=ConvMode.VERTICAL) == [1, 6, 15, 20, 15, 6, 1]


def test_blur_matrix_gauss() -> None:
    assert BlurMatrix.GAUSS() == [138.4479947510548, 1023, 138.4479947510548]
    assert (
        BlurMatrix.GAUSS(3, sigma=1.5, scale_value=1.0, mode=ConvMode.TEMPORAL)
        == [0.1353352832366127, 0.41111229050718745, 0.8007374029168081, 1.0, 0.8007374029168081, 0.41111229050718745, 0.1353352832366127]
    )

    assert BlurMatrix.GAUSS.from_radius(2) == [620.4808648860239, 1023, 620.4808648860239]
    assert BlurMatrix.GAUSS.get_taps(10) == 40


color_bars = core.colorbars.ColorBars(format=vs.YUV444P12).std.Loop(10)

clip_int8 = depth(color_bars, 8, dither_type=DitherType.NONE)
clip_fp16 = depth(color_bars, 16, sample_type=vs.FLOAT, dither_type=DitherType.NONE)
clip_fp32 = depth(color_bars, 32, sample_type=vs.FLOAT, dither_type=DitherType.NONE)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp16, clip_fp32])
@pytest.mark.parametrize("mode", [ConvMode.SQUARE, ConvMode.VERTICAL, ConvMode.HORIZONTAL, ConvMode.HV, ConvMode.TEMPORAL])
@pytest.mark.parametrize("planes", [None, 0, [1, 2]])
@pytest.mark.parametrize("bias", [None, 1.5])
@pytest.mark.parametrize("divisor", [None, 12])
@pytest.mark.parametrize("saturate", [True, False])
def test_blur_matrix_base(
    clip: vs.VideoNode,
    mode: Any,
    planes: PlanesT | None,
    bias: float | None,
    divisor: float | None,
    saturate: bool,
) -> None:
    blur_matrix = BlurMatrixBase([1] * 9, mode)

    try:
        blur_matrix(clip, planes, bias, divisor, saturate)
    except Exception as e:
        pytest.fail(_display_error_msg(clip, blur_matrix, e, mode=mode, planes=planes, bias=bias, divisor=divisor, saturate=saturate))
