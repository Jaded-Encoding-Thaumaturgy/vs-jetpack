import contextlib
import math
from collections.abc import Iterable, Sequence
from typing import Any, cast

import pytest
from jetpytools import clamp

from vsexprtools import ExprOp, ExprToken, expr_func, norm_expr
from vsexprtools.util import _get_akarin_expr_version
from vstools import ColorRange, core, get_lowest_value, get_peak_value, vs

clip_yuv_limited = ColorRange.LIMITED.apply(core.std.BlankClip(width=2, height=2, format=vs.YUV420P8))


@pytest.mark.parametrize(
    ["token", "range_in", "expected"],
    [
        (ExprToken.PlaneMin, None, 16),
        (ExprToken.MaskMax, None, 255),
        (ExprToken.Neutral, None, 128),
        (ExprToken.RangeMin, None, 0),
        (ExprToken.RangeMax, None, 255),
        (ExprToken.RangeSize, None, 256),
    ],
)
def test_expr_token_get_value_limited(token: ExprToken, range_in: ColorRange | None, expected: float) -> None:
    assert token.get_value(clip_yuv_limited, None, range_in) == expected


@pytest.mark.parametrize(
    ["token", "chroma", "range_in", "expected"],
    [
        (ExprToken.PlaneMax, False, None, 235),
        (ExprToken.PlaneMax, True, None, 240),
    ],
)
def test_expr_token_get_value_limited_with_chroma(
    token: ExprToken, chroma: bool, range_in: ColorRange | None, expected: float
) -> None:
    assert token.get_value(clip_yuv_limited, chroma, range_in) == expected


def test_expr_token_getitem() -> None:
    assert ExprToken.PlaneMax[2] == "plane_max_z"


def test_expr_list() -> None: ...


def test_tuple_expr_list() -> None: ...


clip_int8 = core.std.BlankClip(None, 10, 10, vs.YUV420P8, length=10, color=[128, 64, 64], keep=True)
clip_fp32 = core.std.BlankClip(None, 10, 10, vs.YUV420PS, length=10, color=[0.5, 0.25, 0.25], keep=True)


@pytest.mark.parametrize(
    ["clip", "expected"],
    [
        (clip_fp32.std.BlankClip(color=[0, 0, 0]), 0),
        (clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]), -1),
        (clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]), 1),
    ],
)
def test_expr_op_str_sgn(clip: vs.VideoNode, expected: Any) -> None:
    clip = expr_func(clip, f"x {ExprOp.SGN.convert_extra()}")

    for f in clip.frames(close=True):
        assert f[0][0, 0] == expected


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_neg(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.NEG.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == -f_in[0][0, 0]


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_tan(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.TAN.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.tan(f_in[0][0, 0]))


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_atan(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.ATAN.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.atan(f_in[0][0, 0]))


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_asin(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.ASIN.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.asin(f_in[0][0, 0]))


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0, 0, 0]),
        clip_fp32.std.BlankClip(color=[-0.5, -0.5, -0.5]),
        clip_fp32.std.BlankClip(color=[0.5, 0.5, 0.5]),
    ],
)
def test_expr_op_str_acos(input_clip: vs.VideoNode) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.ACOS.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == pytest.approx(math.acos(f_in[0][0, 0]))


def test_expr_op_str_ceil(input_clip: vs.VideoNode = clip_fp32) -> None:
    clip = expr_func(input_clip, f"x {ExprOp.CEIL.convert_extra()}")

    for f, f_in in zip(clip.frames(close=True), input_clip.frames(close=True)):
        assert f[0][0, 0] == math.ceil(f_in[0][0, 0])


@pytest.mark.parametrize(
    ["clip_a", "clip_b", "mask"],
    [
        (
            clip_fp32.std.BlankClip(color=[0.7451596557797295, -0.0897083306063644, -0.04091666168431174]),
            clip_fp32.std.BlankClip(color=[0.07807297537748636, -0.1883522041103517, 0.4003913983861609]),
            clip_fp32.std.BlankClip(color=[0.7994040706646394, 0.41575462854726997, 0.2803351038958446]),
        ),
        (
            clip_int8.std.BlankClip(color=[146, 52, 213]),
            clip_int8.std.BlankClip(color=[73, 125, 111]),
            clip_int8.std.BlankClip(color=[17, 213, 77]),
        ),
    ],
)
def test_expr_op_str_mmg(clip_a: vs.VideoNode, clip_b: vs.VideoNode, mask: vs.VideoNode) -> None:
    expr = norm_expr([clip_a, clip_b, mask], f"x y z {ExprOp.MMG.convert_extra()}")
    std = core.std.MaskedMerge(clip_a, clip_b, mask)

    for f_expr, f_std in zip(expr.frames(close=True), std.frames(close=True)):
        for i in range(f_expr.format.num_planes):
            assert f_expr[i][0, 0] == pytest.approx(f_std[i][0, 0], 1e-7)


@pytest.mark.parametrize(
    ["clip_a", "clip_b", "t"],
    [
        (
            clip_fp32.std.BlankClip(color=[0.4376836998088198, -0.19098552065281704, 0.3494137182200806]),
            clip_fp32.std.BlankClip(color=[0.8609435397529109, -0.2693605227534943, -0.055274461226768934]),
            0.595053469921834,
        ),
        (
            clip_int8.std.BlankClip(color=[25, 196, 106]),
            clip_int8.std.BlankClip(color=[209, 58, 143]),
            0.3852455650184188,
        ),
    ],
)
@pytest.mark.parametrize("legacy", (False, True))
def test_expr_op_str_lerp(clip_a: vs.VideoNode, clip_b: vs.VideoNode, t: float, legacy: bool) -> None:
    def lerp(x: float, y: float, z: float) -> float:
        return (1 - z) * x + z * y

    if not legacy:
        _get_akarin_expr_version.cache_clear()
    else:
        with contextlib.suppress(ValueError):
            cast(list[bytes], _get_akarin_expr_version()["expr_features"]).remove(bytes(ExprOp.LERP.value, "utf-8"))

    expr = norm_expr([clip_a, clip_b], f"x y {t} {ExprOp.LERP.convert_extra()}")

    for f_expr, f_clip_a, f_clip_b in zip(
        expr.frames(close=True), clip_a.frames(close=True), clip_b.frames(close=True)
    ):
        for i in range(f_expr.format.num_planes):
            result = f_expr[i][0, 0]
            expected = lerp(f_clip_a[i][0, 0], f_clip_b[i][0, 0], t)

            if expr.format.sample_type == vs.INTEGER:
                assert result == round(expected)
            else:
                assert result == pytest.approx(expected, rel=1e-7)


@pytest.mark.parametrize(
    "input_clip",
    [
        clip_fp32.std.BlankClip(color=[0.2024082058502097, 0.11259670650840858, 0.11218903003990488]),
        clip_int8.std.BlankClip(color=[25, 196, 106]),
    ],
)
@pytest.mark.parametrize(
    "coeffs",
    [
        [0.1463919616766296],
        [0.14764535441465032, 0.965697950149093],
        [0.5570375595288257, 0.46977481882228445, 0.17221700360247472],
        [0.823899580102139, 0.7176968019631801, 0.6786482169639257, 0.6262636707945298],
        [0.8470205375508597, 0.38407658280419255, 0.7566795503906608, 0.864981682467755, 0.023258137939439538],
        [
            0.13305974216840277,
            0.2768862621336875,
            0.5298824793762182,
            0.8059777592358868,
            0.9658320530651453,
            0.6593204600411745,
        ],
        [
            0.35397591567412223,
            0.2834925813867277,
            0.6198089162255391,
            0.4480158015911534,
            0.12853861477001016,
            0.011699127591215053,
            0.40144166883328813,
        ],
        [
            0.944126740128473,
            0.5613240631030031,
            0.2474108674169353,
            0.47631948030552884,
            0.4076605262025096,
            0.0022524594624114824,
            0.6833393252457135,
            0.7070468102173155,
        ],
        [
            0.9977651120275278,
            0.04279406564480259,
            0.4192512258573716,
            0.1960337949462796,
            0.040689767058342374,
            0.2633260527420662,
            0.327978750589612,
            0.7351580817827231,
            0.9421296510839722,
        ],
        [8],
        [3, 14],
        [252, 77, 158],
        [117, 206, 41, 94],
    ],
)
@pytest.mark.parametrize("legacy", (False, True))
def test_expr_op_str_polyval(input_clip: vs.VideoNode, coeffs: Sequence[float], legacy: bool) -> None:
    def polyval(coeffs: Iterable[float], x: float) -> float:
        result = 0
        for coeff in coeffs:
            result = result * x + coeff
        return result

    if not legacy:
        _get_akarin_expr_version.cache_clear()
    else:
        with contextlib.suppress(ValueError):
            cast(list[bytes], _get_akarin_expr_version()["expr_features"]).remove(b"polyval")

    expr = expr_func(
        input_clip, " ".join(str(c) for c in coeffs) + " x " + ExprOp.POLYVAL.convert_extra(len(coeffs) - 1)
    )

    for f_expr, f_in in zip(expr.frames(close=True), input_clip.frames(close=True)):
        for i in range(f_expr.format.num_planes):
            expected = polyval(coeffs, f_in[i][0, 0])

            if expr.format.sample_type == vs.INTEGER:
                clamped = clamp(
                    expected,
                    get_lowest_value(input_clip, range_in=ColorRange.FULL),
                    get_peak_value(input_clip, range_in=ColorRange.FULL),
                )
                assert f_expr[i][0, 0] == round(clamped)
            else:
                assert f_expr[i][0, 0] == pytest.approx(expected, rel=1e-7)


def test_expr_op_clamp() -> None:
    assert ExprOp.clamp(c="x").to_str() == "x range_min range_max clamp"
