import math
from typing import Any

import pytest

from vsexprtools import ExprOp, ExprToken, expr_func
from vstools import ColorRange, core, vs

clip_yuv_limited = ColorRange.LIMITED.apply(core.std.BlankClip(width=2, height=2, format=vs.YUV420P8))


@pytest.mark.parametrize(
    ["token", "range_in", "expected"],
    [
        (ExprToken.LumaMin, None, 16),
        (ExprToken.ChromaMin, None, 16),
        (ExprToken.LumaMax, None, 235),
        (ExprToken.ChromaMax, None, 240),
        (ExprToken.Neutral, None, 128),
        (ExprToken.RangeHalf, None, 128),
        (ExprToken.RangeSize, None, 256),
        (ExprToken.LumaRangeMin, None, 16),
        (ExprToken.ChromaRangeMin, None, 16),
        (ExprToken.LumaRangeMax, None, 235),
        (ExprToken.ChromaRangeMax, None, 240),
        (ExprToken.LumaRangeInMin, None, 16),
        (ExprToken.LumaRangeInMin, ColorRange.LIMITED, 16),
        (ExprToken.LumaRangeInMin, ColorRange.FULL, 0),
        (ExprToken.ChromaRangeInMin, None, 16),
        (ExprToken.ChromaRangeInMin, ColorRange.LIMITED, 16),
        (ExprToken.ChromaRangeInMin, ColorRange.FULL, 0),
        (ExprToken.LumaRangeInMax, None, 235),
        (ExprToken.LumaRangeInMax, ColorRange.LIMITED, 235),
        (ExprToken.LumaRangeInMax, ColorRange.FULL, 255),
        (ExprToken.ChromaRangeInMax, None, 240),
        (ExprToken.ChromaRangeInMax, ColorRange.LIMITED, 240),
        (ExprToken.ChromaRangeInMax, ColorRange.FULL, 255),
    ],
)
def test_expr_token_get_value_limited(token: ExprToken, range_in: ColorRange | None, expected: float) -> None:
    assert token.get_value(clip_yuv_limited, None, range_in) == expected


@pytest.mark.parametrize(
    ["token", "chroma", "range_in", "expected"],
    [
        (ExprToken.RangeMin, None, None, 0),
        (ExprToken.RangeMin, False, None, 0),
        (ExprToken.RangeMin, True, None, 0),
        (ExprToken.RangeMax, None, None, 255),
        (ExprToken.RangeMax, False, None, 255),
        (ExprToken.RangeMax, True, None, 255),
        (ExprToken.RangeInMin, None, None, 16),
        (ExprToken.RangeInMin, False, None, 16),
        (ExprToken.RangeInMin, True, None, 16),
        (ExprToken.RangeInMin, None, ColorRange.FULL, 0),
        (ExprToken.RangeInMin, False, ColorRange.FULL, 0),
        (ExprToken.RangeInMin, True, ColorRange.FULL, 0),
        (ExprToken.RangeInMax, None, None, 235),
        (ExprToken.RangeInMax, False, None, 235),
        (ExprToken.RangeInMax, True, None, 240),
        (ExprToken.RangeInMax, None, ColorRange.FULL, 255),
        (ExprToken.RangeInMax, False, ColorRange.FULL, 255),
        (ExprToken.RangeInMax, True, ColorRange.FULL, 255),
    ],
)
def test_expr_token_get_value_limited_with_chroma(
    token: ExprToken, chroma: bool, range_in: ColorRange | None, expected: float
) -> None:
    assert token.get_value(clip_yuv_limited, chroma, range_in) == expected


def test_expr_token_getitem() -> None:
    assert ExprToken.LumaMax[2] == "ymax_z"


def test_expr_list() -> None: ...


def test_tuple_expr_list() -> None: ...


clip_int8 = core.std.BlankClip(None, 10, 10, vs.YUV420P8, length=10, color=[128, 64, 64], keep=True)
clip_fp32 = core.std.BlankClip(None, 10, 10, vs.YUV420PS, length=10, color=[0.5, 0.25, 0.25], keep=True)


def test_expr_op_str_pi() -> None:
    clip = expr_func(clip_fp32, str(ExprOp.PI))

    for f in clip.frames(close=True):
        assert f[0][0, 0] == pytest.approx(math.pi)


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


def test_expr_op_clamp() -> None:
    assert ExprOp.clamp(c="x").to_str() == "x ExprToken.RangeMin ExprToken.RangeMax clamp"
