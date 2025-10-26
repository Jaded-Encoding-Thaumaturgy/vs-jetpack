import functools
import itertools
import random
from dataclasses import dataclass

import pytest

from vsdenoise import prefilter_to_full_range
from vsexprtools import inline_expr, norm_expr
from vsmasktools import Sobel
from vstools import ColorRange, ConvMode, core, get_video_format, vs


def _spawn_random(amount: int, format: int = vs.RGB24, length: int | None = None) -> list[vs.VideoNode]:
    clips = list[vs.VideoNode]()

    for _ in range(amount):
        clips.append(
            core.std.BlankClip(
                format=format,
                color=[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)][
                    : get_video_format(format).num_planes
                ],
                length=length,
            )
        )

    return clips


def test_inline_expr_simple_0() -> None:
    clip_a = core.std.BlankClip(format=vs.YUV420P8, color=[255, 0, 0])
    clip_b = core.std.BlankClip(format=vs.YUV420P8, color=[0, 255, 0])

    with inline_expr([clip_a, clip_b]) as ie:
        average = (ie.vars[0] + ie.vars[1]) / 2
        ie.out = average

    post = norm_expr([clip_a, clip_b], tuple(ie.out.to_str_per_plane()))

    for f_inline, f_post in zip(ie.clip.frames(close=True), post.frames(close=True)):
        assert f_inline[0][0, 0] == f_post[0][0, 0]


def test_inline_expr_simple_1() -> None:
    clips = _spawn_random(20)

    with inline_expr(clips) as ie:
        ie.out = sum(ie.vars) / len(ie.vars)

    post = norm_expr(clips, tuple(ie.out.to_str_per_plane()))

    for f_inline, f_post in zip(ie.clip.frames(close=True), post.frames(close=True)):
        assert f_inline[0][0, 0] == f_post[0][0, 0]


@pytest.mark.parametrize("format", [vs.YUV420P16, vs.YUV420PS, vs.GRAYS, vs.GRAY16])
def test_inline_expr_advanced(format: int) -> None:
    def pf_full(clip: vs.VideoNode, slope: float = 2.0, smooth: float = 0.0625) -> vs.VideoNode:
        with inline_expr(clip) as ie:
            x, *_ = ie.vars

            norm_luma = (x - x.PlaneMin) / (x.PlaneMax - x.PlaneMin)
            norm_luma = ie.op.clamp(norm_luma, 0, 1)

            curve_strength = (slope - 1) * smooth

            nonlinear_boost = curve_strength * ((1 + smooth) - ((1 + smooth) * smooth / (norm_luma + smooth)))

            weight_mul = nonlinear_boost + norm_luma * (1 - curve_strength)
            weight_mul *= x.RangeMax

            ie.out.y = weight_mul

            if ColorRange.from_video(clip).is_full() or clip.format.sample_type is vs.FLOAT:
                ie.out.uv = x
            else:
                chroma_expanded = ((x - x.Neutral) / (x.PlaneMax - x.PlaneMin) + 0.5) * x.RangeMax

                ie.out.uv = ie.op.round(chroma_expanded)

        return ColorRange.FULL.apply(ie.clip)

    clip, *_ = _spawn_random(1, format)

    inlined = pf_full(clip, 2.5, 0.0555)
    post = prefilter_to_full_range(clip, 2.5, 0.0555)

    for f_inline, f_post in zip(inlined.frames(close=True), post.frames(close=True)):
        assert f_inline[0][0, 0] == f_post[0][0, 0]


@dataclass
class _LowFreqSettings:
    freq_limit: float = 0.1
    freq_ratio_scale: float = 5.0
    max_reduction: float = 0.95


@pytest.mark.parametrize("strength", [0, 1.5])
@pytest.mark.parametrize("limit", [0, 0.3])
@pytest.mark.parametrize("low_freq", [_LowFreqSettings(freq_limit=0), _LowFreqSettings(freq_limit=0.1)])
def test_inline_expr_complex(strength: float, limit: float, low_freq: _LowFreqSettings) -> None:
    clip = core.std.StackHorizontal(
        [
            core.std.StackVertical(_spawn_random(5, vs.YUV420P16, 5)),
            core.std.StackVertical(_spawn_random(5, vs.YUV420P16, 5)),
        ]
    ).resize.Bicubic(100, 50)

    with inline_expr(clip) as ie:
        x = ie.vars[0]
        blur = ie.op.convolution(x, [1] * 9)
        sharp_diff = (x - blur) * strength
        effective_sharp_diff = sharp_diff
        if low_freq.freq_limit > 0:
            wider_blur = sum(x[i, j] for i, j in itertools.product([-2, 0, 2], repeat=2) if (i, j) != (0, 0))
            wider_blur = ie.as_var(wider_blur) / 9
            high_freq_indicator = abs(blur - wider_blur)

            texture_complexity = ie.op.max(abs(x - blur), abs(blur - wider_blur))

            freq_ratio = ie.op.max(high_freq_indicator / (texture_complexity + 0.01), 0)
            low_freq_factor = 1.0 - ie.op.min(
                freq_ratio * low_freq.freq_ratio_scale * low_freq.freq_limit, low_freq.max_reduction
            )

            effective_sharp_diff = effective_sharp_diff * low_freq_factor

        neighbors = [ie.as_var(x) for x in ie.op.matrix(x, 1, ConvMode.SQUARE, [(0, 0)])]

        local_min = functools.reduce(ie.op.min, neighbors)

        local_max = functools.reduce(ie.op.max, neighbors)

        if limit > 0:
            variance = sum(((n - x) ** 2 for n in neighbors)) / 8

            h_conv, v_conv = Sobel.matrices
            h_edge = ie.op.convolution(x, h_conv, divisor=False, saturate=False)
            v_edge = ie.op.convolution(x, v_conv, divisor=False, saturate=False)
            edge_strength = ie.op.sqrt(h_edge**2 + v_edge**2)

            edge_factor = 1.0 - ie.op.min(edge_strength * 0.01, limit)
            var_factor = 1.0 - ie.op.min(variance * 0.005, limit)
            adaptive_strength = edge_factor * var_factor

            effective_sharp_diff = effective_sharp_diff * adaptive_strength

            final_output = ie.op.clamp(x + effective_sharp_diff, local_min, local_max)
        else:
            final_output = x + effective_sharp_diff

        ie.out = final_output

    post = norm_expr(clip, tuple(ie.out.to_str_per_plane()))

    for f_inline, f_post in zip(ie.clip.frames(close=True), post.frames(close=True)):
        for i in range(len(f_inline)):
            h, w = f_inline[i].shape
            for y, x in itertools.product(range(h), range(w)):
                assert f_inline[i][y, x] == f_post[i][y, x]
