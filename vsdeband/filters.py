from __future__ import annotations

from functools import partial
from math import sqrt

from vsexprtools import norm_expr
from vskernels import Bilinear, Point, Scaler, ScalerLike
from vsrgtools import box_blur, gauss_blur
from vstools import (
    ColorRange, ColorRangeT, PlanesT, check_ref_clip, check_variable, cround, depth, expect_bits, get_plane_sizes,
    normalize_planes, normalize_seq, vs
)

from .types import GuidedFilterMode

__all__ = [
    'guided_filter'
]


def guided_filter(
    clip: vs.VideoNode, guidance: vs.VideoNode | None = None, radius: int | list[int] | None = None,
    thr: float | list[float] = 1 / 3, mode: GuidedFilterMode = GuidedFilterMode.GRADIENT,
    use_gauss: bool = False, planes: PlanesT = None, range_in: ColorRangeT | None = None,
    down_ratio: int = 0, downscaler: ScalerLike = Point, upscaler: ScalerLike = Bilinear
) -> vs.VideoNode:
    assert check_variable(clip, guided_filter)

    planes = normalize_planes(clip, planes)

    downscaler = Scaler.ensure_obj(downscaler, guided_filter)
    upscaler = Scaler.ensure_obj(upscaler, guided_filter)

    range_in = ColorRange.from_param_or_video(range_in, clip, False, guided_filter)

    width, height = clip.width, clip.height

    thr = normalize_seq(thr, clip.format.num_planes)

    size = normalize_seq(
        [220, 225, 225] if range_in.is_full else 256, clip.format.num_planes
    )

    thr = [t / s for t, s in zip(thr, size)]

    if radius is None:
        radius = [
            round(max((w - 1280) / 160 + 12, (h - 720) / 90 + 12))
            for w, h in [
                get_plane_sizes(clip, i) for i in range(clip.format.num_planes)
            ]
        ]

    check_ref_clip(clip, guidance)

    p, bits = expect_bits(clip, 32)
    guidance_clip = g = depth(guidance, 32) if guidance is not None else p

    radius = normalize_seq(radius, clip.format.num_planes)

    if down_ratio:
        down_w, down_h = cround(width / down_ratio), cround(height / down_ratio)

        p = downscaler.scale(p, down_w, down_h)
        g = downscaler.scale(g, down_w, down_h) if guidance is not None else p

        radius = [cround(rad / down_ratio) for rad in radius]

    blur_filter = partial(
        gauss_blur, sigma=[rad / 2 * sqrt(2) for rad in radius], planes=planes
    ) if use_gauss else partial(
        box_blur, radius=[rad + 1 for rad in radius], planes=planes
    )

    blur_filter_corr = partial(
        gauss_blur, sigma=1 / 2 * sqrt(2), planes=planes
    ) if use_gauss else partial(box_blur, radius=2, planes=planes)

    mean_p = blur_filter(p)
    mean_I = blur_filter(g) if guidance is not None else mean_p

    I_square = norm_expr(g, 'x dup *', planes, func=guided_filter)
    corr_I = blur_filter(I_square)
    corr_Ip = blur_filter(norm_expr([g, p], 'x y *', planes, func=guided_filter)) if guidance is not None else corr_I

    var_I = norm_expr([corr_I, mean_I], 'x y dup * -', planes, func=guided_filter)
    cov_Ip = norm_expr([corr_Ip, mean_I, mean_p], 'x y z * -', planes, func=guided_filter) if guidance is not None else var_I

    if mode is GuidedFilterMode.ORIGINAL:
        a = norm_expr([cov_Ip, var_I], 'x y {thr} + /', planes, thr=thr, func=guided_filter)
    else:
        if set(radius) == {1}:
            var_I_1 = var_I
        else:
            mean_I_1 = blur_filter_corr(g)
            corr_I_1 = blur_filter_corr(I_square)
            var_I_1 = norm_expr([corr_I_1, mean_I_1], 'x y dup * -', planes, func=guided_filter)

        if mode is GuidedFilterMode.WEIGHTED:
            weight_in = var_I_1
        else:
            weight_in = norm_expr([var_I, var_I_1], 'x y * sqrt', planes, func=guided_filter)

        denominator = norm_expr([weight_in], '1 x {eps} + /', planes, eps=1e-06, func=guided_filter)

        denominator = denominator.std.PlaneStats(None, 0)

        weight = norm_expr([weight_in, denominator], 'x 1e-06 + y.PlaneStatsAverage *', planes, func=guided_filter)

        if mode is GuidedFilterMode.WEIGHTED:
            a = norm_expr([cov_Ip, var_I, weight], 'x y {thr} z / + /', planes, thr=thr, func=guided_filter)
        else:
            weight_in = weight_in.std.PlaneStats(None, 0)

            a = norm_expr(
                [cov_Ip, weight_in, weight, var_I],
                'x {thr} 1 1 1 -4 y.PlaneStatsMin y.PlaneStatsAverage 1e-6 - - / '
                'y y.PlaneStatsAverage - * exp + / - * z / + a {thr} z / + /',
                planes, thr=thr
            )

    b = norm_expr([mean_p, a, mean_I], 'x y z * -', planes, func=guided_filter)

    mean_a, mean_b = blur_filter(a), blur_filter(b)

    if down_ratio:
        mean_a = upscaler.scale(mean_a, width, height)
        mean_b = upscaler.scale(mean_b, width, height)

    q = norm_expr([mean_a, guidance_clip, mean_b], 'x y * z +', planes, func=guided_filter)

    return depth(q, bits)
