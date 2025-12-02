from __future__ import annotations

from collections.abc import Sequence
from typing import overload

from jetpytools import FuncExcept, StrList, fallback, to_arr

from vsexprtools import ExprOp, ExprVars, norm_expr
from vsrgtools import box_blur, gauss_blur
from vstools import (
    ColorRange,
    DitherType,
    depth,
    get_peak_value,
    get_sample_type,
    get_y,
    limiter,
    plane,
    scale_value,
    vs,
)

from .edge import MinMax
from .morpho import Morpho

__all__ = ["adg_mask", "flat_mask", "retinex", "texture_mask"]


@overload
def adg_mask(
    clip: vs.VideoNode, luma_scaling: float = 8.0, relative: bool = False, func: FuncExcept | None = None
) -> vs.VideoNode: ...


@overload
def adg_mask(
    clip: vs.VideoNode, luma_scaling: Sequence[float], relative: bool = False, func: FuncExcept | None = None
) -> list[vs.VideoNode]: ...


def adg_mask(
    clip: vs.VideoNode,
    luma_scaling: float | Sequence[float] = 8.0,
    relative: bool = False,
    func: FuncExcept | None = None,
) -> vs.VideoNode | list[vs.VideoNode]:
    """
    Generates an adaptive grain mask based on each frame's average luma and pixel value.

    This function is primarily used to create masks for adaptive grain applications but can be used
    in other scenarios requiring luminance-aware masking.

    Args:
        clip: The clip to process.
        luma_scaling: Controls the strength of the adaptive mask. Can be a single float or a sequence of floats. Default
            is 8.0. Negative values invert the mask behavior.
        relative: Enables relative computation based on pixel-to-average luminance ratios.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A single mask or a list of masks (if `luma_scaling` is a sequence), corresponding to the input clip.
    """
    func = func or adg_mask

    luma = plane(clip, 0)

    if clip.format.bits_per_sample > 16 or relative:
        y, y_inv = luma.std.PlaneStats(prop="P"), luma.std.Invert().std.PlaneStats(prop="P")

        peak = get_peak_value(y)

        is_integer = y.format.sample_type == vs.INTEGER

        x_string, aft_int = (f"x {peak} / ", f" {peak} * 0.5 +") if is_integer else ("x ", "0 1 clamp")

        if relative:
            x_string += "Y! Y@ 0.5 < x.PMin 0 max 0.5 / log Y@ * x.PMax 1.0 min 0.5 / log Y@ * ? "

        x_string += "0 0.999 clamp X!"

        def _adgfunc(luma: vs.VideoNode, ls: float) -> vs.VideoNode:
            return norm_expr(
                luma,
                f"{x_string} 1 X@ X@ X@ X@ X@ "
                "18.188 * 45.47 - * 36.624 + * 9.466 - * 1.124 + * - "
                f"x.PAverage 2 pow {ls} * pow {aft_int}",
                func=func,
            )
    else:
        y, y_inv = luma.std.PlaneStats(), luma.std.Invert().std.PlaneStats()

        def _adgfunc(luma: vs.VideoNode, ls: float) -> vs.VideoNode:
            return luma.adg.Mask(ls)

    scaled_clips = [_adgfunc(y_inv if ls < 0 else y, abs(ls)) for ls in to_arr(luma_scaling)]

    if isinstance(luma_scaling, Sequence):
        return scaled_clips

    return scaled_clips[0]


@limiter
def retinex(
    clip: vs.VideoNode,
    sigma: Sequence[float] = [25, 80, 250],
    lower_thr: float = 0.001,
    upper_thr: float = 0.001,
    fast: bool = True,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Multi-Scale Retinex (MSR) implementation for dynamic range and contrast enhancement.

    More information [here](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex).

    Args:
        clip: Input video clip.
        sigma: List of Gaussian sigmas for MSR. Using 3 scales (e.g., [25, 80, 250]) balances speed and quality.
        lower_thr: Lower threshold percentile for output normalization (0-1, exclusive). Affects shadow contrast.
        upper_thr: Upper threshold percentile for output normalization (0-1, exclusive). Affects highlight compression.
        fast: Enables fast mode using downscaled approximation and simplifications. Default is True.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Processed luma-enhanced clip.
    """
    func = func or retinex

    sigma = sorted(sigma)

    y = get_y(clip)

    y = y.std.PlaneStats()
    is_float = get_sample_type(y) is vs.FLOAT

    if is_float:
        luma_float = norm_expr(y, "x x.PlaneStatsMin - x.PlaneStatsMax x.PlaneStatsMin - /", func=func)
    else:
        luma_float = norm_expr(
            y, "1 x.PlaneStatsMax x.PlaneStatsMin - / x x.PlaneStatsMin - *", None, vs.GRAYS, func=func
        )

    slen, slenm = len(sigma), len(sigma) - 1

    expr_msr = StrList([f"{x} 0 <= 1 x {x} / 1 + ? " for x in ExprVars(1, slen + (not fast))])

    if fast:
        expr_msr.append("x.PlaneStatsMax 0 <= 1 x x.PlaneStatsMax / 1 + ? ")
        sigma = sigma[:-1]

    expr_msr.extend(ExprOp.ADD * slenm)
    expr_msr.append(f"log {slen} /")

    msr = norm_expr([luma_float, *(gauss_blur(luma_float, i, _fast=fast) for i in sigma)], expr_msr, func=func)

    msr_stats = msr.vszip.PlaneMinMax(lower_thr, upper_thr)

    expr_balance = "x x.psmMin - x.psmMax x.psmMin - /"

    if not is_float:
        expr_balance = f"{expr_balance} plane_max plane_min - * plane_min + round plane_min plane_max clamp"

    return norm_expr(msr_stats, expr_balance, None, y, func=func)


def flat_mask(src: vs.VideoNode, radius: int = 5, thr: float = 0.011, gauss: bool = False) -> vs.VideoNode:
    luma = get_y(src)

    blur = gauss_blur(luma, radius * 0.361083333) if gauss else box_blur(luma, radius)

    blur, mask = depth(blur, 8), depth(luma, 8)

    mask = mask.vszip.AdaptiveBinarize(blur, int(scale_value(thr, 32, blur)))

    return depth(mask, luma, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range_out=ColorRange.FULL)


def texture_mask(
    clip: vs.VideoNode,
    rady: int = 2,
    radc: int | None = None,
    blur: float = 8,
    thr: float = 0.2,
    stages: list[tuple[int, int]] = [(60, 2), (40, 4), (20, 2)],
    points: list[tuple[bool, float]] = [(False, 1.75), (True, 2.5), (True, 5), (False, 10)],
) -> vs.VideoNode:
    levels = [x for x, _ in points]
    points_ = [scale_value(x, 8, clip) for _, x in points]
    thr = scale_value(thr, 8, 32, ColorRange.FULL)

    qm, peak = len(points), get_peak_value(clip)

    rmask = MinMax(rady, fallback(radc, rady)).edgemask(clip, lthr=0)

    emask = clip.std.Prewitt()

    rm_txt = ExprOp.MIN(
        rmask, (Morpho.minimum(Morpho.binarize(emask, thr, 1.0, 0), iterations=it) for thr, it in stages)
    )

    expr = [f"x {points_[0]} < x {points_[-1]} > or 0"]

    for x in range(len(points_) - 1):
        if points_[x + 1] < points_[-1]:
            expr.append(f"x {points_[x + 1]} <=")

        if levels[x] == levels[x + 1]:
            expr.append(f"{peak if levels[x] else 0}")
        else:
            mean = peak * (levels[x + 1] - levels[x]) / (points_[x + 1] - points_[x])
            expr.append(f"x {points_[x]} - {mean} * {peak * levels[x]} +")

    weighted = norm_expr(rm_txt, [expr, ExprOp.TERN * (qm - 1)], func=texture_mask)

    weighted = box_blur(weighted, blur) if isinstance(blur, int) else gauss_blur(weighted, blur)

    return norm_expr(weighted, f"x {peak * thr} - {1 / (1 - thr)} *", func=texture_mask)
