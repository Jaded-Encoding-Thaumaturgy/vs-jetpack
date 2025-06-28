from __future__ import annotations

from functools import partial

from scipy import interpolate

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode,
    ConvMode,
    FunctionUtil,
    GenericVSFunction,
    PlanesT,
    VSFunctionNoArgs,
    check_ref_clip,
    check_variable,
    vs,
)

from .blur import box_blur, gauss_blur, median_blur
from .enum import BlurMatrix
from .rgtools import repair

__all__ = ["fine_sharp", "soothe", "unsharpen"]


def unsharpen(
    clip: vs.VideoNode,
    strength: float = 1.0,
    blur: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] = partial(gauss_blur, sigma=1.5),
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    assert check_variable(clip, unsharpen)

    if callable(blur):
        blur = blur(clip)

    assert check_variable(blur, unsharpen)
    check_ref_clip(clip, blur, unsharpen)

    return norm_expr([clip, blur], f"x y - {strength} * x +", planes, func=unsharpen)


def fine_sharp(
    clip: vs.VideoNode,
    mode: int = 1,
    sstr: float = 2.0,
    cstr: float | None = None,
    xstr: float = 0.19,
    lstr: float = 1.49,
    pstr: float = 1.272,
    ldmp: float | None = None,
    planes: PlanesT = 0,
) -> ConstantFormatVideoNode:
    from numpy import asarray

    func = FunctionUtil(clip, fine_sharp, planes)

    if cstr is None:
        cs = interpolate.CubicSpline(
            (0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 255.0), (0, 0.1, 0.6, 0.9, 1.0, 1.09, 1.15, 1.19, 1.249, 1.5)
        )
        cstr = float(cs(asarray(sstr)))

    if ldmp is None:
        ldmp = sstr + 0.1

    blur_kernel = BlurMatrix.BINOMIAL()
    blur_kernel2: GenericVSFunction[ConstantFormatVideoNode] = blur_kernel

    if mode < 0:
        cstr **= 0.8
        blur_kernel2 = box_blur

    mode = abs(mode)

    if mode == 1:
        blurred = median_blur(blur_kernel(func.work_clip))
    elif mode > 1:
        blurred = blur_kernel(median_blur(func.work_clip))
    if mode == 3:
        blurred = median_blur(blurred)

    diff = norm_expr(
        [func.work_clip, blurred],
        "range_size 256 / SCL! x y - SCL@ / D! D@ abs DA! DA@ {lstr} / 1 {pstr} / pow {sstr} * "
        "D@ DA@ 0.001 + / * D@ 2 pow D@ 2 pow {ldmp} + / * SCL@ * neutral +",
        lstr=lstr,
        pstr=pstr,
        sstr=sstr,
        ldmp=ldmp,
        func=func.func,
    )

    sharp = func.work_clip

    if sstr:
        sharp = sharp.std.MergeDiff(diff)

    if cstr:
        diff = norm_expr(diff, "x neutral - {cstr} * neutral +", cstr=cstr, func=func.func)
        diff = blur_kernel2(diff)
        sharp = sharp.std.MakeDiff(diff)

    if xstr:
        xysharp = norm_expr([sharp, box_blur(sharp)], "x x y - 9.9 * +", func=func.func)
        rpsharp = repair(xysharp, sharp, 12)
        sharp = rpsharp.std.Merge(sharp, weight=[1 - xstr])

    return func.return_clip(sharp)


def soothe(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    spatial_strength: float = 0.0,
    temporal_strength: float = 0.75,
    spatial_radius: int = 1,
    temporal_radius: int = 1,
    scenechange: bool = False,
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    sharp_diff = src.std.MakeDiff(flt, planes)

    expr = (
        "x neutral - X! y neutral - Y! X@ Y@ xor X@ {strength} * neutral + X@ abs Y@ abs > x y - {strength} * y + x ? ?"
    )

    if spatial_strength:
        blurred = box_blur(sharp_diff, spatial_radius, planes=planes)
        sharp_diff = norm_expr([sharp_diff, blurred], expr, strength=1.0 - spatial_strength, planes=planes, func=soothe)

    if temporal_strength:
        blurred = box_blur(sharp_diff, temporal_radius, 1, ConvMode.TEMPORAL, planes, scenechange=scenechange)
        sharp_diff = norm_expr(
            [sharp_diff, blurred], expr, strength=1.0 - temporal_strength, planes=planes, func=soothe
        )

    return src.std.MakeDiff(sharp_diff, planes)
