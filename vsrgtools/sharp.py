from __future__ import annotations

from scipy import interpolate
from functools import partial

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode, ConvMode, CustomTypeError, FunctionUtil, GenericVSFunction, 
    check_ref_clip, PlanesT, VSFunctionNoArgs, check_variable, normalize_planes, vs
)

from .blur import box_blur, gauss_blur, median_blur, min_blur
from .enum import BlurMatrix
from .limit import limit_filter
from .rgtools import repair
from .util import normalize_radius

__all__ = [
    'unsharpen',
    'unsharp_masked',
    'limit_usm',
    'fine_sharp',
    'soothe'
]


def unsharpen(
    clip: vs.VideoNode, strength: float = 1.0,
    blur: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] = partial(gauss_blur, sigma=1.5),
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:

    assert check_variable(clip, unsharpen)

    if callable(blur):
        blur = blur(clip)

    assert check_variable(blur, unsharpen)
    check_ref_clip(clip, blur, unsharpen)

    return norm_expr([clip, blur], f'x y - {strength} * x +', planes, func=unsharpen)


def unsharp_masked(
    clip: vs.VideoNode, radius: int | list[int] = 1, strength: float = 100.0, planes: PlanesT = None
) -> ConstantFormatVideoNode:

    assert check_variable(clip, unsharp_masked)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, unsharp_masked, radius, planes, strength=strength)

    blurred = BlurMatrix.LOG(radius, strength=strength)(clip, planes)

    return norm_expr([clip, blurred], 'x dup y - +', func=unsharp_masked)


def limit_usm(
    clip: vs.VideoNode, blur: int | vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = 1,
    thr: int | tuple[int, int] = 3, elast: float = 4.0, bright_thr: int | None = None,
    planes: PlanesT = None
) -> ConstantFormatVideoNode:
    """Limited unsharp_masked."""

    if callable(blur):
        blurred = blur(clip)
    elif isinstance(blur, vs.VideoNode):
        blurred = blur
    elif blur <= 0:
        blurred = min_blur(clip, -blur, planes=planes)
    elif blur == 1:
        blurred = BlurMatrix.BINOMIAL()(clip, planes)
    elif blur == 2:
        blurred = BlurMatrix.MEAN()(clip, planes)
    else:
        raise CustomTypeError("'blur' must be an int, clip or a blurring function!", limit_usm, blur)

    sharp = norm_expr([clip, blurred], 'x dup y - +', planes, func=limit_usm)

    return limit_filter(sharp, clip, thr=thr, elast=elast, bright_thr=bright_thr)


def fine_sharp(
    clip: vs.VideoNode, mode: int = 1, sstr: float = 2.0, cstr: float | None = None, xstr: float = 0.19,
    lstr: float = 1.49, pstr: float = 1.272, ldmp: float | None = None, planes: PlanesT = 0
) -> ConstantFormatVideoNode:
    from numpy import asarray

    func = FunctionUtil(clip, fine_sharp, planes)

    if cstr is None:
        cs = interpolate.CubicSpline(
            (0, 0.5, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 255.0),
            (0, 0.1, 0.6, 0.9, 1.0, 1.09, 1.15, 1.19, 1.249, 1.5)
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
        'range_size 256 / SCL! x y - SCL@ / D! D@ abs DA! DA@ {lstr} / 1 {pstr} / pow {sstr} * '
        'D@ DA@ 0.001 + / * D@ 2 pow D@ 2 pow {ldmp} + / * SCL@ * neutral +',
        lstr=lstr, pstr=pstr, sstr=sstr, ldmp=ldmp,
        func=func.func
    )

    sharp = func.work_clip

    if sstr:
        sharp = sharp.std.MergeDiff(diff)

    if cstr:
        diff = norm_expr(diff, 'x neutral - {cstr} * neutral +', cstr=cstr, func=func.func)
        diff = blur_kernel2(diff)
        sharp = sharp.std.MakeDiff(diff)

    if xstr:
        xysharp = norm_expr([sharp, box_blur(sharp)], 'x x y - 9.9 * +', func=func.func)
        rpsharp = repair(xysharp, sharp, 12)
        sharp = rpsharp.std.Merge(sharp, weight=[1 - xstr])

    return func.return_clip(sharp)


def soothe(
    flt: vs.VideoNode, src: vs.VideoNode,
    spatial_strength: int = 0, temporal_strength: int = 25,
    spatial_radius: int = 1, temporal_radius: int = 1,
    scenechange: bool = False,
    planes: PlanesT = 0
) -> ConstantFormatVideoNode:
    sharp_diff = src.std.MakeDiff(flt, planes)

    expr = (
        'x neutral - X! y neutral - Y! X@ 0 < Y@ 0 < xor X@ 100 / {strength} * '
        'X@ abs Y@ abs > X@ {strength} * Y@ 100 {strength} - * + 100 / X@ ? ? neutral +'
    )

    if spatial_strength:
        blurred = box_blur(sharp_diff, radius=spatial_radius, planes=planes)
        strength = 100 - abs(max(min(spatial_strength, 100), 0))
        sharp_diff = norm_expr([sharp_diff, blurred], expr, strength=strength, planes=planes, func=soothe)

    if temporal_strength:
        blurred = (
            BlurMatrix.MEAN(temporal_radius, mode=ConvMode.TEMPORAL)
            (sharp_diff, planes=planes, scenechange=scenechange)
        )
        strength = 100 - abs(max(min(temporal_strength, 100), -100))
        sharp_diff = norm_expr([sharp_diff, blurred], expr, strength=strength, planes=planes, func=soothe)

    return src.std.MakeDiff(sharp_diff, planes)
