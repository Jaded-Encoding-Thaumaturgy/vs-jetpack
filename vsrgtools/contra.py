from __future__ import annotations

from functools import partial
from typing import Callable

from vsexprtools import norm_expr
from vstools import (
    CustomValueError, GenericVSFunction, PlanesT, check_ref_clip,
    check_variable, iterate, normalize_planes, vs, core
)

from .blur import box_blur, median_blur, min_blur
from .enum import BlurMatrix, RemoveGrainMode, RemoveGrainModeT, RepairMode, RepairModeT
from .rgtools import remove_grain, repair
from .util import norm_rmode_planes

__all__ = [
    'contrasharpening',
    'contrasharpening_dehalo',
    'contrasharpening_median'
]


def contrasharpening(
    flt: vs.VideoNode, src: vs.VideoNode, radius: int | list[int] = 1,
    sharp: vs.VideoNode | GenericVSFunction | None = None,
    mode: RepairModeT = RepairMode.MINMAX_SQUARE3, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was previously removed.
    Script by Didée, at the VERY GRAINY thread (http://forum.doom9.org/showthread.php?p=1076491#post1076491)

    :param flt:         Filtered clip
    :param src:         Source clip
    :param radius:      Spatial radius for sharpening.
    :param sharp:       Optional pre-sharpened clip or function to use.
    :param mode:        Mode of rgvs.Repair to limit the difference
    :param planes:      Planes to process, defaults to None

    :return:            Contrasharpened clip
    """

    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    planes = normalize_planes(flt, planes)

    # Damp down remaining spots of the denoised clip
    if callable(sharp):
        sharp = sharp(flt)
    if isinstance(sharp, vs.VideoNode):
        sharp = sharp
    else:
        damp = min_blur(flt, radius, planes=planes)
        blurred = BlurMatrix.BINOMIAL(radius=radius)(damp, planes=planes)

    # Difference of a simple kernel blur
    diff_blur = core.std.MakeDiff(
        sharp if sharp else damp,
        flt if sharp else blurred,
        planes
    )

    # Difference achieved by the filtering
    diff_flt = src.std.MakeDiff(flt, planes)

    # Limit the difference to the max of what the filtering removed locally
    limit = repair(diff_blur, diff_flt, norm_rmode_planes(flt, mode, planes))

    # abs(diff) after limiting may not be bigger than before
    # Apply the limited difference (sharpening is just inverse blurring)
    expr = 'y neutral - Y! z neutral - Z! Y@ abs Z@ abs < Y@ Z@ ? x +'

    return norm_expr([flt, limit, diff_blur], expr, planes)


def contrasharpening_dehalo(
    flt: vs.VideoNode, src: vs.VideoNode, level: float = 1.4, alpha: float = 2.49, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param dehaloed:    Dehaloed clip
    :param src:         Source clip
    :param level:       Strength level
    :return:            Contrasharpened clip
    """
    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    planes = normalize_planes(flt, planes)

    rep_modes = norm_rmode_planes(flt, RepairMode.MINMAX_SQUARE1, planes)

    blur = BlurMatrix.BINOMIAL()(flt, planes)
    blur2 = median_blur(blur, 2, planes=planes)
    blur2 = iterate(blur2, partial(repair, repairclip=blur), 2, mode=rep_modes)

    return norm_expr(
        [flt, src, blur, blur2],
        'z a - {alpha} * {level} * D1! y x - D2! D1@ D2@ xor 0 D1@ abs D2@ abs < D1@ D2@ ? ? x +',
        planes, alpha=alpha, level=level
    )


def contrasharpening_median(
    flt: vs.VideoNode, src: vs.VideoNode,
    mode: RemoveGrainModeT | Callable[..., vs.VideoNode] = box_blur,
    planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param flt:         Filtered clip
    :param src:         Source clip
    :param mode:        Function or the RemoveGrain mode used to blur/repair the filtered clip.
    :param planes:      Planes to process, defaults to None
    :return:            Contrasharpened clip
    """
    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    planes = normalize_planes(flt, planes)

    if isinstance(mode, (int, list, RemoveGrainMode)):
        repaired = remove_grain(flt, norm_rmode_planes(flt, mode, planes))
    elif callable(mode):
        repaired = mode(flt, planes=planes)
    else:
        raise CustomValueError('Invalid mode or function passed!', contrasharpening_median)

    return norm_expr([flt, src, repaired], 'x dup + z - x y min x y max clip', planes)
