from __future__ import annotations

from typing import Callable, Sequence

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode,
    CustomValueError,
    GenericVSFunction,
    PlanesT,
    check_ref_clip,
    check_variable,
    core,
    normalize_planes,
    vs,
)

from .blur import box_blur, median_blur, min_blur
from .enum import BlurMatrix
from .rgtools import RemoveGrain, Repair, remove_grain, repair

__all__ = ["contrasharpening", "contrasharpening_dehalo", "contrasharpening_median"]


def contrasharpening(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    radius: int = 1,
    sharp: vs.VideoNode | GenericVSFunction[vs.VideoNode] | None = None,
    mode: int | Repair.Mode = repair.Mode.MINMAX_SQUARE3,
    planes: PlanesT = 0,
) -> ConstantFormatVideoNode:
    """
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was previously removed.
    Script by Didée, at the [VERY GRAINY thread](http://forum.doom9.org/showthread.php?p=1076491#post1076491).

    Args:
        flt: Filtered clip
        src: Source clip
        radius: Spatial radius for sharpening.
        sharp: Optional pre-sharpened clip or function to use.
        mode: Mode of Repair to limit the difference
        planes: Planes to process, defaults to None

    Returns:
        Contrasharpened clip
    """

    assert check_variable(src, contrasharpening)
    assert check_variable(flt, contrasharpening)
    check_ref_clip(src, flt, contrasharpening)

    # Damp down remaining spots of the denoised clip
    if callable(sharp):
        sharp = sharp(flt)
    if isinstance(sharp, vs.VideoNode):
        sharp = sharp
    else:
        damp = min_blur(flt, radius, planes=planes)
        blurred = BlurMatrix.BINOMIAL(taps=radius)(damp, planes=planes)

    # Difference of a simple kernel blur
    diff_blur = core.std.MakeDiff(sharp if sharp else damp, flt if sharp else blurred, planes)

    # Difference achieved by the filtering
    diff_flt = src.std.MakeDiff(flt, planes)

    # Limit the difference to the max of what the filtering removed locally
    limit = repair(diff_blur, diff_flt, mode, planes)

    # abs(diff) after limiting may not be bigger than before
    # Apply the limited difference (sharpening is just inverse blurring)
    expr = "y neutral - Y! z neutral - Z! Y@ abs Z@ abs < Y@ Z@ ? x +"

    return norm_expr([flt, limit, diff_blur], expr, planes, func=contrasharpening)


def contrasharpening_dehalo(
    flt: vs.VideoNode, src: vs.VideoNode, level: float = 1.4, alpha: float = 2.49, planes: PlanesT = 0
) -> ConstantFormatVideoNode:
    """
    Contrasharpening for dehalo.

    Args:
        flt: Dehaloed clip
        src: Source clip
        level: Strength level

    Returns:
        Contrasharpened clip
    """
    assert check_variable(src, contrasharpening_dehalo)
    assert check_variable(flt, contrasharpening_dehalo)
    check_ref_clip(src, flt, contrasharpening_dehalo)

    blur = BlurMatrix.BINOMIAL()(flt, planes)
    blur2 = median_blur(blur, 2, planes=planes)
    blur2 = repair(blur, repair(blur, blur2, repair.Mode.MINMAX_SQUARE1, planes), repair.Mode.MINMAX_SQUARE1, planes)

    return norm_expr(
        [flt, src, blur, blur2],
        "y x - D1! z a - {alpha} * {level} * D2! D1@ D2@ xor x x D1@ abs D2@ abs < D1@ D2@ ? + ?",
        planes,
        alpha=alpha,
        level=level,
        func=contrasharpening_dehalo,
    )


def contrasharpening_median(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    mode: int | Sequence[int] | RemoveGrain.Mode | Callable[..., ConstantFormatVideoNode] = box_blur,
    planes: PlanesT = 0,
) -> ConstantFormatVideoNode:
    """
    Contrasharpening median.

    Args:
        flt: Filtered clip
        src: Source clip
        mode: Function or the RemoveGrain mode used to blur/repair the filtered clip.
        planes: Planes to process, defaults to None

    Returns:
        Contrasharpened clip
    """
    assert check_variable(src, contrasharpening_median)
    assert check_variable(flt, contrasharpening_median)
    check_ref_clip(src, flt, contrasharpening_median)

    planes = normalize_planes(flt, planes)

    if isinstance(mode, (int, Sequence)):
        repaired = remove_grain(flt, mode, planes)
    elif callable(mode):
        repaired = mode(flt, planes=planes)
    else:
        raise CustomValueError("Invalid mode or function passed!", contrasharpening_median)

    return norm_expr([flt, src, repaired], "x dup + z - x y min x y max clip", planes, func=contrasharpening_median)
