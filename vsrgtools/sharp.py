from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal, Sequence

from vsexprtools import norm_expr
from vskernels import Bilinear

if TYPE_CHECKING:
    from vsmasktools import EdgeDetectT

from vstools import (
    ChromaLocation,
    ConstantFormatVideoNode,
    ConvMode,
    FunctionUtil,
    GenericVSFunction,
    PlanesT,
    VSFunctionNoArgs,
    check_ref_clip,
    check_variable,
    core,
    get_y,
    join,
    pick_func_stype,
    scale_mask,
    vs,
)

from .blur import box_blur, gauss_blur, median_blur
from .enum import BlurMatrix
from .rgtools import repair

__all__ = ["awarpsharp", "fine_sharp", "soothe", "unsharpen"]


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


def awarpsharp(
    clip: vs.VideoNode,
    mask: EdgeDetectT | vs.VideoNode | None = None,
    thresh: int | float = 128,
    blur: int | GenericVSFunction[vs.VideoNode] | Literal[False] = 3,
    depth: int | Sequence[int] | None = None,
    chroma: bool = False,
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    """
    Sharpens edges by warping them.

    Args:
        clip: Clip to process. Must be either the same size as mask, or four times the size of mask in each dimension.
            The latter can be useful if better subpixel interpolation is desired. If clip is upscaled to four times the
            original size, it must be top-left aligned.
        mask: Edge mask.
        thresh: No pixel in the edge mask will have a value greater than thresh. Decrease for weaker sharpening.
        blur: Specifies the blur applied to the edge mask.
               - If an `int`, it sets the number of passes for the default `box_blur` filter.
               - If a callable, a custom blur function will be used instead.
               - If `False`, no blur will be applied.
        depth: Controls how far to warp. Negative values warp in the other direction, i.e. will blur the image instead
            of sharpening.
        chroma: Controls the chroma handling method. False will use the edge mask from the luma to warp the chroma.
            True will create an edge mask from each chroma channel and use those to warp each chroma channel
            individually.
        planes: Planes to process. Defaults to all.

    Returns:
        Warp-sharpened clip.
    """
    from vsmasktools import EdgeDetect, Sobel

    func = FunctionUtil(clip, awarpsharp, planes)

    thresh = scale_mask(thresh, 8, func.work_clip)
    chroma = True if func.work_clip.format.color_family is vs.RGB else chroma
    mask_planes = planes if chroma else 0

    if not isinstance(mask, vs.VideoNode):
        mask = EdgeDetect.ensure_obj(mask if mask else Sobel, awarpsharp).edgemask(
            func.work_clip, clamp=(0, thresh), planes=mask_planes
        )

    if isinstance(blur, int):
        if blur:
            blur = partial(box_blur, radius=2, passes=blur, planes=planes)
    else:
        mask = blur(mask, planes=mask_planes)

    if not chroma:
        loc = ChromaLocation.from_video(func.work_clip)

        mask = get_y(mask)
        mask = join(mask, mask, mask)
        mask = Bilinear().resample(mask, func.work_clip.format.id, chromaloc=loc)

    warp = pick_func_stype(func.work_clip, core.lazy.warp.AWarp, core.lazy.warpsf.AWarp)(
        func.work_clip, mask, depth, 1, planes
    )

    return func.return_clip(warp)


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
    from scipy import interpolate

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
        "range_size 256 / SCL! x y - SCL@ / D! D@ abs DA! D@ 2 pow DP! "
        "DA@ {lstr} / 1 {pstr} / pow {sstr} * "
        "DA@ D@ DA@ / D@ ? * DP@ DP@ {ldmp} + / * SCL@ * neutral +",
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
