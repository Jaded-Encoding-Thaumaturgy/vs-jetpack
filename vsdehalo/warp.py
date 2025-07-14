"""
This module implements dehaloing functions using warping-based techniques
as the core processing method.
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence

from vsexprtools import norm_expr
from vsmasktools import EdgeDetect, EdgeDetectT, Morpho, PrewittStd
from vsrgtools import BlurMatrix, box_blur, min_blur, remove_grain, repair
from vsrgtools.rgtools import Repair
from vstools import (
    FunctionUtil,
    PlanesT,
    cround,
    get_y,
    limiter,
    padder,
    scale_mask,
    vs,
)

__all__ = ["YAHR", "edge_cleaner"]


def edge_cleaner(
    clip: vs.VideoNode,
    strength: float = 10,
    rmode: int | Repair.Mode = 17,
    hot: bool = False,
    smode: bool = False,
    edgemask: EdgeDetectT = PrewittStd,
    planes: PlanesT = 0,
) -> vs.VideoNode:
    """
    Cleans edges in a video clip by applying edge-aware processing.

    Args:
        clip: The input video clip to process.
        strength: The strength of the edge cleaning. Default is 10.
        rmode: Repair mode to use for edge refinement. Default is 17.
        hot: If True, applies additional repair to hot edges. Default is False.
        smode: If True, applies a stronger cleaning mode. Default is False.
        edgemask: The edge detection method to use. Default is PrewittStd.
        planes: The planes to process. Default is 0 (luma only).

    Returns:
        The processed video clip with cleaned edges.
    """
    func = FunctionUtil(clip, edge_cleaner, planes, (vs.YUV, vs.GRAY), (8, 16))

    edgemask = EdgeDetect.ensure_obj(edgemask, edge_cleaner)

    if smode:
        strength += 4

    padded = padder.MIRROR(func.work_clip, 6, 6, 6, 6)
    warped = padded.warp.AWarpSharp2(blur=1, depth=cround(strength / 2), planes=planes)
    warped = warped.std.Crop(6, 6, 6, 6)

    warped = repair(warped, func.work_clip, rmode, func.norm_planes)

    y_mask = get_y(func.work_clip)

    mask = norm_expr(
        edgemask.edgemask(y_mask),
        "x {sc4} < 0 x {sc32} > range_in_max x ? ?",
        sc4=scale_mask(4, 8, func.work_clip),
        sc32=scale_mask(32, 8, func.work_clip),
        func=edge_cleaner,
    )
    mask = box_blur(mask)

    final = warped.std.MaskedMerge(func.work_clip, mask)

    if hot:
        final = repair(final, func.work_clip, 2)

    if smode:
        clean = remove_grain(y_mask, 17)

        diff = y_mask.std.MakeDiff(clean)

        mask = edgemask.edgemask(
            diff.std.Levels(scale_mask(40, 8, func.work_clip), scale_mask(168, 8, func.work_clip), 0.35)
        )
        mask = norm_expr(
            remove_grain(mask, 7),
            "x {sc4} < 0 x {sc16} > range_in_max x ? ?",
            sc4=scale_mask(4, 8, func.work_clip),
            sc16=scale_mask(16, 8, func.work_clip),
            func=edge_cleaner,
        )

        final = final.std.MaskedMerge(func.work_clip, mask)

    return func.return_clip(final)


def YAHR(  # noqa: N802
    clip: vs.VideoNode, blur: int = 2, depth: int | Sequence[int] = 32, expand: float = 5, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Applies YAHR (Yet Another Halo Remover) to reduce halos in a video clip.

    Args:
        clip: The input video clip to process.
        blur: The blur strength for the warping process. Default is 2.
        depth: The depth of the warping process. Default is 32.
        expand: The expansion factor for edge detection. Default is 5.
        planes: The planes to process. Default is 0 (luma only).

    Returns:
        The processed video clip with reduced halos.
    """
    func = FunctionUtil(clip, YAHR, planes, (vs.YUV, vs.GRAY), (8, 16))

    padded = padder.MIRROR(func.work_clip, 6, 6, 6, 6)
    warped = padded.warp.AWarpSharp2(blur=blur, depth=depth, planes=func.norm_planes)
    warped = warped.std.Crop(6, 6, 6, 6)

    blur_diff, blur_warped_diff = [
        c.std.MakeDiff(
            BlurMatrix.BINOMIAL()(min_blur(c, 2, planes=func.norm_planes), planes=func.norm_planes), func.norm_planes
        )
        for c in (func.work_clip, warped)
    ]

    rep_diff = repair(blur_diff, blur_warped_diff, 13, func.norm_planes)

    yahr = func.work_clip.std.MakeDiff(blur_diff.std.MakeDiff(rep_diff, func.norm_planes), func.norm_planes)

    y_mask = get_y(func.work_clip)

    v_edge = norm_expr([y_mask, Morpho.maximum(y_mask, iterations=2)], "y x - 8 range_max * 255 / - 128 *", func=YAHR)

    mask1 = v_edge.tcanny.TCanny(sqrt(expand * 2), mode=-1)

    mask2 = BlurMatrix.BINOMIAL()(v_edge).std.Invert()

    mask = limiter(norm_expr([mask1, mask2], "x 16 * y min", func=YAHR))

    final = func.work_clip.std.MaskedMerge(yahr, mask, func.norm_planes)

    return func.return_clip(final)
