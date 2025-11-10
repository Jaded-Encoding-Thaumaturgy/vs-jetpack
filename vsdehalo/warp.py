"""
This module implements dehaloing functions using warping-based techniques
as the core processing method.
"""

from __future__ import annotations

from typing import Literal, Sequence

from vsexprtools import norm_expr
from vsmasktools import EdgeDetect, EdgeDetectLike, Morpho, Prewitt
from vsrgtools import BlurMatrix, awarpsharp, box_blur, min_blur, remove_grain, repair
from vsrgtools.rgtools import Repair
from vstools import Planes, limiter, scale_mask, scale_value, vs

__all__ = ["YAHR", "edge_cleaner"]


def edge_cleaner(
    clip: vs.VideoNode,
    strength: int = 5,
    rmode: int | Repair.Mode = 17,
    hot: bool = False,
    smode: bool = False,
    edgemask: EdgeDetectLike = Prewitt,
    planes: Planes = 0,
) -> vs.VideoNode:
    """
    Cleans edges in a video clip by applying edge-aware processing.

    Args:
        clip: The input video clip to process.
        strength: The strength of the edge cleaning. Default is 5.
        rmode: Repair mode to use for edge refinement. Default is 17.
        hot: If True, applies additional repair to hot edges. Default is False.
        smode: If True, applies a stronger cleaning mode. Default is False.
        edgemask: The edge detection method to use. Default is Prewitt.
        planes: The planes to process. Default is 0 (luma only).

    Returns:
        The processed video clip with cleaned edges.
    """
    edgemask = EdgeDetect.ensure_obj(edgemask, edge_cleaner)
    lthr = scale_mask(4, 8, 32)
    expr = "x {thr} > mask_max x ?"

    if smode:
        strength += 4

    warped = awarpsharp(clip, blur=2, depth_h=strength, depth_v=strength, planes=planes)
    warped = repair(warped, clip, rmode, planes)

    mask = edgemask.edgemask(clip, lthr, planes=planes)
    mask = norm_expr(mask, expr, planes, thr=scale_mask(32, 8, clip), func=edge_cleaner)
    mask = box_blur(mask, planes=planes)

    final = warped.std.MaskedMerge(clip, mask, planes)

    if hot:
        final = repair(final, clip, 2, planes)

    if smode:
        clean = remove_grain(clip, 17)
        diff = clip.std.MakeDiff(clean)

        mask = edgemask.edgemask(
            diff.std.Levels(scale_value(40, 8, clip), scale_value(168, 8, clip), 0.35), lthr, planes=planes
        )
        mask = norm_expr(remove_grain(mask, 7, planes), expr, planes, thr=scale_mask(16, 8, clip), func=edge_cleaner)

        final = final.std.MaskedMerge(clip, mask, planes)

    return final


def YAHR(  # noqa: N802
    clip: vs.VideoNode,
    blur: int = 3,
    depth: int | Sequence[int] = 32,
    expand: int | Literal[False] = 5,
    shift: int = 8,
    planes: Planes = 0,
) -> vs.VideoNode:
    """
    Applies YAHR (Yet Another Halo Remover) to reduce halos in a video clip.

    Args:
        clip: The input video clip to process.
        blur: The blur strength for the warping process. Default is 3.
        depth: The depth of the warping process. Default is 32.
        expand: The expansion factor for edge detection. Set to False to disable masking. Default is 5.
        shift: Corrective shift for fine-tuning mask iterations.
        planes: The planes to process. Default is 0 (luma only).

    Returns:
        The processed video clip with reduced halos.
    """
    warped = awarpsharp(clip, blur=blur, depth_h=depth, depth_v=depth, planes=planes)

    blur_diff, blur_warped_diff = [
        c.std.MakeDiff(BlurMatrix.BINOMIAL()(min_blur(c, 2, planes=planes), planes=planes), planes)
        for c in (clip, warped)
    ]

    rep_diff = repair(blur_diff, blur_warped_diff, 13, planes)
    yahr = norm_expr([clip, blur_diff, rep_diff], "x y z - -", planes, func=YAHR)

    if expand is not False:
        v_edge = norm_expr(
            [clip, Morpho.maximum(clip, iterations=2, planes=planes)],
            "y x - {shift} mask_max * 255 / - 128 *",
            planes,
            shift=shift,
            func=YAHR,
        )

        mask = norm_expr(
            [
                BlurMatrix.BINOMIAL(radius=expand * 2)(v_edge, planes=planes),
                BlurMatrix.BINOMIAL()(v_edge, planes=planes),
            ],
            "x 16 * mask_max y - min",
            planes,
            func=YAHR,
        )

        yahr = clip.std.MaskedMerge(yahr, limiter(mask, mask=True, planes=planes, func=YAHR), planes)

    return yahr
