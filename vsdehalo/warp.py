"""
This module implements dehaloing functions using warping-based techniques
as the core processing method.
"""

from __future__ import annotations

from typing import Literal, Sequence

from vsexprtools import norm_expr
from vsmasktools import EdgeDetect, EdgeDetectT, Morpho, PrewittStd
from vsrgtools import BlurMatrix, awarpsharp, box_blur, min_blur, remove_grain, repair
from vsrgtools.rgtools import Repair
from vstools import PlanesT, get_y, limiter, scale_mask, vs

__all__ = ["YAHR", "edge_cleaner"]


def edge_cleaner(
    clip: vs.VideoNode,
    strength: int = 5,
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
        strength: The strength of the edge cleaning. Default is 5.
        rmode: Repair mode to use for edge refinement. Default is 17.
        hot: If True, applies additional repair to hot edges. Default is False.
        smode: If True, applies a stronger cleaning mode. Default is False.
        edgemask: The edge detection method to use. Default is PrewittStd.
        planes: The planes to process. Default is 0 (luma only).

    Returns:
        The processed video clip with cleaned edges.
    """
    edgemask = EdgeDetect.ensure_obj(edgemask, edge_cleaner)

    if smode:
        strength += 4

    warped = awarpsharp(clip, blur=2, depth=strength, planes=planes)
    warped = repair(warped, clip, rmode, planes)

    y_mask = get_y(clip)

    mask = norm_expr(
        edgemask.edgemask(y_mask),
        "x {sc4} < 0 x {sc32} > range_in_max x ? ?",
        sc4=scale_mask(4, 8, clip),
        sc32=scale_mask(32, 8, clip),
        func=edge_cleaner,
    )
    mask = box_blur(mask)

    final = warped.std.MaskedMerge(clip, mask, planes)

    if hot:
        final = repair(final, clip, 2, planes)

    if smode:
        clean = remove_grain(y_mask, 17)

        diff = y_mask.std.MakeDiff(clean)

        mask = edgemask.edgemask(diff.std.Levels(scale_mask(40, 8, clip), scale_mask(168, 8, clip), 0.35))
        mask = norm_expr(
            remove_grain(mask, 7),
            "x {sc4} < 0 x {sc16} > range_in_max x ? ?",
            sc4=scale_mask(4, 8, clip),
            sc16=scale_mask(16, 8, clip),
            func=edge_cleaner,
        )

        final = final.std.MaskedMerge(clip, mask, planes)

    return final


def YAHR(  # noqa: N802
    clip: vs.VideoNode,
    blur: int = 3,
    depth: int | Sequence[int] = 32,
    expand: int | Literal[False] = 5,
    planes: PlanesT = 0,
) -> vs.VideoNode:
    """
    Applies YAHR (Yet Another Halo Remover) to reduce halos in a video clip.

    Args:
        clip: The input video clip to process.
        blur: The blur strength for the warping process. Default is 3.
        depth: The depth of the warping process. Default is 32.
        expand: The expansion factor for edge detection. Set to False to disable masking. Default is 5.
        planes: The planes to process. Default is 0 (luma only).

    Returns:
        The processed video clip with reduced halos.
    """
    warped = awarpsharp(clip, blur=blur, depth=depth, planes=planes)

    blur_diff, blur_warped_diff = [
        c.std.MakeDiff(BlurMatrix.BINOMIAL()(min_blur(c, 2, planes=planes), planes=planes), planes)
        for c in (clip, warped)
    ]

    rep_diff = repair(blur_diff, blur_warped_diff, 13, planes)
    yahr = norm_expr([clip, blur_diff, rep_diff], "x y z - -", planes)

    if expand is not False:
        v_edge = norm_expr(
            [clip, Morpho.maximum(clip, iterations=2, planes=planes)],
            "y x - 8 range_max * 255 / - 128 *",
            planes=planes,
            func=YAHR,
        )

        mask = norm_expr(
            [
                BlurMatrix.BINOMIAL(radius=expand * 2)(v_edge, planes=planes),
                BlurMatrix.BINOMIAL()(v_edge, planes=planes),
            ],
            "x 16 * range_max y - min",
            planes,
            func=YAHR,
        )

        yahr = clip.std.MaskedMerge(yahr, limiter(mask, planes=planes), planes)

    return yahr
