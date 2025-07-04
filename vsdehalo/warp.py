from __future__ import annotations

from math import sqrt
from typing import Sequence

from vsexprtools import norm_expr
from vsmasktools import EdgeDetect, EdgeDetectT, Morpho, PrewittStd
from vsrgtools import BlurMatrix, box_blur, min_blur, remove_grain, repair
from vstools import (
    DitherType,
    InvalidColorFamilyError,
    PlanesT,
    check_variable,
    cround,
    depth_func,
    get_y,
    join,
    limiter,
    normalize_planes,
    padder,
    scale_mask,
    split,
    vs,
)

__all__ = ["YAHR", "edge_cleaner"]


def edge_cleaner(
    clip: vs.VideoNode,
    strength: float = 10,
    rmode: int = 17,
    hot: bool = False,
    smode: bool = False,
    planes: PlanesT = 0,
    edgemask: EdgeDetectT = PrewittStd,
) -> vs.VideoNode:
    assert check_variable(clip, edge_cleaner)

    InvalidColorFamilyError.check(clip, (vs.YUV, vs.GRAY), edge_cleaner)

    edgemask = EdgeDetect.ensure_obj(edgemask, edge_cleaner)

    planes = normalize_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip,)
    assert work_clip.format

    is_float = work_clip.format.sample_type == vs.FLOAT

    if smode:
        strength += 4

    padded = padder.MIRROR(work_clip, 6, 6, 6, 6)

    # warpsf is way too slow
    if is_float:
        padded = depth_func(padded, 16, vs.INTEGER, dither_type=DitherType.NONE)

    warped = padded.warp.AWarpSharp2(blur=1, depth=cround(strength / 2), planes=planes)

    warped = warped.std.Crop(6, 6, 6, 6)

    if is_float:
        warped = depth_func(
            warped, work_clip.format.bits_per_sample, work_clip.format.sample_type, dither_type=DitherType.NONE
        )

    warped = repair(warped, work_clip, [rmode if i in planes else 0 for i in range(work_clip.format.num_planes)])

    y_mask = get_y(work_clip)

    mask = norm_expr(
        edgemask.edgemask(y_mask),
        "x {sc4} < 0 x {sc32} > range_in_max x ? ?",
        sc4=scale_mask(4, 8, work_clip),
        sc32=scale_mask(32, 8, work_clip),
        func=edge_cleaner,
    )
    mask = box_blur(mask.std.InvertMask())

    final = work_clip.std.MaskedMerge(warped, mask)

    if hot:
        final = repair(final, work_clip, 2)

    if smode:
        clean = remove_grain(y_mask, 17)

        diff = y_mask.std.MakeDiff(clean)

        mask = edgemask.edgemask(diff.std.Levels(scale_mask(40, 8, work_clip), scale_mask(168, 8, work_clip), 0.35))
        mask = norm_expr(
            remove_grain(mask, 7),
            "x {sc4} < 0 x {sc16} > range_in_max x ? ?",
            sc4=scale_mask(4, 8, work_clip),
            sc16=scale_mask(16, 8, work_clip),
            func=edge_cleaner,
        )

        final = final.std.MaskedMerge(work_clip, mask)

    if chroma:
        return join([final, *chroma], clip.format.color_family)

    return final


def YAHR(  # noqa: N802
    clip: vs.VideoNode, blur: int = 2, depth: int | Sequence[int] = 32, expand: float = 5, planes: PlanesT = 0
) -> vs.VideoNode:
    assert check_variable(clip, edge_cleaner)

    InvalidColorFamilyError.check(clip, (vs.YUV, vs.GRAY), YAHR)

    planes = normalize_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip,)
    assert work_clip.format

    is_float = work_clip.format.sample_type == vs.FLOAT

    padded = padder.MIRROR(work_clip, 6, 6, 6, 6)

    # warpsf is way too slow
    if is_float:
        padded = depth_func(padded, 16, vs.INTEGER, dither_type=DitherType.NONE)

    warped = padded.warp.AWarpSharp2(blur=blur, depth=depth, planes=planes)

    warped = warped.std.Crop(6, 6, 6, 6)

    if is_float:
        warped = depth_func(
            warped, work_clip.format.bits_per_sample, work_clip.format.sample_type, dither_type=DitherType.NONE
        )

    blur_diff, blur_warped_diff = [
        c.std.MakeDiff(BlurMatrix.BINOMIAL()(min_blur(c, 2, planes=planes), planes=planes), planes)
        for c in (work_clip, warped)
    ]

    rep_diff = repair(
        blur_diff, blur_warped_diff, [13 if i in planes else 0 for i in range(work_clip.format.num_planes)]
    )

    yahr = work_clip.std.MakeDiff(blur_diff.std.MakeDiff(rep_diff, planes), planes)

    y_mask = get_y(work_clip)

    v_edge = norm_expr([y_mask, Morpho.maximum(y_mask, iterations=2)], "y x - 8 range_max * 255 / - 128 *", func=YAHR)

    mask1 = v_edge.tcanny.TCanny(sqrt(expand * 2), mode=-1)

    mask2 = BlurMatrix.BINOMIAL()(v_edge, planes=planes).std.Invert()

    mask = limiter(norm_expr([mask1, mask2], "x 16 * y min", func=YAHR))

    final = work_clip.std.MaskedMerge(yahr, mask, planes)

    if chroma:
        return join([final, *chroma], clip.format.color_family)

    return final
