from __future__ import annotations

from math import sqrt
from typing import Sequence

from vsmasktools import EdgeDetect, EdgeDetectT, PrewittStd
from vsrgtools import min_blur, remove_grain, repair, BlurMatrix, box_blur
from vstools import (
    DitherType, PlanesT, core, cround, disallow_variable_format, disallow_variable_resolution, depth_func,
    get_peak_value, get_y, join, normalize_planes, padder, scale_mask, split, vs
)

__all__ = [
    'edge_cleaner', 'YAHR'
]


@disallow_variable_format
@disallow_variable_resolution
def edge_cleaner(
    clip: vs.VideoNode, strength: float = 10, rmode: int = 17,
    hot: bool = False, smode: bool = False, planes: PlanesT = 0,
    edgemask: EdgeDetectT = PrewittStd
) -> vs.VideoNode:
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('edge_cleaner: format not supported')

    edgemask = EdgeDetect.ensure_obj(edgemask, edge_cleaner)

    planes = normalize_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )
    assert work_clip.format

    peak = get_peak_value(work_clip)
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

    warped = repair(warped, work_clip, [
        rmode if i in planes else 0 for i in range(work_clip.format.num_planes)
    ])

    y_mask = get_y(work_clip)

    mask = edgemask.edgemask(y_mask).std.Expr(
        f'x {scale_mask(4, 8, work_clip)} < 0 x '
        f'{scale_mask(32, 8, work_clip)} > {peak} x ? ?'
    ).std.InvertMask()
    mask = box_blur(mask)

    final = work_clip.std.MaskedMerge(warped, mask)

    if hot:
        final = repair(final, work_clip, 2)

    if smode:
        clean = remove_grain(y_mask, 17)

        diff = y_mask.std.MakeDiff(clean)

        mask = edgemask.edgemask(
            diff.std.Levels(
                scale_mask(40, 8, work_clip),
                scale_mask(168, 8, work_clip),
                0.35
            )
        )
        sc4 = scale_mask(4, 8, work_clip)
        sc16 = scale_mask(16, 8, work_clip)
        mask = remove_grain(mask, 7).std.Expr(f'x {sc4} < 0 x {sc16} > {peak} x ? ?')

        final = final.std.MaskedMerge(work_clip, mask)

    if chroma:
        return join([final, *chroma], clip.format.color_family)

    return final


@disallow_variable_format
@disallow_variable_resolution
def YAHR(
    clip: vs.VideoNode, blur: int = 2, depth: int | Sequence[int] = 32, expand: float = 5, planes: PlanesT = 0
) -> vs.VideoNode:
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('YAHR: format not supported')

    planes = normalize_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )
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
        c.std.MakeDiff(
            BlurMatrix.BINOMIAL()(min_blur(c, 2, planes=planes), planes=planes), planes
        ) for c in (work_clip, warped)
    ]

    rep_diff = repair(blur_diff, blur_warped_diff, [
        13 if i in planes else 0 for i in range(work_clip.format.num_planes)
    ])

    yahr = work_clip.std.MakeDiff(blur_diff.std.MakeDiff(rep_diff, planes), planes)

    y_mask = get_y(work_clip)

    vEdge = core.std.Expr(
        [y_mask, y_mask.std.Maximum().std.Maximum()],
        f'y x - {8 * get_peak_value(y_mask) / 255} - 128 *'
    )

    mask1 = vEdge.tcanny.TCanny(sqrt(expand * 2), mode=-1)

    mask2 = BlurMatrix.BINOMIAL()(vEdge, planes=planes).std.Invert()

    mask = core.std.Expr([mask1, mask2], 'x 16 * y min 0 max 1 min' if is_float else 'x 16 * y min')

    final = work_clip.std.MaskedMerge(yahr, mask, planes)

    if chroma:
        return join([final, *chroma], clip.format.color_family)

    return final
