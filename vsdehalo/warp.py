from __future__ import annotations

from math import sqrt

import vapoursynth as vs
from vsmask.better_vsutil import join, split
from vsmask.edge import EdgeDetect, PrewittStd
from vsrgtools import box_blur, min_blur, removegrain, repair
from vsrgtools.util import PlanesT, cround, normalise_planes, wmean_matrix
from vsutil import Dither
from vsutil import Range as CRange
from vsutil import depth as vdepth
from vsutil import disallow_variable_format, disallow_variable_resolution, get_peak_value, get_y, scale_value

from .utils import pad_reflect

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def edge_cleaner(
    clip: vs.VideoNode, strength: float = 10, rmode: int = 17,
    hot: bool = False, smode: bool = False, edgemask: EdgeDetect = PrewittStd()
) -> vs.VideoNode:
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('edge_cleaner: format not supported')

    bits = clip.format.bits_per_sample

    peak = get_peak_value(clip)

    clip_y, *chroma = split(clip)

    if smode:
        strength += 4

    main = pad_reflect(clip_y, 6, 6, 6, 6)

    # warpsf is way too slow
    main = vdepth(main, 16, vs.INTEGER, dither_type=Dither.NONE) if clip.format.sample_type == vs.FLOAT else main
    main = main.warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    main = vdepth(main, bits, clip.format.sample_type, dither_type=Dither.NONE)

    main = repair(main, clip_y, rmode)

    mask = edgemask.edgemask(clip_y).std.Expr(
        f'x {scale_value(4, 8, bits, CRange.FULL)} < 0 x {scale_value(32, 8, bits, CRange.FULL)} > {peak} x ? ?'
    ).std.InvertMask().std.Convolution([1] * 9)

    final = clip_y.std.MaskedMerge(main, mask)

    if hot:
        final = repair(final, clip_y, 2)

    if smode:
        clean = removegrain(clip_y, 17)

        diff = clip_y.std.MakeDiff(clean)

        expr = f'x {scale_value(4, 8, bits, CRange.FULL)} < 0 x {scale_value(16, 8, bits, CRange.FULL)} > {peak} x ? ?'

        mask = edgemask.edgemask(
            diff.std.Levels(scale_value(40, 8, bits, CRange.FULL), scale_value(168, 8, bits, CRange.FULL), 0.35)
        )
        mask = removegrain(mask, 7).std.Expr(expr)

        final = final.std.MaskedMerge(clip_y, mask)

    return join([final, *chroma], clip.format.color_family)


@disallow_variable_format
@disallow_variable_resolution
def YAHR(clip: vs.VideoNode, blur: int = 2, depth: int = 32, expand: float = 5, planes: PlanesT = 0) -> vs.VideoNode:
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError('edge_cleaner: format not supported')

    planes = normalise_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )
    assert work_clip.format

    is_float = work_clip.format.sample_type == vs.FLOAT

    padded = pad_reflect(work_clip, 6, 6, 6, 6)

    # warpsf is way too slow
    if is_float:
        padded = vdepth(padded, 16, vs.INTEGER, dither_type=Dither.NONE)

    warped = padded.warp.AWarpSharp2(blur=blur, depth=depth, planes=planes)

    warped = warped.std.Crop(6, 6, 6, 6)

    if is_float:
        warped = vdepth(
            warped, work_clip.format.bits_per_sample, work_clip.format.sample_type, dither_type=Dither.NONE
        )

    blur_diff, blur_warped_diff = [
        c.std.MakeDiff(
            box_blur(min_blur(c, 2, planes), wmean_matrix, planes), planes
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

    mask2 = box_blur(vEdge, wmean_matrix).std.Invert()

    mask = core.std.Expr([mask1, mask2], 'x 16 * y min 0 max 1 min' if is_float else 'x 16 * y min')

    final = clip.std.MaskedMerge(yahr, mask, planes)

    if chroma:
        return join([final, *chroma], clip.format.color_family)

    return final
