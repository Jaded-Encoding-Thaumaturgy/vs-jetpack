from __future__ import annotations

from functools import partial
from typing import Callable

import vapoursynth as vs
from vsexprtools import PlanesT, aka_expr_available, norm_expr, normalise_planes
from vsutil import disallow_variable_format, disallow_variable_resolution, get_neutral_value

from .blur import blur, box_blur, min_blur
from .enum import RemoveGrainMode, RemoveGrainModeT, RepairMode, RepairModeT
from .rgtools import removegrain, repair
from .util import iterate, norm_rmode_planes, wmean_matrix

__all__ = [
    'contrasharpening', 'contra',
    'contrasharpening_dehalo', 'contra_dehalo',
    'contrasharpening_median', 'contra_median'
]

core = vs.core


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening(
    flt: vs.VideoNode, src: vs.VideoNode, radius: int | None = None,
    mode: RepairModeT = RepairMode.MINMAX_SQUARE_REF3, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was previously removed.
    Script by Didée, at the VERY GRAINY thread (http://forum.doom9.org/showthread.php?p=1076491#post1076491)
    :param flt:         Filtered clip
    :param src:         Source clip
    :param radius:      Spatial radius for contra-sharpening (1-3). Default is 2 for HD / 1 for SD.
    :param mode:        Mode of rgvs.Repair to limit the difference
    :param planes:      Planes to process, defaults to None
    :return:            Contrasharpened clip
    """
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    if flt.format.sample_type == vs.INTEGER:
        neutral = [get_neutral_value(flt), get_neutral_value(flt, True)]
    else:
        neutral = [0.0]

    planes = normalise_planes(flt, planes)

    if radius is None:
        radius = 2 if flt.width > 1024 or flt.height > 576 else 1

    # Damp down remaining spots of the denoised clip
    mblur = min_blur(flt, radius, planes)

    rg11 = blur(mblur, radius, planes=planes)

    # Difference of a simple kernel blur
    diff_blur = mblur.std.MakeDiff(rg11, planes)

    # Difference achieved by the filtering
    diff_flt = src.std.MakeDiff(flt, planes)

    # Limit the difference to the max of what the filtering removed locally
    limit = repair(diff_blur, diff_flt, norm_rmode_planes(flt, mode, planes))

    # abs(diff) after limiting may not be bigger than before
    # Apply the limited difference (sharpening is just inverse blurring)
    if aka_expr_available:
        expr = 'x {mid} - LD! y {mid} - BD@ LD@ abs BD@ abs < LD@ BD@ ? z +'
    else:
        expr = 'x {mid} - abs y {mid} - abs < x y ? {mid} - z +'

    return norm_expr([limit, diff_blur, flt], expr, planes, mid=neutral)


@disallow_variable_format
@disallow_variable_resolution
def contrasharpening_dehalo(
    flt: vs.VideoNode, src: vs.VideoNode, level: float = 1.4, alpha: float = 2.49, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    :param dehaloed:    Dehaloed clip
    :param src:         Source clip
    :param level:       Strengh level
    :return:            Contrasharpened clip
    """
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening_dehalo: Clips must be the same format')

    planes = normalise_planes(flt, planes)

    rep_modes = norm_rmode_planes(flt, RepairMode.MINMAX_SQUARE1, planes)

    weighted = flt.std.Convolution(wmean_matrix, planes=planes)
    weighted2 = weighted.ctmf.CTMF(radius=2, planes=planes)
    weighted2 = iterate(weighted2, partial(repair, repairclip=weighted), 2, mode=rep_modes)

    neutral = [get_neutral_value(flt), get_neutral_value(flt, True)]

    if aka_expr_available:
        clips = [weighted, weighted2, src, flt]
        expr = f'x y - {alpha} * {level} * D! z a - DY! D@ DY@ xor 0 D@ abs DY@ abs < D@ DY@ ? ? a +'
    else:
        diff = norm_expr([weighted, weighted2], f'x y - {alpha} * {level} * {{mid}} +', planes, mid=neutral)

        clips = [diff, src, flt]
        expr = 'x {mid} - y z - xor 0 x {mid} - abs y z - abs < x {mid} - y z - ? ? z +'

    return norm_expr(clips, expr, planes, mid=neutral)


@disallow_variable_format
@disallow_variable_resolution
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
    assert flt.format and src.format

    if flt.format.id != src.format.id:
        raise ValueError('contrasharpening: Clips must be the same format')

    planes = normalise_planes(flt, planes)

    if isinstance(mode, (int, list, RemoveGrainMode)):
        repaired = removegrain(flt, norm_rmode_planes(flt, mode, planes))
    elif callable(mode):
        repaired = mode(flt, planes=planes)
    else:
        raise ValueError('contrasharpening_median: invalid mode/function passed!')

    if aka_expr_available:
        expr = 'x dup + z - D! x y < D@ x y clamp D@ y x clamp ?'
    else:
        expr = 'x dup + z - x y min max x y max min'

    return norm_expr([flt, src, repaired], expr, planes)


contra = contrasharpening
contra_dehalo = contrasharpening_dehalo
contra_median = contrasharpening_median
