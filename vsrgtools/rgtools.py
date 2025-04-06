from __future__ import annotations

import warnings

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode, KwargsNotNone, PlanesT, check_variable, core, normalize_seq, padder, pick_func_stype, vs
)

from .aka_expr import removegrain_aka_exprs, repair_aka_exprs
from .enum import (
    ClenseMode, ClenseModeT, RemoveGrainMode, RemoveGrainModeT, RepairModeT, VerticalCleanerMode, VerticalCleanerModeT
)

__all__ = [
    'repair', 'remove_grain', 'removegrain',
    'clense', 'vertical_cleaner'
]


def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: RepairModeT) -> ConstantFormatVideoNode:
    assert check_variable(clip, repair)
    assert check_variable(repairclip, repair)

    mode = normalize_seq(mode, clip.format.num_planes)

    if not sum(mode):
        return clip

    if clip.format.sample_type == vs.INTEGER and all(m in range(24 + 1) for m in mode):
        sub = max(1 << clip.format.subsampling_w, 1 << clip.format.subsampling_h)

        with padder.ctx(sub, sub) as pad:
            padded = pad.MIRROR(clip)
            padded_repair = pad.MIRROR(repairclip)
            repaired = core.rgvs.Repair(padded, padded_repair, mode)
            return pad.CROP(repaired)

    return norm_expr([clip, repairclip], tuple([repair_aka_exprs[m]() for m in mode]), func=repair)


def remove_grain(clip: vs.VideoNode, mode: RemoveGrainModeT) -> ConstantFormatVideoNode:
    assert check_variable(clip, remove_grain)

    mode = normalize_seq(mode, clip.format.num_planes)

    for m in mode:
        if m in (
            RemoveGrainMode.MINMAX_MEDIAN, RemoveGrainMode.BINOMIAL_BLUR,
            RemoveGrainMode.BOX_BLUR_NO_CENTER, RemoveGrainMode.BOX_BLUR
        ):
            warnings.warn(
                f'Deprecated removegrain mode {m}! '
                f'Use {'median_blur' if m == RemoveGrainMode.MINMAX_MEDIAN else 'BlurMatrix'} instead!',
                DeprecationWarning
            )

    if not sum(mode):
        return clip

    if all(m in range(24 + 1) for m in mode):
        return clip.zsmooth.RemoveGrain(mode)

    return norm_expr(clip, tuple([removegrain_aka_exprs[m]() for m in mode]), func=remove_grain)


def clense(
    clip: vs.VideoNode, previous_clip: vs.VideoNode | None = None, next_clip: vs.VideoNode | None = None,
    mode: ClenseModeT = ClenseMode.NONE, planes: PlanesT = None
) -> ConstantFormatVideoNode:
    assert check_variable(clip, clense)

    warnings.warn('clense is deprecated! Use MeanMode instead!', DeprecationWarning)

    kwargs = KwargsNotNone(previous=previous_clip, next=next_clip)

    if mode == ClenseMode.NONE:
        return clip

    return pick_func_stype(clip, getattr(core.lazy.rgvs, mode), getattr(core.lazy.rgsf, mode))(
        clip, planes=planes, **kwargs
    )


def vertical_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT) -> ConstantFormatVideoNode:
    assert check_variable(clip, vertical_cleaner)

    mode = normalize_seq(mode, clip.format.num_planes)

    if not sum(mode):
        return clip

    for m in mode:
        if m == VerticalCleanerMode.MEDIAN:
            warnings.warn(f'Deprecated verticalcleaner mode {m}! Use median_blur instead!', DeprecationWarning)

    return pick_func_stype(clip, core.lazy.rgvs.VerticalCleaner, core.lazy.rgsf.VerticalCleaner)(clip, mode)


removegrain = remove_grain  # todo: remove
