from __future__ import annotations

import warnings

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode, KwargsNotNone, PlanesT, check_variable, normalize_seq, vs
)

from .aka_expr import removegrain_aka_exprs, repair_aka_exprs
from .enum import (
    ClenseMode, ClenseModeT, RemoveGrainMode, RemoveGrainModeT, RepairModeT, VerticalCleanerModeT
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

    if all(m in range(24 + 1) for m in mode):
        return clip.zsmooth.Repair(repairclip, mode)

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

    kwargs = KwargsNotNone(previous=previous_clip, next=next_clip)

    if mode == ClenseMode.NONE:
        return clip
    
    return getattr(clip.zsmooth, mode)(planes=planes, **kwargs)


def vertical_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT) -> ConstantFormatVideoNode:
    assert check_variable(clip, vertical_cleaner)

    mode = normalize_seq(mode, clip.format.num_planes)

    if not sum(mode):
        return clip

    return clip.zsmooth.VerticalCleaner(mode)


removegrain = remove_grain  # TODO: remove
