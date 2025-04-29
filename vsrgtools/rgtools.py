from __future__ import annotations

from typing import Callable, Generic, Sequence

from jetpytools import CustomIntEnum, CustomStrEnum, P, R

from vsexprtools import norm_expr
from vstools import ConstantFormatVideoNode, KwargsNotNone, PlanesT, check_variable, normalize_seq, vs

from .aka_expr import removegrain_aka_exprs, repair_aka_exprs
from .enum import RemoveGrainModeT, RepairModeT, VerticalCleanerModeT

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

    if not sum(mode):
        return clip

    if all(m in range(24 + 1) for m in mode):
        return clip.zsmooth.RemoveGrain(mode)

    return norm_expr(clip, tuple([removegrain_aka_exprs[m]() for m in mode]), func=remove_grain)


class Clense(Generic[P, R]):
    """
    Class decorator that wraps the [clense][vsrgtools.rgtools.clense] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, clense_func: Callable[P, R]) -> None:
        self._func = clense_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Mode(CustomStrEnum):
        """
        Enum that specifies the temporal clense mode to use.

        Clense modes refer to different ways of applying temporal median filtering over multiple frames.

        Each mode maps to a function provided
        by the [zsmooth](https://github.com/adworacz/zsmooth?tab=readme-ov-file#clense--forwardclense--backwardclense) plugin.
        """

        NONE = ''
        """No clense filtering. Returns the original clip unchanged."""

        BACKWARD = 'BackwardClense'
        """Use the current and previous two frames for temporal median filtering."""

        FORWARD = 'ForwardClense'
        """Use the current and next two frames for temporal median filtering."""

        BOTH = 'Clense'
        """Standard clense: median of previous, current, and next frame."""

        def __call__(
            self,
            clip: vs.VideoNode,
            previous_clip: vs.VideoNode | None = None,
            next_clip: vs.VideoNode | None = None,
            planes: PlanesT = None
        ) -> ConstantFormatVideoNode:
            """
            Apply the selected clense mode to a clip using
            the [zsmooth](https://github.com/adworacz/zsmooth?tab=readme-ov-file#clense--forwardclense--backwardclense) plugin.

            :param clip:             Source clip to process.
            :param previous_clip:    Optional alternate clip to source previous frames. Defaults to `clip`.
            :param next_clip:        Optional alternate clip to source next frames. Defaults to `clip`.
            :param planes:           Planes to process. Defaults to all.
            :return:                 Clensed clip with temporal median filtering.
            """
            return clense(clip, previous_clip, next_clip, self, planes)


@Clense
def clense(
    clip: vs.VideoNode,
    previous_clip: vs.VideoNode | None = None,
    next_clip: vs.VideoNode | None = None,
    mode: Clense.Mode | str = Clense.Mode.NONE,
    planes: PlanesT = None
) -> ConstantFormatVideoNode:
    """
    Apply a clense (temporal median) filter based on the specified mode.

    Example:
        ```py
        clensed = clense(clip, ..., mode=clense.Mode.BOTH)
        ```

        Alternatively, directly using the enum:
        ```py
        clensed = clense.Mode.BOTH(clip, ...)
        ```

    :param clip:             Source clip to process.
    :param previous_clip:    Optional alternate clip to source previous frames. Defaults to `clip`.
    :param next_clip:        Optional alternate clip to source next frames. Defaults to `clip`.
    :param mode:             Mode of filtering. One of [Mode][vsrgtools.rgtools.Clense.Mode] or its string values.
    :param planes:           Planes to process. Defaults to all.
    :return:                 Clensed clip with temporal median filtering.
    """
    assert check_variable(clip, clense)

    kwargs = KwargsNotNone(previous=previous_clip, next=next_clip)

    if mode == Clense.Mode.NONE:
        return clip
    
    return getattr(clip.zsmooth, mode)(planes=planes, **kwargs)


def vertical_cleaner(clip: vs.VideoNode, mode: VerticalCleanerModeT) -> ConstantFormatVideoNode:
    assert check_variable(clip, vertical_cleaner)

    mode = normalize_seq(mode, clip.format.num_planes)

    if not sum(mode):
        return clip

    return clip.zsmooth.VerticalCleaner(mode)


removegrain = remove_grain  # TODO: remove
