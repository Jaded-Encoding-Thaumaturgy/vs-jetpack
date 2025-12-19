from __future__ import annotations

from typing import Any

from jetpytools import FuncExcept, fallback

from vsexprtools import norm_expr
from vstools import VSFunctionNoArgs, core, shift_clip, shift_clip_multi, vs

from .funcs import vinverse
from .utils import telecine_patterns

__all__ = ["deblend", "deblend_bob", "deblend_fix_kf", "deblending_helper"]


def deblending_helper(
    deblended: vs.VideoNode,
    fieldmatched: vs.VideoNode,
    clip: vs.VideoNode | None = None,
    length: int = 5,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Helper function to select a deblended clip pattern from a fieldmatched clip.

    Args:
        deblended: Deblended clip.
        fieldmatched: Source after field matching, must have field=3 and possibly low cthresh.
        length: Length of the pattern.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Deblended clip.
    """
    func = func or deblending_helper

    inters = telecine_patterns(fallback(clip, fieldmatched), deblended, length, func)
    inters += [shift_clip(inter, 1) for inter in inters]

    shifted_clips = shift_clip_multi(fieldmatched)
    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    expr_clips = inters + shifted_clips

    return core.akarin.Select(
        expr_clips, prop_srcs, f"x._Combed N {length * 2} % {len(inters)} ? x._Combed y._Combed and 1 0 ? +"
    )


def deblend(
    clip: vs.VideoNode,
    fieldmatched: vs.VideoNode | None = None,
    decomber: VSFunctionNoArgs | None = vinverse,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Automatically deblends if normal field matching leaves 2 blends every 5 frames. Adopted from jvsfunc.

    Args:
        clip: Input source to fieldmatching.
        fieldmatched: Source after field matching, must have field=3 and possibly low cthresh.
        decomber: Optional post processing decomber after deblending and before pattern matching.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Deblended clip.
    """
    func = func or deblend

    deblended = norm_expr(shift_clip_multi(clip, (-1, 2)), "z a 2 / - y x 2 / - +", func=func)

    if decomber:
        deblended = decomber(deblended, **kwargs)

    if fieldmatched:
        deblended = deblending_helper(deblended, fieldmatched, clip, func=func)

    return deblended


def deblend_bob(
    bobbed: vs.VideoNode, fieldmatched: vs.VideoNode | None = None, func: FuncExcept | None = None
) -> vs.VideoNode:
    """
    Stronger version of [deblend][vsdeinterlace.deblend] that uses a bobbed clip to deblend. Adopted from jvsfunc.

    Args:
        bobbed: Bobbed source.
        fieldmatched: Source after field matching, must have field=3 and possibly low cthresh.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Deblended clip.
    """
    func = func or deblend_bob

    ab0, bc0, c0 = shift_clip_multi(bobbed[::2], (0, 2))
    bc1, ab1, a1 = shift_clip_multi(bobbed[1::2])

    deblended = norm_expr([a1, ab1, ab0, bc1, bc0, c0], "y x - z + b c - a + + 2 /", func=func)

    if fieldmatched:
        return deblending_helper(deblended, fieldmatched, func=func)

    return deblended


def deblend_fix_kf(deblended: vs.VideoNode, fieldmatched: vs.VideoNode, func: FuncExcept | None = None) -> vs.VideoNode:
    """
    Should be used after [deblend_bob][vsdeinterlace.deblend_bob] or [deblend][vsdeinterlace.deblend] to fix scene
    changes. Adopted from jvsfunc.

    Args:
        deblended: Deblended clip.
        fieldmatched: Fieldmatched clip used to debled, must have field=3 and possibly low cthresh.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Deblended clip with fixed blended keyframes.
    """
    func = func or deblend_fix_kf

    shifted_clips = shift_clip_multi(deblended)
    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    return core.akarin.Select(shifted_clips, prop_srcs, "x._Combed x.VFMSceneChange y.VFMSceneChange and 2 0 ? 1 ?")
