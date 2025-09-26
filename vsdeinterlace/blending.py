from __future__ import annotations

from typing import Any

from jetpytools import FuncExcept

from vsexprtools import norm_expr
from vstools import VSFunctionNoArgs, check_ref_clip, core, join, shift_clip, shift_clip_multi, vs

from .funcs import vinverse
from .utils import telecine_patterns

__all__ = ["deblend", "deblend_bob", "deblend_fix_kf", "deblending_helper"]


def deblending_helper(
    deblended: vs.VideoNode, fieldmatched: vs.VideoNode, length: int = 5, func: FuncExcept | None = None
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

    check_ref_clip(fieldmatched, deblended, func)

    inters = telecine_patterns(fieldmatched, deblended, length, func)
    inters += [shift_clip(inter, 1) for inter in inters]

    inters.insert(0, fieldmatched)

    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    index_src = norm_expr(
        prop_srcs,
        "x._Combed N {length} % 1 + y._Combed {length} 0 ? + 0 ?",
        format=vs.GRAY8,
        func=func,
        length=length,
    )

    return core.std.FrameEval(fieldmatched, lambda n, f: inters[int(f[0][0, 0])], index_src)


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
        deblended = deblending_helper(fieldmatched, deblended, func=func)

    return join(fieldmatched or clip, deblended)


def deblend_bob(
    bobbed: vs.VideoNode, fieldmatched: vs.VideoNode | None = None, func: FuncExcept | None = None
) -> vs.VideoNode:
    """
    Stronger version of `deblend` that uses a bobbed clip to deblend. Adopted from jvsfunc.

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

    deblended = norm_expr([a1, ab1, ab0, bc1, bc0, c0], ("b", "y x - z + b c - a + + 2 /"), func=func)

    if fieldmatched:
        return deblending_helper(fieldmatched, deblended, func=func)

    return deblended


def deblend_fix_kf(deblended: vs.VideoNode, fieldmatched: vs.VideoNode, func: FuncExcept | None = None) -> vs.VideoNode:
    """
    Should be used after deblend/_bob to fix scene changes. Adopted from jvsfunc.

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

    index_src = norm_expr(
        prop_srcs, "x._Combed x.VFMSceneChange and y.VFMSceneChange 2 0 ? 1 ?", format=vs.GRAY8, func=func
    )

    return core.std.FrameEval(deblended, lambda n, f: shifted_clips[int(f[0][0, 0])], index_src)
