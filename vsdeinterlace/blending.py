from __future__ import annotations

from typing import Any

from vsexprtools import norm_expr
from vstools import (
    ConstantFormatVideoNode,
    VSFunctionNoArgs,
    check_ref_clip,
    check_variable,
    core,
    join,
    shift_clip,
    shift_clip_multi,
    vs,
)

from .funcs import vinverse
from .utils import telecine_patterns

__all__ = ["deblend", "deblend_bob", "deblend_fix_kf", "deblending_helper"]


def deblending_helper(deblended: vs.VideoNode, fieldmatched: vs.VideoNode, length: int = 5) -> ConstantFormatVideoNode:
    """
    Helper function to select a deblended clip pattern from a fieldmatched clip.

    Args:
        deblended: Deblended clip.
        fieldmatched: Source after field matching, must have field=3 and possibly low cthresh.
        length: Length of the pattern.

    Returns:
        Deblended clip.
    """

    assert check_variable(deblended, deblending_helper)
    assert check_variable(fieldmatched, deblending_helper)
    check_ref_clip(fieldmatched, deblended, deblending_helper)

    inters = telecine_patterns(fieldmatched, deblended, length)
    inters += [shift_clip(inter, 1) for inter in inters]

    inters.insert(0, fieldmatched)

    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    index_src = norm_expr(
        prop_srcs,
        "x._Combed N {length} % 1 + y._Combed {length} 0 ? + 0 ?",
        format=vs.GRAY8,
        func=deblending_helper,
        length=length,
    )

    return core.std.FrameEval(fieldmatched, lambda n, f: inters[f[0][0, 0]], index_src)


def deblend(
    clip: vs.VideoNode,
    fieldmatched: vs.VideoNode | None = None,
    decomber: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = vinverse,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    """
    Automatically deblends if normal field matching leaves 2 blends every 5 frames. Adopted from jvsfunc.

    Args:
        clip: Input source to fieldmatching.
        fieldmatched: Source after field matching, must have field=3 and possibly low cthresh.
        decomber: Optional post processing decomber after deblending and before pattern matching.

    Returns:
        Deblended clip.
    """

    deblended = norm_expr(shift_clip_multi(clip, (-1, 2)), "z a 2 / - y x 2 / - +", func=deblend)

    if decomber:
        deblended = decomber(deblended, **kwargs)

    if fieldmatched:
        deblended = deblending_helper(fieldmatched, deblended)

    return join(fieldmatched or clip, deblended)


def deblend_bob(bobbed: vs.VideoNode, fieldmatched: vs.VideoNode | None = None) -> ConstantFormatVideoNode:
    """
    Stronger version of `deblend` that uses a bobbed clip to deblend. Adopted from jvsfunc.

    Args:
        bobbed: Bobbed source.
        fieldmatched: Source after field matching, must have field=3 and possibly low cthresh.

    Returns:
        Deblended clip.
    """

    assert check_variable(bobbed, deblend_bob)

    ab0, bc0, c0 = shift_clip_multi(bobbed[::2], (0, 2))
    bc1, ab1, a1 = shift_clip_multi(bobbed[1::2])

    deblended = norm_expr([a1, ab1, ab0, bc1, bc0, c0], ("b", "y x - z + b c - a + + 2 /"), func=deblend_bob)

    if fieldmatched:
        return deblending_helper(fieldmatched, deblended)

    return deblended


def deblend_fix_kf(deblended: vs.VideoNode, fieldmatched: vs.VideoNode) -> ConstantFormatVideoNode:
    """
    Should be used after deblend/_bob to fix scene changes. Adopted from jvsfunc.

    Args:
        deblended: Deblended clip.
        fieldmatched: Fieldmatched clip used to debled, must have field=3 and possibly low cthresh.

    Returns:
        Deblended clip with fixed blended keyframes.
    """

    assert check_variable(deblended, deblend_fix_kf)

    shifted_clips = shift_clip_multi(deblended)
    prop_srcs = shift_clip_multi(fieldmatched, (0, 1))

    index_src = norm_expr(
        prop_srcs, "x._Combed x.VFMSceneChange and y.VFMSceneChange 2 0 ? 1 ?", format=vs.GRAY8, func=deblend_fix_kf
    )

    return core.std.FrameEval(deblended, lambda n, f: shifted_clips[f[0][0, 0]], index_src)
