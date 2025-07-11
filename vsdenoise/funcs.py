"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from vstools import (
    ConstantFormatVideoNode,
    KwargsNotNone,
    PlanesT,
    VSFunctionNoArgs,
    check_ref_clip,
    check_variable,
    fallback,
    normalize_seq,
    scale_delta,
    vs,
)

from .mvtools import MotionVectors, MVTools, MVToolsPreset
from .prefilters import PrefilterLike

__all__ = [
    "mc_clamp",
    "mc_degrain",
]


@overload
def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    preset: MVToolsPreset = ...,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: Literal[False] = ...,
    planes: PlanesT = None,
) -> vs.VideoNode: ...


@overload
def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    preset: MVToolsPreset = ...,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: Literal[True] = ...,
    planes: PlanesT = None,
) -> tuple[vs.VideoNode, MVTools]: ...


@overload
def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    preset: MVToolsPreset = ...,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: bool = ...,
    planes: PlanesT = None,
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]: ...


def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    preset: MVToolsPreset = MVToolsPreset.HQ_SAD,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: bool = False,
    planes: PlanesT = None,
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    """
    Perform temporal denoising using motion compensation.

    Motion compensated blocks from previous and next frames are averaged with the current frame.
    The weighting factors for each block depend on their SAD from the current frame.

    Args:
        clip: The clip to process.
        vectors: Motion vectors to use.
        prefilter: Filter or clip to use when performing motion vector search.
        mfilter: Filter or clip to use where degrain couldn't find a matching block.
        preset: MVTools preset defining base values for the MVTools object. Default is HQ_SAD.
        tr: The temporal radius. This determines how many frames are analyzed before/after the current frame.
        blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
        refine: Number of times to recalculate motion vectors with halved block size.
        thsad: Defines the soft threshold of block sum absolute differences. Blocks with SAD above this threshold have
            zero weight for averaging (denoising). Blocks with low SAD have highest weight. The remaining weight is
            taken from pixels of source clip.
        thsad2: Define the SAD soft threshold for frames with the largest temporal distance. The actual SAD threshold
            for each reference frame is interpolated between thsad (nearest frames) and thsad2 (furthest frames). Only
            used with the FLOAT MVTools plugin.
        thsad_recalc: Only bad quality new vectors with a SAD above this will be re-estimated by search. thsad value is
            scaled to 8x8 block size.
        limit: Maximum allowed change in pixel values.
        thscd: Scene change detection thresholds:

               - First value: SAD threshold for considering a block changed between frames.
               - Second value: Percentage of changed blocks needed to trigger a scene change.

        export_globals: Whether to return the MVTools object.
        planes: Which planes to process. Default: None (all planes).

    Returns:
        Motion compensated and temporally filtered clip with reduced noise. If export_globals is true: A tuple
        containing the processed clip and the MVTools object.
    """

    def _floor_div_tuple(x: tuple[int, int]) -> tuple[int, int]:
        return x[0] // 2, x[1] // 2

    mv_args = preset | KwargsNotNone(search_clip=prefilter)

    blksize = blksize if isinstance(blksize, tuple) else (blksize, blksize)
    thsad_recalc = fallback(thsad_recalc, round((thsad[0] if isinstance(thsad, tuple) else thsad) / 2))
    mfilter = mfilter(clip) if callable(mfilter) else fallback(mfilter, clip)

    mv = MVTools(clip, vectors=vectors, planes=planes, **mv_args)

    if not vectors:
        mv.analyze(tr=tr, blksize=blksize, overlap=_floor_div_tuple(blksize))

        for _ in range(refine):
            blksize = _floor_div_tuple(blksize)
            overlap = _floor_div_tuple(blksize)

            mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=overlap)

    den = mv.degrain(mfilter, mv.clip, None, tr, thsad, thsad2, limit, thscd)

    return (den, mv) if export_globals else den


def mc_clamp(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    mv_obj: MVTools,
    ref: vs.VideoNode | None = None,
    clamp: int | float | tuple[int | float, int | float] = 0,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    from vsexprtools import norm_expr
    from vsrgtools import MeanMode

    assert check_variable(flt, mc_clamp)
    assert check_variable(src, mc_clamp)
    check_ref_clip(src, flt, mc_clamp)

    ref = fallback(ref, src)

    undershoot, overshoot = normalize_seq(clamp, 2)

    backward_comp, forward_comp = mv_obj.compensate(ref, interleave=False, **kwargs)

    comp_min = MeanMode.MINIMUM([ref, *backward_comp, *forward_comp])
    comp_max = MeanMode.MAXIMUM([ref, *backward_comp, *forward_comp])

    return norm_expr(
        [flt, comp_min, comp_max],
        "x y {undershoot} - z {overshoot} + clip",
        undershoot=scale_delta(undershoot, 8, flt),
        overshoot=scale_delta(overshoot, 8, flt),
    )
