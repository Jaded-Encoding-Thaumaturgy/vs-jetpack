"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, overload

from jetpytools import MISSING, CustomRuntimeError, FuncExcept, KwargsNotNone, MissingT, fallback, normalize_seq

from vsexprtools import ExprOp, ExprVars, combine_expr, norm_expr
from vskernels import Catrom, Kernel, KernelLike, Lanczos, Scaler, ScalerLike
from vstools import Planes, VSFunctionNoArgs, check_ref_clip, get_color_family, join, normalize_planes, scale_delta, vs

from .mvtools import MotionVectors, MVTools, MVToolsPreset, refine_blksize
from .prefilters import PrefilterLike

__all__ = [
    "ccd",
    "mc_clamp",
    "mc_degrain",
]


@overload
def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs | None = None,
    preset: MVToolsPreset = ...,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    overlap: int | tuple[int, int] = 2,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: Literal[False] = False,
    planes: Planes = None,
) -> vs.VideoNode: ...


@overload
def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs | None = None,
    preset: MVToolsPreset = ...,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    overlap: int | tuple[int, int] = 2,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    *,
    export_globals: Literal[True],
    planes: Planes = None,
) -> tuple[vs.VideoNode, MVTools]: ...


@overload
def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs | None = None,
    preset: MVToolsPreset = ...,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    overlap: int | tuple[int, int] = 2,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: bool = ...,
    planes: Planes = None,
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]: ...


def mc_degrain(
    clip: vs.VideoNode,
    vectors: MotionVectors | None = None,
    prefilter: vs.VideoNode | PrefilterLike | VSFunctionNoArgs | None = None,
    mfilter: vs.VideoNode | VSFunctionNoArgs | None = None,
    preset: MVToolsPreset = MVToolsPreset.HQ_SAD,
    tr: int = 1,
    blksize: int | tuple[int, int] = 16,
    overlap: int | tuple[int, int] = 2,
    refine: int = 1,
    thsad: int | tuple[int, int] = 400,
    thsad_recalc: int | None = None,
    limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None,
    export_globals: bool = False,
    planes: Planes = None,
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
        overlap: The blksize divisor for block overlap. Larger overlapping reduces blocking artifacts.
        refine: Number of times to recalculate motion vectors with halved block size.
        thsad: Defines the soft threshold of block sum absolute differences. Blocks with SAD above this threshold have
            zero weight for averaging (denoising). Blocks with low SAD have highest weight. The remaining weight is
            taken from pixels of source clip.
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
    mv_args = preset | KwargsNotNone(search_clip=prefilter)

    thsad_recalc = fallback(thsad_recalc, round((thsad[0] if isinstance(thsad, tuple) else thsad) / 2))

    mv = MVTools(clip, vectors=vectors, **mv_args)
    mfilter = mfilter(mv.clip) if callable(mfilter) else fallback(mfilter, mv.clip)

    if not vectors:
        mv.analyze(tr=tr, blksize=blksize, overlap=refine_blksize(blksize, overlap))

        for _ in range(refine):
            blksize = refine_blksize(blksize)
            mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=refine_blksize(blksize, overlap))

    den = mv.degrain(mfilter, mv.clip, None, tr, thsad, limit, thscd, planes)

    return (den, mv) if export_globals else den


def mc_clamp(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    mv_obj: MVTools,
    clamp: float | tuple[float, float] = 0,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Motion-compensated clamping of a filtered clip against the source.

    This function clamps the values of a filtered clip `flt` to those of the source clip `src`.
    but instead of using a spatial neighborhood (e.g. 3x3), it computes temporal min/max ranges
    from motion-compensated neighboring frames.

    This helps to preserve temporal consistency and prevent over/undershoot artifacts in motion areas.

    Args:
        flt: The filtered clip to be clamped.
        src: The original source clip, used as a reference for clamping.
        mv_obj: An MVTools object providing motion vectors for compensation.
        clamp: Clamping thresholds. Can be:

               - single value (applied symmetrically to undershoot and overshoot),
               - tuple (undershoot, overshoot) for asymmetric clamping.

            Values are scaled according to clip bit depth.
            Defaults to 0 (no additional clamping margin).
        func: Function returned for custom error handling. This should only be set by VS package developers.
        **kwargs: Additional keyword arguments passed to [mv_obj.compensate][vsdenoise.MVTools.compensate].

    Returns:
        The motion-compensated clamped clip.
    """
    func = func or mc_clamp

    check_ref_clip(src, flt, func)

    undershoot, overshoot = normalize_seq(clamp, 2)

    backward_comp, forward_comp = mv_obj.compensate(src, interleave=False, **kwargs)
    comp_clips = [src, *backward_comp, *forward_comp]

    evars = ExprVars(1, len(comp_clips) + 1, expr_src=True)

    return norm_expr(
        [flt, *comp_clips],
        "src0 {comp_min} {undershoot} - {comp_max} {overshoot} + clamp",
        undershoot=scale_delta(undershoot, 8, flt),
        overshoot=scale_delta(overshoot, 8, flt),
        comp_min=combine_expr(evars, ExprOp.MIN).to_str(),
        comp_max=combine_expr(evars, ExprOp.MAX).to_str(),
        func=func,
    )


def ccd(
    clip: vs.VideoNode,
    thr: float = 4,
    tr: int = 0,
    ref_points: Sequence[bool] = (True, True, False),
    scale: float | None = None,
    pscale: float = 0.0,
    chroma_upscaler: ScalerLike = Lanczos(
        format=lambda clip: clip.format.replace(color_family=vs.RGB, subsampling_w=0, subsampling_h=0)
    ),
    chroma_downscaler: KernelLike = Catrom,
    planes: Planes | MissingT = MISSING,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Camcorder Color Denoise (CCD) - chroma denoiser adapted from the original VirtualDub filter by Sergey Stolyarevsky.

    It works best on analog or low-quality digital sources such as VHS or DVD captures.

    Example usage:
        ```py
        denoised = ccd(clip, thr=6, tr=1, chroma_upscaler=Lanczos(format=vs.RGB48))
        ```

    Args:
        clip: Source clip.
        thr: Euclidean distance threshold for including pixel in the matrix.
            Higher values results in stronger denoising. Automatically scaled to all bit depths internally.
        tr: Temporal radius of processing. Higher values result in more denoising.
        ref_points: Specifies whether to use the low, medium, or high reference points (or any combination),
            respectively, in the processing matrix. The default uses the low and medium, but excludes the high points.
            See [zsmooth.CCD](https://github.com/adworacz/zsmooth?tab=readme-ov-file#ccd) for details.
        scale: Matrix size multiplier.

               - `1` = 25x25 matrix (original CCD)
               - `2` = 50x50, and so on.
        pscale: Correction strength for chroma ringing introduced during downscaling.
        chroma_upscaler: Upscaler for converting YUV input to full-resolution RGB before processing.
        chroma_downscaler: Kernel used to downscale the denoised RGB result back to the input format.
        planes: Planes to process. Defaults to chroma for YUV input, all planes otherwise.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Raises:
        CustomRuntimeError: If `chroma_upscaler` fails to produce full-resolution chroma.

    Returns:
        Denoised clip.
    """
    func = func or ccd

    if planes is MISSING:
        planes = [1, 2] if clip.format.color_family == vs.YUV else None

    planes = normalize_planes(clip, planes)

    if (clip.format.subsampling_w, clip.format.subsampling_h) == (0, 0):
        full = clip
        pscale = 1.0
    else:
        full = Scaler.ensure_obj(chroma_upscaler, func).scale(clip, clip.width, clip.height)

        if (full.format.subsampling_w, full.format.subsampling_h) != (0, 0):
            raise CustomRuntimeError("`chroma_upscaler` didn't upscale chroma planes.", func, repr(full))

    if get_color_family(full) != vs.RGB:
        rgb = vs.core.resize.Point(full, format=full.format.replace(color_family=vs.RGB).id)
    else:
        rgb = full

    processed = vs.core.zsmooth.CCD(rgb, thr, tr, ref_points, scale)

    if clip.format.id != processed.format.id:
        chroma_downscaler = Kernel.ensure_obj(chroma_downscaler, func)
        out = chroma_downscaler.resample(processed, clip, clip)

        if pscale != 1.0:
            no_denoise = chroma_downscaler.resample(rgb, clip, clip)
            out = norm_expr([clip, out, no_denoise], f"x z x - {pscale} * + y z - +", planes=planes, func=func)
    else:
        out = processed

    if planes != normalize_planes(clip, None):
        out = join({None: clip, tuple(planes): out})

    return out
