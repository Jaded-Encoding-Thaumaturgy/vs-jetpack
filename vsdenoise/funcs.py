"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence, overload

from jetpytools import MISSING, CustomRuntimeError, FuncExceptT, MissingT

from vsexprtools import norm_expr
from vskernels import Catrom, Kernel, KernelLike, Scaler, ScalerLike
from vsscale import ArtCNN
from vstools import (
    ConstantFormatVideoNode,
    KwargsNotNone,
    PlanesT,
    VSFunctionNoArgs,
    check_ref_clip,
    check_variable,
    check_variable_format,
    fallback,
    get_color_family,
    get_subsampling,
    get_video_format,
    join,
    normalize_planes,
    normalize_seq,
    scale_delta,
    vs,
)

from .mvtools import MotionVectors, MVTools, MVToolsPreset
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


def ccd(
    clip: vs.VideoNode,
    thr: float = 4,
    tr: int = 0,
    ref_points: Sequence[bool] = (True, True, False),
    scale: float | None = None,
    pscale: float = 0.0,
    chroma_upscaler: ScalerLike = ArtCNN.R8F64_Chroma,
    chroma_downscaler: KernelLike = Catrom,
    planes: PlanesT | MissingT = MISSING,
    func: FuncExceptT | None = None,
) -> vs.VideoNode:
    """
    Camcorder Color Denoise is a VirtualDub filter originally made by Sergey Stolyarevsky.

    It's a chroma denoiser that works great on old sources such as VHS and DVD.

    It works as a convolution of nearby pixels determined by ``ref_points``.
    If the euclidean distance between the RGB values of the center pixel and a given pixel in the convolution
    matrix is less than the threshold, then this pixel is considered in the average.

    Example usage:

        ```py
        denoised = ccd(clip, thr=6, tr=1, chroma_uspcaler=Bicubic(format=vs.RGB48))
        ```

    Args:
        clip: Source clip.
        thr: Euclidean distance threshold for including pixel in the matrix.
            Higher values results in stronger denoising. Automatically scaled to all bit depths internally.
        tr: Temporal radius of processing. Higher values result in more denoising. Defaults to 0.
        ref_points: Specifies whether to use the low, medium, or high reference points (or any combination),
            respectively, in the processing matrix. The default uses the low and medium, but excludes the high points.
            See [zsmooth.CCD](https://github.com/adworacz/zsmooth?tab=readme-ov-file#ccd) for more information.
        scale: Multiplier for the size of the matrix.
            `scale=1` corresponds with a 25x25 matrix (just like the original CCD implementation by Sergey).
            `scale=2` is a 50x50 matrix, and so on.
        pscale: Scale factor for the source clip-denoised process change.
        chroma_upscaler: Chroma upscaler to apply before processing if input clip is YUV.
            Defaults to ArtCNN.R8F64_Chroma.
        chroma_downscaler: Chroma downscaler to apply after processing if input clip is YUV. Defaults to Catrom.
        planes: Planes to process. Default is chroma planes is clip is YUV, else all planes.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Raises:
        CustomRuntimeError: If the `chroma_upscaler` didn't upscale the chroma planes.

    Returns:
        Denoised clip.
    """
    func = func or ccd

    assert check_variable_format(clip, func)

    if planes is MISSING:
        planes = [1, 2] if clip.format.color_family == vs.YUV else None

    planes = normalize_planes(clip, planes)

    if get_subsampling(clip) not in ["444", None]:
        full = Scaler.ensure_obj(chroma_upscaler, func).scale(clip, clip.width, clip.height)
    else:
        full = clip

    full_format = get_video_format(full)

    if (full_format.subsampling_w, full_format.subsampling_h) != (0, 0):
        raise CustomRuntimeError("`chroma_upscaler` didn't upscale chroma planes.", func, repr(full_format))

    if get_color_family(clip) != vs.RGB:
        rgb = vs.core.resize.Point(full, format=full_format.replace(color_family=vs.RGB).id)
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
