"""
This modules implements dehaho functions based on spatial denoising operations.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, log
from typing import Any

from jetpytools import fallback, to_arr

from vsaa import NNEDI3
from vsdenoise import Prefilter, PrefilterLike, frequency_merge, nl_means
from vsexprtools import norm_expr
from vskernels import Catrom, Scaler, ScalerLike
from vsmasktools import Morpho, Prewitt
from vsrgtools import (
    contrasharpening_dehalo,
    gauss_blur,
    median_blur,
    repair,
)
from vstools import (
    FunctionUtil,
    Planes,
    check_progressive,
    check_ref_clip,
    core,
    scale_mask,
    vs,
)

__all__ = ["hq_dering", "smooth_dering", "vine_dehalo"]


def hq_dering(
    clip: vs.VideoNode,
    smooth: vs.VideoNode | PrefilterLike = Prefilter.MINBLUR,
    ringmask: vs.VideoNode | None = None,
    mrad: int = 1,
    msmooth: int = 1,
    minp: int = 1,
    mthr: float = 60,
    incedge: bool = False,
    contra: float = 1.4,
    drrep: int = 24,
    ref: vs.VideoNode | None = None,
    dark_thr: float | Sequence[float] = 12,
    bright_thr: float | Sequence[float] | None = None,
    elast: float | Sequence[float] = 2,
    planes: Planes = 0,
) -> vs.VideoNode:
    """
    Applies deringing by using a smart smoother near edges (where ringing occurs) only.

    Example usage:
        ```py
        from vsdenoise import Prefilter

        dering = hq_dering(clip, Prefilter.BILATERAL, ...)
        ```

        - Bringing back the DFTTest smoothed clip from havsfunc

        ```py
        is_hd = clip.width >= 1280 or clip.height >= 720
        sigma = 128
        sigma2 = sigma / 16
        smoothed = DFTTest().denoise(
            clip,
            [0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0],
            sbsize=8 if is_hd else 6,
            sosize=6 if is_hd else 4,
            tbsize=1,
        )

        dering = hq_dering(clip, smoothed, ...)
        ```

    Args:
        clip: Clip to process.
        smooth: Already smoothed clip, or a Prefilter.
        ringmask: Custom ringing mask.
        mrad: Expanding iterations of edge mask, higher value means more aggressive processing.
        msmooth: Inflating iterations of edge mask, higher value means smoother edges of mask.
        minp: Inpanding iterations of the edge mask, higher value means more aggressive processing.
        mthr: Threshold of the edge mask, lower value means more aggressive processing but for strong ringing, lower
            value will treat some ringing as edge, which "protects" this ringing from being processed.
        incedge: Whether to include edge in ring mask, by default ring mask only include area near edges.
        contra: Whether to use contra-sharpening to resharp deringed clip:

               - 0 means no contra
               - float: represents level for [contrasharpening_dehalo][vsrgtools.contrasharpening_dehalo]

        drrep: Use repair for details retention, recommended values are 24/23/13/12/1.
            See the [repair modes][vsrgtools.rgtools.Repair.Mode] for more information.
        planes: Planes to be processed.
        ref: LimitFilter parameter. Reference clip, to compute the weight to be applied on filtering diff.
        dark_thr: LimitFilter parameter. Threshold (8-bit scale) to limit dark filtering diff.
            Since `dark_thr` is "how much to limit undershoot", increasing this threshold
            will effectively remove more light halos.
        bright_thr: LimitFilter parameter. Threshold (8-bit scale) to limit bright filtering diff.
            Since `bright_thr` is "how much to limit overshoot", increasing this threshold
            will effectively remove more dark halos.
            None defaults to `bright_thr` / 4.
        elast: LimitFilter parameter. Elasticity of the soft threshold.

    Returns:
        Deringed clip.
    """
    func = FunctionUtil(clip, hq_dering, planes)

    if not isinstance(smooth, vs.VideoNode):
        smoothed = smooth(func.work_clip, planes)
    else:
        check_ref_clip(clip, smooth)

        smoothed = smooth

    if contra:
        smoothed = contrasharpening_dehalo(smoothed, func.work_clip, contra, planes=planes)

    repaired = repair(func.work_clip, smoothed, drrep, planes)

    limited = core.vszip.LimitFilter(
        repaired,
        func.work_clip,
        ref,
        dark_thr,
        fallback(bright_thr, [t / 4 for t in to_arr(dark_thr)]),  # type: ignore
        elast,
        planes,
    )

    if ringmask is None:
        edgemask = Prewitt.edgemask(func.work_clip, scale_mask(mthr, 8, 32), planes=planes)
        fmask = median_blur(edgemask, planes=planes).hysteresis.Hysteresis(edgemask, planes)

        omask = Morpho.expand(fmask, mrad, planes=planes)
        omask = Morpho.inflate(omask, msmooth, planes=planes)

        if incedge:
            ringmask = omask
        else:
            if not minp:
                imask = fmask
            elif not minp % 2:
                imask = Morpho.inpand(fmask, minp // 2, planes=planes, func=func.func)
            else:
                imask = Morpho.inpand(
                    Morpho.inflate(fmask, planes=planes, func=func.func), ceil(minp / 2), planes=planes, func=func.func
                )

            ringmask = norm_expr([omask, imask], "x mask_max y - * mask_max / 0 mask_max clip", planes, func=func.func)

    dering = func.work_clip.std.MaskedMerge(limited, ringmask, planes)

    return func.return_clip(dering)


smooth_dering = hq_dering


def vine_dehalo(
    clip: vs.VideoNode,
    strength: float | Sequence[float] = 16.0,
    sharp: float = 0.5,
    sigma: float | list[float] = 1.0,
    supersampler: ScalerLike = NNEDI3,
    downscaler: ScalerLike = Catrom,
    planes: Planes = 0,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Dehalo via non-local errors filtering.

    Args:
        clip: Clip to process.
        strength: Strength of nl_means filtering.
        sharp: Weight to blend supersampled clip.
        sigma: Gaussian sigma for filtering cutoff.
        supersampler: Scaler used for supersampling before dehaloing.
        downscaler: Scaler used for downscaling after supersampling.
        planes: Planes to be processed.
        **kwargs: Additional kwargs to be passed to nl_means.

    Returns:
        Dehaloed clip.
    """
    func = FunctionUtil(clip, vine_dehalo, planes)

    assert check_progressive(clip, func.func)

    strength = to_arr(strength)
    supersampler = Scaler.ensure_obj(supersampler, func.func)
    downscaler = Scaler.ensure_obj(downscaler, func.func)

    sharp = min(max(sharp, 0.0), 1.0)
    s = kwargs.pop("s", None)

    # Only God knows how these were derived.
    constants0 = 0.3926327792690057290863679493724 * sharp
    constants1 = 18.880334973195822973214959957208
    constants2 = 0.5862453661304626725671053478676

    weight = constants0 * log(1 + 1 / constants0)
    h_refine = [constants1 * (s / constants1) ** constants2 for s in strength]

    supersampled = supersampler.supersample(func.work_clip)
    supersampled = nl_means(supersampled, strength, tr=0, s=0, planes=planes, **kwargs)
    supersampled = downscaler.scale(supersampled, func.work_clip.width, func.work_clip.height)

    smoothed = nl_means(func.work_clip, strength, tr=0, s=0, planes=planes, **kwargs)
    smoothed = core.std.Merge(supersampled, smoothed, weight)

    highpassed = frequency_merge(
        func.work_clip,
        smoothed,
        mode_low=func.work_clip,
        mode_high=smoothed,
        lowpass=lambda clip, **kwargs: gauss_blur(clip, sigma, **kwargs),
        planes=planes,
    )

    refined = func.work_clip.std.MakeDiff(highpassed, planes)
    refined = nl_means(refined, h_refine, tr=0, s=s, ref=highpassed, planes=planes, **kwargs)
    refined = highpassed.std.MergeDiff(refined, planes)

    return func.return_clip(refined)
