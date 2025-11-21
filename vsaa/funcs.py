from __future__ import annotations

from functools import partial
from typing import Any, Literal

from jetpytools import CustomValueError, fallback

from vsexprtools import norm_expr
from vskernels import Box, Catrom, NoScale, Scaler, ScalerLike, is_noscale_like
from vsmasktools import EdgeDetect, EdgeDetectLike, Prewitt
from vsrgtools import MeanMode, bilateral, box_blur, gauss_blur, unsharpen
from vsscale import ArtCNN
from vstools import (
    ConvMode,
    FormatsMismatchError,
    FunctionUtil,
    Planes,
    VSFunctionNoArgs,
    scale_mask,
    vs,
)

from .deinterlacers import EEDI3, NNEDI3, AntiAliaser

__all__ = ["based_aa", "pre_aa"]


def pre_aa(
    clip: vs.VideoNode,
    sharpener: VSFunctionNoArgs = partial(unsharpen, blur=partial(gauss_blur, mode=ConvMode.VERTICAL, sigma=1)),
    antialiaser: AntiAliaser = NNEDI3(),
    transpose_first: bool = False,
    direction: AntiAliaser.AADirection = AntiAliaser.AADirection.BOTH,
    planes: Planes = None,
) -> vs.VideoNode:
    func = FunctionUtil(clip, pre_aa, planes)

    wclip = func.work_clip
    tclips = dict[str, Any]()

    for y in sorted((aa_dir for aa_dir in AntiAliaser.AADirection), key=lambda x: x.value, reverse=transpose_first):
        if direction in (y, AntiAliaser.AADirection.BOTH):
            if y == AntiAliaser.AADirection.HORIZONTAL:
                wclip, tclips = antialiaser.transpose(wclip)

            aa = antialiaser.antialias(wclip, AntiAliaser.AADirection.VERTICAL, **tclips)
            sharp = sharpener(wclip)
            limit = MeanMode.MEDIAN(wclip, aa, sharp)

            if y == AntiAliaser.AADirection.HORIZONTAL:
                wclip, tclips = antialiaser.transpose(limit)

    return func.return_clip(wclip)


def based_aa(
    clip: vs.VideoNode,
    rfactor: float = 2.0,
    mask: vs.VideoNode | EdgeDetectLike | Literal[False] = Prewitt,
    mask_thr: int = 60,
    pscale: float = 0.0,
    downscaler: ScalerLike | None = None,
    supersampler: ScalerLike | Literal[False] = ArtCNN,
    antialiaser: AntiAliaser | None = None,
    prefilter: vs.VideoNode | VSFunctionNoArgs | Literal[False] = False,
    postfilter: VSFunctionNoArgs | Literal[False] | dict[str, Any] | None = None,
    show_mask: bool = False,
    **aa_kwargs: Any,
) -> vs.VideoNode:
    """
    Perform based anti-aliasing on a video clip.

    This function works by super- or downsampling the clip and applying an AntiAliaser to that image.
    The result is then merged with the original clip using an edge mask, and it's limited
    to areas where the AntiAliaser was actually applied.

    Sharp supersamplers will yield better results, so long as they do not introduce too much ringing.
    For downscalers, you will want to use a neutral kernel.

    Args:
        clip: Clip to process.
        rfactor: Resize factor for supersampling. Values above 1.0 are recommended. Lower values may be useful for
            particularly extremely aliased content. Values closer to 1.0 will perform faster at the cost of precision.
            This value must be greater than 0.0. Default: 2.0.
        mask: Edge detection mask or function to generate it. Default: Prewitt.
        mask_thr: Threshold for edge detection mask. Only used if an EdgeDetect class is passed to `mask`. Default: 60.
        pscale: Scale factor for the supersample-downscale process change.
        downscaler: Scaler used for downscaling after anti-aliasing. This should ideally be a relatively sharp kernel
            that doesn't introduce too much haloing. If None, downscaler will be set to Box if the scale factor is an
            integer (after rounding), and Catrom otherwise. If rfactor is below 1.0, the downscaler will be used before
            antialiasing instead, and the supersampler will be used to scale the clip back to its original resolution.
            Default: None.
        supersampler: Scaler used for supersampling before anti-aliasing. If False, no supersampling is performed. If
            rfactor is below 1.0, the downscaler will be used before antialiasing instead, and the supersampler will be
            used to scale the clip back to its original resolution. The supersampler should ideally be fairly sharp
            without introducing too much ringing. Default: ArtCNN (R8F64).
        antialiaser: Antialiaser used for anti-aliasing. If None, EEDI3 will be selected with these default settings:
            (alpha=0.125, beta=0.25, vthresh0=12, vthresh1=24, field=1).
        prefilter: Prefilter to apply before anti-aliasing. Must be a VideoNode, a function that takes a VideoNode and
            returns a VideoNode, or False. Default: False.
        postfilter: Postfilter to apply after anti-aliasing. Must be a function that takes a VideoNode and returns a
            VideoNode, or None. If None, applies a median-filtered bilateral smoother to clean halos created during
            antialiasing. Default: None.
        show_mask: If True, returns the edge detection mask instead of the processed clip. Default: False

    Returns:
        Anti-aliased clip or edge detection mask if show_mask is True.

    Raises:
        CustomValueError: If rfactor is not above 0.0, or invalid prefilter/postfilter is passed.
    """

    func = FunctionUtil(clip, based_aa, 0, (vs.YUV, vs.GRAY))

    if rfactor <= 0.0:
        raise CustomValueError("rfactor must be greater than 0!", based_aa, rfactor)

    if mask is not False and not isinstance(mask, vs.VideoNode):
        mask = EdgeDetect.ensure_obj(mask, based_aa).edgemask(func.work_clip, 0)
        mask = mask.std.BinarizeMask(scale_mask(mask_thr, 8, func.work_clip))
        mask = box_blur(mask.std.Maximum())

        if show_mask:
            return mask

    if supersampler is False or is_noscale_like(supersampler):
        supersampler = downscaler = NoScale[Catrom]()
        rfactor = pscale = 1.0

    aaw, aah = [round(dimension * rfactor) for dimension in (func.work_clip.width, func.work_clip.height)]

    if downscaler is None:
        downscaler = (
            Box
            if (
                max(aaw, func.work_clip.width) % min(aaw, func.work_clip.width) == 0
                and max(aah, func.work_clip.height) % min(aah, func.work_clip.height) == 0
            )
            else Catrom
        )

    supersampler = Scaler.ensure_obj(supersampler, based_aa)
    downscaler = Scaler.ensure_obj(downscaler, based_aa)

    if rfactor < 1.0:
        downscaler, supersampler = supersampler, downscaler

    if callable(prefilter):
        ss_clip = prefilter(func.work_clip)
    elif isinstance(prefilter, vs.VideoNode):
        FormatsMismatchError.check(based_aa, func.work_clip, prefilter)
        ss_clip = prefilter
    else:
        ss_clip = func.work_clip

    ss = supersampler.scale(ss_clip, aaw, aah)

    if not antialiaser:
        antialiaser = EEDI3(alpha=0.125, beta=0.25, gamma=40, vthresh=(12, 24, 4), sclip=ss)

    # Only uses mclip if `use_mclip` is True,
    # if mclip isn't in aa_kwargs
    # and antialiaser is an instance of EEDI3
    if aa_kwargs.pop("use_mclip", True) and "mclip" not in aa_kwargs and isinstance(antialiaser, EEDI3):
        mclip = None

        if mask:
            mclip = mask if rfactor == 1 else vs.core.resize.Bilinear(mask, ss.width, ss.height)

        aa_kwargs.update(mclip=mclip)

    aa = antialiaser.antialias(ss, **aa_kwargs)

    aa = downscaler.scale(aa, func.work_clip.width, func.work_clip.height)

    if pscale != 1.0:
        no_aa = downscaler.scale(ss, func.work_clip.width, func.work_clip.height)
        aa = norm_expr([ss_clip, aa, no_aa], "x z x - {pscale} * + y z - +", pscale=pscale, func=func.func)

    if callable(postfilter):
        aa = postfilter(aa)
    elif postfilter is not False:
        postfilter_args = {"sigmaS": 2, "sigmaR": 1 / 255} | fallback(postfilter, {})
        aa = MeanMode.MEDIAN(aa, ss_clip, bilateral(aa, func.work_clip, **postfilter_args))

    if mask:
        aa = func.work_clip.std.MaskedMerge(aa, mask)

    return func.return_clip(aa)
