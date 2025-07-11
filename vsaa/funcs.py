from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from vsexprtools import norm_expr
from vskernels import Box, Catrom, NoScale, Scaler, ScalerLike
from vsmasktools import EdgeDetect, EdgeDetectT, Prewitt
from vsrgtools import MeanMode, bilateral, box_blur, gauss_blur, unsharpen
from vsscale import ArtCNN
from vstools import (
    ConstantFormatVideoNode,
    ConvMode,
    CustomValueError,
    FormatsMismatchError,
    FunctionUtil,
    KwargsT,
    PlanesT,
    VSFunctionNoArgs,
    check_variable_format,
    fallback,
    get_peak_value,
    get_y,
    limiter,
    scale_mask,
    vs,
)

from .deinterlacers import EEDI3, NNEDI3, AntiAliaser

__all__ = ["based_aa", "clamp_aa", "pre_aa"]


def pre_aa(
    clip: vs.VideoNode,
    sharpener: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = partial(
        unsharpen, blur=partial(gauss_blur, mode=ConvMode.VERTICAL, sigma=1)
    ),
    antialiaser: AntiAliaser = NNEDI3(),
    transpose_first: bool = False,
    direction: AntiAliaser.AADirection = AntiAliaser.AADirection.BOTH,
    planes: PlanesT = None,
) -> vs.VideoNode:
    func = FunctionUtil(clip, pre_aa, planes)

    wclip = func.work_clip

    for y in sorted((aa_dir for aa_dir in AntiAliaser.AADirection), key=lambda x: x.value, reverse=transpose_first):
        if direction in (y, AntiAliaser.AADirection.BOTH):
            if y == AntiAliaser.AADirection.HORIZONTAL:
                wclip = antialiaser.transpose(wclip)

            aa = antialiaser.antialias(wclip, AntiAliaser.AADirection.VERTICAL)
            sharp = sharpener(wclip)
            limit = MeanMode.MEDIAN(wclip, aa, sharp)

            if y == AntiAliaser.AADirection.HORIZONTAL:
                wclip = antialiaser.transpose(limit)

    return func.return_clip(wclip)


def clamp_aa(
    clip: vs.VideoNode,
    strength: float = 1.0,
    mthr: float = 0.25,
    mask: vs.VideoNode | EdgeDetectT | Literal[False] = False,
    weak_aa: vs.VideoNode | AntiAliaser | None = None,
    strong_aa: vs.VideoNode | AntiAliaser | None = None,
    ref: vs.VideoNode | None = None,
    planes: PlanesT = 0,
) -> ConstantFormatVideoNode:
    """
    Clamp a strong aa to a weaker one for the purpose of reducing the stronger's artifacts.

    Args:
        clip: Clip to process.
        strength: Set threshold strength for over/underflow value for clamping.
        mthr: Binarize threshold for the mask, float.
        mask: Clip to use for custom mask or an EdgeDetect to use custom masker.
        weak_aa: AntiAliaser for the weaker aa. Default is NNEDI3.
        strong_aa: AntiAliaser for the stronger aa. Default is EEDI3.
        ref: Reference clip for clamping.

    Returns:
        Antialiased clip.
    """
    import warnings

    warnings.warn(
        "clamp_aa is deprecated and will be removed in a future version. Use based_aa instead.",
        DeprecationWarning,
    )

    func = FunctionUtil(clip, clamp_aa, planes, (vs.YUV, vs.GRAY))

    if not isinstance(weak_aa, vs.VideoNode):
        if weak_aa is None:
            weak_aa = NNEDI3()

        weak_aa = weak_aa.antialias(func.work_clip)

    if not isinstance(strong_aa, vs.VideoNode):
        if strong_aa is None:
            strong_aa = EEDI3()

        strong_aa = strong_aa.antialias(func.work_clip)

    ref = fallback(ref, func.work_clip)

    if func.luma_only:
        weak_aa = get_y(weak_aa)
        strong_aa = get_y(strong_aa)
        ref = get_y(ref)

    if func.work_clip.format.sample_type == vs.INTEGER:
        thr = strength * get_peak_value(func.work_clip)
    else:
        thr = strength / 219

    clamped = norm_expr(
        [func.work_clip, ref, weak_aa, strong_aa],
        "y z - D1! y a - D2! D1@ D2@ xor x D1@ abs D2@ abs < a z {thr} - z {thr} + clip a ? ?",
        thr=thr,
        planes=func.norm_planes,
        func=func.func,
    )

    if mask is not False:
        if not isinstance(mask, vs.VideoNode):
            bin_thr = scale_mask(mthr, 32, clip)

            mask = EdgeDetect.ensure_obj(mask).edgemask(func.work_clip)
            mask = box_blur(mask.std.Binarize(bin_thr).std.Maximum())
            mask = mask.std.Minimum().std.Deflate()

        clamped = func.work_clip.std.MaskedMerge(clamped, mask, func.norm_planes)

    return func.return_clip(clamped)


def based_aa(
    clip: vs.VideoNode,
    rfactor: float = 2.0,
    mask: vs.VideoNode | EdgeDetectT | Literal[False] = Prewitt,
    mask_thr: int = 60,
    pscale: float = 0.0,
    downscaler: ScalerLike | None = None,
    supersampler: ScalerLike | Literal[False] = ArtCNN,
    antialiaser: AntiAliaser | None = None,
    prefilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | Literal[False] = False,
    postfilter: VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | Literal[False] | KwargsT | None = None,
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
        mask = mask.std.Binarize(scale_mask(mask_thr, 8, func.work_clip))

        mask = box_blur(mask.std.Maximum())
        mask = limiter(mask, func=based_aa)

        if show_mask:
            return mask

    if supersampler is False:
        supersampler = downscaler = NoScale[Catrom]
        rfactor = 1.0

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

        if TYPE_CHECKING:
            assert check_variable_format(prefilter, func.func)

        ss_clip = prefilter
    else:
        ss_clip = func.work_clip

    ss = supersampler.scale(ss_clip, aaw, aah)

    if not antialiaser:
        antialiaser = EEDI3(alpha=0.125, beta=0.25, gamma=40, vthresh=(12, 24, 4), sclip=ss)

    # Only uses mclip if `use_mclip` is True,
    # if mclip isn't in aa_kwargs
    # and antialiaser is an instance of EEDI3
    if aa_kwargs.pop("use_mclip", False) and "mclip" not in aa_kwargs and isinstance(antialiaser, EEDI3):
        mclip = None

        if mask:
            mclip = mask if isinstance(supersampler, NoScale) else vs.core.resize.Bilinear(mask, ss.width, ss.height)

        aa_kwargs.update(mclip=mclip)

    aa = antialiaser.antialias(ss, **aa_kwargs)

    aa = downscaler.scale(aa, func.work_clip.width, func.work_clip.height)

    if pscale != 1.0 and not isinstance(supersampler, NoScale):
        no_aa = downscaler.scale(ss, func.work_clip.width, func.work_clip.height)
        aa = norm_expr([ss_clip, aa, no_aa], "x z x - {pscale} * + y z - +", pscale=pscale, func=func.func)

    if callable(postfilter):
        aa = postfilter(aa)
    elif postfilter is not False:
        postfilter_args = KwargsT(sigmaS=2, sigmaR=1 / 255) | fallback(postfilter, KwargsT())
        aa = MeanMode.MEDIAN(aa, ss_clip, bilateral(aa, func.work_clip, **postfilter_args))

    if mask:
        aa = func.work_clip.std.MaskedMerge(aa, mask)

    return func.return_clip(aa)
