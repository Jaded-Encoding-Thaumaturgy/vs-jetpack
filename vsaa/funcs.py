from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from jetpytools import CustomTypeError, CustomValueError

from vsjetpack import TypeIs
from vskernels import Box, Catrom, NoScale, Scaler, ScalerLike, is_noscale_like
from vsmasktools import EdgeDetect, EdgeDetectLike, Morpho, Prewitt
from vsrgtools import MeanMode, bilateral, box_blur, gauss_blur, unsharpen
from vsrgtools.blur import Bilateral
from vsscale import ArtCNN, pscale_blend
from vstools import (
    ConvMode,
    FunctionUtil,
    Planes,
    UnsupportedColorFamilyError,
    VSFunctionNoArgs,
    check_ref_clip,
    get_y,
    join,
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

    for y in sorted(AntiAliaser.AADirection, key=lambda x: x.value, reverse=transpose_first):
        if direction in (y, AntiAliaser.AADirection.BOTH):
            if y == AntiAliaser.AADirection.HORIZONTAL:
                wclip, tclips = antialiaser.transpose(wclip)

            aa = antialiaser.antialias(wclip, AntiAliaser.AADirection.VERTICAL, **tclips)
            wclip = MeanMode.MEDIAN(wclip, aa, sharpener(wclip))

            if y == AntiAliaser.AADirection.HORIZONTAL:
                wclip, tclips = antialiaser.transpose(wclip)

    return func.return_clip(wclip)


class BasedAA[**P, R]:
    """
    Class decorator that wraps the [based_aa][vsaa.funcs.based_aa] function and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, func: Callable[P, R]) -> None:
        self._func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    @staticmethod
    def postfilter(
        aa: vs.VideoNode,
        ss: vs.VideoNode,
        luma: vs.VideoNode,
        sigmaS: float = 2.0,  # noqa: N803
        sigmaR: float = 1 / 255,  # noqa: N803
        backend: Bilateral.Backend = bilateral.Backend.CPU,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        The default postfilter function used in based_aa.

        Applies a median-filtered bilateral smoother to clean halos created during antialiasing
        """
        postfilter_args: dict[str, Any] = {"sigmaS": sigmaS, "sigmaR": sigmaR, "backend": backend} | kwargs
        return MeanMode.MEDIAN(aa, ss, bilateral(aa, luma, **postfilter_args))


@BasedAA
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
    postfilter: Callable[[vs.VideoNode], vs.VideoNode]
    | Callable[[vs.VideoNode, vs.VideoNode, vs.VideoNode], vs.VideoNode]
    | Literal[False]
    | dict[str, Any] = BasedAA.postfilter,
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
        postfilter: Postfilter to apply after anti-aliasing.
            Must be either:

                - A function that takes the antialised clip and returns a new clip.
                - A function that takes the antialised clip, the raw supersampled clip
                  and the luma source clip in this order and returns a new clip.
                - A dict to adjust the default `BasedAA.postfilter` function.
                - The boolean False to disable the step entirely.

            The default argument is a callable that applies a median-filtered bilateral smoother
            to clean halos created during antialiasing
        show_mask: If True, returns the edge detection mask instead of the processed clip. Default: False

    Returns:
        Anti-aliased clip or edge detection mask if show_mask is True.

    Raises:
        CustomValueError: If rfactor is not above 0.0, or invalid prefilter/postfilter is passed.
    """

    UnsupportedColorFamilyError.check(clip, (vs.YUV, vs.GRAY), based_aa)

    if rfactor <= 0.0:
        raise CustomValueError("rfactor must be greater than 0!", based_aa, rfactor)

    luma = get_y(clip)

    if mask is not False and not isinstance(mask, vs.VideoNode):
        mask = EdgeDetect.ensure_obj(mask, based_aa).edgemask(luma)
        mask = Morpho.binarize_mask(mask, scale_mask(mask_thr, 8, 32))
        mask = box_blur(mask.std.Maximum())

        if show_mask:
            return mask

    if supersampler is False or is_noscale_like(supersampler):
        supersampler = downscaler = NoScale[Catrom]()
        rfactor = pscale = 1.0

    aaw, aah = [round(dimension * rfactor) for dimension in (clip.width, clip.height)]

    if downscaler is None:
        downscaler = (
            Box
            if (max(aaw, clip.width) % min(aaw, clip.width) == 0 and max(aah, clip.height) % min(aah, clip.height) == 0)
            else Catrom
        )

    supersampler = Scaler.ensure_obj(supersampler, based_aa)
    downscaler = Scaler.ensure_obj(downscaler, based_aa)

    if rfactor < 1.0:
        downscaler, supersampler = supersampler, downscaler

    if callable(prefilter):
        ss_clip = prefilter(luma)
    elif isinstance(prefilter, vs.VideoNode):
        check_ref_clip(prefilter, clip, based_aa)
        ss_clip = get_y(prefilter)
    else:
        ss_clip = luma

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
    aa = downscaler.scale(aa, clip.width, clip.height)

    aa = pscale_blend(ss_clip, aa, lambda: downscaler.scale(ss, clip.width, clip.height), pscale, func=based_aa)

    if callable(postfilter):
        if _has_3_params(postfilter):
            aa = postfilter(aa, ss_clip, luma)
        elif _has_1_param(postfilter):
            aa = postfilter(aa)
        else:
            raise CustomTypeError("Unsupported number of parameters", based_aa, repr(postfilter))
    elif postfilter is not False:
        aa = based_aa.postfilter(aa, ss_clip, luma, **postfilter)

    if mask:
        aa = luma.std.MaskedMerge(aa, mask)

    if clip.format.color_family is vs.YUV:
        aa = join(aa, clip)

    return aa


def _has_x_params(func: Callable[..., Any], n: int) -> bool:
    empty = [
        p
        for p in inspect.signature(func).parameters.values()
        if p.default is p.empty and p.kind not in [p.VAR_KEYWORD, p.VAR_POSITIONAL]
    ]
    return len(empty) == n


def _has_3_params(
    func: Callable[..., Any],
) -> TypeIs[Callable[[vs.VideoNode, vs.VideoNode, vs.VideoNode], vs.VideoNode]]:
    return _has_x_params(func, 3)


def _has_1_param(func: Callable[..., Any]) -> TypeIs[Callable[[vs.VideoNode], vs.VideoNode]]:
    return _has_x_params(func, 1)
