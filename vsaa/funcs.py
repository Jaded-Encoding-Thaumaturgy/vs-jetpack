from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from vsexprtools import norm_expr
from vskernels import Bilinear, Box, Catrom, NoScale, Scaler, ScalerT
from vsmasktools import EdgeDetect, EdgeDetectT, Prewitt, ScharrTCanny
from vsrgtools import MeanMode, bilateral, box_blur, unsharp_masked
from vsscale import ArtCNN
from vstools import (
    ConstantFormatVideoNode, CustomValueError, FormatsMismatchError, FunctionUtil, KwargsT, PlanesT, VSFunction,
    VSFunctionNoArgs, check_variable_format, fallback, get_peak_value, get_y, limiter, scale_mask, vs
)

from .abstract import Antialiaser
from .antialiasers import Eedi3, Nnedi3

__all__ = [
    'pre_aa',
    'clamp_aa',
    'based_aa'
]


class _pre_aa:
    def custom(
        self, clip: vs.VideoNode, sharpen: VSFunction,
        aa: type[Antialiaser] | Antialiaser = Nnedi3,
        planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        func = FunctionUtil(clip, pre_aa, planes)

        field = kwargs.pop('field', 3)
        if field < 2:
            field += 2

        if isinstance(aa, Antialiaser):
            aa = aa.copy(field=field, **kwargs)  # type: ignore
        else:
            aa = aa(field=field, **kwargs)

        wclip = func.work_clip

        for _ in range(2):
            bob = aa.interpolate(wclip, False)
            sharp = sharpen(wclip)
            limit = MeanMode.MEDIAN(sharp, wclip, bob[::2], bob[1::2])
            wclip = limit.std.Transpose()

        return func.return_clip(wclip)

    def __call__(
        self, clip: vs.VideoNode, radius: int = 1, strength: int = 100,
        aa: type[Antialiaser] | Antialiaser = Nnedi3,
        planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.custom(
            clip, partial(unsharp_masked, radius=radius, strength=strength), aa, planes, **kwargs
        )


pre_aa = _pre_aa()


def clamp_aa(
    clip: vs.VideoNode, strength: float = 1.0,
    mthr: float = 0.25, mask: vs.VideoNode | EdgeDetectT | None = None,
    weak_aa: vs.VideoNode | Antialiaser = Nnedi3(),
    strong_aa: vs.VideoNode | Antialiaser = Eedi3(),
    opencl: bool | None = False, ref: vs.VideoNode | None = None,
    planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Clamp a strong aa to a weaker one for the purpose of reducing the stronger's artifacts.

    :param clip:                Clip to process.
    :param strength:            Set threshold strength for over/underflow value for clamping.
    :param mthr:                Binarize threshold for the mask, float.
    :param mask:                Clip to use for custom mask or an EdgeDetect to use custom masker.
    :param weak_aa:             Antialiaser for the weaker aa.
    :param strong_aa:           Antialiaser for the stronger aa.
    :param opencl:              Whether to force OpenCL acceleration, None to leave as is.
    :param ref:                 Reference clip for clamping.

    :return:                    Antialiased clip.
    """

    func = FunctionUtil(clip, clamp_aa, planes, (vs.YUV, vs.GRAY))

    if not isinstance(weak_aa, vs.VideoNode):
        if opencl is not None and hasattr(weak_aa, 'opencl'):
            weak_aa.opencl = opencl

        weak_aa = weak_aa.aa(func.work_clip)

    if not isinstance(strong_aa, vs.VideoNode):
        if opencl is not None and hasattr(strong_aa, 'opencl'):
            strong_aa.opencl = opencl

        strong_aa = strong_aa.aa(func.work_clip)

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
        'y z - D1! y a - D2! D1@ D2@ xor x D1@ abs D2@ abs < a z {thr} - z {thr} + clip a ? ?',
        thr=thr, planes=func.norm_planes, func=func.func
    )

    if mask:
        if not isinstance(mask, vs.VideoNode):
            bin_thr = scale_mask(mthr, 32, clip)

            mask = ScharrTCanny.ensure_obj(mask).edgemask(func.work_clip)  # type: ignore
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
    downscaler: ScalerT | None = None,
    supersampler: ScalerT | Literal[False] = ArtCNN,
    double_rate: bool = False,
    antialiaser: Antialiaser | None = None,
    prefilter: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | Literal[False] = False,
    postfilter: VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | Literal[False] | None = None,
    show_mask: bool = False, **aa_kwargs: Any
) -> vs.VideoNode:
    """
    Perform based anti-aliasing on a video clip.

    This function works by super- or downsampling the clip and applying an antialiaser to that image.
    The result is then merged with the original clip using an edge mask, and it's limited
    to areas where the antialiaser was actually applied.

    Sharp supersamplers will yield better results, so long as they do not introduce too much ringing.
    For downscalers, you will want to use a neutral kernel.

    :param clip:                  Clip to process.
    :param rfactor:               Resize factor for supersampling. Values above 1.0 are recommended.
                                  Lower values may be useful for particularly extremely aliased content.
                                  Values closer to 1.0 will perform faster at the cost of precision.
                                  This value must be greater than 0.0. Default: 2.0.
    :param mask:                  Edge detection mask or function to generate it. Default: Prewitt.
    :param mask_thr:              Threshold for edge detection mask.
                                  Only used if an EdgeDetect class is passed to `mask`. Default: 60.
    :param pscale:                Scale factor for the supersample-downscale process change.
    :param downscaler:            Scaler used for downscaling after anti-aliasing. This should ideally be
                                  a relatively sharp kernel that doesn't introduce too much haloing.
                                  If None, downscaler will be set to Box if the scale factor is an integer
                                  (after rounding), and Catrom otherwise.
                                  If rfactor is below 1.0, the downscaler will be used before antialiasing instead,
                                  and the supersampler will be used to scale the clip back to its original resolution.
                                  Default: None.
    :param supersampler:          Scaler used for supersampling before anti-aliasing. If False, no supersampling
                                  is performed. If rfactor is below 1.0, the downscaler will be used before
                                  antialiasing instead, and the supersampler will be used to scale the clip
                                  back to its original resolution.
                                  The supersampler should ideally be fairly sharp without
                                  introducing too much ringing.
                                  Default: ArtCNN (R8F64).
    :param double_rate:           Whether to use double-rate antialiasing.
                                  If True, both fields will be processed separately, which may improve
                                  anti-aliasing strength at the cost of increased processing time and detail loss.
                                  Default: False.
    :param antialiaser:           Antialiaser used for anti-aliasing. If None, EEDI3 will be selected with these default settings:
                                  (alpha=0.125, beta=0.25, vthresh0=12, vthresh1=24, field=1).
    :param prefilter:             Prefilter to apply before anti-aliasing.
                                  Must be a VideoNode, a function that takes a VideoNode and returns a VideoNode,
                                  or False. Default: False.
    :param postfilter:            Postfilter to apply after anti-aliasing.
                                  Must be a function that takes a VideoNode and returns a VideoNode, or None.
                                  If None, applies a median-filtered bilateral smoother to clean halos
                                  created during antialiasing. Default: None.
    :param show_mask:             If True, returns the edge detection mask instead of the processed clip.
                                  Default: False

    :return:                      Anti-aliased clip or edge detection mask if show_mask is True.

    :raises CustomValueError:     If rfactor is not above 0.0, or invalid prefilter/postfilter is passed.
    """

    func = FunctionUtil(clip, based_aa, 0, (vs.YUV, vs.GRAY))

    if rfactor <= 0.0:
        raise CustomValueError('rfactor must be greater than 0!', based_aa, rfactor)

    if mask is not False and not isinstance(mask, vs.VideoNode):
        mask = EdgeDetect.ensure_obj(mask, based_aa).edgemask(func.work_clip, 0)
        mask = mask.std.Binarize(scale_mask(mask_thr, 8, func.work_clip))

        mask = box_blur(mask.std.Maximum())
        mask = limiter(mask, func=based_aa)
        
        if show_mask:
            return mask

    if supersampler is False:
        supersampler = downscaler = NoScale

    aaw, aah = [round(dimension * rfactor) for dimension in (func.work_clip.width, func.work_clip.height)]

    if downscaler is None:
        downscaler = Box if (
            max(aaw, func.work_clip.width) % min(aaw, func.work_clip.width) == 0
            and max(aah, func.work_clip.height) % min(aah, func.work_clip.height) == 0
        ) else Catrom

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
        antialiaser = Eedi3(mclip=Bilinear.scale(mask, ss.width, ss.height) if mask else None, sclip_aa=True)
        aa_kwargs = KwargsT(alpha=0.125, beta=0.25, vthresh0=12, vthresh1=24, field=1) | aa_kwargs

    aa = getattr(antialiaser, 'draa' if double_rate else 'aa')(ss, **aa_kwargs)

    aa = downscaler.scale(aa, func.work_clip.width, func.work_clip.height)

    if pscale != 1.0:
        no_aa = downscaler.scale(ss, func.work_clip.width, func.work_clip.height)
        aa = norm_expr([func.work_clip, aa, no_aa], 'x z x - {pscale} * + y z - +', pscale=pscale, func=func.func)

    if callable(postfilter):
        aa = postfilter(aa)
    elif postfilter is None:
        aa = MeanMode.MEDIAN(aa, func.work_clip, bilateral(aa))

    if mask:
        aa = func.work_clip.std.MaskedMerge(aa, mask)

    return func.return_clip(aa)
