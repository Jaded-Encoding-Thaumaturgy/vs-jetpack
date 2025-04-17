from __future__ import annotations

from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Any, Literal, Sequence, Union, overload

from jetpytools import FuncExceptT

from vsexprtools import ExprOp, ExprVars, complexpr_available, norm_expr
from vskernels import Bilinear, Gaussian
from vstools import (
    ConstantFormatVideoNode, ConvMode, CustomValueError, KwargsT, OneDimConvModeT, PlanesT, SpatialConvModeT,
    TempConvModeT, VSFunctionNoArgs, check_variable, check_variable_format, core, join, normalize_planes, normalize_seq,
    split, to_arr, vs
)

from .enum import BilateralBackend, BlurMatrix, BlurMatrixBase, LimitFilterMode, VerticalCleanerMode
from .freqs import MeanMode
from .limit import limit_filter
from .rgtools import vertical_cleaner
from .util import normalize_radius, norm_rmode_planes

__all__ = [
    'box_blur', 'side_box_blur',
    'gauss_blur',
    'min_blur', 'sbr', 'median_blur',
    'bilateral', 'flux_smooth'
]


def box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, passes: int = 1,
    mode: OneDimConvModeT | TempConvModeT = ConvMode.HV, planes: PlanesT = None, **kwargs: Any
) -> ConstantFormatVideoNode:
    assert check_variable(clip, box_blur)

    if isinstance(radius, list):
        return normalize_radius(clip, box_blur, radius, planes, passes=passes, mode=mode, **kwargs)

    if not radius:
        return clip

    if mode == ConvMode.TEMPORAL:
        return BlurMatrix.MEAN(radius, mode=mode)(clip, planes, passes=passes, **kwargs)

    if not TYPE_CHECKING:
        if mode == ConvMode.SQUARE:
            raise CustomValueError("Invalid mode specified", box_blur, mode)

    box_args = (
        planes,
        radius, 0 if mode == ConvMode.VERTICAL else passes,
        radius, 0 if mode == ConvMode.HORIZONTAL else passes
    )

    return clip.vszip.BoxBlur(*box_args)


def side_box_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, planes: PlanesT = None,
    inverse: bool = False
) -> ConstantFormatVideoNode:
    assert check_variable_format(clip, side_box_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, side_box_blur, radius, planes, inverse=inverse)

    half_kernel = [(1 if i <= 0 else 0) for i in range(-radius, radius + 1)]

    conv_m1 = partial(core.std.Convolution, matrix=half_kernel, planes=planes)
    conv_m2 = partial(core.std.Convolution, matrix=half_kernel[::-1], planes=planes)
    blur_pt = partial(box_blur, planes=planes)

    vrt_filters, hrz_filters = list[list[partial[ConstantFormatVideoNode]]](
        [
            partial(conv_m1, mode=mode), partial(conv_m2, mode=mode),
            partial(blur_pt, hradius=hr, vradius=vr, hpasses=h, vpasses=v)
        ] for h, hr, v, vr, mode in [
            (0, None, 1, radius, ConvMode.VERTICAL), (1, radius, 0, None, ConvMode.HORIZONTAL)
        ]
    )

    vrt_intermediates = (vrt_flt(clip) for vrt_flt in vrt_filters)
    intermediates = list(
        hrz_flt(vrt_intermediate)
        for i, vrt_intermediate in enumerate(vrt_intermediates)
        for j, hrz_flt in enumerate(hrz_filters) if not i == j == 2
    )

    comp_blur = None if inverse else box_blur(clip, radius, 1, planes=planes)

    if complexpr_available:
        template = '{cum} x - abs {new} x - abs < {cum} {new} ?'

        cum_expr, cumc = '', 'y'
        n_inter = len(intermediates)

        for i, newc, var in zip(count(), ExprVars[2:26], ExprVars[4:26]):
            if i == n_inter - 1:
                break

            cum_expr += template.format(cum=cumc, new=newc)

            if i != n_inter - 2:
                cumc = var.upper()
                cum_expr += f' {cumc}! '
                cumc = f'{cumc}@'

        if comp_blur:
            clips = [clip, *intermediates, comp_blur]
            cum_expr = f'x {cum_expr} - {ExprVars[n_inter + 1]} +'
        else:
            clips = [clip, *intermediates]

        cum = norm_expr(clips, cum_expr, planes, func=side_box_blur)
    else:
        cum = intermediates[0]
        for new in intermediates[1:]:
            cum = limit_filter(clip, cum, new, LimitFilterMode.SIMPLE2_MIN, planes)

        if comp_blur:
            cum = clip.std.MakeDiff(cum).std.MergeDiff(comp_blur)

    if comp_blur:
        return box_blur(cum, 1, min(radius // 2, 1))

    return cum


def gauss_blur(
    clip: vs.VideoNode, sigma: float | list[float] = 0.5, taps: int | None = None,
    mode: ConvMode = ConvMode.HV, planes: PlanesT = None,
    **kwargs: Any
) -> ConstantFormatVideoNode:
    assert check_variable(clip, gauss_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(sigma, list):
        return normalize_radius(clip, gauss_blur, ('sigma', sigma), planes, mode=mode)

    fast = kwargs.pop("_fast", False)

    sigma_constant = 0.9 if fast and not mode.is_temporal else sigma
    taps = BlurMatrix.GAUSS.get_taps(sigma_constant, taps)

    if not mode.is_temporal:
        def _resize2_blur(plane: ConstantFormatVideoNode, sigma: float, taps: int) -> ConstantFormatVideoNode:
            resize_kwargs = dict[str, Any]()

            # Downscale approximation can be used by specifying _fast=True
            # Has a big speed gain when taps is large
            if fast:
                wdown, hdown = plane.width, plane.height

                if ConvMode.VERTICAL in mode:
                    hdown = round(max(round(hdown / sigma), 2) / 2) * 2

                if ConvMode.HORIZONTAL in mode:
                    wdown = round(max(round(wdown / sigma), 2) / 2) * 2

                resize_kwargs.update(width=plane.width, height=plane.height)

                plane = Bilinear.scale(plane, wdown, hdown)  # type: ignore[assignment]
                sigma = sigma_constant
            else:
                resize_kwargs.update({f'force_{k}': k in mode for k in 'hv'})

            return Gaussian(sigma, taps).scale(plane, **resize_kwargs | kwargs)  # type: ignore[return-value]

        if not {*range(clip.format.num_planes)} - {*planes}:
            return _resize2_blur(clip, sigma, taps)

        return join([
            _resize2_blur(p, sigma, taps) if i in planes else p
            for i, p in enumerate(split(clip))
        ])

    kernel: BlurMatrixBase[float] = BlurMatrix.GAUSS(  # type: ignore
        taps, sigma=sigma, mode=mode, scale_value=1023
    )

    return kernel(clip, planes, **kwargs)


def min_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1,
    mode: tuple[ConvMode, ConvMode] = (ConvMode.HV, ConvMode.SQUARE), planes: PlanesT = None,
    **kwargs: Any
) -> ConstantFormatVideoNode:
    """
    MinBlur by Didée (http://avisynth.nl/index.php/MinBlur)
    Nifty Gauss/Median combination
    """
    assert check_variable(clip, min_blur)

    planes = normalize_planes(clip, planes)

    if isinstance(radius, list):
        return normalize_radius(clip, min_blur, radius, planes)

    mode_blur, mode_median = normalize_seq(mode, 2)

    blurred = BlurMatrix.BINOMIAL(radius=radius, mode=mode_blur)(clip, planes=planes, **kwargs)
    median = median_blur(clip, radius, mode_median, planes=planes)

    return MeanMode.MEDIAN([clip, blurred, median], planes=planes)



_SbrBlurT = Union[
    BlurMatrix,
    Sequence[float],
    VSFunctionNoArgs[vs.VideoNode, vs.VideoNode],
]

def sbr(
    clip: vs.VideoNode,
    radius: int | Sequence[int] = 1,
    mode: ConvMode = ConvMode.HV,
    blur: _SbrBlurT | vs.VideoNode = BlurMatrix.BINOMIAL,
    blur_diff: _SbrBlurT = BlurMatrix.BINOMIAL,
    planes: PlanesT = None,
    *,
    func: FuncExceptT | None = None,
    **kwargs: Any
) -> ConstantFormatVideoNode:
    """
    A helper function for high-pass filtering a blur difference, inspired by an AviSynth script by Didée.
    `https://forum.doom9.org/showthread.php?p=1584186#post1584186`

    :param clip:        Source clip.
    :param radius:      Specifies the size of the blur kernels if `blur` or `blur_diff` is a BlurMatrix enum.
                        Default to 1.
    :param mode:        Specifies the convolution mode. Defaults to horizontal + vertical.
    :param blur:        Blur kernel to apply to the original clip. Defaults to binomial.
    :param blur_diff:   Blur kernel to apply to the difference clip. Defaults to binomial.
    :param planes:      Which planes to process. Defaults to all.
    :param **kwargs:    Additional arguments passed to blur kernel call.
    :return:            Sbr'd clip.
    """
    func = func or sbr

    if isinstance(radius, Sequence):
        return normalize_radius(clip, min_blur, list(radius), planes)

    def _apply_blur(clip: ConstantFormatVideoNode, blur: _SbrBlurT | vs.VideoNode) -> ConstantFormatVideoNode:
        if isinstance(blur, Sequence):
            return BlurMatrixBase(blur, mode=mode)(clip, planes, **kwargs)

        if isinstance(blur, BlurMatrix):
            return blur(taps=radius, mode=mode)(clip, planes, **kwargs)

        blurred = blur(clip) if callable(blur) else blur

        assert check_variable_format(blurred, func)

        return blurred

    assert check_variable(clip, func)

    planes = normalize_planes(clip, planes)

    blurred = _apply_blur(clip, blur)

    diff = clip.std.MakeDiff(blurred, planes=planes)
    blurred_diff = _apply_blur(diff, blur_diff)

    return norm_expr(
        [clip, diff, blurred_diff],
        'y neutral - D1! y z - D2! D1@ D2@ xor x x D1@ abs D2@ abs < D1@ D2@ ? - ?',
        planes=planes, func=func
    )


@overload
def median_blur(
    clip: vs.VideoNode, radius: int = ..., mode: Literal[ConvMode.TEMPORAL] = ..., planes: PlanesT = ...
) -> ConstantFormatVideoNode:
    ...


@overload
def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = ..., mode: SpatialConvModeT = ..., planes: PlanesT = None
) -> ConstantFormatVideoNode:
    ...


@overload
def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = ..., mode: ConvMode = ..., planes: PlanesT = None
) -> ConstantFormatVideoNode:
    ...


def median_blur(
    clip: vs.VideoNode, radius: int | list[int] = 1, mode: ConvMode = ConvMode.SQUARE, planes: PlanesT = None
) -> ConstantFormatVideoNode:
    assert check_variable(clip, median_blur)

    if mode == ConvMode.TEMPORAL:
        if isinstance(radius, int):
            return clip.zsmooth.TemporalMedian(radius, planes)

        raise CustomValueError("A list of radius isn't supported for ConvMode.TEMPORAL!", median_blur, radius)

    radius = to_arr(radius)

    if (len((rs := set(radius))) == 1 and rs.pop() == 1):
        if mode == ConvMode.SQUARE:
            return clip.std.Median(planes=planes)
        if mode == ConvMode.VERTICAL:
            return vertical_cleaner(clip, norm_rmode_planes(clip, VerticalCleanerMode.MEDIAN, planes))

    expr_plane = list[list[str]]()

    for r in radius:
        expr_passes = list[str]()

        for mat in ExprOp.matrix('x', r, mode, [(0, 0)]):
            rb = len(mat) + 1
            st = rb - 1
            sp = rb // 2 - 1
            dp = st - 2

            expr_passes.append(f"{mat} sort{st} swap{sp} min! swap{sp} max! drop{dp} x min@ max@ clip")

        expr_plane.append(expr_passes)

    for e in zip(*expr_plane):
        clip = norm_expr(clip, e, planes, func=median_blur)

    return clip


def bilateral(
    clip: vs.VideoNode, ref: vs.VideoNode | None = None, sigmaS: float | Sequence[float] | None = None,
    sigmaR: float | Sequence[float] | None = None, backend: BilateralBackend = BilateralBackend.CPU, **kwargs: Any
) -> ConstantFormatVideoNode:
    assert check_variable_format(clip, bilateral)

    if backend == BilateralBackend.CPU:
        bilateral_args = KwargsT(ref=ref, sigmaS=sigmaS, sigmaR=sigmaR, planes=normalize_planes(clip))
    else:
        bilateral_args = KwargsT(ref=ref, sigma_spatial=sigmaS, sigma_color=sigmaR)

    return getattr(clip, backend).Bilateral(**bilateral_args | kwargs)


def flux_smooth(
    clip: vs.VideoNode,
    temporal_threshold: float | Sequence[float] = 7.0,
    spatial_threshold: float | Sequence[float] | None = None,
    scalep: bool = True,
) -> ConstantFormatVideoNode:
    """
    FluxSmoothT examines each pixel and compares it to the corresponding pixel in the previous and next frames.
    Smoothing occurs if both the previous frame's value and the next frame's value are greater,
    or if both are less than the value in the current frame.

    Smoothing is done by averaging the pixel from the current frame with the pixels from the previous
    and/or next frames, if they are within temporal_threshold.

    FluxSmoothST does the same as FluxSmoothT, except the pixel's eight neighbours from the current frame
    are also included in the average, if they are within spatial_threshold.

    The first and last rows and the first and last columns are not processed by FluxSmoothST.

    :param clip:                    Clip to process.
    :param temporal_threshold:      Temporal neighbour pixels within this threshold from the current pixel
                                    are included in the average. Can be specified as an array,
                                    with values corresonding to each plane of the input clip.
                                    A negative value (such as -1) indicates that the plane should not be processed
                                    and will be copied from the input clip.
    :param spatial_threshold:       Spatial neighbour pixels within this threshold from the current pixel
                                    are included in the average. A negative value (such as -1) indicates that the plane
                                    should not be processed and will be copied from the input clip.
    :param scalep:                  Parameter scaling. If set to true, all threshold values
                                    will be automatically scaled from 8-bit range (0-255) to the corresponding range
                                    of the input clip's bit depth.
    :return:                        Smoothed clip.
    """

    assert check_variable_format(clip, flux_smooth)

    if spatial_threshold:
        return core.zsmooth.FluxSmoothST(clip, temporal_threshold, spatial_threshold, scalep)

    return core.zsmooth.FluxSmoothT(clip, temporal_threshold, scalep)
