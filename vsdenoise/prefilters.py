"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

import warnings

from enum import EnumMeta
from math import sin
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from vsexprtools import ExprOp, complexpr_available, norm_expr
from vskernels import Bilinear, Scaler
from vsmasktools import retinex
from vsrgtools import bilateral, box_blur, flux_smooth, gauss_blur, min_blur
from vstools import (
    MISSING, ColorRange, ConvMode, CustomIntEnum, MissingT, PlanesT, SingleOrArr,
    check_variable, clamp, core, depth, get_neutral_value, get_peak_value, get_y,
    join, normalize_planes, normalize_seq, scale_delta, scale_value, split, vs
)

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, AbstractBM3D, BM3DCuda, BM3DCudaRTC, Profile
from .fft import DFTTest, SLocT
from .nlm import DeviceType, nl_means

__all__ = [
    'Prefilter', 'prefilter_to_full_range',
    'MultiPrefilter'
]

__abstract__ = [
    'CUSTOM'
]


class PrefilterMeta(EnumMeta):
    def __instancecheck__(cls: EnumMeta, instance: Any) -> bool:
        if isinstance(instance, PrefilterPartial):
            return True
        return super().__instancecheck__(instance)  # type: ignore


class PrefilterBase(CustomIntEnum, metaclass=PrefilterMeta):
    @overload
    def __call__(  # type: ignore
        self: Prefilter, *, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> PrefilterPartial:
        ...

    @overload
    def __call__(  # type: ignore
        self: Prefilter, clip: vs.VideoNode, /, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> vs.VideoNode:
        ...

    def __call__(  # type: ignore
        self: Prefilter, clip: vs.VideoNode | MissingT = MISSING, /,
        planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> vs.VideoNode | PrefilterPartial:
        def _run(clip: vs.VideoNode, planes: PlanesT, **kwargs: Any) -> vs.VideoNode:
            assert check_variable(clip, self)

            # TODO: to remove
            if self == Prefilter.AUTO:
                warnings.warn(
                    f"{self} This Prefilter is deprecated and will be removed in a future version."
                    "Use :py:attr:`MINBLUR(radius=3)` instead",
                    DeprecationWarning
                )
                pref_type = Prefilter.MINBLUR
                kwargs = dict(radius=3) | kwargs
            else:
                pref_type = self

            planes = normalize_planes(clip, planes)

            if pref_type == Prefilter.NONE:
                return clip

            if pref_type == Prefilter.MINBLUR:
                return min_blur(clip, **kwargs, planes=planes)

            if pref_type == Prefilter.GAUSS:
                return gauss_blur(clip, kwargs.pop('sigma', 1.5), **kwargs, planes=planes)

            if pref_type == Prefilter.FLUXSMOOTHST:
                temp_thr, spat_thr = kwargs.pop('temp_thr', 2), kwargs.pop('spat_thr', 2)
                return flux_smooth(clip, temp_thr, spat_thr, **kwargs, planes=planes)

            if pref_type == Prefilter.DFTTEST:
                peak = get_peak_value(clip)
                pref_mask: vs.VideoNode | Literal[False] | tuple[int, int] | None = kwargs.pop("pref_mask", None)

                dftt = DFTTest(sloc={0.0: 4, 0.2: 9, 1.0: 15}, tr=0).denoise(
                    clip, kwargs.pop("sloc", None), planes=planes, **kwargs
                )

                if pref_mask is False:
                    return dftt

                lower, upper = 16., 75.

                if isinstance(pref_mask, tuple):
                    lower, upper = pref_mask

                if not isinstance(pref_mask, vs.VideoNode):
                    lower, upper = (scale_value(x, 8, clip) for x in (lower, upper))
                    pref_mask = norm_expr(
                        get_y(clip),
                        f'x {lower} < {peak} x {upper} > 0 {peak} x {lower} - {peak} {upper} {lower} - / * - ? ?'
                    )

                return dftt.std.MaskedMerge(clip, pref_mask, planes)

            if pref_type == Prefilter.NLMEANS:
                kwargs |= dict(strength=7.0, tr=1, sr=2, simr=2) | kwargs | dict(planes=planes)

                return nl_means(clip, **kwargs)

            if pref_type == Prefilter.BM3D:
                bm3d_arch: type[AbstractBM3D] = kwargs.pop('arch', None)
                gpu: bool | None = kwargs.pop('gpu', None)

                if gpu is None:
                    gpu = hasattr(core, 'bm3dcuda')

                if bm3d_arch is None:
                    if gpu:  # type: ignore
                        bm3d_arch = BM3DCudaRTC if hasattr(core, 'bm3dcuda_rtc') else BM3DCuda
                    else:
                        bm3d_arch = BM3DCPU if hasattr(core, 'bm3dcpu') else BM3DM

                if bm3d_arch is BM3DM:
                    sigma, profile = 10, Profile.FAST
                elif bm3d_arch is BM3DCPU:
                    sigma, profile = 10, Profile.LOW_COMPLEXITY
                elif bm3d_arch in (BM3DCuda, BM3DCudaRTC):
                    sigma, profile = 8, Profile.NORMAL
                else:
                    raise ValueError

                sigmas = kwargs.pop(
                    'sigma', [sigma if 0 in planes else 0, sigma if (1 in planes or 2 in planes) else 0]
                )

                bm3d_args = dict[str, Any](sigma=sigmas, tr=1, profile=profile) | kwargs

                return bm3d_arch.denoise(clip, **bm3d_args)

            if pref_type is Prefilter.BILATERAL:
                sigmaS = cast(float | list[float] | tuple[float | list[float], ...], kwargs.pop('sigmaS', 3.0))
                sigmaR = cast(float | list[float] | tuple[float | list[float], ...], kwargs.pop('sigmaR', 0.02))

                if isinstance(sigmaS, tuple):
                    baseS, *otherS = sigmaS
                else:
                    baseS, otherS = sigmaS, []

                if isinstance(sigmaR, tuple):
                    baseR, *otherR = sigmaR
                else:
                    baseR, otherR = sigmaR, []

                base, ref = clip, None
                max_len = max(len(otherS), len(otherR))

                if max_len:
                    otherS = list[float | list[float]](reversed(normalize_seq(otherS or baseS, max_len)))
                    otherR = list[float | list[float]](reversed(normalize_seq(otherR or baseR, max_len)))

                    for siS, siR in zip(otherS, otherR):
                        base, ref = ref or clip, bilateral(base, siS, siR, ref, **kwargs)

                return bilateral(clip, baseS, baseR, ref, **kwargs)

            # TODO: To remove
            if pref_type.value in {Prefilter.MINBLUR1, Prefilter.MINBLUR2, Prefilter.MINBLUR3}:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version."
                    "Use :py:attr:`MINBLUR(radius=...)` instead",
                    DeprecationWarning
                )
                return Prefilter.MINBLUR(clip, planes, full_range, radius=int(pref_type._name_[-1]))

            # TODO: To remove
            if pref_type == Prefilter.MINBLURFLUX:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version."
                    "Use :py:attr:`FLUXSMOOTHST(...)` instead",
                    DeprecationWarning
                )
                temp_thr, spat_thr = kwargs.get('temp_thr', 2), kwargs.get('spat_thr', 2)
                return min_blur(clip, 2, planes=planes).flux.SmoothST(  # type: ignore
                    scale_delta(temp_thr, 8, clip),
                    scale_delta(spat_thr, 8, clip),
                    planes
                )

            # TODO: To remove
            if pref_type == Prefilter.SCALEDBLUR:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version."
                    "Use :py:attr:`GAUSS(...)` instead",
                    DeprecationWarning
                )
                scale = kwargs.pop('scale', 2)
                downscaler = Scaler.ensure_obj(kwargs.pop('downscaler', Bilinear))
                upscaler = downscaler.ensure_obj(kwargs.pop('upscaler', downscaler))

                downscale = downscaler.scale(clip, clip.width // scale, clip.height // scale)

                boxblur = box_blur(downscale, kwargs.pop('radius', 1), mode=kwargs.pop('mode', ConvMode.HV), planes=planes)

                return upscaler.scale(boxblur, clip.width, clip.height)

            # TODO: To remove
            if pref_type == Prefilter.GAUSSBLUR:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version."
                    "Use :py:attr:`GAUSS(...)` instead",
                    DeprecationWarning
                )
                if 'sharp' not in kwargs and 'sigma' not in kwargs:
                    kwargs |= dict(sigma=1.0)

                return gauss_blur(clip, **(kwargs | dict[str, Any](planes=planes)))

            # TODO: To remove
            if pref_type in {Prefilter.GAUSSBLUR1, Prefilter.GAUSSBLUR2}:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version."
                    "Use :py:attr:`GAUSS(...)` instead",
                    DeprecationWarning
                )
                boxblur = box_blur(clip, kwargs.pop('radius', 1), mode=kwargs.get('mode', ConvMode.HV), planes=planes)

                if 'sharp' not in kwargs and 'sigma' not in kwargs:
                    kwargs |= dict(sigma=1.75)

                strg = clamp(kwargs.pop('strength', 50 if pref_type == Prefilter.GAUSSBLUR2 else 90), 0, 98) + 1

                gaussblur = gauss_blur(boxblur, **(kwargs | dict[str, Any](planes=planes)))

                if pref_type == Prefilter.GAUSSBLUR2:
                    i2, i7 = (scale_value(x, 8, clip) for x in (2, 7))

                    merge_expr = f'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x {strg} * y {100 - strg} * + 100 / ? ?'
                else:
                    merge_expr = f'x {strg / 100} * y {(100 - strg) / 100} * +'

                return norm_expr([gaussblur, clip], merge_expr, planes)

            # TODO: To remove
            if pref_type is Prefilter.BMLATERAL:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version.",
                    DeprecationWarning
                )
                sigma = kwargs.pop('sigma', 1.5)
                tr = kwargs.pop('tr', 2)
                radius = kwargs.pop('radius', 7)
                cuda = kwargs.pop('gpu', hasattr(core, 'bm3dcuda'))

                den = Prefilter.BM3D(
                    clip, planes, sigma=[10.0, 8.0] if cuda else [8.0, 6.4], tr=tr, gpu=cuda, **kwargs
                )

                return Prefilter.BILATERAL(den, planes, sigmaS=[sigma, sigma / 3], sigmaR=radius / 255)

            # TODO: To remove
            if pref_type == Prefilter.DFTTEST_SMOOTH:
                warnings.warn(
                    f"{pref_type} This Prefilter is deprecated and will be removed in a future version.\n"
                    "    Use this instead:\n"
                    "    sigma = 128\n"
                    "    sigma2 = sigma / 16\n"
                    "    DFTTEST(\n"
                    "        sbsize=8 if clip.width > 1280 else 6,\n"
                    "        sosize=6 if clip.width > 1280 else 4,\n"
                    "        slocation=[0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0]\n"
                    "    )",
                    DeprecationWarning
                )
                sigma = kwargs.get('sigma', 128.0)
                sigma2 = kwargs.get('sigma2', sigma / 16)
                sbsize = kwargs.get('sbsize', 8 if clip.width > 1280 else 6)
                sosize = kwargs.get('sosize', 6 if clip.width > 1280 else 4)

                kwargs |= dict(
                    sbsize=sbsize, sosize=sosize, slocation=[
                        0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0
                    ]
                )

                return Prefilter.DFTTEST(clip, planes, **kwargs)

            return clip

        if clip is MISSING:
            return PrefilterPartial(self, planes, **kwargs)

        out = _run(clip, planes, **kwargs)

        if full_range is not False:
            if full_range is True:
                full_range = 5.0

            return prefilter_to_full_range(out, full_range)

        return out


class Prefilter(PrefilterBase):
    """
    Enum representing available filters.\n
    These are mainly thought of as prefilters for :py:attr:`MVTools`,
    but can be used standalone as-is.
    """

    NONE = -1
    """Don't do any prefiltering. Returns the clip as-is."""

    MINBLUR = 15
    """Minimum difference of a gaussian/median blur"""

    GAUSS = 13
    """Gaussian blur."""

    FLUXSMOOTHST = 16
    """Perform smoothing using `zsmooth.FluxSmoothST`"""

    DFTTEST = 4
    """Denoising in frequency domain with dfttest and an adaptive mask for retaining details."""

    NLMEANS = 5
    """Denoising with NLMeans."""

    BM3D = 6
    """Normal spatio-temporal denoising using BM3D."""

    BILATERAL = 11
    """Classic bilateral filtering or edge-preserving bilateral multi pass filtering."""

    # TODO: To remove
    AUTO = -2
    """
    Automatically decide what prefilter to use.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`MINBLUR(radius=3)` instead
    """

    MINBLUR1 = 0
    """
    Minimum difference of a gaussian/median blur with a radius of 1.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`MINBLUR(radius=1)` instead
    """

    MINBLUR2 = 1
    """
    Minimum difference of a gaussian/median blur with a radius of 2.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`MINBLUR(radius=2)` instead.
    """

    MINBLUR3 = 2
    """
    Minimum difference of a gaussian/median blur with a radius of 3.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`MINBLUR(radius=3)` instead.
    """

    MINBLURFLUX = 3
    """
    :py:attr:`MINBLUR2` with temporal/spatial average.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`FLUXSMOOTHST(...)` instead.
    """

    SCALEDBLUR = 7
    """
    Perform blurring at a scaled-down resolution, then scaled back up.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`GAUSS()` instead.
    """

    GAUSSBLUR = 8
    """
    Gaussian blurred.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`GAUSS()` instead.
    """

    GAUSSBLUR1 = 9
    """
    Clamped gaussian/box blurring.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`GAUSS()` instead.
    """

    GAUSSBLUR2 = 10
    """
    Clamped gaussian/box blurring with edge preservation.
    This enum is deprecated and will be removed in a future version.
    Use :py:attr:`GAUSS()` instead.
    """

    BMLATERAL = 14
    """
    BM3D + BILATERAL blurring.
    This enum is deprecated and will be removed in a future version.
    """

    DFTTEST_SMOOTH = 12
    """
    Denoising like in DFTTEST but with high defaults for lower frequencies.
    This enum is deprecated and will be removed in a future version.
    Use this instead:
    ```py
    sigma = 128
    sigma2 = sigma / 16

    DFTTEST(
        sbsize=8 if clip.width > 1280 else 6,
        sosize=6 if clip.width > 1280 else 4,
        slocation=[0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0]
    )
    ```
    """

    if TYPE_CHECKING:
        from .prefilters import Prefilter

        @overload  # type: ignore
        def __call__(
            self: Literal[Prefilter.FLUXSMOOTHST], clip: vs.VideoNode, /,
            planes: PlanesT = None, full_range: bool | float = False,
            *, temp_thr: int = 2, spat_thr: int = 2
        ) -> vs.VideoNode:
            """
            Perform smoothing using `zsmooth.FluxSmoothST`

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param temp_thr:    Temporal threshold for the temporal median function.
            :param spat_thr:    Spatial threshold for the temporal median function.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST], clip: vs.VideoNode, /,
            planes: PlanesT = None, full_range: bool | float = False,
            *,
            sloc: SLocT | None = {0.0: 4.0, 0.2: 9.0, 1.0: 15.0},
            pref_mask: vs.VideoNode | Literal[False] | tuple[int, int] = (16, 75),
            tbsize: int = 1, sbsize: int = 12, sosize: int = 6, swin: int = 2,
            **kwargs: Any
        ) -> vs.VideoNode:
            """
            2D/3D frequency domain denoiser.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param pref_mask:   Gradient mask node for details retaining if VideoNode.
                                Disable masking if False.
                                Lower/upper bound pixel values if tuple.
                                Anything below lower bound isn't denoised at all.
                                Anything above upper bound is fully denoised.
                                Values between them are a gradient.

            :return:            Denoised clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.NLMEANS], clip: vs.VideoNode, /,
            planes: PlanesT = None, full_range: bool | float = False, *,
            strength: SingleOrArr[float] = 7.0, tr: SingleOrArr[int] = 1, sr: SingleOrArr[int] = 2,
            simr: SingleOrArr[int] = 2, device_type: DeviceType = DeviceType.AUTO, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Denoising with NLMeans.

            :param clip:            Source clip.
            :param strength:        Controls the strength of the filtering.\n
                                    Larger values will remove more noise.
            :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.\n
                                    Sets the number of past and future frames to uses for denoising the current frame.\n
                                    tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.\n
                                    Usually, larger values result in better denoising.
            :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.\n
                                    Sets the radius of the search window.\n
                                    sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.\n
                                    Usually, larger values result in better denoising.
            :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.\n
                                    Sets the radius of the similarity neighbourhood window.\n
                                    The impact on performance is low, therefore it depends on the nature of the noise.
            :param planes:          Set the clip planes to be processed.
            :param device_type:     Set the device to use for processing. The fastest device will be used by default.
            :param kwargs:          Additional arguments passed to the plugin.

            :return:                Denoised clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BM3D], clip: vs.VideoNode, /,
            planes: PlanesT = None, full_range: bool | float = False, *,
            arch: type[AbstractBM3D] = ..., gpu: bool | None = None,
            sigma: SingleOrArr[float] = ..., tr: SingleOrArr[int] = 1,
            profile: Profile = ..., ref: vs.VideoNode | None = None, refine: int = 1
        ) -> vs.VideoNode:
            """
            Normal spatio-temporal denoising using BM3D.

            :param clip:        Clip to be preprocessed.
            :param sigma:       Strength of denoising, valid range is [0, +inf].
            :param tr:          Temporal radius, valid range is [1, 16].
            :param profile:     See :py:attr:`vsdenoise.bm3d.Profile`.
            :param ref:         Reference clip used in block-matching, replacing the basic estimation.
                                If not specified, the input clip is used instead.
            :param refine:      Times to refine the estimation.
                                * 0 means basic estimate only.
                                * 1 means basic estimate with one final estimate.
                                * n means basic estimate refined with final estimate for n times.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BILATERAL], clip: vs.VideoNode, /,
            planes: PlanesT = None, full_range: bool | float = False, *,
            sigmaS: float | list[float] | tuple[float | list[float], ...] = 3.0,
            sigmaR: float | list[float] | tuple[float | list[float], ...] = 0.02,
            gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Classic bilateral filtering or edge-preserving bilateral multi pass filtering.
            If sigmaS or sigmaR are tuples, first values will be used as base,
            other values as a recursive reference.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param sigmaS:      Sigma of Gaussian function to calculate spatial weight.
            :param sigmaR:      Sigma of Gaussian function to calculate range weight.
            :param gpu:         Whether to use GPU processing if available or not.

            :return:            Preprocessed clip.
            """

        @overload
        def __call__(
            self, clip: vs.VideoNode, /, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Run the selected filter.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param kwargs:      Arguments for the specified filter.

            :return:            Preprocessed clip.
            """

        @overload  # type: ignore
        def __call__(
            self: Literal[Prefilter.FLUXSMOOTHST], *,
            planes: PlanesT = None, full_range: bool | float = False, temp_thr: int = 2, spat_thr: int = 2
        ) -> PrefilterPartial:
            """
            Perform smoothing using `zsmooth.FluxSmoothST`

            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param temp_thr:    Temporal threshold for the temporal median function.
            :param spat_thr:    Spatial threshold for the temporal median function.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.DFTTEST], *,
            planes: PlanesT = None, full_range: bool | float = False,
            sloc: SLocT | None = {0.0: 4.0, 0.2: 9.0, 1.0: 15.0},
            pref_mask: vs.VideoNode | Literal[False] | tuple[int, int] = (16, 75),
            tbsize: int = 1, sbsize: int = 12, sosize: int = 6, swin: int = 2,
            **kwargs: Any
        ) -> PrefilterPartial:
            """
            2D/3D frequency domain denoiser.

            :param clip:        Clip to be preprocessed.
            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param pref_mask:   Gradient mask node for details retaining if VideoNode.
                                Disable masking if False.
                                Lower/upper bound pixel values if tuple.
                                Anything below lower bound isn't denoised at all.
                                Anything above upper bound is fully denoised.
                                Values between them are a gradient.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.NLMEANS], *, planes: PlanesT = None, full_range: bool | float = False,
            strength: SingleOrArr[float] = 7.0, tr: SingleOrArr[int] = 1, sr: SingleOrArr[int] = 2,
            simr: SingleOrArr[int] = 2, device_type: DeviceType = DeviceType.AUTO, **kwargs: Any
        ) -> PrefilterPartial:
            """
            Denoising with NLMeans.

            :param planes:          Set the clip planes to be processed.
            :param strength:        Controls the strength of the filtering.\n
                                    Larger values will remove more noise.
            :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.\n
                                    Sets the number of past and future frames to uses for denoising the current frame.\n
                                    tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.\n
                                    Usually, larger values result in better denoising.
            :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.\n
                                    Sets the radius of the search window.\n
                                    sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.\n
                                    Usually, larger values result in better denoising.
            :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.\n
                                    Sets the radius of the similarity neighbourhood window.\n
                                    The impact on performance is low, therefore it depends on the nature of the noise.
            :param device_type:     Set the device to use for processing. The fastest device will be used by default.
            :param kwargs:          Additional arguments passed to the plugin.

            :return:                Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BM3D], *, planes: PlanesT = None, full_range: bool | float = False,
            arch: type[AbstractBM3D] = ..., gpu: bool = False,
            sigma: SingleOrArr[float] = ..., radius: SingleOrArr[int] = 1,
            profile: Profile = ..., ref: vs.VideoNode | None = None, refine: int = 1
        ) -> PrefilterPartial:
            """
            Normal spatio-temporal denoising using BM3D.

            :param sigma:       Strength of denoising, valid range is [0, +inf].
            :param radius:      Temporal radius, valid range is [1, 16].
            :param profile:     See :py:attr:`vsdenoise.bm3d.Profile`.
            :param ref:         Reference clip used in block-matching, replacing the basic estimation.
                                If not specified, the input clip is used instead.
            :param refine:      Times to refine the estimation.
                                * 0 means basic estimate only.
                                * 1 means basic estimate with one final estimate.
                                * n means basic estimate refined with final estimate for n times.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BILATERAL], *, planes: PlanesT = None, full_range: bool | float = False,
            sigmaS: float | list[float] | tuple[float | list[float], ...] = 3.0,
            sigmaR: float | list[float] | tuple[float | list[float], ...] = 0.02,
            gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            Classic bilateral filtering or edge-preserving bilateral multi pass filtering.
            If sigmaS or sigmaR are tuples, first values will be used as base,
            other values as a recursive reference.

            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param sigmaS:      Sigma of Gaussian function to calculate spatial weight.
            :param sigmaR:      Sigma of Gaussian function to calculate range weight.
            :param gpu:         Whether to use GPU processing if available or not.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.BMLATERAL], *, planes: PlanesT = None, full_range: bool | float = False,
            sigma: float = 1.5, radius: float = 7, tr: int = 2, gpu: bool | None = None, **kwargs: Any
        ) -> vs.VideoNode:
            """
            BM3D + BILATERAL smoothing.

            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param sigma:       ``Bilateral`` spatial weight sigma.
            :param radius:      ``Bilateral`` radius weight sigma.
            :param tr:          Temporal radius for BM3D.
            :param gpu:         Whether to process with GPU or not. None is auto.
            :param **kwargs:    Kwargs passed to BM3D.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(
            self, *, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
        ) -> PrefilterPartial:
            """
            Run the selected filter.

            :param planes:      Planes to be preprocessed.
            :param full_range:  Whether to return a prefiltered clip in full range.
            :param kwargs:      Arguments for the specified filter.

            :return:            Partial Prefilter.
            """

        @overload
        def __call__(  # type: ignore
            self, *, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
        ) -> PrefilterPartial:
            ...

        @overload
        def __call__(  # type: ignore
            self, clip: vs.VideoNode, /, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
        ) -> vs.VideoNode:
            ...

        def __call__(  # type: ignore
            self, clip: vs.VideoNode | MissingT = MISSING, /,
            planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
        ) -> vs.VideoNode | PrefilterPartial:
            ...


if TYPE_CHECKING:
    class PrefBase(Prefilter):  # type: ignore
        ...
else:
    class PrefBase:
        ...


class PrefilterPartial(PrefBase):  # type: ignore
    def __init__(self, prefilter: Prefilter, planes: PlanesT, **kwargs: Any) -> None:
        self.prefilter = prefilter
        self.planes = planes
        self.kwargs = kwargs

    def __call__(  # type: ignore
        self, clip: vs.VideoNode, /, planes: PlanesT | MissingT = MISSING, **kwargs: Any
    ) -> vs.VideoNode:
        return self.prefilter(
            clip, planes=self.planes if planes is MISSING else planes, **kwargs | self.kwargs
        )


class MultiPrefilter(PrefBase):  # type: ignore
    def __init__(self, *prefilters: Prefilter) -> None:
        self.prefilters = prefilters

    def __call__(self, clip: vs.VideoNode, /, **kwargs: Any) -> vs.VideoNode:  # type: ignore
        for pref in self.prefilters:
            clip = pref(clip)

        return clip


def prefilter_to_full_range(clip: vs.VideoNode, range_conversion: float = 5.0, planes: PlanesT = None) -> vs.VideoNode:
    """
    Convert a limited range clip to full range.\n
    Useful for expanding prefiltered clip's ranges to give motion estimation additional information to work with.

    :param clip:                Clip to be preprocessed.
    :param range_conversion:    Value which determines what range conversion method gets used.\n
                                 * >= 1.0 - Expansion with expr based on this coefficient.
                                 * >  0.0 - Expansion with retinex.
                                 * <= 0.0 - Simple conversion with resize plugin.
    :param planes:              Planes to be processed.

    :return:                    Full range clip.
    """
    planes = normalize_planes(clip, planes)

    work_clip, *chroma = split(clip) if planes == [0] else (clip, )

    assert (fmt := work_clip.format) and clip.format

    is_integer = fmt.sample_type == vs.INTEGER

    # Luma expansion TV->PC (up to 16% more values for motion estimation)
    if range_conversion >= 1.0:
        neutral = get_neutral_value(work_clip)
        max_val = get_peak_value(work_clip)

        c = sin(0.0625)
        k = (range_conversion - 1) * c

        if is_integer:
            t = f'x {scale_value(16, 8, clip)} '
            t += f'- {scale_value(219, 8, clip)} '
            t += f'/ {ExprOp.clamp(0, 1)}'
        else:
            t = ExprOp.clamp(0, 1, 'x').to_str()

        head = f'{k} {1 + c} {(1 + c) * c}'

        if complexpr_available:
            head = f'{t} T! {head}'
            t = 'T@'

        luma_expr = f'{head} {t} {c} + / - * {t} 1 {k} - * +'

        if is_integer:
            luma_expr += f' {max_val} *'

        pref_full = norm_expr(work_clip, (luma_expr, f'x {neutral} - 128 * 112 / {neutral} +'), planes)
    elif range_conversion > 0.0:
        pref_full = retinex(work_clip, upper_thr=range_conversion, fast=False)
    else:
        pref_full = depth(work_clip, clip, range_out=ColorRange.FULL)

    if chroma:
        return join(pref_full, *chroma, family=clip.format.color_family)

    return pref_full
