"""
This module implements prefilters for denoisers
"""

from __future__ import annotations

from enum import EnumMeta
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast, overload

from jetpytools import CustomNotImplementedError

from vsexprtools import norm_expr
from vsrgtools import bilateral, flux_smooth, gauss_blur, min_blur
from vstools import (
    MISSING, ColorRange, CustomIntEnum, MissingT, PlanesT, SingleOrArr, check_variable, core, get_video_format,
    get_peak_value, get_y, normalize_planes, normalize_seq, scale_value, vs, InvalidColorFamilyError
)

from .bm3d import BM3D as BM3DM
from .bm3d import BM3DCPU, AbstractBM3D, BM3DCuda, BM3DCudaRTC, Profile
from .fft import DFTTest, SLocT
from .nlm import DeviceType, nl_means

__all__ = [
    'Prefilter', 'prefilter_to_full_range',
    'MultiPrefilter'
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
                return flux_smooth(clip, temp_thr, spat_thr, **kwargs)

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
                        f'x {lower} < {peak} x {upper} > 0 {peak} x {lower} - {peak} {upper} {lower} - / * - ? ?',
                        func=self
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
                        base, ref = ref or clip, bilateral(base, ref, siS, siR, **kwargs)

                return bilateral(clip, ref, baseS, baseR, **kwargs)

            raise CustomNotImplementedError(func=self, reason=self)

        if clip is MISSING:
            return PrefilterPartial(self, planes, **kwargs)

        out = _run(clip, planes, **kwargs)

        if full_range is not False:
            if full_range is True:
                full_range = 2.0

            return prefilter_to_full_range(out, full_range)

        return out


class Prefilter(PrefilterBase):
    """
    Enum representing available filters.\n
    These are mainly thought of as prefilters for :py:attr:`MVTools`,
    but can be used standalone as-is.
    """

    NONE = 0
    """Don't do any prefiltering. Returns the clip as-is."""

    MINBLUR = 0
    """Minimum difference of a gaussian/median blur"""

    GAUSS = 1
    """Gaussian blur."""

    FLUXSMOOTHST = 2
    """Perform smoothing using `zsmooth.FluxSmoothST`"""

    DFTTEST = 3
    """Denoising in frequency domain with dfttest and an adaptive mask for retaining details."""

    NLMEANS = 4
    """Denoising with NLMeans."""

    BM3D = 5
    """Normal spatio-temporal denoising using BM3D."""

    BILATERAL = 6
    """Classic bilateral filtering or edge-preserving bilateral multi pass filtering."""

    if TYPE_CHECKING:
        from .prefilters import Prefilter

        @overload  # type: ignore
        def __call__(  # type: ignore
            self: Literal[Prefilter.FLUXSMOOTHST], clip: vs.VideoNode, /,
            *, full_range: bool | float = False,
            temp_thr: float | Sequence[float] = 2.0, spat_thr: float | Sequence[float] | None = 2.0
        ) -> vs.VideoNode:
            """
            Perform smoothing using `zsmooth.FluxSmoothST`

            :param clip:        Clip to be preprocessed.
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

        @overload
        def __call__(  # type: ignore
            self: Literal[Prefilter.FLUXSMOOTHST], *,
            full_range: bool | float = False,
            temp_thr: float | Sequence[float] = 2.0, spat_thr: float | Sequence[float] | None = 2.0
        ) -> PrefilterPartial:
            """
            Perform smoothing using `zsmooth.FluxSmoothST`

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


def prefilter_to_full_range(clip: vs.VideoNode, slope: float = 2.0, smooth: float = 0.0625) -> vs.VideoNode:
    """
    Converts a clip to full range if necessary and amplifies dark areas.
    Essentially acts like a luma-based multiplier on the SAD when used as an mvtools prefilter.

    :param clip:        Clip to process.
    :param slope:       Slope to amplify the scale of the dark areas relative to bright areas.
    :param smooth:      Indicates the length of the transition between the amplified dark areas and normal range conversion.

    :return:            Range expanded clip.
    """

    InvalidColorFamilyError.check(clip, (vs.YUV, vs.GRAY), prefilter_to_full_range)

    clip_range = ColorRange.from_video(clip)
    clip_fmt = get_video_format(clip)

    curve = (slope - 1) * smooth
    luma_expr = (
        'x yrange_in_min - 1 yrange_in_max yrange_in_min - / * 0 1 clip LUMA! '
        '{k} 1 {c} + {c} sin LUMA@ {c} + / - * LUMA@ 1 {k} - * + range_max * '
    )
    chroma_expr = 'x neutral - range_max crange_in_max crange_in_min - / * range_half + round'

    if clip_fmt.sample_type is vs.INTEGER:
        luma_expr += 'round'

    planes = 0 if clip_range.is_full or clip_fmt.sample_type is vs.FLOAT else None

    return ColorRange.FULL.apply(norm_expr(clip, (luma_expr, chroma_expr), k=curve, c=smooth, planes=planes))
