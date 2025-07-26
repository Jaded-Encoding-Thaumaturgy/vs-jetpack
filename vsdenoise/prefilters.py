"""
This module implements prefilters for denoisers.
"""

from __future__ import annotations

from enum import auto
from typing import Any, Literal, Sequence, cast, overload

from jetpytools import CustomEnum, CustomNotImplementedError, KwargsT

from vsexprtools import norm_expr
from vsrgtools import bilateral, flux_smooth, gauss_blur, min_blur
from vstools import (
    MISSING,
    ColorRange,
    ConstantFormatVideoNode,
    InvalidColorFamilyError,
    MissingT,
    PlanesT,
    check_variable,
    check_variable_format,
    get_peak_value,
    get_y,
    normalize_planes,
    normalize_seq,
    scale_value,
    vs,
)

from .fft import DFTTest, SLocationT

__all__ = [
    "MultiPrefilter",
    "Prefilter",
    "PrefilterLike",
    "PrefilterPartial",
    "prefilter_to_full_range",
]


def _run_prefilter(pref_type: Prefilter, clip: vs.VideoNode, planes: PlanesT, **kwargs: Any) -> vs.VideoNode:
    """
    Internal function for applying a prefilter to a clip.
    """

    assert check_variable(clip, pref_type)

    if pref_type == Prefilter.NONE:
        return clip

    if pref_type == Prefilter.MINBLUR:
        return min_blur(clip, **kwargs, planes=planes)

    if pref_type == Prefilter.GAUSS:
        return gauss_blur(clip, kwargs.pop("sigma", 1.5), **kwargs, planes=planes)

    if pref_type == Prefilter.FLUXSMOOTHST:
        temp_thr, spat_thr = normalize_seq(kwargs.pop("temp_thr", 2), 3), normalize_seq(kwargs.pop("spat_thr", 2), 3)

        return flux_smooth(clip, temp_thr, spat_thr, planes, **kwargs)

    if pref_type == Prefilter.DFTTEST:
        peak = get_peak_value(clip)
        pref_mask: vs.VideoNode | Literal[False] | tuple[int, int] | None = kwargs.pop("pref_mask", None)

        dftt = DFTTest(sloc={0.0: 4, 0.2: 9, 1.0: 15}).denoise(clip, kwargs.pop("sloc", None), planes=planes, **kwargs)

        if pref_mask is False:
            return dftt

        lower, upper = 16.0, 75.0

        if isinstance(pref_mask, tuple):
            lower, upper = pref_mask

        if not isinstance(pref_mask, vs.VideoNode):
            lower, upper = (scale_value(x, 8, clip) for x in (lower, upper))
            pref_mask = norm_expr(
                get_y(clip),
                f"x {lower} < {peak} x {upper} > 0 {peak} x {lower} - {peak} {upper} {lower} - / * - ? ?",
                func=pref_type,
            )

        return dftt.std.MaskedMerge(clip, pref_mask, planes)

    if pref_type == Prefilter.NLMEANS:
        from .nlm import nl_means

        return nl_means(clip, **KwargsT(h=7.0, s=2, planes=planes) | kwargs)

    if pref_type == Prefilter.BM3D:
        from .blockmatch import bm3d

        planes = normalize_planes(clip, planes)

        sigmas = kwargs.pop("sigma", [10 if 0 in planes else 0, 10 if (1 in planes or 2 in planes) else 0])

        return bm3d(clip, **KwargsT(sigma=sigmas, radius=1) | kwargs)

    if pref_type is Prefilter.BILATERAL:
        planes = normalize_planes(clip, planes)

        sigmaS = cast(float | list[float] | tuple[float | list[float], ...], kwargs.pop("sigmaS", 3.0))  # noqa: N806
        sigmaR = cast(float | list[float] | tuple[float | list[float], ...], kwargs.pop("sigmaR", 0.02))  # noqa: N806

        if isinstance(sigmaS, tuple):
            baseS, *otherS = sigmaS  # noqa: N806
        else:
            baseS, otherS = sigmaS, []  # noqa: N806

        if isinstance(sigmaR, tuple):
            baseR, *otherR = sigmaR  # noqa: N806
        else:
            baseR, otherR = sigmaR, []  # noqa: N806

        base, ref = clip, None
        max_len = max(len(otherS), len(otherR))

        if max_len:
            otherS = list[float | list[float]](reversed(normalize_seq(otherS or baseS, max_len)))  # noqa: N806
            otherR = list[float | list[float]](reversed(normalize_seq(otherR or baseR, max_len)))  # noqa: N806

            for siS, siR in zip(otherS, otherR):  # noqa: N806
                base, ref = ref or clip, bilateral(base, ref, siS, siR, **kwargs)

        return bilateral(clip, ref, baseS, baseR, planes=planes, **kwargs)

    raise CustomNotImplementedError(func=pref_type, reason=pref_type)


class AbstractPrefilter:
    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> vs.VideoNode | PrefilterPartial:
        raise NotImplementedError


class Prefilter(AbstractPrefilter, CustomEnum):
    """
    Enum representing available filters.

    These are mainly thought of as prefilters for [MVTools][vsdenoise.mvtools.mvtools.MVTools]
    but can be used standalone as-is.
    """

    NONE = auto()
    """
    Don't do any prefiltering. Returns the clip as-is.
    """

    MINBLUR = auto()
    """
    Minimum difference of a gaussian/median blur.
    """

    GAUSS = auto()
    """
    Gaussian blur.
    """

    FLUXSMOOTHST = auto()
    """
    Perform smoothing using `zsmooth.FluxSmoothST`.
    """

    DFTTEST = auto()
    """
    Denoising in frequency domain with dfttest and an adaptive mask for retaining details.
    """

    NLMEANS = auto()
    """
    Denoising with NLMeans.
    """

    BM3D = auto()
    """
    Normal spatio-temporal denoising using BM3D.
    """

    BILATERAL = auto()
    """
    Classic bilateral filtering or edge-preserving bilateral multi pass filtering.
    """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.FLUXSMOOTHST],
        clip: vs.VideoNode,
        planes: PlanesT = None,
        full_range: bool | float = False,
        *,
        temp_thr: float | Sequence[float] = 2.0,
        spat_thr: float | Sequence[float] | None = 2.0,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Perform smoothing using `zsmooth.FluxSmoothST`

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            temp_thr: Temporal threshold for the temporal median function.
            spat_thr: Spatial threshold for the temporal median function.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Preprocessed clip.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.DFTTEST],
        clip: vs.VideoNode,
        planes: PlanesT = None,
        full_range: bool | float = False,
        *,
        sloc: SLocationT | DFTTest.SLocation.MultiDim | None = {0.0: 4.0, 0.2: 9.0, 1.0: 15.0},
        pref_mask: vs.VideoNode | Literal[False] | tuple[int, int] = (16, 75),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        2D/3D frequency domain denoiser.

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            sloc: Frequency location.
            pref_mask: Gradient mask node for details retaining if VideoNode. Disable masking if False. Lower/upper
                bound pixel values if tuple. Anything below lower bound isn't denoised at all. Anything above upper
                bound is fully denoised. Values between them are a gradient.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Denoised clip.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.NLMEANS],
        clip: vs.VideoNode,
        planes: PlanesT = None,
        full_range: bool | float = False,
        *,
        h: float | Sequence[float] = 7.0,
        s: int | Sequence[int] = 2,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Denoising with NLMeans.

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            h: Controls the strength of the filtering. Larger values will remove more noise.
            s: Similarity Radius. Similarity neighbourhood size = `(2 * s + 1) ** 2`. Sets the radius of the similarity
                neighbourhood window. The impact on performance is low, therefore it depends on the nature of the noise.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Denoised clip.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.BM3D],
        clip: vs.VideoNode,
        planes: PlanesT = None,
        full_range: bool | float = False,
        *,
        sigma: float | Sequence[float] = 10,
        radius: int | Sequence[int | None] | None = 1,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Normal spatio-temporal denoising using BM3D.

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            sigma: Strength of denoising. Valid range is [0, +inf).
            radius: The temporal radius for denoising. Valid range is [1, 16].
            **kwargs: Additional arguments passed to the plugin.

        Returns:
            Preprocessed clip.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.BILATERAL],
        clip: vs.VideoNode,
        planes: PlanesT = None,
        full_range: bool | float = False,
        *,
        sigmaS: float | list[float] | tuple[float | list[float], ...] = 3.0,  # noqa: N803
        sigmaR: float | list[float] | tuple[float | list[float], ...] = 0.02,  # noqa: N803
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Classic bilateral filtering or edge-preserving bilateral multi pass filtering.

        If sigmaS or sigmaR are tuples, first values will be used as base,
        other values as a recursive reference.

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            sigmaS: Sigma of Gaussian function to calculate spatial weight.
            sigmaR: Sigma of Gaussian function to calculate range weight.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Preprocessed clip.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.FLUXSMOOTHST],
        *,
        planes: PlanesT = None,
        full_range: bool | float = False,
        temp_thr: float | Sequence[float] = 2.0,
        spat_thr: float | Sequence[float] | None = 2.0,
        **kwargs: Any,
    ) -> PrefilterPartial:
        """
        Perform smoothing using `zsmooth.FluxSmoothST`

        Args:
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            temp_thr: Temporal threshold for the temporal median function.
            spat_thr: Spatial threshold for the temporal median function.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Partial Prefilter.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.DFTTEST],
        *,
        planes: PlanesT = None,
        full_range: bool | float = False,
        sloc: SLocationT | DFTTest.SLocation.MultiDim | None = {0.0: 4.0, 0.2: 9.0, 1.0: 15.0},
        pref_mask: vs.VideoNode | Literal[False] | tuple[int, int] = (16, 75),
        **kwargs: Any,
    ) -> PrefilterPartial:
        """
        2D/3D frequency domain denoiser.

        Args:
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            sloc: Frequency location.
            pref_mask: Gradient mask node for details retaining if VideoNode. Disable masking if False. Lower/upper
                bound pixel values if tuple. Anything below lower bound isn't denoised at all. Anything above upper
                bound is fully denoised. Values between them are a gradient.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Partial Prefilter.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.NLMEANS],
        *,
        planes: PlanesT = None,
        full_range: bool | float = False,
        h: float | Sequence[float] = 7.0,
        s: int | Sequence[int] = 2,
        **kwargs: Any,
    ) -> PrefilterPartial:
        """
        Denoising with NLMeans.

        Args:
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            h: Controls the strength of the filtering. Larger values will remove more noise.
            s: Similarity Radius. Similarity neighbourhood size = `(2 * s + 1) ** 2`. Sets the radius of the similarity
                neighbourhood window. The impact on performance is low, therefore it depends on the nature of the noise.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Partial Prefilter.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.BM3D],
        *,
        planes: PlanesT = None,
        full_range: bool | float = False,
        sigma: float | Sequence[float] = 10,
        radius: int | Sequence[int | None] | None = 1,
        **kwargs: Any,
    ) -> PrefilterPartial:
        """
        Normal spatio-temporal denoising using BM3D.

        Args:
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            sigma: Strength of denoising. Valid range is [0, +inf).
            radius: The temporal radius for denoising. Valid range is [1, 16].
            **kwargs: Additional arguments passed to the plugin.

        Returns:
            Partial Prefilter.
        """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[Prefilter.BILATERAL],
        *,
        planes: PlanesT = None,
        full_range: bool | float = False,
        sigmaS: float | list[float] | tuple[float | list[float], ...] = 3.0,  # noqa: N803
        sigmaR: float | list[float] | tuple[float | list[float], ...] = 0.02,  # noqa: N803
        **kwargs: Any,
    ) -> PrefilterPartial:
        """
        Classic bilateral filtering or edge-preserving bilateral multi pass filtering.
        If sigmaS or sigmaR are tuples, first values will be used as base,
        other values as a recursive reference.

        Args:
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            sigmaS: Sigma of Gaussian function to calculate spatial weight.
            sigmaR: Sigma of Gaussian function to calculate range weight.
            **kwargs: Additional arguments to pass to the prefilter.

        Returns:
            Partial Prefilter.
        """

    @overload
    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self, *, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> PrefilterPartial: ...

    def __call__(
        self,
        clip: vs.VideoNode | MissingT = MISSING,
        planes: PlanesT = None,
        full_range: bool | float = False,
        **kwargs: Any,
    ) -> vs.VideoNode | PrefilterPartial:
        """
        Run the selected prefilter.

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            **kwargs: Additional arguments to pass to the specified prefilter.

        Returns:
            Preprocessed clip or Partial Prefilter.
        """
        if clip is MISSING:
            return PrefilterPartial(self, planes, full_range, **kwargs)

        out = _run_prefilter(self, clip, planes, **kwargs)

        if full_range is not False:
            if full_range is True:
                full_range = 2.0

            return prefilter_to_full_range(out, full_range)

        return out


class PrefilterPartial(AbstractPrefilter):
    """
    A partially-applied prefilter wrapper.
    """

    def __init__(self, prefilter: Prefilter, planes: PlanesT, full_range: bool | float, **kwargs: Any) -> None:
        """
        Stores a prefilter function, allowing it to be reused with different clips.

        Args:
            prefilter: [Prefilter][vsdenoise.prefilters.Prefilter] enumeration.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            **kwargs: Arguments for the specified prefilter.
        """
        self.prefilter = prefilter
        self.planes = planes
        self.full_range = full_range
        self.kwargs = kwargs

    def __call__(
        self,
        clip: vs.VideoNode,
        planes: PlanesT | MissingT = MISSING,
        full_range: bool | float | MissingT = MISSING,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Apply the prefilter to the given clip with optional argument overrides.

        Args:
            clip: Clip to be preprocessed.
            planes: Optional override for the planes to preprocess. If not provided, the default set in the constructor
                is used.
            full_range: Optional override for full range setting. If not provided, the default set in the constructor is
                used.
            **kwargs: Additional keyword arguments to override or extend the stored ones.

        Returns:
            Preprocessed clip.
        """
        return self.prefilter(
            clip,
            self.planes if planes is MISSING else planes,
            self.full_range if full_range is MISSING else full_range,
            **self.kwargs | kwargs,
        )


class MultiPrefilter(AbstractPrefilter):
    """
    A wrapper to apply multiple prefilters in sequence.
    """

    def __init__(self, *prefilters: Prefilter) -> None:
        """
        Stores a sequence of prefilter functions and applies them one after
        another to a given clip using the same parameters.

        Args:
            *prefilters: One or more prefilter functions to apply in order.
        """
        self.prefilters = prefilters

    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = None, full_range: bool | float = False, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Apply a sequence of prefilters to the given clip.

        Args:
            clip: Clip to be preprocessed.
            planes: Planes to be preprocessed.
            full_range: Whether to return a prefiltered clip in full range.
            **kwargs: Additional keyword arguments passed to each prefilter.

        Returns:
            Preprocessed clip.
        """
        for pref in self.prefilters:
            clip = pref(clip, planes, full_range, **kwargs)

        return clip


PrefilterLike = Prefilter | PrefilterPartial | MultiPrefilter


def prefilter_to_full_range(clip: vs.VideoNode, slope: float = 2.0, smooth: float = 0.0625) -> ConstantFormatVideoNode:
    """
    Converts a clip to full range if necessary and amplifies dark areas.
    Essentially acts like a luma-based multiplier on the SAD when used as an mvtools prefilter.

    Args:
        clip: Clip to process.
        slope: Slope to amplify the scale of the dark areas relative to bright areas.
        smooth: Indicates the length of the transition between the amplified dark areas and normal range conversion.

    Returns:
        Range expanded clip.
    """
    assert check_variable_format(clip, prefilter_to_full_range)

    InvalidColorFamilyError.check(clip, (vs.YUV, vs.GRAY), prefilter_to_full_range)

    clip_range = ColorRange.from_video(clip)

    curve = (slope - 1) * smooth
    luma_expr = (
        "x yrange_in_min - 1 yrange_in_max yrange_in_min - / * 0 1 clip LUMA! "
        "{k} 1 {c} + {c} sin LUMA@ {c} + / - * LUMA@ 1 {k} - * + range_max * "
    )
    chroma_expr = "x neutral - range_max crange_in_max crange_in_min - / * range_half + round"

    if clip.format.sample_type is vs.INTEGER:
        luma_expr += "round"

    planes = 0 if clip_range.is_full or clip.format.sample_type is vs.FLOAT else None

    return ColorRange.FULL.apply(norm_expr(clip, (luma_expr, chroma_expr), k=curve, c=smooth, planes=planes))
