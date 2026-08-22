from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import auto
from math import comb, sqrt
from typing import Any, Literal, Protocol, Self, TypedDict, overload

from jetpytools import CustomIntEnum, CustomValueError, FuncExcept, cachedproperty, fallback, normalize_seq, to_arr

from vsaa import NNEDI3
from vsdeband import Grainer
from vsdenoise import (
    DFTTest,
    MaskMode,
    MotionVectors,
    MVDirection,
    MVTools,
    MVToolsPreset,
    mc_clamp,
    prefilter_to_full_range,
    refine_blksize,
)
from vsexprtools import norm_expr
from vsjetpack import deprecated
from vskernels import Bobber, BobberLike, Catrom
from vsmasktools import Coordinates, Morpho
from vsrgtools import BlurMatrix, gauss_blur, median_blur, remove_grain, repair, unsharpen
from vstools import (
    ConvMode,
    FieldBased,
    FieldBasedLike,
    Planes,
    UnsupportedFieldBasedError,
    VSObject,
    sc_detect,
    scale_delta,
    vs,
)

from .utils import reinterlace, reweave

__all__ = ["QTempGaussMC", "mask_shimmer"]


_DFTTEST_DEFAULT = DFTTest(sigma=8)
_NNEDI3_DEFAULT = NNEDI3(nsize=1)


class _DenoiseFuncTr(Protocol):
    def __call__(self, clip: vs.VideoNode, /, *, tr: int) -> vs.VideoNode: ...


class QTGMCArgs:
    """Namespace containing helper TypedDict definitions for various argument groups."""

    class MaskShimmer(TypedDict, total=False):
        """Arguments accepted by [mask_shimmer][vsdeinterlace.mask_shimmer]."""

        erosion_distance: int
        over_dilation: int

    class PrefilterToFullRange(TypedDict, total=False):
        """Arguments accepted by [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range]."""

        slope: float
        smooth: float

    class Compensate(TypedDict, total=False):
        """Arguments accepted by [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate]."""

        thsad: int | None
        time: float | None

    class Degrain(TypedDict, total=False):
        """Arguments accepted by the internal `_binomial_degrain` method."""

        limit: float | tuple[float, float] | None
        planes: Planes

    class Blur(TypedDict, total=False):
        """Arguments accepted by [MVTools.flow_blur][vsdenoise.mvtools.mvtools.MVTools.flow_blur]."""

        prec: int | None

    class Mask(TypedDict, total=False):
        """Arguments accepted by [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask]."""

        ml: float | None
        gamma: float | None
        time: float | None
        scval: float | None


class _QTGMCBuilder:
    class NoiseDeintMode(CustomIntEnum):
        """How to 'deinterlace' noise taken from an interlaced source."""

        WEAVE = auto()
        """
        Double weave source noise.

        Lags behind by one frame.
        """

        BOB = auto()
        """
        Bob source noise.

        Results in coarse noise.
        """

        GENERATE = auto()
        """
        Generate fresh noise lines.
        """

    class LosslessMode(CustomIntEnum):
        """When to put the original fields into the output."""

        NONE = auto()
        """
        Do not restore the original fields.
        """

        PRESHARPEN = auto()
        """
        Restore the original fields prior to [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen].

        Provides near-lossless fidelity, mitigates most artifacts, and retains sharpness control.
        """

        POSTSMOOTH = auto()
        """
        Restore the original fields after [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] smoothing.

        Provides true lossless output, given [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] `noise_restore` is
        not used. Offers minimal sharpness control and tends to have more significant artifacts.
        """

    class BackBlendMode(CustomIntEnum):
        """When to back-blend the (blurred) difference between the pre- and post-sharpened clips."""

        NONE = auto()
        """
        No back-blending.

        Keeps all [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen] frequencies.
        """

        PRELIMIT = auto()
        """
        Back-blending prior to [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit].

        Provides the weakest low-frequency dampening.
        """

        POSTLIMIT = auto()
        """
        Back-blending after [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit].

        Provides the strongest low-frequency dampening.

        Note:
            Identical to `PRELIMIT` when using `SharpenLimitMode.NONE`, `SharpenLimitMode.SPATIAL_POSTSMOOTH` or
            `SharpenLimitMode.TEMPORAL_POSTSMOOTH`.
        """

        BOTH = auto()
        """
        Back-blending prior to and after [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit].

        Provides a balanced middle ground between `PRELIMIT` and `POSTLIMIT` dampening.

        Note:
            Identical to `PRELIMIT` when using `SharpenLimitMode.NONE`, `SharpenLimitMode.SPATIAL_POSTSMOOTH` or
            `SharpenLimitMode.TEMPORAL_POSTSMOOTH`.
        """

    class SharpenLimitMode(CustomIntEnum):
        """How and when to apply limiting to [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen]."""

        NONE = auto()
        """
        No sharpness limiting.
        """

        SPATIAL_PRESMOOTH = auto()
        """
        Spatial sharpness limiting prior to [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] smoothing.

        Spatial limiting is less accurate, but allows more sharpening. Applying sharpness limiting earlier in the
        algorithm leaves the result softer, but produces fewer artifacts.
        """

        TEMPORAL_PRESMOOTH = auto()
        """
        Temporal sharpness limiting prior to [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] smoothing.

        Temporal limiting is more accurate, but allows less sharpening. Applying sharpness limiting earlier in the
        algorithm leaves the result softer, but produces fewer artifacts.
        """

        SPATIAL_POSTSMOOTH = auto()
        """
        Spatial sharpness limiting after [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] smoothing.

        Spatial limiting is less accurate, but allows more sharpening. Applying sharpness limiting later in the
        algorithm leaves the result sharper, but can produce additional artifacts.
        """

        TEMPORAL_POSTSMOOTH = auto()
        """
        Temporal sharpness limiting after [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] smoothing.

        Temporal limiting is more accurate, but allows less sharpening. Applying sharpness limiting later in the
        algorithm leaves the result sharper, but can produce additional artifacts.
        """

        @cachedproperty
        def is_spatial(self) -> bool:
            """Whether the mode uses spatial sharpness limiting."""
            return self in (self.SPATIAL_PRESMOOTH, self.SPATIAL_POSTSMOOTH)

        @cachedproperty
        def is_temporal(self) -> bool:
            """Whether the mode uses temporal sharpness limiting."""
            return self in (self.TEMPORAL_PRESMOOTH, self.TEMPORAL_POSTSMOOTH)

        @cachedproperty
        def is_presmooth(self) -> bool:
            """Whether the mode applies sharpness limiting prior to smoothing."""
            return self in (self.SPATIAL_PRESMOOTH, self.TEMPORAL_PRESMOOTH)

        @cachedproperty
        def is_postsmooth(self) -> bool:
            """Whether the mode applies sharpness limiting after smoothing."""
            return self in (self.SPATIAL_POSTSMOOTH, self.TEMPORAL_POSTSMOOTH)

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class SearchPostProcess(CustomIntEnum):
        GAUSSBLUR = 0
        GAUSSBLUR_EDGESOFTEN = 1

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class NoiseProcessMode(CustomIntEnum):
        IDENTIFY = 0
        DENOISE = 1

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class SourceMatchMode(CustomIntEnum):
        NONE = 0
        BASIC = 1
        REFINED = 2
        TWICE_REFINED = 3

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class SharpenMode(CustomIntEnum):
        UNSHARP = 0
        UNSHARP_MINMAX = 1

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            **kwargs: Additional arguments to be passed to the parameter category methods. Separate the method name
                from its argument with two underscores, for example: `sharpen_limit__radius=1`.
        """

        settings_methods = (
            self.prefilter,
            self.analyze,
            self.denoise,
            self.basic,
            self.source_match,
            self.lossless,
            self.sharpen,
            self.back_blend,
            self.sharpen_limit,
            self.final,
            self.motion_blur,
        )

        for method in settings_methods:
            prefix = f"{method.__name__}__"

            method(**{k.removeprefix(prefix): kwargs.pop(k) for k in tuple(kwargs) if k.startswith(prefix)})

        if kwargs:
            raise CustomValueError("Unknown arguments were passed.", self.__class__, kwargs)

    def prefilter(
        self,
        *,
        tr: int = 2,
        sc_threshold: float = 0.1,
        strength: tuple[float, float] = (1.9, 0.9),
        limit: tuple[float, float, float] = (3, 7, 2),
        bias: float = 0.51,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = None,
        range_expansion_args: QTGMCArgs.PrefilterToFullRange | None = None,
        postprocess: SearchPostProcess | None = None,
    ) -> Self:
        """
        Configures parameters for the prefilter stage.

        Prepares a suitable search clip to be provided for motion analysis.

        High-level overview:
            - ([QTempGaussMC.deinterlace][vsdeinterlace.QTempGaussMC.deinterlace]) Draft bobbed clip generation:
                Begins with simple spatial interpolation to produce the draft clip, which inherently contains severe
                temporal instability known as bob shimmer.
            - ([QTempGaussMC.repair][vsdeinterlace.QTempGaussMC.repair]) Vertical spatial pre-filtering: Applies a
                vertical binomial blur to filter out residual vertical artifacts.
            - Temporal binomial blurring: Applies a temporal binomial blur to smooth the draft clip, removing the
                shimmer, which prevents [MVTools][vsdenoise.mvtools.mvtools.MVTools] from falsely latching onto the
                shimmer as motion (though this uncompensated blur introduces ghosting).
            - Shimmer masking: Uses a specialized masking process to eliminate the introduced ghosting while retaining
                the shimmer removal.
            - Gaussian blurring post-processing: Applies Gaussian blurring to lower high SAD values caused by sharp
                edges, ensuring edges are properly processed rather than skipped.
            - Edge detail restoration: Conservatively restores essential edge detail from the draft clip back into the
                blurred clip via a limiting process so [MVTools][vsdenoise.mvtools.mvtools.MVTools] retains the ability
                to track motion effectively.
            - Levels optimization:
                Applies level adjustments to brighten dark regions and enhance contrast, enabling
                [MVTools][vsdenoise.mvtools.mvtools.MVTools] to better track dark details, reducing downstream ghosting
                and blurring.

        Args:
            tr: Temporal radius of the binomial blur. Larger values reduce more shimmer but can introduce blurring and
                ghosting. Defaults to 2.
            sc_threshold: Threshold for scene changes. Higher values are less sensitive. Defaults to 0.1.
            strength: Tuple containing the prefilter Gaussian blur sigma and blend weight.

                - First value: Gaussian blur sigma. Higher values result in more blurring.
                - Second value: Blend weight of the Gaussian blur. Higher values give more weight to the
                    Gaussian-blurred clip.

                Defaults to (1.9, 0.9).
            limit: Tuple containing the 3-step limiting thresholds (8-bit) for the Gaussian blur post-processing:

                   - First value: Maximum allowed delta between the temporally blurred clip and the draft clip. Smaller
                    values clamp the draft clip closer to the temporally blurred clip.
                   - Second value: Tolerance threshold for the allowed difference between the clamped clip and the
                    Gaussian-blurred clip before hard clipping triggers. Larger values widen the allowed range for
                    smooth blending before hard clipping is enforced.
                   - Third value: Offset applied to the Gaussian-blurred clip when the second threshold is exceeded.
                    Larger values allow a bigger delta from the Gaussian-blurred clip when clipped.

                Defaults to (3, 7, 2).
            bias: Weight used when blending the Gaussian-blurred clip back with the limited clip. Higher
                values give more weight to the Gaussian-blurred clip. Defaults to 0.51.
            mask_shimmer_args: Additional arguments passed to [mask_shimmer][vsdeinterlace.mask_shimmer]. Defaults
                to None.
            range_expansion_args: Additional arguments passed to
                [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range]. Defaults to None.
        """

        self.prefilter_tr = tr
        self.prefilter_sc_threshold = sc_threshold
        self.prefilter_strength = strength
        self.prefilter_limit = limit
        self.prefilter_bias = bias
        self.prefilter_mask_shimmer_args = fallback(mask_shimmer_args, QTGMCArgs.MaskShimmer())
        self.prefilter_range_expansion_args = fallback(range_expansion_args, QTGMCArgs.PrefilterToFullRange())

        if postprocess is not None and not postprocess.value:  # TODO: remove
            self.prefilter_limit = (0, 0, 0)

        return self

    def analyze(
        self,
        *,
        vectors: MotionVectors | None = None,
        preset: Mapping[str, Any] = MVToolsPreset.HQ_SAD,
        force_tr: int = 0,
        blksize: int | tuple[int, int] = 16,
        overlap: int | tuple[int, int] = 2,
        refine: int = 1,
        thsad_recalc: int | None = None,
        thscd: int | tuple[int | None, float | None] | None = (180, 38.5),
    ) -> Self:
        """
        Configures parameters for motion analysis.

        Performs motion analysis, which is then utilized by all subsequent stages for motion-compensated processing.

        High-level overview:
            - Dynamic temporal radius calculation: Determines the maximum required temporal search radius across all
                actively used settings and processes.
            - Motion vector refinement: Iteratively shrinks block size and calls
                [MVTools.recalculate][vsdenoise.mvtools.mvtools.MVTools.recalculate] to improve motion vector precision.

        Args:
            vectors: Motion vectors to use instead of internally generating them. Defaults to None.
            preset: [MVTools][vsdenoise.mvtools.mvtools.MVTools] preset defining base values for
                [MVTools][vsdenoise.mvtools.mvtools.MVTools]. Defaults to MVToolsPreset.HQ_SAD.
            force_tr: Always analyze motion to at least this value, even if otherwise unnecessary. Useful if you want to
                reuse the generated motion vectors for other tasks. Defaults to 0.
            blksize: Motion analysis block size. Larger values are faster and less sensitive to noise, but less
                accurate. Passing a tuple of values results in asymmetric block sizes. Defaults to 16.
            overlap: The block size divisor for block size overlap. Smaller values reduce blocking artifacts of
                [MVTools][vsdenoise.mvtools.mvtools.MVTools] processes. Passing a tuple of values results in asymmetric
                overlap. Defaults to 2.
            refine: Number of iterations to recalculate motion vectors with halved block size. Improves motion vector
                precision without reducing denoising effectiveness. Defaults to 1.
            thsad_recalc: Only poor-quality new vectors with a SAD above this value will be re-estimated by motion
                search. Only active when refine is used. Defaults to
                [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] `thsad / 2`.
            thscd: Scene-change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

                Defaults to (180, 38.5).
        """

        self.analyze_vectors = vectors
        self.analyze_preset = preset
        self.analyze_force_tr = force_tr
        self.analyze_blksize = blksize
        self.analyze_overlap = overlap
        self.analyze_refine = refine
        self.analyze_thsad_recalc = thsad_recalc
        self.analyze_thscd = thscd

        return self

    @property
    def analyze_thsad_recalc(self) -> int:
        return fallback(self._analyze_thsad_recalc, round(to_arr(self.basic_thsad)[0] / 2))

    @analyze_thsad_recalc.setter
    def analyze_thsad_recalc(self, value: int | None) -> None:
        self._analyze_thsad_recalc = value

    def denoise(
        self,
        *,
        func: DFTTest | _DenoiseFuncTr = _DFTTEST_DEFAULT,
        tr: int = 1,
        mc_denoise: bool = True,
        full_denoise: bool = False,
        deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
        stabilize: float | Literal[False] = 0.4,
        func_comp_args: QTGMCArgs.Compensate | None = None,
        stabilize_comp_args: QTGMCArgs.Compensate | None = None,
        mode: NoiseProcessMode | None = None,
    ) -> Self:
        """
        Configures parameters for the denoise stage.

        Determines how to handle noise in the source, including extraction, removal, deinterlacing, and stabilization.

        High-level overview:
            - Noise handling approaches:
                - Complete denoising: Denoise the source clip entirely, run the denoised clip through the standard
                    deinterlacing routine, and optionally restore a portion of the original noise later in the
                    algorithm.
                - Noise extraction: Denoise the source clip solely to estimate the noise profile, run the source clip
                    through the standard deinterlacing routine (which naturally reduces noise), and optionally restore a
                    portion of the original noise later in the algorithm.

            - Motion-compensated denoising: Motion compensation can optionally be used during the denoising to improve
                the accuracy of the noise estimation.
            - ([QTempGaussMC.deinterlace][vsdeinterlace.QTempGaussMC.deinterlace]) Interlaced noise processing: Because
                the extracted noise is inherently interlaced and standard processing would eliminate it, three
                alternative methods ([QTempGaussMC.NoiseDeintMode][vsdeinterlace.QTempGaussMC.NoiseDeintMode]) are
                available to process it separately.
            - Noise stabilization: The extracted noise can optionally be stabilized at the end of processing using a
                blend of the maximum variance determined through motion compensation and the average of that maximum
                variance and the extracted noise.

        Args:
            func: Denoising function to use. Defaults to DFTTest(sigma=8).
            tr: Temporal radius of the denoising function and its motion compensation. Larger values remove/separate
                more noise. Defaults to 1.
            mc_denoise: Whether to motion-compensate the denoiser being used. Provides more accurate denoising/noise
                extraction when using a non-motion-compensated temporal denoiser. Defaults to True.
            full_denoise: Whether the denoised output will be directly used in all subsequent processing. If `False`,
                the denoising is only for noise extraction. Defaults to False.
            deint: How to 'deinterlace' noise taken from an interlaced source. Defaults to NoiseDeintMode.GENERATE.
            stabilize: Weight used when blending max noise variance with averaged noise. Higher values give more
                weight to the averaged noise. `False` disables stabilization. Defaults to 0.4.
            func_comp_args: Additional arguments passed to
                [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for denoising. Defaults to None.
            stabilize_comp_args: Additional arguments passed to
                [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for stabilization. Defaults to
                None.
        """

        self.denoise_func = func.denoise if isinstance(func, DFTTest) else func
        self.denoise_tr = tr
        self.denoise_mc_denoise = mc_denoise
        self.denoise_full_denoise = full_denoise
        self.denoise_deint = deint
        self.denoise_stabilize = stabilize
        self.denoise_func_comp_args = fallback(func_comp_args, QTGMCArgs.Compensate())
        self.denoise_stabilize_comp_args = fallback(stabilize_comp_args, QTGMCArgs.Compensate())

        if mode is not None:  # TODO: remove
            self.denoise_full_denoise = bool(mode.value)

        return self

    def basic(
        self,
        *,
        bobber: BobberLike = _NNEDI3_DEFAULT,
        tr: int = 2,
        thsad: int | tuple[int, int] = 640,
        thsad2: int | tuple[int, int] | None = None,
        noise_restore: float = 0,
        mask_args: QTGMCArgs.Mask | None = None,
        degrain_args: QTGMCArgs.Degrain | None = None,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = None,
    ) -> Self:
        """
        Configures parameters for the basic stage.

        Creates the basic output of the core algorithm. Intended to eliminate bob shimmer.

        High-level overview:
            - High-quality bobbed clip generation: Begins with high-quality spatial interpolation to produce the bobbed
                clip, which inherently contains severe temporal instability known as bob shimmer.
            - ([QTempGaussMC.repair][vsdeinterlace.QTempGaussMC.repair]) Motion SAD masking: Generates a motion-vector
                SAD mask to blend [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise] output over the bobbed
                clip, protecting static/low-motion detail.
            - Motion-compensated temporal binomial smoothing: Applies a motion-compensated temporal binomial blur to
                smooth the bobbed clip, removing the shimmer while avoiding ghosting artifacts.
            - Shimmer masking: Uses a specialized masking process to eliminate the introduced blurring while retaining
                the shimmer removal.
            - Additional refinements: Passes the temporally smoothed clip through optional fine-tuning processes:
                - [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match]
                - [QTempGaussMC.lossless][vsdeinterlace.QTempGaussMC.lossless]
                - [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen]
                - [QTempGaussMC.back_blend][vsdeinterlace.QTempGaussMC.back_blend]
                - [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit]

            - Noise restoration: Optionally restores noise previously extracted by
                [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise] at the end of this stage.

        Args:
            bobber: Bobber to use for spatial interpolation. Defaults to NNEDI3(nsize=1).
            tr: Temporal radius of the motion-compensated binomial blur. Larger values reduce more shimmer but can
                introduce blurring and ghosting. Defaults to 2.
            thsad: SAD threshold of the motion-compensated binomial blur. Larger values reduce more shimmer but can
                introduce blurring and ghosting. Passing a tuple of values results in per-plane thresholds. Defaults to
                640.
            thsad2: Second SAD threshold of the motion-compensated linear blur. Larger values clean more artifacts but
                can introduce blurring and ghosting. Passing a tuple of values results in per-plane thresholds. Defaults
                to None.
            noise_restore: Amount of noise to restore after this stage. Used to retain stable noise. Defaults to 0.
            mask_args: Additional arguments passed to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask]. Only used
                for [QTempGaussMC.repair][vsdeinterlace.QTempGaussMC.repair]. Defaults to {"ml": 10}.
            degrain_args: Additional arguments passed to the internal `_binomial_degrain` call. Defaults to None.
            mask_shimmer_args: Additional arguments passed to [mask_shimmer][vsdeinterlace.mask_shimmer]. Defaults
                to {"erosion_distance": 0}.
        """

        self.basic_bobber = (
            deepcopy(bobber) if isinstance(bobber, Bobber) else Bobber.ensure_obj(bobber, self.__class__)
        )
        self.basic_tr = tr
        self.basic_thsad = thsad
        self.basic_thsad2 = thsad2
        self.basic_noise_restore = noise_restore
        self.basic_mask_args = QTGMCArgs.Mask(ml=10) | (mask_args or {})
        self.basic_degrain_args = fallback(degrain_args, QTGMCArgs.Degrain())
        self.basic_mask_shimmer_args = QTGMCArgs.MaskShimmer(erosion_distance=0) | (mask_shimmer_args or {})

        return self

    def source_match(
        self,
        *,
        iterations: Literal[0, 1, 2, 3] = 0,
        similarity: float = 0.5,
        bobber: BobberLike | None = None,
        tr: int = 1,
        enhance: float = 0.5,
        degrain_args: QTGMCArgs.Degrain | None = None,
        mode: SourceMatchMode | None = None,
    ) -> Self:
        """
        Configures parameters for source match processing.

        Creates higher-fidelity output with extra processing; acts as an alternative method for sharpness restoration.

        High-level overview:
            - Error-adjusted source matching: Computes a weighted error-correction factor based on temporal radius and
                similarity, adjusting the input clip to compensate for the upcoming blur before re-interpolating and
                applying smoothing.
            - Detail enhancement: Optionally applies unsharpening to the result when `enhance` is used.
            - Residual refinement pass: For multiple iterations, isolates the difference between the original input and
                the current matched clip, interpolates and smooths this residual error (applying an additional
                error-adjustment pass if `iterations` > `2`), and merges it back to restore fine detail missed during
                the initial pass.

        Note:
            - When source matching is used:
                - [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen] is disabled by default, as source matching
                    acts as an alternative form of sharpness restoration.
                - [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit] is disabled by default, as it
                    reduces the accuracy of source matching.

        Args:
            iterations: Number of source match iterations to perform. Higher values are slower and more accurate. Using
                `2` or `3` iterations restores almost exact source detail but is sensitive to noise and introduces
                occasional aliasing (to a lesser extent for `3`). Requires
                [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] `tr` > `0` Defaults to 0.
            similarity: Temporal similarity of the error from frame to frame. Lower values make the result sharper.
                Defaults to 0.5.
            bobber: Bobber to use for refined spatial interpolation. Only used for `iterations` > `1`. Defaults to
                [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] `bobber`.
            tr: Temporal radius of the refinement motion-compensated binomial blur. Larger values reduce more shimmer
                but can introduce blurring and ghosting. Only used for `iterations` > `1`. Defaults to 1.
            enhance: Enhances detail found by `iterations` > `1`. Higher values exaggerate detail more. Defaults to 0.5.
            degrain_args: Additional arguments passed to the internal `_binomial_degrain` call. Defaults to None.
        """

        self.source_match_iterations = iterations
        self.source_match_similarity = similarity
        self.source_match_bobber = bobber
        self.source_match_tr = tr
        self.source_match_enhance = enhance
        self.source_match_degrain_args = fallback(degrain_args, QTGMCArgs.Degrain())

        if mode is not None:  # TODO: remove
            self.source_match_iterations = mode.value  # type: ignore

        return self

    @property
    def source_match_bobber(self) -> Bobber:
        return fallback(self._source_match_bobber, self.basic_bobber)

    @source_match_bobber.setter
    def source_match_bobber(self, value: BobberLike | None) -> None:
        if value is None:
            self._source_match_bobber = value
            return

        if isinstance(value, Bobber):
            self._source_match_bobber = deepcopy(value)
        else:
            self._source_match_bobber = Bobber.ensure_obj(value, self.__class__)

    def lossless(self, *, mode: LosslessMode = LosslessMode.NONE, anti_comb: bool = True) -> Self:
        """
        Configures parameters for lossless processing.

        Creates higher-fidelity output by restoring the original fields.

        High-level overview:
            - Source field weaving: Weaves the original fields together with the newly smoothed fields to preserve the
                original lines, removing the original field alteration introduced by temporal blurring.
            - Residual combing reduction: Applies vertical median filtering to clean up residual combing caused by
                mismatches between the original fields and the processed fields.

        Args:
            mode: When to put the original fields into the output. Defaults to LosslessMode.NONE.
            anti_comb: Whether to apply combing reduction post-processing. Defaults to True.
        """

        self.lossless_mode = mode
        self.lossless_anti_comb = anti_comb

        return self

    def sharpen(
        self,
        *,
        strength: float | None = None,
        offset: float | tuple[float, float] | Literal[False] = 1,
        thin: float = 0,
        mode: SharpenMode | None = None,
    ) -> Self:
        """
        Configures parameters for sharpening.

        Re-sharpens the output after temporal smoothing is performed.

        High-level overview:
            - Pre-blur range limiting: Calculates the local vertical average and offsets it prior to applying the blur
                used for unsharpening to increase vertical sharpening while reducing overshoot/undershoot.
            - Unsharpening: Applies unsharpening onto the temporally smoothed clip to restore image sharpness.
            - Horizontal edge thinning: Optionally thins horizontal edges that have been widened due to interpolation
                into neighboring field lines.

        Args:
            strength: Sharpening strength. Higher values result in more sharpening. Defaults to 1 when
                [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match] `iterations` is `0`, and 0
                otherwise.
            offset: Shifts the unsharpen blur source to the vertical min/max average ± this value (8-bit). Smaller
                values result in more vertical sharpening. Passing a tuple of values results in asymmetric offsetting.
                `False` disables range limiting. Defaults to 1.
            thin: How much to thin down horizontal edges. Higher values result in more thinning. Defaults to 0.
        """

        self.sharpen_strength = strength
        self.sharpen_offset = offset is not False and normalize_seq(offset, 2)
        # Premultiply thin strength by the inverse L2 norm of the binomial blurs used in the algorithm.
        # After multiplication, thin = 1.0 means edges found by median filtering are merged at full strength.
        self.sharpen_thin = thin * 32 / sqrt(105)

        if mode is not None and not mode.value:  # TODO: remove
            self.sharpen_offset = False

        return self

    @property
    def sharpen_strength(self) -> float:
        return fallback(self._sharpen_strength, 0 if self._source_match_enabled else 1)

    @sharpen_strength.setter
    def sharpen_strength(self, value: float | None) -> None:
        self._sharpen_strength = value

    def back_blend(self, *, mode: BackBlendMode = BackBlendMode.BOTH, scale: float = 2) -> Self:
        """
        Configures parameters for back-blending.

        Improves [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen] fidelity by dampening low-frequency
        enhancement caused by unsharpening.

        High-level overview:
            - Low-frequency back-blending: Gaussian-blurs the pre- and post-sharpening difference to isolate broad
                low-frequency shifts, then merges that blurred difference back onto the source to preserve only
                high-frequency edge sharpening.

        Args:
            mode: When to back-blend the (blurred) sharpening difference. Defaults to BackBlendMode.BOTH.
            scale: Scale factor for the Gaussian blur sigma applied to the sharpening difference. Lower values dampen
                sharpening more aggressively. Defaults to 2.
        """

        self.back_blend_mode = mode
        self.back_blend_scale = scale

        return self

    def sharpen_limit(
        self,
        *,
        mode: SharpenLimitMode | None = None,
        radius: int = 1,
        clamp: float | tuple[float, float] = 0,
        comp_args: QTGMCArgs.Compensate | None = None,
    ) -> Self:
        """
        Configures parameters for sharpness limiting.

        Limits the effect of [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen] to reduce oversharpening
        artifacts.

        High-level overview:
            - Sharpness limiting approaches:
                - Spatial limiting: Clamps the sharpened clip's pixel values to the local spatial minimum and maximum
                    bounds of the bobbed clip.
                - Motion-compensated temporal limiting: Clamps the sharpened clip using motion-compensated reference
                    frames from the bobbed clip.

        Args:
            mode: How and when to apply limiting to [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen]. Defaults
                to SharpenLimitMode.TEMPORAL_PRESMOOTH when
                [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match] `iterations` is `0` and
                SharpenLimitMode.NONE otherwise.
            radius: Radius of the sharpness limiting. Larger values allow more sharpening. Defaults to 1.
            clamp: How much undershoot/overshoot to allow (8-bit). Larger values result in less limiting. Passing a
                tuple of values allows for asymmetric limiting. Defaults to 0.
            comp_args: Additional arguments passed to [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate]
                for temporal limiting. Defaults to None.
        """

        self.sharpen_limit_mode = mode
        self.sharpen_limit_radius = radius
        self.sharpen_limit_clamp = normalize_seq(clamp, 2)
        self.sharpen_limit_comp_args = fallback(comp_args, QTGMCArgs.Compensate())

        return self

    @property
    def sharpen_limit_mode(self) -> SharpenLimitMode:
        return fallback(
            self._sharpen_limit_mode,
            self.SharpenLimitMode.NONE if self._source_match_enabled else self.SharpenLimitMode.TEMPORAL_PRESMOOTH,
        )

    @sharpen_limit_mode.setter
    def sharpen_limit_mode(self, value: SharpenLimitMode | None) -> None:
        self._sharpen_limit_mode = value

    def final(
        self,
        *,
        tr: int = 1,
        thsad: int | tuple[int, int] = 256,
        thsad2: int | tuple[int, int] | None = None,
        noise_restore: float = 0,
        degrain_args: QTGMCArgs.Degrain | None = None,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = None,
    ) -> Self:
        """
        Configures parameters for the final stage.

        Creates the final output of the core algorithm. Intended to eliminate residual artifacts.

        High-level overview:
            - Motion-compensated temporal linear smoothing: Applies a motion-compensated temporal linear blur to smooth
                the output of [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic], cleaning any residual artifacts.
            - Shimmer masking: Uses a specialized masking process to eliminate the introduced blurring while retaining
                the artifact removal.
            - Additional refinements: Passes the temporally smoothed clip through optional fine-tuning processes:
                - [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit]
                - [QTempGaussMC.lossless][vsdeinterlace.QTempGaussMC.lossless]

            - Noise restoration: Optionally restores noise previously extracted by
                [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise] at the end of this stage.

        Args:
            tr: Temporal radius of the motion-compensated linear blur. Larger values clean more artifacts but can
                introduce blurring and ghosting. Defaults to 1.
            thsad: SAD threshold of the motion-compensated linear blur. Larger values clean more artifacts but can
                introduce blurring and ghosting. Passing a tuple of values results in per-plane thresholds. Defaults to
                256.
            thsad2: Second SAD threshold of the motion-compensated linear blur. Larger values clean more artifacts but
                can introduce blurring and ghosting. Passing a tuple of values results in per-plane thresholds. Defaults
                to None.
            noise_restore: Amount of noise to restore after this stage. Used to retain any noise. Defaults to 0.
            degrain_args: Additional arguments passed to [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain].
                Defaults to None.
            mask_shimmer_args: Additional arguments passed to [mask_shimmer][vsdeinterlace.mask_shimmer]. Defaults
                to None.
        """

        self.final_tr = tr
        self.final_thsad = thsad
        self.final_thsad2 = thsad2
        self.final_noise_restore = noise_restore
        self.final_degrain_args = fallback(degrain_args, QTGMCArgs.Degrain())
        self.final_mask_shimmer_args = fallback(mask_shimmer_args, QTGMCArgs.MaskShimmer())

        return self

    def motion_blur(
        self,
        *,
        shutter_angle: tuple[float, float] | Literal[False] = False,
        fps_divisor: int = 1,
        blur_args: QTGMCArgs.Blur | None = None,
        mask_args: QTGMCArgs.Mask | None = None,
    ) -> Self:
        """
        Configures parameters for the motion blur stage.

        Simulates realistic camera shutter blur to smooth playback motion, primarily when reducing output frame rate.

        High-level overview:
            - Shutter angle calculation: Computes the required blur intensity based on the estimated input shutter
                angle, the output shutter angle, and the frame rate divisor.
            - Motion-compensated blurring: Applies vector-based directional blur along motion vectors when the required
                blur amount is non-zero.
            - Motion-adaptive masking: Generates a mask based on motion to selectively merge motion blur into the source
                while keeping static areas sharp.
            - Frame rate reduction: Optionally decimates frame rate (e.g., dropping every other frame for single-rate
                output) after motion blur application.

        Args:
            shutter_angle: Tuple containing the source and output shutter angles. Motion blur is applied if they do not
                match. `False` disables motion blur. Defaults to False.
            fps_divisor: Factor by which to smoothly reduce frame rate. Defaults to 1.
            blur_args: Additional arguments passed to [MVTools.flow_blur][vsdenoise.mvtools.mvtools.MVTools.flow_blur].
                Defaults to None.
            mask_args: Additional arguments passed to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask]. Defaults
                to {"ml": 4}.
        """

        self.motion_blur_shutter_angle = shutter_angle
        self.motion_blur_fps_divisor = fps_divisor
        self.motion_blur_blur_args = fallback(blur_args, QTGMCArgs.Blur())
        self.motion_blur_mask_args = QTGMCArgs.Mask(ml=4) | (mask_args or {})

        return self

    @property
    def _noise_restore_enabled(self) -> bool:
        return bool(self.basic_noise_restore or self.final_noise_restore)

    @property
    def _mc_denoise_tr(self) -> int:
        return (
            self.denoise_tr
            if self.denoise_mc_denoise and (self.denoise_full_denoise or self._noise_restore_enabled)
            else 0
        )

    @property
    def _stabilization_enabled(self) -> bool:
        return self.denoise_stabilize is not False and self._noise_restore_enabled

    @property
    def _source_match_enabled(self) -> bool:
        return bool(self.source_match_iterations and self.basic_tr)

    @property
    def _sharpen_sigma(self) -> float:
        # Combined variance of the (binomial) basic blur and (linear) final blur.
        return sqrt(self.basic_tr / 2 + self.final_tr * (self.final_tr + 1) / 3)

    @property
    def _back_blend_sigma(self) -> float:
        passes = 1 + bool(
            self.back_blend_mode is self.BackBlendMode.BOTH
            and self.sharpen_limit_mode.is_presmooth
            and self.sharpen_limit_radius
        )

        return self._sharpen_sigma * (self.back_blend_scale / sqrt(passes))

    @property
    def _sharpening_enabled(self) -> bool:
        return bool(
            ((self.sharpen_strength and self._sharpen_sigma) or self.sharpen_thin)
            and (self.back_blend_mode is self.BackBlendMode.NONE or self._back_blend_sigma)
        )

    @property
    def _sharpness_limiting_enabled(self) -> bool:
        return bool(
            self.sharpen_limit_mode is not self.SharpenLimitMode.NONE
            and self.sharpen_limit_radius
            and self._sharpening_enabled
        )


class QTGMCGraph(VSObject):
    """
    Processing graph for an individual [QTempGaussMC][vsdeinterlace.QTempGaussMC] run.

    The graph exposes each processing stage as a lazily evaluated cached property. It is returned alongside the output
    clip when `return_graph=True` is passed to a [QTempGaussMC][vsdeinterlace.QTempGaussMC] processing method. This
    allows the intermediate clips and motion vectors used to produce the output to be inspected or reused.

    Returned graphs are frozen after evaluation so that access is limited to stages used to construct the output.

    Usage examples:
        - Reusing the [MVTools][vsdenoise.mvtools.mvtools.MVTools] object:
        ```python
        qtgmc = vsdeinterlace.QTempGaussMC()
        deinterlaced, graph = qtgmc.deinterlace(clip, return_graph=True)
        mv = graph.mv
        ```
        - Inspecting the output of [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter]:
        ```python
        qtgmc = vsdeinterlace.QTempGaussMC()
        deinterlaced, graph = qtgmc.deinterlace(clip, return_graph=True)
        prefilter = graph.prefilter
        ```

    Additional usage info: [JET Encoding Guide](https://jaded-encoding-thaumaturgy.github.io/JET-guide/master/filtering/situational/qtgmc/)
    """

    class Mode(CustomIntEnum):
        """Processing mode used to construct the graph."""

        DEINTERLACE = auto()
        """
        Deinterlace interlaced input.

        Interpolates missing fields to reconstruct progressive frames.
        [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur] `fps_divisor` is respected.
        """

        BOB = auto()
        """
        Bob interlaced input.

        Interpolates missing fields to reconstruct progressive frames.
        [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur] `fps_divisor` is ignored.
        """

        REPAIR = auto()
        """
        Repair badly deinterlaced input.

        Drops half the fields to recreate an interlaced clip using the remaining ones.
        """

        DESHIMMER = auto()
        """
        Deshimmer progressive input.

        Removes horizontal shimmering artifacts from progressive sources.
        """

        def __call__(
            self,
            clip: vs.VideoNode,
            tff: FieldBasedLike | bool | None,
            settings: _QTGMCBuilder,
            func: FuncExcept,
        ) -> QTGMCGraph:
            return QTGMCGraph(self, clip, tff, settings, func)

    class _FrozenCache(dict[str, Any]):
        def __contains__(self, key: object) -> bool:
            if super().__contains__(key):
                return True

            raise AttributeError(key)

    def __init__(
        self,
        mode: Mode,
        clip: vs.VideoNode,
        tff: FieldBasedLike | bool | None,
        settings: _QTGMCBuilder,
        func: FuncExcept,
    ) -> None:
        """
        Args:
            mode: Processing mode used to construct the graph.
            clip: Clip to process.
            tff: Field order (top-field-first). If `None`, inferred from the clip.
            settings: `_QTGMCBuilder` instance containing the settings for each processing stage.
            func: Function returned for custom error handling. This should only be set by VS package developers.

        Raises:
            UnsupportedFieldBasedError: If `clip` is [FieldBased.PROGRESSIVE][vstools.FieldBased.PROGRESSIVE] and `mode`
                is not `Mode.DESHIMMER`.
        """

        self.clip = clip
        self.tff = FieldBased.from_param_or_video(tff, clip, True, func)
        self.mode = mode
        self.settings = settings
        self.func = func

        if not (self.tff.is_inter or mode is self.Mode.DESHIMMER):
            raise UnsupportedFieldBasedError("This mode is incompatible with progressive video!", func)

    def freeze(self) -> Self:
        """
        Freeze the graph at its current evaluation state.

        Cached properties remain accessible; however, attempting to access a property that has not been evaluated raises
        an `AttributeError`
        """

        cache = self.__dict__.setdefault(cachedproperty.cache_key, {})

        if not isinstance(cache, self._FrozenCache):
            self.__dict__[cachedproperty.cache_key] = self._FrozenCache(cache)

        return self

    @cachedproperty
    def draft(self) -> vs.VideoNode:
        """
        Draft processed clip.

        Used as a base for [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter] and
        [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise].

        Only available when using denoising or motion-compensated processing.
        """

        if self.mode in (self.Mode.DEINTERLACE, self.Mode.BOB):
            return Catrom().bob(self.clip, tff=self.tff)

        return self.clip

    @cachedproperty
    def prefilter(self) -> vs.VideoNode:
        """
        Output of [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter].

        Only available when motion vectors need to be generated.
        """

        if self.mode is self.Mode.REPAIR:
            search = BlurMatrix.BINOMIAL()(self.draft, mode=ConvMode.VERTICAL, func=self.func)
        else:
            search = self.draft

        if self.settings.prefilter_tr:
            smoothed = BlurMatrix.BINOMIAL(self.settings.prefilter_tr, mode=ConvMode.TEMPORAL)(
                sc_detect(search, self.settings.prefilter_sc_threshold), scenechange=True, func=self.func
            )
            smoothed = mask_shimmer(smoothed, search, **self.settings.prefilter_mask_shimmer_args, func=self.func)
        else:
            smoothed = search

        gauss_sigma, blend_weight = self.settings.prefilter_strength
        lim1, lim2, lim3 = [scale_delta(thr, 8, self.clip) for thr in self.settings.prefilter_limit]
        apply_blur = bool(gauss_sigma and blend_weight)

        blurred = gauss_blur(smoothed, gauss_sigma) if apply_blur else smoothed

        if apply_blur or (self.settings.prefilter_tr and lim1 and (lim2 or lim3)):
            blurred = norm_expr(
                [blurred, smoothed, search],
                "y x y - {weight} * + BLUR! z y {lim1} - y {lim1} + clamp TWEAK! "
                "BLUR@ {lim2} + TWEAK@ < BLUR@ {lim3} + BLUR@ {lim2} - TWEAK@ > BLUR@ {lim3} - "
                "TWEAK@ BLUR@ TWEAK@ - {bias} * + ? ?",
                weight=blend_weight,
                lim1=lim1,
                lim2=lim2,
                lim3=lim3,
                bias=self.settings.prefilter_bias,
                func=self.func,
            )

        return prefilter_to_full_range(blurred, func=self.func, **self.settings.prefilter_range_expansion_args)

    @cachedproperty
    def mv(self) -> MVTools:
        """
        [MVTools][vsdenoise.mvtools.mvtools.MVTools] instance used during processing.

        Only available when using motion-compensated processing.
        """

        preset = dict(self.settings.analyze_preset)
        if not self.settings.analyze_vectors:
            preset.update(search_clip=self.prefilter)

        mv = MVTools(self.draft, vectors=self.settings.analyze_vectors, **preset)

        if self.settings.analyze_vectors:
            return mv

        tr = max(
            self.settings.analyze_force_tr,
            self.settings._mc_denoise_tr,
            self.settings._stabilization_enabled,
            self.settings.basic_tr,
            self._repair_mask_enabled,
            self.settings.source_match_tr
            if self.settings.source_match_iterations > 1 and self.settings._source_match_enabled
            else 0,
            self.settings.sharpen_limit_radius
            if self.settings.sharpen_limit_mode.is_temporal and self.settings._sharpness_limiting_enabled
            else 0,
            self.settings.final_tr,
            bool(self._motion_blur_level),
        )

        blksize = self.settings.analyze_blksize
        mv.analyze(tr=tr, blksize=blksize, overlap_div=self.settings.analyze_overlap)

        for _ in range(self.settings.analyze_refine):
            blksize = refine_blksize(blksize)
            mv.recalculate(
                thsad=self.settings.analyze_thsad_recalc, blksize=blksize, overlap_div=self.settings.analyze_overlap
            )

        return mv

    @cachedproperty
    def denoise(self) -> vs.VideoNode:
        """Output of [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise]."""

        return self._apply_denoise if self.settings.denoise_full_denoise else self.clip

    @cachedproperty
    def noise(self) -> vs.VideoNode:
        """
        Noise extracted by [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise].

        Only available when using noise restoration.
        """

        noise = self.clip.std.MakeDiff(self._apply_denoise)

        if self.mode in (self.Mode.DEINTERLACE, self.Mode.BOB):
            match self.settings.denoise_deint:
                case self.settings.NoiseDeintMode.WEAVE:
                    noise = noise.std.SeparateFields(self.tff.is_tff).std.DoubleWeave(self.tff.is_tff)
                case self.settings.NoiseDeintMode.BOB:
                    noise = Catrom().bob(noise, tff=self.tff)
                case self.settings.NoiseDeintMode.GENERATE:
                    noise = noise.std.SeparateFields(self.tff.is_tff)

                    noise_min = Morpho.inpand(noise, sw=2, sh=1, func=self.func)
                    noise_max = Morpho.expand(noise, sw=2, sh=1, func=self.func)

                    noise_gen = Grainer.GAUSS(
                        noise,
                        # Use the maximum variance that satisfies the 3-sigma rule.
                        ((0.5 * 255) / 3) ** 2,
                        protect_edges=False,
                        protect_neutral_chroma=False,
                        neutral_out=True,
                    )
                    noise_gen = norm_expr(
                        [noise_max, noise_min, noise_gen],
                        # Limitation: This gain map can never reach 1.0 with integer formats.
                        # Peak gain at 8-bit: (255 - 128) / 256 + 0.5 = ~0.996.
                        "y x y - z neutral - range_size / 0.5 + * +",
                        func=self.func,
                    )
                    noise = reweave(noise, noise_gen, self.tff.field, self.func)

            noise = FieldBased.PROGRESSIVE.apply(noise)

        if self.settings._stabilization_enabled:
            noise_comp, _ = self.mv.compensate(
                noise,
                direction=MVDirection.BACKWARD,
                tr=1,
                thscd=self.settings.analyze_thscd,
                interleave=False,
                **self.settings.denoise_stabilize_comp_args,
            )

            noise = norm_expr(
                [noise, *noise_comp],
                "x neutral - abs y neutral - abs > x y ? dup x y + 2 / swap - {weight} * +",
                weight=self.settings.denoise_stabilize,
                func=self.func,
            )

        return noise

    @cachedproperty
    def bob_input(self) -> vs.VideoNode:
        """
        Prepared input clip for high-quality interpolation.

        Used as a base for the internal `_interpolate` method and as a reference for
        [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match].
        """

        if self.mode is self.Mode.REPAIR:
            return reinterlace(self.denoise, self.tff, self.func)

        return self.denoise

    @cachedproperty
    def bobbed(self) -> vs.VideoNode:
        """
        High-quality bobbed clip.

        Used as a spatial interpolation base for [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] and
        [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match].
        """

        bobbed = self._interpolate(self.bob_input, self.settings.basic_bobber)

        if self._repair_mask_enabled:
            mask = self.mv.mask(
                direction=MVDirection.BACKWARD,
                kind=MaskMode.SAD,
                thscd=self.settings.analyze_thscd,
                **self.settings.basic_mask_args,
            )
            bobbed = self.denoise.std.MaskedMerge(bobbed, mask)

        return bobbed

    @cachedproperty
    def basic(self) -> vs.VideoNode:
        """Output of [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic]."""

        smoothed = self._binomial_degrain(self.bobbed, self.settings.basic_tr, **self.settings.basic_degrain_args)

        if self.settings.basic_tr:
            smoothed = mask_shimmer(smoothed, self.bobbed, **self.settings.basic_mask_shimmer_args, func=self.func)

        if self.settings._source_match_enabled:
            smoothed = self._source_match(smoothed)

        if self.settings.lossless_mode is self.settings.LosslessMode.PRESHARPEN:
            smoothed = self._lossless(smoothed)

        if self.settings._sharpening_enabled:
            resharp = self._sharpen(smoothed)

            if self.settings.sharpen_limit_mode.is_presmooth and self.settings._sharpness_limiting_enabled:
                if self.settings.back_blend_mode in (
                    self.settings.BackBlendMode.PRELIMIT,
                    self.settings.BackBlendMode.BOTH,
                ):
                    resharp = self._back_blend(resharp, smoothed)

                resharp = self._sharpen_limit(resharp)

                if self.settings.back_blend_mode in (
                    self.settings.BackBlendMode.POSTLIMIT,
                    self.settings.BackBlendMode.BOTH,
                ):
                    resharp = self._back_blend(resharp, smoothed)
            elif self.settings.back_blend_mode is not self.settings.BackBlendMode.NONE:
                resharp = self._back_blend(resharp, smoothed)
        else:
            resharp = smoothed

        return self._noise_restore(resharp, self.settings.basic_noise_restore)

    @cachedproperty
    def final(self) -> vs.VideoNode:
        """Output of [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final]."""

        if self.settings.final_tr:
            smoothed = self.mv.degrain(
                self.basic,
                tr=self.settings.final_tr,
                thsad=self.settings.final_thsad,
                thsad2=self.settings.final_thsad2,
                thscd=self.settings.analyze_thscd,
                **self.settings.final_degrain_args,
            )
        else:
            smoothed = self.basic

        if smoothed is not self.bobbed:
            smoothed = mask_shimmer(smoothed, self.bobbed, **self.settings.final_mask_shimmer_args, func=self.func)

        if self.settings.sharpen_limit_mode.is_postsmooth and self.settings._sharpness_limiting_enabled:
            smoothed = self._sharpen_limit(smoothed)

        if self.settings.lossless_mode is self.settings.LosslessMode.POSTSMOOTH:
            smoothed = self._lossless(smoothed)

        return self._noise_restore(smoothed, self.settings.final_noise_restore)

    @cachedproperty
    def motion_blur(self) -> vs.VideoNode:
        """Output of [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur]."""

        if self._motion_blur_level:
            blurred = self.mv.flow_blur(
                self.final,
                blur=self._motion_blur_level,
                thscd=self.settings.analyze_thscd,
                **self.settings.motion_blur_blur_args,
            )

            if self.settings.motion_blur_mask_args.get("ml") != 0:
                mask = self.mv.mask(
                    direction=MVDirection.BACKWARD,
                    kind=MaskMode.VECTOR_LENGTH,
                    thscd=self.settings.analyze_thscd,
                    **self.settings.motion_blur_mask_args,
                )

                blurred = self.final.std.MaskedMerge(blurred, mask)
        else:
            blurred = self.final

        if self._motion_blur_fps_divisor > 1:
            blurred = blurred[:: self._motion_blur_fps_divisor]

        return blurred

    @cachedproperty
    def _apply_denoise(self) -> vs.VideoNode:
        if self.settings._mc_denoise_tr:
            denoised = self.mv.compensate(
                tr=self.settings.denoise_tr,
                thscd=self.settings.analyze_thscd,
                temporal_func=lambda clip: self.settings.denoise_func(clip, tr=self.settings.denoise_tr),
                **self.settings.denoise_func_comp_args,
            )
        else:
            denoised = self.settings.denoise_func(self.draft, tr=self.settings.denoise_tr)

        if self.mode in (self.Mode.DEINTERLACE, self.Mode.BOB):
            denoised = reinterlace(denoised, self.tff, self.func)

        return denoised

    @cachedproperty
    def _repair_mask_enabled(self) -> bool:
        return bool(self.mode is self.Mode.REPAIR and self.settings.basic_mask_args.get("ml") != 0)

    @cachedproperty
    def _motion_blur_fps_divisor(self) -> int:
        return 1 if self.mode is self.Mode.BOB else self.settings.motion_blur_fps_divisor

    @cachedproperty
    def _motion_blur_level(self) -> float:
        if self.settings.motion_blur_shutter_angle is False:
            return 0

        angle_in, angle_out = self.settings.motion_blur_shutter_angle

        return (angle_out * self._motion_blur_fps_divisor - angle_in) / 3.60

    def _interpolate(self, clip: vs.VideoNode, bobber: Bobber) -> vs.VideoNode:
        if self.mode is not self.Mode.DESHIMMER:
            clip = bobber.bob(clip, tff=self.tff)

        return clip

    def _binomial_degrain(self, clip: vs.VideoNode, tr: int, **degrain_args: Any) -> vs.VideoNode:
        if not tr:
            return clip

        return self.mv.degrain(
            clip,
            tr=tr,
            thsad=self.settings.basic_thsad,
            thsad2=self.settings.basic_thsad2,
            thscd=self.settings.analyze_thscd,
            weights=BlurMatrix.BINOMIAL(radius=tr),
            **degrain_args,
        )

    def _source_match(self, clip: vs.VideoNode) -> vs.VideoNode:
        def error_adjustment(ref: vs.VideoNode, clip: vs.VideoNode, tr: int) -> vs.VideoNode:
            if not tr:
                return ref

            tr_f = 2 * tr - 1
            tr_s = 2**tr_f
            binomial_coeff = comb(tr_f, tr)
            error_adj = tr_s / (binomial_coeff + self.settings.source_match_similarity * (tr_s - binomial_coeff))

            return unsharpen(ref, error_adj, clip, func=error_adjustment)

        if self.mode is not self.Mode.DESHIMMER:
            clip = reinterlace(clip, self.tff, self._source_match)

        adjusted = error_adjustment(self.bob_input, clip, self.settings.basic_tr)
        new_bobbed = self._interpolate(adjusted, self.settings.basic_bobber)
        matched = self._binomial_degrain(new_bobbed, self.settings.basic_tr, **self.settings.basic_degrain_args)

        if self.settings.source_match_iterations > 1:
            if self.settings.source_match_enhance:
                matched = unsharpen(
                    matched, self.settings.source_match_enhance, BlurMatrix.BINOMIAL(), func=self._source_match
                )

            if self.mode is not self.Mode.DESHIMMER:
                clip = reinterlace(matched, self.tff, self._source_match)
            else:
                clip = matched

            diff = self.bob_input.std.MakeDiff(clip)
            refine_bobbed = self._interpolate(diff, self.settings.source_match_bobber)
            refine_matched = self._binomial_degrain(
                refine_bobbed, self.settings.source_match_tr, **self.settings.source_match_degrain_args
            )

            if self.settings.source_match_iterations > 2:
                refine_adjusted = error_adjustment(refine_bobbed, refine_matched, self.settings.source_match_tr)
                refine_matched = self._binomial_degrain(
                    refine_adjusted, self.settings.source_match_tr, **self.settings.source_match_degrain_args
                )

            return matched.std.MergeDiff(refine_matched)

        return matched

    def _lossless(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.mode is self.Mode.DESHIMMER or clip is self.bobbed:
            return clip

        fields_src = self.denoise.std.SeparateFields(self.tff.is_tff)
        if self.mode is self.Mode.REPAIR:
            fields_src = fields_src.std.SelectEvery(4, (0, 3))
        fields_flt = clip.std.SeparateFields(self.tff.is_tff).std.SelectEvery(4, (1, 2))

        woven = reweave(fields_src, fields_flt, self.tff.field, self._lossless)

        if self.settings.lossless_anti_comb:
            median_diff = median_blur(woven, mode=ConvMode.VERTICAL, func=self._lossless).std.MakeDiff(woven)
            fields_diff = median_diff.std.SeparateFields(self.tff.is_tff).std.SelectEvery(4, (1, 2))

            cleaned_diff = norm_expr(
                [median_blur(fields_diff, mode=ConvMode.VERTICAL, func=self._lossless), fields_diff],
                "x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?",
                func=self._lossless,
            )
            cleaned_diff = repair.Mode.MINMAX_SQUARE1(cleaned_diff, remove_grain.Mode.MINMAX_AROUND2(cleaned_diff))
            woven = reweave(fields_src, fields_flt.std.MergeDiff(cleaned_diff), self.tff.field, self._lossless)

        return FieldBased.PROGRESSIVE.apply(woven)

    def _sharpen(self, clip: vs.VideoNode) -> vs.VideoNode:
        resharp = clip

        if self.settings.sharpen_strength and self.settings._sharpen_sigma:
            if self.settings.sharpen_offset is not False:
                dark_offset, bright_offset = self.settings.sharpen_offset

                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL, func=self._sharpen)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL, func=self._sharpen)

                resharp = norm_expr(
                    [clip, source_min, source_max],
                    "y z + 2 / AVG! AVG@ x > AVG@ {dark_offset} - AVG@ x < AVG@ {bright_offset} + x ? ?",
                    dark_offset=scale_delta(dark_offset, 8, self.clip),
                    bright_offset=scale_delta(bright_offset, 8, self.clip),
                    func=self._sharpen,
                )

            resharp = unsharpen(
                clip,
                self.settings.sharpen_strength,
                gauss_blur(resharp, self.settings._sharpen_sigma),
                func=self._sharpen,
            )

        if self.settings.sharpen_thin:
            median_diff = median_blur(clip, mode=ConvMode.VERTICAL, func=self._sharpen).std.MakeDiff(clip)
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff, func=self._sharpen)

            resharp = norm_expr(
                [resharp, BlurMatrix.BINOMIAL()(blurred_diff, func=self._sharpen), blurred_diff],
                "y neutral - dup abs z neutral - abs > swap {thin} * x + x ?",
                thin=self.settings.sharpen_thin,
                func=self._sharpen,
            )

        return resharp

    def _back_blend(self, flt: vs.VideoNode, src: vs.VideoNode) -> vs.VideoNode:
        return flt.std.MergeDiff(gauss_blur(src.std.MakeDiff(flt), self.settings._back_blend_sigma))

    def _sharpen_limit(self, clip: vs.VideoNode) -> vs.VideoNode:
        undershoot, overshoot = self.settings.sharpen_limit_clamp

        if self.settings.sharpen_limit_mode.is_spatial:
            if self.settings.sharpen_limit_radius == 1 and undershoot == overshoot == 0:
                clip = repair.Mode.MINMAX_SQUARE1(clip, self.bobbed)
            else:
                inpand = Morpho.minimum(
                    self.bobbed, iterations=self.settings.sharpen_limit_radius, func=self._sharpen_limit
                )
                expand = Morpho.maximum(
                    self.bobbed, iterations=self.settings.sharpen_limit_radius, func=self._sharpen_limit
                )
                clip = norm_expr(
                    [clip, inpand, expand],
                    "x y {undershoot} - z {overshoot} + clamp",
                    undershoot=scale_delta(undershoot, 8, self.clip),
                    overshoot=scale_delta(overshoot, 8, self.clip),
                    func=self._sharpen_limit,
                )
        elif self.settings.sharpen_limit_mode.is_temporal:
            clip = mc_clamp(
                clip,
                self.bobbed,
                self.mv,
                (undershoot, overshoot),
                self._sharpen_limit,
                tr=self.settings.sharpen_limit_radius,
                thscd=self.settings.analyze_thscd,
                **self.settings.sharpen_limit_comp_args,
            )

        return clip

    def _noise_restore(self, clip: vs.VideoNode, restore: float) -> vs.VideoNode:
        if restore:
            clip = norm_expr(
                [clip, self.noise], "x y neutral - {restore} * +", restore=restore, func=self._noise_restore
            )

        return clip


class QTempGaussMC(_QTGMCBuilder):
    """
    Quick Temporal Gaussian Motion Compensated (QTGMC)

    A very high-quality deinterlacer with a range of features for quality and convenience. This includes extensive noise
    processing capabilities, support for repair of progressive material, precision source matching, shutter speed
    simulation, and more.

    Originally based on QTGMC by Vit and TempGaussMC by Didée.

    Usage examples:
        - Basic call with defaults:
        ```python
        deinterlaced = QTempGaussMC().deinterlace(clip)
        ```
        - Using [EEDI3][vsaa.EEDI3] for interpolation:
        ```python
        deinterlaced = QTempGaussMC().basic(bobber=EEDI3()).deinterlace(clip)
        ```
        - Enabling [QTempGaussMC.lossless][vsdeinterlace.QTempGaussMC.lossless] and increasing
            [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final] `tr`:
        ```python
        deinterlaced = QTempGaussMC().lossless(QTempGaussMC.LosslessMode.PRESHARPEN).final(tr=2).deinterlace(clip)
        ```

    Additional usage info: [JET Encoding Guide](https://jaded-encoding-thaumaturgy.github.io/JET-guide/master/filtering/situational/qtgmc/)
    """

    @overload
    def deinterlace(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: Literal[False] = False
    ) -> vs.VideoNode: ...

    @overload
    def deinterlace(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, *, return_graph: Literal[True]
    ) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def deinterlace(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None, return_graph: Literal[True]
    ) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def deinterlace(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: bool = ...
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]: ...

    def deinterlace(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: bool = False
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]:
        """
        Deinterlace interlaced input.

        Interpolates missing fields to reconstruct progressive frames.
        [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur] `fps_divisor` is respected.

        Args:
            clip: Clip to process.
            tff: Field order (top-field-first). If `None`, inferred from the clip. Defaults to None.
            return_graph: Whether to return the [QTGMCGraph][vsdeinterlace.qtgmc.QTGMCGraph] object. It can be used for
                inspecting the output or [QTGMCGraph.mv][vsdeinterlace.qtgmc.QTGMCGraph.mv] can be reused for other
                motion-compensated processing. Defaults to False.

        Returns:
            The deinterlaced clip, or a tuple of (clip, graph) containing the deinterlaced clip and its `QTGMCGraph`
            object if `return_graph` is `True`.
        """

        run = QTGMCGraph.Mode.DEINTERLACE(clip, tff, self, self.deinterlace)

        if return_graph:
            return run.motion_blur, run.freeze()

        return run.motion_blur

    @overload
    def bob(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: Literal[False] = False
    ) -> vs.VideoNode: ...

    @overload
    def bob(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, *, return_graph: Literal[True]
    ) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def bob(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None, return_graph: Literal[True]
    ) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def bob(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: bool = ...
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]: ...

    def bob(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: bool = False
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]:
        """
        Bob interlaced input.

        Interpolates missing fields to reconstruct progressive frames.
        [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur] `fps_divisor` is ignored.

        Args:
            clip: Clip to process.
            tff: Field order (top-field-first). If `None`, inferred from the clip. Defaults to None.
            return_graph: Whether to return the [QTGMCGraph][vsdeinterlace.qtgmc.QTGMCGraph] object. It can be used for
                inspecting the output or [QTGMCGraph.mv][vsdeinterlace.qtgmc.QTGMCGraph.mv] can be reused for other
                motion-compensated processing. Defaults to False.

        Returns:
            The bobbed clip, or a tuple of (clip, graph) containing the bobbed clip and its `QTGMCGraph` object if
            `return_graph` is `True`.
        """

        run = QTGMCGraph.Mode.BOB(clip, tff, self, self.bob)

        if return_graph:
            return run.motion_blur, run.freeze()

        return run.motion_blur

    @overload
    def repair(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: Literal[False] = False
    ) -> vs.VideoNode: ...

    @overload
    def repair(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, *, return_graph: Literal[True]
    ) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def repair(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None, return_graph: Literal[True]
    ) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def repair(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: bool = ...
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]: ...

    def repair(
        self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, return_graph: bool = False
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]:
        """
        Repair badly deinterlaced input.

        Drops half the fields to recreate an interlaced clip using the remaining ones.

        Args:
            clip: Clip to process.
            tff: Field order (top-field-first). If `None`, inferred from the clip. Defaults to None.
            return_graph: Whether to return the [QTGMCGraph][vsdeinterlace.qtgmc.QTGMCGraph] object. It can be used for
                inspecting the output or [QTGMCGraph.mv][vsdeinterlace.qtgmc.QTGMCGraph.mv] can be reused for other
                motion-compensated processing. Defaults to False.

        Returns:
            The repaired clip, or a tuple of (clip, graph) containing the repaired clip and its `QTGMCGraph` object if
            `return_graph` is `True`.
        """

        run = QTGMCGraph.Mode.REPAIR(clip, tff, self, self.repair)

        if return_graph:
            return run.motion_blur, run.freeze()

        return run.motion_blur

    @overload
    def deshimmer(self, clip: vs.VideoNode, return_graph: Literal[False] = False) -> vs.VideoNode: ...

    @overload
    def deshimmer(self, clip: vs.VideoNode, return_graph: Literal[True]) -> tuple[vs.VideoNode, QTGMCGraph]: ...

    @overload
    def deshimmer(
        self, clip: vs.VideoNode, return_graph: bool = ...
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]: ...

    def deshimmer(
        self, clip: vs.VideoNode, return_graph: bool = False
    ) -> vs.VideoNode | tuple[vs.VideoNode, QTGMCGraph]:
        """
        Deshimmer progressive input.

        Removes horizontal shimmering artifacts from progressive sources.

        Args:
            clip: Clip to process.
            return_graph: Whether to return the [QTGMCGraph][vsdeinterlace.qtgmc.QTGMCGraph] object. It can be used for
                inspecting the output or [QTGMCGraph.mv][vsdeinterlace.qtgmc.QTGMCGraph.mv] can be reused for other
                motion-compensated processing. Defaults to False.

        Returns:
            The deshimmered clip, or a tuple of (clip, graph) containing the deshimmered clip and its `QTGMCGraph`
                object if `return_graph` is `True`.
        """

        run = QTGMCGraph.Mode.DESHIMMER(clip, FieldBased.PROGRESSIVE, self, self.deshimmer)

        if return_graph:
            return run.motion_blur, run.freeze()

        return run.motion_blur


def mask_shimmer(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    erosion_distance: int = 4,
    over_dilation: int = 0,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Filters out differences unrelated to bob shimmer by isolating only thin horizontal areas.

    High-level overview:
        - Vertical morphological analysis: Extracts the difference between source and filtered clips, running
            vertical opening and closing operations to collapse thin bob shimmer while leaving large motion
            artifacts intact.

    Args:
        flt: Filtered clip to perform masking on.
        src: Source clip to restore from.
        erosion_distance: Vertical radius for shimmer detection. Larger values capture more spread-out artifacts on soft
            sources. Defaults to 4.
        over_dilation: Extra dilation passes to restore beyond the detected lines. Larger values restore more beyond the
            mask boundary. Defaults to 0.
        func: Function returned for custom error handling. This should only be set by VS package developers. Defaults to
            None.

    Returns:
        Clip with only bob shimmer fixes kept.
    """
    func = func or mask_shimmer

    if not erosion_distance:
        return flt

    ed1 = 1 + erosion_distance // 3
    ed2 = (erosion_distance + 4) // 3
    ed_res = erosion_distance % 3
    od, od_res = divmod(over_dilation, 3)

    ops = ((Morpho.maximum, Morpho.inflate), (Morpho.minimum, Morpho.deflate))

    diff = src.std.MakeDiff(flt)

    processed = list[vs.VideoNode]()
    for (expand_op, inflate_op), (inpand_op, deflate_op) in (ops, ops[::-1]):
        clip = expand_op(diff, iterations=ed1, coords=Coordinates.VERTICAL, func=func)

        if ed_res:
            clip = inflate_op(clip, func=func)
        if ed_res == 2:
            clip = median_blur(clip, func=func)

        clip = inpand_op(clip, iterations=ed2, coords=Coordinates.VERTICAL, func=func)

        if over_dilation:
            clip = inpand_op(clip, iterations=od, func=func)
            clip = deflate_op(clip, iterations=od_res, func=func)

        processed.append(clip)

    return norm_expr([flt, diff, *processed], "x y z neutral min a neutral max clamp neutral - +", func=func)
