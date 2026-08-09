# ruff: noqa: B006, B008, RUF003
from collections.abc import Generator, Mapping
from contextlib import contextmanager, suppress
from copy import deepcopy
from enum import auto
from math import comb
from typing import Any, Literal, Protocol, Self, TypedDict

from jetpytools import CustomIntEnum, CustomValueError, FuncExcept, fallback, normalize_seq

from vsaa import NNEDI3
from vsdeband import Grainer
from vsdenoise import (
    DFTTest,
    MaskMode,
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
    core,
    sc_detect,
    scale_delta,
    vs,
)

from .utils import reinterlace, reweave

__all__ = ["QTempGaussMC"]


class _DenoiseFuncTr(Protocol):
    def __call__(self, clip: vs.VideoNode, /, *, tr: int) -> vs.VideoNode: ...


class QTGMCArgs:
    """Namespace containing helper TypedDict definitions for various argument groups."""

    class PrefilterToFullRange(TypedDict, total=False):
        """Arguments accepted by [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range]."""

        slope: float
        smooth: float

    class MaskShimmer(TypedDict, total=False):
        """
        Arguments accepted by the internal `_mask_shimmer` method through
        [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter],
        [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] and
        [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final].

        Removes areas of difference between a temporally blurred clip and a reference clip that are not due to
        bob shimmer by only allowing thin horizontal areas of difference.

        High-level overview:
            - Vertical morphological analysis: Extracts the difference between source and filtered clips, running
                vertical opening and closing operations to collapse thin bob shimmer while leaving large motion
                artifacts intact.
            - Safety margin adjustment: Applies extra dilation passes to fine-tune detection boundaries on soft sources,
                ensuring proper mask coverage on blurry edges.
        """

        erosion_distance: int
        """
        Vertical radius for shimmer detection.

        Larger values capture more spread-out shimmer artifacts on soft sources.
        """

        over_dilation: int
        """
        Extra dilation passes for safety margins.

        Larger values shrink the mask boundary to prevent artifacts on blurry edges. Defaults to 0.
        """

    class Compensate(TypedDict, total=False):
        """Arguments accepted by [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate]."""

        thsad: int | None
        time: float | None

    class Degrain(TypedDict, total=False):
        """
        Arguments accepted by the internal `_binomial_degrain` method, calling
        [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain] through
        [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] and
        [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match], or directly through
        [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final].
        """

        limit: float | tuple[float, float] | None
        planes: Planes

    class Mask(TypedDict, total=False):
        """Arguments accepted by [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask]."""

        delta: int
        ml: float | None
        gamma: float | None
        time: float | None
        scval: float | None

    class Blur(TypedDict, total=False):
        """Arguments accepted by [MVTools.flow_blur][vsdenoise.mvtools.mvtools.MVTools.flow_blur]."""

        prec: int | None


class QTempGaussMC(VSObject):
    """
    Quick Temporal Gaussian Motion Compensated (QTGMC)

    A very high-quality deinterlacer with a range of features for both quality and convenience. These include extensive
    noise processing capabilities, support for repair of progressive material, precision source matching, shutter speed
    simulation, and more.

    Originally based on TempGaussMC by Didée.

    Basic usage: [JET guide](https://jaded-encoding-thaumaturgy.github.io/JET-guide/master/filtering/situational/qtgmc/)

    Alternate documentation reference: [AviSynth documentation](http://avisynth.nl/index.php/QTGMC)
    """

    mv: MVTools
    """[MVTools][vsdenoise.mvtools.mvtools.MVTools] instance used during processing."""

    clip: vs.VideoNode
    """Clip to process."""

    draft: vs.VideoNode
    """
    Draft processed clip, used as a base for [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter] and
    [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise].
    """

    input: vs.VideoNode
    """
    Prepared input clip for high-quality interpolation. Used as a base for the internal `_interpolate` method.
    """

    bobbed: vs.VideoNode
    """
    High-quality bobbed clip, acting as a spatial interpolation base for
    [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] and
    [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match].
    """

    noise: vs.VideoNode
    """Noise extracted by [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise]."""

    prefilter_output: vs.VideoNode
    """Output of [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter]."""

    denoise_output: vs.VideoNode
    """Output of [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise]."""

    basic_output: vs.VideoNode
    """Output of [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic]."""

    final_output: vs.VideoNode
    """Output of [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final]."""

    motion_blur_output: vs.VideoNode
    """Output of [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur]."""

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class SearchPostProcess(CustomIntEnum):
        GAUSSBLUR = 0
        GAUSSBLUR_EDGESOFTEN = 1

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class NoiseProcessMode(CustomIntEnum):
        IDENTIFY = 0
        DENOISE = 1

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class SharpenMode(CustomIntEnum):
        UNSHARP = 0
        UNSHARP_MINMAX = 1

    @deprecated("This enum is deprecated and will be removed in a future version.", category=DeprecationWarning)
    class SourceMatchMode(CustomIntEnum):
        NONE = 0
        BASIC = 1
        REFINED = 2
        TWICE_REFINED = 3

    class NoiseDeintMode(CustomIntEnum):
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

    class SharpenLimitMode(CustomIntEnum):
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

    class BackBlendMode(CustomIntEnum):
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
        Back-blending both before and after [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit].

        Provides a balanced middle ground between `PRELIMIT` sharpening strength and `POSTLIMIT` dampening.

        Note:
            Identical to `PRELIMIT` when using `SharpenLimitMode.NONE`, `SharpenLimitMode.SPATIAL_POSTSMOOTH` or
            `SharpenLimitMode.TEMPORAL_POSTSMOOTH`.
        """

    class LosslessMode(CustomIntEnum):
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

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            **kwargs: Additional arguments to be passed to the parameter category methods. Use the method's name as a
                prefix to pass an argument to the respective method.
        """

        # Set default parameters for all the categories in this exact order
        self._settings_methods = (
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

        for method in self._settings_methods:
            prefix = f"{method.__name__}_"

            method(**{k.removeprefix(prefix): kwargs.pop(k) for k in tuple(kwargs) if k.startswith(prefix)})

        if kwargs:
            raise CustomValueError("Unknown arguments were passed.", self.__class__, kwargs)

    def prefilter(
        self,
        *,
        tr: int = 2,
        sc_threshold: float = 0.1,
        postprocess: SearchPostProcess | None = None,
        strength: tuple[float, float] = (1.9, 0.9),
        limit: tuple[float, float, float] = (3, 7, 2),
        bias: float = 0.51,
        range_expansion_args: QTGMCArgs.PrefilterToFullRange | None = None,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = {"erosion_distance": 4},
    ) -> Self:
        """
        Configures parameters for the prefilter stage.

        Prepares a suitable search clip to be provided for motion analysis purposes.

        High-level overview:
            - [[QTempGaussMC.deinterlace][vsdeinterlace.QTempGaussMC.deinterlace]] Draft bobbed clip generation:
                Begins with simple spatial interpolation to produce
                [QTempGaussMC.draft][vsdeinterlace.QTempGaussMC.draft], which inherently contains severe temporal
                instability known as bob shimmer.
            - [[QTempGaussMC.repair][vsdeinterlace.QTempGaussMC.repair]] Vertical spatial pre-filtering: Applies a
                vertical binomial blur to filter out residual vertical artifacts.
            - Temporal binomial blurring: Applies a temporal binomial blur to smooth
                [QTempGaussMC.draft][vsdeinterlace.QTempGaussMC.draft], removing the shimmer, which prevents
                [MVTools][vsdenoise.mvtools.mvtools.MVTools] from falsely latching onto the shimmer as motion (though
                this uncompensated blur introduces ghosting).
            - Shimmer masking: Uses a specialized masking process to eliminate the introduced ghosting while retaining
                the shimmer removal.
            - Gaussian blurring post-processing: Applies Gaussian blurring to lower high SAD values caused by sharp
                edges, ensuring edges are properly processed rather than skipped.
            - Edge detail restoration: Conservatively restores essential edge detail from
                [QTempGaussMC.draft][vsdeinterlace.QTempGaussMC.draft] back into the blurred clip via a limiting process
                so [MVTools][vsdenoise.mvtools.mvtools.MVTools] retains the ability to track motion effectively.
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
            limit: Tuple containing the 3-step limiting (8-bit) thresholds for the Gaussian blur post-processing:

                   - First value: Maximum allowed delta between the temporally blurred clip and
                   [QTempGaussMC.draft][vsdeinterlace.QTempGaussMC.draft]. Smaller values clamp
                   [QTempGaussMC.draft][vsdeinterlace.QTempGaussMC.draft] closer to the temporally blurred clip.
                   - Second value: Tolerance threshold for the allowed difference between the clamped clip and the
                    Gaussian-blurred clip before hard clipping triggers. Larger values widen the allowed range for
                    smooth blending before hard clipping is enforced.
                   - Third value: Offset applied to the Gaussian-blurred clip when the second threshold is exceeded.
                    Larger values allow a bigger delta from the Gaussian-blurred clip when clipped.

                Defaults to (3, 7, 2).
            bias: Weight used when blending the Gaussian-blurred clip back with the limited clip. Higher
                values give more weight to the Gaussian-blurred clip. Defaults to 0.51.
            range_expansion_args: Additional arguments passed to
                [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range]. Defaults to None.
            mask_shimmer_args: Additional arguments passed to the internal `_mask_shimmer` call. Defaults to
                {"erosion_distance": 4}.
        """

        self.prefilter_tr = tr
        self.prefilter_sc_threshold = sc_threshold
        self.prefilter_postprocess = postprocess
        self.prefilter_strength = strength
        self.prefilter_limit = limit
        self.prefilter_bias = bias
        self.prefilter_range_expansion_args = fallback(range_expansion_args, QTGMCArgs.PrefilterToFullRange())
        self.prefilter_mask_shimmer_args = fallback(mask_shimmer_args, QTGMCArgs.MaskShimmer())

        if postprocess is not None and not postprocess.value:  # TODO: remove
            self.prefilter_limit = (0, 0, 0)

        return self

    def analyze(
        self,
        *,
        force_tr: int = 0,
        preset: Mapping[str, Any] = MVToolsPreset.HQ_SAD,
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
            force_tr: Always analyze motion to at least this value, even if otherwise unnecessary. Useful if you want to
                reuse the generated motion vectors for other tasks. Defaults to 0.
            preset: [MVTools][vsdenoise.mvtools.mvtools.MVTools] preset defining base values for
                [MVTools][vsdenoise.mvtools.mvtools.MVTools]. Defaults to MVToolsPreset.HQ_SAD.
            blksize: Motion analysis block size. Larger blocks are faster and less sensitive to noise, but less
                accurate. Defaults to 16.
            overlap: The block size divisor for block size overlap. Larger overlap reduces blocking artifacts of
                [MVTools][vsdenoise.mvtools.mvtools.MVTools] processes. Defaults to 2.
            refine: Number of iterations to recalculate motion vectors with halved block size. Improves motion vector
                precision without reducing denoising effectiveness. Defaults to 1.
            thsad_recalc: Only poor-quality new vectors with a SAD above this value will be re-estimated by motion
                search. Only active when refine is used. Defaults to
                [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] `thsad` / 2.
            thscd: Scene-change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.

                Defaults to (180, 38.5).
        """

        self.analyze_force_tr = force_tr
        self.analyze_preset = preset
        self.analyze_blksize = blksize
        self.analyze_overlap = overlap
        self.analyze_refine = refine
        self.analyze_thsad_recalc = thsad_recalc
        self.analyze_thscd = thscd

        return self

    def denoise(
        self,
        *,
        tr: int = 1,
        func: DFTTest | _DenoiseFuncTr = DFTTest(sigma=8),
        mode: NoiseProcessMode | None = None,
        deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
        full_denoise: bool = False,
        mc_denoise: bool = True,
        stabilize: float | Literal[False] = 0.4,
        func_comp_args: QTGMCArgs.Compensate | None = None,
        stabilize_comp_args: QTGMCArgs.Compensate | None = None,
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
            - [[QTempGaussMC.deinterlace][vsdeinterlace.QTempGaussMC.deinterlace]] Interlaced noise processing: Because
                the extracted noise is inherently interlaced and standard processing would eliminate it, three
                alternative methods ([QTempGaussMC.NoiseDeintMode][vsdeinterlace.QTempGaussMC.NoiseDeintMode]) are
                available to process it separately.
            - Noise stabilization: The extracted noise can optionally be stabilized at the end of processing using a
                blend of the maximum variance determined through motion compensation and the average of that maximum
                variance and the extracted noise.

        Args:
            tr: Temporal radius of the denoising function and its motion compensation. Larger values remove/separate
                more noise. Defaults to 1.
            func: Denoising function to use. Defaults to DFTTest(sigma=8).
            deint: Specifies how to 'deinterlace' noise taken from an interlaced source before restoration. Defaults to
                NoiseDeintMode.GENERATE.
            full_denoise: Whether the denoised output will be directly used in all subsequent processing. If `False`,
                the denoising is only for noise extraction. Defaults to False.
            mc_denoise: Whether to motion-compensate the denoiser being used. Provides more accurate denoising/noise
                extraction when using a non-motion-compensated temporal denoiser. Defaults to True.
            stabilize: Weight used when blending max noise variance with averaged noise. Higher values give more
                weight to the averaged noise. `False` disables stabilization. Defaults to 0.4.
            func_comp_args: Additional arguments passed to
                [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for denoising. Defaults to None.
            stabilize_comp_args: Additional arguments passed to
                [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for noise stabilization. Defaults to
                None.
        """

        self.denoise_tr = tr
        self.denoise_func = func.denoise if isinstance(func, DFTTest) else func
        self.denoise_deint = deint
        self.denoise_full_denoise = full_denoise
        self.denoise_mc_denoise = mc_denoise
        self.denoise_stabilize = stabilize
        self.denoise_func_comp_args = fallback(func_comp_args, QTGMCArgs.Compensate())
        self.denoise_stabilize_comp_args = fallback(stabilize_comp_args, QTGMCArgs.Compensate())

        if mode is not None:  # TODO: remove
            self.denoise_full_denoise = bool(mode)

        return self

    def basic(
        self,
        *,
        tr: int = 2,
        thsad: int | tuple[int, int] = 640,
        bobber: BobberLike = NNEDI3(nsize=1),
        noise_restore: float = 0,
        degrain_args: QTGMCArgs.Degrain | None = None,
        mask_args: QTGMCArgs.Mask | None = {"ml": 10},
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = {"erosion_distance": 0},
    ) -> Self:
        """
        Configures parameters for the basic stage.

        Creates the basic output of the core algorithm. Intended to eliminate bob shimmer.

        High-level overview:
            - High-quality bobbed clip generation: Begins with high-quality spatial interpolation to produce
                [QTempGaussMC.bobbed][vsdeinterlace.QTempGaussMC.bobbed], which inherently contains severe temporal
                instability known as bob shimmer.
            - [[QTempGaussMC.repair][vsdeinterlace.QTempGaussMC.repair]] Motion SAD masking: Generates a motion-vector
                SAD mask to blend [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise] output over
                [QTempGaussMC.bobbed][vsdeinterlace.QTempGaussMC.bobbed], protecting static/low-motion detail.
            - Motion-compensated temporal binomial smoothing: Applies a motion-compensated temporal binomial blur to
                smooth [QTempGaussMC.bobbed][vsdeinterlace.QTempGaussMC.bobbed], removing the shimmer while avoiding
                ghosting artifacts.
            - Shimmer masking: Uses a specialized masking process to eliminate the introduced blurring while retaining
                the shimmer removal.
            - Additional refinements: Passes the temporally smoothed clip through optional fine-tuning processes:
                - [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match]
                - [QTempGaussMC.lossless][vsdeinterlace.QTempGaussMC.lossless]
                - [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen]
                - [QTempGaussMC.back_blend][vsdeinterlace.QTempGaussMC.back_blend]
                - [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit]

            - Noise restoration: Optionally restores noise previously extracted by
                [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise] at the end of the pipeline.

        Args:
            tr: Temporal radius of the motion-compensated binomial blur. Larger values reduce more shimmer but can
                introduce blurring and ghosting. Defaults to 2.
            thsad: SAD threshold of the motion-compensated binomial blur. Larger values reduce more shimmer but can
                introduce blurring and ghosting. Defaults to 640.
            bobber: Bobber to use for spatial interpolation. Defaults to NNEDI3(nsize=1).
            noise_restore: Amount of noise to restore after this stage. Used to retain stable noise. Defaults to 0.
            degrain_args: Additional arguments passed to the internal `_binomial_degrain` call. Defaults to None.
            mask_args: Additional arguments passed to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask]. Only used
                for [QTempGaussMC.repair][vsdeinterlace.QTempGaussMC.repair]. Defaults to {"ml": 10}.
            mask_shimmer_args: Additional arguments passed to the internal `_mask_shimmer` call. Defaults to
                {"erosion_distance": 0}.
        """

        self.basic_tr = tr
        self.basic_thsad = thsad
        self.basic_bobber = (
            deepcopy(bobber) if isinstance(bobber, Bobber) else Bobber.ensure_obj(bobber, self.__class__)
        )
        self.basic_noise_restore = noise_restore
        self.basic_degrain_args = fallback(degrain_args, QTGMCArgs.Degrain())
        self.basic_mask_args = fallback(mask_args, QTGMCArgs.Mask())
        self.basic_mask_shimmer_args = fallback(mask_shimmer_args, QTGMCArgs.MaskShimmer())

        return self

    def source_match(
        self,
        *,
        tr: int = 1,
        bobber: BobberLike | None = None,
        iterations: Literal[0, 1, 2, 3] = 0,
        mode: SourceMatchMode | None = None,
        similarity: float = 0.5,
        enhance: float = 0.5,
        degrain_args: QTGMCArgs.Degrain | None = None,
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
                error-adjustment pass if `iterations` > 2), and merges it back to restore fine detail missed during the
                initial pass.

        Note:
            - When source matching is used:
                - [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen] is disabled by default, as source matching
                    acts as an alternative form of sharpness restoration.
                - [QTempGaussMC.sharpen_limit][vsdeinterlace.QTempGaussMC.sharpen_limit] is disabled by default, as it
                    reduces the accuracy of source matching.

        Args:
            tr: Temporal radius of the refinement motion-compensated binomial blur. Larger values reduce more shimmer
                but can introduce blurring and ghosting. Only used for `iterations` > 1. Defaults to 1.
            bobber: Bobber to use for refined spatial interpolation. Only used for `iterations` > 1. Defaults to
                [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic] `bobber`.
            iterations: Number of source match iterations to perform. Higher values are slower and more accurate. Using
                2 or 3 iterations restores almost exact source detail but is sensitive to noise and introduces
                occasional aliasing (to a lesser extent for 3). Defaults to 0.
            similarity: Temporal similarity of the error from frame to frame. Lower values make the result sharper.
                Defaults to 0.5.
            enhance: Enhances detail found by `iterations` > 1. Higher values exaggerate detail more. Defaults to 0.5.
            degrain_args: Additional arguments passed to the internal `_binomial_degrain` call. Defaults to None.
        """

        self.source_match_tr = tr
        self.source_match_bobber = bobber
        self.source_match_iterations = iterations
        self.source_match_similarity = similarity
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

        Restoring original fields significantly improves fidelity, but may introduce minor shimmering, combing,
        or noise.

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
        mode: SharpenMode | None = None,
        strength: float | None = None,
        offset: float | tuple[float, float] | Literal[False] = 1,
        thin: float = 0,
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
                [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match] `iterations` is 0, and 0 otherwise.
            offset: Offsets the blur source to the vertical min/max average ±this value. Smaller values result in more
                vertical sharpening. Passing a tuple of values results in asymmetric offsetting. `False` disables
                range limiting. Defaults to 1.
            thin: How much to thin down horizontal edges. Higher values result in more thinning. Defaults to 0.
        """

        self.sharpen_strength = strength
        self.sharpen_offset = offset is not False and normalize_seq(offset, 2)
        self.sharpen_thin = thin

        if mode is not None and not mode.value:  # TODO: remove
            self.sharpen_offset = False

        return self

    @property
    def sharpen_strength(self) -> float:
        return fallback(self._sharpen_strength, 0 if self.source_match_iterations else 1)

    @sharpen_strength.setter
    def sharpen_strength(self, value: float | None) -> None:
        self._sharpen_strength = value

    def back_blend(self, *, mode: BackBlendMode = BackBlendMode.BOTH, sigma: float = 1.4) -> Self:
        """
        Configures parameters for back-blending.

        Improves [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen] fidelity by dampening low-frequency
        enhancement caused by unsharpening.

        High-level overview:
            - Low-frequency back-blending: Gaussian-blurs the pre- and post-sharpening difference to isolate broad
                low-frequency shifts, then merges that blurred difference back onto the source to preserve only
                high-frequency edge sharpening.

        Args:
            mode: When to back-blend the (blurred) difference between the pre- and post-sharpened clips. Defaults to
                BackBlendMode.BOTH.
            sigma: Gaussian blur sigma applied to the pre- and post-sharpening difference. Lower values dampen
                sharpening more aggressively. Defaults to 1.4.
        """

        self.backblend_mode = mode
        self.backblend_sigma = sigma

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
                    bounds of [QTempGaussMC.bobbed][vsdeinterlace.QTempGaussMC.bobbed].
                - Motion-compensated temporal limiting: Clamps the sharpened clip using motion-compensated reference
                    frames from [QTempGaussMC.bobbed][vsdeinterlace.QTempGaussMC.bobbed].

        Args:
            mode: How and when to apply limiting to [QTempGaussMC.sharpen][vsdeinterlace.QTempGaussMC.sharpen]. Defaults
                to SharpenLimitMode.TEMPORAL_PRESMOOTH when
                [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match] `iterations` is 0 and
                SharpenLimitMode.NONE otherwise.
            radius: Radius of the sharpness limiting. Larger values allow more sharpening. Defaults to 1.
            clamp: How much undershoot/overshoot to allow. Larger values result in less limiting. Passing a tuple of
                values allows for asymmetric limiting. Defaults to 0.
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
            self.SharpenLimitMode.NONE if self.source_match_iterations else self.SharpenLimitMode.TEMPORAL_PRESMOOTH,
        )

    @sharpen_limit_mode.setter
    def sharpen_limit_mode(self, value: SharpenLimitMode | None) -> None:
        self._sharpen_limit_mode = value

    def final(
        self,
        *,
        tr: int = 1,
        thsad: int | tuple[int, int] = 256,
        noise_restore: float = 0,
        degrain_args: QTGMCArgs.Degrain | None = None,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = {"erosion_distance": 4},
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
                [QTempGaussMC.denoise][vsdeinterlace.QTempGaussMC.denoise] at the end of the pipeline.

        Args:
            tr: Temporal radius of the motion-compensated linear blur. Larger values clean more artifacts but can
                introduce blurring and ghosting. Defaults to 1.
            thsad: SAD threshold of the motion-compensated linear blur. Larger values clean more artifacts but can
                introduce blurring and ghosting. Defaults to 256.
            noise_restore: Amount of noise to restore after this stage. Used to retain any noise. Defaults to 0.
            degrain_args: Additional arguments passed to [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain].
                Defaults to None.
            mask_shimmer_args: Additional arguments passed to the internal `_mask_shimmer` call. Defaults to
                {"erosion_distance": 4}.
        """

        self.final_tr = tr
        self.final_thsad = thsad
        self.final_noise_restore = noise_restore
        self.final_degrain_args = fallback(degrain_args, QTGMCArgs.Degrain())
        self.final_mask_shimmer_args = fallback(mask_shimmer_args, QTGMCArgs.MaskShimmer())

        return self

    def motion_blur(
        self,
        *,
        shutter_angle: tuple[float, float] = (180, 180),
        fps_divisor: int = 1,
        blur_args: QTGMCArgs.Blur | None = None,
        mask_args: QTGMCArgs.Mask | None = {"ml": 4},
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
                match. Defaults to (180, 180).
            fps_divisor: Factor by which to smoothly reduce frame rate. Defaults to 1.
            blur_args: Additional arguments passed to [MVTools.flow_blur][vsdenoise.mvtools.mvtools.MVTools.flow_blur].
                Defaults to None.
            mask_args: Additional arguments passed to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask]. Defaults
                to {"ml": 4}.
        """

        self.motion_blur_shutter_angle = shutter_angle
        self.motion_blur_fps_divisor = fps_divisor
        self.motion_blur_blur_args = fallback(blur_args, QTGMCArgs.Blur())
        self.motion_blur_mask_args = fallback(mask_args, QTGMCArgs.Mask())

        return self

    def _mask_shimmer(
        self,
        flt: vs.VideoNode,
        src: vs.VideoNode,
        erosion_distance: int,
        over_dilation: int = 0,
    ) -> vs.VideoNode:
        if not erosion_distance:
            return flt

        ed_iter1 = 1 + erosion_distance // 3
        ed_iter2 = 1 + (erosion_distance + 1) // 3
        ed_res = erosion_distance % 3
        od_iter1, od_iter2 = divmod(over_dilation, 3)

        diff = src.std.MakeDiff(flt)

        processed = list[vs.VideoNode]()
        for grow_op, shrink_op, inflate_op, deflate_op in [
            (Morpho.maximum, Morpho.minimum, Morpho.inflate, Morpho.deflate),
            (Morpho.minimum, Morpho.maximum, Morpho.deflate, Morpho.inflate),
        ]:
            clip = grow_op(diff, iterations=ed_iter1, coords=Coordinates.VERTICAL, func=self._mask_shimmer)

            if ed_res:
                clip = inflate_op(clip, func=self._mask_shimmer)
                if ed_res == 2:
                    clip = median_blur(clip, func=self._mask_shimmer)

            clip = shrink_op(clip, iterations=ed_iter2, coords=Coordinates.VERTICAL, func=self._mask_shimmer)

            if over_dilation:
                clip = shrink_op(clip, iterations=od_iter1, func=self._mask_shimmer)
                clip = deflate_op(clip, iterations=od_iter2, func=self._mask_shimmer)

            processed.append(clip)

        return norm_expr(
            [flt, diff, *processed], "x y z neutral min a neutral max clamp neutral - +", func=self._mask_shimmer
        )

    def _interpolate(self, clip: vs.VideoNode, bobber: Bobber) -> vs.VideoNode:
        if self.tff.is_inter:
            clip = bobber.bob(clip, tff=self.tff)

        return clip

    def _binomial_degrain(self, clip: vs.VideoNode, tr: int, **degrain_args: Any) -> vs.VideoNode:
        if not tr:
            return clip

        return self.mv.degrain(
            clip,
            tr=tr,
            thsad=self.basic_thsad,
            thscd=self.analyze_thscd,
            weights=BlurMatrix.BINOMIAL(radius=tr),
            **degrain_args,
        )

    def _apply_prefilter(self) -> None:
        self.draft = Catrom().bob(self.clip, tff=self.tff) if self.tff.is_inter and not self.is_repair else self.clip

        if self.is_repair:
            search = BlurMatrix.BINOMIAL()(self.draft, mode=ConvMode.VERTICAL, func=self._apply_prefilter)
        else:
            search = self.draft

        if self.prefilter_tr:
            smoothed = BlurMatrix.BINOMIAL(self.prefilter_tr, mode=ConvMode.TEMPORAL)(
                sc_detect(search, self.prefilter_sc_threshold), scenechange=True, func=self._apply_prefilter
            )
            smoothed = self._mask_shimmer(smoothed, search, **self.prefilter_mask_shimmer_args)
        else:
            smoothed = search

        gauss_sigma, blend_weight = self.prefilter_strength
        lim1, lim2, lim3 = [scale_delta(thr, 8, self.clip) for thr in self.prefilter_limit]

        # TODO: Figure out early exits
        blurred = gauss_blur(smoothed, gauss_sigma) if gauss_sigma and blend_weight else smoothed
        blurred = norm_expr(
            [blurred, smoothed, search],
            "y x y - {weight} * + BLUR! z y {lim1} - y {lim1} + clamp TWEAK! "
            "BLUR@ {lim2} + TWEAK@ < BLUR@ {lim3} + BLUR@ {lim2} - TWEAK@ > BLUR@ {lim3} - "
            "TWEAK@ BLUR@ TWEAK@ - {bias} * + ? ?",
            weight=blend_weight,
            lim1=lim1,
            lim2=lim2,
            lim3=lim3,
            bias=self.prefilter_bias,
            func=self._apply_prefilter,
        )

        self.prefilter_output = prefilter_to_full_range(
            blurred, func=self._apply_prefilter, **self.prefilter_range_expansion_args
        )

    def _apply_analyze(self) -> None:
        angle_in, angle_out = self.motion_blur_shutter_angle

        tr = max(
            # Unconditional radii
            self.analyze_force_tr,
            self.basic_tr,
            self.source_match_tr,
            self.final_tr,
            # Conditional radii
            self.denoise_tr if self.denoise_mc_denoise else 0,
            self.sharpen_limit_radius
            if self.sharpen_limit_mode
            in {self.SharpenLimitMode.TEMPORAL_PRESMOOTH, self.SharpenLimitMode.TEMPORAL_POSTSMOOTH}
            and (self.sharpen_strength or self.sharpen_thin)
            else 0,
            # Feature flags
            int(self.denoise_stabilize is not False and (self.basic_noise_restore or self.final_noise_restore)),
            int(bool(self.is_repair and self.basic_mask_args.get("ml"))),
            int(angle_out * self.motion_blur_fps_divisor != angle_in),
        )

        blksize = self.analyze_blksize
        thsad_recalc = fallback(
            self.analyze_thsad_recalc,
            round((self.basic_thsad[0] if isinstance(self.basic_thsad, tuple) else self.basic_thsad) / 2),
        )

        self.mv = MVTools(self.draft, **{**self.analyze_preset, "search_clip": self.prefilter_output})
        self.mv.analyze(tr=tr, blksize=blksize, overlap_div=self.analyze_overlap)

        for _ in range(self.analyze_refine):
            blksize = refine_blksize(blksize)
            self.mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap_div=self.analyze_overlap)

    def _apply_denoise(self) -> None:
        self.denoise_output = self.clip

        no_restore = self.basic_noise_restore == self.final_noise_restore == 0

        if not self.denoise_full_denoise and no_restore:
            return

        if self.denoise_mc_denoise:
            denoised = self.mv.compensate(
                tr=self.denoise_tr,
                thscd=self.analyze_thscd,
                temporal_func=lambda clip: self.denoise_func(clip, tr=self.denoise_tr),
                **self.denoise_func_comp_args,
            )
        else:
            denoised = self.denoise_func(self.draft, tr=self.denoise_tr)

        if self.tff.is_inter and not self.is_repair:
            denoised = reinterlace(denoised, self.tff, self._apply_denoise)

        if self.denoise_full_denoise:
            self.denoise_output = denoised

        self.noise = self.clip.std.MakeDiff(denoised)

        if no_restore:
            return

        if self.tff.is_inter and not self.is_repair:
            match self.denoise_deint:
                case self.NoiseDeintMode.WEAVE:
                    new_noise = self.noise.std.SeparateFields(self.tff.is_tff).std.DoubleWeave(self.tff.is_tff)
                case self.NoiseDeintMode.BOB:
                    new_noise = Catrom().bob(self.noise, tff=self.tff)
                case self.NoiseDeintMode.GENERATE:
                    noise_source = self.noise.std.SeparateFields(self.tff.is_tff)

                    noise_max = Morpho.expand(noise_source, sw=2, sh=1, func=self._apply_denoise)
                    noise_min = Morpho.inpand(noise_source, sw=2, sh=1, func=self._apply_denoise)

                    gen_noise = Grainer.GAUSS(
                        noise_source,
                        ((0.5 * 255) / 3) ** 2,  # 3σ rule
                        protect_edges=False,
                        protect_neutral_chroma=False,
                        neutral_out=True,
                    )
                    gen_noise = norm_expr(
                        [noise_max, noise_min, gen_noise],
                        "y x y - z neutral - range_size / 0.5 + * +",
                        func=self._apply_denoise,
                    )
                    new_noise = reweave(noise_source, gen_noise, self.tff.field, self._apply_denoise)

            self.noise = FieldBased.PROGRESSIVE.apply(new_noise)

        if self.denoise_stabilize is not False:
            noise_comp, _ = self.mv.compensate(
                self.noise,
                direction=MVDirection.BACKWARD,
                tr=1,
                thscd=self.analyze_thscd,
                interleave=False,
                **self.denoise_stabilize_comp_args,
            )

            self.noise = norm_expr(
                [self.noise, *noise_comp],
                "x neutral - abs y neutral - abs > x y ? {weight2} * x y + {weight1} * +",
                weight1=self.denoise_stabilize / 2,
                weight2=1 - self.denoise_stabilize,
                func=self._apply_denoise,
            )

    def _apply_basic(self) -> None:
        if self.is_repair:
            self.input = reinterlace(self.denoise_output, self.tff, self._interpolate)
        else:
            self.input = self.denoise_output

        self.bobbed = self._interpolate(self.input, self.basic_bobber)

        if self.is_repair and self.basic_mask_args.get("ml", 0):
            mask = self.mv.mask(
                direction=MVDirection.BACKWARD,
                kind=MaskMode.SAD,
                thscd=self.analyze_thscd,
                **self.basic_mask_args,
            )
            self.bobbed = self.denoise_output.std.MaskedMerge(self.bobbed, mask)

        smoothed = self._binomial_degrain(self.bobbed, self.basic_tr, **self.basic_degrain_args)

        if self.basic_tr:
            smoothed = self._mask_shimmer(smoothed, self.bobbed, **self.basic_mask_shimmer_args)

            if self.source_match_iterations:
                smoothed = self._apply_source_match(smoothed)

        if self.lossless_mode == self.LosslessMode.PRESHARPEN:
            smoothed = self._apply_lossless(smoothed)

        resharp = self._apply_sharpen(smoothed)

        if self.sharpen_limit_mode in {
            self.SharpenLimitMode.SPATIAL_PRESMOOTH,
            self.SharpenLimitMode.TEMPORAL_PRESMOOTH,
        }:
            if self.backblend_mode in {self.BackBlendMode.PRELIMIT, self.BackBlendMode.BOTH}:
                resharp = self._apply_back_blend(resharp, smoothed)

            resharp = self._apply_sharpen_limit(resharp)

            if self.backblend_mode in {self.BackBlendMode.POSTLIMIT, self.BackBlendMode.BOTH}:
                resharp = self._apply_back_blend(resharp, smoothed)
        elif self.backblend_mode != self.BackBlendMode.NONE:
            resharp = self._apply_back_blend(resharp, smoothed)

        self.basic_output = self._apply_noise_restore(resharp, self.basic_noise_restore)

    def _apply_source_match(self, clip: vs.VideoNode) -> vs.VideoNode:
        def _error_adjustment(ref: vs.VideoNode, clip: vs.VideoNode, tr: int) -> vs.VideoNode:
            if not tr:
                return ref

            tr_f = 2 * tr - 1
            tr_s = 2**tr_f
            binomial_coeff = comb(tr_f, tr)
            error_adj = tr_s / (binomial_coeff + self.source_match_similarity * (tr_s - binomial_coeff))

            return norm_expr([ref, clip], "x x y - {error_adj} * +", error_adj=error_adj, func=_error_adjustment)

        if self.tff.is_inter:
            clip = reinterlace(clip, self.tff, self._apply_source_match)

        adjusted = _error_adjustment(self.input, clip, self.basic_tr)
        new_bobbed = self._interpolate(adjusted, self.basic_bobber)
        matched = self._binomial_degrain(new_bobbed, self.basic_tr, **self.basic_degrain_args)

        if self.source_match_iterations > 1:
            if self.source_match_enhance:
                matched = unsharpen(
                    matched, self.source_match_enhance, BlurMatrix.BINOMIAL(), func=self._apply_source_match
                )

            clip = reinterlace(matched, self.tff, self._apply_source_match) if self.tff.is_inter else matched

            diff = self.input.std.MakeDiff(clip)
            refine_bobbed = self._interpolate(diff, self.source_match_bobber)
            refine_matched = self._binomial_degrain(
                refine_bobbed, self.source_match_tr, **self.source_match_degrain_args
            )

            if self.source_match_iterations > 2:
                refine_adjusted = _error_adjustment(refine_bobbed, refine_matched, self.source_match_tr)
                refine_matched = self._binomial_degrain(
                    refine_adjusted, self.source_match_tr, **self.source_match_degrain_args
                )

            return matched.std.MergeDiff(refine_matched)

        return matched

    def _apply_lossless(self, clip: vs.VideoNode) -> vs.VideoNode:
        if not self.tff.is_inter or clip is self.bobbed:
            return clip

        fields_src = self.denoise_output.std.SeparateFields(self.tff.is_tff)
        if self.is_repair:
            fields_src = core.std.SelectEvery(fields_src, 4, (0, 3))
        fields_flt = clip.std.SeparateFields(self.tff.is_tff).std.SelectEvery(4, (1, 2))

        woven = reweave(fields_src, fields_flt, self.tff.field, self._apply_lossless)

        if self.lossless_anti_comb:
            median_diff = median_blur(woven, mode=ConvMode.VERTICAL, func=self._apply_lossless).std.MakeDiff(woven)
            fields_diff = median_diff.std.SeparateFields(self.tff.is_tff).std.SelectEvery(4, (1, 2))

            cleaned_diff = norm_expr(
                [median_blur(fields_diff, mode=ConvMode.VERTICAL, func=self._apply_lossless), fields_diff],
                "x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?",
                func=self._apply_lossless,
            )
            cleaned_diff = repair.Mode.MINMAX_SQUARE1(cleaned_diff, remove_grain.Mode.MINMAX_AROUND2(cleaned_diff))
            woven = reweave(fields_src, fields_flt.std.MergeDiff(cleaned_diff), self.tff.field, self._apply_lossless)

        return FieldBased.PROGRESSIVE.apply(woven)

    def _apply_sharpen(self, clip: vs.VideoNode) -> vs.VideoNode:
        resharp = clip

        if self.sharpen_strength:
            if self.sharpen_offset is not False:
                dark_offset, bright_offset = self.sharpen_offset

                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL, func=self._apply_sharpen)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL, func=self._apply_sharpen)

                resharp = norm_expr(
                    [clip, source_min, source_max],
                    "y z + 2 / AVG! AVG@ x > AVG@ {dark_offset} - AVG@ x < AVG@ {bright_offset} + x ? ?",
                    dark_offset=scale_delta(dark_offset, 8, clip),
                    bright_offset=scale_delta(bright_offset, 8, clip),
                    func=self._apply_sharpen,
                )

            resharp = unsharpen(
                clip,
                self.sharpen_strength,
                BlurMatrix.BINOMIAL()(resharp, func=self._apply_sharpen),
                func=self._apply_sharpen,
            )

        if self.sharpen_thin:
            median_diff = norm_expr(
                [clip, median_blur(clip, mode=ConvMode.VERTICAL)],
                "y x - {thin} * neutral +",
                thin=self.sharpen_thin,
                func=self._apply_sharpen,
            )
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff, func=self._apply_sharpen)

            resharp = norm_expr(
                [resharp, BlurMatrix.BINOMIAL()(blurred_diff), blurred_diff],
                "y neutral - dup abs z neutral - abs > swap x + x ?",
                func=self._apply_sharpen,
            )

        return resharp

    def _apply_back_blend(self, flt: vs.VideoNode, src: vs.VideoNode) -> vs.VideoNode:
        if self.sharpen_strength or self.sharpen_thin:
            flt = src.std.MergeDiff(gauss_blur(flt.std.MakeDiff(src), self.backblend_sigma))

        return flt

    def _apply_sharpen_limit(self, clip: vs.VideoNode) -> vs.VideoNode:
        undershoot, overshoot = self.sharpen_limit_clamp

        if self.sharpen_strength or self.sharpen_thin:
            if self.sharpen_limit_mode in {
                self.SharpenLimitMode.SPATIAL_PRESMOOTH,
                self.SharpenLimitMode.SPATIAL_POSTSMOOTH,
            }:
                if self.sharpen_limit_radius == 1 and undershoot == overshoot == 0:
                    clip = repair.Mode.MINMAX_SQUARE1(clip, self.bobbed)
                else:
                    inpand = Morpho.minimum(
                        self.bobbed, iterations=self.sharpen_limit_radius, func=self._apply_sharpen_limit
                    )
                    expand = Morpho.maximum(
                        self.bobbed, iterations=self.sharpen_limit_radius, func=self._apply_sharpen_limit
                    )
                    clip = norm_expr(
                        [clip, inpand, expand],
                        "x y {undershoot} - z {overshoot} + clamp",
                        undershoot=undershoot,
                        overshoot=overshoot,
                        func=self._apply_sharpen_limit,
                    )
            elif self.sharpen_limit_mode in {
                self.SharpenLimitMode.TEMPORAL_PRESMOOTH,
                self.SharpenLimitMode.TEMPORAL_POSTSMOOTH,
            }:
                clip = mc_clamp(
                    clip,
                    self.bobbed,
                    self.mv,
                    (undershoot, overshoot),
                    self._apply_sharpen_limit,
                    tr=self.sharpen_limit_radius,
                    thscd=self.analyze_thscd,
                    **self.sharpen_limit_comp_args,
                )

        return clip

    def _apply_noise_restore(self, clip: vs.VideoNode, restore: float) -> vs.VideoNode:
        if restore:
            clip = norm_expr(
                [clip, self.noise], "x y neutral - {restore} * +", restore=restore, func=self._apply_noise_restore
            )

        return clip

    def _apply_final(self) -> None:
        if self.final_tr:
            smoothed = self.mv.degrain(
                self.basic_output,
                tr=self.final_tr,
                thsad=self.final_thsad,
                thscd=self.analyze_thscd,
                **self.final_degrain_args,
            )
        else:
            smoothed = self.basic_output

        if smoothed is not self.bobbed:
            smoothed = self._mask_shimmer(smoothed, self.bobbed, **self.final_mask_shimmer_args)

        if self.sharpen_limit_mode in {
            self.SharpenLimitMode.SPATIAL_POSTSMOOTH,
            self.SharpenLimitMode.TEMPORAL_POSTSMOOTH,
        }:
            smoothed = self._apply_sharpen_limit(smoothed)

        if self.lossless_mode == self.LosslessMode.POSTSMOOTH:
            smoothed = self._apply_lossless(smoothed)

        self.final_output = self._apply_noise_restore(smoothed, self.final_noise_restore)

    def _apply_motion_blur(self) -> None:
        angle_in, angle_out = self.motion_blur_shutter_angle
        blur_level = (angle_out * self.motion_blur_fps_divisor - angle_in) * 100 / 360

        if blur_level:
            blurred = self.mv.flow_blur(
                self.final_output, blur=blur_level, thscd=self.analyze_thscd, **self.motion_blur_blur_args
            )

            if self.motion_blur_mask_args.get("ml", 0):
                mask = self.mv.mask(
                    direction=MVDirection.BACKWARD,
                    kind=MaskMode.VECTOR_LENGTH,
                    thscd=self.analyze_thscd,
                    **self.motion_blur_mask_args,
                )

                blurred = self.final_output.std.MaskedMerge(blurred, mask)
        else:
            blurred = self.final_output

        if self.motion_blur_fps_divisor != 1:
            blurred = blurred[:: self.motion_blur_fps_divisor]

        self.motion_blur_output = blurred

    def _run_process(self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None, func: FuncExcept) -> vs.VideoNode:
        attrs = (
            "tff",
            "is_repair",
            "mv",
            "clip",
            "draft",
            "input",
            "bobbed",
            "noise",
            "prefilter_output",
            "denoise_output",
            "basic_output",
            "final_output",
            "motion_blur_output",
        )

        for attr in attrs:
            with suppress(AttributeError):
                delattr(self, attr)

        self.clip = clip
        self.tff = FieldBased.from_param_or_video(tff, self.clip, True, func)
        self.is_repair = func == self.repair

        if not self.tff.is_inter and func != self.deshimmer:
            raise UnsupportedFieldBasedError("This method is incompatible with progressive video!", func)

        self._apply_prefilter()
        self._apply_analyze()
        self._apply_denoise()
        self._apply_basic()
        self._apply_final()
        self._apply_motion_blur()

        return self.motion_blur_output

    @contextmanager
    def _disable_fps_divisor(self) -> Generator[None]:
        orig = self.motion_blur_fps_divisor
        self.motion_blur_fps_divisor = 1

        try:
            yield
        finally:
            self.motion_blur_fps_divisor = orig

    def deinterlace(self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None) -> vs.VideoNode:
        """
        Deinterlace interlaced input. [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur] `fps_divisor`
        is respected.

        Interpolates missing fields to reconstruct progressive frames.

        Args:
            clip: Clip to process.
            tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.

        Returns:
            Deinterlaced clip.
        """
        return self._run_process(clip, tff, self.deinterlace)

    def bob(self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None) -> vs.VideoNode:
        """
        Bob interlaced input. [QTempGaussMC.motion_blur][vsdeinterlace.QTempGaussMC.motion_blur] `fps_divisor` is
        ignored.

        Interpolates missing fields to reconstruct double-framerate progressive frames.

        Args:
            clip: Clip to process.
            tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.

        Returns:
            Bobbed clip.
        """
        with self._disable_fps_divisor():
            return self._run_process(clip, tff, self.bob)

    def repair(self, clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None) -> vs.VideoNode:
        """
        Repair badly deinterlaced input.

        Drops half the fields to recreate an interlaced stream using the remaining ones.

        Args:
            clip: Clip to process.
            tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.

        Returns:
            Repaired clip.
        """
        return self._run_process(clip, tff, self.repair)

    def deshimmer(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Deshimmer progressive input.

        Removes horizontal shimmering artifacts from progressive sources.

        Args:
            clip: Clip to process.

        Returns:
            Deshimmered clip.
        """
        return self._run_process(clip, FieldBased.PROGRESSIVE, self.deshimmer)
