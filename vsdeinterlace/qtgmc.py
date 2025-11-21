from copy import deepcopy
from math import factorial
from typing import Any, Iterable, Literal, Mapping, Protocol, Self, TypedDict

from jetpytools import CustomIntEnum, CustomValueError, fallback, normalize_seq

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
    get_y,
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
        """
        Arguments available when passing to [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range].
        """

        slope: float
        smooth: float

    class MaskShimmer(TypedDict, total=False):
        """
        Arguments available when passing to the internal `_mask_shimmer` method through
        [QTempGaussMC.prefilter][vsdeinterlace.QTempGaussMC.prefilter],
        [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic]
        and [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final].
        """

        erosion_distance: int
        """How much to deflate then reflate to remove thin areas."""
        over_dilation: int
        """Extra inflation to ensure areas to restore back are fully caught."""

    class Compensate(TypedDict, total=False):
        """
        Arguments available when passing to [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate].
        """

        scbehavior: bool | None
        thsad: int | None
        time: float | None

    class Degrain(TypedDict, total=False):
        """
        Arguments available when passing to the internal `binomial_degrain` method,
        calling [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain] through
        [QTempGaussMC.basic][vsdeinterlace.QTempGaussMC.basic]
        and [QTempGaussMC.source_match][vsdeinterlace.QTempGaussMC.source_match]
        or directly calling [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain] through
        [QTempGaussMC.final][vsdeinterlace.QTempGaussMC.final]
        """

        limit: float | tuple[float | None, float | None] | None
        planes: Planes

    class Mask(TypedDict, total=False):
        """
        Arguments available when passing to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask].
        """

        delta: int
        ml: float | None
        gamma: float | None
        time: float | None
        ysc: int | None

    class Blur(TypedDict, total=False):
        """
        Arguments available when passing to [MVTools.flow_blur][vsdenoise.mvtools.mvtools.MVTools.flow_blur].
        """

        prec: int | None


class QTempGaussMC(VSObject):
    """
    Quick Temporal Gaussian Motion Compensated (QTGMC)

    A very high quality deinterlacer with a range of features for both quality and convenience.
    These include extensive noise processing capabilities, support for repair of progressive material,
    precision source matching, shutter speed simulation, etc.

    Originally based on TempGaussMC by Didée.

    Basic usage:
        ```py
        deinterlace = QTempGaussMC(clip).deinterlace()
        ```

    Refer to the [AviSynth QTGMC documentation](http://avisynth.nl/index.php/QTGMC)
    and the [havsfunc implementation](https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/aa79ebc9eb5517a3a76a74caa98a9d41993ac39b/havsfunc/havsfunc.py#L598-L848)
    for detailed explanations of the underlying algorithm.

    These resources remain relevant, as the core algorithm used here is largely similar.

    Note that parameter names differ in this implementation due to a complete rewrite.
    A mapping between vsjetpack and havsfunc parameters is available [here](https://gist.github.com/emotion3459/33bd2b2a2c21afc6497f65adaf7f0b02).

    Examples:
        - ...
        - Passing a progressive input to reduce shimmering (equivalent to `InputType=2, ProgSADMask=12`):
        ```python
        clip = (
            QTempGaussMC(clip, QTempGaussMC.InputType.REPAIR)
            .basic(mask_args={"ml": 12})
            .deinterlace()
        )
        ```
        Or:
        ```python
        clip = QTempGaussMC(clip, QTempGaussMC.InputType.REPAIR, basic_mask_args={"ml": 12}).deinterlace()
        ```
    """  # fmt: skip

    clip: vs.VideoNode
    """Clip to process."""

    draft: vs.VideoNode
    """Draft processed clip, used as a base for prefiltering & denoising."""

    input: vs.VideoNode
    """Prepared input clip for high quality interpolation."""

    bobbed: vs.VideoNode
    """High quality bobbed clip, initial spatial interpolation."""

    noise: vs.VideoNode
    """Extracted noise when noise processing is enabled."""

    prefilter_output: vs.VideoNode
    """Output of the prefilter stage."""

    denoise_output: vs.VideoNode
    """Output of the denoise stage."""

    basic_output: vs.VideoNode
    """Output of the basic stage."""

    final_output: vs.VideoNode
    """Output of the final stage."""

    motion_blur_output: vs.VideoNode
    """Output of the motion blur stage."""

    class InputType(CustomIntEnum):
        """
        Processing routine to use for the input.
        """

        INTERLACE = 0
        """
        Deinterlace interlaced input.
        """

        PROGRESSIVE = 1
        """
        Deshimmer general progressive material that contains less severe problems.
        """

        REPAIR = 2
        """
        Repair badly deinterlaced material with considerable horizontal artifacts.
        """

    class SearchPostProcess(CustomIntEnum):
        """
        Prefiltering to apply in order to assist with motion search.
        """

        GAUSSBLUR = 0
        """
        Gaussian blur.
        """

        GAUSSBLUR_EDGESOFTEN = 1
        """
        Gaussian blur & edge softening.
        """

    class NoiseProcessMode(CustomIntEnum):
        """
        How to handle processing noise in the source.
        """

        IDENTIFY = 0
        """
        Identify noise only & optionally restore some noise back at the end of basic or final stages.
        """

        DENOISE = 1
        """
        Denoise source & optionally restore some noise back at the end of basic or final stages.
        """

    class NoiseDeintMode(CustomIntEnum):
        """
        When noise is taken from interlaced source, how to 'deinterlace' it before restoring.
        """

        WEAVE = 0
        """
        Double weave source noise, lags behind by one frame.
        """

        BOB = 1
        """
        Bob source noise, results in coarse noise.
        """

        GENERATE = 2
        """
        Generates fresh noise lines.
        """

    class SharpenMode(CustomIntEnum):
        """
        How to re-sharpen the clip after temporal smoothing.
        """

        UNSHARP = 0
        """
        Re-sharpening using unsharpening.
        """

        UNSHARP_MINMAX = 1
        """
        Re-sharpening using unsharpening clamped to the local 3x3 min/max average.
        """

    class SharpenLimitMode(CustomIntEnum):
        """
        How to limit and when to apply re-sharpening of the clip.
        """

        SPATIAL_PRESMOOTH = 0
        """
        Spatial sharpness limiting prior to final stage.
        """

        TEMPORAL_PRESMOOTH = 1
        """
        Temporal sharpness limiting prior to final stage.
        """

        SPATIAL_POSTSMOOTH = 2
        """
        Spatial sharpness limiting after the final stage.
        """

        TEMPORAL_POSTSMOOTH = 3
        """
        Temporal sharpness limiting after the final stage.
        """

    class BackBlendMode(CustomIntEnum):
        """
        When to back blend (blurred) difference between pre & post sharpened clip.
        """

        PRELIMIT = 0
        """
        Perform back-blending prior to sharpness limiting.
        """

        POSTLIMIT = 1
        """
        Perform back-blending after sharpness limiting.
        """

        BOTH = 2
        """
        Perform back-blending both before and after sharpness limiting.
        """

    class SourceMatchMode(CustomIntEnum):
        """
        Creates higher fidelity output with extra processing.
        Will capture more source detail and reduce oversharpening / haloing.
        """

        NONE = 0
        """
        No source match processing.
        """

        BASIC = 1
        """
        Conservative halfway stage that rarely introduces artifacts.
        """

        REFINED = 2
        """
        Restores almost exact source detail but is sensitive to noise & can introduce occasional aliasing.
        """

        TWICE_REFINED = 3
        """
        Restores almost exact source detail.
        """

    class LosslessMode(CustomIntEnum):
        """
        When to put exact source fields into result & clean any artifacts.
        """

        NONE = 0
        """
        Do not restore source fields.
        """

        PRESHARPEN = 1
        """
        Restore source fields prior to re-sharpening. Not exactly lossless.
        """

        POSTSMOOTH = 2
        """
        Restore source fields after final temporal smooth. True lossless but less stable.
        """

    def __init__(
        self,
        clip: vs.VideoNode,
        input_type: InputType = InputType.INTERLACE,
        tff: FieldBasedLike | bool | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            clip: Clip to process.
            input_type: Indicates processing routine.
            tff: Field order of the clip.
            **kwargs: Additional arguments to be passed to the parameter stage methods.
                Use the method's name as prefix to pass an argument to the respective method.

                Example for passing tr=1 to the prefilter stage: `prefilter_tr=1`.
        """
        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        self.clip = clip
        self.input_type = input_type

        if self.input_type == self.InputType.PROGRESSIVE and clip_fieldbased.is_inter():
            raise UnsupportedFieldBasedError(f"{self.input_type} incompatible with interlaced video!", self.__class__)
        elif self.input_type in (self.InputType.INTERLACE, self.InputType.REPAIR) and not clip_fieldbased.is_inter():
            raise UnsupportedFieldBasedError(f"{self.input_type} incompatible with progressive video!", self.__class__)

        self.tff = None if self.input_type == self.InputType.PROGRESSIVE else clip_fieldbased.is_tff()

        # Set default parameters for all the stages in this exact order
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
            raise CustomValueError("Unknown arguments were passed", self.__class__, kwargs)

    def prefilter(
        self,
        *,
        tr: int = 2,
        sc_threshold: float = 0.1,
        postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
        strength: tuple[float, float] | Literal[False] = (1.9, 0.1),
        limit: tuple[float, float, float] = (3, 7, 2),
        bias: float = 0.51,
        range_expansion_args: QTGMCArgs.PrefilterToFullRange | None = None,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = None,
    ) -> Self:
        """
        Configure parameters for the prefilter stage.

        Args:
            tr: Radius of the initial temporal binomial smooth.
            sc_threshold: Threshold for scene changes, disables sc detection if False.
            postprocess: Post-processing routine to use.
            strength: Tuple containing gaussian blur sigma & blend weight of the blur.
            limit: 3-step limiting (8-bit) thresholds for the gaussian blur post-processing. Only for
                [SearchPostProcess.GAUSSBLUR_EDGESOFTEN][vsdeinterlace.qtgmc.QTempGaussMC.SearchPostProcess.GAUSSBLUR_EDGESOFTEN].
            bias: Bias for blending the gaussian blurred clip with the limited output. Only for
                [SearchPostProcess.GAUSSBLUR_EDGESOFTEN][vsdeinterlace.qtgmc.QTempGaussMC.SearchPostProcess.GAUSSBLUR_EDGESOFTEN].
            range_expansion_args: Arguments passed to
                [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range].
            mask_shimmer_args: mask_shimmer_args: Arguments passed to the mask_shimmer call.
        """

        self.prefilter_tr = tr
        self.prefilter_sc_threshold = sc_threshold
        self.prefilter_postprocess = postprocess
        self.prefilter_strength: tuple[float, float] | Literal[False] = strength
        self.prefilter_limit = limit
        self.prefilter_bias = bias
        self.prefilter_range_expansion_args = fallback(range_expansion_args, QTGMCArgs.PrefilterToFullRange())
        self.prefilter_mask_shimmer_args = fallback(mask_shimmer_args, QTGMCArgs.MaskShimmer())

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
        Configure parameters for the motion analysis stage.

        Args:
            force_tr: Always analyze motion to at least this, even if otherwise unnecessary.
            preset: MVTools preset defining base values for the MVTools object.
            blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
            overlap: The blksize divisor for block overlap. Larger overlapping reduces blocking artifacts.
            refine: Number of times to recalculate motion vectors with halved block size.
            thsad_recalc: Only bad quality new vectors with a SAD above this will be re-estimated by search. thsad value
                is scaled to 8x8 block size.
            thscd: Scene change detection thresholds:

                   - First value: SAD threshold for considering a block changed between frames.
                   - Second value: Percentage of changed blocks needed to trigger a scene change.
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
        mode: NoiseProcessMode = NoiseProcessMode.IDENTIFY,
        deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
        mc_denoise: bool = True,
        stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
        func_comp_args: QTGMCArgs.Compensate | None = None,
        stabilize_comp_args: QTGMCArgs.Compensate | None = None,
    ) -> Self:
        """
        Configure parameters for the denoise stage.

        Args:
            tr: Temporal radius of the denoising function & it's motion compensation.
            func: Denoising function to use.
            mode: Noise handling method to use.
            deint: Noise deinterlacing method to use.
            mc_denoise: Whether to perform motion-compensated denoising.
            stabilize: Weights to use when blending source noise with compensated noise.
            func_comp_args: Arguments passed to [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for
                denoising.
            stabilize_comp_args: Arguments passed to [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate]
                for stabilization.
        """

        self.denoise_tr = tr
        self.denoise_func = func.denoise if isinstance(func, DFTTest) else func
        self.denoise_mode = mode
        self.denoise_deint = deint
        self.denoise_mc_denoise = mc_denoise
        self.denoise_stabilize: tuple[float, float] | Literal[False] = stabilize
        self.denoise_func_comp_args = fallback(func_comp_args, QTGMCArgs.Compensate())
        self.denoise_stabilize_comp_args = fallback(stabilize_comp_args, QTGMCArgs.Compensate())

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
        Configure parameters for the basic stage.

        Args:
            tr: Temporal radius of the motion compensated binomial smooth.
            thsad: Thsad of the motion compensated binomial smooth.
            bobber: Bobber to use for initial spatial interpolation.
            noise_restore: How much noise to restore after this stage.
            degrain_args: Arguments passed to the binomial_degrain call.
            mask_args: Arguments passed to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask] for
                [InputType.REPAIR][vsdeinterlace.qtgmc.QTempGaussMC.InputType.REPAIR].
            mask_shimmer_args: Arguments passed to the mask_shimmer call.
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
        mode: SourceMatchMode = SourceMatchMode.NONE,
        similarity: float = 0.5,
        enhance: float = 0.5,
        degrain_args: QTGMCArgs.Degrain | None = None,
    ) -> Self:
        """
        Configure parameters for the source_match stage.

        Args:
            tr: Temporal radius of the refinement motion compensated binomial smooth.
            bobber: Bobber to use for refined spatial interpolation. Defaults to the basic bobber.
            mode: Specifies number of refinement steps to perform.
            similarity: Temporal similarity of the error created by smoothing.
            enhance: Sharpening strength prior to source match refinement.
            degrain_args: Arguments passed to the binomial_degrain call.
        """

        self.source_match_tr = tr
        self.source_match_bobber = bobber
        self.source_match_mode = mode
        self.source_match_similarity = similarity
        self.source_match_enhance = enhance
        self.source_match_degrain_args = fallback(degrain_args, QTGMCArgs.Degrain())

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
        Configure parameter for the lossless stage.

        Args:
            mode: Specifies at which stage to re-weave the original fields.
            anti_comb: Whether to apply combing reduction post-processing.
        """

        self.lossless_mode = mode
        self.lossless_anti_comb = anti_comb

        return self

    def sharpen(
        self,
        *,
        mode: SharpenMode | None = None,
        strength: float | None = None,
        clamp: float | tuple[float, float] = 1,
        thin: float = 0,
    ) -> Self:
        """
        Configure parameters for the sharpen stage.

        Args:
            mode: Specifies the type of sharpening to use.
                Defaults to [SharpenMode.UNSHARP][vsdeinterlace.qtgmc.QTempGaussMC.SharpenMode.UNSHARP] for
                    [InputType.PROGRESSIVE][vsdeinterlace.qtgmc.QTempGaussMC.InputType.PROGRESSIVE] or
                    [SharpenMode.UNSHARP_MINMAX][vsdeinterlace.qtgmc.QTempGaussMC.SharpenMode.UNSHARP_MINMAX] otherwise.
            strength: Sharpening strength. Defaults to 1 for
                [SourceMatchMode.NONE][vsdeinterlace.qtgmc.QTempGaussMC.SourceMatchMode.NONE] or 0 otherwise.
            clamp: Clamp the sharpening strength of
                [SharpenMode.UNSHARP_MINMAX][vsdeinterlace.qtgmc.QTempGaussMC.SharpenMode.UNSHARP_MINMAX] to the min/max
                average plus/minus this.
            thin: How much to vertically thin edges.
        """

        self.sharpen_mode = mode
        self.sharpen_strength = strength
        self.sharpen_clamp = normalize_seq(clamp, 2)
        self.sharpen_thin = thin

        return self

    @property
    def sharpen_mode(self) -> SharpenMode:
        return fallback(
            self._sharpen_mode,
            self.SharpenMode.UNSHARP
            if self.input_type == self.InputType.PROGRESSIVE
            else self.SharpenMode.UNSHARP_MINMAX,
        )

    @sharpen_mode.setter
    def sharpen_mode(self, value: SharpenMode | None) -> None:
        self._sharpen_mode = value

    @property
    def sharpen_strength(self) -> float:
        return fallback(self._sharpen_strength, 0 if self.source_match_mode else 1)

    @sharpen_strength.setter
    def sharpen_strength(self, value: float | None) -> None:
        self._sharpen_strength = value

    def back_blend(self, *, mode: BackBlendMode = BackBlendMode.PRELIMIT, sigma: float = 1.4) -> Self:
        """
        Configure parameters for the back_blend stage.

        Args:
            mode: Specifies at which stage to perform back-blending.
            sigma: Gaussian blur sigma.
        """

        self.backblend_mode = mode
        self.backblend_sigma = sigma

        return self

    def sharpen_limit(
        self,
        *,
        mode: SharpenLimitMode = SharpenLimitMode.TEMPORAL_PRESMOOTH,
        radius: int | None = None,
        clamp: float | tuple[float, float] = 0,
        comp_args: QTGMCArgs.Compensate | None = None,
    ) -> Self:
        """
        Configure parameters for the sharpen_limit stage.

        Args:
            mode: Specifies type of limiting & at which stage to perform it.
            radius: Radius of sharpness limiting. Defaults to 1 for
                [SourceMatchMode.NONE][vsdeinterlace.qtgmc.QTempGaussMC.SourceMatchMode.NONE] or 0 otherwise.
            clamp: How much undershoot/overshoot to allow.
            comp_args: Arguments passed to [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for
                temporal limiting.
        """

        self.sharpen_limit_mode = mode
        self.sharpen_limit_radius = radius
        self.sharpen_limit_clamp = clamp
        self.sharpen_limit_comp_args = fallback(comp_args, QTGMCArgs.Compensate())

        return self

    @property
    def sharpen_limit_radius(self) -> int:
        return fallback(self._sharpen_limit_radius, 0 if self.source_match_mode else 1)

    @sharpen_limit_radius.setter
    def sharpen_limit_radius(self, value: int | None) -> None:
        self._sharpen_limit_radius = value

    def final(
        self,
        *,
        tr: int = 1,
        thsad: int | tuple[int, int] = 256,
        noise_restore: float = 0,
        degrain_args: QTGMCArgs.Degrain | None = None,
        mask_shimmer_args: QTGMCArgs.MaskShimmer | None = None,
    ) -> Self:
        """
        Configure parameters for the final stage.

        Args:
            tr: Temporal radius of the motion compensated smooth.
            thsad: Thsad of the motion compensated smooth.
            noise_restore: How much noise to restore after this stage.
            degrain_args: Arguments passed to [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain].
            mask_shimmer_args: Arguments passed to the mask_shimmer call.
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
        Configure parameters for the motion blur stage.

        Args:
            shutter_angle: Tuple containing the source and output shutter angle. Will apply motion blur if they do not
                match.
            fps_divisor: Factor by which to reduce framerate.
            blur_args: Arguments passed to [MVTools.flow_blur][vsdenoise.mvtools.mvtools.MVTools.flow_blur].
            mask_args: Arguments passed to [MVTools.mask][vsdenoise.mvtools.mvtools.MVTools.mask].
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
        erosion_distance: int = 4,
        over_dilation: int = 0,
    ) -> vs.VideoNode:
        """
        Compare processed clip with reference clip,
        only allow thin, horizontal areas of difference, i.e. bob shimmer fixes.

        Args:
            flt: Processed clip to perform masking on.
            src: Unprocessed clip to restore from.
            erosion_distance: How much to deflate then reflate to remove thin areas.
            over_dilation: Extra inflation to ensure areas to restore back are fully caught.
        """
        if not erosion_distance:
            return flt

        ed_iter1, ed_iter2 = (1 + erosion_distance // 3, 1 + (erosion_distance + 1) // 3)
        od_iter1, od_iter2 = (over_dilation // 3, over_dilation % 3)

        diff = src.std.MakeDiff(flt)

        closing = Morpho.maximum(diff, iterations=ed_iter1, coords=Coordinates.VERTICAL, func=self._mask_shimmer)
        opening = Morpho.minimum(diff, iterations=ed_iter1, coords=Coordinates.VERTICAL, func=self._mask_shimmer)

        if erosion_residual := erosion_distance % 3:
            closing = Morpho.inflate(closing, func=self._mask_shimmer)
            opening = Morpho.deflate(opening, func=self._mask_shimmer)

            if erosion_residual == 2:
                closing = median_blur(closing, func=self._mask_shimmer)
                opening = median_blur(opening, func=self._mask_shimmer)

        closing = Morpho.minimum(closing, iterations=ed_iter2, coords=Coordinates.VERTICAL, func=self._mask_shimmer)
        opening = Morpho.maximum(opening, iterations=ed_iter2, coords=Coordinates.VERTICAL, func=self._mask_shimmer)

        if over_dilation:
            closing = Morpho.minimum(closing, iterations=od_iter1, func=self._mask_shimmer)
            opening = Morpho.maximum(opening, iterations=od_iter1, func=self._mask_shimmer)

            closing = Morpho.deflate(closing, iterations=od_iter2, func=self._mask_shimmer)
            opening = Morpho.inflate(opening, iterations=od_iter2, func=self._mask_shimmer)

        return norm_expr(
            [flt, diff, closing, opening], "x y z neutral min a neutral max clip neutral - +", func=self._mask_shimmer
        )

    def _interpolate(self, clip: vs.VideoNode, bobber: Bobber) -> vs.VideoNode:
        if self.input_type != self.InputType.PROGRESSIVE:
            clip = bobber.bob(clip, tff=self.tff)

        return clip

    def _binomial_degrain(self, clip: vs.VideoNode, tr: int, **degrain_args: Any) -> vs.VideoNode:
        from numpy import linalg, zeros

        def _get_weights(n: int) -> Iterable[Any]:
            k, rhs = 1, list[int]()
            mat = zeros((n + 1, n + 1))

            for i in range(1, n + 2):
                mat[n + 1 - i, i - 1] = mat[n, i - 1] = 1 / 3
                rhs.append(k)
                k = k * (2 * n + 1 - i) // i

            mat[n, 0] = 1

            return linalg.solve(mat, rhs)

        if not tr:
            return clip

        backward, forward = self.mv.vectors.get_vectors(tr=tr)
        vectors = MotionVectors()
        degrained = list[vs.VideoNode]()

        sup = self.mv.super(clip, levels=1)

        for delta in range(tr):
            vectors.set_vector(backward[delta], MVDirection.BACKWARD, 1)
            vectors.set_vector(forward[delta], MVDirection.FORWARD, 1)

            degrained.append(
                self.mv.degrain(clip, sup, vectors, thsad=self.basic_thsad, thscd=self.analyze_thscd, **degrain_args)
            )
            vectors.clear()

        return BlurMatrix.custom(_get_weights(tr), ConvMode.TEMPORAL)([clip, *degrained], func=self._binomial_degrain)

    def _apply_prefilter(self) -> None:
        self.draft = Catrom().bob(self.clip, tff=self.tff) if self.input_type == self.InputType.INTERLACE else self.clip

        search = self.draft

        if not self.analyze_preset.get("chroma", True):
            search = get_y(search)

        if self.input_type == self.InputType.REPAIR:
            search = BlurMatrix.BINOMIAL()(search, mode=ConvMode.VERTICAL, func=self._apply_prefilter)

        if self.prefilter_tr:
            scenes = sc_detect(search, self.prefilter_sc_threshold)
            smoothed = BlurMatrix.BINOMIAL(self.prefilter_tr, mode=ConvMode.TEMPORAL)(
                scenes, scenechange=True, func=self._apply_prefilter
            )
            smoothed = self._mask_shimmer(smoothed, search, **self.prefilter_mask_shimmer_args)
        else:
            smoothed = search

        if self.prefilter_strength:
            gauss_sigma, blend_weight = self.prefilter_strength

            blurred = core.std.Merge(gauss_blur(smoothed, gauss_sigma), smoothed, blend_weight)

            if self.prefilter_postprocess == self.SearchPostProcess.GAUSSBLUR_EDGESOFTEN:
                lim1, lim2, lim3 = [scale_delta(thr, 8, self.clip) for thr in self.prefilter_limit]

                blurred = norm_expr(
                    [blurred, smoothed, search],
                    "z y {lim1} - y {lim1} + clip TWEAK! "
                    "x {lim2} + TWEAK@ < x {lim3} + x {lim2} - TWEAK@ > x {lim3} - "
                    "x {bias} * TWEAK@ 1 {bias} - * + ? ?",
                    lim1=lim1,
                    lim2=lim2,
                    lim3=lim3,
                    bias=self.prefilter_bias,
                    func=self._apply_prefilter,
                )
        else:
            blurred = smoothed

        self.prefilter_output = prefilter_to_full_range(
            blurred, func=self._apply_prefilter, **self.prefilter_range_expansion_args
        )

    def _apply_analyze(self) -> None:
        tr = max(
            self.analyze_force_tr,
            self.denoise_tr,
            self.basic_tr,
            self.source_match_tr,
            self.sharpen_limit_radius
            if self.sharpen_limit_mode
            in (self.SharpenLimitMode.TEMPORAL_PRESMOOTH, self.SharpenLimitMode.TEMPORAL_POSTSMOOTH)
            else 0,
            self.final_tr,
        )

        blksize, overlap = self.analyze_blksize, self.analyze_overlap
        thsad_recalc = fallback(
            self.analyze_thsad_recalc,
            round((self.basic_thsad[0] if isinstance(self.basic_thsad, tuple) else self.basic_thsad) / 2),
        )

        self.mv = MVTools(self.draft, **{**self.analyze_preset, "search_clip": self.prefilter_output})
        self.mv.analyze(tr=tr, blksize=blksize, overlap=refine_blksize(blksize, overlap))

        for _ in range(self.analyze_refine):
            blksize = refine_blksize(blksize)
            self.mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=refine_blksize(blksize, overlap))

    def _apply_denoise(self) -> None:
        self.denoise_output = self.clip

        no_restore = self.basic_noise_restore == self.final_noise_restore == 0

        if self.denoise_mode == self.NoiseProcessMode.IDENTIFY and no_restore:
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

        if self.input_type == self.InputType.INTERLACE:
            denoised = reinterlace(denoised, self.tff, self._apply_denoise)

        if self.denoise_mode == self.NoiseProcessMode.DENOISE:
            self.denoise_output = denoised

        self.noise = self.clip.std.MakeDiff(denoised)

        if no_restore:
            return

        if self.input_type == self.InputType.INTERLACE:
            match self.denoise_deint:
                case self.NoiseDeintMode.WEAVE:
                    new_noise = self.noise.std.SeparateFields(self.tff).std.DoubleWeave(self.tff)
                case self.NoiseDeintMode.BOB:
                    new_noise = Catrom().bob(self.noise, tff=self.tff)
                case self.NoiseDeintMode.GENERATE:
                    noise_source = self.noise.std.SeparateFields(self.tff)

                    noise_max = Morpho.maximum(
                        Morpho.maximum(noise_source), coords=Coordinates.HORIZONTAL, func=self._apply_denoise
                    )
                    noise_min = Morpho.minimum(
                        Morpho.minimum(noise_source), coords=Coordinates.HORIZONTAL, func=self._apply_denoise
                    )

                    gen_noise = Grainer.GAUSS(
                        noise_source, 2048, protect_edges=False, protect_neutral_chroma=False, neutral_out=True
                    )
                    gen_noise = norm_expr(
                        [noise_max, noise_min, gen_noise], "x y - z * range_size / y +", func=self._apply_denoise
                    )
                    new_noise = reweave(noise_source, gen_noise, self.tff, self._apply_denoise)

            self.noise = FieldBased.PROGRESSIVE.apply(new_noise)

        if self.denoise_stabilize:
            weight1, weight2 = self.denoise_stabilize

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
                "x neutral - abs y neutral - abs > x y ? {weight1} * x y + {weight2} * +",
                weight1=weight1,
                weight2=weight2,
                func=self._apply_denoise,
            )

    def _apply_basic(self) -> None:
        if self.input_type == self.InputType.REPAIR:
            self.input = reinterlace(self.denoise_output, self.tff, self._interpolate)
        else:
            self.input = self.denoise_output

        self.bobbed = self._interpolate(self.input, self.basic_bobber)

        if self.input_type == self.InputType.REPAIR and self.basic_mask_args.get("ml", 0):
            mask = self.mv.mask(
                self.prefilter_output,
                direction=MVDirection.BACKWARD,
                kind=MaskMode.SAD,
                thscd=self.analyze_thscd,
                **self.basic_mask_args,
            )
            self.bobbed = self.denoise_output.std.MaskedMerge(self.bobbed, mask)

        self.bobbed = self.mv.super(self.bobbed, levels=1)

        smoothed = self._binomial_degrain(self.bobbed, self.basic_tr, **self.basic_degrain_args)
        if self.basic_tr:
            smoothed = self._mask_shimmer(smoothed, self.bobbed, **self.basic_mask_shimmer_args)

        if self.source_match_mode:
            smoothed = self._apply_source_match(smoothed)

        if self.lossless_mode == self.LosslessMode.PRESHARPEN:
            smoothed = self._apply_lossless(smoothed)

        resharp = self._apply_sharpen(smoothed)

        if self.backblend_mode in (self.BackBlendMode.PRELIMIT, self.BackBlendMode.BOTH):
            resharp = self._apply_back_blend(resharp, smoothed)

        if self.sharpen_limit_mode in (
            self.SharpenLimitMode.SPATIAL_PRESMOOTH,
            self.SharpenLimitMode.TEMPORAL_PRESMOOTH,
        ):
            resharp = self._apply_sharpen_limit(resharp)

        if self.backblend_mode in (self.BackBlendMode.POSTLIMIT, self.BackBlendMode.BOTH):
            resharp = self._apply_back_blend(resharp, smoothed)

        self.basic_output = self._apply_noise_restore(resharp, self.basic_noise_restore)

    def _apply_source_match(self, clip: vs.VideoNode) -> vs.VideoNode:
        def _error_adjustment(ref: vs.VideoNode, clip: vs.VideoNode, tr: int) -> vs.VideoNode:
            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2**tr_f / (binomial_coeff + self.source_match_similarity * (2**tr_f - binomial_coeff))

            return norm_expr([ref, clip], "x {adj} 1 + * y {adj} * -", adj=error_adj, func=_error_adjustment)

        if self.input_type != self.InputType.PROGRESSIVE:
            clip = reinterlace(clip, self.tff, self._apply_source_match)

        adjusted = _error_adjustment(self.input, clip, self.basic_tr)
        new_bobbed = self._interpolate(adjusted, self.basic_bobber)
        matched = self._binomial_degrain(new_bobbed, self.basic_tr, **self.basic_degrain_args)

        if self.source_match_mode > self.SourceMatchMode.BASIC:
            if self.source_match_enhance:
                matched = unsharpen(
                    matched, self.source_match_enhance, BlurMatrix.BINOMIAL(), func=self._apply_source_match
                )

            if self.input_type != self.InputType.PROGRESSIVE:
                clip = reinterlace(matched, self.tff, self._apply_source_match)
            else:
                clip = matched

            diff = self.input.std.MakeDiff(clip)
            refine_bobbed = self._interpolate(diff, self.source_match_bobber)
            refine_matched = self._binomial_degrain(
                refine_bobbed, self.source_match_tr, **self.source_match_degrain_args
            )

            if self.source_match_mode == self.SourceMatchMode.TWICE_REFINED:
                refine_adjusted = _error_adjustment(refine_bobbed, refine_matched, self.source_match_tr)
                refine_matched = self._binomial_degrain(
                    refine_adjusted, self.source_match_tr, **self.source_match_degrain_args
                )

            return matched.std.MergeDiff(refine_matched)

        return matched

    def _apply_lossless(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.input_type == self.InputType.PROGRESSIVE:
            return clip

        fields_src = self.denoise_output.std.SeparateFields(self.tff)
        if self.input_type == self.InputType.REPAIR:
            fields_src = core.std.SelectEvery(fields_src, 4, (0, 3))
        fields_flt = clip.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = reweave(fields_src, fields_flt, self.tff, self._apply_lossless)

        if self.lossless_anti_comb:
            median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL, func=self._apply_lossless))
            fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

            processed_diff = norm_expr(
                [median_blur(fields_diff, mode=ConvMode.VERTICAL, func=self._apply_lossless), fields_diff],
                "x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?",
                func=self._apply_lossless,
            )
            processed_diff = repair.Mode.MINMAX_SQUARE1(
                processed_diff, remove_grain.Mode.MINMAX_AROUND2(processed_diff)
            )
            woven = reweave(fields_src, fields_flt.std.MakeDiff(processed_diff), self.tff, self._apply_lossless)

        return FieldBased.PROGRESSIVE.apply(woven)

    def _apply_sharpen(self, clip: vs.VideoNode) -> vs.VideoNode:
        resharp = clip

        if self.sharpen_strength:
            if self.sharpen_mode == self.SharpenMode.UNSHARP:
                resharp = unsharpen(clip, self.sharpen_strength, BlurMatrix.BINOMIAL(), func=self._apply_sharpen)
            elif self.sharpen_mode == self.SharpenMode.UNSHARP_MINMAX:
                undershoot, overshoot = self.sharpen_clamp

                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL, func=self._apply_sharpen)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL, func=self._apply_sharpen)

                clamp = norm_expr(
                    [clip, source_min, source_max],
                    "y z + 2 / AVG! AVG@ x > AVG@ {undershoot} - AVG@ x < AVG@ {overshoot} + AVG@ ? ?",
                    undershoot=scale_delta(undershoot, 8, clip),
                    overshoot=scale_delta(overshoot, 8, clip),
                    func=self._apply_sharpen,
                )
                resharp = unsharpen(
                    clip,
                    self.sharpen_strength,
                    BlurMatrix.BINOMIAL()(clamp, func=self._apply_sharpen),
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
        if self.backblend_sigma and (self.sharpen_strength or self.sharpen_thin):
            flt = flt.std.MakeDiff(gauss_blur(flt.std.MakeDiff(src), self.backblend_sigma))

        return flt

    def _apply_sharpen_limit(self, clip: vs.VideoNode) -> vs.VideoNode:
        if (self.sharpen_strength or self.sharpen_thin) and self.sharpen_limit_radius:
            if self.sharpen_limit_mode in (
                self.SharpenLimitMode.SPATIAL_PRESMOOTH,
                self.SharpenLimitMode.SPATIAL_POSTSMOOTH,
            ):
                if self.sharpen_limit_radius == 1:
                    clip = repair.Mode.MINMAX_SQUARE1(clip, self.bobbed)
                else:
                    inpand = Morpho.minimum(
                        self.bobbed, iterations=self.sharpen_limit_radius, func=self._apply_sharpen_limit
                    )
                    expand = Morpho.maximum(
                        self.bobbed, iterations=self.sharpen_limit_radius, func=self._apply_sharpen_limit
                    )
                    clip = norm_expr([clip, inpand, expand], "x y z clip", func=self._apply_sharpen_limit)
            elif self.sharpen_limit_mode in (
                self.SharpenLimitMode.TEMPORAL_PRESMOOTH,
                self.SharpenLimitMode.TEMPORAL_POSTSMOOTH,
            ):
                clip = mc_clamp(
                    clip,
                    self.bobbed,
                    self.mv,
                    self.sharpen_limit_clamp,
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
        smoothed = self._mask_shimmer(smoothed, self.bobbed, **self.final_mask_shimmer_args)

        if self.sharpen_limit_mode in (
            self.SharpenLimitMode.SPATIAL_POSTSMOOTH,
            self.SharpenLimitMode.TEMPORAL_POSTSMOOTH,
        ):
            smoothed = self._apply_sharpen_limit(smoothed)

        if self.lossless_mode == self.LosslessMode.POSTSMOOTH:
            smoothed = self._apply_lossless(smoothed)

        self.final_output = self._apply_noise_restore(smoothed, self.final_noise_restore)

    def _apply_motion_blur(self) -> None:
        angle_in, angle_out = self.motion_blur_shutter_angle

        if angle_out * self.motion_blur_fps_divisor != angle_in:
            blur_level = (angle_out * self.motion_blur_fps_divisor - angle_in) * 100 / 360

            blurred = self.mv.flow_blur(
                self.final_output, blur=blur_level, thscd=self.analyze_thscd, **self.motion_blur_blur_args
            )

            if self.motion_blur_mask_args.get("ml", 0):
                mask = self.mv.mask(
                    self.prefilter_output,
                    direction=MVDirection.BACKWARD,
                    kind=MaskMode.MOTION,
                    thscd=self.analyze_thscd,
                    **self.motion_blur_mask_args,
                )

                blurred = self.final_output.std.MaskedMerge(blurred, mask)
        else:
            blurred = self.final_output

        if self.motion_blur_fps_divisor > 1:
            blurred = blurred[:: self.motion_blur_fps_divisor]

        self.motion_blur_output = blurred

    def deinterlace(self) -> vs.VideoNode:
        """
        Start the deinterlacing process.

        Returns:
            Deinterlaced clip.
        """

        self._apply_prefilter()
        self._apply_analyze()
        self._apply_denoise()
        self._apply_basic()
        self._apply_final()
        self._apply_motion_blur()

        return self.motion_blur_output
