from copy import deepcopy
from functools import partial
from math import factorial
from typing import Any, Iterable, Literal, Protocol, Self

from jetpytools import CustomIntEnum, KwargsT, fallback, normalize_seq

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
    UnsupportedFieldBasedError,
    UnsupportedSampleTypeError,
    VSFunctionKwArgs,
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
    def __call__(self, clip: vs.VideoNode, /, *, tr: int = ...) -> vs.VideoNode: ...


class QTempGaussMC(VSObject):
    """
    Quasi Temporal Gaussian Motion Compensated (QTGMC)

    A very high quality deinterlacer with a range of features for both quality and convenience.
    These include extensive noise processing capabilities, support for repair of progressive material,
    precision source matching, shutter speed simulation, etc.

    Originally based on TempGaussMC by Didée.

    Basic usage:
        ```py
        deinterlace = (
            QTempGaussMC(clip)
            .prefilter()
            .analyze()
            .denoise()
            .basic()
            .source_match()
            .lossless()
            .sharpen()
            .back_blend()
            .sharpen_limit()
            .final()
            .motion_blur()
            .deinterlace()
        )
        ```
    """

    clip: vs.VideoNode
    """Clip to process."""

    draft: vs.VideoNode
    """Draft processed clip, used as a base for prefiltering & denoising."""

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
        Repair badly deinterlaced material with considerable horizontal artefacts.
        """

    class SearchPostProcess(CustomIntEnum):
        """
        Prefiltering to apply in order to assist with motion search.
        """

        NONE = 0
        """
        No post-processing.
        """

        GAUSSBLUR = 1
        """
        Gaussian blur.
        """

        GAUSSBLUR_EDGESOFTEN = 2
        """
        Gaussian blur & edge softening.
        """

    class NoiseProcessMode(CustomIntEnum):
        """
        How to handle processing noise in the source.
        """

        NONE = 0
        """
        No noise processing.
        """

        DENOISE = 1
        """
        Denoise source & optionally restore some noise back at the end of basic or final stages.
        """

        IDENTIFY = 2
        """
        Identify noise only & optionally restore some noise back at the end of basic or final stages.
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

    class SharpMode(CustomIntEnum):
        """
        How to re-sharpen the clip after temporally blurring.
        """

        NONE = 0
        """
        No re-sharpening.
        """

        UNSHARP = 1
        """
        Re-sharpening using unsharpening.
        """

        UNSHARP_MINMAX = 2
        """
        Re-sharpening using unsharpening clamped to the local 3x3 min/max average.
        """

    class SharpLimitMode(CustomIntEnum):
        """
        How to limit and when to apply re-sharpening of the clip.
        """

        NONE = 0
        """
        No sharpness limiting.
        """

        SPATIAL_PRESMOOTH = 1
        """
        Spatial sharpness limiting prior to final stage.
        """

        TEMPORAL_PRESMOOTH = 2
        """
        Temporal sharpness limiting prior to final stage.
        """

        SPATIAL_POSTSMOOTH = 3
        """
        Spatial sharpness limiting after the final stage.
        """

        TEMPORAL_POSTSMOOTH = 4
        """
        Temporal sharpness limiting after the final stage.
        """

    class BackBlendMode(CustomIntEnum):
        """
        When to back blend (blurred) difference between pre & post sharpened clip.
        """

        NONE = 0
        """
        No back-blending.
        """

        PRELIMIT = 1
        """
        Perform back-blending prior to sharpness limiting.
        """

        POSTLIMIT = 2
        """
        Perform back-blending after sharpness limiting.
        """

        BOTH = 3
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
        Conservative halfway stage that rarely introduces artefacts.
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
        When to put exact source fields into result & clean any artefacts.
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
    ) -> None:
        """
        Args:
            clip: Clip to process.
            input_type: Indicates processing routine.
            tff: Field order of the clip.
        """
        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        self.clip = clip
        self.input_type = input_type
        self.tff = clip_fieldbased.is_tff()
        self.double_rate = self.input_type != self.InputType.REPAIR

        UnsupportedSampleTypeError.check(self.clip, vs.INTEGER, self.__class__, "FLOAT input is not supported!")

        if self.input_type == self.InputType.PROGRESSIVE and clip_fieldbased.is_inter():
            raise UnsupportedFieldBasedError(f"{self.input_type} incompatible with interlaced video!", self.__class__)
        elif self.input_type in (self.InputType.INTERLACE, self.InputType.REPAIR) and not clip_fieldbased.is_inter():
            raise UnsupportedFieldBasedError(f"{self.input_type} incompatible with progressive video!", self.__class__)

    def prefilter(
        self,
        *,
        tr: int = 2,
        sc_threshold: float | Literal[False] = 0.1,
        postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
        strength: tuple[float, float] = (1.9, 0.1),
        limit: tuple[int | float, int | float, int | float] = (3, 7, 2),
        range_expansion_args: dict[str, Any] | None | Literal[False] = None,
        mask_shimmer_args: dict[str, Any] | None = None,
    ) -> Self:
        """
        Configure parameters for the prefilter stage.

        Args:
            tr: Radius of the initial temporal binomial smooth.
            sc_threshold: Threshold for scene changes, disables sc detection if False.
            postprocess: Post-processing routine to use.
            strength: Tuple containing gaussian blur sigma & blend weight of the blur.
            limit: 3-step limiting thresholds for the gaussian blur post-processing.
            range_expansion_args: Arguments passed to
                [prefilter_to_full_range][vsdenoise.prefilters.prefilter_to_full_range].
            mask_shimmer_args: mask_shimmer_args: Arguments passed to the mask_shimmer call.
        """

        self.prefilter_tr = tr
        self.prefilter_sc_threshold = sc_threshold
        self.prefilter_postprocess = postprocess
        self.prefilter_blur_strength = strength
        self.prefilter_soften_limit = limit
        self.prefilter_range_expansion_args: dict[str, Any] | Literal[False] = fallback(range_expansion_args, {})
        self.prefilter_mask_shimmer_args = fallback(mask_shimmer_args, {})

        return self

    def analyze(
        self,
        *,
        force_tr: int = 1,
        preset: MVToolsPreset = MVToolsPreset.HQ_SAD,
        chroma: bool = True,
        blksize: int | tuple[int, int] = 16,
        overlap: int | tuple[int, int] = 2,
        refine: int = 1,
        thsad_recalc: int | None = None,
        thscd: int | tuple[int | None, int | float | None] | None = (180, 38.5),
    ) -> Self:
        """
        Configure parameters for the motion analysis stage.

        Args:
            force_tr: Always analyze motion to at least this, even if otherwise unnecessary.
            preset: MVTools preset defining base values for the MVTools object.
            chroma: Whether to consider chroma in motion vector calculations.
            blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
            overlap: The blksize divisor for block overlap. Larger overlapping reduces blocking artifacts.
            refine: Number of times to recalculate motion vectors with halved block size.
            thsad_recalc: Only bad quality new vectors with a SAD above this will be re-estimated by search. thsad value
                is scaled to 8x8 block size.
            thscd: Scene change detection thresholds: - First value: SAD threshold for considering a block changed
                between frames. - Second value: Percentage of changed blocks needed to trigger a scene change.
        """

        preset.pop("search_clip", None)

        self.analyze_tr = force_tr
        self.analyze_preset = preset
        self.analyze_chroma = chroma
        self.analyze_blksize = blksize
        self.analyze_overlap = overlap
        self.analyze_refine = refine
        self.analyze_thsad_recalc = thsad_recalc
        self.analyze_thscd = thscd

        return self

    def denoise(
        self,
        *,
        tr: int = 2,
        func: _DenoiseFuncTr | VSFunctionKwArgs = partial(DFTTest().denoise, sigma=8),
        mode: NoiseProcessMode = NoiseProcessMode.IDENTIFY,
        deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
        mc_denoise: bool = True,
        stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
        func_comp_args: KwargsT | None = None,
        stabilize_comp_args: KwargsT | None = None,
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
        self.denoise_func = func
        self.denoise_mode = mode
        self.denoise_deint = deint
        self.denoise_mc_denoise = mc_denoise
        self.denoise_stabilize: tuple[float, float] | Literal[False] = stabilize
        self.denoise_func_comp_args = fallback(func_comp_args, KwargsT())
        self.denoise_stabilize_comp_args = fallback(stabilize_comp_args, KwargsT())

        return self

    def basic(
        self,
        *,
        tr: int = 2,
        thsad: int | tuple[int, int] = 640,
        bobber: BobberLike = NNEDI3(nsize=1),
        noise_restore: float = 0.0,
        degrain_args: KwargsT | None = None,
        mask_args: KwargsT | None | Literal[False] = None,
        mask_shimmer_args: KwargsT | None = KwargsT(erosion_distance=0),
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
        self.basic_bobber.kwargs.update(tff=self.tff, double_rate=self.double_rate)

        self.basic_noise_restore = noise_restore
        self.basic_degrain_args = fallback(degrain_args, KwargsT())
        self.basic_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())
        self.basic_mask_args: KwargsT | Literal[False] = fallback(mask_args, KwargsT())

        return self

    def source_match(
        self,
        *,
        tr: int = 1,
        bobber: BobberLike | None = None,
        mode: SourceMatchMode = SourceMatchMode.NONE,
        similarity: float = 0.5,
        enhance: float = 0.5,
        degrain_args: KwargsT | None = None,
    ) -> Self:
        """
        Configure parameters for the source_match stage.

        Args:
            tr: Temporal radius of the refinement motion compensated binomial smooth.
            bobber: Bobber to use for refined spatial interpolation.
            mode: Specifies number of refinement steps to perform.
            similarity: Temporal similarity of the error created by smoothing.
            enhance: Sharpening strength prior to source match refinement.
            degrain_args: Arguments passed to the binomial_degrain call.
        """

        self.match_tr = tr

        if bobber is None:
            self.match_bobber = deepcopy(self.basic_bobber)
        elif isinstance(bobber, Bobber):
            self.match_bobber = deepcopy(bobber)
        else:
            self.match_bobber = Bobber.ensure_obj(bobber, self.__class__)

        self.match_bobber.kwargs.update(tff=self.tff, double_rate=self.double_rate)

        self.match_mode = mode
        self.match_similarity = similarity
        self.match_enhance = enhance
        self.match_degrain_args = fallback(degrain_args, KwargsT())

        return self

    def lossless(
        self,
        *,
        mode: LosslessMode = LosslessMode.NONE,
    ) -> Self:
        """
        Configure parameter for the lossless stage.

        Args:
            mode: Specifies at which stage to re-weave the original fields.
        """

        self.lossless_mode = mode

        return self

    def sharpen(
        self,
        *,
        mode: SharpMode | None = None,
        strength: tuple[int, float] = (1, 1.0),
        clamp: int | float | tuple[int | float, int | float] = 1,
        thin: float = 0.0,
    ) -> Self:
        """
        Configure parameters for the sharpen stage.

        Args:
            mode: Specifies the type of sharpening to use.
            strength: Sharpening radius and strength.
            clamp: Clamp the sharpening strength of
                [SharpMode.UNSHARP_MINMAX][vsdeinterlace.qtgmc.QTempGaussMC.SharpMode.UNSHARP_MINMAX] to the min/max
                average plus/minus this.
            thin: How much to vertically thin edges.
        """

        self.sharp_mode = fallback(mode, self.SharpMode.NONE if self.match_mode else self.SharpMode.UNSHARP_MINMAX)
        self.sharp_strength = strength
        self.sharp_clamp = normalize_seq(clamp, 2)
        self.sharp_thin = thin

        return self

    def back_blend(
        self,
        *,
        mode: BackBlendMode = BackBlendMode.BOTH,
        sigma: float = 1.4,
    ) -> Self:
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
        mode: SharpLimitMode | None = None,
        radius: int = 3,
        clamp: int | float | tuple[int | float, int | float] = 0,
        comp_args: KwargsT | None = None,
    ) -> Self:
        """
        Configure parameters for the sharpen_limit stage.

        Args:
            mode: Specifies type of limiting & at which stage to perform it.
            radius: Radius of sharpness limiting.
            clamp: How much undershoot/overshoot to allow.
            comp_args: Arguments passed to [MVTools.compensate][vsdenoise.mvtools.mvtools.MVTools.compensate] for
                temporal limiting.
        """

        self.limit_mode = fallback(
            mode, self.SharpLimitMode.NONE if self.match_mode else self.SharpLimitMode.TEMPORAL_PRESMOOTH
        )
        self.limit_radius = radius
        self.limit_clamp = clamp
        self.limit_comp_args = fallback(comp_args, KwargsT())

        return self

    def final(
        self,
        *,
        tr: int = 3,
        thsad: int | tuple[int, int] = 256,
        noise_restore: float = 0.0,
        degrain_args: KwargsT | None = None,
        mask_shimmer_args: KwargsT | None = None,
    ) -> Self:
        """
        Configure parameters for the final stage.

        Args:
            tr: Temporal radius of the motion compensated smooth.
            thsad: Thsad of the motion compensated smooth.
            noise_restore: How much noise to restore after this stage.
            degrain_args: Arguments passed to [MVTools.degrain][vsdenoise.mvtools.mvtools.MVTools.degrain].
            mask_shimmer_args: mask_shimmer_args: Arguments passed to the mask_shimmer call.
        """

        self.final_tr = tr
        self.final_thsad = thsad
        self.final_noise_restore = noise_restore
        self.final_degrain_args = fallback(degrain_args, KwargsT())
        self.final_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())

        return self

    def motion_blur(
        self,
        *,
        shutter_angle: tuple[int | float, int | float] = (180, 180),
        fps_divisor: int = 1,
        blur_args: KwargsT | None = None,
        mask_args: KwargsT | None | Literal[False] = KwargsT(ml=4),
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
        self.motion_blur_args = fallback(blur_args, KwargsT())
        self.motion_blur_mask_args: KwargsT | Literal[False] = fallback(mask_args, KwargsT())

        return self

    def _mask_shimmer(
        self,
        flt: vs.VideoNode,
        src: vs.VideoNode,
        threshold: int | float = 1,
        erosion_distance: int = 4,
        over_dilation: int = 0,
    ) -> vs.VideoNode:
        """
        Compare processed clip with reference clip,
        only allow thin, horizontal areas of difference, i.e. bob shimmer fixes.

        Args:
            flt: Processed clip to perform masking on.
            src: Unprocessed clip to restore from.
            threshold: Threshold of change to perform masking.
            erosion_distance: How much to deflate then reflate to remove thin areas.
            over_dilation: Extra inflation to ensure areas to restore back are fully caught.
        """
        if not erosion_distance:
            return flt

        iterations = (1 + (erosion_distance + 1) // 3, 1 + (erosion_distance + 2) // 3)
        over_dilations = (over_dilation // 3, over_dilation % 3)

        diff = src.std.MakeDiff(flt)

        closing = Morpho.maximum(diff, iterations=iterations[0], coords=Coordinates.VERTICAL, func=self._mask_shimmer)
        opening = Morpho.minimum(diff, iterations=iterations[0], coords=Coordinates.VERTICAL, func=self._mask_shimmer)

        if erosion_residual := erosion_distance % 3:
            closing = Morpho.inflate(closing, func=self._mask_shimmer)
            opening = Morpho.deflate(opening, func=self._mask_shimmer)

            if erosion_residual == 2:
                closing = median_blur(closing, func=self._mask_shimmer)
                opening = median_blur(opening, func=self._mask_shimmer)

        closing = Morpho.minimum(
            closing, iterations=iterations[1], coords=Coordinates.VERTICAL, func=self._mask_shimmer
        )
        opening = Morpho.maximum(
            opening, iterations=iterations[1], coords=Coordinates.VERTICAL, func=self._mask_shimmer
        )

        if over_dilation:
            closing = Morpho.minimum(closing, iterations=over_dilations[0], func=self._mask_shimmer)
            opening = Morpho.maximum(opening, iterations=over_dilations[0], func=self._mask_shimmer)

            closing = Morpho.deflate(closing, iterations=over_dilations[1], func=self._mask_shimmer)
            opening = Morpho.inflate(opening, iterations=over_dilations[1], func=self._mask_shimmer)

        return norm_expr(
            [flt, diff, closing, opening],
            "x y neutral - abs {thr} > y z neutral min a neutral max clip y ? neutral - +",
            thr=scale_delta(threshold, 8, flt),
            func=self._mask_shimmer,
        )

    def _interpolate(self, clip: vs.VideoNode, bobber: Bobber) -> vs.VideoNode:
        if self.input_type != self.InputType.PROGRESSIVE:
            clip = bobber.deinterlace(clip)

        return clip

    def _binomial_degrain(self, clip: vs.VideoNode, tr: int) -> vs.VideoNode:
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

        backward, forward = self.mv.get_vectors(tr=tr)
        vectors = MotionVectors()
        degrained = list[vs.VideoNode]()

        for delta in range(tr):
            vectors.set_vector(backward[delta], MVDirection.BACKWARD, 1)
            vectors.set_vector(forward[delta], MVDirection.FORWARD, 1)
            vectors.tr = 1

            degrained.append(
                self.mv.degrain(
                    clip, vectors=vectors, thsad=self.basic_thsad, thscd=self.analyze_thscd, **self.basic_degrain_args
                )
            )
            vectors.clear()

        return BlurMatrix.custom(_get_weights(tr), ConvMode.TEMPORAL)([clip, *degrained], func=self._binomial_degrain)

    def _apply_prefilter(self) -> None:
        self.draft = Catrom(tff=self.tff).bob(self.clip) if self.input_type == self.InputType.INTERLACE else self.clip

        search = self.draft

        if not self.analyze_chroma:
            search = get_y(search)

        if self.input_type == self.InputType.REPAIR:
            search = BlurMatrix.BINOMIAL()(search, mode=ConvMode.VERTICAL, func=self._apply_prefilter)

        if self.prefilter_tr:
            scenechange = self.prefilter_sc_threshold is not False

            scenes = sc_detect(search, self.prefilter_sc_threshold) if scenechange else search
            smoothed = BlurMatrix.BINOMIAL(self.prefilter_tr, mode=ConvMode.TEMPORAL)(
                scenes, scenechange=scenechange, func=self._apply_prefilter
            )
            smoothed = self._mask_shimmer(smoothed, search, **self.prefilter_mask_shimmer_args)
        else:
            smoothed = search

        if self.prefilter_postprocess:
            gauss_sigma, blend_weight = self.prefilter_blur_strength

            blurred = core.std.Merge(gauss_blur(smoothed, gauss_sigma), smoothed, blend_weight)

            if self.prefilter_postprocess == self.SearchPostProcess.GAUSSBLUR_EDGESOFTEN:
                lim1, lim2, lim3 = [scale_delta(thr, 8, self.clip) for thr in self.prefilter_soften_limit]

                blurred = norm_expr(
                    [blurred, smoothed, search],
                    "z y {lim1} - y {lim1} + clip TWEAK! "
                    "x {lim2} + TWEAK@ < x {lim3} + x {lim2} - TWEAK@ > x {lim3} - x 0.51 * TWEAK@ 0.49 * + ? ?",
                    lim1=lim1,
                    lim2=lim2,
                    lim3=lim3,
                    func=self._apply_prefilter,
                )
        else:
            blurred = smoothed

        if self.prefilter_range_expansion_args is not False:
            blurred = prefilter_to_full_range(
                blurred, func=self._apply_prefilter, **self.prefilter_range_expansion_args
            )

        self.prefilter_output = blurred

    def _apply_analyze(self) -> None:
        tr = max(
            self.analyze_tr,
            self.denoise_tr,
            self.basic_tr,
            self.match_tr,
            self.limit_radius
            if self.limit_mode in (self.SharpLimitMode.TEMPORAL_PRESMOOTH, self.SharpLimitMode.TEMPORAL_POSTSMOOTH)
            else 1,
            self.final_tr,
        )

        blksize, overlap = self.analyze_blksize, self.analyze_overlap
        thsad_recalc = fallback(
            self.analyze_thsad_recalc,
            round((self.basic_thsad[0] if isinstance(self.basic_thsad, tuple) else self.basic_thsad) / 2),
        )

        self.mv = MVTools(self.draft, self.prefilter_output, chroma=self.analyze_chroma, **self.analyze_preset)
        self.mv.analyze(tr=tr, blksize=blksize, overlap=refine_blksize(blksize, overlap))

        for _ in range(self.analyze_refine):
            blksize = refine_blksize(blksize)
            self.mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=refine_blksize(blksize, overlap))

    def _apply_denoise(self) -> None:
        self.denoise_output = self.clip

        if self.denoise_mode == self.NoiseProcessMode.NONE:
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

        if self.basic_noise_restore == self.final_noise_restore == 0.0:
            return

        if self.input_type == self.InputType.INTERLACE:
            match self.denoise_deint:
                case self.NoiseDeintMode.WEAVE:
                    self.noise = self.noise.std.SeparateFields(self.tff).std.DoubleWeave(self.tff)
                case self.NoiseDeintMode.BOB:
                    self.noise = Catrom(tff=self.tff).bob(self.noise)
                case self.NoiseDeintMode.GENERATE:
                    noise_source = self.noise.std.SeparateFields(self.tff)

                    noise_max = Morpho.maximum(
                        Morpho.maximum(noise_source), coords=Coordinates.HORIZONTAL, func=self._apply_denoise
                    )
                    noise_min = Morpho.minimum(
                        Morpho.minimum(noise_source), coords=Coordinates.HORIZONTAL, func=self._apply_denoise
                    )

                    noise_new = Grainer.GAUSS(
                        noise_source, 2048, protect_edges=False, protect_neutral_chroma=False, neutral_out=True
                    )
                    noise_new = norm_expr(
                        [noise_max, noise_min, noise_new], "x y - z * range_size / y +", func=self._apply_denoise
                    )

                    self.noise = reweave(noise_source, noise_new, self.tff, self._apply_denoise)

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
        self.bobbed = self._interpolate(self.denoise_output, self.basic_bobber)

        if self.input_type == self.InputType.REPAIR and self.basic_mask_args is not False:
            mask = self.mv.mask(
                self.prefilter_output,
                direction=MVDirection.BACKWARD,
                kind=MaskMode.SAD,
                thscd=self.analyze_thscd,
                **self.basic_mask_args,
            )
            self.bobbed = self.denoise_output.std.MaskedMerge(self.bobbed, mask)

        smoothed = self._binomial_degrain(self.bobbed, self.basic_tr)
        if self.basic_tr:
            smoothed = self._mask_shimmer(smoothed, self.bobbed, **self.basic_mask_shimmer_args)

        if self.match_mode:
            smoothed = self._apply_source_match(smoothed, self.denoise_output)

        if self.lossless_mode == self.LosslessMode.PRESHARPEN:
            smoothed = self._apply_lossless(smoothed)

        resharp = self._apply_sharpen(smoothed)

        if self.backblend_mode in (self.BackBlendMode.PRELIMIT, self.BackBlendMode.BOTH):
            resharp = self._apply_back_blend(resharp, smoothed)

        if self.limit_mode in (self.SharpLimitMode.SPATIAL_PRESMOOTH, self.SharpLimitMode.TEMPORAL_PRESMOOTH):
            resharp = self._apply_sharpen_limit(resharp)

        if self.backblend_mode in (self.BackBlendMode.POSTLIMIT, self.BackBlendMode.BOTH):
            resharp = self._apply_back_blend(resharp, smoothed)

        self.basic_output = self._apply_noise_restore(resharp, self.basic_noise_restore)

    def _apply_source_match(self, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
        def _error_adjustment(clip: vs.VideoNode, ref: vs.VideoNode, tr: int) -> vs.VideoNode:
            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2**tr_f / (binomial_coeff + self.match_similarity * (2**tr_f - binomial_coeff))

            return norm_expr([clip, ref], "y 1 {adj} + * x {adj} * -", adj=error_adj, func=_error_adjustment)

        if self.input_type == self.InputType.INTERLACE:
            clip = reinterlace(clip, self.tff, self._apply_source_match)

        adjusted1 = _error_adjustment(clip, ref, self.basic_tr)
        bobbed1 = self._interpolate(adjusted1, self.basic_bobber)
        match1 = self._binomial_degrain(bobbed1, self.basic_tr)

        if self.match_mode > self.SourceMatchMode.BASIC:
            if self.match_enhance:
                match1 = unsharpen(match1, self.match_enhance, BlurMatrix.BINOMIAL(), func=self._apply_source_match)

            if self.input_type == self.InputType.INTERLACE:
                clip = reinterlace(match1, self.tff, self._apply_source_match)

            diff = ref.std.MakeDiff(clip)
            bobbed2 = self._interpolate(diff, self.match_bobber)
            match2 = self._binomial_degrain(bobbed2, self.match_tr)

            if self.match_mode == self.SourceMatchMode.TWICE_REFINED:
                adjusted2 = _error_adjustment(match2, bobbed2, self.match_tr)
                match2 = self._binomial_degrain(adjusted2, self.match_tr)

            return match1.std.MergeDiff(match2)

        return match1

    def _apply_lossless(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.input_type == self.InputType.PROGRESSIVE:
            return clip

        fields_src = self.denoise_output.std.SeparateFields(self.tff)

        if self.input_type == self.InputType.REPAIR:
            fields_src = core.std.SelectEvery(fields_src, 4, (0, 3))

        fields_flt = clip.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = reweave(fields_src, fields_flt, self.tff, self._apply_lossless)

        median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL, func=self._apply_lossless))
        fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        processed_diff = norm_expr(
            [fields_diff, median_blur(fields_diff, mode=ConvMode.VERTICAL, func=self._apply_lossless)],
            "x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?",
            func=self._apply_lossless,
        )
        processed_diff = repair.Mode.MINMAX_SQUARE1(processed_diff, remove_grain.Mode.MINMAX_AROUND2(processed_diff))

        out = reweave(fields_src, fields_flt.std.MakeDiff(processed_diff), self.tff, self._apply_lossless)

        return core.std.SetFieldBased(out, FieldBased.PROGRESSIVE)

    def _apply_sharpen(self, clip: vs.VideoNode) -> vs.VideoNode:
        match self.sharp_mode:
            case self.SharpMode.NONE:
                resharp = clip
            case self.SharpMode.UNSHARP:
                resharp = unsharpen(
                    clip, self.sharp_strength[1], BlurMatrix.BINOMIAL(self.sharp_strength[0]), func=self._apply_sharpen
                )
            case self.SharpMode.UNSHARP_MINMAX:
                undershoot, overshoot = self.sharp_clamp

                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL, func=self._apply_sharpen)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL, func=self._apply_sharpen)

                clamp = norm_expr(
                    [clip, source_min, source_max],
                    "y z + 2 / x {undershoot} - x {overshoot} + clip",
                    undershoot=scale_delta(undershoot, 8, clip),
                    overshoot=scale_delta(overshoot, 8, clip),
                    func=self._apply_sharpen,
                )
                resharp = unsharpen(
                    clip,
                    self.sharp_strength[1],
                    BlurMatrix.BINOMIAL(self.sharp_strength[0])(clamp, func=self._apply_sharpen),
                    func=self._apply_sharpen,
                )

        if self.sharp_thin:
            median_diff = norm_expr(
                [clip, median_blur(clip, mode=ConvMode.VERTICAL)],
                "y x - {thin} * neutral +",
                thin=self.sharp_thin,
                func=self._apply_sharpen,
            )
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff, func=self._apply_sharpen)

            resharp = norm_expr(
                [resharp, blurred_diff, BlurMatrix.BINOMIAL()(blurred_diff)],
                "y neutral - dup abs z neutral - abs < swap x + x ?",
                func=self._apply_sharpen,
            )

        return resharp

    def _apply_back_blend(self, flt: vs.VideoNode, src: vs.VideoNode) -> vs.VideoNode:
        if self.backblend_sigma and (self.sharp_mode or self.sharp_thin):
            flt = flt.std.MakeDiff(gauss_blur(flt.std.MakeDiff(src), self.backblend_sigma))

        return flt

    def _apply_sharpen_limit(self, clip: vs.VideoNode) -> vs.VideoNode:
        if (self.sharp_mode or self.sharp_thin) and self.limit_radius:
            if self.limit_mode in (self.SharpLimitMode.SPATIAL_PRESMOOTH, self.SharpLimitMode.SPATIAL_POSTSMOOTH):
                if self.limit_radius == 1:
                    clip = repair.Mode.MINMAX_SQUARE1(clip, self.bobbed)
                else:
                    inpand = Morpho.minimum(self.bobbed, iterations=self.limit_radius, func=self._apply_sharpen_limit)
                    expand = Morpho.maximum(self.bobbed, iterations=self.limit_radius, func=self._apply_sharpen_limit)
                    clip = norm_expr([clip, inpand, expand], "x y z clip", func=self._apply_sharpen_limit)

            elif self.limit_mode in (self.SharpLimitMode.TEMPORAL_PRESMOOTH, self.SharpLimitMode.TEMPORAL_POSTSMOOTH):
                clip = mc_clamp(
                    clip,
                    self.bobbed,
                    self.mv,
                    self.limit_clamp,
                    self._apply_sharpen_limit,
                    tr=self.limit_radius,
                    thscd=self.analyze_thscd,
                    **self.limit_comp_args,
                )

        return clip

    def _apply_noise_restore(self, clip: vs.VideoNode, restore: float = 0.0) -> vs.VideoNode:
        if restore and hasattr(self, "noise"):
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

        if self.limit_mode in (self.SharpLimitMode.SPATIAL_POSTSMOOTH, self.SharpLimitMode.TEMPORAL_POSTSMOOTH):
            smoothed = self._apply_sharpen_limit(smoothed)

        if self.lossless_mode == self.LosslessMode.POSTSMOOTH:
            smoothed = self._apply_lossless(smoothed)

        self.final_output = self._apply_noise_restore(smoothed, self.final_noise_restore)

    def _apply_motion_blur(self) -> None:
        angle_in, angle_out = self.motion_blur_shutter_angle

        if angle_out * self.motion_blur_fps_divisor != angle_in:
            blur_level = (angle_out * self.motion_blur_fps_divisor - angle_in) * 100 / 360

            processed = self.mv.flow_blur(
                self.final_output, blur=blur_level, thscd=self.analyze_thscd, **self.motion_blur_args
            )

            if self.motion_blur_mask_args is not False:
                mask = self.mv.mask(
                    self.prefilter_output,
                    direction=MVDirection.BACKWARD,
                    kind=MaskMode.MOTION,
                    thscd=self.analyze_thscd,
                    **self.motion_blur_mask_args,
                )

                processed = self.final_output.std.MaskedMerge(processed, mask)
        else:
            processed = self.final_output

        if self.motion_blur_fps_divisor > 1:
            processed = processed[:: self.motion_blur_fps_divisor]

        self.motion_blur_output = processed

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
