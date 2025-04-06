from functools import partial
from math import factorial
from typing import Any, Literal, MutableMapping, Protocol

from numpy import linalg, zeros
from typing_extensions import Self

from vsaa import Nnedi3
from vsaa.abstract import _Antialiaser
from vsdeband import AddNoise
from vsdenoise import (
    DFTTest, MaskMode, MotionVectors, MVDirection, MVTools, MVToolsPreset, MVToolsPresets, prefilter_to_full_range
)
from vsexprtools import norm_expr
from vsmasktools import Coordinates, Morpho
from vsrgtools import (
    BlurMatrix, MeanMode, RemoveGrainMode, RepairMode, gauss_blur, median_blur, remove_grain, repair, unsharpen
)
from vstools import (
    ConstantFormatVideoNode, ConvMode, CustomRuntimeError, FieldBased, FieldBasedT, KwargsT, VSFunctionKwArgs,
    VSFunctionNoArgs, check_variable, core, fallback, scale_delta, vs, vs_object
)

from .enums import (
    BackBlendMode, InputType, LosslessMode, NoiseDeintMode, NoiseProcessMode, SearchPostProcess, SharpLimitMode,
    SharpMode, SourceMatchMode
)
from .utils import reinterlace, scdetect

__all__ = [
    'QTempGaussMC'
]


class _DenoiseFuncTr(Protocol):
    def __call__(self, clip: vs.VideoNode, /, *, tr: int = ...) -> vs.VideoNode:
        ...


class QTempGaussMC(vs_object):
    """
    Quasi Temporal Gaussian Motion Compensated (QTGMC)
    """

    clip: ConstantFormatVideoNode

    draft: ConstantFormatVideoNode

    bobbed: ConstantFormatVideoNode

    noise: ConstantFormatVideoNode | None

    prefilter_output: ConstantFormatVideoNode

    denoise_output: ConstantFormatVideoNode

    basic_output: ConstantFormatVideoNode

    final_output: ConstantFormatVideoNode

    motion_blur_output: ConstantFormatVideoNode

    def __init__(
        self,
        clip: vs.VideoNode,
        input_type: InputType = InputType.INTERLACE,
        tff: FieldBasedT | bool | None = None,
    ) -> None:
        assert check_variable(clip, self.__class__)

        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        self.clip = clip
        self.input_type = input_type
        self.tff = clip_fieldbased.is_tff
        self.field = clip_fieldbased.field + 2

        if self.input_type == InputType.PROGRESSIVE and clip_fieldbased.is_inter:
            raise CustomRuntimeError(f"{self.input_type} incompatible with interlaced video!", self.__class__)

    def prefilter(
        self,
        *,
        tr: int = 2,
        sc_threshold: float | None | Literal[False] = None,
        postprocess: SearchPostProcess = SearchPostProcess.GAUSSBLUR_EDGESOFTEN,
        strength: tuple[float, float] = (1.9, 0.1),
        limit: tuple[int | float, int | float, int | float] = (3, 7, 2),
        range_conversion_args: KwargsT | None | Literal[False] = KwargsT(range_conversion=2.0),
        mask_shimmer_args: KwargsT | None = None,
    ) -> Self:
        self.prefilter_tr = tr
        self.prefilter_sc_threshold = sc_threshold
        self.prefilter_postprocess = postprocess
        self.prefilter_blur_strength = strength
        self.prefilter_soften_limit = limit
        self.prefilter_range_conversion_args = fallback(range_conversion_args, KwargsT())
        self.prefilter_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())

        return self

    def denoise(
        self,
        *,
        tr: int = 2,
        func: _DenoiseFuncTr | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] = partial(DFTTest.denoise, sigma=2),
        mode: NoiseProcessMode = NoiseProcessMode.IDENTIFY,
        deint: NoiseDeintMode = NoiseDeintMode.GENERATE,
        stabilize: tuple[float, float] | Literal[False] = (0.6, 0.2),
        func_comp_args: KwargsT | None = None,
        stabilize_comp_args: KwargsT | None = None,
    ) -> Self:
        self.denoise_tr = tr
        self.denoise_func = func
        self.denoise_mode = mode
        self.denoise_deint = deint
        self.denoise_stabilize: tuple[float, float] | Literal[False] = stabilize
        self.denoise_func_comp_args = fallback(func_comp_args, KwargsT())
        self.denoise_stabilize_comp_args = fallback(stabilize_comp_args, KwargsT())

        return self

    def basic(
        self,
        *,
        tr: int = 2,
        bobber: _Antialiaser | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] = Nnedi3(qual=2, nsize=0, nns=4, pscrn=1),
        noise_restore: float = 0,
        degrain_args: KwargsT | None = KwargsT(thsad=640),
        mask_shimmer_args: KwargsT | None = KwargsT(erosion_distance=0),
    ) -> Self:
        self.basic_tr = tr

        if isinstance(bobber, _Antialiaser):
            bobber = bobber.copy()
            bobber.field = self.field

            def _bobber_func(clip: vs.VideoNode) -> vs.VideoNode:
                return bobber.interpolate(clip, double_y=False, **bobber.get_aa_args(clip))

            self.basic_bobber = _bobber_func
        else:
            self.basic_bobber = bobber

        self.basic_noise_restore = noise_restore
        self.basic_degrain_args = fallback(degrain_args, KwargsT())
        self.basic_mask_shimmer_args = fallback(mask_shimmer_args, KwargsT())

        return self

    def source_match(
        self,
        *,
        tr: int = 1,
        bobber: _Antialiaser | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
        mode: SourceMatchMode = SourceMatchMode.NONE,
        similarity: float = 0.5,
        enhance: float = 0.5,
        degrain_args: KwargsT | None = None,
    ) -> Self:
        self.match_tr = tr

        if isinstance(bobber, _Antialiaser):
            bobber = bobber.copy()
            bobber.field = self.field

            def _bobber_func(clip: vs.VideoNode) -> vs.VideoNode:
                return bobber.interpolate(clip, double_y=False, **bobber.get_aa_args(clip))

            self.match_bobber = _bobber_func
        elif bobber:
            self.match_bobber = bobber
        else:
            self.match_bobber = self.basic_bobber

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
        self.lossless_mode = mode

        return self

    def sharpen(
        self,
        *,
        mode: SharpMode | None = None,
        strength: float = 1.0,
        clamp: int | float = 1,
        thin: float = 0.0,
    ) -> Self:
        if mode is None:
            self.sharp_mode = SharpMode.NONE if self.match_mode else SharpMode.UNSHARP_MINMAX
        else:
            self.sharp_mode = mode

        self.sharp_strength = strength
        self.sharp_clamp = clamp
        self.sharp_thin = thin

        return self

    def back_blend(
        self,
        *,
        mode: BackBlendMode = BackBlendMode.BOTH,
        sigma: float = 1.4,
    ) -> Self:
        self.backblend_mode = mode
        self.backblend_sigma = sigma

        return self

    def sharpen_limit(
        self,
        *,
        mode: SharpLimitMode | None = None,
        radius: int = 3,
        overshoot: int | float = 0,
        comp_args: KwargsT | None = None,
    ) -> Self:
        if mode is None:
            self.limit_mode = SharpLimitMode.NONE if self.match_mode else SharpLimitMode.TEMPORAL_PRESMOOTH
        else:
            self.limit_mode = mode

        self.limit_radius = radius
        self.limit_overshoot = overshoot
        self.limit_comp_args = fallback(comp_args, KwargsT())

        return self

    def final(
        self,
        *,
        tr: int = 3,
        noise_restore: float = 0.0,
        degrain_args: KwargsT | None = KwargsT(thsad=256),
        mask_shimmer_args: KwargsT | None = None,
    ) -> Self:
        self.final_tr = tr
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
        self.motion_blur_shutter_angle = shutter_angle
        self.motion_blur_fps_divisor = fps_divisor
        self.motion_blur_args = fallback(blur_args, KwargsT())
        self.motion_blur_mask_args: KwargsT | Literal[False] = fallback(mask_args, KwargsT())

        return self

    def mask_shimmer(
        self,
        flt: vs.VideoNode,
        src: vs.VideoNode,
        threshold: float | int = 1,
        erosion_distance: int = 4,
        over_dilation: int = 0,
    ) -> ConstantFormatVideoNode:
        assert check_variable(flt, self.mask_shimmer)

        if not erosion_distance:
            return flt

        iter1 = 1 + (erosion_distance + 1) // 3
        iter2 = 1 + (erosion_distance + 2) // 3

        over1 = over_dilation // 3
        over2 = over_dilation % 3

        diff = src.std.MakeDiff(flt)

        opening = Morpho.minimum(diff, iterations=iter1, coords=Coordinates.VERTICAL)
        if erosion_distance % 3:
            opening = Morpho.deflate(opening)
            if erosion_distance % 3 == 2:
                opening = median_blur(opening)
        opening = Morpho.maximum(opening, iterations=iter2, coords=Coordinates.VERTICAL)

        closing = Morpho.maximum(diff, iterations=iter1, coords=Coordinates.VERTICAL)
        if erosion_distance % 3:
            closing = Morpho.inflate(closing)
            if erosion_distance % 3 == 2:
                closing = median_blur(closing)
        closing = Morpho.minimum(closing, iterations=iter2, coords=Coordinates.VERTICAL)

        if over_dilation:
            opening = Morpho.maximum(opening, iterations=over1)
            opening = Morpho.inflate(opening, iterations=over2)

            closing = Morpho.minimum(closing, iterations=over1)
            closing = Morpho.deflate(closing, iterations=over2)

        return norm_expr(
            [flt, diff, opening, closing],
            'y neutral - abs {thr} > y a neutral min z neutral max clip y ? neutral - x +',
            thr=scale_delta(threshold, 8, flt)
        )

    def binomial_degrain(self, clip: vs.VideoNode, tr: int, **kwargs: Any) -> ConstantFormatVideoNode:
        def _get_weights(n: int) -> list[int]:
            k, rhs = 1, []
            mat = zeros((n + 1, n + 1))

            for i in range(1, n + 2):
                mat[n + 1 - i, i - 1] = mat[n, i - 1] = 1 / 3
                rhs.append(k)
                k = k * (2 * n + 1 - i) // i

            mat[n, 0] = 1

            return list(linalg.solve(mat, rhs))

        assert check_variable(clip, self.binomial_degrain)

        if not tr:
            return clip

        backward, forward = self.mv.get_vectors(tr=tr)
        vectors = MotionVectors()
        degrained = list[ConstantFormatVideoNode]()

        for delta in range(tr):
            vectors.set_vector(backward[delta], MVDirection.BACKWARD, 1)
            vectors.set_vector(forward[delta], MVDirection.FORWARD, 1)
            vectors.tr = 1

            degrained.append(self.mv.degrain(clip, tr=1, vectors=vectors, **kwargs))  # type: ignore
            vectors.clear()

        return core.std.AverageFrames([clip, *degrained], _get_weights(tr))

    def apply_prefilter(self) -> None:
        if self.input_type == InputType.REPAIR:
            search = BlurMatrix.BINOMIAL()(self.draft, mode=ConvMode.VERTICAL)
        else:
            search = self.draft

        if self.prefilter_tr:
            scenechange = self.prefilter_sc_threshold is not False

            scenes = scdetect(search, self.prefilter_sc_threshold) if scenechange else search
            smoothed = BlurMatrix.BINOMIAL(self.prefilter_tr, mode=ConvMode.TEMPORAL, scenechange=scenechange)(scenes)
            smoothed = self.mask_shimmer(smoothed, search, **self.prefilter_mask_shimmer_args)
        else:
            smoothed = search

        if self.prefilter_postprocess:
            gauss_sigma, blend_weight = self.prefilter_blur_strength

            blurred = core.std.Merge(gauss_blur(smoothed, gauss_sigma), smoothed, blend_weight)

            if self.prefilter_postprocess == SearchPostProcess.GAUSSBLUR_EDGESOFTEN:
                lim1, lim2, lim3 = [scale_delta(_, 8, self.clip) for _ in self.prefilter_soften_limit]

                blurred = norm_expr(
                    [blurred, smoothed, search],
                    'z y {lim1} - y {lim1} + clip TWEAK! '
                    'x {lim2} + TWEAK@ < x {lim3} + x {lim2} - TWEAK@ > x {lim3} - x 51 * y 49 * + 100 / ? ?',
                    lim1=lim1, lim2=lim2, lim3=lim3,
                )
        else:
            blurred = smoothed

        if self.prefilter_range_conversion_args is not False:
            blurred = prefilter_to_full_range(blurred, **self.prefilter_range_conversion_args)  # type: ignore

        self.prefilter_output = blurred

    def apply_denoise(self) -> None:
        if self.denoise_mode:
            if self.denoise_tr:
                denoised = self.mv.compensate(
                    self.draft, tr=self.denoise_tr,
                    temporal_func=lambda clip: self.denoise_func(clip, tr=self.denoise_tr),
                    **self.denoise_func_comp_args,
                )
            else:
                denoised = self.denoise_func(self.draft)

            if self.input_type == InputType.INTERLACE:
                denoised = reinterlace(denoised, self.tff)

            noise = self.clip.std.MakeDiff(denoised)

            if self.basic_noise_restore or self.final_noise_restore:
                if self.input_type == InputType.INTERLACE:
                    match self.denoise_deint:
                        case NoiseDeintMode.WEAVE:
                            noise = core.std.Interleave([noise] * 2)
                        case NoiseDeintMode.BOB:
                            noise = noise.resize.Bob(tff=self.tff)
                        case NoiseDeintMode.GENERATE:
                            noise_source = noise.std.SeparateFields(self.tff)

                            noise_max = Morpho.maximum(Morpho.maximum(noise_source), coords=Coordinates.HORIZONTAL)
                            noise_min = Morpho.minimum(Morpho.minimum(noise_source), coords=Coordinates.HORIZONTAL)

                            noise_new = AddNoise.GAUSS.grain(
                                noise_source, 2048, protect_chroma=False, fade_limits=False, neutral_out=True
                            )
                            noise_limit = norm_expr([noise_max, noise_min, noise_new], 'x y - z * range_size / y +')

                            noise = core.std.Interleave([noise_source, noise_limit]).std.DoubleWeave(self.tff)

                if self.denoise_stabilize:
                    weight1, weight2 = self.denoise_stabilize

                    noise_comp, _ = self.mv.compensate(
                        noise, direction=MVDirection.BACKWARD,
                        tr=1, interleave=False,
                        **self.denoise_stabilize_comp_args,
                    )

                    noise = norm_expr(
                        [noise, *noise_comp],
                        'x neutral - abs y neutral - abs > x y ? {weight1} * x y + {weight2} * +',
                        weight1=weight1, weight2=weight2,
                    )  # type: ignore

            self.noise = noise
            self.denoise_output = denoised if self.denoise_mode == NoiseProcessMode.DENOISE else self.clip  # type: ignore
        else:
            self.noise = None
            self.denoise_output = self.clip

    def apply_basic(self) -> None:
        smoothed = self.binomial_degrain(self.bobbed, tr=self.basic_tr, **self.basic_degrain_args)
        smoothed = self.mask_shimmer(smoothed, self.bobbed, **self.basic_mask_shimmer_args)

        if self.match_mode:
            smoothed = self.apply_source_match(smoothed)

        if self.lossless_mode == LosslessMode.PRESHARPEN and self.input_type != InputType.PROGRESSIVE:
            smoothed = self.apply_lossless(smoothed)

        resharp = self.apply_sharpen(smoothed)

        if self.backblend_mode in (BackBlendMode.PRELIMIT, BackBlendMode.BOTH):
            resharp = self.apply_back_blend(resharp, smoothed)

        if self.limit_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.TEMPORAL_PRESMOOTH):
            resharp = self.apply_sharpen_limit(resharp)

        if self.backblend_mode in (BackBlendMode.POSTLIMIT, BackBlendMode.BOTH):
            resharp = self.apply_back_blend(resharp, smoothed)

        self.basic_output = self.apply_noise_restore(resharp, self.basic_noise_restore)

    def apply_source_match(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        def _error_adjustment(clip: vs.VideoNode, ref: vs.VideoNode, tr: int) -> ConstantFormatVideoNode:
            tr_f = 2 * tr - 1
            binomial_coeff = factorial(tr_f) // factorial(tr) // factorial(tr_f - tr)
            error_adj = 2**tr_f / (binomial_coeff + self.match_similarity * (2**tr_f - binomial_coeff))

            return norm_expr([clip, ref], 'y {adj} 1 + * x {adj} * -', adj=error_adj)  # type: ignore

        if self.input_type != InputType.PROGRESSIVE:
            clip = reinterlace(clip, self.tff)

        adjusted1 = _error_adjustment(clip, self.denoise_output, self.basic_tr)
        bobbed1 = self.basic_bobber(adjusted1)
        match1 = self.binomial_degrain(bobbed1, self.basic_tr, **self.basic_degrain_args)

        if self.match_mode > SourceMatchMode.BASIC:
            if self.match_enhance:
                match1 = unsharpen(match1, self.match_enhance, BlurMatrix.BINOMIAL())

            if self.input_type != InputType.PROGRESSIVE:
                clip = reinterlace(match1, self.tff)

            diff = self.denoise_output.std.MakeDiff(clip)
            bobbed2 = self.match_bobber(diff)
            match2 = self.binomial_degrain(bobbed2, self.match_tr)

            if self.match_mode == SourceMatchMode.TWICE_REFINED:
                adjusted2 = _error_adjustment(match2, bobbed2, self.match_tr)
                match2 = self.binomial_degrain(adjusted2, self.match_tr)

            out = match1.std.MergeDiff(match2)
        else:
            out = match1

        return out

    def apply_lossless(self, flt: vs.VideoNode) -> ConstantFormatVideoNode:
        def _reweave(clipa: vs.VideoNode, clipb: vs.VideoNode) -> ConstantFormatVideoNode:
            return core.std.Interleave([clipa, clipb]).std.SelectEvery(4, (0, 1, 3, 2)).std.DoubleWeave(self.tff)[::2]

        fields_src = self.denoise_output.std.SeparateFields(self.tff)

        if self.input_type == InputType.REPAIR:
            fields_src = fields_src.std.SelectEvery(4, (0, 3))  # type: ignore

        fields_flt = flt.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        woven = _reweave(fields_src, fields_flt)

        median_diff = woven.std.MakeDiff(median_blur(woven, mode=ConvMode.VERTICAL))
        fields_diff = median_diff.std.SeparateFields(self.tff).std.SelectEvery(4, (1, 2))

        processed_diff = norm_expr(
            [fields_diff, median_blur(fields_diff, mode=ConvMode.VERTICAL)],
            'x neutral - X! y neutral - Y! X@ Y@ xor neutral X@ abs Y@ abs < x y ? ?',
        )
        processed_diff = repair(
            processed_diff, remove_grain(processed_diff, RemoveGrainMode.MINMAX_AROUND2), RepairMode.MINMAX_SQUARE1
        )

        return _reweave(fields_src, core.std.MakeDiff(fields_flt, processed_diff))

    def apply_sharpen(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.apply_sharpen)

        blur_kernel = BlurMatrix.BINOMIAL()

        match self.sharp_mode:
            case SharpMode.NONE:
                resharp = clip
            case SharpMode.UNSHARP:
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel)
            case SharpMode.UNSHARP_MINMAX:
                source_min = Morpho.minimum(clip, coords=Coordinates.VERTICAL)
                source_max = Morpho.maximum(clip, coords=Coordinates.VERTICAL)

                clamp = norm_expr(
                    [clip, source_min, source_max],
                    'y z + 2 / AVG! x AVG@ {thr} - AVG@ {thr} + clip',
                    thr=scale_delta(self.sharp_clamp, 8, clip),
                )
                resharp = unsharpen(clip, self.sharp_strength, blur_kernel(clamp))

        if self.sharp_thin:
            median_diff = norm_expr(
                [clip, median_blur(clip, mode=ConvMode.VERTICAL)], 'y x - {thin} * neutral +', thin=self.sharp_thin
            )
            blurred_diff = BlurMatrix.BINOMIAL(mode=ConvMode.HORIZONTAL)(median_diff)

            resharp = norm_expr(
                [resharp, blurred_diff, blur_kernel(blurred_diff)],
                'y neutral - Y! z neutral - Z! Y@ abs Z@ abs < Y@ 0 ? x +',
            )

        return resharp

    def apply_back_blend(self, flt: vs.VideoNode, src: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(flt, self.apply_back_blend)

        if self.backblend_sigma:
            flt = flt.std.MakeDiff(gauss_blur(flt.std.MakeDiff(src), self.backblend_sigma))

        return flt

    def apply_sharpen_limit(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.apply_sharpen_limit)

        if self.sharp_mode:
            if self.limit_mode in (SharpLimitMode.SPATIAL_PRESMOOTH, SharpLimitMode.SPATIAL_POSTSMOOTH):
                if self.limit_radius == 1:
                    clip = repair(clip, self.bobbed, RepairMode.MINMAX_SQUARE1)
                elif self.limit_radius > 1:
                    clip = repair(
                        clip, repair(clip, self.bobbed, RepairMode.MINMAX_SQUARE_REF2), RepairMode.MINMAX_SQUARE1
                    )

            if self.limit_mode in (SharpLimitMode.TEMPORAL_PRESMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
                backward_comp, forward_comp = self.mv.compensate(
                    self.bobbed, tr=self.limit_radius, interleave=False, **self.limit_comp_args
                )

                comp_min = MeanMode.MINIMUM([self.bobbed, *backward_comp, *forward_comp])
                comp_max = MeanMode.MAXIMUM([self.bobbed, *backward_comp, *forward_comp])

                clip = norm_expr(
                    [clip, comp_min, comp_max],
                    'x y {thr} - z {thr} + clip',
                    thr=scale_delta(self.limit_overshoot, 8, clip),
                )

        return clip

    def apply_noise_restore(self, clip: vs.VideoNode, restore: float = 0.0) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.apply_noise_restore)

        if restore and self.noise:
            clip = norm_expr([clip, self.noise], 'y neutral - {restore} * x +', restore=restore)

        return clip

    def apply_final(self) -> None:
        smoothed = self.mv.degrain(self.basic_output, tr=self.final_tr, **self.final_degrain_args)
        smoothed = self.mask_shimmer(smoothed, self.bobbed, **self.final_mask_shimmer_args)

        if self.limit_mode in (SharpLimitMode.SPATIAL_POSTSMOOTH, SharpLimitMode.TEMPORAL_POSTSMOOTH):
            smoothed = self.apply_sharpen_limit(smoothed)

        if self.lossless_mode == LosslessMode.POSTSMOOTH and self.input_type != InputType.PROGRESSIVE:
            smoothed = self.apply_lossless(smoothed)

        self.final_output = self.apply_noise_restore(smoothed, self.final_noise_restore)

    def apply_motion_blur(self) -> None:
        angle_in, angle_out = self.motion_blur_shutter_angle

        if not angle_out * self.motion_blur_fps_divisor == angle_in:
            blur_level = (angle_out * self.motion_blur_fps_divisor - angle_in) * 100 / 360

            processed = self.mv.flow_blur(self.final_output, blur=blur_level, **self.motion_blur_args)

            if self.motion_blur_mask_args is not False:
                mask = self.mv.mask(
                    self.prefilter_output, direction=MVDirection.BACKWARD, kind=MaskMode.MOTION, **self.motion_blur_mask_args
                )

                processed = self.final_output.std.MaskedMerge(processed, mask)
        else:
            processed = self.final_output

        if self.motion_blur_fps_divisor > 1:
            processed = processed[:: self.motion_blur_fps_divisor]

        self.motion_blur_output = processed  # type: ignore

    def process(
        self,
        *,
        force_tr: int = 1,
        preset: MVToolsPreset = MVToolsPresets.HQ_SAD,
        mask_args: KwargsT | None | Literal[False] = None,
    ) -> ConstantFormatVideoNode:
        mask_args_norm: KwargsT | Literal[False] = fallback(mask_args, KwargsT())

        self.draft = self.clip.resize.Bob(tff=self.tff) if self.input_type == InputType.INTERLACE else self.clip

        tr = max(force_tr, self.denoise_tr, self.basic_tr, self.match_tr, self.final_tr)

        if preset:
            preset.pop('search_clip', None)

        self.apply_prefilter()

        self.mv = MVTools(self.draft, self.prefilter_output, **preset)
        self.mv.analyze(tr=tr)

        self.apply_denoise()

        if self.input_type == InputType.REPAIR:
            self.denoise_output = reinterlace(self.denoise_output, self.tff)  # type: ignore

        self.bobbed = self.basic_bobber(self.denoise_output)  # type: ignore

        if mask_args_norm is not False and self.input_type == InputType.REPAIR:
            mask = self.mv.mask(self.prefilter_output, direction=MVDirection.BACKWARD, kind=MaskMode.SAD, **mask_args_norm)
            self.bobbed = self.denoise_output.std.MaskedMerge(self.bobbed, mask)

        self.apply_basic()
        self.apply_final()
        self.apply_motion_blur()

        return self.motion_blur_output

    def __vs_del__(self, core_id: int) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, MutableMapping):
                v.clear()
            if isinstance(v, vs.VideoNode):
                setattr(self, k, None)
