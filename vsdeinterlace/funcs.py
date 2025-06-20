from __future__ import annotations

from functools import partial
from typing import Sequence

from jetpytools import CustomIntEnum, KwargsT

from vsdenoise import MVTools, MVToolsPreset, prefilter_to_full_range
from vsexprtools import norm_expr
from vsrgtools import BlurMatrix, sbr
from vstools import (
    ConvMode, ConstantFormatVideoNode, FormatsMismatchError, FunctionUtil, VSFunctionKwArgs, PlanesT,
    check_ref_clip, check_variable, core, limiter, scale_delta, shift_clip, vs,
)

from .enums import IVTCycles

__all__ = [
    'InterpolateOverlay',
    'FixInterlacedFades',
    'vinverse'
]


class InterpolateOverlay(CustomIntEnum):
    IVTC_TXT60 = 0
    """For 60i overlaid ontop 24t."""

    DEC_TXT60 = 1
    """For 60i overlaid ontop 24d."""

    IVTC_TXT30 = 2
    """For 30p overlaid ontop 24t."""

    def __call__(
        self,
        clip: vs.VideoNode,
        bobbed: vs.VideoNode,
        pattern: int,
        preset: MVToolsPreset = MVToolsPreset.HQ_COHERENCE,
        blksize: int | tuple[int, int] = 8,
        refine: int = 1,
        thsad_recalc: int | None = None,
    ) -> ConstantFormatVideoNode:
        """
        Virtually oversamples the video to 120 fps with motion interpolation on credits only, and decimates to 24 fps.
        Requires manually specifying the 3:2 pulldown pattern (the clip must be split into parts if it changes).

        :param clip:             Original interlaced clip.
        :param bobbed:           Bob-deinterlaced clip.
        :param pattern:          First frame of any clean-combed-combed-clean-clean sequence.
        :param preset:           MVTools preset defining base values for the MVTools object. Default is HQ_COHERENCE.
        :param blksize:          Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
        :param refine:           Number of times to recalculate motion vectors with halved block size.
        :param thsad_recalc:     Only bad quality new vectors with a SAD above this will be re-estimated by search.
                                 thsad value is scaled to 8x8 block size.

        :return:                 Decimated clip with text resampled down to 24p.
        """

        def select_every(clip: vs.VideoNode, cycle: int, offsets: int | list[int]) -> vs.VideoNode:
            if isinstance(offsets, int):
                offsets = [offsets]

            clips = list[vs.VideoNode]()
            for x in offsets:
                shifted = shift_clip(clip, x)
                if cycle != 1:
                    shifted = shifted.std.SelectEvery(cycle, 0)
                clips.append(shifted)

            return core.std.Interleave(clips)

        def _floor_div_tuple(x: tuple[int, int]) -> tuple[int, int]:
            return (x[0] // 2, x[1] // 2)

        assert check_variable(clip, self.__class__)
        assert check_variable(bobbed, self.__class__)
        check_ref_clip(bobbed, clip, self.__class__)

        field_ref = pattern * 2 % 5
        invpos = (5 - field_ref) % 5

        blksize = blksize if isinstance(blksize, tuple) else (blksize, blksize)

        match self:
            case InterpolateOverlay.IVTC_TXT60:
                clean = select_every(bobbed, 5, 1 - invpos)
                judder = select_every(bobbed, 5, [3 - invpos, 4 - invpos])
            case InterpolateOverlay.DEC_TXT60:
                clean = select_every(bobbed, 5, 4 - invpos)
                judder = select_every(bobbed, 5, [1 - invpos, 2 - invpos])
            case InterpolateOverlay.IVTC_TXT30:
                offsets = list(range(0, 10))
                offsets.pop(8)

                clean = IVTCycles.CYCLE_05.decimate(clip, pattern % 5)
                judder = select_every(bobbed, 1, -1 - invpos).std.SelectEvery(10, offsets)

        mv = MVTools(judder, **preset | KwargsT(search_clip=partial(prefilter_to_full_range, slope=1)))
        mv.analyze(tr=1, blksize=blksize, overlap=_floor_div_tuple(blksize))

        if refine:
            for _ in range(refine):
                blksize = _floor_div_tuple(blksize)
                overlap = _floor_div_tuple(blksize)

                mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=overlap)

        if self == InterpolateOverlay.IVTC_TXT30:
            comp = mv.flow_fps(fps=clean.fps)
            fixed = core.std.Interleave([clean, comp])
        else:
            comp = mv.flow_interpolate(interleave=False)[0]
            fixed = core.std.Interleave([clean, comp[::2]])

        match self:
            case InterpolateOverlay.IVTC_TXT60:
                return fixed[invpos // 2 :]
            case InterpolateOverlay.DEC_TXT60:
                return fixed[invpos // 3 :]
            case InterpolateOverlay.IVTC_TXT30:
                return core.std.SelectEvery(fixed, 8, (3, 5, 7, 6))


class FixInterlacedFades(CustomIntEnum):
    AVERAGE = 0
    """Adjust the average of each field to `color`."""

    MATCH = 1
    """Match to the field closest to `color`."""

    def __call__(
        self, clip: vs.VideoNode, color: float | Sequence[float] = 0.0, planes: PlanesT = None
    ) -> ConstantFormatVideoNode:
        """
        Give a mathematically perfect solution to decombing fades made *after* telecine
        (which made perfect IVTC impossible) that start or end in a solid color.

        Steps between the frames are not adjusted, so they will remain uneven depending on the telecine pattern,
        but the decombing is blur-free, ensuring minimum information loss. However, this may cause small amounts
        of combing to remain due to error amplification, especially near the solid-color end of the fade.

        This is an improved version of the Fix-Telecined-Fades plugin.

        Make sure to run this *after* IVTC!

        :param clip:      Clip to process.
        :param color:     Fade source/target color (floating-point plane averages).
                          Accepts a single float or a sequence of floats to control the color per plane.

        :return:          Clip with fades to/from `color` accurately deinterlaced.
                          Frames that don't contain such fades may be damaged.
        """

        func = FunctionUtil(clip, self.__class__, planes, vs.YUV, 32)

        fields = limiter(func.work_clip).std.SeparateFields(tff=True)

        fields = norm_expr(fields, 'x {color} - abs', planes, color=color, func=self.__class__)
        for i in func.norm_planes:
            fields = fields.std.PlaneStats(None, i, f'P{i}')

        props_clip = core.akarin.PropExpr(
            [func.work_clip, fields[::2], fields[1::2]],
            lambda: {
                f'f{f}Avg{i}': f'{c}.P{i}Average' for f, c in zip('tb', 'yz') for i in func.norm_planes
            }
        )

        expr = (
            'Y 2 % x.fbAvg{i} x.ftAvg{i} ? AVG! '
            'AVG@ 0 = x x {color} - x.ftAvg{i} x.fbAvg{i} {expr_mode} AVG@ / * {color} + ?'
        )

        fix = norm_expr(
            props_clip, expr, planes,
            i=func.norm_planes, color=color,
            expr_mode='+ 2 /' if self == self.AVERAGE else 'min',
            func=self.__class__,
        )

        return func.return_clip(fix)



def vinverse(
    clip: vs.VideoNode,
    comb_blur: VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] | vs.VideoNode = partial(sbr, mode=ConvMode.VERTICAL),
    contra_blur: VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] | vs.VideoNode = BlurMatrix.BINOMIAL(
        mode=ConvMode.VERTICAL
    ),
    contra_str: float = 2.7,
    amnt: int | float | None = None,
    scl: float = 0.25,
    thr: int | float = 0,
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    """
    A simple but effective script to remove residual combing. Based on an AviSynth script by Didée.

    :param clip:            Clip to process.
    :param comb_blur:       Filter used to remove combing.
    :param contra_blur:     Filter used to calculate contra sharpening.
    :param contra_str:      Strength of contra sharpening.
    :param amnt:            Change no pixel by more than this in 8bit.
    :param thr:             Skip processing if abs(clip - comb_blur(clip)) < thr
    :param scl:             Scale factor for vshrpD * vblurD < 0.
    """

    if callable(comb_blur):
        blurred = comb_blur(clip, planes=planes)
    else:
        blurred = comb_blur

    if callable(contra_blur):
        blurred2 = contra_blur(blurred, planes=planes)
    else:
        blurred2 = contra_blur

    assert check_variable(clip, vinverse)
    assert check_variable(blurred, vinverse)
    assert check_variable(blurred2, vinverse)

    FormatsMismatchError.check(vinverse, clip, blurred, blurred2)

    expr = (
        'x y - D1! D1@ abs D1A! D1A@ {thr} < x y z - {sstr} * D2! D1A@ D2@ abs < D1@ D2@ ? D3! '
        'D1@ D2@ xor D3@ {scl} * D3@ ? y + '
    )

    if amnt is not None:
        expr += 'x {amnt} - x {amnt} + clip '
        amnt = scale_delta(amnt, 8, clip)

    return norm_expr(
        [clip, blurred, blurred2],
        f'{expr} ?',
        planes, sstr=contra_str, amnt=amnt,
        scl=scl, thr=scale_delta(thr, 8, clip),
        func=vinverse
    )
