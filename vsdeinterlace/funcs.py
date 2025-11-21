from __future__ import annotations

from functools import partial
from typing import Any, Literal, Mapping, Sequence, overload

from jetpytools import CustomEnum, CustomIntEnum

from vsdenoise import MotionVectors, MVTools, MVToolsPreset, prefilter_to_full_range, refine_blksize
from vsexprtools import norm_expr
from vsrgtools import BlurMatrix, sbr
from vstools import (
    ConvMode,
    FormatsMismatchError,
    FunctionUtil,
    Planes,
    VSFunctionKwArgs,
    check_ref_clip,
    core,
    scale_delta,
    vs,
)

__all__ = ["FixInterlacedFades", "InterpolateOverlay", "vinverse"]


class InterpolateOverlay(CustomEnum):
    """
    Enum defining interpolation patterns for handling overlaid 60i/30p text on top of telecined 24p footage.
    """

    IVTC_TXT60 = 10, (4, 2, 0, 8, 6)
    """
    For 60i overlaid on top of 24t.
    """

    DEC_TXT60 = 10, (6, 4, 2, 0, 8)
    """
    For 60i overlaid on top of 24d.
    """

    IVTC_TXT30 = 9, (5, 13, 21, 29, 37)
    """
    For 30p overlaid on top of 24t.
    """

    @overload
    def __call__(
        self,
        clip: vs.VideoNode,
        pattern: int,
        vectors: MotionVectors | None = None,
        preset: Mapping[str, Any] = ...,
        blksize: int | tuple[int, int] = 8,
        overlap: int | tuple[int, int] = 2,
        refine: int = 1,
        thsad_recalc: int | None = None,
        export_globals: Literal[False] = False,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self,
        clip: vs.VideoNode,
        pattern: int,
        vectors: MotionVectors | None = None,
        preset: Mapping[str, Any] = ...,
        blksize: int | tuple[int, int] = 8,
        overlap: int | tuple[int, int] = 2,
        refine: int = 1,
        thsad_recalc: int | None = None,
        *,
        export_globals: Literal[True],
    ) -> tuple[vs.VideoNode, MVTools]: ...

    @overload
    def __call__(
        self,
        clip: vs.VideoNode,
        pattern: int,
        vectors: MotionVectors | None = None,
        preset: Mapping[str, Any] = ...,
        blksize: int | tuple[int, int] = 8,
        overlap: int | tuple[int, int] = 2,
        refine: int = 1,
        thsad_recalc: int | None = None,
        export_globals: bool = ...,
    ) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]: ...

    def __call__(
        self,
        clip: vs.VideoNode,
        pattern: int,
        vectors: MotionVectors | None = None,
        preset: Mapping[str, Any] = MVToolsPreset.HQ_COHERENCE,
        blksize: int | tuple[int, int] = 8,
        overlap: int | tuple[int, int] = 2,
        refine: int = 1,
        thsad_recalc: int | None = None,
        export_globals: bool = False,
    ) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
        """
        Virtually oversamples the video to 120 fps with motion interpolation on credits only, and decimates to 24 fps.
        Requires manually specifying the 3:2 pulldown pattern (the clip must be split into parts if it changes).

        Args:
            clip: Bob-deinterlaced clip.
            vectors: Motion vectors to use.
            pattern: First frame of any clean-combed-combed-clean-clean sequence.
            preset: MVTools preset defining base values for the MVTools object. Default is HQ_COHERENCE.
            blksize: Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
            overlap: The blksize divisor for block overlap. Larger overlapping reduces blocking artifacts.
            refine: Number of times to recalculate motion vectors with halved block size.
            thsad_recalc: Only bad quality new vectors with a SAD above this will be re-estimated by search. thsad value
                is scaled to 8x8 block size.
            export_globals: Whether to return the MVTools object.

        Returns:
            Decimated clip with text resampled down to 24p.
        """
        step, lookup = self.value
        offset = lookup[pattern % 5]
        offsets = [(offset + i * step) % 40 for i in range(4)]

        mv = MVTools(
            clip,
            vectors=vectors,
            **{**preset, "search_clip": partial(prefilter_to_full_range, slope=1, func=self.__class__)},
        )

        if not vectors:
            mv.analyze(tr=1, blksize=blksize, overlap=refine_blksize(blksize, overlap))

            for _ in range(refine):
                blksize = refine_blksize(blksize)
                mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=refine_blksize(blksize, overlap))

        comp = mv.flow_fps(fps=clip.fps * 4)
        comp += comp[-1] * 3
        fixed = core.std.SelectEvery(comp, 40, sorted(offsets))

        return (fixed, mv) if export_globals else fixed


class FixInterlacedFades(CustomIntEnum):
    """
    Enum for mathematically decombing fades that were applied *after* telecine.
    """

    AVERAGE = 0
    """
    Adjust the average of each field to `color`.
    """

    MATCH = 1
    """
    Match to the field closest to `color`.
    """

    def __call__(
        self, clip: vs.VideoNode, color: float | Sequence[float] | vs.VideoNode = 0.0, planes: Planes = None
    ) -> vs.VideoNode:
        """
        Give a mathematically perfect solution to decombing fades made *after* telecine
        (which made perfect IVTC impossible) that start or end in a solid color.

        Steps between the frames are not adjusted, so they will remain uneven depending on the telecine pattern,
        but the decombing is blur-free, ensuring minimum information loss. However, this may cause small amounts
        of combing to remain due to error amplification, especially near the solid-color end of the fade.

        This is an improved version of the Fix-Telecined-Fades plugin.

        Make sure to run this *after* IVTC!

        Args:
            clip: Clip to process.
            color: Fade source/target color (floating-point plane averages or a single-frame clip matching the format of
                `clip`). Accepts a single float or a sequence of floats to control the color per plane.

        Returns:
            Clip with fades to/from `color` accurately deinterlaced.
                Frames that don't contain such fades may be damaged.
        """
        func = FunctionUtil(clip, self.__class__, planes, vs.YUV, 32)

        expr_clips = list[vs.VideoNode]()
        fields = func.work_clip.std.SeparateFields(tff=True)

        if isinstance(color, vs.VideoNode):
            check_ref_clip(color, func.work_clip, self.__class__)

            expr_clips.append(color)
            clipb, prop_name, expr_color = color.std.SeparateFields(tff=True), "Diff", "y"
        else:
            fields = norm_expr(fields, "x 0 1 clip {color} - abs", planes, color=color, func=self.__class__)
            clipb, prop_name, expr_color = None, "Average", color

        for i in func.norm_planes:
            fields = fields.std.PlaneStats(clipb, i, f"P{i}")

        props_clip = core.akarin.PropExpr(
            [func.work_clip, fields[::2], fields[1::2]],
            lambda: {f"f{f}Avg{i}": f"{c}.P{i}{prop_name}" for f, c in zip("tb", "yz") for i in func.norm_planes},
        )
        expr_clips.insert(0, props_clip)

        expr = (
            "Y 2 % x.fbAvg{i} x.ftAvg{i} ? AVG! "
            "AVG@ x {color} - x.ftAvg{i} x.fbAvg{i} {expr_mode} AVG@ / * {color} + x ?"
        )

        fix = norm_expr(
            expr_clips,
            expr,
            planes,
            i=func.norm_planes,
            color=expr_color,
            expr_mode="+ 2 /" if self == self.AVERAGE else "min",
            func=self.__class__,
        )

        return func.return_clip(fix)


def vinverse(
    clip: vs.VideoNode,
    comb_blur: VSFunctionKwArgs | vs.VideoNode = partial(sbr, mode=ConvMode.VERTICAL),
    contra_blur: VSFunctionKwArgs | vs.VideoNode = BlurMatrix.BINOMIAL(mode=ConvMode.VERTICAL),
    contra_str: float = 2.7,
    amnt: float | None = None,
    scl: float = 0.25,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    A simple but effective script to remove residual combing. Based on an AviSynth script by Didée.

    Args:
        clip: Clip to process.
        comb_blur: Filter used to remove combing.
        contra_blur: Filter used to calculate contra sharpening.
        contra_str: Strength of contra sharpening.
        amnt: Change no pixel by more than this in 8bit.
        scl: Scale factor for vshrpD * vblurD < 0.
    """

    blurred = comb_blur(clip, planes=planes) if callable(comb_blur) else comb_blur
    blurred2 = contra_blur(blurred, planes=planes) if callable(contra_blur) else contra_blur

    FormatsMismatchError.check(vinverse, clip, blurred, blurred2)

    expr = "y z - {sstr} * D1! x y - D2! D1@ abs D2@ abs < D1@ D2@ ? D3! D1@ D2@ xor D3@ {scl} * D3@ ? y +"

    if amnt is not None:
        expr += " x {amnt} - x {amnt} + clip"
        amnt = scale_delta(amnt, 8, clip)

    return norm_expr([clip, blurred, blurred2], expr, planes, sstr=contra_str, amnt=amnt, scl=scl, func=vinverse)
