"""
This module contains general denoising functions built on top of base denoisers.
"""

from __future__ import annotations

from typing import Any, Literal, overload

from vsscale import Waifu2x
from vsscale.scale import BaseWaifu2x
from vstools import CustomIndexError, KwargsNotNone, PlanesT, VSFunction, fallback, normalize_seq, vs

from .mvtools import MotionVectors, MVTools, MVToolsPreset, MVToolsPresets, RFilterMode
from .prefilters import PrefilterPartial

__all__ = [
    'mc_degrain',

    'waifu2x_denoise'
]


@overload
def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | PrefilterPartial | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    preset: MVToolsPreset = MVToolsPresets.HQ_SAD, tr: int = 1,
    rfilter: int | tuple[int, int] = (RFilterMode.CUBIC, RFilterMode.TRIANGLE),
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: Literal[False] = ...,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    ...


@overload
def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | PrefilterPartial | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    preset: MVToolsPreset = MVToolsPresets.HQ_SAD, tr: int = 1,
    rfilter: int | tuple[int, int] = (RFilterMode.CUBIC, RFilterMode.TRIANGLE),
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: Literal[True] = ...,
    planes: PlanesT = None, **kwargs: Any
) -> tuple[vs.VideoNode, MVTools]:
    ...


@overload
def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | PrefilterPartial | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    preset: MVToolsPreset = MVToolsPresets.HQ_SAD, tr: int = 1,
    rfilter: int | tuple[int, int] = (RFilterMode.CUBIC, RFilterMode.TRIANGLE),
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: bool = ...,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    ...


def mc_degrain(
    clip: vs.VideoNode, vectors: MotionVectors | MVTools | None = None,
    prefilter: vs.VideoNode | PrefilterPartial | VSFunction | None = None, mfilter: vs.VideoNode | VSFunction | None = None,
    preset: MVToolsPreset = MVToolsPresets.HQ_SAD, tr: int = 1,
    rfilter: int | tuple[int, int] = (RFilterMode.CUBIC, RFilterMode.TRIANGLE),
    blksize: int | tuple[int, int] = 16, refine: int = 1,
    thsad: int | tuple[int, int] = 400, thsad2: int | tuple[int | None, int | None] | None = None,
    thsad_recalc: int | None = None, limit: int | tuple[int | None, int | None] | None = None,
    thscd: int | tuple[int | None, int | None] | None = None, export_globals: bool = False,
    planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode | tuple[vs.VideoNode, MVTools]:
    """
    Perform temporal denoising using motion compensation.

    Motion compensated blocks from previous and next frames are averaged with the current frame.
    The weighting factors for each block depend on their SAD from the current frame.

    :param clip:              The clip to process.
    :param vectors:           Motion vectors to use. Can be a MotionVectors object or another MVTools instance.
    :param prefilter:         Filter or clip to use when performing motion vector search.
    :param mfilter:           Filter or clip to use where degrain couldn't find a matching block.
    :param preset:            MVTools preset defining base values for the MVTools object.
    :param tr:                The temporal radius. This determines how many frames are analyzed before/after the current frame.
    :param rfilter:           Hierarchical levels smoothing and reducing (halving) filter.
    :param blksize:           Size of a block. Larger blocks are less sensitive to noise, are faster, but also less accurate.
    :param refine:            Number of times to recalculate motion vectors with halved block size.
    :param thsad:             Defines the soft threshold of block sum absolute differences.
                              Blocks with SAD above this threshold have zero weight for averaging (denoising).
                              Blocks with low SAD have highest weight.
                              The remaining weight is taken from pixels of source clip.
    :param thsad2:            Define the SAD soft threshold for frames with the largest temporal distance.
                              The actual SAD threshold for each reference frame is interpolated between thsad (nearest frames)
                              and thsad2 (furthest frames).
                              Only used with the FLOAT MVTools plugin.
    :param thsad_recalc:      Only bad quality new vectors with a SAD above thid will be re-estimated by search.
                              thsad value is scaled to 8x8 block size.
    :param limit:             Maximum allowed change in pixel values.
    :param thscd:             Scene change detection thresholds:
                              - First value: SAD threshold for considering a block changed between frames.
                              - Second value: Number of changed blocks needed to trigger a scene change.
    :param export_globals:    Whether to return the MVTools object.
    :param planes:            Which planes to process. Default: None (all planes).

    :return:                  Motion compensated and temporally filtered clip with reduced noise.
    """
    def _floor_div_tuple(x: tuple[int, int], div: int = 2) -> tuple[int, int]:
        return (x[0] // div, x[1] // div)

    mv_args = preset | kwargs | KwargsNotNone(search_clip=prefilter, tr=tr)

    rfilter_srch, rfilter_render = normalize_seq(rfilter, 2)

    blksize = blksize if isinstance(blksize, tuple) else (blksize, blksize)
    thsad = thsad if isinstance(thsad, tuple) else (thsad, thsad)

    mfilter = mfilter(clip) if callable(mfilter) else fallback(mfilter, clip)

    mv = MVTools(clip, vectors=vectors, planes=planes, **mv_args)

    if mv.search_clip != mv.clip or rfilter_render != rfilter_srch:
        mv.super(mv.clip, rfilter=rfilter_render)

    mv.super(mv.search_clip, rfilter=rfilter_srch)

    if not vectors:
        mv.analyze(blksize=blksize, overlap=_floor_div_tuple(blksize))

        if refine:
            if thsad_recalc is None:
                thsad_recalc = thsad[0] // 2

            for _ in range(refine):
                blksize = _floor_div_tuple(blksize)
                overlap = _floor_div_tuple(blksize)

                mv.recalculate(thsad=thsad_recalc, blksize=blksize, overlap=overlap)

    den = mv.degrain(mfilter, mv.clip, None, tr, thsad, thsad2, limit, thscd)

    return (den, mv) if export_globals else den


def waifu2x_denoise(
    clip: vs.VideoNode, noise: int = 1, model: type[BaseWaifu2x] = Waifu2x.Cunet, **kwargs: Any
) -> vs.VideoNode:
    if noise < 0 or noise > 3:
        raise CustomIndexError('"noise" must be in range 0-3 (inclusive).')

    if not isinstance(model, type):
        model = model.__class__  # type: ignore

    return model(**kwargs).scale(clip, clip.width, clip.height, _static_args=dict(scale=1, noise=noise, force=True))
