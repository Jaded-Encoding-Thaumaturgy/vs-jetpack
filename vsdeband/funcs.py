from __future__ import annotations

from math import ceil
from typing import Any

from vsdenoise import Prefilter, frequency_merge
from vsexprtools import ExprOp
from vsmasktools import FDoG, Morpho, flat_mask, texture_mask
from vsrgtools import (
    BlurMatrix, MeanMode, RemoveGrainMode, RemoveGrainModeT,
    box_blur, gauss_blur, limit_filter, remove_grain
)
from vsexprtools import norm_expr
from vstools import (
    ColorRange, PlanesT, VSFunction, FunctionUtil, check_ref_clip, check_variable, depth,
     expect_bits, fallback, normalize_planes, normalize_seq, scale_value, to_arr, vs
)

from .abstract import Debander
from .f3kdb import F3kdb
from .filters import guided_filter
from .mask import deband_detail_mask
from .placebo import Placebo
from .types import GuidedFilterMode

__all__ = [
    'mdb_bilateral',

    'masked_deband',

    'pfdeband',

    'guided_deband',

    'DebandPassPresets', 'multi_deband'
]


def mdb_bilateral(
    clip: vs.VideoNode, radius: int = 16,
    thr: int | list[int] = 260,
    lthr: int | tuple[int, int] = (153, 0), elast: float = 3.0,
    bright_thr: int | None = None,
    debander: type[Debander] | Debander = F3kdb
) -> vs.VideoNode:
    """
    Multi stage debanding, bilateral-esque filter.

    This function is more of a last resort for extreme banding.
    Recommend values are ~40-60 for luma and chroma strengths.

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param lthr:        Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :param elast:       Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :param bright_thr:  Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :param debander:    Specify what Debander to use. You can pass an instance with custom arguments.

    :return:            Debanded clip.
    """

    assert check_variable(clip, mdb_bilateral)

    if not isinstance(debander, Debander):
        debander = debander()

    clip, bits = expect_bits(clip, 16)

    rad1, rad2, rad3 = round(radius * 4 / 3), round(radius * 2 / 3), round(radius / 3)

    db1 = debander.deband(clip, radius=rad1, thr=[max(1, th // 2) for th in to_arr(thr)], grain=0.0)
    db2 = debander.deband(db1, radius=rad2, thr=thr, grain=0)
    db3 = debander.deband(db2, radius=rad3, thr=thr, grain=0)

    limit = limit_filter(db3, db2, clip, thr=lthr, elast=elast, bright_thr=bright_thr)

    return depth(limit, bits)


def masked_deband(
    clip: vs.VideoNode, radius: int = 16,
    thr: float | list[float] = 96, grain: float | list[float] = [0.23, 0],
    sigma: float = 1.25, rxsigma: list[int] = [50, 220, 300],
    pf_sigma: float | None = 1.25, brz: tuple[float, float] = (0.038, 0.068),
    rg_mode: RemoveGrainModeT = RemoveGrainMode.MINMAX_MEDIAN_OPP,
    debander: type[Debander] | Debander = F3kdb, **kwargs: Any
) -> vs.VideoNode:
    clip, bits = expect_bits(clip, 16)

    if not isinstance(debander, Debander):
        debander = debander()

    deband_mask = deband_detail_mask(clip, sigma, rxsigma, pf_sigma, brz, rg_mode)

    deband = debander.deband(clip, radius=radius, thr=thr, grain=grain, **kwargs)

    masked = deband.std.MaskedMerge(clip, deband_mask)

    return depth(masked, bits)


def pfdeband(
    clip: vs.VideoNode, radius: int = 16, thr: float | list[float] = 96,
    lthr: int | tuple[int, int] = 0.5, elast: float = 1.5,
    bright_thr: int | None = None, prefilter: Prefilter | VSFunction = gauss_blur,
    debander: type[Debander] | Debander = F3kdb, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Prefilter and deband a clip.

    :param clip:          Input clip.
    :param radius:        Banding detection range.
    :param thr:           Banding detection thr(s) for planes.
    :param lthr:          Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :param elast:         Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :param bright_thr:    Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :param prefilter:     Prefilter used to blur the clip before debanding.
    :param debander:      Specify what Debander to use. You can pass an instance with custom arguments.
    :param planes:        Planes to process

    :return:              Debanded clip.
    """
    func = FunctionUtil(clip, pfdeband, planes, (vs.YUV, vs.GRAY), (8, 16))

    if not isinstance(debander, Debander):
        debander = debander()

    blur = prefilter(func.work_clip, planes=planes, **kwargs)
    diff = func.work_clip.std.MakeDiff(blur, planes=planes)

    deband = debander.deband(blur, radius=radius, thr=thr, planes=planes)

    merge = deband.std.MergeDiff(diff, planes=planes)
    limit = limit_filter(merge, func.work_clip, thr=lthr, elast=elast, bright_thr=bright_thr)

    return func.return_clip(limit)


def guided_deband(
    clip: vs.VideoNode, radius: int | list[int] | None = None, strength: float = 0.3,
    thr: int | tuple[int, int] | None = None, mode: GuidedFilterMode = GuidedFilterMode.GRADIENT,
    rad: int = 0, bin_thr: float | list[float] | None = 0, planes: PlanesT = None,
    range_in: ColorRange | None = None, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, guided_deband)

    planes = normalize_planes(clip, planes)

    range_in = ColorRange.from_param_or_video(range_in, clip)

    rad = fallback(rad, ceil(clip.height / 540))

    if bin_thr is None:
        if clip.format.sample_type is vs.FLOAT:
            bin_thr = 1.5 / 255 if range_in.is_full else [1.5 / 219, 1.5 / 224]
        else:
            bin_thr = scale_value(0.005859375, 32, clip, range_out=range_in)

    bin_thr = normalize_seq(bin_thr, clip.format.num_planes)

    deband = guided_filter(clip, None, radius, strength, mode, planes=planes, **kwargs)

    if thr:
        deband = limit_filter(deband, clip, thr=thr)

    if rad:
        morpho = Morpho(planes)
        rmask = ExprOp.SUB.combine(morpho.expand(clip, rad), morpho.inpand(clip, rad), planes=planes)

        if bin_thr and max(bin_thr) > 0:
            rmask = rmask.std.Binarize(threshold=bin_thr, planes=planes)

        rmask = remove_grain(rmask, RemoveGrainMode.OPP_CLIP_AVG_FAST)
        rmask = BlurMatrix.BINOMIAL()(rmask)
        rmask = remove_grain(rmask, RemoveGrainMode.MIN_SHARP)

        deband = deband.std.MaskedMerge(clip, rmask, planes=planes)

    return deband


class DebandPassPresets:
    LIGHT = (
        (F3kdb(16, 120), False, False),
        (F3kdb(31, 160), False, True),
        (Placebo(8, 2.5), False, False),
        (Placebo(16, 1.5), False, True),
        (Placebo(24, 1.0), False, True),
    )
    MEDIUM = (
        (F3kdb(16, 120), False, False),
        (F3kdb(31, 160), False, True),
        (Placebo(8, 1.75), False, False),
        (Placebo(16, 1.275), False, True),
        (Placebo(24, 0.8), False, True),
    )
    STRONG = (
        (F3kdb(16, 120), False, False),
        (F3kdb(31, 160), True, False),
        (Placebo(8, 2.5), False, False),
        (Placebo(16, 1.5), True, False),
        (Placebo(24, 1.0), True, False),
    )


def multi_deband(
    clip: vs.VideoNode, *passes: Debander | tuple[Debander, bool] | tuple[Debander, bool, bool],
    ref: vs.VideoNode | None = None, base_db: Debander = F3kdb(8, 100), lowpass_db: Debander = Placebo(24, 6.0),
    edgemask: vs.VideoNode | None = None, textures: vs.VideoNode | None = None, **freq_merge_kwargs: Any
) -> vs.VideoNode:
    ref = check_ref_clip(clip, ref, multi_deband)

    if edgemask is None:
        edgemask = FDoG.edgemask(ref, 0.25, 1.0, 2, planes=(0, True))

    if textures is None:
        edges = flat_mask(ref, 6, 0.025, False)

        textures = ExprOp.MAX(
            texture_mask(ref, thr=0.001, blur=3, points=[
                (False, 1.15), (True, 1.75), (False, 2.5),
                (True, 5.0), (False, 7.5), (False, 10.0)
            ]).resize.Bicubic(format=vs.YUV444P16), split_planes=True
        )
        textures = box_blur(ExprOp.SUB(textures, edges), 2)
        textures = norm_expr(textures, 'x 2 *', func=guided_filter)

    line_big = ExprOp.ADD(
        edgemask, gauss_blur(edgemask.std.Maximum(), 0.75), expr_suffix='4 *'
    ).std.Maximum()

    linemask_deband = gauss_blur(line_big.std.BinarizeMask(20 << 10), 0.45)

    base_deband = base_db.deband(clip)
    lowpass_deband = lowpass_db.deband(clip).std.MaskedMerge(base_deband, linemask_deband)

    freq_merge_kwargs = dict(mode_high=MeanMode.HARMONIC) | freq_merge_kwargs

    def _norm_pass(
        dbpass: Debander | tuple[Debander, bool] | tuple[Debander, bool, bool]
    ) -> tuple[Debander, bool, bool]:
        mask, base_db = False, True

        if isinstance(dbpass, tuple):
            debander, mask, *other = dbpass

            if other:
                base_db = other[0]
        else:
            debander = dbpass

        return debander, mask, base_db

    deband = frequency_merge(
        base_deband,
        (
            debanded
            if (debanded := debander.deband(clip if base_db else base_deband))
            and not mask else debanded.std.MaskedMerge(base_deband, textures)
            for debander, mask, base_db in map(_norm_pass, passes)
        ),
        lowpass=lambda *args, **kwargs: lowpass_deband, **freq_merge_kwargs
    )

    return deband.std.MaskedMerge(clip.std.Merge(base_deband, 0.5), textures)
