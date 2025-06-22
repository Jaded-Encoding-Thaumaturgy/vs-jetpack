from __future__ import annotations

from vsexprtools import ExprOp
from vsrgtools import bilateral, gauss_blur, remove_grain
from vsrgtools.rgtools import RemoveGrain
from vstools import ConstantFormatVideoNode, check_variable, get_y, limiter, plane, vs

from .edge import Kirsch, MinMax, Prewitt, PrewittTCanny
from .masks import range_mask
from .morpho import Morpho
from .types import GenericMaskT
from .utils import normalize_mask

__all__ = [
    "detail_mask",
    "detail_mask_neo",
    "multi_detail_mask",
    "simple_detail_mask",
]


@limiter
def detail_mask(
    clip: vs.VideoNode,
    brz_mm: float,
    brz_ed: float,
    minmax: MinMax = MinMax(rady=3, radc=2),
    edge: GenericMaskT = Kirsch,
) -> ConstantFormatVideoNode:
    assert check_variable(clip, detail_mask)

    range_mask = Morpho.binarize(minmax.edgemask(clip), brz_mm)

    edges = Morpho.binarize(normalize_mask(edge, clip), brz_ed)

    mask = ExprOp.MAX.combine(range_mask, edges)

    mask = remove_grain(mask, 22)
    mask = remove_grain(mask, 11)

    return mask


@limiter
def detail_mask_neo(
    clip: vs.VideoNode,
    sigma: float = 1.0,
    detail_brz: float = 0.05,
    lines_brz: float = 0.08,
    edgemask: GenericMaskT = Prewitt,
    rg_mode: RemoveGrain.Mode = remove_grain.Mode.MINMAX_MEDIAN_OPP,
) -> ConstantFormatVideoNode:
    assert check_variable(clip, detail_mask_neo)

    clip_y = get_y(clip)
    blur_pf = gauss_blur(clip_y, sigma * 0.75)

    blur_pref = bilateral(clip_y, blur_pf, sigma)
    blur_pref_diff = ExprOp.SUB.combine(blur_pref, clip_y).std.Deflate()
    blur_pref = Morpho.inflate(blur_pref_diff, iterations=4)

    prew_mask = normalize_mask(edgemask, clip_y).std.Deflate().std.Inflate()

    if detail_brz > 0:
        blur_pref = Morpho.binarize(blur_pref, detail_brz)

    if lines_brz > 0:
        prew_mask = Morpho.binarize(prew_mask, lines_brz)

    merged = ExprOp.ADD.combine(blur_pref, prew_mask)

    return remove_grain(merged, rg_mode)


@limiter
def simple_detail_mask(
    clip: vs.VideoNode, sigma: float | None = None, rad: int = 3, brz_a: float = 0.025, brz_b: float = 0.045
) -> ConstantFormatVideoNode:
    y = plane(clip, 0)

    blur = gauss_blur(y, sigma) if sigma else y

    mask_a = Morpho.binarize(range_mask(blur, rad=rad), brz_a)

    mask_b = Morpho.binarize(PrewittTCanny.edgemask(blur), brz_b)

    mask = ExprOp.MAX.combine(mask_a, mask_b)

    return remove_grain(remove_grain(mask, 22), 11)


def multi_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> ConstantFormatVideoNode:
    general_mask = simple_detail_mask(clip, rad=1, brz_a=1, brz_b=24.3 * thr)

    return ExprOp.MIN.combine(
        ExprOp.MIN.combine(
            simple_detail_mask(clip, brz_a=1, brz_b=2 * thr), Morpho.maximum(general_mask, iterations=4).std.Inflate()
        ),
        general_mask.std.Maximum(),
    )
