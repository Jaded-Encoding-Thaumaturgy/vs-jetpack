from __future__ import annotations

from vsexprtools import ExprOp
from vsjetpack import deprecated
from vsrgtools import bilateral, gauss_blur, remove_grain
from vsrgtools.rgtools import RemoveGrain
from vstools import get_y, limiter, vs

from .edge import Kirsch, MinMax, Prewitt
from .morpho import Morpho
from .types import MaskLike
from .utils import normalize_mask

__all__ = [
    "detail_mask",
    "detail_mask_neo",
    "multi_detail_mask",
    "simple_detail_mask",
]


def detail_mask(
    clip: vs.VideoNode,
    brz_mm: float = 0.025,
    brz_ed: float = 0.045,
    minmax: MinMax = MinMax(rady=3, radc=2),
    edge: MaskLike = Kirsch,
    sigma: float | None = None,
) -> vs.VideoNode:
    if sigma:
        clip = gauss_blur(clip, sigma)

    range_mask = Morpho.binarize_mask(minmax.edgemask(clip), brz_mm)
    edges = Morpho.binarize_mask(normalize_mask(edge, clip), brz_ed)

    mask = ExprOp.MAX.combine(range_mask, edges)

    mask = remove_grain(mask, 22)
    mask = remove_grain(mask, 11)

    return mask


@limiter(mask=True)
def detail_mask_neo(
    clip: vs.VideoNode,
    sigma: float = 5.0,
    detail_brz: float = 0.005,
    lines_brz: float = 0.08,
    edgemask: MaskLike = Prewitt,
    rg_mode: RemoveGrain.Mode = remove_grain.Mode.MINMAX_MEDIAN_OPP,
) -> vs.VideoNode:
    clip_y = get_y(clip)
    blur_pf = gauss_blur(clip_y, sigma)

    blur_pref = bilateral(clip_y, blur_pf, sigma)
    blur_pref_diff = ExprOp.SUB.combine(blur_pref, clip_y).std.Deflate()
    blur_pref = Morpho.inflate(blur_pref_diff, iterations=4)

    prew_mask = normalize_mask(edgemask, clip_y).std.Deflate().std.Inflate()

    if detail_brz > 0:
        blur_pref = Morpho.binarize_mask(blur_pref, detail_brz)

    if lines_brz > 0:
        prew_mask = Morpho.binarize_mask(prew_mask, lines_brz)

    merged = ExprOp.ADD.combine(blur_pref, prew_mask)

    return remove_grain(merged, rg_mode)


@deprecated("simple_detail_mask is deprecated. Use detail_mask instead.", category=DeprecationWarning)
def simple_detail_mask(
    clip: vs.VideoNode, sigma: float | None = None, rad: int = 3, brz_a: float = 0.025, brz_b: float = 0.045
) -> vs.VideoNode:
    return detail_mask(get_y(clip), brz_a, brz_b, MinMax(rady=rad, radc=0), Prewitt, sigma)


def multi_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
    y = get_y(clip)
    general_mask = detail_mask(y, 1, 24.3 * thr, MinMax(rady=1, radc=0), Prewitt)

    return ExprOp.MIN.combine(
        ExprOp.MIN.combine(
            detail_mask(y, 1, 2 * thr, MinMax(rady=3, radc=0), Prewitt),
            Morpho.maximum(general_mask, iterations=4).std.Inflate(),
        ),
        general_mask.std.Maximum(),
    )
