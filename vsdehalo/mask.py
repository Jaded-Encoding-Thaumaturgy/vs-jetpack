from __future__ import annotations

from math import sqrt

from vsaa import Nnedi3
from vsexprtools import norm_expr
from vskernels import Point
from vsmasktools import Morpho, PrewittTCanny
from vsrgtools import BlurMatrix
from vstools import check_progressive, get_y, scale_delta, vs

__all__ = [
    'base_dehalo_mask'
]


def base_dehalo_mask(
    src: vs.VideoNode, expand: float = 0.5, iterations: int = 2,
    brz0: float = 0.31, brz1: float = 1.0, shift: int = 8,
    pre_ss: bool = True, multi: float = 1.0
) -> vs.VideoNode:
    """
    Based on `muvsfunc.YAHRmask`, stand-alone version with some tweaks. Adopted from jvsfunc.

    :param src:         Input clip.
    :param expand:      Expansion of edge mask.
    :param iterations:  Protects parallel lines and corners that are usually damaged by strong dehaloing.
    :param brz:         Adjusts the internal line thickness.
    :param shift:       8-bit corrective shift value for fine-tuning expansion.
    :param pre_ss:      Perform the mask creation at 2x.

    :return:            Dehalo mask.
    """

    assert check_progressive(src, base_dehalo_mask)

    luma = get_y(src)

    if pre_ss:
        luma = Nnedi3.scale(luma, luma.width * 2, luma.height * 2)

    exp_edges = norm_expr(
        [luma, Morpho.maximum(luma, iterations=2)], 'y x - {shift} - range_half *',
        shift=scale_delta(shift, 8, luma), func=base_dehalo_mask
    )

    edgemask = PrewittTCanny.edgemask(exp_edges, sigma=sqrt(expand * 2), mode=-1, multi=16)

    halo_mask = Morpho.maximum(exp_edges, iterations=iterations)
    halo_mask = Morpho.minimum(halo_mask, iterations=iterations)
    halo_mask = Morpho.binarize(halo_mask, brz0, 1.0, 0.0)

    if brz1 != 1.0:
        halo_mask = Morpho.inflate(halo_mask, iterations=2)
        halo_mask = Morpho.binarize(halo_mask, brz1)

    mask = norm_expr(
        [edgemask, BlurMatrix.BINOMIAL()(halo_mask)], 'x y min {multi} *', multi=multi, func=base_dehalo_mask
    )

    if pre_ss:
        return Point.scale(mask, src.width, src.height)

    return mask
