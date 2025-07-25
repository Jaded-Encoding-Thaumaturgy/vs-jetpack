from __future__ import annotations

from math import sqrt

from vsaa import NNEDI3
from vsexprtools import norm_expr
from vsmasktools import Morpho, PrewittTCanny
from vsrgtools import BlurMatrix
from vstools import check_progressive, get_y, scale_delta, vs

__all__ = ["base_dehalo_mask"]


def base_dehalo_mask(
    src: vs.VideoNode,
    expand: float = 0.5,
    iterations: int = 2,
    brz0: float = 0.31,
    brz1: float = 1.0,
    shift: int = 8,
    pre_ss: bool = True,
    multi: float = 1.0,
) -> vs.VideoNode:
    """
    Based on `muvsfunc.YAHRmask`, stand-alone version with some tweaks. Adopted from jvsfunc.

    Args:
        src: Input clip.
        expand: Expansion of edge mask.
        iterations: Protects parallel lines and corners that are usually damaged by strong dehaloing.
        brz0: Adjusts the internal line thickness.
        brz1: Adjusts the internal line thickness.
        shift: 8-bit corrective shift value for fine-tuning expansion.
        pre_ss: Perform the mask creation at 2x.
        multi: Final pixel value multiplier.

    Returns:
        Dehalo mask.
    """

    assert check_progressive(src, base_dehalo_mask)

    luma = get_y(src)

    if pre_ss:
        luma = NNEDI3().supersample(luma)

    exp_edges = norm_expr(
        [luma, Morpho.maximum(luma, iterations=2)],
        "y x - {shift} - range_half *",
        shift=scale_delta(shift, 8, luma),
        func=base_dehalo_mask,
    )

    edgemask = PrewittTCanny.edgemask(exp_edges, sigma=sqrt(expand * 2), mode=-1, multi=16)

    halo_mask = Morpho.maximum(exp_edges, iterations=iterations)
    halo_mask = Morpho.minimum(halo_mask, iterations=iterations)
    halo_mask = Morpho.binarize(halo_mask, brz0, 1.0, 0.0)

    if brz1 != 1.0:
        halo_mask = Morpho.inflate(halo_mask, iterations=2)
        halo_mask = Morpho.binarize(halo_mask, brz1)

    mask = norm_expr(
        [edgemask, BlurMatrix.BINOMIAL()(halo_mask)], "x y min {multi} *", multi=multi, func=base_dehalo_mask
    )

    if pre_ss:
        return vs.core.resize.Point(mask, src.width, src.height)

    return mask
