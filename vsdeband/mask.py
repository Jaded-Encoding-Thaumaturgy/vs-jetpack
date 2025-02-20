from __future__ import annotations

from vsexprtools import ExprOp
from vsmasktools import Morpho, Prewitt, retinex
from vsrgtools import RemoveGrainMode, RemoveGrainModeT, gauss_blur, remove_grain
from vstools import get_y, vs

__all__ = [
    'deband_detail_mask'
]


def deband_detail_mask(
    clip: vs.VideoNode, sigma: float = 1.0, rxsigma: list[int] = [50, 200, 350],
    pf_sigma: float | None = 1.0, brz: tuple[float, float] = (0.038, 0.068),
    rg_mode: RemoveGrainModeT = RemoveGrainMode.MINMAX_MEDIAN_OPP
) -> vs.VideoNode:
    clip_y = get_y(clip)

    ret = retinex(
        clip_y if pf_sigma is None else gauss_blur(clip_y, pf_sigma), rxsigma, upper_thr=0.005
    )

    brz0, brz1 = brz

    blur_ret = gauss_blur(ret, sigma)
    blur_ret_diff = Morpho.deflate(ExprOp.SUB(blur_ret, ret))

    blur_ret_brz = Morpho.inflate(blur_ret_diff, 4)
    blur_ret_brz = Morpho.closing(Morpho.binarize(blur_ret_brz, brz0), 3)

    prewitt_mask = Morpho.inflate(Morpho.deflate(Prewitt.edgemask(clip_y, brz1, brz1)))
    prewitt_brz = Morpho.closing(Morpho.binarize(prewitt_mask, brz1), 2)

    merged = ExprOp.ADD(blur_ret_brz, prewitt_brz)

    return remove_grain(merged, rg_mode)
