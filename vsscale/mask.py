from __future__ import annotations

from vsexprtools import ExprOp, norm_expr
from vsmasktools import Morpho, XxpandMode
from vsrgtools import BlurMatrix, box_blur, gauss_blur
from vstools import ConstantFormatVideoNode, ConvMode, core, get_y, iterate, limiter, shift_clip_multi, split, vs

__all__ = ["descale_detail_mask", "descale_error_mask"]


@limiter
def descale_detail_mask(
    clip: vs.VideoNode, rescaled: vs.VideoNode, thr: float = 0.05, inflate: int = 2, xxpand: tuple[int, int] = (4, 0)
) -> ConstantFormatVideoNode:
    """
    Mask non-native resolution detail to prevent detail loss and artifacting.

    Descaling without masking is very dangerous, as descaling FHD material often leads to
    heavy artifacting and fine detail loss.

    Args:
        clip: Original clip.
        rescaled: Clip rescaled using the presumed native kernel.
        thr: Binarizing threshold. Lower will catch more. Assumes float bitdepth input. Default: 0.05.
        inflate: Amount of times to ``inflate`` the mask. Default: 2.
        xxpand: Amount of times to ``Maximum`` the clip by. The first ``Maximum`` is done before inflating, the second
            after. Default: 4 times pre-inflating, 0 times post-inflating.

    Returns:
        Mask containing all the native FHD detail.
    """
    mask = norm_expr([get_y(clip), get_y(rescaled)], "x y - abs", func=descale_detail_mask)

    mask = Morpho.binarize(mask, thr)

    if xxpand[0]:
        mask = iterate(mask, core.std.Maximum if xxpand[0] > 0 else core.std.Minimum, xxpand[0])

    if inflate:
        mask = iterate(mask, core.std.Inflate, inflate)

    if xxpand[1]:
        mask = iterate(mask, core.std.Maximum if xxpand[1] > 0 else core.std.Minimum, xxpand[1])

    return mask


@limiter
def descale_error_mask(
    clip: vs.VideoNode,
    rescaled: vs.VideoNode,
    thr: float | list[float] = 0.038,
    expands: int | tuple[int, int, int] = (2, 2, 3),
    blur: int | float = 3,
    bwbias: int = 1,
    tr: int = 0,
) -> ConstantFormatVideoNode:
    """
    Create an error mask from the original and rescaled clip.

    Args:
        clip: Original clip.
        rescaled: Rescaled clip.
        thr: Threshold of the minimum difference.
        expands: Iterations of mask expand at each step (diff, expand, binarize).
        blur: How much to blur the clip. If int, it will be a box_blur, else gauss_blur.
        bwbias: Calculate a bias with the clip's chroma.
        tr: Make the error mask temporally stable with a temporal radius.

    Returns:
        Descale error mask.
    """
    y, *chroma = split(clip)

    error = norm_expr([y, rescaled], "x y - abs", func=descale_error_mask)

    if bwbias > 1 and chroma:
        chroma_abs = norm_expr(chroma, "x neutral - abs y neutral - abs max")
        chroma_abs = core.resize.Bicubic(chroma_abs, y.width, y.height)

        bias = norm_expr([y, chroma_abs], f"x ymax >= x ymin <= or y not and {bwbias} 1 ?", func=descale_error_mask)
        bias = Morpho.expand(bias, 2, func=descale_error_mask)

        error = ExprOp.MUL(error, bias)

    if isinstance(expands, int):
        exp1 = exp2 = exp3 = expands
    else:
        exp1, exp2, exp3 = expands

    if exp1:
        error = Morpho.expand(error, exp1, func=descale_error_mask)

    if exp2:
        error = Morpho.expand(error, exp2, mode=XxpandMode.ELLIPSE, func=descale_error_mask)

    thrs = [thr] if isinstance(thr, (float, int)) else thr

    error = Morpho.binarize(error, thrs[0])

    for scaled_thr in thrs[1:]:
        bin2 = Morpho.binarize(error, scaled_thr)
        error = bin2.hysteresis.Hysteresis(error)

    if exp3:
        error = Morpho.expand(error, exp2, mode=XxpandMode.ELLIPSE, func=descale_error_mask)

    if tr:
        avg = Morpho.binarize(BlurMatrix.MEAN(taps=tr, mode=ConvMode.TEMPORAL)(error), 0.5)

        error = ExprOp.MIN(error, ExprOp.MAX(shift_clip_multi(ExprOp.MIN(error, avg), (-tr, tr))))

    error = box_blur(error, blur) if isinstance(blur, int) else gauss_blur(error, blur)

    return error
