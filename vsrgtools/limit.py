from __future__ import annotations

from typing import Sequence

from jetpytools import fallback, to_arr

from vsexprtools import norm_expr
from vsjetpack import deprecated
from vstools import Planes, scale_delta, vs

__all__ = ["limit_filter"]


@deprecated(
    "limit_filter is deprecated and will be removed in a future version. Use vszip.LimitFilter instead.",
    category=DeprecationWarning,
)
def limit_filter(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    ref: vs.VideoNode | None = None,
    dark_thr: float | Sequence[float] = 1.0,
    bright_thr: float | Sequence[float] = 1.0,
    elast: float | Sequence[float] = 2.0,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    Performs a soft-limiting between two clips to limit the difference of filtering while avoiding artifacts.

    Args:
        flt: Filtered clip.
        src: Source clip.
        ref: Reference clip, to compute the weight to be applied on filtering diff.
        dark_thr: Threshold (8-bit scale) to limit dark filtering diff.
        bright_thr: Threshold (8-bit scale) to limit bright filtering diff.
        elast: Elasticity of the soft threshold.
        planes: Planes to process. Defaults to all.

    Returns:
        Limited clip.
    """

    return norm_expr(
        [flt, src, fallback(ref, src)],
        "x z - dup {bright_thr} {dark_thr} ? THR1! abs DIFF! THR1@ {elast} * THR2! "
        "DIFF@ THR1@ <= x DIFF@ THR2@ >= y y x y - THR2@ DIFF@ - * THR2@ THR1@ - / + ? ?",
        dark_thr=[scale_delta(x, 8, flt) for x in to_arr(dark_thr)],
        bright_thr=[scale_delta(x, 8, flt) for x in to_arr(bright_thr)],
        elast=elast,
        planes=planes,
    )
