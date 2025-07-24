from __future__ import annotations

from typing import Sequence

from vsexprtools import norm_expr
from vstools import PlanesT, fallback, vs

__all__ = ["limit_filter"]


def limit_filter(
    flt: vs.VideoNode,
    src: vs.VideoNode,
    ref: vs.VideoNode | None = None,
    dark_thr: float | Sequence[float] = 1.0,
    bright_thr: float | Sequence[float] = 1.0,
    elast: float | Sequence[float] = 2.0,
    planes: PlanesT = None,
) -> vs.VideoNode:
    ref = fallback(ref, src)

    base_expr = "x y - DIFF! x z - abs DIFF_REF_ABS!"
    dark_expr = (
        "{dark_thr} range_size 256 / * THR1! THR1@ {elast} * THR2! "
        "1 THR2@ THR1@ - / SLOPE! "
        "DIFF_REF_ABS@ THR1@ <= x DIFF_REF_ABS@ THR2@ >= y "
        "y DIFF@ THR2@ DIFF_REF_ABS@ - * SLOPE@ * + ? ?"
    )
    bright_expr = dark_expr.replace("dark_thr", "bright_thr")

    return norm_expr(
        [flt, src, ref],
        f"{base_expr} x z < {dark_expr} {bright_expr} ?",
        dark_thr=dark_thr,
        bright_thr=bright_thr,
        elast=elast,
        planes=planes,
    )
