"""
1D matrices
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Sequence

from vsexprtools import norm_expr
from vstools import ConstantFormatVideoNode, Planes, core

from ._abstract import EdgeDetect, EuclideanDistance

__all__ = [
    "Matrix1D",
    "TEdge",
    "TEdgeTedgemask",
]


class Matrix1D(EdgeDetect, ABC):
    """
    Abstract base class for one-dimensional convolution-based edge detectors.
    """


class TEdge(EuclideanDistance, Matrix1D):
    """
    (TEdgeMasktype=2) Avisynth plugin.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[12, -74, 0, 74, -12], [-12, 74, 0, -74, 12]]
    divisors: ClassVar[Sequence[float] | None] = [62, 62]
    mode_types: ClassVar[Sequence[str] | None] = ["h", "v"]

class TEdgeTedgemask(Matrix1D):
    """
    (tedgemask.TEdgeMask(threshold=0.0, type=2)) Vapoursynth plugin.
    """

    def _compute_edge_mask(
        self,
        clip: ConstantFormatVideoNode,
        *,
        multi: float | Sequence[float] = 1.0,
        planes: Planes = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        if not isinstance(multi, Sequence):
            return clip.tedgemask.TEdgeMask(0, 2, scale=multi, planes=planes, **kwargs)

        return norm_expr(
            clip.tedgemask.TEdgeMask(0, 2, **kwargs), "x {multi} *", planes, func=self.__class__, multi=multi
        )
