"""
1D matrices
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Sequence

from vstools import ConstantFormatVideoNode, vs

from ._abstract import EdgeDetect, EuclideanDistance

__all__ = [
    "Matrix1D",
    "TEdge",
    "TEdgeTedgemask",
]


class Matrix1D(EdgeDetect, ABC): ...


class TEdge(EuclideanDistance, Matrix1D):
    """
    (TEdgeMasktype=2) Avisynth plugin.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[12, -74, 0, 74, -12], [-12, 74, 0, -74, 12]]
    divisors: ClassVar[Sequence[float] | None] = [62, 62]
    mode_types: ClassVar[Sequence[str] | None] = ["h", "v"]


class TEdgeTedgemask(Matrix1D, EdgeDetect):
    """
    (tedgemask.TEdgeMask(threshold=0.0, type=2)) Vapoursynth plugin.
    """

    def _compute_edge_mask(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return clip.tedgemask.TEdgeMask(threshold=0, type=2)
