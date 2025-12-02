"""
1D matrices
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import Any, ClassVar

from vstools import core, vs

from ._abstract import EdgeDetect, EdgeMasksEdgeDetect, EuclideanDistance, RidgeDetect

__all__ = ["Matrix1D", "TEdge", "Tritical"]


class Matrix1D(EdgeDetect, ABC):
    """
    Abstract base class for one-dimensional convolution-based edge detectors.
    """

    mode_types: ClassVar[Sequence[str] | None] = ["h", "v"]


class Tritical(Matrix1D, EdgeMasksEdgeDetect, RidgeDetect, EuclideanDistance):
    """
    Operator used in Tritical's original TCanny filter.
    Plain and simple orthogonal first order derivative.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[-1, 0, 1], [1, 0, -1]]


class TEdge(Matrix1D, EuclideanDistance):
    """
    (TEdgeMasktype=2) Avisynth plugin.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[12, -74, 0, 74, -12], [-12, 74, 0, -74, 12]]
    divisors: ClassVar[Sequence[float] | None] = [62, 62]

    def _compute_edge_mask(
        self,
        clip: vs.VideoNode,
        *,
        multi: float | Sequence[float] = 1,
        planes: int | Sequence[int] | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        if (
            hasattr(core, "tedgemask")
            and max(clip.format.subsampling_h, clip.format.subsampling_w) <= 2
            and clip.format.bits_per_sample <= 16
            and not isinstance(multi, Sequence)
        ):
            return clip.tedgemask.TEdgeMask(0, 2, scale=multi, planes=planes, **kwargs)

        return super()._compute_edge_mask(clip, multi=multi, planes=planes, **kwargs)
