"""
2D matrices.
"""

from __future__ import annotations

import math
from abc import ABC
from typing import Any, ClassVar, Sequence

from vsexprtools import ExprOp, norm_expr
from vstools import (
    ConstantFormatVideoNode,
    KwargsT,
    get_depth,
    join,
    split,
    vs,
)

from ..morpho import Morpho
from ..types import XxpandMode
from ._abstract import (
    EdgeDetect,
    EuclideanDistance,
    MagnitudeMatrix,
    MatrixEdgeDetect,
    Max,
    NormalizeProcessor,
    RidgeDetect,
    SingleMatrix,
)

# ruff: noqa: RUF022

__all__ = [
    "Matrix3x3",
    # Single matrix
    "Laplacian1",
    "Laplacian2",
    "Laplacian3",
    "Laplacian4",
    "Kayyali",
    # Euclidean Distance
    "Tritical",
    "TriticalTCanny",
    "Cross",
    "Prewitt",
    "PrewittStd",
    "PrewittTCanny",
    "Sobel",
    "SobelStd",
    "SobelTCanny",
    "ASobel",
    "Scharr",
    "RScharr",
    "ScharrTCanny",
    "Kroon",
    "KroonTCanny",
    "FreyChenG41",
    "FreyChen",
    # Max
    "Robinson3",
    "Robinson5",
    "TheToof",
    "Kirsch",
    "KirschTCanny",
    # Misc
    "MinMax",
]


class Matrix3x3(EdgeDetect, ABC): ...


# Single matrix
class Laplacian1(SingleMatrix, Matrix3x3):
    """
    Pierre-Simon de Laplace operator 1st implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[0, -1, 0, -1, 4, -1, 0, -1, 0]]


class Laplacian2(SingleMatrix, Matrix3x3):
    """
    Pierre-Simon de Laplace operator 2nd implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[1, -2, 1, -2, 4, -2, 1, -2, 1]]


class Laplacian3(SingleMatrix, Matrix3x3):
    """
    Pierre-Simon de Laplace operator 3rd implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[2, -1, 2, -1, -4, -1, 2, -1, 2]]


class Laplacian4(SingleMatrix, Matrix3x3):
    """
    Pierre-Simon de Laplace operator 4th implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[-1, -1, -1, -1, 8, -1, -1, -1, -1]]


class Kayyali(SingleMatrix, Matrix3x3):
    """
    Kayyali operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[6, 0, -6, 0, 0, 0, -6, 0, 6]]


# Euclidean Distance
class Tritical(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    Operator used in Tritical's original TCanny filter.
    Plain and simple orthogonal first order derivative.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[0, 0, 0, -1, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, -1, 0]]


class TriticalTCanny(Matrix3x3, EdgeDetect):
    """
    Operator used in Tritical's original TCanny filter.
    Plain and simple orthogonal first order derivative.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        _scale_constant = 1 if clip.format.sample_type == vs.FLOAT else (7 / 3 * 1.002 - 0.0001) / 2

        return clip.tcanny.TCanny(op=0, **(KwargsT(sigma=0, mode=1, scale=_scale_constant) | kwargs))


class Cross(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    "HotDoG" Operator from AVS ExTools by Dogway.
    Plain and simple cross first order derivative.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, -1, 0, 0, 0, 1, 0, 0]]


class Prewitt(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    Judith M. S. Prewitt operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[1, 0, -1, 1, 0, -1, 1, 0, -1], [1, 1, 1, 0, 0, 0, -1, -1, -1]]


class PrewittStd(Matrix3x3, EdgeDetect):
    """
    Judith M. S. Prewitt Vapoursynth plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return clip.std.Prewitt(**kwargs)


class PrewittTCanny(Matrix3x3, EdgeDetect):
    """
    Judith M. S. Prewitt TCanny plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        _scale_constant = 2 if clip.format.sample_type == vs.FLOAT else 7 / 3 * 1.002 - 0.0001

        return clip.tcanny.TCanny(op=1, **(KwargsT(sigma=0, mode=1, scale=_scale_constant) | kwargs))


class Sobel(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    Sobel-Feldman operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[1, 0, -1, 2, 0, -2, 1, 0, -1], [1, 2, 1, 0, 0, 0, -1, -2, -1]]


class SobelStd(Matrix3x3, EdgeDetect):
    """
    Sobel-Feldman Vapoursynth plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return clip.std.Sobel(**kwargs)


class SobelTCanny(Matrix3x3, EdgeDetect):
    """
    Sobel-Feldman Vapoursynth plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        _scale_constant = 1.0 if clip.format.sample_type == vs.FLOAT else (7 / 3 * 1.002 - 0.0001) / 2

        return clip.tcanny.TCanny(op=2, **(KwargsT(sigma=0, mode=1, scale=_scale_constant) | kwargs))


class ASobel(Matrix3x3, EdgeDetect):
    """
    Modified Sobel-Feldman operator from AWarpSharp.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return (vs.core.warp.ASobel if get_depth(clip) < 32 else vs.core.warpsf.ASobel)(clip, 255, **kwargs)


class Scharr(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    Original H. Scharr optimised operator which attempts
    to achieve the perfect rotational symmetry with coefficients 3 and 10.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[-3, 0, 3, -10, 0, 10, -3, 0, 3], [-3, -10, -3, 0, 0, 0, 3, 10, 3]]
    divisors: ClassVar[Sequence[float] | None] = [3, 3]


class RScharr(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    Refined H. Scharr operator to more accurately calculate
    1st derivatives for a 3x3 kernel with coeffs 47 and 162.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [-47, 0, 47, -162, 0, 162, -47, 0, 47],
        [-47, -162, -47, 0, 0, 0, 47, 162, 47],
    ]
    divisors: ClassVar[Sequence[float] | None] = [47, 47]


class ScharrTCanny(Matrix3x3, EdgeDetect):
    """
    H. Scharr optimised TCanny Vapoursynth plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        _scale_constant = 1 / 3 if clip.format.sample_type == vs.FLOAT else 1 / ((3 - math.sqrt(3) / 4) - 0.00053)

        return clip.tcanny.TCanny(op=3, **(KwargsT(sigma=0, mode=1, scale=_scale_constant) | kwargs))


class Kroon(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    Dirk-Jan Kroon operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [-17, 0, 17, -61, 0, 61, -17, 0, 17],
        [-17, -61, -17, 0, 0, 0, 17, 61, 17],
    ]
    divisors: ClassVar[Sequence[float] | None] = [17, 17]


class KroonTCanny(Matrix3x3, EdgeDetect):
    """
    Dirk-Jan Kroon TCanny Vapoursynth plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        _scale_constant = 1 / 17 if clip.format.sample_type == vs.FLOAT else 1 / 14.54325

        return clip.tcanny.TCanny(op=4, **(KwargsT(sigma=0, mode=1, scale=_scale_constant) | kwargs))


class FreyChen(NormalizeProcessor, MatrixEdgeDetect):
    """
    Chen Frei operator. 3x3 matrices properly implemented.
    """

    sqrt2 = math.sqrt(2)
    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [1, sqrt2, 1, 0, 0, 0, -1, -sqrt2, -1],
        [1, 0, -1, sqrt2, 0, -sqrt2, 1, 0, -1],
        [0, -1, sqrt2, 1, 0, -1, -sqrt2, 1, 0],
        [sqrt2, -1, 0, -1, 0, 1, 0, 1, -sqrt2],
        [0, 1, 0, -1, 0, -1, 0, 1, 0],
        [-1, 0, 1, 0, 0, 0, 1, 0, -1],
        [1, -2, 1, -2, 4, -2, 1, -2, 1],
        [-2, 1, -2, 1, 4, 1, -2, 1, -2],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
    divisors: ClassVar[Sequence[float] | None] = [2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2 * sqrt2, 2, 2, 6, 6, 3]

    def _merge_edge(self, clips: Sequence[ConstantFormatVideoNode], **kwargs: Any) -> ConstantFormatVideoNode:
        M = "x dup * y dup * + z dup * + a dup * +"  # noqa: N806
        S = f"b dup * c dup * + d dup * + e dup * + f dup * + {M} +"  # noqa: N806
        return norm_expr(clips, f"{M} {S} / sqrt", kwargs.get("planes"), func=self.__class__)


class FreyChenG41(RidgeDetect, EuclideanDistance, Matrix3x3):
    """
    "Chen Frei" operator. 3x3 matrices from G41Fun.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [[-7, 0, 7, -10, 0, 10, -7, 0, 7], [-7, -10, -7, 0, 0, 0, 7, 10, 7]]
    divisors: ClassVar[Sequence[float] | None] = [7, 7]


# Max
class Robinson3(MagnitudeMatrix, Max, Matrix3x3):
    """
    Robinson compass operator level 3.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [1, 1, 1, 0, 0, 0, -1, -1, -1],  # N
        [1, 1, 0, 1, 0, -1, 0, -1, -1],  # NW
        [1, 0, -1, 1, 0, -1, 1, 0, -1],  # W
        [0, -1, -1, 1, 0, -1, 1, 1, 0],  # SW
        [],
        [],
        [],
        [],
    ]


class Robinson5(MagnitudeMatrix, Max, Matrix3x3):
    """
    Robinson compass operator level 5.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [1, 2, 1, 0, 0, 0, -1, -2, -1],  # N
        [2, 1, 0, 1, 0, -1, 0, -1, -2],  # NW
        [1, 0, -1, 2, 0, -2, 1, 0, -1],  # W
        [0, -1, -2, 1, 0, -1, 2, 1, 0],  # SW
        [],
        [],
        [],
        [],
    ]


class TheToof(MagnitudeMatrix, Max, Matrix3x3):
    """
    TheToof compass operator from SharpAAMCmod.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [5, 10, 5, 0, 0, 0, -5, -10, -5],  # N
        [10, 5, 0, 5, 0, -5, 0, -5, -10],  # NW
        [5, 0, -5, 10, 0, -10, 5, 0, -5],  # W
        [0, -5, -10, 5, 0, -5, 10, 5, 0],  # SW
        [],
        [],
        [],
        [],
    ]
    divisors: ClassVar[Sequence[float] | None] = [4] * 4


class Kirsch(MagnitudeMatrix, Max, Matrix3x3):
    """
    Russell Kirsch compass operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [5, 5, 5, -3, 0, -3, -3, -3, -3],  # N
        [5, 5, -3, 5, 0, -3, -3, -3, -3],  # NW
        [5, -3, -3, 5, 0, -3, 5, -3, -3],  # W
        [-3, -3, -3, 5, 0, -3, 5, 5, -3],  # SW
        [-3, -3, -3, -3, 0, -3, 5, 5, 5],  # S
        [-3, -3, -3, -3, 0, 5, -3, 5, 5],  # SE
        [-3, -3, 5, -3, 0, 5, -3, -3, 5],  # E
        [-3, 5, 5, -3, 0, 5, -3, -3, -3],  # NE
    ]


class KirschTCanny(Matrix3x3, EdgeDetect):
    """
    Russell Kirsch compass TCanny Vapoursynth plugin operator.
    """

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return clip.tcanny.TCanny(op=5, **(KwargsT(sigma=0, mode=1) | kwargs))


# Misc
class MinMax(EdgeDetect):
    """
    Min/max mask with separate luma/chroma radii.
    """

    rady: int
    radc: int

    def __init__(self, rady: int = 2, radc: int = 0, **kwargs: Any) -> None:
        self.rady = rady
        self.radc = radc
        super().__init__(**kwargs)

    def _compute_edge_mask(self, clip: ConstantFormatVideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return join(
            [
                ExprOp.SUB.combine(
                    Morpho.expand(p, rad, rad, XxpandMode.ELLIPSE, **kwargs),
                    Morpho.inpand(p, rad, rad, XxpandMode.ELLIPSE, **kwargs),
                    planes=kwargs.get("planes"),
                    func=self.__class__,
                )
                if rad > 0
                else p
                for p, rad in zip(split(clip), (self.rady, self.radc, self.radc)[: clip.format.num_planes])
            ],
            clip.format.color_family,
        )
