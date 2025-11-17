"""
2D matrices.
"""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Sequence

from vsjetpack import deprecated

from ._abstract import (
    EdgeDetect,
    EdgeMasksEdgeDetect,
    EuclideanDistance,
    MagnitudeEdgeMasks,
    Max,
    NormalizeProcessor,
    RidgeDetect,
    SingleMatrix,
    TCannyEdgeDetect,
)

# ruff: noqa: RUF022

__all__ = [
    "Matrix5x5",
    # Single matrix
    "ExLaplacian1",
    "ExLaplacian2",
    "ExLaplacian3",
    "ExLaplacian4",
    "LoG",
    # Euclidean distance
    "ExPrewitt",
    "ExSobel",
    "FDoG",
    "FDoGTCanny",
    "Farid",
    # Max
    "ExKirsch",
]


class Matrix5x5(EdgeDetect, ABC):
    """
    Abstract base class for 5x5 convolution-based edge detectors.
    """


# Single matrix
class ExLaplacian1(SingleMatrix, Matrix5x5):
    """
    Extended Pierre-Simon de Laplace operator, 1st implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [0, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, -1, 8, -1, -1, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0]
    ]


class ExLaplacian2(SingleMatrix, Matrix5x5):
    """
    Extended Pierre-Simon de Laplace operator, 2nd implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [0, 1, -1, 1, 0, 1, 1, -4, 1, 1, -1, -4, 8, -4, -1, 1, 1, -4, 1, 1, 0, 1, -1, 1, 0]
    ]


class ExLaplacian3(SingleMatrix, Matrix5x5):
    """
    Extended Pierre-Simon de Laplace operator, 3rd implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [-1, 1, -1, 1, -1, 1, 2, -4, 2, 1, -1, -4, 8, -4, -1, 1, 2, -4, 2, 1, -1, 1, -1, 1, -1]
    ]


class ExLaplacian4(SingleMatrix, Matrix5x5):
    """
    Extended Pierre-Simon de Laplace operator, 4th implementation.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]


class LoG(SingleMatrix, Matrix5x5):
    """
    Laplacian of Gaussian operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0]
    ]


# Euclidean distance
class ExPrewitt(EdgeMasksEdgeDetect, RidgeDetect, EuclideanDistance, Matrix5x5):
    """
    Extended Judith M. S. Prewitt operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
        [2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -2, -2, -2, -2, -2],
    ]


class ExSobel(EdgeMasksEdgeDetect, RidgeDetect, EuclideanDistance, Matrix5x5):
    """
    Extended Sobel-Feldman operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [2, 1, 0, -1, -2, 2, 1, 0, -1, -2, 4, 2, 0, -2, -4, 2, 1, 0, -1, -2, 2, 1, 0, -1, -2],
        [2, 2, 4, 2, 2, 1, 1, 2, 1, 1, 0, 0, 0, 0, 0, -1, -1, -2, -1, -1, -2, -2, -4, -2, -2],
    ]


class FDoG(EdgeMasksEdgeDetect, RidgeDetect, EuclideanDistance, Matrix5x5):
    """
    Flow-based Difference of Gaussian
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1],
        [1, 2, 3, 2, 1, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, -1, -2, -3, -2, -1, -1, -2, -3, -2, -1],
    ]
    divisors: ClassVar[Sequence[float] | None] = [2, 2]


@deprecated(
    "FDoGTCanny is deprecated and will be removed in a future version. "
    "Please use FDoG and install the 'edgemasks' plugin instead.",
    category=DeprecationWarning,
)
class FDoGTCanny(TCannyEdgeDetect, Matrix5x5):
    """
    Flow-based Difference of Gaussian TCanny Vapoursynth plugin.
    """

    _op = 6
    _scale = 1 / 2


class Farid(NormalizeProcessor, RidgeDetect, EuclideanDistance, Matrix5x5):
    """
    Farid & Simoncelli operator.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [
            0.004127602875174862,
            0.027308149775363867,
            0.04673225765917656,
            0.027308149775363867,
            0.004127602875174862,
            0.010419993699470744,
            0.06893849946536831,
            0.11797400212587895,
            0.06893849946536831,
            0.010419993699470744,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.010419993699470744,
            -0.06893849946536831,
            -0.11797400212587895,
            -0.06893849946536831,
            -0.010419993699470744,
            -0.004127602875174862,
            -0.027308149775363867,
            -0.04673225765917656,
            -0.027308149775363867,
            -0.004127602875174862,
        ],
        [
            0.004127602875174862,
            0.010419993699470744,
            0.0,
            -0.010419993699470744,
            -0.004127602875174862,
            0.027308149775363867,
            0.06893849946536831,
            0.0,
            -0.06893849946536831,
            -0.027308149775363867,
            0.04673225765917656,
            0.11797400212587895,
            0.0,
            -0.11797400212587895,
            -0.04673225765917656,
            0.027308149775363867,
            0.06893849946536831,
            0.0,
            -0.06893849946536831,
            -0.027308149775363867,
            0.004127602875174862,
            0.010419993699470744,
            0.0,
            -0.010419993699470744,
            -0.004127602875174862,
        ],
    ]


# Max
class ExKirsch(MagnitudeEdgeMasks, Max):
    """
    Extended Russell Kirsch compass operator. 5x5 matrices.
    """

    matrices: ClassVar[Sequence[Sequence[float]]] = [
        [9, 9, 9, 9, 9, 9, 5, 5, 5, 9, -7, -3, 0, -3, -7, -7, -3, -3, -3, -7, -7, -7, -7, -7, -7],
        [9, 9, 9, 9, -7, 9, 5, 5, -3, -7, 9, 5, 0, -3, -7, 9, -3, -3, -3, -7, -7, -7, -7, -7, -7],
        [9, 9, -7, -7, -7, 9, 5, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, -3, -3, -7, 9, 9, -7, -7, -7],
        [-7, -7, -7, -7, -7, 9, -3, -3, -3, -7, 9, 5, 0, -3, -7, 9, 5, 5, -3, -7, 9, 9, 9, 9, -7],
        [-7, -7, -7, -7, -7, -7, -3, -3, -3, -7, -7, -3, 0, -3, -7, 9, 5, 5, 5, 9, 9, 9, 9, 9, 9],
        [-7, -7, -7, -7, -7, -7, -3, -3, -3, 9, -7, -3, 0, 5, 9, -7, -3, 5, 5, 9, -7, 9, 9, 9, 9],
        [-7, -7, -7, 9, 9, -7, -3, -3, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, 5, 9, -7, -7, -7, 9, 9],
        [-7, 9, 9, 9, 9, -7, -3, 5, 5, 9, -7, -3, 0, 5, 9, -7, -3, -3, -3, 9, -7, -7, -7, -7, -7],
    ]
