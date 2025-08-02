from __future__ import annotations

from typing import Literal, TypeAlias

from jetpytools import CustomIntEnum, CustomStrEnum

__all__ = [
    "Align",
    "BaseAlign",
    "ConvMode",
    "OneDimConvModeT",
    "OnePassConvModeT",
    "SpatialConvModeT",
    "TempConvModeT",
    "TwoDimConvModeT",
    "TwoPassesConvModeT",
]


class ConvMode(CustomStrEnum):
    """
    Convolution mode for .std.Convolution or std.AverageFrames
    """

    SQUARE = "s"
    """
    Square horizontal/vertical convolution.
    """

    VERTICAL = "v"
    """
    Vertical-only convolution.
    """

    HORIZONTAL = "h"
    """
    Horizontal-only convolution.
    """

    HV = "hv"
    """
    Horizontal and Vertical convolution
    """

    TEMPORAL = "t"
    """
    Temporal convolution
    """

    S = SQUARE
    """
    Alias for `ConvMode.SQUARE`
    """

    V = VERTICAL
    """
    Alias for `ConvMode.VERTICAL`
    """

    H = HORIZONTAL
    """
    Alias for `ConvMode.HORIZONTAL`
    """

    T = TEMPORAL
    """
    Alias for `ConvMode.TEMPORAL`
    """

    @property
    def is_one_dim(self) -> bool:
        return self in ["v", "h", "hv"]

    @property
    def is_two_dim(self) -> bool:
        return self in ["s"]

    @property
    def is_spatial(self) -> bool:
        return self in ["s", "v", "h", "hv"]

    @property
    def is_temporal(self) -> bool:
        return self in ["t"]


OnePassConvModeT: TypeAlias = Literal[ConvMode.SQUARE, ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.TEMPORAL]
"""Type alias for one pass convolution mode"""

TwoPassesConvModeT: TypeAlias = Literal[ConvMode.HV]
"""Type alias for two passes convolution mode"""

OneDimConvModeT: TypeAlias = Literal[ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.HV]
"""Type alias for one dimension convolution mode"""

TwoDimConvModeT: TypeAlias = Literal[ConvMode.SQUARE]
"""Type alias for two dimensions convolution mode"""

SpatialConvModeT: TypeAlias = Literal[ConvMode.SQUARE, ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.HV]
"""Type alias for spatial convolution mode only"""

TempConvModeT: TypeAlias = Literal[ConvMode.TEMPORAL]
"""Type alias for temporal convolution mode only"""


class BaseAlign(CustomIntEnum):
    TOP = 1
    MIDDLE = 2
    BOTTOM = 4
    LEFT = 8
    CENTER = 16
    RIGHT = 32


class Align(CustomIntEnum):
    TOP_LEFT = BaseAlign.TOP | BaseAlign.LEFT
    TOP_CENTER = BaseAlign.TOP | BaseAlign.CENTER
    TOP_RIGHT = BaseAlign.TOP | BaseAlign.RIGHT
    MIDDLE_LEFT = BaseAlign.MIDDLE | BaseAlign.LEFT
    MIDDLE_CENTER = BaseAlign.MIDDLE | BaseAlign.CENTER
    MIDDLE_RIGHT = BaseAlign.MIDDLE | BaseAlign.RIGHT
    BOTTOM_LEFT = BaseAlign.BOTTOM | BaseAlign.LEFT
    BOTTOM_CENTER = BaseAlign.BOTTOM | BaseAlign.CENTER
    BOTTOM_RIGHT = BaseAlign.BOTTOM | BaseAlign.RIGHT
