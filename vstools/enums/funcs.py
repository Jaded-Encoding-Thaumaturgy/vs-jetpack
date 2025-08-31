from __future__ import annotations

from typing import Literal

from jetpytools import CustomIntEnum, CustomStrEnum

__all__ = [
    "Align",
    "BaseAlign",
    "ConvMode",
    "OneDimConvMode",
    "OnePassConvMode",
    "SpatialConvMode",
    "TempConvMode",
    "TwoDimConvMode",
    "TwoPassesConvMode",
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


type OnePassConvMode = Literal[ConvMode.SQUARE, ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.TEMPORAL]
"""Type alias for one pass convolution mode"""

type TwoPassesConvMode = Literal[ConvMode.HV]
"""Type alias for two passes convolution mode"""

type OneDimConvMode = Literal[ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.HV]
"""Type alias for one dimension convolution mode"""

type TwoDimConvMode = Literal[ConvMode.SQUARE]
"""Type alias for two dimensions convolution mode"""

type SpatialConvMode = Literal[ConvMode.SQUARE, ConvMode.HORIZONTAL, ConvMode.VERTICAL, ConvMode.HV]
"""Type alias for spatial convolution mode only"""

type TempConvMode = Literal[ConvMode.TEMPORAL]
"""Type alias for temporal convolution mode only"""

OnePassConvModeT = OnePassConvMode
"""Deprecated alias of OnePassConvMode"""

TwoPassesConvModeT = TwoPassesConvMode
"""Deprecated alias of TwoPassesConvMode"""

OneDimConvModeT = OneDimConvMode
"""Deprecated alias of OneDimConvMode"""

TwoDimConvModeT = TwoDimConvMode
"""Deprecated alias of TwoDimConvMode"""

SpatialConvModeT = SpatialConvMode
"""Deprecated alias of SpatialConvMode"""

TempConvModeT = TempConvMode
"""Deprecated alias of TempConvMode"""


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
