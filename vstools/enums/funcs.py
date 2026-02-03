from __future__ import annotations

from enum import Flag, auto
from typing import Literal

from jetpytools import CustomStrEnum

__all__ = [
    "Align",
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
        return self == "s"

    @property
    def is_spatial(self) -> bool:
        return self in ["s", "v", "h", "hv"]

    @property
    def is_temporal(self) -> bool:
        return self == "t"


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


class Align(Flag):
    """Defines alignment flags for positioning elements vertically and horizontally."""

    TOP = auto()
    """Align to the top."""

    MIDDLE = auto()
    """Align to the vertical center."""

    BOTTOM = auto()
    """Align to the bottom."""

    LEFT = auto()
    """Align to the left."""

    CENTER = auto()
    """Align to the horizontal center."""

    RIGHT = auto()
    """Align to the right."""

    TOP_LEFT = TOP | LEFT
    """Align to the top-left corner."""

    TOP_CENTER = TOP | CENTER
    """Align to the top-center."""

    TOP_RIGHT = TOP | RIGHT
    """Align to the top-right corner."""

    MIDDLE_LEFT = MIDDLE | LEFT
    """Align to the middle-left."""

    MIDDLE_CENTER = MIDDLE | CENTER
    """Align to the center (both vertically and horizontally)."""

    MIDDLE_RIGHT = MIDDLE | RIGHT
    """Align to the middle-right."""

    BOTTOM_LEFT = BOTTOM | LEFT
    """Align to the bottom-left corner."""

    BOTTOM_CENTER = BOTTOM | CENTER
    """Align to the bottom-center."""

    BOTTOM_RIGHT = BOTTOM | RIGHT
    """Align to the bottom-right corner."""
