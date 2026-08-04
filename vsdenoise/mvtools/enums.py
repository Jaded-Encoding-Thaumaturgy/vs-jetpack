from __future__ import annotations

from enum import IntFlag

from jetpytools import CustomIntEnum

__all__ = ["MVDirection", "MaskMode", "PenaltyMode", "RFilterMode", "SearchMode", "SharpMode"]


class MVDirection(IntFlag):
    """
    Motion vector analyze direction.
    """

    BACKWARD = 1
    """
    Backward motion compensation.
    """

    FORWARD = 2
    """
    Forward motion compensation.
    """

    BOTH = BACKWARD | FORWARD
    """
    Backward and forward motion compensation.
    """


class SharpMode(CustomIntEnum):
    """
    Subpixel interpolation method for pel = 2 or 4.

    This enum controls the calculation of the first level only.
    If pel=4, bilinear interpolation is always used to compute the second level.
    """

    BILINEAR = 0
    """
    Soft bilinear interpolation.
    """

    BICUBIC = 1
    """
    Bicubic interpolation (4-tap Catmull-Rom).
    """

    WIENER = 2
    """
    Sharper Wiener interpolation (6-tap, similar to Lanczos).
    """


class RFilterMode(CustomIntEnum):
    """
    Hierarchical levels smoothing and reducing (halving) filter.
    """

    AVERAGE = 0
    """
    Simple 4 pixels averaging.
    """

    BILINEAR = 1
    """
    Triangle filter for even more smoothing.
    """

    CUBIC = 2
    """
    Cubic filter for even more smoothing.
    """


class SearchMode(CustomIntEnum):
    """
    Decides the type of search at every level.
    """

    DIAMOND = 0
    """
    Logarithmic search, also named Diamond Search.
    """

    EXHAUSTIVE = 1
    """
    Exhaustive search, square side is 2 * radius + 1. It's slow, but gives the best results SAD-wise.
    """

    HEXAGON = 2
    """
    Hexagon search (similar to x264).
    """

    UMH = 3
    """
    Uneven Multi Hexagon search (similar to x264).
    """

    EXHAUSTIVE_H = 4
    """
    Pure horizontal exhaustive search, width is 2 * radius + 1.
    """

    EXHAUSTIVE_V = 5
    """
    Pure vertical exhaustive search, height is 2 * radius + 1.
    """


class PenaltyMode(CustomIntEnum):
    """
    Controls how motion estimation penalties scale with hierarchical levels.
    """

    NONE = 0
    """
    Penalties remain constant across all hierarchical levels.
    """

    LINEAR = 1
    """
    Penalties scale linearly with hierarchical level size.
    """

    QUADRATIC = 2
    """
    Penalties scale quadratically with hierarchical level size.
    """


class MaskMode(CustomIntEnum):
    """
    Defines the type of analysis mask to generate.
    """

    VECTOR_LENGTH = 0
    """
    Generates a mask based on motion vector magnitudes.
    """

    SAD = 1
    """
    Generates a mask based on SAD (Sum of Absolute Differences) values.
    """

    OCCLUSION = 2
    """
    Generates a mask highlighting areas where motion estimation fails due to occlusion.
    """
