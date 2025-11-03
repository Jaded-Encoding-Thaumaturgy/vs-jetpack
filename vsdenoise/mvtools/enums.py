from __future__ import annotations

from enum import IntFlag

from jetpytools import CustomIntEnum

__all__ = [
    "FlowMode",
    "MVDirection",
    "MaskMode",
    "MotionMode",
    "PenaltyMode",
    "RFilterMode",
    "SADMode",
    "SearchMode",
    "SharpMode",
]


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

    TRIANGLE_SHIFTED = 1
    """
    Triangle (shifted) filter for more smoothing (decrease aliasing).
    """

    TRIANGLE = 2
    """
    Triangle filter for even more smoothing.
    """

    QUADRATIC = 3
    """
    Quadratic filter for even more smoothing.
    """

    CUBIC = 4
    """
    Cubic filter for even more smoothing.
    """


class SearchMode(CustomIntEnum):
    """
    Decides the type of search at every level.
    """

    ONETIME = 0
    """
    One time search.
    """

    NSTEP = 1
    """
    N step searches.
    """

    DIAMOND = 2
    """
    Logarithmic search, also named Diamond Search.
    """

    EXHAUSTIVE = 3
    """
    Exhaustive search, square side is 2 * radius + 1. It's slow, but gives the best results SAD-wise.
    """

    HEXAGON = 4
    """
    Hexagon search (similar to x264).
    """

    UMH = 5
    """
    Uneven Multi Hexagon search (similar to x264).
    """

    EXHAUSTIVE_H = 6
    """
    Pure horizontal exhaustive search, width is 2 * radius + 1.
    """

    EXHAUSTIVE_V = 7
    """
    Pure vertical exhaustive search, height is 2 * radius + 1.
    """


class SADMode(CustomIntEnum):
    """
    Specifies how block differences (SAD) are calculated between frames.
    Can use spatial data, DCT coefficients, SATD, or combinations to improve motion estimation.
    """

    SPATIAL = 0
    """
    Calculate differences using raw pixel values in spatial domain.
    """

    DCT = 1
    """
    Calculate differences using DCT coefficients. Slower, especially for block sizes other than 8x8.
    """

    MIXED_SPATIAL_DCT = 2
    """
    Use both spatial and DCT data, weighted based on the average luma difference between frames.
    """

    ADAPTIVE_SPATIAL_MIXED = 3
    """
    Adaptively choose between spatial data or an equal mix of spatial and DCT data for each block.
    """

    ADAPTIVE_SPATIAL_DCT = 4
    """
    Adaptively choose between spatial data or DCT-weighted mixed mode for each block.
    """

    SATD = 5
    """
    Use Sum of Absolute Transformed Differences (SATD) instead of SAD for luma comparison.
    """

    MIXED_SATD_DCT = 6
    """
    Use both SATD and DCT data, weighted based on the average luma difference between frames.
    """

    ADAPTIVE_SATD_MIXED = 7
    """
    Adaptively choose between SATD data or an equal mix of SATD and DCT data for each block.
    """

    ADAPTIVE_SATD_DCT = 8
    """
    Adaptively choose between SATD data or DCT-weighted mixed mode for each block.
    """

    MIXED_SADEQSATD_DCT = 9
    """
    Mix of SAD, SATD and DCT data. Weight varies from SAD-only to equal SAD/SATD mix.
    """

    ADAPTIVE_SATD_LUMA = 10
    """
    Adaptively use SATD weighted by SAD, but only when there are significant luma changes.
    """


class MotionMode(CustomIntEnum):
    """
    Controls how motion vectors are searched and selected.

    Provides presets that configure multiple motion estimation parameters like lambda,
    LSAD threshold, and penalty values to optimize for either raw SAD scores or motion coherence.
    """

    SAD = 0
    """
    Optimize purely for lowest SAD scores when searching motion vectors.
    """

    COHERENCE = 1
    """
    Optimize for motion vector coherence, preferring vectors that align with surrounding blocks.
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


class FlowMode(CustomIntEnum):
    """
    Controls how motion vectors are applied to pixels.
    """

    ABSOLUTE = 0
    """
    Motion vectors point directly to destination pixels.
    """

    RELATIVE = 1
    """
    Motion vectors describe how source pixels should be shifted.
    """


class MaskMode(CustomIntEnum):
    """
    Defines the type of analysis mask to generate.
    """

    MOTION = 0
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

    HORIZONTAL = 3
    """
    Visualizes horizontal motion vector components. Values are in pixels + 128.
    """

    VERTICAL = 4
    """
    Visualizes vertical motion vector components. Values are in pixels + 128.
    """

    COLORMAP = 5
    """
    Creates a color visualization of motion vectors, mapping x/y components to U/V planes.
    """
