from __future__ import annotations

from enum import IntFlag
from typing import Any

from vstools import (
    ConstantFormatVideoNode,
    CustomIntEnum,
    CustomValueError,
    VSFunctionAllArgs,
    check_variable_format,
    core,
    fallback,
    vs,
)

__all__ = [
    "FlowMode",
    "MVDirection",
    "MVToolsPlugin",
    "MaskMode",
    "MotionMode",
    "PenaltyMode",
    "RFilterMode",
    "SADMode",
    "SearchMode",
    "SharpMode",
    "SmoothMode",
]


# ruff: noqa: N802
class MVToolsPlugin(CustomIntEnum):
    """
    Abstraction around both mvtools plugin versions.
    """

    INTEGER = 0
    """
    Original plugin. Only accepts integer 8-16 bits clips.
    """

    FLOAT = 1
    """
    Fork by IFeelBloated. Only works with float single precision clips.
    """

    @property
    def namespace(self) -> Any:
        """
        Get the appropriate MVTools namespace based on plugin type.
        """

        return core.proxied.mv if self is MVToolsPlugin.INTEGER else core.proxied.mvsf

    @property
    def Super(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Super function for creating motion vector clips.
        """

        return self.namespace.Super

    @property
    def Analyze(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Analyze function for analyzing motion vectors.
        """

        return self.namespace.Analyze if self is MVToolsPlugin.FLOAT else self.namespace.Analyse

    @property
    def Recalculate(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Recalculate function for refining motion vectors.
        """

        return self.namespace.Recalculate

    @property
    def Compensate(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Compensate function for motion compensation.
        """

        return self.namespace.Compensate

    @property
    def Flow(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Flow function for motion vector visualization.
        """

        return self.namespace.Flow

    @property
    def FlowInter(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the FlowInter function for motion-compensated frame interpolation.
        """

        return self.namespace.FlowInter

    @property
    def FlowBlur(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the FlowBlur function for motion-compensated frame blending.
        """

        return self.namespace.FlowBlur

    @property
    def FlowFPS(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the FlowFPS function for motion-compensated frame rate conversion.
        """

        return self.namespace.FlowFPS

    @property
    def BlockFPS(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the BlockFPS function for block-based frame rate conversion.
        """

        return self.namespace.BlockFPS

    @property
    def Mask(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Mask function for generating motion masks.
        """

        return self.namespace.Mask

    @property
    def SCDetection(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the SCDetection function for scene change detection.
        """

        return self.namespace.SCDetection

    def Degrain(self, tr: int | None = None) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the Degrain function for motion compensated denoising.
        """

        if tr is None and self is not MVToolsPlugin.FLOAT:
            raise CustomValueError("This implementation needs a temporal radius!", f"{self.name}.Degrain")

        try:
            return getattr(self.namespace, f"Degrain{fallback(tr, '')}")
        except AttributeError:
            raise CustomValueError("This temporal radius isn't supported!", f"{self.name}.Degrain", tr)

    @classmethod
    def from_video(cls, clip: vs.VideoNode) -> MVToolsPlugin:
        """
        Automatically select the appropriate plugin based on the given clip.

        Args:
            clip: The clip to process.

        Returns:
            The accompanying MVTools plugin for the clip.
        """
        assert check_variable_format(clip, cls.from_video)

        if clip.format.sample_type is vs.FLOAT:
            return MVToolsPlugin.FLOAT

        return MVToolsPlugin.INTEGER


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


class SmoothMode(CustomIntEnum):
    """
    This is method for dividing coarse blocks into smaller ones.
    """

    NEAREST = 0
    """
    Use motion of nearest block.
    """

    BILINEAR = 1
    """
    Bilinear interpolation of 4 neighbors.
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
