from vstools import CustomEnum, CustomIntEnum, vs

__all__ = [
    'VFMMode',
    'IVTCycles',

    'InputType',
    'SearchPostProcess',
    'LosslessMode',
    'NoiseProcessMode',
    'NoiseDeintMode',
    'SharpMode',
    'SharpLimitMode',
    'BackBlendMode',
    'SourceMatchMode'
]


class VFMMode(CustomIntEnum):
    """
    Enum representing different matching modes for VFM.

    The mode determines the strategy used for matching fields and frames.
    Higher modes generally offer better matching in complex scenarios but
    may introduce more risk of jerkiness or duplicate frames.
    """

    TWO_WAY_MATCH = 0
    """2-way match (p/c). Safest option, but may output combed frames in cases of bad edits or blended fields."""

    TWO_WAY_MATCH_THIRD_COMBED = 1
    """2-way match + 3rd match on combed (p/c + n). Default mode."""

    TWO_WAY_MATCH_THIRD_SAME_ORDER = 2
    """2-way match + 3rd match (same order) on combed (p/c + u)."""

    TWO_WAY_MATCH_THIRD_FOURTH_FIFTH = 3
    """2-way match + 3rd match on combed + 4th/5th matches if still combed (p/c + n + u/b)."""

    THREE_WAY_MATCH = 4
    """3-way match (p/c/n)."""

    THREE_WAY_MATCH_FOURTH_FIFTH = 5
    """
    3-way match + 4th/5th matches on combed (p/c/n + u/b).
    Highest risk of jerkiness but best at finding good matches.
    """


class IVTCycles(CustomEnum):
    """
    Enum representing different decimation patterns for IVTC (Inverse Telecine) processes.

    These patterns are used to remove duplicate frames after double weaving fields into frames.
    Each pattern defines a sequence of frame indices to keep during decimation.
    """

    cycle_10 = [[0, 3, 6, 8], [0, 2, 5, 8], [0, 2, 4, 7], [2, 4, 6, 9], [1, 4, 6, 8]]
    """Pattern for standard field-based 2:3 pulldown."""

    cycle_08 = [[0, 3, 4, 6], [0, 2, 5, 6], [0, 2, 4, 7], [0, 2, 4, 7], [1, 2, 4, 6]]
    """Pattern for 2:3:3:2 pulldown."""

    cycle_05 = [[0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3], [1, 2, 3, 4], [0, 2, 3, 4]]
    """Pattern for standard frame-based 2:3 pulldown."""

    @property
    def pattern_length(self) -> int:
        """Get the length of the pattern cycle in frames."""
        return int(self._name_[6:])

    @property
    def cycle (self) -> int:
        """Get the total number of available pattern variations for this cycle."""

        return len(self.value)

    def decimate(self, clip: vs.VideoNode, pattern: int = 0) -> vs.VideoNode:
        """Apply the decimation pattern to a video clip with the given pattern index."""

        assert 0 <= pattern < self.cycle
        return clip.std.SelectEvery(self.pattern_length, self.value[pattern])


class InputType(CustomIntEnum):
    """Processing routine to use for the input."""

    INTERLACE = 0
    """Deinterlace interlaced input."""

    PROGRESSIVE = 1
    """Deshimmer general progressive material that contains less severe problems."""

    REPAIR = 2
    """Repair badly deinterlaced material with considerable horizontal artefacts."""


class SearchPostProcess(CustomIntEnum):
    """Prefiltering to apply in order to assist with motion search."""

    NONE = 0
    """No post-processing."""

    GAUSSBLUR = 1
    """Gaussian blur."""

    GAUSSBLUR_EDGESOFTEN = 2
    """Gaussian blur & edge softening."""


class NoiseProcessMode(CustomIntEnum):
    """How to handle processing noise in the source."""

    NONE = 0
    """No noise processing."""

    DENOISE = 1
    """Denoise source & optionally restore some noise back at the end of basic or final stages."""

    IDENTIFY = 2
    """Identify noise only & optionally restore some noise back at the end of basic or final stages."""


class NoiseDeintMode(CustomIntEnum):
    """When noise is taken from interlaced source, how to 'deinterlace' it before restoring."""

    WEAVE = 0
    """Double weave source noise, lags behind by one frame."""

    BOB = 1
    """Bob source noise, results in coarse noise."""

    GENERATE = 2
    """Gnerates fresh noise lines."""


class SharpMode(CustomIntEnum):
    """How to re-sharpen the clip after temporally blurring."""

    NONE = 0
    """No re-sharpening."""

    UNSHARP = 1
    """Re-sharpening using unsharpening."""

    UNSHARP_MINMAX = 2
    """Re-sharpening using unsharpening clamped to the local 3x3 min/max average."""


class SharpLimitMode(CustomIntEnum):
    """How to limit and when to apply re-sharpening of the clip."""

    NONE = 0
    """No sharpness limiting."""

    SPATIAL_PRESMOOTH = 1
    """Spatial sharpness limiting prior to final stage."""

    TEMPORAL_PRESMOOTH = 2
    """Temporal sharpness limiting prior to final stage."""

    SPATIAL_POSTSMOOTH = 3
    """Spatial sharpness limiting after the final stage."""

    TEMPORAL_POSTSMOOTH = 4
    """Temporal sharpness limiting after the final stage."""


class BackBlendMode(CustomIntEnum):
    """When to back blend (blurred) difference between pre & post sharpened clip."""

    NONE = 0
    """No back-blending."""

    PRELIMIT = 1
    """Perform back-blending prior to sharpness limiting."""

    POSTLIMIT = 2
    """Perform back-blending after sharpness limiting."""

    BOTH = 3
    """Perform back-blending both before and after sharpness limiting."""


class SourceMatchMode(CustomIntEnum):
    """Creates higher fidelity output with extra processing. will capture more source detail and reduce oversharpening / haloing."""

    NONE = 0
    """No source match processing."""

    BASIC = 1
    """Conservative halfway stage that rarely introduces artefacts."""

    REFINED = 2
    """Restores almost exact source detail but is sensitive to noise & can introduce occasional aliasing."""

    TWICE_REFINED = 3
    """Restores almost exact source detail."""


class LosslessMode(CustomIntEnum):
    """When to put exact source fields into result & clean any artefacts."""

    NONE = 0
    """Do not restore source fields."""

    PRESHARPEN = 1
    """Restore source fields prior to re-sharpening. Not exactly lossless."""

    POSTSMOOTH = 2
    """Restore source fields after final temporal smooth. True lossless but less stable."""
