from vstools import ConstantFormatVideoNode, CustomEnum, CustomIntEnum, check_variable, core, vs

__all__ = [
    "IVTCycles",
    "VFMMode",
]


class VFMMode(CustomIntEnum):
    """
    Enum representing different matching modes for VFM.

    The mode determines the strategy used for matching fields and frames.
    Higher modes generally offer better matching in complex scenarios but
    may introduce more risk of jerkiness or duplicate frames.
    """

    TWO_WAY_MATCH = 0
    """
    2-way match (p/c). Safest option, but may output combed frames in cases of bad edits or blended fields.
    """

    TWO_WAY_MATCH_THIRD_COMBED = 1
    """
    2-way match + 3rd match on combed (p/c + n). Default mode.
    """

    TWO_WAY_MATCH_THIRD_SAME_ORDER = 2
    """
    2-way match + 3rd match (same order) on combed (p/c + u).
    """

    TWO_WAY_MATCH_THIRD_FOURTH_FIFTH = 3
    """
    2-way match + 3rd match on combed + 4th/5th matches if still combed (p/c + n + u/b).
    """

    THREE_WAY_MATCH = 4
    """
    3-way match (p/c/n).
    """

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

    CYCLE_10 = ((0, 3, 6, 8), (0, 2, 5, 8), (0, 2, 4, 7), (2, 4, 6, 9), (1, 4, 6, 8))
    """
    Pattern for standard field-based 2:3 pulldown.
    """

    CYCLE_08 = ((0, 3, 4, 6), (0, 2, 5, 6), (0, 2, 4, 7), (0, 2, 4, 7), (1, 2, 4, 6))
    """
    Pattern for 2:3:3:2 pulldown.
    """

    CYCLE_05 = ((0, 1, 3, 4), (0, 1, 2, 4), (0, 1, 2, 3), (1, 2, 3, 4), (0, 2, 3, 4))
    """
    Pattern for standard frame-based 2:3 pulldown.
    """

    @property
    def pattern_length(self) -> int:
        """
        Get the length of the pattern cycle in frames.
        """

        return int(self._name_[6:])

    @property
    def cycle(self) -> int:
        """
        Get the total number of available pattern variations for this cycle.
        """

        return len(self.value)

    def decimate(self, clip: vs.VideoNode, pattern: int = 0) -> ConstantFormatVideoNode:
        """
        Apply the decimation pattern to a video clip with the given pattern index.
        """

        assert check_variable(clip, self.decimate)
        assert 0 <= pattern < self.cycle

        return core.std.SelectEvery(clip, self.pattern_length, self.value[pattern])
