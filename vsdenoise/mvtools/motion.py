from __future__ import annotations

from collections import defaultdict
from contextlib import suppress
from typing import Any

from vstools import check_variable_format, vs, vs_object

from .enums import MVDirection

__all__ = [
    "MotionVectors",
]


class MotionVectors(defaultdict[MVDirection, dict[int, vs.VideoNode]], vs_object):
    """
    Class for storing and managing motion vectors for a video clip.

    Contains both backward and forward motion vectors.
    """

    mv_multi: vs.VideoNode
    """Multi-vector clip."""

    tr: int
    """Temporal radius of the motion vectors."""

    analysis_data: dict[str, Any]
    """Dictionary containing motion vector analysis data."""

    scaled: bool
    """Whether motion vectors have been scaled."""

    def __init__(self) -> None:
        super().__init__(None, {w: {} for w in MVDirection})
        self.tr = 0
        self.analysis_data = {}
        self.scaled = False

    def clear(self) -> None:
        """
        Clear all stored motion vectors.
        """

        for v in self.values():
            v.clear()

        with suppress(AttributeError):
            del self.mv_multi

        self.tr = 0
        self.analysis_data.clear()
        self.scaled = False

    @property
    def has_vectors(self) -> bool:
        """
        Check if motion vectors are available.
        """

        return bool((self[MVDirection.BACKWARD] and self[MVDirection.FORWARD]) or hasattr(self, "mv_multi"))

    def set_vector(self, vector: vs.VideoNode, direction: MVDirection, delta: int) -> None:
        """
        Store a motion vector.

        Args:
            vector: Motion vector clip to store.
            direction: Direction of the motion vector (forward or backward).
            delta: Frame distance for the motion vector.
        """
        assert check_variable_format(vector, self.set_vector)

        self[direction][delta] = vector

    def __vs_del__(self, core_id: int) -> None:
        self.clear()
