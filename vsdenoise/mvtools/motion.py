from __future__ import annotations

from typing import Any

from vstools import vs

from .enums import MVDirection

__all__ = [
    'MotionVectors',
]


class MotionVectors:
    """Class for storing and managing motion vectors for a video clip."""

    vmulti: vs.VideoNode
    """Super-sampled clip used for motion vector analysis."""

    analysis_data: dict[str, Any]
    """Dictionary containing motion vector analysis data."""

    scaled: bool
    """Whether motion vectors have been scaled."""

    temporal_vectors: dict[MVDirection, dict[int, vs.VideoNode]]
    """Dictionary containing both backward and forward motion vectors."""

    def __init__(self) -> None:
        self._init_vects()
        self.analysis_data = dict()
        self.scaled = False
        self.kwargs = dict[str, Any]()

    def _init_vects(self) -> None:
        self.temporal_vectors = {w: {} for w in MVDirection}

    @property
    def has_vectors(self) -> bool:
        """Check if motion vectors are available."""

        return bool(
            (self.temporal_vectors[MVDirection.BACK] and self.temporal_vectors[MVDirection.FWRD]) or self.vmulti
        )

    def get_mv(self, direction: MVDirection, delta: int) -> vs.VideoNode:
        """
        Retrieve a specific motion vector.

        :param direction:    Direction of the motion vector (forward or backward).
        :param delta:        Frame distance for the motion vector.

        :return:             The requested motion vector clip.
        """

        return self.temporal_vectors[direction][delta]

    def set_mv(self, direction: MVDirection, delta: int, vector: vs.VideoNode) -> None:
        """
        Store a motion vector.

        :param direction:    Direction of the motion vector (forward or backward).
        :param delta:        Frame distance for the motion vector.
        :param vect:         Motion vector clip to store.
        """

        self.temporal_vectors[direction][delta] = vector

    def clear(self) -> None:
        """Clear all stored motion vectors and reset the instance."""

        del self.vmulti
        self.analysis_data.clear()
        self.scaled = False
        self.kwargs.clear()
        self.temporal_vectors.clear()
        self._init_vects()
