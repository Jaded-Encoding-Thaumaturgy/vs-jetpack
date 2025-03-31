from __future__ import annotations

from typing import Any

from vstools import vs, vs_object

from .enums import MVDirection

__all__ = [
    'MotionVectors',
]


class MotionVectors(vs_object):
    """Class for storing and managing motion vectors for a video clip."""

    motion_vectors: dict[MVDirection, dict[int, vs.VideoNode]]
    """Dictionary containing both backward and forward motion vectors."""

    mv_multi: vs.VideoNode | None
    """Multi-vector clip."""

    tr: int
    """Temporal radius of the motion vectors."""

    analysis_data: dict[str, Any]
    """Dictionary containing motion vector analysis data."""

    scaled: bool
    """Whether motion vectors have been scaled."""

    def __init__(self) -> None:
        self._init_vects()
        self.mv_multi = None
        self.tr = 0
        self.analysis_data = dict[str, Any]()
        self.scaled = False

    def _init_vects(self) -> None:
        self.motion_vectors = {w: {} for w in MVDirection}

    def clear(self) -> None:
        """Clear all stored motion vectors and reset the instance."""

        for v in self.motion_vectors.values():
            v.clear()

        self.motion_vectors.clear()
        self.mv_multi = None
        self.tr = 0
        self.analysis_data.clear()
        self.scaled = False
        self._init_vects()

    @property
    def has_vectors(self) -> bool:
        """Check if motion vectors are available."""

        return bool(
            (self.motion_vectors[MVDirection.BACKWARD] and self.motion_vectors[MVDirection.FORWARD]) or self.mv_multi
        )

    def set_vector(self, vector: vs.VideoNode, direction: MVDirection, delta: int) -> None:
        """
        Store a motion vector.

        :param vector:       Motion vector clip to store.
        :param direction:    Direction of the motion vector (forward or backward).
        :param delta:        Frame distance for the motion vector.
        """

        self.motion_vectors[direction][delta] = vector

    def __vs_del__(self, core_id: int) -> None:
        for v in self.motion_vectors.values():
            v.clear()

        self.mv_multi = None
