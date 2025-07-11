from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vstools import ConstantFormatVideoNode, check_variable_format, vs, vs_object

from .enums import MVDirection

__all__ = [
    "MotionVectors",
]


class MotionVectors(vs_object):
    """
    Class for storing and managing motion vectors for a video clip.
    """

    motion_vectors: dict[MVDirection, dict[int, ConstantFormatVideoNode]]
    """Dictionary containing both backward and forward motion vectors."""

    mv_multi: ConstantFormatVideoNode
    """Multi-vector clip."""

    tr: int
    """Temporal radius of the motion vectors."""

    analysis_data: dict[str, Any]
    """Dictionary containing motion vector analysis data."""

    scaled: bool
    """Whether motion vectors have been scaled."""

    def __init__(self) -> None:
        self._init_vects()
        self.tr = 0
        self.analysis_data = dict[str, Any]()
        self.scaled = False

    def _init_vects(self) -> None:
        self.motion_vectors = {w: {} for w in MVDirection}

    def clear(self) -> None:
        """
        Clear all stored motion vectors and reset the instance.
        """

        for v in self.motion_vectors.values():
            v.clear()

        self.motion_vectors.clear()
        if hasattr(self, "mv_multi"):
            del self.mv_multi
        self.tr = 0
        self.analysis_data.clear()
        self.scaled = False
        self._init_vects()

    @property
    def has_vectors(self) -> bool:
        """
        Check if motion vectors are available.
        """

        return bool(
            (self.motion_vectors[MVDirection.BACKWARD] and self.motion_vectors[MVDirection.FORWARD])
            or hasattr(self, "mv_multi")
        )

    def set_vector(self, vector: vs.VideoNode, direction: MVDirection, delta: int) -> None:
        """
        Store a motion vector.

        Args:
            vector: Motion vector clip to store.
            direction: Direction of the motion vector (forward or backward).
            delta: Frame distance for the motion vector.
        """
        assert check_variable_format(vector, self.set_vector)

        self.motion_vectors[direction][delta] = vector

    def __vs_del__(self, core_id: int) -> None:
        for v in self.motion_vectors.values():
            for k in v:
                if not TYPE_CHECKING:
                    v[k] = None

        if not TYPE_CHECKING:
            self.mv_multi = None
