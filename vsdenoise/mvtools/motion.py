from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType
from typing import Any, Literal

from jetpytools import CustomRuntimeError, cachedproperty, fallback, normalize_seq

from vstools import VSObject, get_props, vs

from .enums import MVDirection

__all__ = [
    "MotionVectors",
]


class MotionVectors(defaultdict[MVDirection, dict[int, vs.VideoNode]], VSObject):
    """
    Class for storing and managing motion vectors for a video clip.

    Contains both backward and forward motion vectors.
    """

    def __init__(self) -> None:
        super().__init__(None, {w: {} for w in MVDirection})

    def clear(self) -> None:
        """
        Clear all stored motion vectors.
        """

        for v in self.values():
            v.clear()

        cachedproperty.clear_cache(self)

    def set_vector(self, vector: vs.VideoNode, direction: MVDirection, delta: int) -> None:
        """
        Store a motion vector.

        Args:
            vector: Motion vector clip to store.
            direction: Direction of the motion vector (forward or backward).
            delta: Frame distance for the motion vector.
        """

        self[direction][delta] = vector

    def get_vector(self, direction: MVDirection, delta: int) -> vs.VideoNode:
        """
        Get a single motion vector.

        Args:
            direction: Motion vector direction to get.
            delta: Motion vector delta to get.

        Returns:
            A single motion vector VideoNode
        """

        if delta > self.tr:
            raise CustomRuntimeError(
                "Tried to get a motion vector delta larger than what exists!",
                self.get_vector,
                f"{delta} > {self.tr}",
            )

        return self[direction][delta]

    def get_vectors(
        self,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None,
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Get the backward and forward vectors.

        Args:
            direction: Motion vector direction to get.
            tr: The number of frames to get the vectors for.

        Returns:
            A tuple containing two lists of motion vectors.
            The first list contains backward vectors and the second contains forward vectors.
        """

        tr = fallback(tr, self.tr)

        if tr > self.tr:
            raise CustomRuntimeError(
                "Tried to obtain more motion vectors than what exist!", self.get_vectors, f"{tr} > {self.tr}"
            )

        vectors_backward = list[vs.VideoNode]()
        vectors_forward = list[vs.VideoNode]()

        for delta in range(1, tr + 1):
            if direction in [MVDirection.BACKWARD, MVDirection.BOTH]:
                vectors_backward.append(self.get_vector(MVDirection.BACKWARD, delta))
            if direction in [MVDirection.FORWARD, MVDirection.BOTH]:
                vectors_forward.append(self.get_vector(MVDirection.FORWARD, delta))

        return (vectors_backward, vectors_forward)

    @cachedproperty
    def analysis_data(self) -> MappingProxyType[str, Any]:
        """Mapping containing motion vector analysis data."""

        vect = self.get_vector(MVDirection.BACKWARD, 1).manipmv.ExpandAnalysisData()

        props_list = (
            "Analysis_BlockSize",
            "Analysis_Pel",
            "Analysis_LevelCount",
            "Analysis_CpuFlags",
            "Analysis_MotionFlags",
            "Analysis_FrameSize",
            "Analysis_Overlap",
            "Analysis_BlockCount",
            "Analysis_BitsPerSample",
            "Analysis_ChromaRatio",
            "Analysis_Padding",
        )

        return MappingProxyType(
            get_props(vect, props_list, (int, list), func=self.__class__.__name__ + ".analysis_data")
        )

    @analysis_data.deleter  # type: ignore[no-redef]
    def analysis_data(self) -> None:
        cachedproperty.clear_cache(self, "analysis_data")

    def scale_vectors(self, scale: int | tuple[int, int], strict: bool = True) -> None:
        """
        Scales image_size, block_size, overlap, padding, and the individual motion_vectors contained in Analyse output
        by arbitrary and independent x and y factors.

        Args:
            scale: Factor to scale motion vectors by.
        """

        supported_blksize = (
            (4, 4),
            (8, 4),
            (8, 8),
            (16, 2),
            (16, 8),
            (16, 16),
            (32, 16),
            (32, 32),
            (64, 32),
            (64, 64),
            (128, 64),
            (128, 128),
        )

        scalex, scaley = normalize_seq(scale, 2)

        if scalex > 1 or scaley > 1:
            blksizex, blksizev = self.analysis_data["Analysis_BlockSize"]

            scaled_blksize = (blksizex * scalex, blksizev * scaley)

            if strict and scaled_blksize not in supported_blksize:
                raise CustomRuntimeError("Unsupported block size!", self.scale_vectors, scaled_blksize)

            del self.analysis_data
            cachedproperty.update_cache(self, "scaled", True)

            for delta in range(1, self.tr + 1):
                for direction in MVDirection:
                    self[direction][delta] = self[direction][delta].manipmv.ScaleVect(scalex, scaley)

    def show_vector(
        self,
        clip: vs.VideoNode,
        direction: Literal[MVDirection.FORWARD, MVDirection.BACKWARD] = MVDirection.FORWARD,
        delta: int = 1,
        scenechange: bool | None = None,
    ) -> vs.VideoNode:
        """
        Draws generated vectors onto a clip.

        Args:
            clip: The clip to overlay the motion vectors on.
            direction: Motion vector direction to use.
            delta: Motion vector delta to use.
            scenechange: Skips drawing vectors if frame props indicate they are from a different scene than the current
                frame of the clip.

        Returns:
            Clip with motion vectors overlaid.
        """

        vect = self.get_vector(direction, delta)

        return clip.manipmv.ShowVect(vect, scenechange)

    @cachedproperty
    def scaled(self) -> bool:
        """Whether motion vectors have been scaled."""

        return False

    @property
    def tr(self) -> int:
        """
        Temporal radius of the motion vectors.
        """

        return max(len(self[MVDirection.BACKWARD]), len(self[MVDirection.FORWARD]))
