from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from contextlib import suppress
from types import MappingProxyType
from typing import Any, Literal

from jetpytools import CustomRuntimeError, cachedproperty, fallback, normalize_seq, to_arr

from vstools import VSObject, vs

from .enums import MVDirection

__all__ = ["MotionVectors"]


class MotionVectors(VSObject, defaultdict[MVDirection, dict[int, vs.VideoNode]]):
    """
    Class for storing and managing motion vectors for a video clip.

    Contains both backward and forward motion vectors.
    """

    def __init__(
        self,
        blksize: int | tuple[int, int] | None = None,
        overlap_div: int | tuple[int, int] | None = None,
    ) -> None:
        super().__init__(None, {w: {} for w in MVDirection})
        self.scaled = False
        self._blksize: tuple[int, int] | None = None
        self._overlap_div: tuple[int, int] | None = None
        self.blksize = blksize
        self.overlap_div = overlap_div

    @property
    def blksize(self) -> tuple[int, int] | None:
        if self._blksize is not None:
            return self._blksize

        with suppress(KeyError, CustomRuntimeError):
            return (
                self.analysis_data["MVUtensilsAnalysisBlkSizeX"],
                self.analysis_data["MVUtensilsAnalysisBlkSizeY"],
            )

        return None

    @blksize.setter
    def blksize(self, value: int | tuple[int, int] | None) -> None:
        self._blksize = tuple(normalize_seq(value, 2)) if value is not None else None  # type: ignore[assignment]

    @property
    def overlap_div(self) -> tuple[int, int] | None:
        if self._overlap_div is not None:
            return self._overlap_div

        if self.blksize is None:
            return None

        try:
            overlap = (
                self.analysis_data["MVUtensilsAnalysisOverlapX"],
                self.analysis_data["MVUtensilsAnalysisOverlapY"],
            )
        except (KeyError, CustomRuntimeError):
            return None

        return self.blksize[0] // overlap[0], self.blksize[1] // overlap[1]

    @overlap_div.setter
    def overlap_div(self, value: int | tuple[int, int] | None) -> None:
        self._overlap_div = tuple(normalize_seq(value, 2)) if value is not None else None  # type: ignore[assignment]

    def clear(self) -> None:
        """
        Clear all stored motion vectors.
        """

        for v in self.values():
            v.clear()

        self._blksize = None
        self._overlap_div = None
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

        if delta not in self[direction]:
            raise CustomRuntimeError(
                "Tried to get a motion vector delta that does not exist!", self.get_vector, f"{delta}"
            )

        return self[direction][delta]

    def get_vectors(
        self,
        direction: MVDirection = MVDirection.BOTH,
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Get the backward and forward vectors.

        Args:
            direction: Motion vector direction to get.
            tr: The number of frames to get the vectors for.
            delta: Specific delta(s) of motion vectors to retrieve.

        Returns:
            A tuple containing two lists of motion vectors.
            The first list contains backward vectors and the second contains forward vectors.
        """
        if delta is not None:
            deltas = [delta] if isinstance(delta, int) else list(delta)
        else:
            tr = fallback(tr, self.tr)
            deltas = range(1, tr + 1)

        vectors_backward = list[vs.VideoNode]()
        vectors_forward = list[vs.VideoNode]()

        for d in deltas:
            if direction in [MVDirection.BACKWARD, MVDirection.BOTH] and d in self[MVDirection.BACKWARD]:
                vectors_backward.append(self[MVDirection.BACKWARD][d])
            if direction in [MVDirection.FORWARD, MVDirection.BOTH] and d in self[MVDirection.FORWARD]:
                vectors_forward.append(self[MVDirection.FORWARD][d])

        return (vectors_backward, vectors_forward)

    @cachedproperty
    def analysis_data(self) -> MappingProxyType[str, Any]:
        """Mapping containing motion vector analysis data."""

        # vect = self.get_vector(MVDirection.BACKWARD, 1).manipmv.ExpandAnalysisData()

        # props_list = (
        #     "Analysis_BlockSize",
        #     "Analysis_Pel",
        #     "Analysis_LevelCount",
        #     "Analysis_CpuFlags",
        #     "Analysis_MotionFlags",
        #     "Analysis_FrameSize",
        #     "Analysis_Overlap",
        #     "Analysis_BlockCount",
        #     "Analysis_BitsPerSample",
        #     "Analysis_ChromaRatio",
        #     "Analysis_Padding",
        # )

        # return MappingProxyType(
        #     get_props(vect, props_list, (int, list), func=self.__class__.__name__ + ".analysis_data")
        # )

        with self.get_vector(MVDirection.BACKWARD, 1).get_frame(0) as fr:
            return MappingProxyType({key: fr.props[key] for key in fr.props if key.startswith("MVUtensils")})

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
        # supported_blksize = (
        #     (4, 4),
        #     (8, 4),
        #     (8, 8),
        #     (16, 2),
        #     (16, 8),
        #     (16, 16),
        #     (32, 16),
        #     (32, 32),
        #     (64, 32),
        #     (64, 64),
        #     (128, 64),
        #     (128, 128),
        # )

        # scalex, scaley = normalize_seq(scale, 2)

        # if scalex > 1 or scaley > 1:
        #     blksizex, blksizev = self.analysis_data["Analysis_BlockSize"]

        #     scaled_blksize = (blksizex * scalex, blksizev * scaley)

        #     if strict and scaled_blksize not in supported_blksize:
        #         raise CustomRuntimeError("Unsupported block size!", self.scale_vectors, scaled_blksize)

        #     del self.analysis_data
        #     self.scaled = True

        #     for delta in range(1, self.tr + 1):
        #         for direction in MVDirection:
        #             self[direction][delta] = self[direction][delta].manipmv.ScaleVect(scalex, scaley)
        raise NotImplementedError("scale_vectors is not supported with MVUtensils.")

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
        # vect = self.get_vector(direction, delta)

        # return clip.manipmv.ShowVect(vect, scenechange)

        raise NotImplementedError("show_vector is not supported with MVUtensils.")

    @property
    def deltas(self) -> list[int]:
        """List of active deltas."""
        return sorted(set(self[MVDirection.BACKWARD].keys()) | set(self[MVDirection.FORWARD].keys()))

    @property
    def tr(self) -> int:
        """
        Temporal radius of the motion vectors.
        """
        return max(self.deltas) if self.deltas else 0
