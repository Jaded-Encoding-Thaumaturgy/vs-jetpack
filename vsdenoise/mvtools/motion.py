from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from jetpytools import to_arr

from vstools import VSObject, vs

from .enums import MVDirection

__all__ = ["MotionVectors"]


class MotionVectors(VSObject, dict[int, vs.VideoNode]):
    """
    Class for storing and managing motion vectors for a video clip.

    Contains both backward and forward motion vectors.
    """

    def __init__(
        self,
        *args: Mapping[int, vs.VideoNode] | Iterable[tuple[int, vs.VideoNode]],
        blksize: tuple[int, int] | None = None,
        overlap: tuple[int, int] | None = None,
        pad: tuple[int, int] | None = None,
        pel: int | None = None,
    ) -> None:
        super().__init__(*args)

        self.blksize = blksize
        self.overlap = overlap
        self.pad = pad
        self.pel = pel

    def clear(self) -> None:
        """
        Clear all stored motion vectors.
        """

        super().clear()

        self.blksize = None
        self.overlap = None
        self.pad = None
        self.pel = None

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
            delta: Specific delta(s) of motion vectors to use.

        Returns:
            A tuple containing two lists of motion vectors.
            The first list contains backward vectors and the second contains forward vectors.
        """

        if delta:
            delta = to_arr(delta)
        elif tr:
            range_start = -tr if direction & MVDirection.FORWARD else 1
            range_end = tr + 1 if direction & MVDirection.BACKWARD else 0
            delta = range(range_start, range_end)
        else:
            delta = self.keys()

        vectors_backward = list[vs.VideoNode]()
        vectors_forward = list[vs.VideoNode]()

        for d in sorted(delta, key=abs):
            if d > 0:
                vectors_backward.append(self[d])
            elif d < 0:
                vectors_forward.append(self[d])

        return (vectors_backward, vectors_forward)
