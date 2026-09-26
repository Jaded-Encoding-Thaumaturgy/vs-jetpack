from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from jetpytools import normalize_seq, to_arr

from vstools import VSObject, core, vs

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
        tr: int | None = None,
        delta: int | Sequence[int] | None = None,
    ) -> tuple[list[vs.VideoNode], list[vs.VideoNode]]:
        """
        Get the backward and forward vectors.

        Args:
            tr: The number of frames to get the vectors for.
            delta: Specific delta(s) of motion vectors to use.

        Returns:
            A tuple containing two lists of motion vectors.
            The first list contains backward vectors and the second contains forward vectors.
        """

        if delta:
            delta = to_arr(delta)
        elif tr:
            delta = range(-tr, tr + 1)
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

    def scale_vectors(self, scale: int | tuple[int, int]) -> None:
        """
        Scales image_size, block_size, overlap, padding, and the individual motion_vectors contained in Analyse output
        by arbitrary and independent x and y factors.

        Args:
            scale: Factor to scale motion vectors by.
        """

        scalex, scaley = normalize_seq(scale, 2)

        if scalex > 1 or scaley > 1:
            for attr in ("blksize", "overlap", "pad"):
                x, y = getattr(self, attr)
                setattr(self, attr, (x * scalex, y * scaley))

            for delta, vect in self.items():
                self[delta] = core.manipmv.ScaleVect(vect, scalex, scaley)

    def show_vector(
        self,
        clip: vs.VideoNode,
        delta: int,
        scenechange: bool = True,
    ) -> vs.VideoNode:
        """
        Draws generated vectors onto a clip.

        Args:
            clip: The clip to overlay the motion vectors on.
            delta: Motion vector delta to use.
            scenechange: Skips drawing vectors if frame props indicate they are from a different scene than the current
                frame of the clip.

        Returns:
            Clip with motion vectors overlaid.
        """

        return core.manipmv.ShowVect(clip, self[delta], scenechange)
