from __future__ import annotations

from jetpytools import CustomValueError, normalize_seq

from vstools import Planes, normalize_planes, vs

__all__ = ["planes_to_mvtools", "refine_blksize"]


def planes_to_mvtools(clip: vs.VideoNode, planes: Planes) -> int:
    """
    Convert a regular Planes parameter to MVTools' plane parameter value.

    MVTools uses a single integer to represent which planes to process:

    - 0: Process Y plane only
    - 1: Process U plane only
    - 2: Process V plane only
    - 3: Process UV planes only
    - 4: Process all planes

    Args:
        clip: Input clip.
        planes: Which planes to process.

    Returns:
        Integer value used by MVTools to specify which planes to process.
    """
    norm_planes = set(normalize_planes(clip, planes))

    if norm_planes in [{0}, {1}, {2}]:
        return norm_planes.pop()

    if norm_planes == {1, 2}:
        return 3

    if norm_planes == {0, 1, 2}:
        return 4

    raise CustomValueError("Invalid planes specified!", planes_to_mvtools)


def refine_blksize(blksize: int | tuple[int, int], divisor: int | tuple[int, int] = (2, 2)) -> tuple[int, int]:
    """
    Normalize and refine blksize.

    Args:
        blksize: Block size to refine.
        divisor: Block size divisor.

    Returns:
        Normalized and refined blksize tuple.
    """

    nblksize = normalize_seq(blksize, 2)
    ndivisor = normalize_seq(divisor, 2)

    return (
        nblksize[0] // ndivisor[0] if ndivisor[0] else 0,
        nblksize[1] // ndivisor[1] if ndivisor[1] else 0,
    )
