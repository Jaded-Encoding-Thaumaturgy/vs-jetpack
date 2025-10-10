from __future__ import annotations

from jetpytools import CustomValueError, normalize_seq

from vstools import Planes, normalize_planes, vs

__all__ = ["normalize_thscd", "planes_to_mvtools", "refine_blksize"]


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


def normalize_thscd(
    thscd: int | tuple[int | None, int | float | None] | None, scale: bool = True
) -> tuple[int | None, int | None]:
    """
    Normalize and scale the thscd parameter.

    Args:
        thscd: thscd value to scale and/or normalize.
        scale: Whether to scale thscd2 from 0-100 percentage threshold to 0-255.

    Returns:
        Scaled and/or normalized thscd tuple.
    """

    thscd1, thscd2 = thscd if isinstance(thscd, tuple) else (thscd, None)

    if scale and thscd2 is not None:
        thscd2 = round(thscd2 / 100 * 255)

    if isinstance(thscd2, float):
        thscd2 = int(thscd2)

    return (thscd1, thscd2)


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
