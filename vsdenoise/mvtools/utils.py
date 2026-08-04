from __future__ import annotations

from jetpytools import normalize_seq

__all__ = ["normalize_thscd", "refine_blksize"]


def refine_blksize(blksize: int | tuple[int, ...], divisor: int | tuple[int, ...] = (2, 2)) -> tuple[int, int]:
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


def normalize_thscd(thscd: int | tuple[int | None, float | None] | None) -> tuple[int | None, float | None]:
    return (None, None) if thscd is None else thscd if isinstance(thscd, tuple) else (thscd, None)
