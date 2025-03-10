from __future__ import annotations

from typing import Callable, Iterable, Sequence, TypeAlias, TypeVar

import vapoursynth as vs
from jetpytools import (
    F0, F1, F2, P0, P1, P2, R0, R1, R2, T0, T1, T2, ByteData, ComparatorFunc, F, KwargsT, Nb, P, R, R_contra,
    SimpleByteData, SimpleByteDataArray, SingleOrArr, SingleOrArrOpt, SingleOrSeq, SingleOrSeqOpt, SoftRange,
    SoftRangeN, SoftRangesN, SupportsAllComparisons, SupportsDunderGE, SupportsDunderGT, SupportsDunderLE,
    SupportsDunderLT, SupportsFloatOrIndex, SupportsIndexing, SupportsKeysAndGetItem, SupportsRichComparison,
    SupportsRichComparisonT, SupportsString, SupportsTrunc, T, T_contra
)

__all__ = [
    'T', 'T0', 'T1', 'T2', 'T_contra',

    'F', 'F0', 'F1', 'F2', 'F_VD',

    'P', 'P0', 'P1', 'P2',
    'R', 'R0', 'R1', 'R2', 'R_contra',

    'Nb',

    'PlanesT', 'VideoNodeIterable',

    'FrameRange', 'FrameRangeN', 'FrameRangesN',

    'SingleOrArr', 'SingleOrArrOpt',
    'SingleOrSeq', 'SingleOrSeqOpt',

    'SimpleByteData', 'SimpleByteDataArray',
    'ByteData',

    'KwargsT',

    'SupportsTrunc',

    'SupportsString',

    'SupportsDunderLT', 'SupportsDunderGT',
    'SupportsDunderLE', 'SupportsDunderGE',

    'SupportsFloatOrIndex',

    'SupportsIndexing',
    'SupportsKeysAndGetItem',

    'SupportsAllComparisons',
    'SupportsRichComparison', 'SupportsRichComparisonT',
    'ComparatorFunc'
]

F_VD = TypeVar('F_VD', bound=Callable[..., vs.VideoNode])

PlanesT: TypeAlias = int | Sequence[int] | None

FrameRange: TypeAlias = SoftRange
FrameRangeN: TypeAlias = SoftRangeN
FrameRangesN: TypeAlias = SoftRangesN


VideoNodeIterable: TypeAlias = vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]] | Iterable[
    vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]
]
