from __future__ import annotations

from collections.abc import Sequence

from jetpytools import SoftRange, SoftRangeN, SoftRangesN

__all__ = ["FrameRange", "FrameRangeN", "FrameRangesN", "Planes", "PlanesT"]


type Planes = int | Sequence[int] | None
PlanesT = Planes

type FrameRange = SoftRange
type FrameRangeN = SoftRangeN
type FrameRangesN = SoftRangesN
