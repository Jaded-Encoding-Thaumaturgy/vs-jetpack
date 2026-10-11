from __future__ import annotations

from collections.abc import Sequence

from jetpytools import SoftRange, SoftRangeN, SoftRangesN

__all__ = ["FrameRange", "FrameRangeN", "FrameRangesN", "Planes"]


type Planes = int | Sequence[int] | None
"""
Type alias for planes.

- `None`: All planes.
- `int`: Single plane.
- `Sequence[int]`: Multiple planes.

Values outside the range of the clip's format will throw an error.

Example of processing only the luma (Y) plane of a YUV clip:

```py
planes=[0]  # Y
```

Example of processing only the chroma (U and V) planes of a YUV clip:

```py
planes=[1, 2]  #  U, V
```

Example of processing all planes of a clip (regardless of the format):

```py
planes=None  # All planes
```
"""

type FrameRange = SoftRange
type FrameRangeN = SoftRangeN
type FrameRangesN = SoftRangesN
