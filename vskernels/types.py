from __future__ import annotations

from functools import cache
from typing import Any, TypeAlias, Union

from jetpytools import CustomNotImplementedError

from vstools import CustomIntEnum, KwargsT, padder, vs

__all__ = [
    "BorderHandling",
    "BotFieldLeftShift",
    "BotFieldTopShift",
    "Center",
    "LeftShift",
    "SampleGridModel",
    "ShiftT",
    "Slope",
    "TopFieldLeftShift",
    "TopFieldTopShift",
    "TopShift",
]


class BorderHandling(CustomIntEnum):
    MIRROR = 0
    ZERO = 1
    REPEAT = 2

    def prepare_clip(
        self, clip: vs.VideoNode, min_pad: int = 2, shift: tuple[TopShift, LeftShift] = (0, 0)
    ) -> tuple[vs.VideoNode, tuple[TopShift, LeftShift]]:
        pad_w, pad_h = (self.pad_amount(size, min_pad) for size in (clip.width, clip.height))

        if pad_w == pad_h == 0:
            return clip, shift

        match self:
            case BorderHandling.ZERO:
                padded = padder.COLOR(clip, pad_w, pad_w, pad_h, pad_h)
            case BorderHandling.REPEAT:
                padded = padder.REPEAT(clip, pad_w, pad_w, pad_h, pad_h)
            case _:
                raise CustomNotImplementedError

        shift = tuple(s + ((p - c) // 2) for s, c, p in zip(shift, *((x.height, x.width) for x in (clip, padded))))

        return padded, shift

    @cache
    def pad_amount(self, size: int, min_amount: int = 2) -> int:
        if self is BorderHandling.MIRROR:
            return 0

        return (((size + min_amount) + 7) & -8) - size


class SampleGridModel(CustomIntEnum):
    MATCH_EDGES = 0
    MATCH_CENTERS = 1

    def __call__(
        self, width: int, height: int, src_width: float, src_height: float, shift: tuple[float, float], kwargs: KwargsT
    ) -> tuple[KwargsT, tuple[float, float]]:
        if self is SampleGridModel.MATCH_CENTERS:
            src_width = src_width * (width - 1) / (src_width - 1)
            src_height = src_height * (height - 1) / (src_height - 1)

            kwargs |= {"src_width": src_width, "src_height": src_height}
            shift_x, shift_y, *_ = tuple(
                (x / 2 + y for x, y in zip(((height - src_height), (width - src_width)), shift))
            )
            shift = shift_x, shift_y

        return kwargs, shift

    def for_dst(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float], **kwargs: Any
    ) -> tuple[KwargsT, tuple[float, float]]:
        src_width = kwargs.get("src_width", width)
        src_height = kwargs.get("src_height", height)

        return self(src_width, src_height, width, height, shift, kwargs)

    def for_src(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float], **kwargs: Any
    ) -> tuple[KwargsT, tuple[float, float]]:
        src_width = kwargs.get("src_width", clip.width)
        src_height = kwargs.get("src_height", clip.height)

        return self(width, height, src_width, src_height, shift, kwargs)


TopShift: TypeAlias = float
"""
Type alias for vertical shift in pixels (top).

Represents the amount of vertical offset when scaling a video.
"""

LeftShift: TypeAlias = float
"""
Type alias for horizontal shift in pixels (left).

Represents the amount of horizontal offset when scaling a video.
"""

TopFieldTopShift: TypeAlias = float
"""
Type alias for the top field's vertical shift in pixels.

Used when processing interlaced video to describe the vertical shift of the top field.
"""

TopFieldLeftShift: TypeAlias = float
"""
Type alias for the top field's horizontal shift in pixels.

Used when processing interlaced video to describe the horizontal shift of the top field.
"""

BotFieldTopShift: TypeAlias = float
"""
Type alias for the bottom field's vertical shift in pixels.

Used when processing interlaced video to describe the vertical shift of the bottom field.
"""

BotFieldLeftShift: TypeAlias = float
"""
Type alias for the bottom field's horizontal shift in pixels.

Used when processing interlaced video to describe the horizontal shift of the bottom field.
"""

ShiftT = Union[
    tuple[TopShift, LeftShift],
    tuple[
        TopShift | tuple[TopFieldTopShift, BotFieldTopShift], LeftShift | tuple[TopFieldLeftShift, BotFieldLeftShift]
    ],
]
"""
Type alias for shift in both horizontal and vertical directions.

Can either represent a single shift (for progressive video)
or separate shifts for top and bottom fields (for interlaced video).

The first value in the tuple represents vertical shift, and the second represents horizontal shift.
"""

Slope: TypeAlias = float
"""
Type alias for the slope of the sigmoid curve, controlling the steepness of the transition.
"""

Center: TypeAlias = float
"""
Type alias for the center point of the sigmoid curve, determining the midpoint of the transition.
"""
