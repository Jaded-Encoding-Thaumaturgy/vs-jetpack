from __future__ import annotations

from math import ceil
from typing import Any, assert_never

from jetpytools import CustomIntEnum, fallback

from vstools import padder, vs

__all__ = [
    "BorderHandling",
    "BotFieldLeftShift",
    "BotFieldTopShift",
    "Center",
    "FieldShift",
    "LeftShift",
    "SampleGridModel",
    "Slope",
    "TopFieldLeftShift",
    "TopFieldTopShift",
    "TopShift",
]


class BorderHandling(CustomIntEnum):
    """
    Border padding strategy used when a clip requires alignment padding.
    """

    MIRROR = 0
    """Assume the image was resized with mirror padding."""

    ZERO = 1
    """Assume the image was resized with zero padding."""

    REPEAT = 2
    """Assume the image was resized with extend padding, where the outermost row was extended infinitely far."""

    def prepare_clip(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[TopShift, LeftShift],
        kernel_radius: int,
        **kwargs: Any,
    ) -> tuple[vs.VideoNode, tuple[TopShift, LeftShift]]:
        """
        Apply required padding and adjust shift.

        Args:
            clip: Input clip.
            width: Output width.
            height: Output height.
            shift: Current (top, left) shift.
            kernel_radius: Kernel radius.
            **kwargs: Optional src_width/src_height.

        Returns:
            (padded clip, updated shift).
        """

        if self is BorderHandling.MIRROR:
            return (clip, shift)

        src_width = fallback(kwargs.get("src_width"), clip.width)
        src_height = fallback(kwargs.get("src_height"), clip.height)

        shift = kwargs.pop("src_stop", shift[0]), kwargs.pop("src_left", shift[1])

        left, right, top, bottom = self.pad_amount(
            clip,
            width,
            height,
            shift,
            kernel_radius,
            src_width,
            src_height,
        )

        match self:
            case BorderHandling.ZERO:
                padded = padder.COLOR(clip, left, right, top, bottom)
            case BorderHandling.REPEAT:
                padded = padder.REPEAT(clip, left, right, top, bottom)
            case _:
                assert_never(self)

        shift = tuple(s + c for s, c in zip(shift, (top, left)))  # type: ignore

        return padded, shift

    def pad_amount(
        self,
        clip: vs.VideoNode,
        width: int,
        height: int,
        shift: tuple[TopShift, LeftShift],
        kernel_radius: int,
        src_width: float,
        src_height: float,
    ) -> tuple[int, int, int, int]:
        """
        Return required padding.

        Args:
            clip: Input clip.
            width: Output width.
            height: Output height.
            shift: Current (top, left) shift.
            kernel_radius: Kernel radius.
            src_width: Width source region.
            src_height: Height source region.

        Returns:
            Padding amount.
        """
        top_shift, left_shift = shift

        w_factor = kernel_radius * max(src_width / width, 1)
        left, right = (
            ceil((w_factor - left_shift) / 2) * 2**clip.format.subsampling_w,
            ceil((w_factor + left_shift) / 2) * 2**clip.format.subsampling_w,
        )

        h_factor = kernel_radius * max(src_height / height, 1)
        top, bottom = (
            ceil((h_factor - top_shift) / 2) * 2**clip.format.subsampling_h,
            ceil((h_factor + top_shift) / 2) * 2**clip.format.subsampling_h,
        )

        return (left, right, top, bottom)


class SampleGridModel(CustomIntEnum):
    """
    Sampling grid alignment model.

    While match edges will align the edges of the outermost pixels in the target image,
    match centers will instead align the *centers* of the outermost pixels.

    Here's a visual example for a 3x1 image upsampled to 7x1:

    - Match edges:
        ```
        +-------------+-------------+-------------+
        |      ·      |      ·      |      ·      |
        +-------------+-------------+-------------+
        ↓                                         ↓
        +-----+-----+-----+-----+-----+-----+-----+
        |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |
        +-----+-----+-----+-----+-----+-----+-----+
        ```

    - Match centers:
        ```
        +-----------------+-----------------+-----------------+
        |        ·        |        ·        |        ·        |
        +-----------------+-----------------+-----------------+
                 ↓                                   ↓
              +-----+-----+-----+-----+-----+-----+-----+
              |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |  ·  |
              +-----+-----+-----+-----+-----+-----+-----+
        ```

    For a more detailed explanation, refer to this page: <https://entropymine.com/imageworsener/matching/>.

    The formula for calculating values we can use during desampling is simple:

    - width: `base_width * (target_width - 1) / (base_width - 1)`
    - height: `base_height * (target_height - 1) / (base_height - 1)`
    """

    MATCH_EDGES = 0
    """Align edges."""

    MATCH_CENTERS = 1
    """Align pixel centers."""

    def __call__(
        self,
        width: float,
        height: float,
        src_width: float,
        src_height: float,
        shift: tuple[float, float],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], tuple[float, float]]:
        """
        Apply sampling model to sizes and shift.

        Args:
            width: Destination width.
            height: Destination height.
            src_width: Current source width.
            src_height: Current source height.
            shift: Top, left sampling shift.
            kwargs: Parameter dict to update.

        Returns:
            (updated kwargs, updated shift).
        """

        if self is SampleGridModel.MATCH_CENTERS:
            src_width = src_width * (width - 1) / (src_width - 1)
            src_height = src_height * (height - 1) / (src_height - 1)

            shift = kwargs.pop("src_stop", shift[0]), kwargs.pop("src_left", shift[1])

            kwargs.update(src_width=src_width, src_height=src_height)
            shift_x, shift_y, *_ = tuple(
                (x / 2 + sh for x, sh in zip(((height - src_height), (width - src_width)), shift))
            )
            shift = shift_x, shift_y

        return kwargs, shift

    def for_dst(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float], **kwargs: Any
    ) -> tuple[dict[str, Any], tuple[float, float]]:
        """
        Apply grid model using destination sizes.

        Args:
            clip: Source clip.
            width: Destination width.
            height: Destination height.
            shift: Current shift.
            **kwargs: Optional src_width/src_height.

        Returns:
            (updated kwargs, updated shift).
        """

        src_width = fallback(kwargs.get("src_width"), width)
        src_height = fallback(kwargs.get("src_height"), height)

        return self(src_width, src_height, width, height, shift, kwargs)

    def for_src(
        self, clip: vs.VideoNode, width: int, height: int, shift: tuple[float, float], **kwargs: Any
    ) -> tuple[dict[str, Any], tuple[float, float]]:
        """
        Apply grid model using source sizes.

        Args:
            clip: Source clip (fallback for src dimensions).
            width: Source width.
            height: Source height.
            shift: Current shift.
            **kwargs: Optional overrides.

        Returns:
            (updated kwargs, updated shift).
        """

        src_width = fallback(kwargs.get("src_width"), clip.width)
        src_height = fallback(kwargs.get("src_height"), clip.height)

        return self(width, height, src_width, src_height, shift, kwargs)


type TopShift = float
"""
Type alias for vertical shift in pixels (top).

Represents the amount of vertical offset when scaling a video.
"""

type LeftShift = float
"""
Type alias for horizontal shift in pixels (left).

Represents the amount of horizontal offset when scaling a video.
"""

type TopFieldTopShift = float
"""
Type alias for the top field's vertical shift in pixels.

Used when processing interlaced video to describe the vertical shift of the top field.
"""

type TopFieldLeftShift = float
"""
Type alias for the top field's horizontal shift in pixels.

Used when processing interlaced video to describe the horizontal shift of the top field.
"""

type BotFieldTopShift = float
"""
Type alias for the bottom field's vertical shift in pixels.

Used when processing interlaced video to describe the vertical shift of the bottom field.
"""

type BotFieldLeftShift = float
"""
Type alias for the bottom field's horizontal shift in pixels.

Used when processing interlaced video to describe the horizontal shift of the bottom field.
"""


type FieldShift = tuple[
    TopShift | tuple[TopFieldTopShift, BotFieldTopShift], LeftShift | tuple[TopFieldLeftShift, BotFieldLeftShift]
]
"""
Type alias for shifts in interlaced content.

Represents separate shifts for top and bottom fields.
"""


type Slope = float
"""
Type alias for the slope of the sigmoid curve, controlling the steepness of the transition.
"""

type Center = float
"""
Type alias for the center point of the sigmoid curve, determining the midpoint of the transition.
"""
