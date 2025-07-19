from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, overload

from vsexprtools import ExprOp
from vsrgtools import box_blur
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    CustomTypeError,
    FrameRangeN,
    FrameRangesN,
    FramesLengthError,
    Position,
    Size,
    check_variable,
    depth,
    inject_self,
    limiter,
    normalize_seq,
    replace_ranges,
    vs,
)

__all__ = ["BoundingBox", "DeferredMask", "GeneralMask"]


class GeneralMask(ABC):
    """
    Abstract GeneralMask interface
    """

    @abstractmethod
    def get_mask(self, clip: vs.VideoNode, /, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode: ...

    @inject_self.init_kwargs.clean
    def apply_mask(
        self, _clipa: vs.VideoNode, _clipb: vs.VideoNode, _ref: vs.VideoNode | None = None, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return _clipa.std.MaskedMerge(_clipb, self.get_mask(_ref or _clipa, **kwargs))


class BoundingBox(GeneralMask):
    size: Size
    pos: Position
    invert: bool

    @overload
    def __init__(self, width: int, height: int, offset_x: int, offset_y: int, /, *, invert: bool = False) -> None: ...

    @overload
    def __init__(
        self, size: tuple[int, int] | Size, pos: tuple[int, int] | Position, /, *, invert: bool = False
    ) -> None: ...

    def __init__(self, *args: Any, invert: bool = False) -> None:
        if len(args) == 4:
            size, pos = (args[0], args[1]), (args[2], args[3])
        elif len(args) == 2:
            size, pos = args[0], args[1]
        else:
            raise CustomTypeError(None, self.__class__, args)

        self.size, self.pos, self.invert = Size(size), Position(pos), invert

    def get_mask(self, ref: vs.VideoNode, /, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
        from .utils import squaremask

        return squaremask(ref, self.size.x, self.size.y, self.pos.x, self.pos.y, self.invert, False, func=self.get_mask)


class DeferredMask(GeneralMask):
    """
    Abstract DeferredMask interface
    """

    ranges: FrameRangesN
    bound: BoundingBox | None
    refframes: list[int | None]
    blur: bool

    def __init__(
        self,
        ranges: FrameRangeN | FrameRangesN | None = None,
        bound: BoundingBox | None = None,
        *,
        blur: bool = False,
        refframes: int | list[int | None] | None = None,
    ) -> None:
        """
        Args:
            ranges: The frame ranges that the mask should be applied to.
            bound: An optional bounding box that defines the area of the frame where the mask will be applied. If None,
                the mask applies to the whole frame.
            blur: Whether to apply a box blur effect to the mask.
            refframes: A list of reference frames used in building the final mask for each specified range. Must have
                the same length as `ranges`.
        """
        self.ranges = ranges if isinstance(ranges, Sequence) else [(0, None)] if ranges is None else [ranges]
        self.blur = blur
        self.bound = bound

        if refframes is None:
            self.refframes = []
        else:
            self.refframes = refframes if isinstance(refframes, list) else normalize_seq(refframes, len(self.ranges))

        if len(self.refframes) > 0 and len(self.refframes) != len(self.ranges):
            raise FramesLengthError(self.__class__, "", "Received reference frame and range list size mismatch!")

    @limiter
    def get_mask(self, clip: vs.VideoNode, /, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        """
        Get the constructed mask

        Args:
            clip: Source clip.
            ref: Reference clip.
            **kwargs: Keyword arguments passed to the internal `_mask` method.

        Returns:
            Constructed mask
        """
        assert check_variable(clip, self.get_mask)
        assert check_variable(ref, self.get_mask)

        if self.refframes:
            hm = vs.core.std.BlankClip(
                ref, format=ref.format.replace(color_family=vs.GRAY, subsampling_h=0, subsampling_w=0).id, keep=True
            )

            for ran, rf in zip(self.ranges, self.refframes):
                if rf is None:
                    rf = ref.num_frames - 1
                elif rf < 0:
                    rf = ref.num_frames - 1 + rf

                mask = depth(
                    self._mask(clip[rf], ref[rf], **kwargs), clip, range_out=ColorRange.FULL, range_in=ColorRange.FULL
                )
                mask = vs.core.std.Loop(mask, hm.num_frames)

                hm = replace_ranges(hm, ExprOp.MAX.combine(hm, mask), ran)
        else:
            hm = depth(self._mask(clip, ref, **kwargs), clip, range_out=ColorRange.FULL, range_in=ColorRange.FULL)

        if self.bound:
            bm = self.bound.get_mask(hm, **kwargs)

            if self.blur:
                bm = box_blur(bm, 5, 5)

            hm = hm.std.BlankClip(keep=True).std.MaskedMerge(hm, bm)

        return hm

    @abstractmethod
    def _mask(self, clip: vs.VideoNode, ref: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode: ...
