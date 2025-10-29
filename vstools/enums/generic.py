from __future__ import annotations

from typing import Any, Mapping, NoReturn, SupportsInt

from jetpytools import CustomNotImplementedError, FuncExcept

from ..exceptions import (
    UndefinedChromaLocationError,
    UndefinedFieldBasedError,
    UndefinedFieldError,
    UnsupportedFieldBasedError,
)
from ..types import HoldsVideoFormat, VideoFormatLike
from ..vs_proxy import vs
from .base import PropEnum, _base_from_video

__all__ = ["ChromaLocation", "ChromaLocationLike", "ChromaLocationT", "FieldBased", "FieldBasedLike", "FieldBasedT"]


class ChromaLocation(PropEnum):
    """
    Chroma sample position in YUV formats.
    """

    LEFT = 0
    CENTER = 1
    TOP_LEFT = 2
    TOP = 3
    BOTTOM_LEFT = 4
    BOTTOM = 5

    def get_offsets(self, src: SupportsInt | VideoFormatLike | HoldsVideoFormat) -> tuple[float, float]:
        """
        Get (left,top) shift for chroma relative to luma.

        This is only useful if you MUST use a pre-specified chroma location and shift the chroma yourself.
        """
        from ..utils import get_video_format

        fmt = get_video_format(src)

        off_left = off_top = 0.0

        if self in [ChromaLocation.LEFT, ChromaLocation.TOP_LEFT, ChromaLocation.BOTTOM_LEFT]:
            off_left = 0.5 - 2 ** (fmt.subsampling_w - 1)

        if self in [ChromaLocation.TOP, ChromaLocation.TOP_LEFT]:
            off_top = 0.5 - 2 ** (fmt.subsampling_h - 1)
        elif self in [ChromaLocation.BOTTOM, ChromaLocation.BOTTOM_LEFT]:
            off_top = 2 ** (fmt.subsampling_h - 1) - 0.5

        return off_left, off_top

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> ChromaLocation:
        """
        Guess the chroma location based on the clip's resolution.

        Args:
            frame: Input clip or frame.

        Returns:
            ChromaLocation object.
        """
        return ChromaLocation.LEFT

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> ChromaLocation:
        """
        Obtain the chroma location of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the properties. The result may NOT be an unknown value.

        Returns:
            ChromaLocation object.

        Raises:
            UndefinedChromaLocationError: If chroma location is undefined or chroma location can not be determined
                from the frame properties.
        """
        return _base_from_video(cls, src, UndefinedChromaLocationError, strict, func)


class FieldBased(PropEnum):
    """
    Whether the frame is composed of two independent fields (interlaced) and their order if so.
    """

    PROGRESSIVE = 0
    """
    The frame is progressive.
    """

    BFF = 1
    """
    The frame is interlaced and the field order is bottom field first.
    """

    TFF = 2
    """
    The frame is interlaced and the field order is top field first.
    """

    @property
    def field(self) -> Field:
        """
        Check what field the enum signifies.

        Raises:
            UnsupportedFieldBasedError: If PROGRESSIVE value is passed.
        """
        if self is self.PROGRESSIVE:
            raise UnsupportedFieldBasedError(
                "Progressive video aren't field based!", f"{self.__class__.__name__}.field"
            )

        return Field.from_param(self.value - 1)

    @property
    def inverted_field(self) -> FieldBased:
        """
        Get the inverted field order.

        Raises:
            UnsupportedFieldBasedError: PROGRESSIVE value is passed.
        """
        if self == self.PROGRESSIVE:
            raise UnsupportedFieldBasedError(
                "Progressive video aren't field based!", f"{self.__class__.__name__}.inverted_field"
            )

        return FieldBased.BFF if self.is_tff() else FieldBased.TFF

    @property
    def pretty_string(self) -> str:
        if self.is_inter():
            return f"{'Top' if self.is_tff() else 'Bottom'} Field First"

        return super().pretty_string

    def is_inter(self) -> bool:
        """
        Check whether the value belongs to an interlaced value.
        """
        return self != FieldBased.PROGRESSIVE

    def is_tff(self) -> bool:
        """
        Check whether the value is Top-Field-First.
        """
        return self is self.TFF

    @classmethod
    def from_param(cls, value: Any, func_except: FuncExcept | None = None) -> FieldBased:
        """
        Determine the type of field through a parameter.

        Args:
            value: Value or FieldBased object. If it's bool, it specifies whether it's TFF or BFF.
            func_except: Function returned for custom error handling.

        Returns:
            FieldBased object.
        """
        if isinstance(value, bool):
            return cls(1 + value)

        return super().from_param(value, func_except)

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> FieldBased:
        """
        Guess the Field order from the frame resolution.
        """
        return cls.PROGRESSIVE

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> FieldBased:
        """
        Obtain the Field order of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the properties. Will ALWAYS error if the FieldBased is missing.
            func_except: Function returned for custom error handling.

        Returns:
            FieldBased object.

        Raises:
            UndefinedFieldBasedError: If the Field order is undefined or can not be determined
                from the frame properties.
        """
        return _base_from_video(cls, src, UndefinedFieldBasedError, strict, func)

    @classmethod
    def ensure_presence(cls, clip: vs.VideoNode, value: Any, func: FuncExcept | None = None) -> vs.VideoNode:
        """
        Ensure the presence of the property in the clip.

        Args:
            clip: Input clip.
            value: Value or FieldBased object. If it's bool, it specifies whether it's TFF or BFF.
            func_except: Function returned for custom error handling.

        Returns:
            Clip with the FieldBased set.
        """
        return clip.std.SetFieldBased(cls.from_param_or_video(value, clip, True, func))


class Field(PropEnum):
    """
    Indicates which field (top or bottom) was used to generate the current frame.

    This property is typically set when the frame is produced by a function such as `.std.SeparateFields`.
    """

    BOTTOM = 0
    """Frame generated from the bottom field."""

    TOP = 1
    """Frame generated from the top field."""

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> NoReturn:
        raise CustomNotImplementedError

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> Field:
        """
        Obtain the Field of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the properties. Will ALWAYS error if the Field is missing.
            func_except: Function returned for custom error handling.

        Returns:
            Field object.

        Raises:
            UndefinedFieldError: if the Field is undefined or can not be determined from the frame properties.
        """
        return _base_from_video(cls, src, UndefinedFieldError, strict, func)


type ChromaLocationLike = int | vs.ChromaLocation | ChromaLocation
"""Type alias for values that can be used to initialize a [ChromaLocation][vstools.ChromaLocation]."""

type FieldBasedLike = int | vs.FieldBased | FieldBased
"""Type alias for values that can be used to initialize a [FieldBased][vstools.FieldBased]."""

type FieldLike = int | Field
"""Type alias for values that can be used to initialize a [Field][vstools.Field]."""

ChromaLocationT = ChromaLocationLike
"""Deprecated alias of ChromaLocationLike"""

FieldBasedT = FieldBasedLike
"""Deprecated alias of FieldBasedT = FieldBasedLike"""
