from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, TypeAlias, overload

import vapoursynth as vs
from jetpytools import FuncExcept
from typing_extensions import Self

from ..exceptions import (
    UndefinedChromaLocationError,
    UndefinedFieldBasedError,
    UnsupportedChromaLocationError,
    UnsupportedFieldBasedError,
)
from ..types import HoldsVideoFormat, VideoFormatLike
from .base import PropEnum, _base_from_video

__all__ = ["ChromaLocation", "ChromaLocationLike", "ChromaLocationT", "FieldBased", "FieldBasedLike", "FieldBasedT"]


class ChromaLocation(PropEnum):
    """
    Chroma sample position in YUV formats.
    """

    _value_: int

    @classmethod
    def _missing_(cls, value: Any) -> ChromaLocation | None:
        if value is None:
            return cls.LEFT

        if value > cls.BOTTOM:
            raise UnsupportedChromaLocationError(f"ChromaLocation({value}) is unsupported.", cls)

        return None

    LEFT = 0
    CENTER = 1
    TOP_LEFT = 2
    TOP = 3
    BOTTOM_LEFT = 4
    BOTTOM = 5

    if TYPE_CHECKING:

        @overload
        @classmethod
        def from_param(cls, value: None, func_except: FuncExcept | None = None) -> None: ...

        @overload
        @classmethod
        def from_param(cls, value: ChromaLocationLike, func_except: FuncExcept | None = None) -> Self: ...

        @overload
        @classmethod
        def from_param(cls, value: ChromaLocationLike | None, func_except: FuncExcept | None = None) -> Self | None: ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """
            Determine the ChromaLocation through a parameter.

            Args:
                value: Value or ChromaLocation object.
                func_except: Function returned for custom error handling. This should only be set by VS package
                    developers.

            Returns:
                ChromaLocation object or None.
            """

        @classmethod
        def from_param_or_video(
            cls,
            value: ChromaLocation | None,
            src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any],
            strict: bool = False,
            func_except: FuncExcept | None = None,
        ) -> ChromaLocation: ...

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
            UndefinedChromaLocationError: Chroma location is undefined.
            UndefinedChromaLocationError: Chroma location can not be determined from the frame properties.
        """

        return _base_from_video(cls, src, UndefinedChromaLocationError, strict, func)

    def get_offsets(self, src: int | VideoFormatLike | HoldsVideoFormat) -> tuple[float, float]:
        """
        Get (left,top) shift for chroma relative to luma.

        This is only useful if you MUST use a pre-specified chroma location and shift the chroma yourself.
        """
        from ..utils import get_video_format

        fmt = get_video_format(src)

        off_left = off_top = 0.0

        if self in {ChromaLocation.LEFT, ChromaLocation.TOP_LEFT, ChromaLocation.BOTTOM_LEFT}:
            off_left = 0.5 - 2 ** (fmt.subsampling_w - 1)

        if self in {ChromaLocation.TOP, ChromaLocation.TOP_LEFT}:
            off_top = 0.5 - 2 ** (fmt.subsampling_h - 1)
        elif self in {ChromaLocation.BOTTOM, ChromaLocation.BOTTOM_LEFT}:
            off_top = 2 ** (fmt.subsampling_h - 1) - 0.5

        return off_left, off_top


class FieldBased(PropEnum):
    """
    Whether the frame is composed of two independent fields (interlaced) and their order if so.
    """

    _value_: int

    @classmethod
    def _missing_(cls: type[FieldBased], value: Any) -> FieldBased | None:
        value = super()._missing_(value)

        if value is None:
            return cls.PROGRESSIVE

        if value > cls.TFF:
            raise UnsupportedFieldBasedError(f"FieldBased({value}) is unsupported.", cls)

        return None

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

    @overload
    @classmethod
    def from_param(cls, value: None, func_except: FuncExcept | None = None) -> None: ...

    @overload
    @classmethod
    def from_param(cls, value: FieldBasedLike | bool, func_except: FuncExcept | None = None) -> Self: ...

    @overload
    @classmethod
    def from_param(cls, value: FieldBasedLike | bool | None, func_except: FuncExcept | None = None) -> Self | None: ...

    @classmethod
    def from_param(cls, value: FieldBasedLike | bool | None, func_except: FuncExcept | None = None) -> Self | None:
        """
        Determine the type of field through a parameter.

        Args:
            value: Value or FieldBased object. If it's bool, it specifies whether it's TFF or BFF.
            func_except: Function returned for custom error handling. This should only be set by VS package
                developers.

        Returns:
            FieldBased object or None.
        """
        if isinstance(value, bool):
            return cls(1 + value)

        return super().from_param(value)

    if TYPE_CHECKING:

        @classmethod
        def from_param_or_video(
            cls,
            value: FieldBasedLike | bool | None,
            src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any],
            strict: bool = False,
            func_except: FuncExcept | None = None,
        ) -> FieldBased: ...

    @classmethod
    def ensure_presence(
        cls, clip: vs.VideoNode, tff: FieldBasedLike | bool | None, func: FuncExcept | None = None
    ) -> vs.VideoNode:
        field_based = cls.from_param_or_video(tff, clip, True, func)

        return vs.core.std.SetFieldBased(clip, field_based.value)

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

        Returns:
            FieldBased object.

        Raises:
            UndefinedFieldBasedError: Field order is undefined.
            UndefinedFieldBasedError: Field order can not be determined from the frame properties.
        """

        return _base_from_video(cls, src, UndefinedFieldBasedError, strict, func)

    @property
    def is_inter(self) -> bool:
        """
        Check whether the value belongs to an interlaced value.
        """

        return self != FieldBased.PROGRESSIVE

    @property
    def field(self) -> int:
        """
        Check what field the enum signifies.

        Raises:
            UnsupportedFieldBasedError: PROGRESSIVE value is passed.
        """

        if self == self.PROGRESSIVE:
            raise UnsupportedFieldBasedError(
                "Progressive video aren't field based!", f"{self.__class__.__name__}.field"
            )

        return self.value - 1

    @property
    def is_tff(self) -> bool:
        """
        Check whether the value is Top-Field-First.
        """

        return self is self.TFF

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

        return FieldBased.BFF if self.is_tff else FieldBased.TFF

    @property
    def pretty_string(self) -> str:
        if self.is_inter:
            return f"{'Top' if self.is_tff else 'Bottom'} Field First"

        return super().pretty_string


ChromaLocationLike: TypeAlias = int | vs.ChromaLocation | ChromaLocation
"""Type alias for values that can be used to initialize a [ChromaLocation][vstools.ChromaLocation]."""

FieldBasedLike: TypeAlias = int | vs.FieldBased | FieldBased
"""Type alias for values that can be used to initialize a [FieldBased][vstools.FieldBased]."""

ChromaLocationT = ChromaLocationLike
"""Deprecated alias of ChromaLocationLike"""

FieldBasedT = FieldBasedLike
"""Deprecated alias of FieldBasedT = FieldBasedLike"""
