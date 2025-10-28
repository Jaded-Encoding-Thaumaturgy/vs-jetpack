from __future__ import annotations

from abc import abstractmethod
from contextlib import suppress
from string import capwords
from typing import Any, Iterable, Mapping, Self, overload

from jetpytools import (
    CustomIntEnum,
    CustomValueError,
    EnumABCMeta,
    FuncExcept,
    NotFoundEnumValueError,
    cachedproperty,
    classproperty,
    fallback,
)

from ..vs_proxy import vs

__all__ = ["PropEnum"]


class PropEnum(CustomIntEnum, metaclass=EnumABCMeta):
    """
    Base class for enumerations representing frame or clip properties in VapourSynth.
    """

    def __new__(cls, value: Any, *_: str | None) -> Self:
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    def __init__(self, _: Any, string: str | None = None, pretty_string: str | None = None) -> None:
        self._string = fallback(string, self._name_.lower())
        self._pretty_string = fallback(pretty_string, capwords(self._string.replace("_", " ")))

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, (vs.VideoNode, vs.VideoFrame, Mapping)):
            return cls.from_video(value, func=cls)

        return None

    @cachedproperty
    def pretty_string(self) -> str:
        """
        Get a pretty, displayable string of the enum member.
        """
        return self._pretty_string

    @cachedproperty
    def string(self) -> str:
        """
        Get the string representation used in resize plugin/encoders.
        """
        return self._string

    @classproperty
    @classmethod
    def prop_key(cls) -> str:
        """
        The key used in props to store the enum.
        """
        return f"_{cls.__name__}"

    def apply(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Applies the property to the VideoNode.
        """
        return clip.std.SetFrameProp(self.prop_key, self.value)

    def is_unspecified(self) -> bool:
        """
        Whether the value is unspecified.
        """
        return False

    @classmethod
    @abstractmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> Self:
        """
        Get an enum member from the video resolution with heuristics.
        """

    @classmethod
    @abstractmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> Self:
        """
        Get an enum member from the frame properties or optionally fall back to resolution when strict=False.
        """

    @overload
    @classmethod
    def from_param_with_fallback(cls, value: Any, fallback: None = None) -> Self | None: ...
    @overload
    @classmethod
    def from_param_with_fallback[T](cls, value: Any, fallback: T) -> Self | T: ...
    @classmethod
    def from_param_with_fallback[T](cls, value: Any, fallback: T | None = None) -> Self | T | None:
        """
        Attempts to obtain an enum member from a parameter value, returning a fallback if the value cannot be converted
        or represents an unspecified enum member.

        Args:
            value: The input value to convert into an enum member.
            fallback: The value to return if the input cannot be converted. Defaults to `None`.

        Returns:
            The corresponding enum member if conversion succeeds, otherwise the provided `fallback` value.
        """
        with suppress(NotFoundEnumValueError, CustomValueError):
            # Will raise a NotFoundEnumValueError if the value can't be casted
            casted = cls.from_param(value)

            # If unspecified, fallbacks to `fallback` value
            if casted.is_unspecified():
                raise CustomValueError

            return casted

        return fallback

    @classmethod
    def from_param_or_video(
        cls,
        value: Any,
        src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any],
        strict: bool = False,
        func_except: FuncExcept | None = None,
    ) -> Self:
        """
        Get the enum member from a value that can be casted to this prop value or grab it from frame properties.

        If `strict=False`, gather the heuristics using the clip's size or format.

        Args:
            value: Value to cast.
            src: Clip to get prop from.
            strict: Be strict about the frame properties. Default: False.
            func_except: Function returned for custom error handling.
        """
        prop = cls.from_param_with_fallback(value, None)

        # Delay the `from_video` call here to avoid unnecessary frame rendering
        if prop is None:
            return cls.from_video(src, strict, func_except)

        return prop

    @classmethod
    def ensure_presence(cls, clip: vs.VideoNode, value: Any, func: FuncExcept | None = None) -> vs.VideoNode:
        """
        Ensure the presence of the property in the VideoNode.
        """
        enum_value = cls.from_param_or_video(value, clip, True, func)

        return clip.std.SetFrameProp(enum_value.prop_key, enum_value.value)

    @staticmethod
    def ensure_presences(
        clip: vs.VideoNode, prop_enums: Iterable[type[PropEnum] | PropEnum], func: FuncExcept | None = None
    ) -> vs.VideoNode:
        """
        Ensure the presence of multiple PropEnums at once.
        """
        return clip.std.SetFrameProps(
            **{
                value.prop_key: value.value
                for value in (
                    prop_enum if isinstance(prop_enum, PropEnum) else prop_enum.from_video(clip, True, func)
                    for prop_enum in prop_enums
                )
            },
        )


def _base_from_video[PropEnumT: PropEnum](
    cls: type[PropEnumT],
    src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any],
    exception: type[CustomValueError],
    strict: bool,
    func: FuncExcept | None = None,
) -> PropEnumT:
    from ..utils import get_prop

    func = func or cls.from_video

    value = get_prop(src, cls, int, cast=cls, default=None, func=func)

    if value is None or value.is_unspecified():
        if strict:
            raise exception(f"{cls.__name__} is undefined.", func, value)

        if isinstance(src, Mapping):
            raise exception("Can't determine {class_name} from FrameProps.", func, class_name=cls)

        value = cls.from_res(src)

    return value
