from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Self, TypeVar

import vapoursynth as vs
from jetpytools import CustomEnum, CustomError, CustomIntEnum, CustomStrEnum, FuncExcept, classproperty

__all__ = [
    "CustomEnum",
    "CustomIntEnum",
    "CustomStrEnum",
    "PropEnum",
    "_base_from_video",
]


class PropEnum(CustomIntEnum):
    @classmethod
    def is_unknown(cls, value: int | Self) -> bool:
        """
        Whether the value represents an unknown value.
        """

        return False

    @classproperty
    @classmethod
    def prop_key(cls) -> str:
        """
        The key used in props to store the enum.
        """

        return f"_{cls.__name__}"

    if TYPE_CHECKING:

        def __new__(cls, value: Any) -> Self: ...

    @classmethod
    def _missing_(cls, value: Any) -> Self | None:
        if isinstance(value, vs.VideoNode | vs.VideoFrame | vs.FrameProps):
            return cls.from_video(value)
        return super().from_param(value)

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> Self:
        """
        Get an enum member from the video resolution with heuristics.
        """

        raise NotImplementedError

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> Self:
        """
        Get an enum member from the frame properties or optionally fall back to resolution when strict=False.
        """

        raise NotImplementedError

    @classmethod
    def from_param_or_video(
        cls,
        value: Any,
        src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
        strict: bool = False,
        func_except: FuncExcept | None = None,
    ) -> Self:
        """
        Get the enum member from a value that can be casted to this prop value
        or grab it from frame properties.

        If `strict=False`, gather the heuristics using the clip's size or format.

        Args:
            value: Value to cast.
            src: Clip to get prop from.
            strict: Be strict about the frame properties. Default: False.
            func_except: Function returned for custom error handling.
        """
        value = cls.from_param(value, func_except)

        if value is not None:
            return value

        return cls.from_video(src, strict, func_except)

    @classmethod
    def ensure_presence(
        cls, clip: vs.VideoNode, value: int | Self | None, /, func: FuncExcept | None = None
    ) -> vs.VideoNode:
        """
        Ensure the presence of the property in the VideoNode.
        """

        enum_value = cls.from_param_or_video(value, clip, True, func)

        return vs.core.std.SetFrameProp(clip, enum_value.prop_key, enum_value.value)

    def apply(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Applies the property to the VideoNode.
        """

        return vs.core.std.SetFrameProp(clip, self.prop_key, self.value)

    @staticmethod
    def ensure_presences(
        clip: vs.VideoNode, prop_enums: Iterable[type[PropEnum] | PropEnum], func: FuncExcept | None = None
    ) -> vs.VideoNode:
        """
        Ensure the presence of multiple PropEnums at once.
        """

        return vs.core.std.SetFrameProps(
            clip,
            **{
                value.prop_key: value.value
                for value in (
                    prop_enum if isinstance(prop_enum, PropEnum) else prop_enum.from_video(clip, True, func)
                    for prop_enum in prop_enums
                )
            },
        )

    @property
    def pretty_string(self) -> str:
        """
        Get a pretty, displayable string of the enum member.
        """
        from string import capwords

        return capwords(self.string.replace("_", " "))

    @property
    def string(self) -> str:
        """
        Get the string representation used in resize plugin/encoders.
        """

        return self._name_.lower()

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """
        Check if the given value is a valid int value of this enum.
        """
        return int(value) in map(int, cls.__members__.values())


PropEnumT = TypeVar("PropEnumT", bound=PropEnum)


def _base_from_video(
    cls: type[PropEnumT],
    src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any],
    exception: type[CustomError],
    strict: bool,
    func: FuncExcept | None = None,
) -> PropEnumT:
    from ..utils import get_prop

    func = func or cls.from_video

    value = get_prop(src, cls, int, default=None, func=func)

    if value is None or cls.is_unknown(value):
        if strict:
            raise exception("{class_name} is undefined.", func, class_name=cls, reason=value)

        if isinstance(src, Mapping):
            raise exception("Can't determine {class_name} from FrameProps.", func, class_name=cls)

        if src.width and src.height:
            return cls.from_res(src)

    return cls(value)
