from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterable, Mapping, Self

from jetpytools import (
    CustomEnum,
    CustomIntEnum,
    CustomStrEnum,
    CustomValueError,
    EnumABCMeta,
    FuncExcept,
    classproperty,
)

from ..vs_proxy import vs

__all__ = ["PropEnum"]


class PropEnum(CustomIntEnum, metaclass=EnumABCMeta):
    @classmethod
    @abstractmethod
    # Subclasses must implement a default value when value is None
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, (vs.VideoNode, vs.VideoFrame, Mapping)):
            return cls.from_video(value, func=cls)

        return None

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

    def is_unknown(self) -> bool:
        """
        Whether the value represents an unknown value.
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
        prop = cls.from_param(value, func_except)

        return prop if not prop.is_unknown() else cls.from_video(src, strict, func_except)

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

    value = get_prop(src, cls, int, cast=cls, default=cls.from_param(None, func), func=func)

    if value.is_unknown():
        if strict:
            raise exception("{class_name} is undefined.", func, class_name=cls, reason=value)

        if isinstance(src, Mapping):
            raise exception("Can't determine {class_name} from FrameProps.", func, class_name=cls)

        if src.width and src.height:
            return cls.from_res(src)

    return value
