from __future__ import annotations

from string import capwords
from typing import TYPE_CHECKING, Any, Iterable, TypeVar, overload

import vapoursynth as vs

from jetpytools import CustomError, CustomIntEnum, FuncExceptT, classproperty
from typing_extensions import Self

__all__ = [
    'PropEnum',

    '_base_from_video',

    '_MatrixMeta',
    '_TransferMeta',
    '_PrimariesMeta',
    '_ColorRangeMeta',
    '_ChromaLocationMeta',
    '_FieldBasedMeta'
]


class PropEnum(CustomIntEnum):
    __name__: str

    @classmethod
    def is_unknown(cls, value: int | Self) -> bool:
        """Whether the value represents an unknown value."""

        return False

    @classproperty
    def prop_key(cls) -> str:
        """The key used in props to store the enum."""

        return f'_{cls.__name__}'

    if TYPE_CHECKING:
        def __new__(
            cls, value: int | Self | vs.VideoNode | vs.VideoFrame | vs.FrameProps | None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Self, func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Self | None, func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """Get the enum member from its int representation."""

    @classmethod
    def _missing_(cls, value: Any) -> Self | None:
        if isinstance(value, vs.VideoNode | vs.VideoFrame | vs.FrameProps):
            return cls.from_video(value)
        return super().from_param(value)

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> Self:
        """Get an enum member from the video resolution with heuristics."""

        raise NotImplementedError

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | vs.FrameProps, strict: bool = False,
        func: FuncExceptT | None = None
    ) -> Self:
        """Get an enum member from the frame properties or optionally fall back to resolution when strict=False."""

        raise NotImplementedError

    @classmethod
    def from_param_or_video(
        cls, value: Any,
        src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
        strict: bool = False, func_except: FuncExceptT | None = None
    ) -> Self:
        """
        Get the enum member from a value that can be casted to this prop value
        or grab it from frame properties.

        If `strict=False`, gather the heuristics using the clip's size or format.

        :param value:           Value to cast.
        :param src:             Clip to get prop from.
        :param strict:          Be strict about the frame properties. Default: False.
        :param func_except:     Function returned for custom error handling.
        """
        value = cls.from_param(value, func_except)

        if value is not None:
            return value

        return cls.from_video(src, strict, func_except)

    @classmethod
    def ensure_presence(
        cls, clip: vs.VideoNode, value: int | Self | None, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Ensure the presence of the property in the VideoNode."""

        enum_value = cls.from_param_or_video(value, clip, True, func)

        return clip.std.SetFrameProp(enum_value.prop_key, enum_value.value)

    def apply(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Applies the property to the VideoNode."""

        return clip.std.SetFrameProp(self.prop_key, self.value)

    @staticmethod
    def ensure_presences(
        clip: vs.VideoNode, prop_enums: Iterable[type[PropEnumT] | PropEnumT], func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """Ensure the presence of multiple PropEnums at once."""

        return clip.std.SetFrameProps(**{
            value.prop_key: value.value
            for value in [
                cls if isinstance(cls, PropEnum) else cls.from_video(clip, True)
                for cls in prop_enums
            ]
        })

    @property
    def pretty_string(self) -> str:
        """Get a pretty, displayable string of the enum member."""

        return capwords(self.string.replace('_', ' '))

    @property
    def string(self) -> str:
        """Get the string representation used in resize plugin/encoders."""

        return self._name_.lower()

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """Check if the given value is a valid int value of this enum."""
        return int(value) in map(int, cls.__members__.values())


PropEnumT = TypeVar('PropEnumT', bound=PropEnum)


def _base_from_video(
    cls: type[PropEnumT], src: vs.VideoNode | vs.VideoFrame | vs.FrameProps, exception: type[CustomError],
    strict: bool, func: FuncExceptT | None = None
) -> PropEnumT:
    from ..utils import get_prop

    func = func or cls.from_video

    value = get_prop(src, cls, int, default=None, func=func)

    if value is None or cls.is_unknown(value):
        if strict:
            raise exception('{class_name} is undefined.', func, class_name=cls, reason=value)

        if isinstance(src, vs.FrameProps):
            raise exception('Can\'t determine {class_name} from FrameProps.', func, class_name=cls)

        if all(hasattr(src, x) for x in ('width', 'height')):
            return cls.from_res(src)

    return cls(value)


if TYPE_CHECKING:
    from .color import ColorRange, ColorRangeT, Matrix, MatrixT, Primaries, PrimariesT, Transfer, TransferT
    from .generic import ChromaLocation, ChromaLocationT, FieldBased, FieldBasedT

    class _MatrixMeta(PropEnum, vs.MatrixCoefficients):  # type: ignore[misc]
        def __new__(cls, value: MatrixT) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Matrix | MatrixT, func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Matrix | MatrixT | None, func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """
            Determine the Matrix through a parameter.

            :param value:           Value or Matrix object.
            :param func_except:     Function returned for custom error handling.

            :return:                Matrix object or None.
            """

        @classmethod
        def from_param_or_video(
            cls, value: int | Matrix | MatrixT | None,
            src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
            strict: bool = False, func_except: FuncExceptT | None = None
        ) -> Matrix:
            ...

    class _TransferMeta(PropEnum, vs.TransferCharacteristics):  # type: ignore[misc]
        def __new__(cls, value: TransferT) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Transfer | TransferT, func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Transfer | TransferT | None, func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """
            Determine the Transfer through a parameter.

            :param value:           Value or Transfer object.
            :param func_except:     Function returned for custom error handling.
                                    This should only be set by VS package developers.

            :return:                Transfer object or None.
            """

        @classmethod
        def from_param_or_video(
            cls, value: int | Transfer | TransferT | None,
            src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
            strict: bool = False, func_except: FuncExceptT | None = None
        ) -> Transfer:
            ...

    class _PrimariesMeta(PropEnum, vs.ColorPrimaries):  # type: ignore[misc]
        def __new__(cls, value: PrimariesT) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Primaries | PrimariesT, func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | Primaries | PrimariesT | None, func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """
            Determine the Primaries through a parameter.

            :param value:           Value or Primaries object.
            :param func_except:     Function returned for custom error handling.
                                    This should only be set by VS package developers.

            :return:                Primaries object or None.
            """

        @classmethod
        def from_param_or_video(
            cls, value: int | Primaries | PrimariesT | None,
            src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
            strict: bool = False, func_except: FuncExceptT | None = None
        ) -> Primaries:
            ...

    class _ColorRangeMeta(PropEnum, vs.ColorPrimaries):  # type: ignore[misc]
        def __new__(cls, value: ColorRangeT) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | ColorRange | ColorRangeT, func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | ColorRange | ColorRangeT | None, func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """
            Determine the ColorRange through a parameter.

            :param value:           Value or ColorRange object.
            :param func_except:     Function returned for custom error handling.
                                    This should only be set by VS package developers.

            :return:                ColorRange object or None.
            """

        @classmethod
        def from_param_or_video(
            cls, value: int | ColorRange | ColorRangeT | None,
            src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
            strict: bool = False, func_except: FuncExceptT | None = None
        ) -> ColorRange:
            ...

    class _ChromaLocationMeta(PropEnum, vs.ChromaLocation):  # type: ignore[misc]
        def __new__(cls, value: ChromaLocationT) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | ChromaLocation | ChromaLocationT,
            func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value: int | ChromaLocation | ChromaLocationT | None,
            func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value: Any, func_except: Any = None) -> Self | None:
            """
            Determine the ChromaLocation through a parameter.

            :param value:           Value or ChromaLocation object.
            :param func_except:     Function returned for custom error handling.
                                    This should only be set by VS package developers.

            :return:                ChromaLocation object or None.
            """

        @classmethod
        def from_param_or_video(
            cls, value: int | ChromaLocation | ChromaLocationT | None,
            src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
            strict: bool = False, func_except: FuncExceptT | None = None
        ) -> ChromaLocation:
            ...

    class _FieldBasedMeta(PropEnum, vs.FieldBased):  # type: ignore[misc]
        def __new__(cls, value: FieldBasedT) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value_or_tff: None, func_except: FuncExceptT | None = None
        ) -> None:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value_or_tff: int | FieldBasedT | bool, func_except: FuncExceptT | None = None
        ) -> Self:
            ...

        @overload
        @classmethod
        def from_param(
            cls, value_or_tff: int | FieldBasedT | bool | None, func_except: FuncExceptT | None = None
        ) -> Self | None:
            ...

        @classmethod
        def from_param(cls, value_or_tff: Any, func_except: Any = None) -> Self | None:
            """
            Determine the type of field through a parameter.

            :param value_or_tff:    Value or FieldBased object.
                                    If it's bool, it specifies whether it's TFF or BFF.
            :param func_except:     Function returned for custom error handling.
                                    This should only be set by VS package developers.

            :return:                FieldBased object or None.
            """

        @classmethod
        def from_param_or_video(
            cls, value_or_tff: int | FieldBasedT | bool | None,
            src: vs.VideoNode | vs.VideoFrame | vs.FrameProps,
            strict: bool = False, func_except: FuncExceptT | None = None
        ) -> FieldBased:
            ...

        @classmethod
        def ensure_presence(
            cls, clip: vs.VideoNode, tff: bool | int | FieldBasedT | None, func: FuncExceptT | None = None
        ) -> vs.VideoNode:
            ...
else:
    _MatrixMeta = _TransferMeta = _PrimariesMeta = _ColorRangeMeta = _ChromaLocationMeta = PropEnum

    class _FieldBasedMeta(PropEnum):
        @classmethod
        def ensure_presence(
            cls, clip: vs.VideoNode, tff: int | FieldBasedT | bool | None, func: FuncExceptT | None = None
        ) -> vs.VideoNode:
            field_based = cls.from_param_or_video(tff, clip, True, func)

            return clip.std.SetFieldBased(field_based.value)
