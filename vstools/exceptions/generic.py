from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterable, Sized

import vapoursynth as vs
from jetpytools import (
    CustomKeyError,
    CustomOverflowError,
    CustomValueError,
    FuncExceptT,
    MismatchError,
    MismatchRefError,
    SupportsString,
)

from ..types import HoldsVideoFormatT, VideoFormatT

if TYPE_CHECKING:
    from ..enums import Resolution


__all__ = [
    "ClipLengthError",
    "FormatsMismatchError",
    "FormatsRefClipMismatchError",
    "FramePropError",
    "FramerateMismatchError",
    "FramerateRefClipMismatchError",
    "FramesLengthError",
    "InvalidColorFamilyError",
    "InvalidFramerateError",
    "InvalidSubsamplingError",
    "InvalidTimecodeVersionError",
    "InvalidVideoFormatError",
    "LengthMismatchError",
    "LengthRefClipMismatchError",
    "MismatchError",
    "MismatchRefError",
    "ResolutionsMismatchError",
    "ResolutionsRefClipMismatchError",
    "TopFieldFirstError",
    "UnsupportedColorFamilyError",
    "UnsupportedSubsamplingError",
    "UnsupportedVideoFormatError",
    "VariableFormatError",
    "VariableResolutionError",
]


class FramesLengthError(CustomOverflowError):
    def __init__(
        self,
        func: FuncExceptT,
        var_name: str,
        message: SupportsString = '"{var_name}" can\'t be greater than the clip length!',
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, var_name=var_name, **kwargs)


class ClipLengthError(CustomOverflowError):
    """
    Raised when a generic clip length error occurs.
    """


class VariableFormatError(CustomValueError):
    """
    Raised when clip is of a variable format.
    """

    def __init__(
        self, func: FuncExceptT, message: SupportsString = "Variable-format clips not supported!", **kwargs: Any
    ) -> None:
        super().__init__(message, func, **kwargs)


class VariableResolutionError(CustomValueError):
    """
    Raised when clip is of a variable resolution.
    """

    def __init__(
        self, func: FuncExceptT, message: SupportsString = "Variable-resolution clips not supported!", **kwargs: Any
    ) -> None:
        super().__init__(message, func, **kwargs)


class UnsupportedVideoFormatError(CustomValueError):
    """
    Raised when an undefined video format value is passed.
    """


class InvalidVideoFormatError(CustomValueError):
    """
    Raised when the given clip has an invalid format.
    """

    def __init__(
        self,
        func: FuncExceptT,
        format: VideoFormatT | HoldsVideoFormatT,
        message: SupportsString = "The format {format.name} is not supported!",
        **kwargs: Any,
    ) -> None:
        from ..utils import get_video_format

        super().__init__(message, func, format=get_video_format(format), **kwargs)


class UnsupportedColorFamilyError(CustomValueError):
    """
    Raised when an undefined color family value is passed.
    """


class InvalidColorFamilyError(CustomValueError):
    """
    Raised when the given clip uses an invalid format.
    """

    def __init__(
        self,
        func: FuncExceptT | None,
        wrong: VideoFormatT | HoldsVideoFormatT | vs.ColorFamily,
        correct: VideoFormatT
        | HoldsVideoFormatT
        | vs.ColorFamily
        | Iterable[VideoFormatT | HoldsVideoFormatT | vs.ColorFamily] = vs.YUV,
        message: SupportsString = "Input clip must be of {correct} color family, not {wrong}!",
        **kwargs: Any,
    ) -> None:
        from ..functions import to_arr
        from ..utils import get_color_family

        wrong_str = get_color_family(wrong).name

        super().__init__(
            message,
            func,
            wrong=wrong_str,
            correct=iter({get_color_family(c).name for c in to_arr(correct)}),  # type: ignore[arg-type]
            **kwargs,
        )

    @staticmethod
    def check(
        to_check: VideoFormatT | HoldsVideoFormatT | vs.ColorFamily,
        correct: VideoFormatT
        | HoldsVideoFormatT
        | vs.ColorFamily
        | Iterable[VideoFormatT | HoldsVideoFormatT | vs.ColorFamily],
        func: FuncExceptT | None = None,
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Check whether the given values are correct, and if not, throw this exception.

        Args:
            to_check: Value to check. Must be either a ColorFamily value, or a value a ColorFamily can be derived from
                such as VideoFormat.
            correct: A correct value or an array of correct color families.
            func: Function returned for custom error handling. This should only be set by VS package developers.
            message: Message to print when throwing the exception. The message will be formatted to display the correct
                and wrong values (`{correct}` and `{wrong}` respectively).
            **kwargs: Keyword arguments to pass on to the exception.

        Raises:
            InvalidColorFamilyError: Given color family is not in list of correct color families.
        """
        from ..functions import to_arr
        from ..utils import get_color_family

        to_check = get_color_family(to_check)
        correct_list = [get_color_family(c) for c in to_arr(correct)]  # type: ignore[arg-type]

        if to_check not in correct_list:
            if message is not None:
                kwargs.update(message=message)
            raise InvalidColorFamilyError(func, to_check, correct_list, **kwargs)


class UnsupportedSubsamplingError(CustomValueError):
    """
    Raised when an undefined subsampling value is passed.
    """


class InvalidSubsamplingError(CustomValueError):
    """
    Raised when the given clip has invalid subsampling.
    """

    def __init__(
        self,
        func: FuncExceptT,
        subsampling: str | VideoFormatT | HoldsVideoFormatT,
        message: SupportsString = "The subsampling {subsampling} is not supported!",
        **kwargs: Any,
    ) -> None:
        from ..utils import get_video_format

        subsampling = subsampling if isinstance(subsampling, str) else get_video_format(subsampling).name

        super().__init__(message, func, subsampling=subsampling, **kwargs)


class FormatsMismatchError(MismatchError):
    """
    Raised when clips with different formats are given.
    """

    @classmethod
    def _item_to_name(cls, item: VideoFormatT | HoldsVideoFormatT) -> str:
        from ..utils import get_video_format

        return get_video_format(item).name

    def __init__(
        self,
        func: FuncExceptT,
        formats: Iterable[VideoFormatT | HoldsVideoFormatT],
        message: SupportsString = "All specified formats must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, formats, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(cls, func: FuncExceptT, /, *formats: VideoFormatT | HoldsVideoFormatT, **kwargs: Any) -> None: ...


class FormatsRefClipMismatchError(MismatchRefError, FormatsMismatchError):
    """
    Raised when a ref clip and the main clip have different formats.
    """

    def __init__(
        self,
        func: FuncExceptT,
        clip: VideoFormatT | HoldsVideoFormatT,
        ref: VideoFormatT | HoldsVideoFormatT,
        message: SupportsString = "The format of ref and main clip must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls,
            func: FuncExceptT,
            clip: VideoFormatT | HoldsVideoFormatT,
            ref: VideoFormatT | HoldsVideoFormatT,
            /,
            **kwargs: Any,
        ) -> None: ...


class ResolutionsMismatchError(MismatchError):
    """
    Raised when clips with different resolutions are given.
    """

    @classmethod
    def _item_to_name(cls, item: Resolution | vs.VideoNode) -> str:
        from ..enums import Resolution

        return str(item if isinstance(item, Resolution) else Resolution.from_video(item))

    def __init__(
        self,
        func: FuncExceptT,
        resolutions: Iterable[Resolution | vs.VideoNode],
        message: SupportsString = "All the resolutions must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, resolutions, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(cls, func: FuncExceptT, /, *resolutions: Resolution | vs.VideoNode, **kwargs: Any) -> None: ...


class ResolutionsRefClipMismatchError(MismatchRefError, ResolutionsMismatchError):
    """
    Raised when a ref clip and the main clip have different resolutions.
    """

    def __init__(
        self,
        func: FuncExceptT,
        clip: Resolution | vs.VideoNode,
        ref: Resolution | vs.VideoNode,
        message: SupportsString = "The resolution of ref and main clip must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls, func: FuncExceptT, clip: Resolution | vs.VideoNode, ref: Resolution | vs.VideoNode, /, **kwargs: Any
        ) -> None: ...


class LengthMismatchError(MismatchError):
    """
    Raised when clips with a different number of total frames are given.
    """

    @classmethod
    def _item_to_name(cls, item: int | Sized) -> str:
        return str(int(item if isinstance(item, int) else len(item)))

    def __init__(
        self,
        func: FuncExceptT,
        lengths: Iterable[int | Sized],
        message: SupportsString = "All the lengths must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, lengths, message, iter(map(self._item_to_name, lengths)), **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(cls, func: FuncExceptT, /, *lengths: int | Sized, **kwargs: Any) -> None: ...


class LengthRefClipMismatchError(MismatchRefError, LengthMismatchError):
    """
    Raised when a ref clip and the main clip have a different number of total frames.
    """

    def __init__(
        self,
        func: FuncExceptT,
        clip: int | vs.RawNode,
        ref: int | vs.RawNode,
        message: SupportsString = "The main clip and ref clip length must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls, func: FuncExceptT, clip: int | vs.RawNode, ref: int | vs.RawNode, /, **kwargs: Any
        ) -> None: ...


class FramerateMismatchError(MismatchError):
    """
    Raised when clips with a different framerate are given.
    """

    @classmethod
    def _item_to_name(cls, item: vs.VideoNode | Fraction | tuple[int, int] | float) -> str:
        from ..utils import get_framerate

        return str(get_framerate(item))

    def __init__(
        self,
        func: FuncExceptT,
        framerates: Iterable[vs.VideoNode | Fraction | tuple[int, int] | float],
        message: SupportsString = "All the framerates must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, framerates, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(
            cls, func: FuncExceptT, /, *framerates: vs.VideoNode | Fraction | tuple[int, int] | float, **kwargs: Any
        ) -> None: ...


class FramerateRefClipMismatchError(MismatchRefError, FramerateMismatchError):
    """
    Raised when a ref clip and the main clip have a different framerate.
    """

    def __init__(
        self,
        func: FuncExceptT,
        clip: vs.VideoNode | Fraction | tuple[int, int] | float,
        ref: vs.VideoNode | Fraction | tuple[int, int] | float,
        message: SupportsString = "The framerate of the ref and main clip must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls,
            func: FuncExceptT,
            clip: vs.VideoNode | Fraction | tuple[int, int] | float,
            ref: vs.VideoNode | Fraction | tuple[int, int] | float,
            /,
            **kwargs: Any,
        ) -> None: ...


class FramePropError(CustomKeyError):
    """
    Raised when there is an error with the frame props.
    """

    def __init__(
        self,
        func: FuncExceptT,
        key: str,
        message: SupportsString = 'Error while trying to get frame prop "{key}"!',
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, key=key, **kwargs)


class TopFieldFirstError(CustomValueError):
    """
    Raised when the user must pass a TFF argument.
    """

    def __init__(
        self, func: FuncExceptT, message: SupportsString = "You must set `tff` for this clip!", **kwargs: Any
    ) -> None:
        super().__init__(message, func, **kwargs)


class InvalidFramerateError(CustomValueError):
    """
    Raised when the given clip has an invalid framerate.
    """

    def __init__(
        self,
        func: FuncExceptT,
        clip: vs.VideoNode | Fraction,
        message: SupportsString = "{fps} clips are not allowed!",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, fps=clip.fps if isinstance(clip, vs.VideoNode) else clip, **kwargs)

    @staticmethod
    def check(
        func: FuncExceptT,
        to_check: vs.VideoNode | Fraction | tuple[int, int] | float,
        correct: vs.VideoNode
        | Fraction
        | tuple[int, int]
        | float
        | Iterable[vs.VideoNode | Fraction | tuple[int, int] | float],
        message: SupportsString = "Input clip must have {correct} framerate, not {wrong}!",
        **kwargs: Any,
    ) -> None:
        """
        Check whether the given values are correct, and if not, throw this exception.

        Args:
            to_check: Value to check. Must be either a VideoNode holding the correct framerate, a Fraction, a tuple
                representing a fraction, or a float.
            correct: A correct value or an array of correct values.
            func: Function returned for custom error handling. This should only be set by VS package developers.
            message: Message to print when throwing the exception. The message will be formatted to display the correct
                and wrong values (`{correct}` and `{wrong}` respectively).
            **kwargs: Keyword arguments to pass on to the exception.

        Raises:
            InvalidFramerateError: Given framerate is not in list of correct framerates.
        """
        from ..utils import get_framerate

        to_check = get_framerate(to_check)

        def _resolve_correct(val: Any) -> Iterable[vs.VideoNode | Fraction | tuple[int, int] | float]:
            if isinstance(val, Iterable):
                if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, int) for x in val):
                    return [val]
                return val
            return [val]

        correct_list = [get_framerate(c) for c in _resolve_correct(correct)]

        if to_check not in correct_list:
            raise InvalidFramerateError(
                func, to_check, message, wrong=to_check, correct=iter(set(correct_list)), **kwargs
            )


class InvalidTimecodeVersionError(CustomValueError):
    """
    Raised when an invalid timecode version is passed.
    """

    def __init__(
        self,
        func: FuncExceptT,
        version: int,
        message: SupportsString = "{version} is not a valid timecodes version!",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, version=version, **kwargs)

    @staticmethod
    def check(
        func: FuncExceptT,
        to_check: int,
        correct: int | Iterable[int] = [1, 2],
        message: SupportsString = "Timecodes version be in {correct}, not {wrong}!",
        **kwargs: Any,
    ) -> None:
        """
        Check whether the given values are correct, and if not, throw this exception.

        Args:
            func: Function returned for custom error handling. This should only be set by VS package developers.
            to_check: Value to check. Must be an integer representing the timecodes version.
            correct: A correct value or an array of correct values. Defaults to [1, 2] (V1, V2).
            message: Message to print when throwing the exception. The message will be formatted to display the correct
                and wrong values (`{correct}` and `{wrong}` respectively).
            **kwargs: Keyword arguments to pass on to the exception.

        Raises:
            InvalidTimecodeVersionError: Given timecodes version is not in list of correct versions.
        """
        from ..functions import to_arr

        correct_list = to_arr(correct)

        if to_check not in correct_list:
            raise InvalidTimecodeVersionError(
                func, to_check, message, wrong=to_check, correct=iter(set(correct_list)), **kwargs
            )
