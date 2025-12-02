from __future__ import annotations

from collections.abc import Callable, Iterable, Sized
from fractions import Fraction
from typing import TYPE_CHECKING, Any, SupportsInt

from jetpytools import (
    CustomKeyError,
    CustomOverflowError,
    CustomValueError,
    FuncExcept,
    MismatchError,
    MismatchRefError,
    SupportsString,
    to_arr,
)

from ..types import HoldsVideoFormat, VideoFormatLike
from ..vs_proxy import vs

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
    "LengthMismatchError",
    "LengthRefClipMismatchError",
    "MismatchError",
    "MismatchRefError",
    "ResolutionsMismatchError",
    "ResolutionsRefClipMismatchError",
    "TopFieldFirstError",
    "UnsupportedColorFamilyError",
    "UnsupportedFramerateError",
    "UnsupportedSampleTypeError",
    "UnsupportedSubsamplingError",
    "UnsupportedTimecodeVersionError",
    "UnsupportedVideoFormatError",
    "VariableFormatError",
    "VariableResolutionError",
]


class FramesLengthError(CustomOverflowError):
    def __init__(
        self,
        func: FuncExcept,
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
        self, func: FuncExcept, message: SupportsString = "Variable-format clips not supported!", **kwargs: Any
    ) -> None:
        super().__init__(message, func, **kwargs)


class VariableResolutionError(CustomValueError):
    """
    Raised when clip is of a variable resolution.
    """

    def __init__(
        self, func: FuncExcept, message: SupportsString = "Variable-resolution clips not supported!", **kwargs: Any
    ) -> None:
        super().__init__(message, func, **kwargs)


class _UnsupportedError(CustomValueError):
    def __init__(
        self,
        func: FuncExcept | None,
        wrong: Iterable[SupportsString],
        correct: Iterable[SupportsString],
        message: SupportsString,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, wrong=iter(wrong), correct=iter(correct), **kwargs)

    @classmethod
    def _check_builder[T](
        cls,
        norm_function: Callable[[T], Any],
        to_check: T | Iterable[T],
        correct: T | Iterable[T],
        func: FuncExcept | None = None,
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        to_check_set = {norm_function(c) for c in to_arr(to_check)}
        correct_set = {norm_function(c) for c in to_arr(correct)}

        if to_check_set - correct_set:
            if message is not None:
                kwargs.update(message=message)
            raise cls(func, to_check_set, correct_set, **kwargs)


class UnsupportedVideoFormatError(_UnsupportedError):
    """
    Raised when an video format value is not supported.
    """

    def __init__(
        self,
        func: FuncExcept | None,
        wrong: VideoFormatLike | HoldsVideoFormat | Iterable[VideoFormatLike | HoldsVideoFormat],
        correct: VideoFormatLike | HoldsVideoFormat | Iterable[VideoFormatLike | HoldsVideoFormat],
        message: SupportsString = "Input clip must be of {correct} format, not {wrong}.",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the exception.

        Args:
            func: Function where the exception was raised.
            wrong: The unsupported video format(s).
            correct: The expected video format(s).
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.
        """
        from ..utils import get_video_format

        super().__init__(
            func,
            {get_video_format(f).name for f in to_arr(wrong)},  # type: ignore[arg-type]
            {get_video_format(f).name for f in to_arr(correct)},  # type: ignore[arg-type]
            message,
            **kwargs,
        )

    @classmethod
    def check(
        cls,
        to_check: VideoFormatLike | HoldsVideoFormat | Iterable[VideoFormatLike | HoldsVideoFormat],
        correct: VideoFormatLike | HoldsVideoFormat | Iterable[VideoFormatLike | HoldsVideoFormat],
        func: FuncExcept | None = None,
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Ensure that the given video format(s) match the expected format(s).

        Args:
            to_check: The video format(s) to check.
            correct: The expected video format(s).
            func: Function where the exception was raised.
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.

        Raises:
            UnsupportedVideoFormatError: If any given format does not match the expected format(s).
        """
        from ..utils import get_video_format

        cls._check_builder(get_video_format, to_check, correct, func, message, **kwargs)  # type: ignore[arg-type]


class UnsupportedColorFamilyError(_UnsupportedError):
    """
    Raised when an color family value is not supported.
    """

    def __init__(
        self,
        func: FuncExcept | None,
        wrong: VideoFormatLike
        | HoldsVideoFormat
        | vs.ColorFamily
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.ColorFamily],
        correct: VideoFormatLike
        | HoldsVideoFormat
        | vs.ColorFamily
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.ColorFamily] = vs.YUV,
        message: SupportsString = "Input clip must be of {correct} color family, not {wrong}.",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the exception.

        Args:
            func: Function where the exception was raised.
            wrong: The unsupported color family(ies).
            correct: The expected color family(ies).
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.
        """
        from ..utils import get_color_family

        super().__init__(
            func,
            {get_color_family(c).name for c in to_arr(wrong)},  # type: ignore[arg-type]
            {get_color_family(c).name for c in to_arr(correct)},  # type: ignore[arg-type]
            message,
            **kwargs,
        )

    @classmethod
    def check(
        cls,
        to_check: VideoFormatLike
        | HoldsVideoFormat
        | vs.ColorFamily
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.ColorFamily],
        correct: VideoFormatLike
        | HoldsVideoFormat
        | vs.ColorFamily
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.ColorFamily],
        func: FuncExcept | None = None,
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Ensure that the given color family(ies) match the expected family(ies).

        Args:
            to_check: The color family(ies) to check.
            correct: The expected color family(ies).
            func: Function where the exception was raised.
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.

        Raises:
            UnsupportedColorFamilyError: If any given color family does not match the expected color family(ies).
        """
        from ..utils import get_color_family

        cls._check_builder(get_color_family, to_check, correct, func, message, **kwargs)  # type: ignore[arg-type]


class UnsupportedSampleTypeError(_UnsupportedError):
    """
    Raised when an sample type value is not supported.
    """

    def __init__(
        self,
        func: FuncExcept | None,
        wrong: VideoFormatLike
        | HoldsVideoFormat
        | vs.SampleType
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.SampleType],
        correct: VideoFormatLike
        | HoldsVideoFormat
        | vs.SampleType
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.SampleType],
        message: SupportsString = "Input clip must be of {correct} sample type, not {wrong}.",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the exception.

        Args:
            func: Function where the exception was raised.
            wrong: The unsupported sample type(s).
            correct: The expected sample type(s).
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.
        """
        from ..utils import get_sample_type

        super().__init__(
            func,
            {get_sample_type(c).name for c in to_arr(wrong)},  # type: ignore[arg-type]
            {get_sample_type(c).name for c in to_arr(correct)},  # type: ignore[arg-type]
            message,
            **kwargs,
        )

    @classmethod
    def check(
        cls,
        to_check: VideoFormatLike
        | HoldsVideoFormat
        | vs.SampleType
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.SampleType],
        correct: VideoFormatLike
        | HoldsVideoFormat
        | vs.SampleType
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.SampleType],
        func: FuncExcept | None = None,
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Ensure that the given sample type(s) match the expected type(s).

        Args:
            to_check: The sample type(s) to check.
            correct: The expected sample type(s).
            func: Function where the exception was raised.
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.

        Raises:
            UnsupportedSampleTypeError: If any given color family does not match the expected sample type(s).
        """
        from ..utils import get_sample_type

        cls._check_builder(get_sample_type, to_check, correct, func, message, **kwargs)  # type: ignore[arg-type]


class UnsupportedSubsamplingError(_UnsupportedError):
    """
    Raised when an subsampling value is not supported.
    """

    def __init__(
        self,
        func: FuncExcept,
        wrong: str | VideoFormatLike | HoldsVideoFormat | Iterable[str | VideoFormatLike | HoldsVideoFormat],
        correct: str | VideoFormatLike | HoldsVideoFormat | Iterable[str | VideoFormatLike | HoldsVideoFormat],
        message: SupportsString = "Input clip must be of {correct} subsampling, not {wrong}.",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the exception.

        Args:
            func: Function where the exception was raised.
            wrong: The unsupported subsampling(s).
            correct: The expected subsampling(s).
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.
        """
        from ..utils import get_subsampling

        super().__init__(
            func,
            {f if isinstance(f, str) else get_subsampling(f) for f in to_arr(wrong)},  # type: ignore[arg-type]
            {f if isinstance(f, str) else get_subsampling(f) for f in to_arr(correct)},  # type: ignore[arg-type]
            message,
            **kwargs,
        )

    @classmethod
    def check(
        cls,
        to_check: VideoFormatLike | HoldsVideoFormat | Iterable[VideoFormatLike | HoldsVideoFormat],
        correct: VideoFormatLike | HoldsVideoFormat | Iterable[VideoFormatLike | HoldsVideoFormat],
        func: FuncExcept | None = None,
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Ensure that the given subsampling(s) match the expected subsampling(s).

        Args:
            to_check: The subsampling(s) to check.
            correct: The expected subsampling(s).
            func: Function where the exception was raised.
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.

        Raises:
            UnsupportedSubsamplingError: If any given format does not match the expected format(s).
        """
        from ..utils import get_subsampling

        cls._check_builder(get_subsampling, to_check, correct, func, message, **kwargs)  # type: ignore[arg-type]


class UnsupportedFramerateError(_UnsupportedError):
    """
    Raised when a clip's framerate does not match the expected framerate.
    """

    def __init__(
        self,
        func: FuncExcept,
        wrong: vs.VideoNode | Fraction | Iterable[vs.VideoNode | Fraction],
        correct: vs.VideoNode | Fraction | Iterable[vs.VideoNode | Fraction],
        message: SupportsString = "Input clip must be of {correct} framerate, not {wrong}.",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the exception.

        Args:
            func: Function where the exception was raised.
            wrong: The unsupported framerate(s).
            correct: The expected framerate(s).
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.
        """
        from ..utils import get_framerate

        super().__init__(
            func,
            {get_framerate(f) for f in to_arr(wrong)},  # type: ignore[arg-type]
            {get_framerate(f) for f in to_arr(correct)},  # type: ignore[arg-type]
            message,
            **kwargs,
        )

    @classmethod
    def check(
        cls,
        to_check: vs.VideoNode
        | Fraction
        | tuple[int, int]
        | float
        | Iterable[vs.VideoNode | Fraction | tuple[int, int] | float],
        correct: vs.VideoNode
        | Fraction
        | tuple[int, int]
        | float
        | Iterable[vs.VideoNode | Fraction | tuple[int, int] | float],
        func: FuncExcept | None = None,
        message: SupportsString = "Input clip must have {correct} framerate, not {wrong}!",
        **kwargs: Any,
    ) -> None:
        """
        Ensure that the given framerate(s) match the expected framerate(s).

        Args:
            to_check: The framerate(s) to check.
            correct: The expected framerate(s).
            func: Function where the exception was raised.
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.

        Raises:
            UnsupportedFramerateError: If any given format does not match the expected format(s).
        """
        from ..utils import get_framerate

        def _resolve_correct(val: Any) -> Iterable[vs.VideoNode | Fraction | tuple[int, int] | float]:
            if isinstance(val, Iterable):
                if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, int) for x in val):
                    return [val]
                return val
            return [val]

        cls._check_builder(
            get_framerate, _resolve_correct(to_check), _resolve_correct(correct), func, message, **kwargs
        )


class UnsupportedTimecodeVersionError(_UnsupportedError):
    """
    Raised when a timecode version does not match the expected version.
    """

    def __init__(
        self,
        func: FuncExcept,
        wrong: SupportsInt | Iterable[SupportsInt],
        correct: SupportsInt | Iterable[SupportsInt] = (1, 2),
        message: SupportsString = "Timecodes version be in {correct}, not {wrong}.",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the exception.

        Args:
            func: Function where the exception was raised.
            wrong: The unsupported timecode(s).
            correct: The expected timecode(s).
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.
        """
        super().__init__(func, {int(t) for t in to_arr(wrong)}, {int(t) for t in to_arr(correct)}, message, **kwargs)

    @classmethod
    def check(
        cls,
        func: FuncExcept,
        to_check: SupportsInt | Iterable[SupportsInt],
        correct: SupportsInt | Iterable[SupportsInt] = (1, 2),
        message: SupportsString | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Ensure that the given timecode(s) match the expected timecode(s).

        Args:
            to_check: The timecode(s) to check.
            correct: The expected timecode(s).
            func: Function where the exception was raised.
            message: Exception message template supporting `{correct}` and `{wrong}` placeholders.
            **kwargs: Additional keyword arguments passed to the exception.

        Raises:
            UnsupportedTimecodeVersionError: If any given format does not match the expected format(s).
        """
        cls._check_builder(int, to_check, correct, func, message, **kwargs)


class FormatsMismatchError(MismatchError):
    """
    Raised when clips with different formats are given.
    """

    @classmethod
    def _item_to_name(cls, item: VideoFormatLike | HoldsVideoFormat) -> str:
        from ..utils import get_video_format

        return get_video_format(item).name

    def __init__(
        self,
        func: FuncExcept,
        formats: Iterable[VideoFormatLike | HoldsVideoFormat],
        message: SupportsString = "All specified formats must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, formats, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(cls, func: FuncExcept, /, *formats: VideoFormatLike | HoldsVideoFormat, **kwargs: Any) -> None: ...


class FormatsRefClipMismatchError(MismatchRefError, FormatsMismatchError):
    """
    Raised when a ref clip and the main clip have different formats.
    """

    def __init__(
        self,
        func: FuncExcept,
        clip: VideoFormatLike | HoldsVideoFormat,
        ref: VideoFormatLike | HoldsVideoFormat,
        message: SupportsString = "The format of ref and main clip must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls,
            func: FuncExcept,
            clip: VideoFormatLike | HoldsVideoFormat,
            ref: VideoFormatLike | HoldsVideoFormat,
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
        func: FuncExcept,
        resolutions: Iterable[Resolution | vs.VideoNode],
        message: SupportsString = "All the resolutions must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, resolutions, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(cls, func: FuncExcept, /, *resolutions: Resolution | vs.VideoNode, **kwargs: Any) -> None: ...


class ResolutionsRefClipMismatchError(MismatchRefError, ResolutionsMismatchError):
    """
    Raised when a ref clip and the main clip have different resolutions.
    """

    def __init__(
        self,
        func: FuncExcept,
        clip: Resolution | vs.VideoNode,
        ref: Resolution | vs.VideoNode,
        message: SupportsString = "The resolution of ref and main clip must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls, func: FuncExcept, clip: Resolution | vs.VideoNode, ref: Resolution | vs.VideoNode, /, **kwargs: Any
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
        func: FuncExcept,
        lengths: Iterable[int | Sized],
        message: SupportsString = "All the lengths must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, lengths, message, iter(map(self._item_to_name, lengths)), **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(cls, func: FuncExcept, /, *lengths: int | Sized, **kwargs: Any) -> None: ...


class LengthRefClipMismatchError(MismatchRefError, LengthMismatchError):
    """
    Raised when a ref clip and the main clip have a different number of total frames.
    """

    def __init__(
        self,
        func: FuncExcept,
        clip: int | vs.RawNode,
        ref: int | vs.RawNode,
        message: SupportsString = "The main clip and ref clip length must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, clip, ref, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(  # type: ignore[override]
            cls, func: FuncExcept, clip: int | vs.RawNode, ref: int | vs.RawNode, /, **kwargs: Any
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
        func: FuncExcept,
        framerates: Iterable[vs.VideoNode | Fraction | tuple[int, int] | float],
        message: SupportsString = "All the framerates must be equal!",
        **kwargs: Any,
    ) -> None:
        super().__init__(func, framerates, message, **kwargs)

    if TYPE_CHECKING:

        @classmethod
        def check(
            cls, func: FuncExcept, /, *framerates: vs.VideoNode | Fraction | tuple[int, int] | float, **kwargs: Any
        ) -> None: ...


class FramerateRefClipMismatchError(MismatchRefError, FramerateMismatchError):
    """
    Raised when a ref clip and the main clip have a different framerate.
    """

    def __init__(
        self,
        func: FuncExcept,
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
            func: FuncExcept,
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
        func: FuncExcept,
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
        self, func: FuncExcept, message: SupportsString = "You must set `tff` for this clip!", **kwargs: Any
    ) -> None:
        super().__init__(message, func, **kwargs)
