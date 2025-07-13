from __future__ import annotations

import inspect
from abc import abstractmethod
from functools import partial, wraps
from typing import Any, Callable, Literal, Union, overload

from jetpytools import CustomValueError, FuncExceptT, KwargsT, P, T

from ..enums import (
    ChromaLocation,
    ChromaLocationT,
    ColorRange,
    ColorRangeT,
    FieldBased,
    FieldBasedT,
    Matrix,
    MatrixT,
    Primaries,
    PrimariesT,
    PropEnum,
    Transfer,
    TransferT,
)
from ..functions import DitherType, check_variable_format, depth
from ..types import ConstantFormatVideoNode, HoldsVideoFormatT, VideoFormatT, VideoNodeT
from . import vs_proxy as vs
from .cache import DynamicClipsCache
from .info import get_depth

__all__ = [
    "ProcessVariableClip",
    "ProcessVariableFormatClip",
    "ProcessVariableResClip",
    "ProcessVariableResFormatClip",
    "finalize_clip",
    "finalize_output",
    "initialize_clip",
    "initialize_input",
]


def finalize_clip(
    clip: vs.VideoNode,
    bits: VideoFormatT | HoldsVideoFormatT | int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    *,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    """
    Finalize a clip for output to the encoder.

    Args:
        clip: Clip to output.
        bits: Bitdepth to output to.
        clamp_tv_range: Whether to clamp to tv range.
        dither_type: Dithering used for the bitdepth conversion.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Dithered down and optionally clamped clip.
    """
    from ..functions import limiter

    assert check_variable_format(clip, func or finalize_clip)

    if bits:
        clip = depth(clip, bits, dither_type=dither_type)

    if clamp_tv_range:
        clip = limiter(clip, tv_range=clamp_tv_range)

    return clip


@overload
def finalize_output(
    function: Callable[P, vs.VideoNode],
    /,
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None,
) -> Callable[P, ConstantFormatVideoNode]: ...


@overload
def finalize_output(
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None,
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, ConstantFormatVideoNode]]: ...


def finalize_output(
    function: Callable[P, vs.VideoNode] | None = None,
    /,
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None,
) -> Union[Callable[P, vs.VideoNode], Callable[[Callable[P, vs.VideoNode]], Callable[P, ConstantFormatVideoNode]]]:
    """
    Decorator implementation of [finalize_clip][vstools.finalize_clip].
    """

    if function is None:
        return partial(finalize_output, bits=bits, clamp_tv_range=clamp_tv_range, dither_type=dither_type, func=func)

    @wraps(function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> ConstantFormatVideoNode:
        return finalize_clip(function(*args, **kwargs), bits, clamp_tv_range, dither_type, func=func)

    return _wrapper


def initialize_clip(
    clip: vs.VideoNode,
    bits: int | None = None,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    *,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    """
    Initialize a clip with default props.

    It is HIGHLY recommended to always use this function at the beginning of your scripts!

    Args:
        clip: Clip to initialize.
        bits: Bits to dither to.

               - If 0, no dithering is applied.
               - If None, 16 if bit depth is lower than it, else leave untouched.
               - If positive integer, dither to that bitdepth.

        matrix: Matrix property to set. If None, tries to get the Matrix from existing props. If no props are set or
            Matrix=2, guess from the video resolution.
        transfer: Transfer property to set. If None, tries to get the Transfer from existing props. If no props are set
            or Transfer=2, guess from the video resolution.
        primaries: Primaries property to set. If None, tries to get the Primaries from existing props. If no props are
            set or Primaries=2, guess from the video resolution.
        chroma_location: ChromaLocation prop to set. If None, tries to get the ChromaLocation from existing props. If no
            props are set, guess from the video resolution.
        color_range: ColorRange prop to set. If None, tries to get the ColorRange from existing props. If no props are
            set, assume Limited Range.
        field_based: FieldBased prop to set. If None, tries to get the FieldBased from existing props. If no props are
            set, assume PROGRESSIVE.
        strict: Whether to be strict about existing properties. If True, throws an exception if certain frame properties
            are not found.
        dither_type: Dithering used for the bitdepth conversion.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Clip with relevant frame properties set, and optionally dithered up to 16 bits by default.
    """

    func = func or initialize_clip

    assert check_variable_format(clip, func)

    values: list[tuple[type[PropEnum], Any]] = [
        (Matrix, matrix),
        (Transfer, transfer),
        (Primaries, primaries),
        (ChromaLocation, chroma_location),
        (ColorRange, color_range),
        (FieldBased, field_based),
    ]

    clip = PropEnum.ensure_presences(
        clip,
        [
            (cls if strict else cls.from_video(clip, False, func)) if value is None else cls.from_param(value, func)
            for cls, value in values
        ],
        func,
    )

    if bits is None:
        bits = max(get_depth(clip), 16)
    elif bits <= 0:
        return clip

    return depth(clip, bits, dither_type=dither_type)


@overload
def initialize_input(
    function: Callable[P, VideoNodeT],
    /,
    *,
    bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None,
) -> Callable[P, VideoNodeT]: ...


@overload
def initialize_input(
    *,
    bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None,
) -> Callable[[Callable[P, VideoNodeT]], Callable[P, VideoNodeT]]: ...


def initialize_input(
    function: Callable[P, vs.VideoNode] | None = None,
    /,
    *,
    bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None,
) -> Union[Callable[P, VideoNodeT], Callable[[Callable[P, VideoNodeT]], Callable[P, VideoNodeT]]]:
    """
    Decorator implementation of [initialize_clip][vstools.initialize_clip]
    """

    if function is None:
        return partial(
            initialize_input,
            bits=bits,
            matrix=matrix,
            transfer=transfer,
            primaries=primaries,
            chroma_location=chroma_location,
            color_range=color_range,
            field_based=field_based,
            strict=strict,
            dither_type=dither_type,
            func=func,
        )

    init_args = dict[str, Any](
        bits=bits,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
        chroma_location=chroma_location,
        color_range=color_range,
        field_based=field_based,
        strict=strict,
        dither_type=dither_type,
        func=func,
    )

    @wraps(function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> VideoNodeT:
        args_l = list(args)

        for i, obj in enumerate(args_l):
            if isinstance(obj, vs.VideoNode):
                args_l[i] = initialize_clip(obj, **init_args)
                return function(*args_l, **kwargs)  # type: ignore

        kwargs2 = kwargs.copy()

        for name, obj in kwargs2.items():
            if isinstance(obj, vs.VideoNode):
                kwargs2[name] = initialize_clip(obj, **init_args)
                return function(*args, **kwargs2)  # type: ignore

        for name, param in inspect.signature(function).parameters.items():
            if param.default is not inspect.Parameter.empty and isinstance(param.default, vs.VideoNode):
                return function(*args, **kwargs2 | {name: initialize_clip(param.default, **init_args)})  # type: ignore

        raise CustomValueError(
            "No VideoNode found in positional, keyword, nor default arguments!", func or initialize_input
        )

    return _wrapper


class ProcessVariableClip(DynamicClipsCache[T, VideoNodeT]):
    """
    A helper class for processing variable format/resolution clip.
    """

    def __init__(
        self,
        clip: VideoNodeT,
        out_dim: tuple[int, int] | Literal[False] | None = None,
        out_fmt: int | vs.VideoFormat | Literal[False] | None = None,
        cache_size: int = 10,
    ) -> None:
        """
        Args:
            clip: Clip to process
            out_dim: Ouput dimension.
            out_fmt: Output format.
            cache_size: The maximum number of items allowed in the cache. Defaults to 10.
        """
        bk_args = KwargsT(length=clip.num_frames, keep=True, varformat=None)

        if out_dim is None:
            out_dim = (clip.width, clip.height)

        if out_fmt is None:
            out_fmt = clip.format or False

        if out_dim is not False and 0 in out_dim:
            out_dim = False

        if out_dim is False:
            bk_args.update(width=8, height=8, varsize=True)
        else:
            bk_args.update(width=out_dim[0], height=out_dim[1])

        if out_fmt is False:
            bk_args.update(format=vs.GRAY8, varformat=True)
        else:
            bk_args.update(format=out_fmt if isinstance(out_fmt, int) else out_fmt.id)

        super().__init__(cache_size)

        self.clip = clip
        self.out = vs.core.std.BlankClip(clip, **bk_args)

    def eval_clip(self) -> VideoNodeT:
        if self.out.format and (0 not in (self.out.width, self.out.height)):
            try:
                return self.get_clip(self.get_key(self.clip))
            except Exception:
                ...

        return vs.core.std.FrameEval(self.out, lambda n, f: self[self.get_key(f)], self.clip)

    def get_clip(self, key: T) -> VideoNodeT:
        return self.process(self.normalize(self.clip, key))

    @classmethod
    def from_clip(cls, clip: VideoNodeT) -> VideoNodeT:
        """
        Process a variable format/resolution clip.

        Args:
            clip: Clip to process.

        Returns:
            Processed clip.
        """
        return cls(clip).eval_clip()

    @classmethod
    def from_func(
        cls,
        clip: VideoNodeT,
        func: Callable[[VideoNodeT], VideoNodeT],
        out_dim: tuple[int, int] | Literal[False] | None = None,
        out_fmt: int | vs.VideoFormat | Literal[False] | None = None,
        cache_size: int = 10,
    ) -> VideoNodeT:
        """
        Process a variable format/resolution clip with a given function

        Args:
            clip: Clip to process.
            func: Function that takes and returns a single VideoNode.
            out_dim: Ouput dimension.
            out_fmt: Output format.
            cache_size: The maximum number of VideoNode allowed in the cache. Defaults to 10.

        Returns:
            Processed variable clip.
        """

        def process(self: ProcessVariableClip[T, VideoNodeT], clip: VideoNodeT) -> VideoNodeT:
            return func(clip)

        ns = cls.__dict__.copy()
        ns[process.__name__] = process

        return type(cls.__name__, cls.__bases__, ns)(clip, out_dim, out_fmt, cache_size).eval_clip()

    @abstractmethod
    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> T:
        """
        Generate a unique key based on the node or frame.
        This key will be used to temporarily assert a resolution and format for the clip to process.

        Args:
            frame: Node or frame from which the unique key is generated.

        Returns:
            Unique identifier.
        """

    @abstractmethod
    def normalize(self, clip: VideoNodeT, cast_to: T) -> VideoNodeT:
        """
        Normalize the given node to the format/resolution specified by the unique key `cast_to`.

        Args:
            clip: Clip to normalize.
            cast_to: The target resolution or format to which the clip should be cast or normalized.

        Returns:
            Normalized clip.
        """

    def process(self, clip: VideoNodeT) -> VideoNodeT:
        """
        Process the given clip.

        Args:
            clip: Clip to process.

        Returns:
            Processed clip.
        """
        return clip


class ProcessVariableResClip(ProcessVariableClip[tuple[int, int], VideoNodeT]):
    """
    A helper class for processing variable resolution clip.
    """

    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> tuple[int, int]:
        return (frame.width, frame.height)

    def normalize(self, clip: VideoNodeT, cast_to: tuple[int, int]) -> VideoNodeT:
        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), *cast_to)
        return vs.core.std.CopyFrameProps(normalized, clip)


class ProcessVariableFormatClip(ProcessVariableClip[vs.VideoFormat, vs.VideoNode]):
    """
    A helper class for processing variable format clip.
    """

    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> vs.VideoFormat:
        assert frame.format
        return frame.format

    def normalize(self, clip: vs.VideoNode, cast_to: vs.VideoFormat) -> ConstantFormatVideoNode:
        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), format=cast_to.id)
        return vs.core.std.CopyFrameProps(normalized, clip)


class ProcessVariableResFormatClip(ProcessVariableClip[tuple[int, int, vs.VideoFormat], vs.VideoNode]):
    """
    A helper class for processing variable format and resolution clip.
    """

    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> tuple[int, int, vs.VideoFormat]:
        assert frame.format
        return (frame.width, frame.height, frame.format)

    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int, vs.VideoFormat]) -> vs.VideoNode:
        w, h, fmt = cast_to

        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), w, h, fmt.id)

        return vs.core.std.CopyFrameProps(normalized, clip)
