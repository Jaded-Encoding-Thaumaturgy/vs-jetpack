from __future__ import annotations

from abc import abstractmethod
from functools import partial, wraps
from inspect import signature
from typing import Any, Callable, Literal, overload

from jetpytools import CustomValueError, FuncExcept, StrictRange

from ..enums import (
    ChromaLocation,
    ChromaLocationLike,
    ColorRange,
    ColorRangeLike,
    FieldBased,
    FieldBasedLike,
    Matrix,
    MatrixLike,
    Primaries,
    PrimariesLike,
    PropEnum,
    Transfer,
    TransferLike,
)
from ..exceptions import FramesLengthError
from ..types import HoldsVideoFormat, VideoFormatLike
from ..utils import DynamicClipsCache, get_depth
from ..vs_proxy import vs
from .utils import DitherType, depth, limiter

__all__ = [
    "ProcessVariableClip",
    "ProcessVariableFormatClip",
    "ProcessVariableResClip",
    "ProcessVariableResFormatClip",
    "finalize_clip",
    "finalize_output",
    "initialize_clip",
    "initialize_input",
    "sc_detect",
    "shift_clip",
    "shift_clip_multi",
]


class ProcessVariableClip[T](DynamicClipsCache[T]):
    """
    A helper class for processing variable format/resolution clip.
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        out_dim: tuple[int, int] | Literal[False] | None = None,
        out_fmt: int | vs.VideoFormat | Literal[False] | None = None,
        cache_size: int = 10,
    ) -> None:
        """
        Initializes the class.

        Args:
            clip: Clip to process
            out_dim: Ouput dimension.
            out_fmt: Output format.
            cache_size: The maximum number of items allowed in the cache. Defaults to 10.
        """
        bk_args = {"length": clip.num_frames, "keep": True, "varformat": None}

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

    def eval_clip(self) -> vs.VideoNode:
        if self.out.format and (0 not in (self.out.width, self.out.height)):
            try:
                return self.get_clip(self.get_key(self.clip))
            except Exception:
                ...

        return vs.core.std.FrameEval(self.out, lambda n, f: self[self.get_key(f)], self.clip)

    def get_clip(self, key: T) -> vs.VideoNode:
        return self.process(self.normalize(self.clip, key))

    @classmethod
    def from_clip(cls, clip: vs.VideoNode) -> vs.VideoNode:
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
        clip: vs.VideoNode,
        func: Callable[[vs.VideoNode], vs.VideoNode],
        out_dim: tuple[int, int] | Literal[False] | None = None,
        out_fmt: int | vs.VideoFormat | Literal[False] | None = None,
        cache_size: int = 10,
    ) -> vs.VideoNode:
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

        def process(self: ProcessVariableClip[T], clip: vs.VideoNode) -> vs.VideoNode:
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
    def normalize(self, clip: vs.VideoNode, cast_to: T) -> vs.VideoNode:
        """
        Normalize the given node to the format/resolution specified by the unique key `cast_to`.

        Args:
            clip: Clip to normalize.
            cast_to: The target resolution or format to which the clip should be cast or normalized.

        Returns:
            Normalized clip.
        """

    def process(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Process the given clip.

        Args:
            clip: Clip to process.

        Returns:
            Processed clip.
        """
        return clip


class ProcessVariableResClip(ProcessVariableClip[tuple[int, int]]):
    """
    A helper class for processing variable resolution clip.
    """

    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> tuple[int, int]:
        return (frame.width, frame.height)

    def normalize(self, clip: vs.VideoNode, cast_to: tuple[int, int]) -> vs.VideoNode:
        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), *cast_to)
        return vs.core.std.CopyFrameProps(normalized, clip)


class ProcessVariableFormatClip(ProcessVariableClip[vs.VideoFormat]):
    """
    A helper class for processing variable format clip.
    """

    def get_key(self, frame: vs.VideoNode | vs.VideoFrame) -> vs.VideoFormat:
        assert frame.format
        return frame.format

    def normalize(self, clip: vs.VideoNode, cast_to: vs.VideoFormat) -> vs.VideoNode:
        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), format=cast_to.id)
        return vs.core.std.CopyFrameProps(normalized, clip)


class ProcessVariableResFormatClip(ProcessVariableClip[tuple[int, int, vs.VideoFormat]]):
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


def finalize_clip(
    clip: vs.VideoNode,
    bits: VideoFormatLike | HoldsVideoFormat | int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    *,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
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
    if bits:
        clip = depth(clip, bits, dither_type=dither_type)

    if clamp_tv_range:
        clip = limiter(clip, tv_range=clamp_tv_range)

    return clip


@overload
def finalize_output[**P](
    function: Callable[P, vs.VideoNode],
    /,
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExcept | None = None,
) -> Callable[P, vs.VideoNode]: ...


@overload
def finalize_output[**P](
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExcept | None = None,
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]: ...


def finalize_output[**P](
    function: Callable[P, vs.VideoNode] | None = None,
    /,
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExcept | None = None,
) -> Callable[P, vs.VideoNode] | Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]:
    """
    Decorator implementation of [finalize_clip][vstools.finalize_clip].
    """

    if function is None:
        return partial(finalize_output, bits=bits, clamp_tv_range=clamp_tv_range, dither_type=dither_type, func=func)

    @wraps(function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
        return finalize_clip(function(*args, **kwargs), bits, clamp_tv_range, dither_type, func=func)

    return _wrapper


def initialize_clip(
    clip: vs.VideoNode,
    bits: int | None = None,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    *,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
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

    values: list[tuple[type[PropEnum], Any]] = [
        (Matrix, matrix),
        (Transfer, transfer),
        (Primaries, primaries),
        (ChromaLocation, chroma_location),
        (ColorRange, color_range),
        (FieldBased, field_based),
    ]

    to_ensure_presence = list[type[PropEnum] | PropEnum]()

    for prop_t, prop_v in values:
        if strict:
            to_ensure_presence.append(prop_t)
        else:
            to_ensure_presence.append(prop_t.from_param_or_video(prop_v, clip, False, func))

    clip = PropEnum.ensure_presences(clip, to_ensure_presence, func)

    if bits is None:
        bits = max(get_depth(clip), 16)
    elif bits <= 0:
        return clip

    return depth(clip, bits, dither_type=dither_type)


@overload
def initialize_input[**P](
    function: Callable[P, vs.VideoNode],
    /,
    *,
    bits: int | None = 16,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExcept | None = None,
) -> Callable[P, vs.VideoNode]: ...


@overload
def initialize_input[**P](
    *,
    bits: int | None = 16,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExcept | None = None,
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]: ...


def initialize_input[**P](
    function: Callable[P, vs.VideoNode] | None = None,
    /,
    *,
    bits: int | None = 16,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: ColorRangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExcept | None = None,
) -> Callable[P, vs.VideoNode] | Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]:
    """
    Decorator implementation of [initialize_clip][vstools.initialize_clip].

    Initializes the first clip found in this order: positional arguments -> keyword arguments ->  default arguments.
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
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
        args_l = list(args)

        for i, obj in enumerate(args_l):
            if isinstance(obj, vs.VideoNode):
                args_l[i] = initialize_clip(obj, **init_args)
                return function(*args_l, **kwargs)  # type: ignore[arg-type]

        kwargs2 = kwargs.copy()

        for name, obj in kwargs2.items():
            if isinstance(obj, vs.VideoNode):
                kwargs2[name] = initialize_clip(obj, **init_args)
                return function(*args, **kwargs2)  # type: ignore[arg-type]

        for name, param in signature(function).parameters.items():
            if isinstance(param.default, vs.VideoNode):
                return function(*args, **{name: initialize_clip(param.default, **init_args)}, **kwargs)  # type: ignore[arg-type]

        raise CustomValueError(
            "No VideoNode found in positional, keyword, nor default arguments!", func or initialize_input
        )

    return _wrapper


def shift_clip(clip: vs.VideoNode, offset: int) -> vs.VideoNode:
    """
    Shift a clip forwards or backwards by N frames.

    This is useful for cases where you must compare every frame of a clip
    with the frame that comes before or after the current frame,
    like for example when performing temporal operations.

    Both positive and negative integers are allowed.
    Positive values will shift a clip forward, negative will shift a clip backward.

    Args:
        clip: Input clip.
        offset: Number of frames to offset the clip with. Negative values are allowed. Positive values will shift a clip
            forward, negative will shift a clip backward.

    Returns:
        Clip that has been shifted forwards or backwards by *N* frames.
    """

    if offset > clip.num_frames - 1:
        raise FramesLengthError(shift_clip, "offset")

    if offset < 0:
        return clip[0] * abs(offset) + clip[:offset]

    if offset > 0:
        return clip[offset:] + clip[-1] * offset

    return clip


def shift_clip_multi(clip: vs.VideoNode, offsets: StrictRange = (-1, 1)) -> list[vs.VideoNode]:
    """
    Shift a clip forwards or backwards multiple times by a varying amount of frames.

    This will return a clip for every shifting operation performed.
    This is a convenience function that makes handling multiple shifts easier.

    Example:

        >>> shift_clip_multi(clip, (-3, 3))
            [VideoNode, VideoNode, VideoNode, VideoNode, VideoNode, VideoNode, VideoNode]
                -3         -2         -1          0         +1         +2         +3

    Args:
        clip: Input clip.
        offsets: Tuple of offsets representing an inclusive range.
            A clip will be returned for every offset. Default: (-1, 1).

    Returns:
        A list of clips, the amount determined by the amount of offsets.
    """
    return [shift_clip(clip, x) for x in range(offsets[0], offsets[1] + 1)]


def sc_detect(clip: vs.VideoNode, threshold: float = 0.1) -> vs.VideoNode:
    """
    Detects scene changes in a video clip based on frame difference statistics.

    Args:
        clip: The input clip.
        threshold: Sensitivity for scene change detection. Higher values make detection less sensitive. Default is 0.1.

    Returns:
        vs.VideoNode: A clip with scene change props (`_SceneChangePrev` and `_SceneChangeNext`) set for each frame.
    """
    stats = vs.core.std.PlaneStats(shift_clip(clip, -1), clip)

    return vs.core.akarin.PropExpr(
        [clip, stats, stats[1:]],
        lambda: {
            "_SceneChangePrev": f"y.PlaneStatsDiff {threshold} > 1 0 ?",
            "_SceneChangeNext": f"z.PlaneStatsDiff {threshold} > 1 0 ?",
        },
    )
