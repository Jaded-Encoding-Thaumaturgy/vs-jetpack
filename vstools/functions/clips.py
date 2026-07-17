from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any, Literal, Never, SupportsInt, cast, overload

from jetpytools import MISSING, CustomValueError, FuncExcept, StrictRange

from ..enums import (
    ChromaLocation,
    ChromaLocationLike,
    FieldBased,
    FieldBasedLike,
    Matrix,
    MatrixLike,
    Primaries,
    PrimariesLike,
    PropEnum,
    Range,
    RangeLike,
    Transfer,
    TransferLike,
)
from ..exceptions import FramesLengthError
from ..types import HoldsVideoFormat, VideoFormatLike
from ..utils import DynamicClipsCache
from ..vs_proxy import vs
from .utils import depth, limiter

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
        out_fmt: SupportsInt | Literal[False] | None = None,
        cache_size: int = 10,
    ) -> None:
        """
        Initializes the class.

        Args:
            clip: Clip to process
            out_dim: Output dimension.
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
            bk_args.update(format=int(out_fmt))

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
            out_dim: Output dimension.
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
        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), format=cast_to)
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

        normalized = vs.core.resize.Point(vs.core.std.RemoveFrameProps(clip), w, h, fmt)

        return vs.core.std.CopyFrameProps(normalized, clip)


def finalize_clip(
    clip: vs.VideoNode,
    bits: VideoFormatLike | HoldsVideoFormat | int | None = 10,
    clamp_tv_range: bool = False,
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Finalize a clip for output to the encoder.

    Args:
        clip: Clip to output.
        bits: Bitdepth to output to.
        clamp_tv_range: Whether to clamp to tv range.
        func: Function returned for custom error handling. This should only be set by VS package developers.
        **kwargs: Additional arguments passed to [depth][vstools.utils.depth].

    Returns:
        Converted and optionally clamped clip.
    """
    if bits:
        clip = depth(clip, bits, **kwargs)

    if clip.format.sample_type is vs.FLOAT or clip.format.bits_per_sample > 16:
        raise CustomValueError("Unsuitable output format!", func, clip.format)

    return limiter(clip, tv_range=clamp_tv_range, func=func)


@overload
def finalize_output[**P](
    function: Callable[P, vs.VideoNode],
    /,
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> Callable[P, vs.VideoNode]: ...


@overload
def finalize_output[**P](
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]: ...


def finalize_output[**P](
    function: Callable[P, vs.VideoNode] | None = None,
    /,
    *,
    bits: int | None = 10,
    clamp_tv_range: bool = False,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> Callable[P, vs.VideoNode] | Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]:
    """
    Decorator implementation of [finalize_clip][vstools.finalize_clip].
    """

    final_args = dict[str, Any](bits=bits, clamp_tv_range=clamp_tv_range, func=func, **kwargs)

    if function is None:
        return lambda function: finalize_output(function, **final_args)

    @wraps(function)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> vs.VideoNode:
        return finalize_clip(function(*args, **kwargs), **final_args)

    return _wrapper


def initialize_clip(
    clip: vs.VideoNode,
    bits: int | None = 32,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: RangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    strict: Never = cast(Never, MISSING),
    *,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Initialize a clip with core frame properties and convert its bit depth.

    This function ensures that key video metadata properties are explicitly set on the clip's frame properties.

    If any property is not explicitly provided, it will be resolved by:
    1. Checking the clip's existing frame properties.
    2. Failing that, applying heuristics based on the clip's resolution and format.

    It is HIGHLY recommended to always use this function at the beginning of your scripts!

    Args:
        clip: The input clip to initialize.
        bits: The target bit depth to convert the clip to. If set to `None`, no bit depth
            conversion will be performed. Defaults to 32.
        matrix: The color matrix of the clip. If `None`, it is inferred.
        transfer: The transfer characteristics. If `None`, it is inferred.
        primaries: The color primaries. If `None`, it is inferred.
        chroma_location: The chroma sample location. If `None`, it is inferred.
        color_range: The color range. If `None`, it is inferred.
        field_based: The field order type. If `None`, it is inferred.
        func: Function returned for custom error handling
        **kwargs: Additional keyword arguments forwarded to [depth][vstools.utils.depth]

    Returns:
        The initialized clip with all essential frame properties set and converted to the target bit depth.
    """
    func = func or initialize_clip

    if strict is not cast(Never, MISSING):
        warnings.warn("The 'strict' argument has been removed and is deprecated.", RuntimeWarning)

    user_props: dict[type[PropEnum], Any] = {
        Matrix: matrix,
        Transfer: transfer,
        Primaries: primaries,
        ChromaLocation: chroma_location,
        Range: color_range,
        FieldBased: field_based,
    }

    clip = clip.std.SetFrameProps(
        **{p.prop_key: p.from_param_or_video(v, clip, func_except=func) for p, v in user_props.items()}
    )

    return depth(clip, bits, **kwargs)


@overload
def initialize_input[**P](
    function: Callable[P, vs.VideoNode],
    /,
    *,
    bits: int | None = 32,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: RangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> Callable[P, vs.VideoNode]: ...


@overload
def initialize_input[**P](
    *,
    bits: int | None = 32,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: RangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]: ...


def initialize_input[**P](
    function: Callable[P, vs.VideoNode] | None = None,
    /,
    *,
    bits: int | None = 32,
    matrix: MatrixLike | None = None,
    transfer: TransferLike | None = None,
    primaries: PrimariesLike | None = None,
    chroma_location: ChromaLocationLike | None = None,
    color_range: RangeLike | None = None,
    field_based: FieldBasedLike | None = None,
    strict: Never = cast(Never, MISSING),
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> Callable[P, vs.VideoNode] | Callable[[Callable[P, vs.VideoNode]], Callable[P, vs.VideoNode]]:
    """
    Decorator implementation of [initialize_clip][vstools.initialize_clip].

    Initializes the first clip found in this order: positional arguments -> keyword arguments -> default arguments.
    """

    init_args = dict[str, Any](
        bits=bits,
        matrix=matrix,
        transfer=transfer,
        primaries=primaries,
        chroma_location=chroma_location,
        color_range=color_range,
        field_based=field_based,
        strict=strict,
        func=func,
        **kwargs,
    )

    if function is None:
        return lambda function: initialize_input(function, **init_args)

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
