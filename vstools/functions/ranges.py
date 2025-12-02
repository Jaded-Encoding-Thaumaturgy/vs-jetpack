from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from typing import Any, Literal, Protocol, TypeGuard, TypeVar, overload

from jetpytools import CustomValueError, SoftRange, StrictRange, fallback, flatten, interleave_arr
from jetpytools import invert_ranges as jetp_invert_ranges
from jetpytools import normalize_list_to_ranges as jetp_normalize_list_to_ranges
from jetpytools import normalize_range as jetp_normalize_range
from jetpytools import normalize_ranges as jetp_normalize_ranges
from jetpytools import normalize_ranges_to_list as jetp_normalize_ranges_to_list

from ..types import FrameRangeN, FrameRangesN, Planes
from ..utils import check_ref_clip
from ..vs_proxy import vs

__all__ = [
    "invert_ranges",
    "normalize_franges",
    "normalize_list_to_ranges",
    "normalize_ranges",
    "normalize_ranges_to_list",
    "remap_frames",
    "replace_every",
    "replace_ranges",
]


def normalize_franges(ranges: SoftRange, /, exclusive: bool | None = None) -> Sequence[int]:
    """
    Normalize ranges represented by a tuple to an iterable of frame numbers.

    Args:
        ranges: Ranges to normalize.
        exclusive: Whether to use exclusive (Python-style) ranges.

    Returns:
        Sequence of positive frame ranges.
    """
    return jetp_normalize_range(ranges, fallback(exclusive, replace_ranges.exclusive, False))


def normalize_list_to_ranges(
    flist: Iterable[int], min_length: int = 0, exclusive: bool | None = None
) -> list[StrictRange]:
    return jetp_normalize_list_to_ranges(flist, min_length, fallback(exclusive, replace_ranges.exclusive, False))


def normalize_ranges(
    clip: vs.VideoNode, ranges: FrameRangeN | FrameRangesN, exclusive: bool | None = None
) -> list[tuple[int, int]]:
    """
    Normalize ranges to a list of positive ranges.

    - Frame ranges can include `None` and negative values.
    - None will be converted to either 0 if it's the first value in a FrameRange, or the clip's length
     if it's the second item.
    - Negative values will be subtracted from the clip's length.

    Examples:
        ```py
        clip.num_frames
        # 1000
        normalize_ranges(clip, (None, None))
        # [(0, 999)]
        normalize_ranges(clip, (24, -24))
        # [(24, 975)]
        normalize_ranges(clip, [(24, 100), (80, 150)])
        # [(24, 150)]
        ```

    Args:
        clip: Input clip.
        ranges: Frame range or list of frame ranges.
        exclusive: Whether to use exclusive (Python-style) ranges.

    Returns:
        List of positive frame ranges.
    """
    return jetp_normalize_ranges(ranges, clip.num_frames, fallback(exclusive, replace_ranges.exclusive, False))


def normalize_ranges_to_list(ranges: Iterable[SoftRange], exclusive: bool | None = None) -> list[int]:
    return jetp_normalize_ranges_to_list(ranges, fallback(exclusive, replace_ranges.exclusive, False))


def invert_ranges(
    clipa: vs.VideoNode, clipb: vs.VideoNode | None, ranges: FrameRangeN | FrameRangesN, exclusive: bool | None = None
) -> list[tuple[int, int]]:
    """
    Invert FrameRanges.

    Example:
        ```py
        franges = [(100, 200), 600, (1200, 2400)]
        invert_ranges(core.std.BlankClip(length=10000), core.std.BlankClip(length=10000), franges)
        # [(0, 99), (201, 599), (601, 1199), (2401, 9999)]
        ```

    Args:
        clipa: Original clip.
        clipb: Replacement clip.
        ranges: Ranges to replace clipa (original clip) with clipb (replacement clip). These ranges will be inverted.
            For more info, see `replace_ranges`.
        exclusive: Whether to use exclusive (Python-style) ranges.

    Returns:
        A list of inverted frame ranges.
    """
    return jetp_invert_ranges(
        ranges,
        clipa.num_frames,
        None if clipb is None else clipb.num_frames,
        fallback(exclusive, replace_ranges.exclusive, False),
    )


_VideoFrameT_contra = TypeVar(
    "_VideoFrameT_contra",
    vs.VideoFrame,
    Sequence[vs.VideoFrame],
    vs.VideoFrame | Sequence[vs.VideoFrame],
    contravariant=True,
)


class _RangesCallBack(Protocol):
    def __call__(self, n: int, /) -> bool: ...


class _RangesCallBackF(Protocol[_VideoFrameT_contra]):
    def __call__(self, f: _VideoFrameT_contra, /) -> bool: ...


class _RangesCallBackNF(Protocol[_VideoFrameT_contra]):
    def __call__(self, n: int, f: _VideoFrameT_contra, /) -> bool: ...


_RangesCallBackLike = (
    _RangesCallBack
    | _RangesCallBackF[vs.VideoFrame]
    | _RangesCallBackNF[vs.VideoFrame]
    | _RangesCallBackF[Sequence[vs.VideoFrame]]
    | _RangesCallBackNF[Sequence[vs.VideoFrame]]
)


def _is_cb_nf(
    cb: Callable[..., bool], params: set[str]
) -> TypeGuard[_RangesCallBackNF[vs.VideoFrame | Sequence[vs.VideoFrame]]]:
    return "f" in params and "n" in params


def _is_cb_f(
    cb: Callable[..., bool], params: set[str]
) -> TypeGuard[_RangesCallBackF[vs.VideoFrame | Sequence[vs.VideoFrame]]]:
    return "f" in params


def _is_cb_n(cb: Callable[..., bool], params: set[str]) -> TypeGuard[_RangesCallBack]:
    return "n" in params


class ReplaceRanges[**P, R]:
    """
    Class decorator that wraps the [replace_ranges][vstools.replace_ranges] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    exclusive: bool | None
    """
    Whether to use exclusive ranges globally (Default: None).

    If set to True, all calls of replace_ranges will use exclusive ranges.
    If set to False, all calls of replace_ranges will use inclusive ranges.
    """

    def __init__(self, replace_ranges: Callable[P, R]) -> None:
        self._func = replace_ranges
        self.exclusive = None

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN,
        exclusive: bool | None = None,
        mismatch: Literal[False] = ...,
        planes: Planes = None,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self, clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: _RangesCallBack, *, mismatch: bool = False
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBackF[vs.VideoFrame] | _RangesCallBackNF[vs.VideoFrame],
        *,
        mismatch: bool = False,
        prop_src: vs.VideoNode,
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBackF[Sequence[vs.VideoFrame]] | _RangesCallBackNF[Sequence[vs.VideoFrame]],
        *,
        mismatch: bool = False,
        prop_src: Sequence[vs.VideoNode],
    ) -> vs.VideoNode: ...

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN | _RangesCallBackLike | None,
        exclusive: bool | None = None,
        mismatch: bool = False,
        planes: Planes = None,
        *,
        prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None,
    ) -> vs.VideoNode: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)


@ReplaceRanges
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | _RangesCallBackLike | None,
    exclusive: bool | None = None,
    mismatch: bool = False,
    planes: Planes = None,
    *,
    prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None,
) -> vs.VideoNode:
    """
    Replaces frames in a clip, either with pre-calculated indices or on-the-fly with a callback.

    Frame ranges are inclusive by default.
    This behaviour can be changed by setting `exclusive=True` for one-time use,
    or set `replace_ranges.exclusive = True` to apply the change globally.

    Examples with clips ``black`` and ``white`` of equal length:
        ``` py
        # Replaces frames 0 and 1 with ``white``
        replace_ranges(black, white, [(0, 1)])

        # Replaces the entire clip with ``white``
        replace_ranges(black, white, [(None, None)])

        # Same as previous
        replace_ranges(black, white, [(0, None)])

        # Replaces 200 until the end with ``white``
        replace_ranges(black, white, [(200, None)])

        # Replaces 200 until the end with ``white``, leaving 1 frame of ``black``
        replace_ranges(black, white, [(200, -1)])
        ```

    A callback function can be used to replace frames based on frame properties or frame numbers.
    The function must return a boolean value.

    Example of using a callback function:
        ```py
        # Replaces frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P.
        replace_ranges(clip_a, clip_b, lambda f: get_prop(f, '_PictType', str) == 'P', prop_src=clip_a)``
        ```

    Optional Dependencies:
        - [vs-zip](https://github.com/dnjulek/vapoursynth-zip) (highly recommended!)

    Args:
        clip_a: Original clip.
        clip_b: Replacement clip.
        ranges:
            Ranges to replace clip_a (original clip) with clip_b (replacement clip).

            Integer values in the list indicate single frames, tuple values indicate inclusive ranges.
            Callbacks must return true to replace a with b.
            Negative integer values will be wrapped around based on clip_b's length.
            None values are context dependent:

                * None provided as sole value to ranges: no-op
                * Single None value in list: Last frame in clip_b
                * None as first value of tuple: 0
                * None as second value of tuple: Last frame in clip_b

        exclusive: Force the use of exclusive (Python-style) ranges.
        mismatch: Accept format or resolution mismatch between clips.
        planes: Which planes to process. Only available when `vs-zip` is installed.
        prop_src: Source clip(s) to use for frame properties in the callback.
            This is required if you're using a callback.

    Raises:
        CustomValueError: If ``prop_src`` isn't specified and a callback needs it.
        CustomValueError: If a wrong callback signature is provided.
        CustomValueError: If ``planes`` is specified and `vs-zip` is not installed..

    Returns:
        Clip with ranges from clip_a replaced with clip_b.
    """
    if (ranges != 0 and not ranges) or clip_a is clip_b:
        return clip_a

    if not mismatch:
        check_ref_clip(clip_a, clip_b, replace_ranges)

    if callable(ranges):
        from inspect import Signature

        signature = Signature.from_callable(ranges, eval_str=True)

        params = set(signature.parameters.keys())

        base_clip = clip_a.std.BlankClip(keep=True, varformat=mismatch, varsize=mismatch)

        callback = ranges

        if "f" in params and not prop_src:
            raise CustomValueError(
                'To use frame properties in the callback (parameter "f"), '
                "you must specify one or more source clips via `prop_src`!",
                replace_ranges,
            )

        if _is_cb_nf(callback, params):
            return vs.core.std.FrameEval(
                base_clip,
                lambda n, f: clip_b if callback(n, f) else clip_a,
                prop_src,
                [clip_a, clip_b],
            )
        if _is_cb_f(callback, params):
            return vs.core.std.FrameEval(
                base_clip,
                lambda n, f: clip_b if callback(f) else clip_a,
                prop_src,
                [clip_a, clip_b],
            )
        if _is_cb_n(callback, params):
            return vs.core.std.FrameEval(base_clip, lambda n: clip_b if callback(n) else clip_a, None, [clip_a, clip_b])

        raise CustomValueError("Callback must have signature ((n, f) | (n) | (f)) -> bool!", replace_ranges, callback)

    exclusive = fallback(exclusive, replace_ranges.exclusive, False)

    b_ranges = normalize_ranges(clip_b, ranges, exclusive)

    with suppress(AttributeError):
        return vs.core.vszip.RFS(
            clip_a, clip_b, [y for (s, e) in b_ranges for y in range(s, e + (not exclusive))], mismatch, planes
        )

    if planes is not None:
        raise CustomValueError(
            "The `planes` argument is only available when vszip is installed.", replace_ranges, planes
        )

    a_ranges = invert_ranges(clip_a, clip_b, b_ranges, exclusive)

    a_trims = [clip_a[max(0, start - exclusive) : end + (not exclusive)] for start, end in a_ranges]
    b_trims = [clip_b[start : end + (not exclusive)] for start, end in b_ranges]

    if a_ranges:
        main, other = (a_trims, b_trims) if (a_ranges[0][0] == 0) else (b_trims, a_trims)
    else:
        main, other = (b_trims, a_trims) if (b_ranges[0][0] == 0) else (a_trims, b_trims)

    return vs.core.std.Splice(list(interleave_arr(main, other, 1)), mismatch)


def remap_frames(clip: vs.VideoNode, ranges: Sequence[int | tuple[int, int]]) -> vs.VideoNode:
    """
    Remap frames of a clip according to specified ranges.

    This function creates a new clip where frames are reordered or repeated based on the given ranges.
    Ranges can be specified as single frame indices or as ranges of frames.

    Example:
        Remap frames 0, 5, and frames 10 through 15
        ```
        new_clip = remap_frames(clip, [0, 5, (10, 15)])
        ```

        This will produce a new clip containing frames:
        `[0, 5, 10, 11, 12, 13, 14, 15]` from the original.

    Args:
        clip: The source clip to remap frames from.
        ranges: A sequence of frame indices or tuples representing ranges.

               - If an element is an integer, that frame is included.
               - If an element is a tuple ``(start, end)``, all frames from
                ``start`` through ``end`` (inclusive, the default, or exclusive through `replace_ranges.exclusive`)
                are included.

    Returns:
        A new clip with frames reordered according to the given ranges.
    """
    frame_map = list[int](
        flatten(f if isinstance(f, int) else range(f[0], f[1] + (not replace_ranges.exclusive)) for f in ranges)
    )

    base = vs.core.std.BlankClip(clip, length=len(frame_map))

    return vs.core.std.FrameEval(base, lambda n: clip[frame_map[n]], None, clip)


def replace_every(
    clipa: vs.VideoNode, clipb: vs.VideoNode, cycle: int, offsets: Sequence[int], modify_duration: bool = True
) -> vs.VideoNode:
    """
    Replace frames in one clip with frames from another at regular intervals.

    This function interleaves two clips and then selects frames so that, within each cycle, frames from `clipa`
    are replaced with frames from `clipb` at the specified offsets.

    Example:
        Replace every 3rd frame with a frame from another clip:
        ```
        new_clip = replace_every(clipa, clipb, cycle=3, offsets=[2])
        ```

        In this example, within every group of 3 frames:

        - Frame 0 and 1 come from `clipa`.
        - Frame 2 comes from `clipb`.

    Args:
        clipa: The base clip to use as the primary source.
        clipb: The replacement clip to take frames from.
        cycle: The size of the repeating cycle in frames.
        offsets: The positions within each cycle where frames from `clipb` should replace frames from `clipa`.
        modify_duration: Whether to adjust the clip's duration to reflect inserted frames.

    Returns:
        A new clip where frames from `clipb` replace frames in `clipa` at the specified offsets.
    """
    offsets_a = [x * 2 for x in range(cycle) if x not in offsets]
    offsets_b = [x * 2 + 1 for x in offsets]
    offsets = sorted(offsets_a + offsets_b)

    interleaved = vs.core.std.Interleave([clipa, clipb])

    return vs.core.std.SelectEvery(interleaved, cycle * 2, offsets, modify_duration)
