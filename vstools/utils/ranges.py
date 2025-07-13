from __future__ import annotations

import contextlib
from typing import Callable, Literal, Protocol, Sequence, TypeGuard, TypeVar, Union, overload

import vapoursynth as vs
from jetpytools import CustomValueError, flatten, interleave_arr, ranges_product

from ..functions import check_ref_clip
from ..types import ConstantFormatVideoNode, FrameRangeN, FrameRangesN, VideoNodeT

__all__ = [
    "interleave_arr",
    "ranges_product",
    "remap_frames",
    "replace_every",
    "replace_ranges",
]


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


_RangesCallBackLike = Union[
    _RangesCallBack,
    _RangesCallBackF[vs.VideoFrame],
    _RangesCallBackNF[vs.VideoFrame],
    _RangesCallBackF[Sequence[vs.VideoFrame]],
    _RangesCallBackNF[Sequence[vs.VideoFrame]],
]


def _is_cb_nf(
    cb: Callable[..., bool], params: set[str]
) -> TypeGuard[_RangesCallBackNF[vs.VideoFrame] | _RangesCallBackNF[Sequence[vs.VideoFrame]]]:
    return bool("f" in params and "n" in params)


def _is_cb_f(
    cb: Callable[..., bool], params: set[str]
) -> TypeGuard[_RangesCallBackF[vs.VideoFrame] | _RangesCallBackF[Sequence[vs.VideoFrame]]]:
    return "f" in params


def _is_cb_n(cb: Callable[..., bool], params: set[str]) -> TypeGuard[_RangesCallBack]:
    return "n" in params


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN,
    exclusive: bool = False,
    mismatch: Literal[False] = ...,
) -> ConstantFormatVideoNode: ...


@overload
def replace_ranges(
    clip_a: VideoNodeT,
    clip_b: VideoNodeT,
    ranges: FrameRangeN | FrameRangesN,
    exclusive: bool = False,
    mismatch: bool = ...,
) -> VideoNodeT: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: _RangesCallBack, *, mismatch: bool = False
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: _RangesCallBackF[vs.VideoFrame] | _RangesCallBackNF[vs.VideoFrame],
    *,
    mismatch: bool = False,
    prop_src: vs.VideoNode,
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: _RangesCallBackF[Sequence[vs.VideoFrame]] | _RangesCallBackNF[Sequence[vs.VideoFrame]],
    *,
    mismatch: bool = False,
    prop_src: Sequence[vs.VideoNode],
) -> vs.VideoNode: ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | _RangesCallBackLike | None,
    exclusive: bool = False,
    mismatch: bool = False,
    *,
    prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None,
) -> vs.VideoNode: ...


def replace_ranges(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | _RangesCallBackLike | None,
    exclusive: bool = False,
    mismatch: bool = False,
    *,
    prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None,
) -> vs.VideoNode:
    """
    Replaces frames in a clip with pre-calculated indices or on-the-fly with a callback.

    Frame ranges are inclusive. This behaviour can be changed by setting `exclusive=True`.

    Examples with clips ``black`` and ``white`` of equal length:

    * ``replace_ranges(black, white, [(0, 1)])``: replace frames 0 and 1 with ``white``
    * ``replace_ranges(black, white, [(None, None)])``: replace the entire clip with ``white``
    * ``replace_ranges(black, white, [(0, None)])``: same as previous
    * ``replace_ranges(black, white, [(200, None)])``: replace 200 until the end with ``white``
    * ``replace_ranges(black, white, [(200, -1)])``: replace 200 until the end with ``white``,
                                                        leaving 1 frame of ``black``

    A callback function can be used to replace frames based on frame properties.
    The function must return a boolean value.

    Examples with a callback:

    ```py
    # Replace frames from `clip_a` with `clip_b` if the frame number is between 200 and 299 (inclusive).
    clip = replace_ranges(clip_a, clip_b, lambda n: n in range(200, 300))
    ```

    ```py
    # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P.
    clip = replace_ranges(clip_a, clip_b, lambda f: get_prop(f, "_PictType", str) == "P", prop_src=clip_a)

    # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P
    # and if the frame number is between 200 and 299 (inclusive)
    clip = replace_ranges(
        clip_a, clip_b, lambda n, f: get_prop(f, "_PictType", str) == "P" and n in range(200, 300), prop_src=clip_a
    )
    ```

    ```py
    prop_srcs: list[vs.VideNode]()

    # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of all the ``prop_srcs`` is P.
    clip = replace_ranges(
        clip_a, clip_b, lambda f: all(get_prop(frame, "_PictType", str) == "P" for frame in f), prop_src=prop_srcs
    )

    # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of all the ``prop_srcs`` is P
    # and if the frame number is between 200 and 299 (inclusive)
    clip = replace_ranges(
        clip_a,
        clip_b,
        lambda n, f: all(get_prop(frame, "_PictType", str) == "P" for frame in f) and n in range(200, 300),
        prop_src=prop_srcs,
    )
    ```

    Optional dependency:
        [vs-zip](https://github.com/dnjulek/vapoursynth-zip) (highly recommended!)

    Args:
        clip_a: Original clip.
        clip_b: Replacement clip.
        ranges: Ranges to replace clip_a (original clip) with clip_b (replacement clip). Integer values in the list
            indicate single frames, Tuple values indicate inclusive ranges. Negative integer values will be wrapped
            around based on clip_b's length. None values are context dependent:

               * None provided as sole value to ranges: no-op
               * Single None value in list: Last frame in clip_b
               * None as first value of tuple: 0
               * None as second value of tuple: Last frame in clip_b

            Can also be a callback to replace clip_a (original clip) with clip_b (replacement clip).
            Must return True to replace a with b.

        exclusive: Use exclusive ranges (Default: False).
        mismatch: Accept format or resolution mismatch between clips.
        prop_src: Source clip(s) to use for frame properties in the callback. This is required if you're using a
            callback.

    Returns:
        Clip with ranges from clip_a replaced with clip_b.
    """
    from ..functions import invert_ranges, normalize_ranges

    if (ranges != 0 and not ranges) or clip_a is clip_b:
        return clip_a

    if not mismatch:
        check_ref_clip(clip_a, clip_b)

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
                base_clip, lambda n, f: clip_b if callback(n, f) else clip_a, prop_src, [clip_a, clip_b]
            )
        if _is_cb_f(callback, params):
            return vs.core.std.FrameEval(
                base_clip, lambda n, f: clip_b if callback(f) else clip_a, prop_src, [clip_a, clip_b]
            )
        if _is_cb_n(callback, params):
            return vs.core.std.FrameEval(base_clip, lambda n: clip_b if callback(n) else clip_a, None, [clip_a, clip_b])

        raise CustomValueError("Callback must have signature ((n, f) | (n) | (f)) -> bool!", replace_ranges, callback)

    shift = 1 - exclusive
    b_ranges = normalize_ranges(clip_b, ranges)

    with contextlib.suppress(AttributeError):
        return vs.core.vszip.RFS(
            clip_a,
            clip_b,
            [
                y
                for (s, e) in b_ranges
                for y in range(
                    s, e + (not exclusive if s != e else 1) + (1 if e == clip_b.num_frames - 1 and exclusive else 0)
                )
            ],
            mismatch=mismatch,
        )

    a_ranges = invert_ranges(clip_a, clip_b, b_ranges)

    a_trims = [clip_a[max(0, start - exclusive) : end + shift + exclusive] for start, end in a_ranges]
    b_trims = [clip_b[start : end + shift] for start, end in b_ranges]

    if a_ranges:
        main, other = (a_trims, b_trims) if (a_ranges[0][0] == 0) else (b_trims, a_trims)
    else:
        main, other = (b_trims, a_trims) if (b_ranges[0][0] == 0) else (a_trims, b_trims)

    return vs.core.std.Splice(list(interleave_arr(main, other, 1)), mismatch)


def remap_frames(clip: vs.VideoNode, ranges: Sequence[int | tuple[int, int]]) -> ConstantFormatVideoNode:
    frame_map = list[int](flatten(f if isinstance(f, int) else range(f[0], f[1] + 1) for f in ranges))

    base = vs.core.std.BlankClip(clip, length=len(frame_map))

    return vs.core.std.FrameEval(base, lambda n: clip[frame_map[n]], None, clip)


def replace_every(
    clipa: vs.VideoNode, clipb: vs.VideoNode, cycle: int, offsets: Sequence[int], modify_duration: bool = True
) -> ConstantFormatVideoNode:
    offsets_a = [x * 2 for x in range(cycle) if x not in offsets]
    offsets_b = [x * 2 + 1 for x in offsets]
    offsets = sorted(offsets_a + offsets_b)

    interleaved = vs.core.std.Interleave([clipa, clipb])

    return vs.core.std.SelectEvery(interleaved, cycle * 2, offsets, modify_duration)
