from __future__ import annotations

from typing import Callable, Sequence, Union, overload

import vapoursynth as vs
from jetpytools import CustomValueError, flatten, interleave_arr, ranges_product

from ..functions import check_ref_clip
from ..types import ConstantFormatVideoNode, FrameRangeN, FrameRangesN

__all__ = [
    'replace_ranges',

    'remap_frames',

    'replace_every',

    'ranges_product',

    'interleave_arr',
]


_gc_func_gigacope = []

RangesCallback = Union[
    Callable[[int], bool],
    Callable[[vs.VideoFrame], bool],
    Callable[[list[vs.VideoFrame]], bool],
    Callable[[vs.VideoFrame | list[vs.VideoFrame]], bool],
    Callable[[int, vs.VideoFrame], bool],
    Callable[[int, list[vs.VideoFrame]], bool],
    Callable[[int, vs.VideoFrame | list[vs.VideoFrame]], bool]
]


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | Callable[[vs.VideoFrame], bool] | Callable[[int, vs.VideoFrame], bool] | None,
    exclusive: bool = False, mismatch: bool = False,
    *, prop_src: vs.VideoNode
) -> vs.VideoNode:
    ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: FrameRangeN | FrameRangesN | Callable[
        [list[vs.VideoFrame]], bool
    ] | Callable[[int, list[vs.VideoFrame]], bool] | None, exclusive: bool = False,
    mismatch: bool = False,
    *, prop_src: list[vs.VideoNode]
) -> vs.VideoNode:
    ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: FrameRangeN | FrameRangesN | Callable[
        [vs.VideoFrame | list[vs.VideoFrame]], bool
    ] | Callable[[int, vs.VideoFrame | list[vs.VideoFrame]], bool] | None, exclusive: bool = False,
    mismatch: bool = False, *, prop_src: vs.VideoNode | list[vs.VideoNode] | None = None
) -> vs.VideoNode:
    ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | Callable[[int], bool] | None,
    exclusive: bool = False, mismatch: bool = False, *, prop_src: None = None
) -> vs.VideoNode:
    ...


@overload
def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | RangesCallback | None,
    exclusive: bool = False, mismatch: bool = False,
    *, prop_src: vs.VideoNode | list[vs.VideoNode] | None = None
) -> vs.VideoNode:
    ...


def replace_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode,
    ranges: FrameRangeN | FrameRangesN | RangesCallback | None,
    exclusive: bool = False, mismatch: bool = False,
    *, prop_src: vs.VideoNode | list[vs.VideoNode] | None = None
) -> vs.VideoNode:
    """
    Replaces frames in a clip, either with pre-calculated indices or on-the-fly with a callback.
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

    Example of using a callback function:
        * ``replace_ranges(clip_a, clip_b, lambda f: get_prop(f, '_PictType', str) == 'P', prop_src=clip_a)``:
          Replace frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P.

    Optional Dependencies:
        * `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`_ (highly recommended!)

    :param clip_a:      Original clip.
    :param clip_b:      Replacement clip.
    :param ranges:      Ranges to replace clip_a (original clip) with clip_b (replacement clip).
                        Integer values in the list indicate single frames,
                        Tuple values indicate inclusive ranges.
                        Callbacks must return true to replace a with b.
                        Negative integer values will be wrapped around based on clip_b's length.
                        None values are context dependent:
                            * None provided as sole value to ranges: no-op
                            * Single None value in list: Last frame in clip_b
                            * None as first value of tuple: 0
                            * None as second value of tuple: Last frame in clip_b
    :param exclusive:   Use exclusive ranges (Default: False).
    :param mismatch:    Accept format or resolution mismatch between clips.
    :param prop_src:    Source clip(s) to use for frame properties in the callback.
                        This is required if you're using a callback.

    :return:            Clip with ranges from clip_a replaced with clip_b.
    """

    from ..functions import invert_ranges, normalize_ranges

    if ranges != 0 and not ranges or clip_a is clip_b:
        return clip_a

    if not mismatch:
        check_ref_clip(clip_a, clip_b)

    if callable(ranges):
        from inspect import Signature

        signature = Signature.from_callable(ranges, eval_str=True)

        params = set(signature.parameters.keys())

        base_clip = clip_a.std.BlankClip(
            keep=True, varformat=(clip_a.format != clip_b.format),
            varsize=(clip_a.width, clip_a.height) != (clip_b.width, clip_b.height)
        )

        callback: RangesCallback = ranges

        if 'f' in params and not prop_src:
            raise CustomValueError(
                'To use frame properties in the callback (parameter "f"), '
                'you must specify one or more source clips via `prop_src`!',
                replace_ranges
            )

        if 'f' in params and 'n' in params:
            def _func(n: int, f: vs.VideoFrame) -> vs.VideoNode:
                return clip_b if callback(n, f) else clip_a  # type: ignore
        elif 'f' in params:
            def _func(n: int, f: vs.VideoFrame) -> vs.VideoNode:
                return clip_b if callback(f) else clip_a  # type: ignore
        elif 'n' in params:
            def _func(n: int) -> vs.VideoNode:  # type: ignore
                return clip_b if callback(n) else clip_a  # type: ignore
        else:
            raise CustomValueError('Callback must have signature ((n, f) | (n) | (f)) -> bool!')

        _func.__callback = callback  # type: ignore
        _gc_func_gigacope.append(_func)

        return base_clip.std.FrameEval(_func, prop_src if 'f' in params else None, [clip_a, clip_b])

    shift = 1 - exclusive
    b_ranges = normalize_ranges(clip_b, ranges)

    if hasattr(vs.core, 'vszip'):
        return vs.core.vszip.RFS(
            clip_a, clip_b,
            [y for (s, e) in b_ranges
             for y in range(
                 s, e + (not exclusive if s != e else 1) + (1 if e == clip_b.num_frames - 1 and exclusive else 0)
             )
            ],
            mismatch=mismatch
        )

    a_ranges = invert_ranges(clip_a, clip_b, b_ranges)

    a_trims = [clip_a[max(0, start - exclusive):end + shift + exclusive] for start, end in a_ranges]
    b_trims = [clip_b[start:end + shift] for start, end in b_ranges]

    if a_ranges:
        main, other = (a_trims, b_trims) if (a_ranges[0][0] == 0) else (b_trims, a_trims)
    else:
        main, other = (b_trims, a_trims) if (b_ranges[0][0] == 0) else (a_trims, b_trims)

    return vs.core.std.Splice(list(interleave_arr(main, other, 1)), mismatch)


def remap_frames(clip: vs.VideoNode, ranges: Sequence[int | tuple[int, int]]) -> ConstantFormatVideoNode:
    frame_map = list[int](flatten(
        f if isinstance(f, int) else range(f[0], f[1] + 1) for f in ranges
    ))

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
