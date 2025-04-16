from __future__ import annotations

from functools import partial
from typing import Any, Callable, Literal, Protocol, Sequence, TypeVar, Union, overload

import vapoursynth as vs

from jetpytools import CustomValueError, flatten, interleave_arr, ranges_product

from ..functions import check_ref_clip
from ..types import ConstantFormatVideoNode, FrameRangeN, FrameRangesN, VideoNodeT

__all__ = [
    'replace_ranges',

    'remap_frames',

    'replace_every',

    'ranges_product',

    'interleave_arr',
]


_gc_func_gigacope = list[Any]()

_VideoFrameT_contra = TypeVar(
    "_VideoFrameT_contra",
    vs.VideoFrame, Sequence[vs.VideoFrame], vs.VideoFrame | Sequence[vs.VideoFrame],
    contravariant=True
)


class _RangesCallBack(Protocol):
    def __call__(self, n: int, /) -> bool:
        ...


class _RangesCallBackF(Protocol[_VideoFrameT_contra]):
    def __call__(self, f: _VideoFrameT_contra, /) -> bool:
        ...


class _RangesCallBackNF(Protocol[_VideoFrameT_contra]):
    def __call__(self, n: int, f: _VideoFrameT_contra, /) -> bool:
        ...


_RangesCallBackT = Union[
    _RangesCallBack,
    _RangesCallBackF[vs.VideoFrame],
    _RangesCallBackNF[vs.VideoFrame],
    _RangesCallBackF[Sequence[vs.VideoFrame]],
    _RangesCallBackNF[Sequence[vs.VideoFrame]],
]


class _replace_ranges:
    exclusive: bool | None

    def __init__(self) -> None:
        self.exclusive = None

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN,
        exclusive: bool = False,
        mismatch: Literal[False] = ...
    ) -> ConstantFormatVideoNode:
        """
        Replaces frames in a clip with pre-calculated indices.
        Frame ranges are inclusive. This behaviour can be changed by setting `exclusive=True`.

        Examples with clips ``black`` and ``white`` of equal length:
            * ``replace_ranges(black, white, [(0, 1)])``: replace frames 0 and 1 with ``white``
            * ``replace_ranges(black, white, [(None, None)])``: replace the entire clip with ``white``
            * ``replace_ranges(black, white, [(0, None)])``: same as previous
            * ``replace_ranges(black, white, [(200, None)])``: replace 200 until the end with ``white``
            * ``replace_ranges(black, white, [(200, -1)])``: replace 200 until the end with ``white``,
                                                            leaving 1 frame of ``black``

        Optional Dependencies:
            * `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`_ (highly recommended!)

        :param clip_a:      Original clip.
        :param clip_b:      Replacement clip.
        :param ranges:      Ranges to replace clip_a (original clip) with clip_b (replacement clip).
                            Integer values in the list indicate single frames,
                            Tuple values indicate inclusive ranges.
                            Negative integer values will be wrapped around based on clip_b's length.
                            None values are context dependent:
                                * None provided as sole value to ranges: no-op
                                * Single None value in list: Last frame in clip_b
                                * None as first value of tuple: 0
                                * None as second value of tuple: Last frame in clip_b
        :param exclusive:   Use exclusive ranges (Default: False).
        :param mismatch:    Accept format or resolution mismatch between clips.

        :return:            Clip with ranges from clip_a replaced with clip_b.
        """

    @overload
    def __call__(
        self,
        clip_a: VideoNodeT,
        clip_b: VideoNodeT,
        ranges: FrameRangeN | FrameRangesN,
        exclusive: bool = False,
        mismatch: Literal[True] | bool = ...
    ) -> VideoNodeT:
        """
        Replaces frames in a clip with pre-calculated indices.
        Frame ranges are inclusive. This behaviour can be changed by setting `exclusive=True`.

        Examples with clips ``black`` and ``white`` of equal length:
            * ``replace_ranges(black, white, [(0, 1)])``: replace frames 0 and 1 with ``white``
            * ``replace_ranges(black, white, [(None, None)])``: replace the entire clip with ``white``
            * ``replace_ranges(black, white, [(0, None)])``: same as previous
            * ``replace_ranges(black, white, [(200, None)])``: replace 200 until the end with ``white``
            * ``replace_ranges(black, white, [(200, -1)])``: replace 200 until the end with ``white``,
                                                            leaving 1 frame of ``black``

        Optional Dependencies:
            * `vs-zip <https://github.com/dnjulek/vapoursynth-zip>`_ (highly recommended!)

        :param clip_a:      Original clip.
        :param clip_b:      Replacement clip.
        :param ranges:      Ranges to replace clip_a (original clip) with clip_b (replacement clip).
                            Integer values in the list indicate single frames,
                            Tuple values indicate inclusive ranges.
                            Negative integer values will be wrapped around based on clip_b's length.
                            None values are context dependent:
                                * None provided as sole value to ranges: no-op
                                * Single None value in list: Last frame in clip_b
                                * None as first value of tuple: 0
                                * None as second value of tuple: Last frame in clip_b
        :param exclusive:   Use exclusive ranges (Default: False).
        :param mismatch:    Accept format or resolution mismatch between clips.

        :return:            Clip with ranges from clip_a replaced with clip_b.
        """

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBack,
        *,
        mismatch: bool = False
    ) -> vs.VideoNode:
        """
        Replaces frames in a clip on-the-fly with a callback.

        Example usage:
        ```py
        # Replace frames from `clip_a` with `clip_b` if the frame number is between 200 and 299 (inclusive).
        clip = replace_ranges(clip_a, clip_b, lambda n: n in range(200, 300))
        ```

        :param clip_a:      Original clip.
        :param clip_b:      Replacement clip.
        :param ranges:      Callback to replace clip_a (original clip) with clip_b (replacement clip).
                            Must return True to replace a with b.
        :param mismatch:    Accept format or resolution mismatch between clips.

        :return:            Clip with ranges from clip_a replaced with clip_b.
        """

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBackF[vs.VideoFrame] | _RangesCallBackNF[vs.VideoFrame],
        *,
        mismatch: bool = False,
        prop_src: vs.VideoNode
    ) -> vs.VideoNode:
        """
        Replaces frames in a clip on-the-fly with a callback.

        Example usage:
        ```py
        # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P.
        clip = replace_ranges(clip_a, clip_b, lambda f: get_prop(f, '_PictType', str) == 'P', prop_src=clip_a)

        # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of ``clip_a`` is P
        # and if the frame number is between 200 and 299 (inclusive)
        clip = replace_ranges(
            clip_a, clip_b,
            lambda n, f: get_prop(f, '_PictType', str) == 'P' and n in range(200, 300),
            prop_src=clip_a
        )
        ```

        :param clip_a:      Original clip.
        :param clip_b:      Replacement clip.
        :param ranges:      Callback to replace clip_a (original clip) with clip_b (replacement clip).
                            Must return True to replace a with b.
        :param mismatch:    Accept format or resolution mismatch between clips.
        :param prop_src:    Source clip to use for frame properties in the callback.

        :return:            Clip with ranges from clip_a replaced with clip_b.
        """

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: _RangesCallBackF[Sequence[vs.VideoFrame]] | _RangesCallBackNF[Sequence[vs.VideoFrame]],
        *,
        mismatch: bool = False,
        prop_src: Sequence[vs.VideoNode]
    ) -> vs.VideoNode:
        """
        Replaces frames in a clip on-the-fly with a callback.

        Example usage:
        ```py
        prop_srcs: list[vs.VideNode]()
        ...

        # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of all the ``prop_srcs`` is P.
        clip = replace_ranges(
            clip_a, clip_b,
            lambda f: all(get_prop(frame, '_PictType', str) == 'P' for frame in f),
            prop_src=prop_srcs
        )

        # Replace frames from ``clip_a`` with ``clip_b`` if the picture type of all the ``prop_srcs`` is P
        # and if the frame number is between 200 and 299 (inclusive)
        clip = replace_ranges(
            clip_a, clip_b,
            lambda n, f: all(get_prop(frame, '_PictType', str) == 'P' for frame in f) and n in range(200, 300),
            prop_src=prop_srcs
        )
        ```

        :param clip_a:      Original clip.
        :param clip_b:      Replacement clip.
        :param ranges:      Callback to replace clip_a (original clip) with clip_b (replacement clip).
                            Must return True to replace a with b.
        :param mismatch:    Accept format or resolution mismatch between clips.
        :param prop_src:    Source clips to use for frame properties in the callback.

        :return:            Clip with ranges from clip_a replaced with clip_b.
        """

    @overload
    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN | _RangesCallBackT | None,
        exclusive: bool = False,
        mismatch: bool = False,
        *,
        prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None
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

    def __call__(
        self,
        clip_a: vs.VideoNode,
        clip_b: vs.VideoNode,
        ranges: FrameRangeN | FrameRangesN | _RangesCallBackT | None,
        exclusive: bool = False,
        mismatch: bool = False,
        *,
        prop_src: vs.VideoNode | Sequence[vs.VideoNode] | None = None
    ) -> vs.VideoNode:
        from ..functions import invert_ranges, normalize_ranges

        if ranges != 0 and not ranges or clip_a is clip_b:
            return clip_a

        if not mismatch:
            check_ref_clip(clip_a, clip_b)

        if callable(ranges):
            from inspect import Signature

            signature = Signature.from_callable(ranges, eval_str=True)

            params = set(signature.parameters.keys())

            base_clip = clip_a.std.BlankClip(keep=True, varformat=mismatch, varsize=mismatch)

            callback = ranges

            if 'f' in params and not prop_src:
                raise CustomValueError(
                    'To use frame properties in the callback (parameter "f"), '
                    'you must specify one or more source clips via `prop_src`!',
                    replace_ranges
                )

            def _func_nf(
                n: int, f: vs.VideoFrame | Sequence[vs.VideoFrame],
                callback: _RangesCallBackNF[vs.VideoFrame | Sequence[vs.VideoFrame]]
            ) -> vs.VideoNode:
                return clip_b if callback(n, f) else clip_a

            def _func_f(
                n: int, f: vs.VideoFrame | Sequence[vs.VideoFrame],
                callback: _RangesCallBackF[vs.VideoFrame | Sequence[vs.VideoFrame]]
            ) -> vs.VideoNode:
                return clip_b if callback(f) else clip_a

            def _func_n(n: int, callback: _RangesCallBack) -> vs.VideoNode:
                return clip_b if callback(n) else clip_a

            _func: Callable[..., vs.VideoNode]

            if 'f' in params and 'n' in params:
                _func = _func_nf
            elif 'f' in params:
                _func = _func_f
            elif 'n' in params:
                _func = _func_n
            else:
                raise CustomValueError(
                    'Callback must have signature ((n, f) | (n) | (f)) -> bool!', replace_ranges, callback
                )

            _func.__callback = callback  # type: ignore[attr-defined]
            _gc_func_gigacope.append(_func)

            return vs.core.std.FrameEval(
                base_clip, partial(_func, callback=callback), prop_src if 'f' in params else None, [clip_a, clip_b]
            )


        b_ranges = normalize_ranges(clip_b, ranges, exclusive)

        if hasattr(vs.core, 'vszip'):
            return vs.core.vszip.RFS(
                clip_a, clip_b, [y for (s, e) in b_ranges for y in range(s, e + (not exclusive))], mismatch=mismatch
            )

        a_ranges = invert_ranges(clip_a, clip_b, b_ranges, exclusive)

        a_trims = [clip_a[max(0, start - exclusive):end + (not exclusive)] for start, end in a_ranges]
        b_trims = [clip_b[start:end + (not exclusive)] for start, end in b_ranges]

        if a_ranges:
            main, other = (a_trims, b_trims) if (a_ranges[0][0] == 0) else (b_trims, a_trims)
        else:
            main, other = (b_trims, a_trims) if (b_ranges[0][0] == 0) else (a_trims, b_trims)

        return vs.core.std.Splice(list(interleave_arr(main, other, 1)), mismatch)


replace_ranges = _replace_ranges()


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
