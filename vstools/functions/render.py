from __future__ import annotations

import operator
from collections import deque
from dataclasses import dataclass
from math import floor
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, Callable, Literal, Union, overload

import vapoursynth as vs
from jetpytools import (
    CustomRuntimeError,
    CustomValueError,
    Sentinel,
    SentinelT,
    SPath,
    SPathLike,
    T,
)

from ..exceptions import InvalidColorFamilyError
from ..types import VideoNodeT
from .normalize import normalize_list_to_ranges

__all__ = ["AsyncRenderConf", "clip_async_render", "clip_data_gather", "find_prop", "find_prop_rfs", "prop_compare_cb"]


@dataclass
class AsyncRenderConf:
    n: int = 2
    one_pix_frame: bool = False
    parallel_input: bool = False


@overload
def clip_async_render(
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: None = None,
    prefetch: int = 0,
    backlog: int = -1,
    y4m: bool | None = None,
    async_requests: int | bool | AsyncRenderConf = False,
) -> None: ...


@overload
def clip_async_render(
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: Callable[[int, vs.VideoFrame], T] = ...,
    prefetch: int = 0,
    backlog: int = -1,
    y4m: bool | None = None,
    async_requests: int | bool | AsyncRenderConf = False,
) -> list[T]: ...


@overload
def clip_async_render(
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: Callable[[int, vs.VideoFrame], T] | None = ...,
    prefetch: int = 0,
    backlog: int = -1,
    y4m: bool | None = None,
    async_requests: int | bool | AsyncRenderConf = False,
) -> list[T] | None: ...


def clip_async_render(
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: Callable[[int, vs.VideoFrame], T] | None = None,
    prefetch: int = 0,
    backlog: int = -1,
    y4m: bool | None = None,
    async_requests: int | bool | AsyncRenderConf = False,
) -> list[T] | None:
    """
    Iterate over an entire clip and optionally write results to a file.

    This is mostly useful for metric gathering that must be performed before any other processing.
    This could be for example gathering scenechanges, per-frame heuristics, etc.

    It's highly recommended to perform as little filtering as possible on the input clip for speed purposes.

    Example usage:

        # Gather scenechanges.
        >>> scenechanges = clip_async_render(
            clip, None, 'Searching for scenechanges...', lambda n, f: get_prop(f, "_SceneChange", int)
        )

        # Gather average planes stats.
        >>> avg_planes = clip_async_render(
            clip, None, 'Calculating average planes...', lambda n, f: get_prop(f, "PlaneStatsAverage", float)
        )

    Args:
        clip: Clip to render.
        outfile: Optional binary output or path to write to.
        progress: A message to display during rendering. This is shown alongside the progress.
        callback: Callback function. Must accept `n` and `f` (like a frameeval would) and return some value. This
            function is used to determine what information gets returned per frame. Default: None.
        prefetch: The amount of frames to prefetch. 0 means automatically determine. Default: 0.
        backlog: How many frames to hold. Useful for if your write of callback is slower than your frame rendering.
        y4m: Whether to add YUV4MPEG2 headers to the rendered output. If None, automatically determine. Default: None.
        async_requests: Whether to render frames non-consecutively. If int, determines the number of requests. Default:
            False.
    """

    from .funcs import fallback

    if isinstance(outfile, (str, PathLike, Path, SPath)) and outfile is not None:
        with open(outfile, "wb") as f:
            return clip_async_render(clip, f, progress, callback, prefetch, backlog, y4m, async_requests)

    result = dict[int, T]()
    async_conf: AsyncRenderConf | Literal[False]

    if async_requests is True:
        async_conf = AsyncRenderConf(1)
    elif isinstance(async_requests, int):
        if isinstance(async_requests, int) and async_requests <= 1:
            async_conf = False
        else:
            async_conf = AsyncRenderConf(async_requests)
    else:
        async_conf = False if async_requests.n <= 1 else async_requests

    if async_conf and async_conf.one_pix_frame and y4m:
        raise CustomValueError("You cannot have y4m=True and one_pix_frame in AsyncRenderConf!")

    num_frames = len(clip)

    pr_update: Callable[[], None]
    pr_update_custom: Callable[[int, int], None]

    if callback:

        def get_callback(shift: int = 0) -> Callable[[int, vs.VideoFrame], vs.VideoFrame]:
            if shift:
                if outfile is None and progress is not None:
                    if isinstance(progress, str):

                        def _cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                            n += shift
                            result[n] = callback(n, f)
                            pr_update()
                            return f
                    else:

                        def _cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                            n += shift
                            result[n] = callback(n, f)
                            pr_update_custom(n, num_frames)
                            return f
                else:

                    def _cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                        n += shift
                        result[n] = callback(n, f)
                        return f
            else:
                if outfile is None and progress is not None:
                    if isinstance(progress, str):

                        def _cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                            result[n] = callback(n, f)
                            pr_update()
                            return f
                    else:

                        def _cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                            result[n] = callback(n, f)
                            pr_update_custom(n, num_frames)
                            return f
                else:

                    def _cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                        result[n] = callback(n, f)
                        return f

            return _cb

        if async_conf and async_conf.one_pix_frame and (clip.width != clip.height != 1):
            clip = clip.std.CropAbs(1, 1)

        if not async_conf or async_conf.n == 1:
            blankclip = clip.std.BlankClip(keep=True)

            _cb = get_callback()

            if async_conf:
                rend_clip = blankclip.std.FrameEval(lambda n: blankclip.std.ModifyFrame(clip, _cb))
            else:
                rend_clip = blankclip.std.ModifyFrame(clip, _cb)
        else:
            if outfile:
                raise CustomValueError("You cannot have and output file with multi async request!", clip_async_render)

            chunk = floor(clip.num_frames / async_conf.n)
            cl = chunk * async_conf.n

            blankclip = clip.std.BlankClip(length=chunk, keep=True)

            stack = async_conf.parallel_input and not async_conf.one_pix_frame

            if stack:
                rend_clip = vs.core.std.StackHorizontal(
                    [
                        blankclip.std.ModifyFrame(clip[chunk * i : chunk * (i + 1)], get_callback(chunk * i))
                        for i in range(async_conf.n)
                    ]
                )
            else:
                _cb = get_callback()

                clip_indices = list(range(cl))
                range_indices = list(range(async_conf.n))

                indices = [clip_indices[i :: async_conf.n] for i in range_indices]

                def _var(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
                    for i, fi in zip(range_indices, f):
                        _cb(indices[i][n], fi)

                    return f[0]

                rend_clip = blankclip.std.ModifyFrame([clip[i :: async_conf.n] for i in range_indices], _var)

            if cl != clip.num_frames:
                rend_rest = blankclip[: clip.num_frames - cl].std.ModifyFrame(clip[cl:], get_callback(cl))
                rend_clip = vs.core.std.Splice([rend_clip, rend_rest], stack)
    else:
        rend_clip = clip

    if outfile is None:
        if y4m:
            raise CustomValueError("You cannot have y4m=False without any output file!", clip_async_render)

        clip_it = rend_clip.frames(prefetch, backlog, True)

        if progress is None:
            deque(clip_it, 0)
        elif isinstance(progress, str):
            from .progress import get_render_progress

            with get_render_progress(progress, clip.num_frames) as pr:
                if callback:
                    pr_update = pr.update
                    deque(clip_it, 0)
                else:
                    for _ in clip_it:
                        pr.update()
        else:
            if callback:
                pr_update_custom = progress
                deque(clip_it, 0)
            else:
                for i, _ in enumerate(clip_it):
                    progress(i, num_frames)

    else:
        y4m = fallback(y4m, bool(rend_clip.format and (rend_clip.format.color_family is vs.YUV)))

        if y4m:
            if rend_clip.format is None:
                raise CustomValueError(
                    "You cannot have y4m=True when rendering a variable resolution clip!", clip_async_render
                )
            else:
                InvalidColorFamilyError.check(
                    rend_clip,
                    (vs.YUV, vs.GRAY),
                    clip_async_render,
                    message="Can only render to y4m clips with {correct} color family, not {wrong}!",
                )

        if progress is None:
            rend_clip.output(outfile, y4m, None, prefetch, backlog)
        elif isinstance(progress, str):
            from .progress import get_render_progress

            with get_render_progress(progress, clip.num_frames) as pr:
                rend_clip.output(outfile, y4m, pr.update, prefetch, backlog)
        else:
            rend_clip.output(outfile, y4m, progress, prefetch, backlog)

    if callback:
        try:
            return [result[i] for i in range(clip.num_frames)]
        except KeyError:
            raise CustomRuntimeError(
                "There was an error with the rendering and one frame request was rejected!", clip_async_render
            )

    return None


def clip_data_gather(
    clip: vs.VideoNode,
    progress: str | Callable[[int, int], None] | None,
    callback: Callable[[int, vs.VideoFrame], SentinelT | T],
    async_requests: int | bool | AsyncRenderConf = False,
    prefetch: int = 0,
    backlog: int = -1,
) -> list[T]:
    frames = clip_async_render(clip, None, progress, callback, prefetch, backlog, False, async_requests)

    return list(Sentinel.filter(frames))


_operators: dict[str, tuple[Callable[[Any, Any], bool], str]] = {
    "<": (operator.lt, "<"),
    "<=": (operator.le, "<="),
    "==": (operator.eq, "="),
    "!=": (operator.ne, "= not"),
    ">": (operator.gt, ">"),
    ">=": (operator.ge, ">="),
}


@overload
def prop_compare_cb(
    src: VideoNodeT,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    return_frame_n: Literal[False] = ...,
) -> tuple[VideoNodeT, Callable[[int, vs.VideoFrame], bool]]: ...


@overload
def prop_compare_cb(
    src: VideoNodeT,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    return_frame_n: Literal[True] = ...,
) -> tuple[VideoNodeT, Callable[[int, vs.VideoFrame], int | SentinelT]]: ...


def prop_compare_cb(
    src: VideoNodeT,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    return_frame_n: bool = False,
) -> Union[
    tuple[VideoNodeT, Callable[[int, vs.VideoFrame], bool]],
    tuple[VideoNodeT, Callable[[int, vs.VideoFrame], int | SentinelT]],
]:
    bool_check = isinstance(ref, bool)
    one_pix = hasattr(vs.core, "akarin") and not (callable(op) or " " in prop)
    assert (op is None) if bool_check else (op is not None)

    if isinstance(op, str):
        assert op in _operators

    callback: Callable[[int, vs.VideoFrame], SentinelT | int]
    if one_pix:
        clip = (
            vs.core.std.BlankClip(None, 1, 1, vs.GRAY8 if bool_check else vs.GRAYS, length=src.num_frames)
            .std.CopyFrameProps(src)
            .akarin.Expr(
                f"x.{prop}" if bool_check else f"x.{prop} {ref} {_operators[op][1]}"  # type: ignore[index]
            )
        )
        src = clip  # type: ignore[assignment]

        def _cb_one_px_return_frame_n(n: int, f: vs.VideoFrame) -> int | SentinelT:
            return Sentinel.check(n, bool(f[0][0, 0]))

        def _cb_one_px_not_return_frame_n(n: int, f: vs.VideoFrame) -> bool:
            return bool(f[0][0, 0])

        callback = _cb_one_px_return_frame_n if return_frame_n else _cb_one_px_not_return_frame_n
    else:
        from ..utils import get_prop

        _op = _operators[op][0] if isinstance(op, str) else op

        def _cb_return_frame_n(n: int, f: vs.VideoFrame) -> int | SentinelT:
            assert _op
            return Sentinel.check(n, _op(get_prop(f, prop, (float, bool)), ref))

        def _cb_not_return_frame_n(n: int, f: vs.VideoFrame) -> bool:
            assert _op
            return _op(get_prop(f, prop, (float, bool)), ref)

        callback = _cb_return_frame_n if return_frame_n else _cb_not_return_frame_n

    return src, callback


@overload
def find_prop(
    src: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    range_length: Literal[0] = ...,
    async_requests: int = 1,
) -> list[int]: ...


@overload
def find_prop(
    src: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    range_length: int = ...,
    async_requests: int = 1,
) -> list[tuple[int, int]]: ...


def find_prop(
    src: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    range_length: int = 0,
    async_requests: int = 1,
) -> list[int] | list[tuple[int, int]]:
    """
    Find specific frame props in the clip and return a list of frame ranges that meets the conditions.

    Example usage:

        # Return a list of all frames that were marked as combed.
        >>> find_prop(clip, "_Combed", None, True, 0)

    Args:
        src: Input clip.
        prop: Frame prop to perform checks on.
        op: Conditional operator to apply between prop and ref ("<", "<=", "==", "!=", ">" or ">="). If None, check
            whether a prop is truthy.
        ref: Value to be compared with prop.
        range_length: Amount of frames to finish a sequence, to avoid false negatives. This will create ranges with a
            sequence of start-end tuples.
        async_requests: Whether to render frames non-consecutively. If int, determines the number of requests. Default:
            1.

    Returns:
        Frame ranges at the specified conditions.
    """

    prop_src, callback = prop_compare_cb(src, prop, op, ref, True)

    aconf = AsyncRenderConf(async_requests, (prop_src.width, prop_src.height) == (1, 1), False)

    frames = clip_data_gather(prop_src, f"Searching {prop} {op} {ref}...", callback, aconf)

    if range_length > 0:
        return normalize_list_to_ranges(frames, range_length)

    return frames


def find_prop_rfs(
    clip_a: vs.VideoNode,
    clip_b: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    prop_ref: float | bool,
    ref: vs.VideoNode | None = None,
    mismatch: bool = False,
) -> vs.VideoNode:
    """
    Conditional replace frames from the original clip with a replacement clip by comparing frame properties.

    Example usage:

        # Replace a rescaled clip with the original clip for frames where the error
        # (defined on another clip) is equal to or greater than 0.025.
        >>> find_prop_rfs(scaled, src, "PlaneStatsAverage", ">=", 0.025, err_clip)

    Args:
        clip_a: Original clip.
        clip_b: Replacement clip.
        prop: Frame prop to perform checks on.
        op: Conditional operator to apply between prop and ref ("<", "<=", "==", "!=", ">" or ">="). If None, check
            whether a prop is truthy. Default: None.
        prop_ref: Value to be compared with prop.
        ref: Optional reference clip to read frame properties from. Default: None.
        mismatch: Accept format or resolution mismatch between clips. Default: False.

    Returns:
        Clip where frames that meet the specified criteria were replaced with a different clip.
    """

    from ..utils import replace_ranges

    prop_src, callback = prop_compare_cb(ref or clip_a, prop, op, prop_ref, False)

    return replace_ranges(clip_a, clip_b, callback, False, mismatch, prop_src=prop_src)
