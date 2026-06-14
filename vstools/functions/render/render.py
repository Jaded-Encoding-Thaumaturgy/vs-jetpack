from __future__ import annotations

import contextlib
import operator
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from math import floor
from os import PathLike
from typing import Any, BinaryIO, Literal, Protocol, overload

from jetpytools import CustomRuntimeError, CustomValueError, Sentinel, SentinelT, SPathLike, fallback

from ...utils import get_prop
from ...vs_proxy import core, vs
from ..ranges import normalize_list_to_ranges, replace_ranges

__all__ = ["AsyncRenderConf", "clip_async_render", "clip_data_gather", "find_prop", "find_prop_rfs", "prop_compare_cb"]


class _CallbackModifyFrame(Protocol):
    def __call__(self, n: int, f: vs.VideoFrame) -> vs.VideoFrame: ...


@dataclass(frozen=True)
class AsyncRenderConf:
    """
    Configuration for asynchronous/non-consecutive frame requests during clip rendering.

    This configuration allows splitting the rendering workload into multiple chunks or steps
    to speed up processing by requesting frames out of order or in parallel.
    """

    n: int = 2
    """
    The number of chunks or step size to use for parallelizing requests.
    Must be greater than 1.
    """
    parallel_input: bool = False
    """
    If True, uses `core.std.StackHorizontal` on chunk boundaries to request frames
    in parallel across the chunks.
    If False, requests frames with a step of `n` inside a single `ModifyFrame` call.
    """

    def __post_init__(self) -> None:
        if self.n < 2:
            raise CustomValueError("AsyncRenderConf.n must be greater than 1")

    def get_clip[T](
        self,
        clip: vs.VideoNode,
        callback: Callable[[int, vs.VideoFrame], T],
        result: dict[int, T],
        progress_up: Callable[[int], None] | None,
    ) -> vs.VideoNode:
        """
        Set up the rendering pipeline using the configured asynchronous request method.

        Args:
            clip: The clip to render.
            callback: The callback function to execute on each frame.
            result: A dictionary to store the callback results, mapped by frame index.
            progress_up: An optional progress callback that receives the frame index.

        Returns:
            A `VideoNode` that, when rendered, triggers the callback for each frame.
        """
        chunk = floor(clip.num_frames / self.n)
        cl = chunk * self.n

        blankclip = clip.std.BlankClip(length=chunk, keep=True)

        if self.parallel_input:
            rend_clip = core.std.StackHorizontal(
                [
                    blankclip.std.ModifyFrame(
                        clip[chunk * i : chunk * (i + 1)],
                        _get_render_callback(callback, result, progress_up, chunk * i),
                    )
                    for i in range(self.n)
                ]
            )
        else:
            cb = _get_render_callback(callback, result, progress_up)
            indices = [range(i, cl, self.n) for i in range(self.n)]

            def _var(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
                for idx, fi in enumerate(f):
                    cb(indices[idx][n], fi)

                return f[0]

            rend_clip = blankclip.std.ModifyFrame([clip[i :: self.n] for i in range(self.n)], _var)

        if cl != clip.num_frames:
            rend_rest = (
                clip[cl:]
                .std.BlankClip(keep=True)
                .std.ModifyFrame(clip[cl:], _get_render_callback(callback, result, progress_up, cl))
            )
            rend_clip = core.std.Splice([rend_clip, rend_rest], self.parallel_input)

        return rend_clip


def _get_render_callback[T](
    callback: Callable[[int, vs.VideoFrame], T],
    result: dict[int, T],
    progress_up: Callable[[int], None] | None,
    shift: int = 0,
) -> _CallbackModifyFrame:
    if shift:

        def cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            real_n = n + shift
            result[real_n] = callback(real_n, f)
            if progress_up:
                progress_up(real_n)
            return f
    else:

        def cb(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            result[n] = callback(n, f)
            if progress_up:
                progress_up(n)
            return f

    return cb


@overload
def clip_async_render(  # pyright: ignore[reportOverlappingOverload]
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: None = None,
    prefetch: int | None = None,
    backlog: int | None = None,
    y4m: bool = False,
) -> None: ...


@overload
def clip_async_render[T](
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: Callable[[int, vs.VideoFrame], T] = ...,
    prefetch: int | None = None,
    backlog: int | None = None,
    y4m: bool = False,
    async_requests: int | Literal[False] | AsyncRenderConf = False,
) -> list[T]: ...


def clip_async_render[T](
    clip: vs.VideoNode,
    outfile: BinaryIO | SPathLike | None = None,
    progress: str | Callable[[int, int], None] | None = None,
    callback: Callable[[int, vs.VideoFrame], T] | None = None,
    prefetch: int | None = None,
    backlog: int | None = None,
    y4m: bool = False,
    async_requests: int | Literal[False] | AsyncRenderConf = False,
) -> list[T] | None:
    """
    Iterate over an entire clip and optionally write results to a file or gather data.

    This is mostly useful for metric gathering that must be performed before any other processing.
    This could be for example gathering scenechanges, per-frame heuristics, etc.

    It's highly recommended to perform as little filtering as possible on the input clip for speed purposes.

    Example usage:
        ```py
        # Gather scenechanges.
        scenechanges = clip_async_render(
            clip, None, "Searching for scenechanges...", lambda n, f: get_prop(f, "_SceneChange", int)
        )

        # Gather average planes stats.
        avg_planes = clip_async_render(
            clip, None, "Calculating average planes...", lambda n, f: get_prop(f, "PlaneStatsAverage", float)
        )
        ```

    Args:
        clip: Clip to render.
        outfile: Optional binary output or path to write to.
        progress: Progress reporting configuration. Can be:

               - `str`: A description message displayed next to the rendering progress bar.
               - `Callable[[int, int], None]`: A custom progress callback function
                 that receives the current frame index and total frames `(current_frame, total_frames)`.
               - `None`: Disables progress reporting completely.
        callback: Callback function invoked for every frame during rendering.
            Must accept `n` and `f` (like a frameeval would) and return some value of type `T`.
            The returned values are collected in order of frame index and returned as a `list[T]`.
            Only active when `outfile` is not provided.
        prefetch: Defines how many frames are rendered concurrently.
        backlog: Defines how many unconsumed frames vapoursynth buffers at most
            before it stops rendering additional frames.
        y4m: Whether to add YUV4MPEG2 headers to the rendered output.
        async_requests: Whether to render frames non-consecutively. If int, determines the number of requests.
    """
    if isinstance(outfile, (str, PathLike)) and outfile is not None:
        with open(outfile, "wb") as f:
            return clip_async_render(clip, f, progress, callback, prefetch, backlog, y4m, async_requests)  # type: ignore[misc,arg-type]

    async_conf: AsyncRenderConf | Literal[False]

    if isinstance(async_requests, AsyncRenderConf):
        async_conf = async_requests
    elif async_requests is False:
        async_conf = False
    else:
        async_conf = AsyncRenderConf(async_requests)

    if async_conf and (not callback or outfile):
        raise CustomValueError(
            "You cannot use async requests without a callback and with an outfile", clip_async_render
        )

    total_frames = clip.num_frames

    if not progress:
        pr_ctx = contextlib.nullcontext()
        pr_up = lambda *_: None  # noqa: E731
    elif callable(progress):
        pr_ctx = contextlib.nullcontext()
        pr_up = progress
    else:
        from .progress import get_render_progress

        pr_ctx = get_render_progress()
        task = pr_ctx.add_task(progress)
        pr_up = lambda n, total: pr_ctx.update(task, total=total, completed=n)  # noqa: E731

    with pr_ctx:
        if outfile:
            clip.output(outfile, y4m, pr_up, prefetch or 0, fallback(backlog, -1))
        elif not callback:
            for i, _ in enumerate(clip.frames(prefetch, backlog, True)):
                pr_up(i, total_frames)
        else:
            result = dict[int, T]()

            if not async_conf:
                clip = core.std.ModifyFrame(
                    clip.std.BlankClip(keep=True),
                    clip,
                    _get_render_callback(callback, result, lambda n: pr_up(n, total_frames)),
                )
            else:
                clip = async_conf.get_clip(clip, callback, result, lambda n: pr_up(n, total_frames))

            deque(clip.frames(prefetch, backlog, True), 0)

            try:
                return [result[i] for i in range(total_frames)]
            except KeyError:
                raise CustomRuntimeError(
                    "There was an error with the rendering and one frame request was rejected!", clip_async_render
                )

    return None


def clip_data_gather[T](
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
    src: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    return_frame_n: Literal[False] = False,
) -> tuple[vs.VideoNode, Callable[[int, vs.VideoFrame], bool]]: ...


@overload
def prop_compare_cb(
    src: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    *,
    return_frame_n: Literal[True],
) -> tuple[vs.VideoNode, Callable[[int, vs.VideoFrame], int | SentinelT]]: ...


def prop_compare_cb(
    src: vs.VideoNode,
    prop: str,
    op: str | Callable[[float, float], bool] | None,
    ref: float | bool,
    return_frame_n: bool = False,
) -> (
    tuple[vs.VideoNode, Callable[[int, vs.VideoFrame], bool]]
    | tuple[vs.VideoNode, Callable[[int, vs.VideoFrame], int | SentinelT]]
):
    bool_check = isinstance(ref, bool)
    assert (op is None) if bool_check else (op is not None)

    if isinstance(op, str):
        assert op in _operators

    op_ = _operators[op][0] if isinstance(op, str) else op

    def _cb_return_frame_n(n: int, f: vs.VideoFrame) -> int | SentinelT:
        assert op_
        return Sentinel.check(n, op_(get_prop(f, prop, (float, bool)), ref))

    def _cb_not_return_frame_n(n: int, f: vs.VideoFrame) -> bool:
        assert op_
        return op_(get_prop(f, prop, (float, bool)), ref)

    callback = _cb_return_frame_n if return_frame_n else _cb_not_return_frame_n

    return src, callback


@overload
def find_prop(  # pyright: ignore[reportOverlappingOverload]
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
        ```py
        # Return a list of all frames that were marked as combed.
        find_prop(clip, "_Combed", None, True, 0)
        ```

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

    prop_src, callback = prop_compare_cb(src, prop, op, ref, return_frame_n=True)

    aconf = AsyncRenderConf(async_requests, False)

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
        ```py
        # Replace a rescaled clip with the original clip for frames where the error
        # (defined on another clip) is equal to or greater than 0.025.
        find_prop_rfs(scaled, src, "PlaneStatsAverage", ">=", 0.025, err_clip)
        ```

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
    prop_src, callback = prop_compare_cb(ref or clip_a, prop, op, prop_ref, False)

    return replace_ranges(clip_a, clip_b, callback, False, mismatch, prop_src=prop_src)
