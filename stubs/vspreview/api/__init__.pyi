from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import Any, Sequence, overload

from vapoursynth import AudioNode, RawNode, VideoNode

from vstools import Keyframes, vs, VideoNodeIterable, AudioNodeIterable, RawNodeIterable

__all__ = ["is_preview", "set_output", "set_scening", "set_timecodes", "start_preview", "update_node_info"]

def start_preview(path: str, *args: str) -> None: ...
def is_preview() -> bool: ...
def set_timecodes(
    index: int,
    timecodes: str | Path | dict[tuple[int | None, int | None], float | tuple[int, int] | Fraction] | list[Fraction],
    node: VideoNode | None = None,
    den: int = 1001,
) -> None: ...
def set_scening(
    scenes: Keyframes | list[tuple[int, int]] | list[Keyframes | list[tuple[int, int]]], node: VideoNode, name: str
) -> None: ...
def update_node_info(node_type: type[RawNode | VideoNode | AudioNode], index: int, **kwargs: Any) -> None: ...

type TimecodesT = (
    str
    | PathLike[str]
    | dict[tuple[int | None, int | None], float | tuple[int, int] | Fraction]
    | list[Fraction]
    | None
)
type ScenesT = Keyframes | list[tuple[int, int]] | list[Keyframes | list[tuple[int, int]]] | None

@overload
def set_output(
    node: vs.VideoNode,
    index: int = ...,
    /,
    *,
    alpha: vs.VideoNode | None = ...,
    timecodes: TimecodesT = None,
    denominator: int = 1001,
    scenes: ScenesT = None,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: vs.VideoNode,
    name: str | bool | None = ...,
    /,
    *,
    alpha: vs.VideoNode | None = ...,
    timecodes: TimecodesT = None,
    denominator: int = 1001,
    scenes: ScenesT = None,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: vs.VideoNode,
    index: int = ...,
    name: str | bool | None = ...,
    /,
    alpha: vs.VideoNode | None = ...,
    *,
    timecodes: TimecodesT = None,
    denominator: int = 1001,
    scenes: ScenesT = None,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: VideoNodeIterable | AudioNodeIterable | RawNodeIterable, index: int | Sequence[int] = ..., /, **kwargs: Any
) -> None: ...
@overload
def set_output(
    node: VideoNodeIterable | AudioNodeIterable | RawNodeIterable, name: str | bool | None = ..., /, **kwargs: Any
) -> None: ...
@overload
def set_output(
    node: VideoNodeIterable | AudioNodeIterable | RawNodeIterable,
    index: int | Sequence[int] = ...,
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None: ...
