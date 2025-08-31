from fractions import Fraction
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Sequence, overload

from vapoursynth import AudioNode, RawNode, VideoNode

from vstools import Keyframes, vs

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
def set_output(node: vs.AudioNode, index: int = ..., /, **kwargs: Any) -> None: ...
@overload
def set_output(node: vs.AudioNode, name: str | bool | None = ..., /, **kwargs: Any) -> None: ...
@overload
def set_output(node: vs.AudioNode, index: int = ..., name: str | bool | None = ..., /, **kwargs: Any) -> None: ...
@overload
def set_output(
    node: Iterable[vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]],
    index: int | Sequence[int] = ...,
    /,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: Iterable[vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]],
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: Iterable[vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]]],
    index: int | Sequence[int] = ...,
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: Iterable[vs.AudioNode | Iterable[vs.AudioNode | Iterable[vs.AudioNode]]],
    index: int | Sequence[int] = ...,
    /,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: Iterable[vs.AudioNode | Iterable[vs.AudioNode | Iterable[vs.AudioNode]]],
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: Iterable[vs.AudioNode | Iterable[vs.AudioNode | Iterable[vs.AudioNode]]],
    index: int | Sequence[int] = ...,
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None: ...
@overload
def set_output(
    node: vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode]]],
    index: int | Sequence[int] = ...,
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None: ...
