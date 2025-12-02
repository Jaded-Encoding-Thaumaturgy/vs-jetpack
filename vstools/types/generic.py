from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from jetpytools import MissingT

from ..vs_proxy import vs
from .builtins import Planes

__all__ = [
    "AudioNodeIterable",
    "GenericVSFunction",
    "HoldsPropValue",
    "HoldsVideoFormat",
    "MissingT",
    "RawNodeIterable",
    "VSFunction",
    "VSFunctionAllArgs",
    "VSFunctionArgs",
    "VSFunctionKwArgs",
    "VSFunctionNoArgs",
    "VSFunctionPlanesArgs",
    "VideoFormatLike",
    "VideoNodeIterable",
]

type VideoNodeIterable = vs.VideoNode | Iterable[VideoNodeIterable]
type AudioNodeIterable = vs.AudioNode | Iterable[AudioNodeIterable]
type RawNodeIterable = vs.RawNode | Iterable[RawNodeIterable]


VideoFormatLike = vs.PresetVideoFormat | vs.VideoFormat
"""
Types representing a clear VideoFormat.
"""

HoldsVideoFormat = vs.VideoNode | vs.VideoFrame | vs.VideoFormat
"""
Types from which a clear VideoFormat can be retrieved.
"""

HoldsPropValue = vs.FrameProps | vs.VideoFrame | vs.AudioFrame | vs.VideoNode | vs.AudioNode | Mapping[str, Any]
"""
Types that can hold a vs.FrameProps.
"""


@runtime_checkable
class VSFunctionNoArgs(Protocol):
    def __call__(self, clip: vs.VideoNode) -> vs.VideoNode: ...


@runtime_checkable
class VSFunctionArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any) -> vs.VideoNode: ...


@runtime_checkable
class VSFunctionKwArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode: ...


@runtime_checkable
class VSFunctionAllArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode: ...


VSFunction = VSFunctionNoArgs | VSFunctionArgs | VSFunctionKwArgs | VSFunctionAllArgs
"""
Function that takes a VideoNode as its first argument and returns a VideoNode.
"""


class VSFunctionPlanesArgs(VSFunctionKwArgs, Protocol):
    def __call__(self, clip: vs.VideoNode, *, planes: Planes = ..., **kwargs: Any) -> vs.VideoNode: ...


GenericVSFunction = Callable[..., vs.VideoNode]
