from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Protocol, TypeVar, Union

from jetpytools import MissingT

from ..vs_proxy import vs
from .builtins import Planes

__all__ = [
    "F_VD",
    "ConstantFormatVideoNode",
    "GenericVSFunction",
    "HoldsPropValue",
    "HoldsPropValueT",  # Deprecated alias
    "HoldsVideoFormat",
    "HoldsVideoFormatT",  # Deprecated alias
    "MissingT",
    "VSFunction",
    "VSFunctionAllArgs",
    "VSFunctionArgs",
    "VSFunctionKwArgs",
    "VSFunctionNoArgs",
    "VSFunctionPlanesArgs",
    "VideoFormatLike",
    "VideoFormatT",  # Deprecated alias
    "VideoNodeIterable",
    "VideoNodeIterableT",  # Deprecated alias
]

type VideoNodeIterable = vs.VideoNode | Iterable[VideoNodeIterable]

VideoNodeIterableT = VideoNodeIterable


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

VideoFormatT = VideoFormatLike
"""
Deprecated alias of VideoFormatLike.
"""

HoldsVideoFormatT = HoldsVideoFormat
"""
Deprecated alias of HoldsVideoFormat.
"""

HoldsPropValueT = HoldsPropValue
"""
Deprecated alias of HoldsPropValue.
"""


F_VD = TypeVar("F_VD", bound=Callable[..., vs.VideoNode])


class VSFunctionNoArgs(Protocol):
    def __call__(self, clip: vs.VideoNode) -> vs.VideoNode: ...


class VSFunctionArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any) -> vs.VideoNode: ...


class VSFunctionKwArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode: ...


class VSFunctionAllArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode: ...


VSFunction = Union[VSFunctionNoArgs, VSFunctionArgs, VSFunctionKwArgs, VSFunctionAllArgs]
"""
Function that takes a VideoNode as its first argument and returns a VideoNode.
"""


class VSFunctionPlanesArgs(VSFunctionKwArgs, Protocol):
    def __call__(self, clip: vs.VideoNode, *, planes: Planes = ..., **kwargs: Any) -> vs.VideoNode: ...


GenericVSFunction = Callable[..., vs.VideoNode]

ConstantFormatVideoNode = vs.VideoNode
"""
Deprecated alias of vs.VideoNode
"""
