from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Protocol, TypeAlias, TypeVar, Union

import vapoursynth as vs
from jetpytools import MISSING, DataType, FuncExcept, MissingT, PassthroughC, SingleOrSeq, StrArr, StrArrOpt

from .builtins import Planes

__all__ = [
    "F_VD",
    "MISSING",
    "BoundVSMapValue",
    "ConstantFormatVideoNode",
    "DataType",
    "FuncExcept",
    "FuncExceptT",  # Deprecated alias
    "GenericVSFunction",
    "HoldsPropValue",
    "HoldsPropValueT",  # Deprecated alias
    "HoldsVideoFormat",
    "HoldsVideoFormatT",  # Deprecated alias
    "MissingT",
    "PassthroughC",
    "StrArr",
    "StrArrOpt",
    "VSFunction",
    "VSFunctionAllArgs",
    "VSFunctionArgs",
    "VSFunctionKwArgs",
    "VSFunctionNoArgs",
    "VSFunctionPlanesArgs",
    "VSMapValue",
    "VSMapValueCallback",
    "VideoFormatLike",
    "VideoFormatT",  # Deprecated alias
    "VideoNodeIterableT",
]


FuncExceptT = FuncExcept

VideoNodeIterableT: TypeAlias = Union[vs.VideoNode, Iterable["VideoNodeIterableT"]]

_VSMapValue = Union[
    SingleOrSeq[int],
    SingleOrSeq[float],
    SingleOrSeq[DataType],
    SingleOrSeq[vs.VideoNode],
    SingleOrSeq[vs.VideoFrame],
    SingleOrSeq[vs.AudioNode],
    SingleOrSeq[vs.AudioFrame],
]
VSMapValue = Union[_VSMapValue, SingleOrSeq[Callable[..., _VSMapValue]]]
"""
Values that a VSMap can hold, so all that a vs.Function can accept in args and can return.
"""

BoundVSMapValue = TypeVar("BoundVSMapValue", bound=VSMapValue)
"""
Type variable that can be one of the types in a VSMapValue.
"""

VSMapValueCallback = Callable[..., VSMapValue]
"""
Callback that can be held in a VSMap. It can only return values representable in a VSMap.
"""

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
