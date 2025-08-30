from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol, TypeAlias, TypeVar, Union

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
    "FuncExcept",
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
    "VideoNodeT",
]


FuncExcept = FuncExcept

VideoNodeT = TypeVar("VideoNodeT", bound=vs.VideoNode)
VideoNodeT_contra = TypeVar("VideoNodeT_contra", bound=vs.VideoNode, contravariant=True)
VideoNodeT_co = TypeVar("VideoNodeT_co", bound=vs.VideoNode, covariant=True)

VideoNodeIterableT: TypeAlias = Union[VideoNodeT, Iterable["VideoNodeIterableT[VideoNodeT]"]]

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


class VSFunctionNoArgs(Protocol[VideoNodeT_contra, VideoNodeT_co]):
    def __call__(self, clip: VideoNodeT_contra) -> VideoNodeT_co: ...


class VSFunctionArgs(Protocol[VideoNodeT_contra, VideoNodeT_co]):
    def __call__(self, clip: VideoNodeT_contra, *args: Any) -> VideoNodeT_co: ...


class VSFunctionKwArgs(Protocol[VideoNodeT_contra, VideoNodeT_co]):
    def __call__(self, clip: VideoNodeT_contra, **kwargs: Any) -> VideoNodeT_co: ...


class VSFunctionAllArgs(Protocol[VideoNodeT_contra, VideoNodeT_co]):
    def __call__(self, clip: VideoNodeT_contra, *args: Any, **kwargs: Any) -> VideoNodeT_co: ...


VSFunction = Union[
    VSFunctionNoArgs[VideoNodeT, VideoNodeT],
    VSFunctionArgs[VideoNodeT, VideoNodeT],
    VSFunctionKwArgs[VideoNodeT, VideoNodeT],
    VSFunctionAllArgs[VideoNodeT, VideoNodeT],
]
"""
Function that takes a VideoNode as its first argument and returns a VideoNode.
"""


class VSFunctionPlanesArgs(VSFunctionKwArgs[VideoNodeT_contra, VideoNodeT_co], Protocol):
    def __call__(self, clip: VideoNodeT_contra, *, planes: Planes = ..., **kwargs: Any) -> VideoNodeT_co: ...


GenericVSFunction = Callable[..., VideoNodeT]

if TYPE_CHECKING:

    class ConstantFormatVideoNode(vs.VideoNode):
        format: vs.VideoFormat

else:
    ConstantFormatVideoNode = vs.VideoNode
