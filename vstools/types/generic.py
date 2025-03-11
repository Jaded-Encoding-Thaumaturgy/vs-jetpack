from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, TypeVar, Union

import vapoursynth as vs

from jetpytools import MISSING, DataType, FuncExceptT, MissingT, PassthroughC, SingleOrArr, StrArr, StrArrOpt

__all__ = [
    'MissingT', 'MISSING',

    'FuncExceptT',

    'DataType', 'VSMapValue', 'BoundVSMapValue', 'VSMapValueCallback',

    'VideoFormatT',

    'HoldsVideoFormatT', 'HoldsPropValueT',

    'VSFunction', 'VSFunctionNoArgs', 'VSFunctionArgs', 'VSFunctionKwArgs', 'VSFunctionAllArgs', 'GenericVSFunction',

    'StrArr', 'StrArrOpt',

    'PassthroughC',

    'ConstantFormatVideoNode', 'VideoNodeT'
]

_VSMapValue = Union[
    SingleOrArr[int],
    SingleOrArr[float],
    SingleOrArr[DataType],
    SingleOrArr[vs.VideoNode],
    SingleOrArr[vs.VideoFrame],
    SingleOrArr[vs.AudioNode],
    SingleOrArr[vs.AudioFrame]
]
VSMapValue = Union[
    _VSMapValue,
    SingleOrArr[Callable[..., _VSMapValue]]
]
"""Values that a VSMap can hold, so all that a :py:attr:`vs.Function`` can accept in args and can return."""

BoundVSMapValue = TypeVar('BoundVSMapValue', bound=VSMapValue)
"""Type variable that can be one of the types in a VSMapValue."""

VSMapValueCallback = Callable[..., VSMapValue]
"""Callback that can be held in a VSMap. It can only return values representable in a VSMap."""

VideoFormatT = vs.PresetVideoFormat | vs.VideoFormat
"""Types representing a clear VideoFormat."""

HoldsVideoFormatT = vs.VideoNode | vs.VideoFrame | vs.VideoFormat
"""Types from which a clear VideoFormat can be retrieved."""

HoldsPropValueT = vs.FrameProps | vs.VideoFrame | vs.AudioFrame | vs.VideoNode | vs.AudioNode
"""Types that can hold :py:attr:`vs.FrameProps`."""


class VSFunctionNoArgs(Protocol):
    def __call__(self, clip: vs.VideoNode) -> vs.VideoNode:
        ...


class VSFunctionArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any) -> vs.VideoNode:
        ...


class VSFunctionKwArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
        ...


class VSFunctionAllArgs(Protocol):
    def __call__(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...


VSFunction = VSFunctionNoArgs | VSFunctionArgs | VSFunctionKwArgs | VSFunctionAllArgs
"""Function that takes a :py:attr:`vs.VideoNode` as its first argument and returns a :py:attr:`vs.VideoNode`."""

GenericVSFunction = Callable[..., vs.VideoNode]

if TYPE_CHECKING:
    class ConstantFormatVideoNode(vs.VideoNode):
        format: vs.VideoFormat

else:
    ConstantFormatVideoNode = vs.VideoNode


VideoNodeT = TypeVar("VideoNodeT", bound=vs.VideoNode)
