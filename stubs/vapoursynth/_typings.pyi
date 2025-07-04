from abc import abstractmethod
from typing import Any, Callable, Generic, Protocol, Sequence, TypeAlias, TypeVar, Union, runtime_checkable

from ._frames import AudioFrame, VideoFrame
from ._nodes import AudioNode, VideoNode

###
# Typing

_T = TypeVar("_T")
_SingleAndSequence: TypeAlias = _T | Sequence[_T]

@runtime_checkable
class _SupportsString(Protocol):
    @abstractmethod
    def __str__(self) -> str: ...

_DataType: TypeAlias = str | bytes | bytearray | _SupportsString

_VapourSynthMapValue: TypeAlias = Union[
    _SingleAndSequence[int],
    _SingleAndSequence[float],
    _SingleAndSequence[_DataType],
    _SingleAndSequence[VideoNode],
    _SingleAndSequence[VideoFrame],
    _SingleAndSequence[AudioNode],
    _SingleAndSequence[AudioFrame],
    _SingleAndSequence[_VSMapValueCallback[Any]],
]

_BoundVSMapValue = TypeVar("_BoundVSMapValue", bound=_VapourSynthMapValue)

_VSMapValueCallback: TypeAlias = Callable[..., _BoundVSMapValue]

class _Future(Generic[_T]):
    def set_result(self, value: _T) -> None: ...
    def set_exception(self, exception: BaseException) -> None: ...
    def result(self) -> _T: ...
    def exception(self) -> None: ...
