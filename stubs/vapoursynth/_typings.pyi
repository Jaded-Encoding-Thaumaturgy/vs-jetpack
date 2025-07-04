from abc import abstractmethod
from typing import Any, Callable, Generic, Protocol, Sequence, TypedDict, TypeVar, Union, runtime_checkable, TypeAlias

from ._enums import ColorFamily, SampleType
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

###
# Typed dicts

class _VideoFormatInfo(TypedDict):
    id: int
    name: str
    color_family: ColorFamily
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int
