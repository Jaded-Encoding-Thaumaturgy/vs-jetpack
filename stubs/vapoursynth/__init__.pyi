# This file is auto-generated. DO NOT EDIT.
# ruff: noqa
# flake8: noqa
# fmt: off
# isort: skip_file

from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, MutableMapping
from concurrent.futures import Future
from ctypes import c_void_p
from enum import Enum, IntEnum, IntFlag
from fractions import Fraction
from inspect import Signature
from logging import Handler, LogRecord, StreamHandler
from types import MappingProxyType, TracebackType
from typing import Any, Concatenate, Final, IO, Literal, NamedTuple, Protocol, Self, SupportsFloat, SupportsIndex, SupportsInt, TextIO, TypedDict, final, overload
from warnings import deprecated
from weakref import ReferenceType


__all__ = [
    "CHROMA_BOTTOM",
    "CHROMA_BOTTOM_LEFT",
    "CHROMA_CENTER",
    "CHROMA_LEFT",
    "CHROMA_TOP",
    "CHROMA_TOP_LEFT",
    "FIELD_BOTTOM",
    "FIELD_PROGRESSIVE",
    "FIELD_TOP",
    "FLOAT",
    "GRAY",
    "GRAY8",
    "GRAY9",
    "GRAY10",
    "GRAY12",
    "GRAY14",
    "GRAY16",
    "GRAY32",
    "GRAYH",
    "GRAYS",
    "INTEGER",
    "NONE",
    "RANGE_FULL",
    "RANGE_LIMITED",
    "RGB",
    "RGB24",
    "RGB27",
    "RGB30",
    "RGB36",
    "RGB42",
    "RGB48",
    "RGBH",
    "RGBS",
    "YUV",
    "YUV410P8",
    "YUV411P8",
    "YUV420P8",
    "YUV420P9",
    "YUV420P10",
    "YUV420P12",
    "YUV420P14",
    "YUV420P16",
    "YUV420PH",
    "YUV420PS",
    "YUV422P8",
    "YUV422P9",
    "YUV422P10",
    "YUV422P12",
    "YUV422P14",
    "YUV422P16",
    "YUV422PH",
    "YUV422PS",
    "YUV440P8",
    "YUV444P8",
    "YUV444P9",
    "YUV444P10",
    "YUV444P12",
    "YUV444P14",
    "YUV444P16",
    "YUV444PH",
    "YUV444PS",
    "clear_output",
    "clear_outputs",
    "core",
    "get_output",
    "get_outputs",
]

type _AnyStr = str | bytes | bytearray
type _IntLike = SupportsInt | SupportsIndex | Buffer
type _FloatLike = SupportsFloat | SupportsIndex | Buffer

type _VSValueSingle = (
    int | float | _AnyStr | RawFrame | VideoFrame | AudioFrame | RawNode | VideoNode | AudioNode | Callable[..., Any]
)

type _VSValueIterable = (
    _SupportsIter[_IntLike]
    | _SupportsIter[_FloatLike]
    | _SupportsIter[_AnyStr]
    | _SupportsIter[RawFrame]
    | _SupportsIter[VideoFrame]
    | _SupportsIter[AudioFrame]
    | _SupportsIter[RawNode]
    | _SupportsIter[VideoNode]
    | _SupportsIter[AudioNode]
    | _SupportsIter[Callable[..., Any]]
    | _GetItemIterable[_IntLike]
    | _GetItemIterable[_FloatLike]
    | _GetItemIterable[_AnyStr]
    | _GetItemIterable[RawFrame]
    | _GetItemIterable[VideoFrame]
    | _GetItemIterable[AudioFrame]
    | _GetItemIterable[RawNode]
    | _GetItemIterable[VideoNode]
    | _GetItemIterable[AudioNode]
    | _GetItemIterable[Callable[..., Any]]
)
type _VSValue = _VSValueSingle | _VSValueIterable

class _SupportsIter[T](Protocol):
    def __iter__(self) -> Iterator[T]: ...

class _SequenceLike[T](Protocol):
    def __iter__(self) -> Iterator[T]: ...
    def __len__(self) -> int: ...

class _GetItemIterable[T](Protocol):
    def __getitem__(self, i: SupportsIndex, /) -> T: ...

class _SupportsKeysAndGetItem[KT, VT](Protocol):
    def __getitem__(self, key: KT, /) -> VT: ...
    def keys(self) -> Iterable[KT]: ...

class _VSCallback(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> _VSValue: ...

# Known callback signatures
# _VSCallback_{plugin_namespace}_{Function_name}_{parameter_name}
class _VSCallback_akarin_PropExpr_dict(Protocol):
    def __call__(
        self,
    ) -> Mapping[
        str,
        _IntLike
        | _FloatLike
        | _AnyStr
        | _SupportsIter[_IntLike]
        | _SupportsIter[_AnyStr]
        | _SupportsIter[_FloatLike]
        | _GetItemIterable[_IntLike]
        | _GetItemIterable[_FloatLike]
        | _GetItemIterable[_AnyStr],
    ]: ...

class _VSCallback_descale_Decustom_custom_kernel(Protocol):
    def __call__(self, *, x: float) -> _FloatLike: ...

class _VSCallback_descale_ScaleCustom_custom_kernel(Protocol):
    def __call__(self, *, x: float) -> _FloatLike: ...

class _VSCallback_std_FrameEval_eval_0(Protocol):
    def __call__(self, *, n: int) -> VideoNode: ...

class _VSCallback_std_FrameEval_eval_1(Protocol):
    def __call__(self, *, n: int, f: VideoFrame) -> VideoNode: ...

class _VSCallback_std_FrameEval_eval_2(Protocol):
    def __call__(self, *, n: int, f: list[VideoFrame]) -> VideoNode: ...

class _VSCallback_std_FrameEval_eval_3(Protocol):
    def __call__(self, *, n: int, f: VideoFrame | list[VideoFrame]) -> VideoNode: ...

type _VSCallback_std_FrameEval_eval = (  # noqa: PYI047
    _VSCallback_std_FrameEval_eval_0
    | _VSCallback_std_FrameEval_eval_1
    | _VSCallback_std_FrameEval_eval_2
    | _VSCallback_std_FrameEval_eval_3
)

class _VSCallback_std_Lut_function_0(Protocol):
    def __call__(self, *, x: int) -> _IntLike: ...

class _VSCallback_std_Lut_function_1(Protocol):
    def __call__(self, *, x: float) -> _FloatLike: ...

type _VSCallback_std_Lut_function = _VSCallback_std_Lut_function_0 | _VSCallback_std_Lut_function_1  # noqa: PYI047

class _VSCallback_std_Lut2_function_0(Protocol):
    def __call__(self, *, x: int, y: int) -> _IntLike: ...

class _VSCallback_std_Lut2_function_1(Protocol):
    def __call__(self, *, x: float, y: float) -> _FloatLike: ...

type _VSCallback_std_Lut2_function = _VSCallback_std_Lut2_function_0 | _VSCallback_std_Lut2_function_1  # noqa: PYI047

class _VSCallback_std_ModifyFrame_selector_0(Protocol):
    def __call__(self, *, n: int, f: VideoFrame) -> VideoFrame: ...

class _VSCallback_std_ModifyFrame_selector_1(Protocol):
    def __call__(self, *, n: int, f: list[VideoFrame]) -> VideoFrame: ...

class _VSCallback_std_ModifyFrame_selector_2(Protocol):
    def __call__(self, *, n: int, f: VideoFrame | list[VideoFrame]) -> VideoFrame: ...

type _VSCallback_std_ModifyFrame_selector = (  # noqa: PYI047
    _VSCallback_std_ModifyFrame_selector_0
    | _VSCallback_std_ModifyFrame_selector_1
    | _VSCallback_std_ModifyFrame_selector_2
)

class _VSCallback_resize2_Custom_custom_kernel(Protocol):
    def __call__(self, *, x: float) -> _FloatLike: ...

class LogHandle: ...

class PythonVSScriptLoggingBridge(Handler):
    def __init__(self, parent: StreamHandler[TextIO], level: int | str = ...) -> None: ...
    def emit(self, record: LogRecord) -> None: ...

class Error(Exception):
    value: Any
    def __init__(self, value: Any) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

# Environment SubSystem
@final
class EnvironmentData: ...

class EnvironmentPolicy:
    def on_policy_registered(self, special_api: EnvironmentPolicyAPI) -> None: ...
    def on_policy_cleared(self) -> None: ...
    def get_current_environment(self) -> EnvironmentData | None: ...
    def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData | None: ...
    def is_alive(self, environment: EnvironmentData) -> bool: ...

@final
class StandaloneEnvironmentPolicy:
    def on_policy_registered(self, api: EnvironmentPolicyAPI) -> None: ...
    def on_policy_cleared(self) -> None: ...
    def get_current_environment(self) -> EnvironmentData: ...
    def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData: ...
    def is_alive(self, environment: EnvironmentData) -> bool: ...
    def _on_log_message(self, level: MessageType, msg: str) -> None: ...

@final
class VSScriptEnvironmentPolicy:
    def on_policy_registered(self, policy_api: EnvironmentPolicyAPI) -> None: ...
    def on_policy_cleared(self) -> None: ...
    def get_current_environment(self) -> EnvironmentData | None: ...
    def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData | None: ...
    def is_alive(self, environment: EnvironmentData) -> bool: ...

@final
class EnvironmentPolicyAPI:
    def wrap_environment(self, environment_data: EnvironmentData) -> Environment: ...
    def create_environment(self, flags: _IntLike = 0) -> EnvironmentData: ...
    def set_logger(self, env: EnvironmentData, logger: Callable[[int, str], None]) -> None: ...
    def get_vapoursynth_api(self, version: int) -> c_void_p: ...
    def get_core_ptr(self, environment_data: EnvironmentData) -> c_void_p: ...
    def destroy_environment(self, env: EnvironmentData) -> None: ...
    def unregister_policy(self) -> None: ...

def register_policy(policy: EnvironmentPolicy) -> None: ...
def has_policy() -> bool: ...
def register_on_destroy(callback: Callable[..., None]) -> None: ...
def unregister_on_destroy(callback: Callable[..., None]) -> None: ...
def _try_enable_introspection(version: int | None = None) -> bool: ...
@final
class _FastManager:
    def __enter__(self) -> None: ...
    def __exit__(self, *_: object) -> None: ...

class Environment:
    env: Final[ReferenceType[EnvironmentData]]
    def __repr__(self) -> str: ...
    @overload
    def __eq__(self, other: Environment) -> bool: ...
    @overload
    def __eq__(self, other: object) -> bool: ...
    @property
    def alive(self) -> bool: ...
    @property
    def single(self) -> bool: ...
    @classmethod
    def is_single(cls) -> bool: ...
    @property
    def env_id(self) -> int: ...
    @property
    def active(self) -> bool: ...
    def copy(self) -> Self: ...
    def use(self) -> _FastManager: ...

class Local:
    def __getattr__(self, key: str) -> Any: ...
    def __setattr__(self, key: str, value: Any) -> None: ...
    def __delattr__(self, key: str) -> None: ...

def get_current_environment() -> Environment: ...

class CoreTimings:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def enabled(self) -> bool: ...
    @enabled.setter
    def enabled(self, enabled: bool) -> bool: ...
    @property
    def freed_nodes(self) -> bool: ...
    @freed_nodes.setter
    def freed_nodes(self, value: Literal[0]) -> bool: ...

# VapourSynth & plugin versioning

class VapourSynthVersion(NamedTuple):
    release_major: int
    release_minor: int
    def __str__(self) -> str: ...

class VapourSynthAPIVersion(NamedTuple):
    api_major: int
    api_minor: int
    def __str__(self) -> str: ...

__version__: VapourSynthVersion
__api_version__: VapourSynthAPIVersion

# Vapoursynth constants from vapoursynth.pyx

class MediaType(IntEnum):
    VIDEO = ...
    AUDIO = ...

VIDEO: Literal[MediaType.VIDEO]
AUDIO: Literal[MediaType.AUDIO]

class ColorFamily(IntEnum):
    UNDEFINED = ...
    GRAY = ...
    RGB = ...
    YUV = ...

UNDEFINED: Literal[ColorFamily.UNDEFINED]
GRAY: Literal[ColorFamily.GRAY]
RGB: Literal[ColorFamily.RGB]
YUV: Literal[ColorFamily.YUV]

class SampleType(IntEnum):
    INTEGER = ...
    FLOAT = ...

INTEGER: Literal[SampleType.INTEGER]
FLOAT: Literal[SampleType.FLOAT]

class PresetVideoFormat(IntEnum):
    NONE = ...

    GRAY8 = ...
    GRAY9 = ...
    GRAY10 = ...
    GRAY12 = ...
    GRAY14 = ...
    GRAY16 = ...
    GRAY32 = ...

    GRAYH = ...
    GRAYS = ...

    YUV420P8 = ...
    YUV422P8 = ...
    YUV444P8 = ...
    YUV410P8 = ...
    YUV411P8 = ...
    YUV440P8 = ...

    YUV420P9 = ...
    YUV422P9 = ...
    YUV444P9 = ...

    YUV420P10 = ...
    YUV422P10 = ...
    YUV444P10 = ...

    YUV420P12 = ...
    YUV422P12 = ...
    YUV444P12 = ...

    YUV420P14 = ...
    YUV422P14 = ...
    YUV444P14 = ...

    YUV420P16 = ...
    YUV422P16 = ...
    YUV444P16 = ...

    YUV420PH = ...
    YUV420PS = ...

    YUV422PH = ...
    YUV422PS = ...

    YUV444PH = ...
    YUV444PS = ...

    RGB24 = ...
    RGB27 = ...
    RGB30 = ...
    RGB36 = ...
    RGB42 = ...
    RGB48 = ...

    RGBH = ...
    RGBS = ...

NONE: Literal[PresetVideoFormat.NONE]

GRAY8: Literal[PresetVideoFormat.GRAY8]
GRAY9: Literal[PresetVideoFormat.GRAY9]
GRAY10: Literal[PresetVideoFormat.GRAY10]
GRAY12: Literal[PresetVideoFormat.GRAY12]
GRAY14: Literal[PresetVideoFormat.GRAY14]
GRAY16: Literal[PresetVideoFormat.GRAY16]
GRAY32: Literal[PresetVideoFormat.GRAY32]

GRAYH: Literal[PresetVideoFormat.GRAYH]
GRAYS: Literal[PresetVideoFormat.GRAYS]

YUV420P8: Literal[PresetVideoFormat.YUV420P8]
YUV422P8: Literal[PresetVideoFormat.YUV422P8]
YUV444P8: Literal[PresetVideoFormat.YUV444P8]
YUV410P8: Literal[PresetVideoFormat.YUV410P8]
YUV411P8: Literal[PresetVideoFormat.YUV411P8]
YUV440P8: Literal[PresetVideoFormat.YUV440P8]

YUV420P9: Literal[PresetVideoFormat.YUV420P9]
YUV422P9: Literal[PresetVideoFormat.YUV422P9]
YUV444P9: Literal[PresetVideoFormat.YUV444P9]

YUV420P10: Literal[PresetVideoFormat.YUV420P10]
YUV422P10: Literal[PresetVideoFormat.YUV422P10]
YUV444P10: Literal[PresetVideoFormat.YUV444P10]

YUV420P12: Literal[PresetVideoFormat.YUV420P12]
YUV422P12: Literal[PresetVideoFormat.YUV422P12]
YUV444P12: Literal[PresetVideoFormat.YUV444P12]

YUV420P14: Literal[PresetVideoFormat.YUV420P14]
YUV422P14: Literal[PresetVideoFormat.YUV422P14]
YUV444P14: Literal[PresetVideoFormat.YUV444P14]

YUV420P16: Literal[PresetVideoFormat.YUV420P16]
YUV422P16: Literal[PresetVideoFormat.YUV422P16]
YUV444P16: Literal[PresetVideoFormat.YUV444P16]

YUV420PH: Literal[PresetVideoFormat.YUV420PH]
YUV420PS: Literal[PresetVideoFormat.YUV420PS]

YUV422PH: Literal[PresetVideoFormat.YUV422PH]
YUV422PS: Literal[PresetVideoFormat.YUV422PS]

YUV444PH: Literal[PresetVideoFormat.YUV444PH]
YUV444PS: Literal[PresetVideoFormat.YUV444PS]

RGB24: Literal[PresetVideoFormat.RGB24]
RGB27: Literal[PresetVideoFormat.RGB27]
RGB30: Literal[PresetVideoFormat.RGB30]
RGB36: Literal[PresetVideoFormat.RGB36]
RGB42: Literal[PresetVideoFormat.RGB42]
RGB48: Literal[PresetVideoFormat.RGB48]

RGBH: Literal[PresetVideoFormat.RGBH]
RGBS: Literal[PresetVideoFormat.RGBS]

class FilterMode(IntEnum):
    PARALLEL = ...
    PARALLEL_REQUESTS = ...
    UNORDERED = ...
    FRAME_STATE = ...

PARALLEL: Literal[FilterMode.PARALLEL]
PARALLEL_REQUESTS: Literal[FilterMode.PARALLEL_REQUESTS]
UNORDERED: Literal[FilterMode.UNORDERED]
FRAME_STATE: Literal[FilterMode.FRAME_STATE]

class AudioChannels(IntEnum):
    FRONT_LEFT = ...
    FRONT_RIGHT = ...
    FRONT_CENTER = ...
    LOW_FREQUENCY = ...
    BACK_LEFT = ...
    BACK_RIGHT = ...
    FRONT_LEFT_OF_CENTER = ...
    FRONT_RIGHT_OF_CENTER = ...
    BACK_CENTER = ...
    SIDE_LEFT = ...
    SIDE_RIGHT = ...
    TOP_CENTER = ...
    TOP_FRONT_LEFT = ...
    TOP_FRONT_CENTER = ...
    TOP_FRONT_RIGHT = ...
    TOP_BACK_LEFT = ...
    TOP_BACK_CENTER = ...
    TOP_BACK_RIGHT = ...
    STEREO_LEFT = ...
    STEREO_RIGHT = ...
    WIDE_LEFT = ...
    WIDE_RIGHT = ...
    SURROUND_DIRECT_LEFT = ...
    SURROUND_DIRECT_RIGHT = ...
    LOW_FREQUENCY2 = ...

FRONT_LEFT: Literal[AudioChannels.FRONT_LEFT]
FRONT_RIGHT: Literal[AudioChannels.FRONT_RIGHT]
FRONT_CENTER: Literal[AudioChannels.FRONT_CENTER]
LOW_FREQUENCY: Literal[AudioChannels.LOW_FREQUENCY]
BACK_LEFT: Literal[AudioChannels.BACK_LEFT]
BACK_RIGHT: Literal[AudioChannels.BACK_RIGHT]
FRONT_LEFT_OF_CENTER: Literal[AudioChannels.FRONT_LEFT_OF_CENTER]
FRONT_RIGHT_OF_CENTER: Literal[AudioChannels.FRONT_RIGHT_OF_CENTER]
BACK_CENTER: Literal[AudioChannels.BACK_CENTER]
SIDE_LEFT: Literal[AudioChannels.SIDE_LEFT]
SIDE_RIGHT: Literal[AudioChannels.SIDE_RIGHT]
TOP_CENTER: Literal[AudioChannels.TOP_CENTER]
TOP_FRONT_LEFT: Literal[AudioChannels.TOP_FRONT_LEFT]
TOP_FRONT_CENTER: Literal[AudioChannels.TOP_FRONT_CENTER]
TOP_FRONT_RIGHT: Literal[AudioChannels.TOP_FRONT_RIGHT]
TOP_BACK_LEFT: Literal[AudioChannels.TOP_BACK_LEFT]
TOP_BACK_CENTER: Literal[AudioChannels.TOP_BACK_CENTER]
TOP_BACK_RIGHT: Literal[AudioChannels.TOP_BACK_RIGHT]
STEREO_LEFT: Literal[AudioChannels.STEREO_LEFT]
STEREO_RIGHT: Literal[AudioChannels.STEREO_RIGHT]
WIDE_LEFT: Literal[AudioChannels.WIDE_LEFT]
WIDE_RIGHT: Literal[AudioChannels.WIDE_RIGHT]
SURROUND_DIRECT_LEFT: Literal[AudioChannels.SURROUND_DIRECT_LEFT]
SURROUND_DIRECT_RIGHT: Literal[AudioChannels.SURROUND_DIRECT_RIGHT]
LOW_FREQUENCY2: Literal[AudioChannels.LOW_FREQUENCY2]

class MessageType(IntFlag):
    MESSAGE_TYPE_DEBUG = ...
    MESSAGE_TYPE_INFORMATION = ...
    MESSAGE_TYPE_WARNING = ...
    MESSAGE_TYPE_CRITICAL = ...
    MESSAGE_TYPE_FATAL = ...

MESSAGE_TYPE_DEBUG: Literal[MessageType.MESSAGE_TYPE_DEBUG]
MESSAGE_TYPE_INFORMATION: Literal[MessageType.MESSAGE_TYPE_INFORMATION]
MESSAGE_TYPE_WARNING: Literal[MessageType.MESSAGE_TYPE_WARNING]
MESSAGE_TYPE_CRITICAL: Literal[MessageType.MESSAGE_TYPE_CRITICAL]
MESSAGE_TYPE_FATAL: Literal[MessageType.MESSAGE_TYPE_FATAL]

class CoreCreationFlags(IntFlag):
    ENABLE_GRAPH_INSPECTION = ...
    DISABLE_AUTO_LOADING = ...
    DISABLE_LIBRARY_UNLOADING = ...

ENABLE_GRAPH_INSPECTION: Literal[CoreCreationFlags.ENABLE_GRAPH_INSPECTION]
DISABLE_AUTO_LOADING: Literal[CoreCreationFlags.DISABLE_AUTO_LOADING]
DISABLE_LIBRARY_UNLOADING: Literal[CoreCreationFlags.DISABLE_LIBRARY_UNLOADING]

# Vapoursynth constants from vsconstants.pyd

class ColorRange(IntEnum):
    RANGE_FULL = ...
    RANGE_LIMITED = ...

RANGE_FULL: Literal[ColorRange.RANGE_FULL]
RANGE_LIMITED: Literal[ColorRange.RANGE_LIMITED]

class ChromaLocation(IntEnum):
    CHROMA_LEFT = ...
    CHROMA_CENTER = ...
    CHROMA_TOP_LEFT = ...
    CHROMA_TOP = ...
    CHROMA_BOTTOM_LEFT = ...
    CHROMA_BOTTOM = ...

CHROMA_LEFT: Literal[ChromaLocation.CHROMA_LEFT]
CHROMA_CENTER: Literal[ChromaLocation.CHROMA_CENTER]
CHROMA_TOP_LEFT: Literal[ChromaLocation.CHROMA_TOP_LEFT]
CHROMA_TOP: Literal[ChromaLocation.CHROMA_TOP]
CHROMA_BOTTOM_LEFT: Literal[ChromaLocation.CHROMA_BOTTOM_LEFT]
CHROMA_BOTTOM: Literal[ChromaLocation.CHROMA_BOTTOM]

class FieldBased(IntEnum):
    FIELD_PROGRESSIVE = ...
    FIELD_TOP = ...
    FIELD_BOTTOM = ...

FIELD_PROGRESSIVE: Literal[FieldBased.FIELD_PROGRESSIVE]
FIELD_TOP: Literal[FieldBased.FIELD_TOP]
FIELD_BOTTOM: Literal[FieldBased.FIELD_BOTTOM]

class MatrixCoefficients(IntEnum):
    MATRIX_RGB = ...
    MATRIX_BT709 = ...
    MATRIX_UNSPECIFIED = ...
    MATRIX_FCC = ...
    MATRIX_BT470_BG = ...
    MATRIX_ST170_M = ...
    MATRIX_ST240_M = ...
    MATRIX_YCGCO = ...
    MATRIX_BT2020_NCL = ...
    MATRIX_BT2020_CL = ...
    MATRIX_CHROMATICITY_DERIVED_NCL = ...
    MATRIX_CHROMATICITY_DERIVED_CL = ...
    MATRIX_ICTCP = ...

MATRIX_RGB: Literal[MatrixCoefficients.MATRIX_RGB]
MATRIX_BT709: Literal[MatrixCoefficients.MATRIX_BT709]
MATRIX_UNSPECIFIED: Literal[MatrixCoefficients.MATRIX_UNSPECIFIED]
MATRIX_FCC: Literal[MatrixCoefficients.MATRIX_FCC]
MATRIX_BT470_BG: Literal[MatrixCoefficients.MATRIX_BT470_BG]
MATRIX_ST170_M: Literal[MatrixCoefficients.MATRIX_ST170_M]
MATRIX_ST240_M: Literal[MatrixCoefficients.MATRIX_ST240_M]
MATRIX_YCGCO: Literal[MatrixCoefficients.MATRIX_YCGCO]
MATRIX_BT2020_NCL: Literal[MatrixCoefficients.MATRIX_BT2020_NCL]
MATRIX_BT2020_CL: Literal[MatrixCoefficients.MATRIX_BT2020_CL]
MATRIX_CHROMATICITY_DERIVED_NCL: Literal[MatrixCoefficients.MATRIX_CHROMATICITY_DERIVED_NCL]
MATRIX_CHROMATICITY_DERIVED_CL: Literal[MatrixCoefficients.MATRIX_CHROMATICITY_DERIVED_CL]
MATRIX_ICTCP: Literal[MatrixCoefficients.MATRIX_ICTCP]

class TransferCharacteristics(IntEnum):
    TRANSFER_BT709 = ...
    TRANSFER_UNSPECIFIED = ...
    TRANSFER_BT470_M = ...
    TRANSFER_BT470_BG = ...
    TRANSFER_BT601 = ...
    TRANSFER_ST240_M = ...
    TRANSFER_LINEAR = ...
    TRANSFER_LOG_100 = ...
    TRANSFER_LOG_316 = ...
    TRANSFER_IEC_61966_2_4 = ...
    TRANSFER_IEC_61966_2_1 = ...
    TRANSFER_BT2020_10 = ...
    TRANSFER_BT2020_12 = ...
    TRANSFER_ST2084 = ...
    TRANSFER_ST428 = ...
    TRANSFER_ARIB_B67 = ...

TRANSFER_BT709: Literal[TransferCharacteristics.TRANSFER_BT709]
TRANSFER_UNSPECIFIED: Literal[TransferCharacteristics.TRANSFER_UNSPECIFIED]
TRANSFER_BT470_M: Literal[TransferCharacteristics.TRANSFER_BT470_M]
TRANSFER_BT470_BG: Literal[TransferCharacteristics.TRANSFER_BT470_BG]
TRANSFER_BT601: Literal[TransferCharacteristics.TRANSFER_BT601]
TRANSFER_ST240_M: Literal[TransferCharacteristics.TRANSFER_ST240_M]
TRANSFER_LINEAR: Literal[TransferCharacteristics.TRANSFER_LINEAR]
TRANSFER_LOG_100: Literal[TransferCharacteristics.TRANSFER_LOG_100]
TRANSFER_LOG_316: Literal[TransferCharacteristics.TRANSFER_LOG_316]
TRANSFER_IEC_61966_2_4: Literal[TransferCharacteristics.TRANSFER_IEC_61966_2_4]
TRANSFER_IEC_61966_2_1: Literal[TransferCharacteristics.TRANSFER_IEC_61966_2_1]
TRANSFER_BT2020_10: Literal[TransferCharacteristics.TRANSFER_BT2020_10]
TRANSFER_BT2020_12: Literal[TransferCharacteristics.TRANSFER_BT2020_12]
TRANSFER_ST2084: Literal[TransferCharacteristics.TRANSFER_ST2084]
TRANSFER_ST428: Literal[TransferCharacteristics.TRANSFER_ST428]
TRANSFER_ARIB_B67: Literal[TransferCharacteristics.TRANSFER_ARIB_B67]

class ColorPrimaries(IntEnum):
    PRIMARIES_BT709 = ...
    PRIMARIES_UNSPECIFIED = ...
    PRIMARIES_BT470_M = ...
    PRIMARIES_BT470_BG = ...
    PRIMARIES_ST170_M = ...
    PRIMARIES_ST240_M = ...
    PRIMARIES_FILM = ...
    PRIMARIES_BT2020 = ...
    PRIMARIES_ST428 = ...
    PRIMARIES_ST431_2 = ...
    PRIMARIES_ST432_1 = ...
    PRIMARIES_EBU3213_E = ...

PRIMARIES_BT709: Literal[ColorPrimaries.PRIMARIES_BT709]
PRIMARIES_UNSPECIFIED: Literal[ColorPrimaries.PRIMARIES_UNSPECIFIED]
PRIMARIES_BT470_M: Literal[ColorPrimaries.PRIMARIES_BT470_M]
PRIMARIES_BT470_BG: Literal[ColorPrimaries.PRIMARIES_BT470_BG]
PRIMARIES_ST170_M: Literal[ColorPrimaries.PRIMARIES_ST170_M]
PRIMARIES_ST240_M: Literal[ColorPrimaries.PRIMARIES_ST240_M]
PRIMARIES_FILM: Literal[ColorPrimaries.PRIMARIES_FILM]
PRIMARIES_BT2020: Literal[ColorPrimaries.PRIMARIES_BT2020]
PRIMARIES_ST428: Literal[ColorPrimaries.PRIMARIES_ST428]
PRIMARIES_ST431_2: Literal[ColorPrimaries.PRIMARIES_ST431_2]
PRIMARIES_ST432_1: Literal[ColorPrimaries.PRIMARIES_ST432_1]
PRIMARIES_EBU3213_E: Literal[ColorPrimaries.PRIMARIES_EBU3213_E]

class _VideoFormatDict(TypedDict):
    id: int
    name: str
    color_family: ColorFamily
    sample_type: SampleType
    bits_per_sample: Literal[
        8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
    ]
    bytes_per_sample: int
    subsampling_w: Literal[0, 1, 2, 3, 4]
    subsampling_h: Literal[0, 1, 2, 3, 4]
    num_planes: Literal[1, 3]

class VideoFormat:
    id: Final[int]
    name: Final[str]
    color_family: Final[ColorFamily]
    sample_type: Final[SampleType]
    bits_per_sample: Final[
        Literal[8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    ]
    bytes_per_sample: Final[int]
    subsampling_w: Final[Literal[0, 1, 2, 3, 4]]
    subsampling_h: Final[Literal[0, 1, 2, 3, 4]]
    num_planes: Final[Literal[1, 3]]
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def replace(
        self,
        *,
        color_family: ColorFamily = ...,
        sample_type: SampleType = ...,
        bits_per_sample: _IntLike = ...,
        subsampling_w: _IntLike = ...,
        subsampling_h: _IntLike = ...,
    ) -> Self: ...
    def _as_dict(self) -> _VideoFormatDict: ...

# Behave like a Collection
class ChannelLayout(int):
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __contains__(self, layout: AudioChannels) -> bool: ...
    def __iter__(self) -> Iterator[AudioChannels]: ...
    def __len__(self) -> int: ...

type _PropValue = (
    int
    | float
    | str
    | bytes
    | RawFrame
    | VideoFrame
    | AudioFrame
    | RawNode
    | VideoNode
    | AudioNode
    | Callable[..., Any]
    | list[int]
    | list[float]
    | list[str]
    | list[bytes]
    | list[RawFrame]
    | list[VideoFrame]
    | list[AudioFrame]
    | list[RawNode]
    | list[VideoNode]
    | list[AudioNode]
    | list[Callable[..., Any]]
)

# Only the _PropValue types are allowed in FrameProps but passing _VSValue is allowed.
# Just keep in mind that _SupportsIter and _GetItemIterable will only yield their keys if they're Mapping-like.
# Consider storing Mapping-likes as two separate props. One for the keys and one for the values as list.
class FrameProps(MutableMapping[str, _PropValue]):
    def __repr__(self) -> str: ...
    def __dir__(self) -> list[str]: ...
    def __getitem__(self, name: str) -> _PropValue: ...
    def __setitem__(self, name: str, value: _VSValue) -> None: ...
    def __delitem__(self, name: str) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def __setattr__(self, name: str, value: _VSValue) -> None: ...
    def __delattr__(self, name: str) -> None: ...
    def __getattr__(self, name: str) -> _PropValue: ...
    @overload
    def setdefault(self, key: str, default: Literal[0] = 0, /) -> _PropValue | Literal[0]: ...
    @overload
    def setdefault(self, key: str, default: _VSValue, /) -> _PropValue: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def copy(self) -> dict[str, _PropValue]: ...

class FuncData:
    def __call__(self, **kwargs: Any) -> Any: ...

class Func:
    def __call__(self, **kwargs: Any) -> Any: ...

class Function:
    plugin: Final[Plugin]
    name: Final[str]
    signature: Final[str]
    return_signature: Final[str]
    def __repr__(self) -> str: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    @property
    def __signature__(self) -> Signature: ...

class PluginVersion(NamedTuple):
    major: int
    minor: int

class Plugin:
    identifier: Final[str]
    namespace: Final[str]
    name: Final[str]

    def __repr__(self) -> str: ...
    def __dir__(self) -> list[str]: ...
    def __getattr__(self, name: str) -> Function: ...
    @property
    def version(self) -> PluginVersion: ...
    @property
    def plugin_path(self) -> str: ...
    def functions(self) -> Iterator[Function]: ...

_VSPlugin = Plugin
_VSFunction = Function

class _Wrapper:
    class Function[**P, R](_VSFunction):
        def __init__[PluginT: Plugin](self, function: Callable[Concatenate[PluginT, P], R]) -> None: ...
        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

class _Wrapper_Core_bound_FrameEval:
    class Function(_VSFunction):
        def __init__[PluginT: Plugin](self, function: Callable[Concatenate[PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(
            self,
            clip: VideoNode,
            eval: _VSCallback_std_FrameEval_eval_0,
            prop_src: None = None,
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            clip: VideoNode,
            eval: _VSCallback_std_FrameEval_eval_1,
            prop_src: VideoNode,
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            clip: VideoNode,
            eval: _VSCallback_std_FrameEval_eval_2,
            prop_src: _SequenceLike[VideoNode],
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            clip: VideoNode,
            eval: _VSCallback_std_FrameEval_eval_3,
            prop_src: VideoNode | _SequenceLike[VideoNode],
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            clip: VideoNode,
            eval: _VSCallback_std_FrameEval_eval,
            prop_src: VideoNode | _SequenceLike[VideoNode] | None,
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...

class _Wrapper_VideoNode_bound_FrameEval:
    class Function(_VSFunction):
        def __init__[PluginT: Plugin](self, function: Callable[Concatenate[PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(
            self,
            eval: _VSCallback_std_FrameEval_eval_0,
            prop_src: None = None,
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            eval: _VSCallback_std_FrameEval_eval_1,
            prop_src: VideoNode,
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            eval: _VSCallback_std_FrameEval_eval_2,
            prop_src: _SequenceLike[VideoNode],
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            eval: _VSCallback_std_FrameEval_eval_3,
            prop_src: VideoNode | _SequenceLike[VideoNode],
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            eval: _VSCallback_std_FrameEval_eval,
            prop_src: VideoNode | _SequenceLike[VideoNode] | None,
            clip_src: VideoNode | _SequenceLike[VideoNode] | None = None,
        ) -> VideoNode: ...

class _Wrapper_Core_bound_ModifyFrame:
    class Function(_VSFunction):
        def __init__[PluginT: Plugin](self, function: Callable[Concatenate[PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(
            self, clip: VideoNode, clips: VideoNode, selector: _VSCallback_std_ModifyFrame_selector_0
        ) -> VideoNode: ...
        @overload
        def __call__(
            self, clip: VideoNode, clips: _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector_1
        ) -> VideoNode: ...
        @overload
        def __call__(
            self,
            clip: VideoNode,
            clips: VideoNode | _SequenceLike[VideoNode],
            selector: _VSCallback_std_ModifyFrame_selector,
        ) -> VideoNode: ...

class _Wrapper_VideoNode_bound_ModifyFrame:
    class Function(_VSFunction):
        def __init__[PluginT: Plugin](self, function: Callable[Concatenate[PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(self, clips: VideoNode, selector: _VSCallback_std_ModifyFrame_selector_0) -> VideoNode: ...
        @overload
        def __call__(
            self, clips: _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector_1
        ) -> VideoNode: ...
        @overload
        def __call__(
            self, clips: VideoNode | _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector
        ) -> VideoNode: ...

class FramePtr: ...

# These memoryview-likes don't exist at runtime.
class _video_view(memoryview):  # type: ignore[misc]
    def __getitem__(self, index: tuple[int, int]) -> float: ...  # type: ignore[override]
    def __setitem__(self, index: tuple[int, int], other: float) -> None: ...  # type: ignore[override]
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def strides(self) -> tuple[int, int]: ...
    @property
    def ndim(self) -> Literal[2]: ...
    @property
    def obj(self) -> FramePtr: ...  # type: ignore[override]
    def tolist(self) -> list[float]: ...  # type: ignore[override]

class _audio_view(memoryview):  # type: ignore[misc]
    def __getitem__(self, index: int) -> float: ...  # type: ignore[override]
    def __setitem__(self, index: int, other: float) -> None: ...  # type: ignore[override]
    @property
    def shape(self) -> tuple[int]: ...
    @property
    def strides(self) -> tuple[int]: ...
    @property
    def ndim(self) -> Literal[1]: ...
    @property
    def obj(self) -> FramePtr: ...  # type: ignore[override]
    def tolist(self) -> list[float]: ...  # type: ignore[override]

class RawFrame:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self, exc: type[BaseException] | None = None, val: BaseException | None = None, tb: TracebackType | None = None
    ) -> bool | None: ...
    def __getitem__(self, index: SupportsIndex) -> memoryview: ...
    def __len__(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    @property
    def props(self) -> FrameProps: ...
    @props.setter
    def props(self, new_props: _SupportsKeysAndGetItem[str, _VSValue]) -> None: ...
    @property
    def readonly(self) -> bool: ...
    def copy(self) -> Self: ...
    def close(self) -> None: ...
    def get_write_ptr(self, plane: _IntLike) -> c_void_p: ...
    def get_read_ptr(self, plane: _IntLike) -> c_void_p: ...
    def get_stride(self, plane: _IntLike) -> int: ...

# Behave like a Sequence
class VideoFrame(RawFrame):
    format: Final[VideoFormat]
    width: Final[int]
    height: Final[int]

    def __getitem__(self, index: SupportsIndex) -> _video_view: ...
    def readchunks(self) -> Iterator[_video_view]: ...

# Behave like a Sequence
class AudioFrame(RawFrame):
    sample_type: Final[SampleType]
    bits_per_sample: Final[int]
    bytes_per_sample: Final[int]
    channel_layout: Final[int]
    num_channels: Final[int]

    def __getitem__(self, index: SupportsIndex) -> _audio_view: ...
    @property
    def channels(self) -> ChannelLayout: ...

class RawNode:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __dir__(self) -> list[str]: ...
    def __getitem__(self, index: int | slice[int | None, int | None, int | None]) -> Self: ...
    def __len__(self) -> int: ...
    def __add__(self, other: Self) -> Self: ...
    def __mul__(self, other: int) -> Self: ...
    def __getattr__(self, name: str) -> Plugin: ...
    @property
    def node_name(self) -> str: ...
    @property
    def timings(self) -> int: ...
    @timings.setter
    def timings(self, value: Literal[0]) -> None: ...
    @property
    def mode(self) -> FilterMode: ...
    @property
    def dependencies(self) -> tuple[Self, ...]: ...
    @property
    def _name(self) -> str: ...
    @property
    def _inputs(self) -> dict[str, _VSValue]: ...
    def get_frame(self, n: _IntLike) -> RawFrame: ...
    @overload
    def get_frame_async(self, n: _IntLike) -> Future[RawFrame]: ...
    @overload
    def get_frame_async(self, n: _IntLike, cb: Callable[[RawFrame | None, Exception | None], None]) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[RawFrame]: ...
    def set_output(self, index: _IntLike = 0) -> None: ...
    def clear_cache(self) -> None: ...
    def is_inspectable(self, version: int | None = None) -> bool: ...

type _CurrentFrame = int
type _TotalFrames = int

# Behave like a Sequence
class VideoNode(RawNode):
    format: Final[VideoFormat]
    width: Final[int]
    height: Final[int]
    num_frames: Final[int]
    fps_num: Final[int]
    fps_den: Final[int]
    fps: Final[Fraction]
    def get_frame(self, n: _IntLike) -> VideoFrame: ...
    @overload  # type: ignore[override]
    def get_frame_async(self, n: _IntLike) -> Future[VideoFrame]: ...
    @overload
    def get_frame_async(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, n: _IntLike, cb: Callable[[VideoFrame | None, Exception | None], None]
    ) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[VideoFrame]: ...
    def set_output(self, index: _IntLike = 0, alpha: Self | None = None, alt_output: Literal[0, 1, 2] = 0) -> None: ...
    def output(
        self,
        fileobj: IO[bytes],
        y4m: bool = False,
        progress_update: Callable[[_CurrentFrame, _TotalFrames], None] | None = None,
        prefetch: int = 0,
        backlog: int = -1,
    ) -> None: ...

# <plugins/bound/VideoNode>
# <attribute/VideoNode_bound/adg>
    adg: Final[_adg._VideoNode_bound.Plugin]
    """Adaptive grain"""
# </attribute/VideoNode_bound/adg>
# <attribute/VideoNode_bound/akarin>
    akarin: Final[_akarin._VideoNode_bound.Plugin]
    """Akarin's Experimental Filters"""
# </attribute/VideoNode_bound/akarin>
# <attribute/VideoNode_bound/awarp>
    awarp: Final[_awarp._VideoNode_bound.Plugin]
    """AWarp filter from AWarpSharp2"""
# </attribute/VideoNode_bound/awarp>
# <attribute/VideoNode_bound/bilateralgpu>
    bilateralgpu: Final[_bilateralgpu._VideoNode_bound.Plugin]
    """Bilateral filter using CUDA"""
# </attribute/VideoNode_bound/bilateralgpu>
# <attribute/VideoNode_bound/bilateralgpu_rtc>
    bilateralgpu_rtc: Final[_bilateralgpu_rtc._VideoNode_bound.Plugin]
    """Bilateral filter using CUDA (NVRTC)"""
# </attribute/VideoNode_bound/bilateralgpu_rtc>
# <attribute/VideoNode_bound/bm3d>
    bm3d: Final[_bm3d._VideoNode_bound.Plugin]
    """Implementation of BM3D denoising filter for VapourSynth."""
# </attribute/VideoNode_bound/bm3d>
# <attribute/VideoNode_bound/bm3dcpu>
    bm3dcpu: Final[_bm3dcpu._VideoNode_bound.Plugin]
    """BM3D algorithm implemented in AVX and AVX2 intrinsics"""
# </attribute/VideoNode_bound/bm3dcpu>
# <attribute/VideoNode_bound/bm3dcuda>
    bm3dcuda: Final[_bm3dcuda._VideoNode_bound.Plugin]
    """BM3D algorithm implemented in CUDA"""
# </attribute/VideoNode_bound/bm3dcuda>
# <attribute/VideoNode_bound/bm3dcuda_rtc>
    bm3dcuda_rtc: Final[_bm3dcuda_rtc._VideoNode_bound.Plugin]
    """BM3D algorithm implemented in CUDA (NVRTC)"""
# </attribute/VideoNode_bound/bm3dcuda_rtc>
# <attribute/VideoNode_bound/bwdif>
    bwdif: Final[_bwdif._VideoNode_bound.Plugin]
    """BobWeaver Deinterlacing Filter"""
# </attribute/VideoNode_bound/bwdif>
# <attribute/VideoNode_bound/cs>
    cs: Final[_cs._VideoNode_bound.Plugin]
    """carefulsource"""
# </attribute/VideoNode_bound/cs>
# <attribute/VideoNode_bound/dctf>
    dctf: Final[_dctf._VideoNode_bound.Plugin]
    """DCT/IDCT Frequency Suppressor"""
# </attribute/VideoNode_bound/dctf>
# <attribute/VideoNode_bound/deblock>
    deblock: Final[_deblock._VideoNode_bound.Plugin]
    """It does a deblocking of the picture, using the deblocking filter of h264"""
# </attribute/VideoNode_bound/deblock>
# <attribute/VideoNode_bound/descale>
    descale: Final[_descale._VideoNode_bound.Plugin]
    """Undo linear interpolation"""
# </attribute/VideoNode_bound/descale>
# <attribute/VideoNode_bound/dfttest>
    dfttest: Final[_dfttest._VideoNode_bound.Plugin]
    """2D/3D frequency domain denoiser"""
# </attribute/VideoNode_bound/dfttest>
# <attribute/VideoNode_bound/dfttest2_cpu>
    dfttest2_cpu: Final[_dfttest2_cpu._VideoNode_bound.Plugin]
    """DFTTest2 (CPU)"""
# </attribute/VideoNode_bound/dfttest2_cpu>
# <attribute/VideoNode_bound/dfttest2_cuda>
    dfttest2_cuda: Final[_dfttest2_cuda._VideoNode_bound.Plugin]
    """DFTTest2 (CUDA)"""
# </attribute/VideoNode_bound/dfttest2_cuda>
# <attribute/VideoNode_bound/dfttest2_nvrtc>
    dfttest2_nvrtc: Final[_dfttest2_nvrtc._VideoNode_bound.Plugin]
    """DFTTest2 (NVRTC)"""
# </attribute/VideoNode_bound/dfttest2_nvrtc>
# <attribute/VideoNode_bound/edgemasks>
    edgemasks: Final[_edgemasks._VideoNode_bound.Plugin]
    """Creates an edge mask using various operators"""
# </attribute/VideoNode_bound/edgemasks>
# <attribute/VideoNode_bound/eedi2>
    eedi2: Final[_eedi2._VideoNode_bound.Plugin]
    """EEDI2"""
# </attribute/VideoNode_bound/eedi2>
# <attribute/VideoNode_bound/eedi3m>
    eedi3m: Final[_eedi3m._VideoNode_bound.Plugin]
    """Enhanced Edge Directed Interpolation 3"""
# </attribute/VideoNode_bound/eedi3m>
# <attribute/VideoNode_bound/fmtc>
    fmtc: Final[_fmtc._VideoNode_bound.Plugin]
    """Format converter"""
# </attribute/VideoNode_bound/fmtc>
# <attribute/VideoNode_bound/hysteresis>
    hysteresis: Final[_hysteresis._VideoNode_bound.Plugin]
    """Hysteresis filter."""
# </attribute/VideoNode_bound/hysteresis>
# <attribute/VideoNode_bound/imwri>
    imwri: Final[_imwri._VideoNode_bound.Plugin]
    """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
# </attribute/VideoNode_bound/imwri>
# <attribute/VideoNode_bound/knlm>
    knlm: Final[_knlm._VideoNode_bound.Plugin]
    """KNLMeansCL for VapourSynth"""
# </attribute/VideoNode_bound/knlm>
# <attribute/VideoNode_bound/manipmv>
    manipmv: Final[_manipmv._VideoNode_bound.Plugin]
    """Manipulate Motion Vectors"""
# </attribute/VideoNode_bound/manipmv>
# <attribute/VideoNode_bound/mv>
    mv: Final[_mv._VideoNode_bound.Plugin]
    """MVTools v24"""
# </attribute/VideoNode_bound/mv>
# <attribute/VideoNode_bound/neo_f3kdb>
    neo_f3kdb: Final[_neo_f3kdb._VideoNode_bound.Plugin]
    """Neo F3KDB Deband Filter r10"""
# </attribute/VideoNode_bound/neo_f3kdb>
# <attribute/VideoNode_bound/nlm_cuda>
    nlm_cuda: Final[_nlm_cuda._VideoNode_bound.Plugin]
    """Non-local means denoise filter implemented in CUDA"""
# </attribute/VideoNode_bound/nlm_cuda>
# <attribute/VideoNode_bound/nlm_ispc>
    nlm_ispc: Final[_nlm_ispc._VideoNode_bound.Plugin]
    """Non-local means denoise filter implemented in ISPC"""
# </attribute/VideoNode_bound/nlm_ispc>
# <attribute/VideoNode_bound/noise>
    noise: Final[_noise._VideoNode_bound.Plugin]
    """Noise generator"""
# </attribute/VideoNode_bound/noise>
# <attribute/VideoNode_bound/placebo>
    placebo: Final[_placebo._VideoNode_bound.Plugin]
    """libplacebo plugin for VapourSynth"""
# </attribute/VideoNode_bound/placebo>
# <attribute/VideoNode_bound/resize>
    resize: Final[_resize._VideoNode_bound.Plugin]
    """VapourSynth Resize"""
# </attribute/VideoNode_bound/resize>
# <attribute/VideoNode_bound/resize2>
    resize2: Final[_resize2._VideoNode_bound.Plugin]
    """Built-in VapourSynth resizer based on zimg with some modifications."""
# </attribute/VideoNode_bound/resize2>
# <attribute/VideoNode_bound/sangnom>
    sangnom: Final[_sangnom._VideoNode_bound.Plugin]
    """VapourSynth Single Field Deinterlacer"""
# </attribute/VideoNode_bound/sangnom>
# <attribute/VideoNode_bound/scxvid>
    scxvid: Final[_scxvid._VideoNode_bound.Plugin]
    """VapourSynth Scxvid Plugin"""
# </attribute/VideoNode_bound/scxvid>
# <attribute/VideoNode_bound/sneedif>
    sneedif: Final[_sneedif._VideoNode_bound.Plugin]
    """Setsugen No Ensemble of Edge Directed Interpolation Functions"""
# </attribute/VideoNode_bound/sneedif>
# <attribute/VideoNode_bound/std>
    std: Final[_std._VideoNode_bound.Plugin]
    """VapourSynth Core Functions"""
# </attribute/VideoNode_bound/std>
# <attribute/VideoNode_bound/sub>
    sub: Final[_sub._VideoNode_bound.Plugin]
    """A subtitling filter based on libass and FFmpeg."""
# </attribute/VideoNode_bound/sub>
# <attribute/VideoNode_bound/tcanny>
    tcanny: Final[_tcanny._VideoNode_bound.Plugin]
    """Build an edge map using canny edge detection"""
# </attribute/VideoNode_bound/tcanny>
# <attribute/VideoNode_bound/text>
    text: Final[_text._VideoNode_bound.Plugin]
    """VapourSynth Text"""
# </attribute/VideoNode_bound/text>
# <attribute/VideoNode_bound/vivtc>
    vivtc: Final[_vivtc._VideoNode_bound.Plugin]
    """VFM"""
# </attribute/VideoNode_bound/vivtc>
# <attribute/VideoNode_bound/vszip>
    vszip: Final[_vszip._VideoNode_bound.Plugin]
    """VapourSynth Zig Image Process"""
# </attribute/VideoNode_bound/vszip>
# <attribute/VideoNode_bound/wnnm>
    wnnm: Final[_wnnm._VideoNode_bound.Plugin]
    """Weighted Nuclear Norm Minimization Denoiser"""
# </attribute/VideoNode_bound/wnnm>
# <attribute/VideoNode_bound/wwxd>
    wwxd: Final[_wwxd._VideoNode_bound.Plugin]
    """Scene change detection approximately like Xvid's"""
# </attribute/VideoNode_bound/wwxd>
# <attribute/VideoNode_bound/znedi3>
    znedi3: Final[_znedi3._VideoNode_bound.Plugin]
    """Neural network edge directed interpolation (3rd gen.)"""
# </attribute/VideoNode_bound/znedi3>
# <attribute/VideoNode_bound/zsmooth>
    zsmooth: Final[_zsmooth._VideoNode_bound.Plugin]
    """Smoothing functions in Zig"""
# </attribute/VideoNode_bound/zsmooth>
# </plugins/bound/VideoNode>

# Behave like a Sequence
class AudioNode(RawNode):
    sample_type: Final[SampleType]
    bits_per_sample: Final[int]
    bytes_per_sample: Final[int]
    channel_layout: Final[int]
    num_channels: Final[int]
    sample_rate: Final[int]
    num_samples: Final[int]
    num_frames: Final[int]
    @property
    def channels(self) -> ChannelLayout: ...
    def get_frame(self, n: _IntLike) -> AudioFrame: ...
    @overload  # type: ignore[override]
    def get_frame_async(self, n: _IntLike) -> Future[AudioFrame]: ...
    @overload
    def get_frame_async(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, n: _IntLike, cb: Callable[[AudioFrame | None, Exception | None], None]
    ) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[AudioFrame]: ...

# <plugins/bound/AudioNode>
# <attribute/AudioNode_bound/std>
    std: Final[_std._AudioNode_bound.Plugin]
    """VapourSynth Core Functions"""
# </attribute/AudioNode_bound/std>
# </plugins/bound/AudioNode>

class Core:
    timings: Final[CoreTimings]
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __dir__(self) -> list[str]: ...
    def __getattr__(self, name: str) -> Plugin: ...
    @property
    def api_version(self) -> VapourSynthAPIVersion: ...
    @property
    def core_version(self) -> VapourSynthVersion: ...
    @property
    def num_threads(self) -> int: ...
    @num_threads.setter
    def num_threads(self, value: _IntLike) -> None: ...
    @property
    def max_cache_size(self) -> int: ...
    @max_cache_size.setter
    def max_cache_size(self, mb: _IntLike) -> None: ...
    @property
    def used_cache_size(self) -> int: ...
    @property
    def flags(self) -> int: ...
    def plugins(self) -> Iterator[Plugin]: ...
    def query_video_format(
        self,
        color_family: _IntLike,
        sample_type: _IntLike,
        bits_per_sample: _IntLike,
        subsampling_w: _IntLike = 0,
        subsampling_h: _IntLike = 0,
    ) -> VideoFormat: ...
    def get_video_format(self, id: _IntLike) -> VideoFormat: ...
    def create_video_frame(self, format: VideoFormat, width: _IntLike, height: _IntLike) -> VideoFrame: ...
    def log_message(self, message_type: _IntLike, message: str) -> None: ...
    def add_log_handler(self, handler_func: Callable[[MessageType, str], None]) -> LogHandle: ...
    def remove_log_handler(self, handle: LogHandle) -> None: ...
    def clear_cache(self) -> None: ...
    @deprecated("core.version() is deprecated, use str(core)!", category=DeprecationWarning)
    def version(self) -> str: ...
    @deprecated(
        "core.version_number() is deprecated, use core.core_version.release_major!", category=DeprecationWarning
    )
    def version_number(self) -> int: ...

# <plugins/bound/Core>
# <attribute/Core_bound/adg>
    adg: Final[_adg._Core_bound.Plugin]
    """Adaptive grain"""
# </attribute/Core_bound/adg>
# <attribute/Core_bound/akarin>
    akarin: Final[_akarin._Core_bound.Plugin]
    """Akarin's Experimental Filters"""
# </attribute/Core_bound/akarin>
# <attribute/Core_bound/awarp>
    awarp: Final[_awarp._Core_bound.Plugin]
    """AWarp filter from AWarpSharp2"""
# </attribute/Core_bound/awarp>
# <attribute/Core_bound/bilateralgpu>
    bilateralgpu: Final[_bilateralgpu._Core_bound.Plugin]
    """Bilateral filter using CUDA"""
# </attribute/Core_bound/bilateralgpu>
# <attribute/Core_bound/bilateralgpu_rtc>
    bilateralgpu_rtc: Final[_bilateralgpu_rtc._Core_bound.Plugin]
    """Bilateral filter using CUDA (NVRTC)"""
# </attribute/Core_bound/bilateralgpu_rtc>
# <attribute/Core_bound/bm3d>
    bm3d: Final[_bm3d._Core_bound.Plugin]
    """Implementation of BM3D denoising filter for VapourSynth."""
# </attribute/Core_bound/bm3d>
# <attribute/Core_bound/bm3dcpu>
    bm3dcpu: Final[_bm3dcpu._Core_bound.Plugin]
    """BM3D algorithm implemented in AVX and AVX2 intrinsics"""
# </attribute/Core_bound/bm3dcpu>
# <attribute/Core_bound/bm3dcuda>
    bm3dcuda: Final[_bm3dcuda._Core_bound.Plugin]
    """BM3D algorithm implemented in CUDA"""
# </attribute/Core_bound/bm3dcuda>
# <attribute/Core_bound/bm3dcuda_rtc>
    bm3dcuda_rtc: Final[_bm3dcuda_rtc._Core_bound.Plugin]
    """BM3D algorithm implemented in CUDA (NVRTC)"""
# </attribute/Core_bound/bm3dcuda_rtc>
# <attribute/Core_bound/bs>
    bs: Final[_bs._Core_bound.Plugin]
    """Best Source 2"""
# </attribute/Core_bound/bs>
# <attribute/Core_bound/bwdif>
    bwdif: Final[_bwdif._Core_bound.Plugin]
    """BobWeaver Deinterlacing Filter"""
# </attribute/Core_bound/bwdif>
# <attribute/Core_bound/cs>
    cs: Final[_cs._Core_bound.Plugin]
    """carefulsource"""
# </attribute/Core_bound/cs>
# <attribute/Core_bound/dctf>
    dctf: Final[_dctf._Core_bound.Plugin]
    """DCT/IDCT Frequency Suppressor"""
# </attribute/Core_bound/dctf>
# <attribute/Core_bound/deblock>
    deblock: Final[_deblock._Core_bound.Plugin]
    """It does a deblocking of the picture, using the deblocking filter of h264"""
# </attribute/Core_bound/deblock>
# <attribute/Core_bound/descale>
    descale: Final[_descale._Core_bound.Plugin]
    """Undo linear interpolation"""
# </attribute/Core_bound/descale>
# <attribute/Core_bound/dfttest>
    dfttest: Final[_dfttest._Core_bound.Plugin]
    """2D/3D frequency domain denoiser"""
# </attribute/Core_bound/dfttest>
# <attribute/Core_bound/dfttest2_cpu>
    dfttest2_cpu: Final[_dfttest2_cpu._Core_bound.Plugin]
    """DFTTest2 (CPU)"""
# </attribute/Core_bound/dfttest2_cpu>
# <attribute/Core_bound/dfttest2_cuda>
    dfttest2_cuda: Final[_dfttest2_cuda._Core_bound.Plugin]
    """DFTTest2 (CUDA)"""
# </attribute/Core_bound/dfttest2_cuda>
# <attribute/Core_bound/dfttest2_nvrtc>
    dfttest2_nvrtc: Final[_dfttest2_nvrtc._Core_bound.Plugin]
    """DFTTest2 (NVRTC)"""
# </attribute/Core_bound/dfttest2_nvrtc>
# <attribute/Core_bound/dvdsrc2>
    dvdsrc2: Final[_dvdsrc2._Core_bound.Plugin]
    """Dvdsrc 2nd tour"""
# </attribute/Core_bound/dvdsrc2>
# <attribute/Core_bound/edgemasks>
    edgemasks: Final[_edgemasks._Core_bound.Plugin]
    """Creates an edge mask using various operators"""
# </attribute/Core_bound/edgemasks>
# <attribute/Core_bound/eedi2>
    eedi2: Final[_eedi2._Core_bound.Plugin]
    """EEDI2"""
# </attribute/Core_bound/eedi2>
# <attribute/Core_bound/eedi3m>
    eedi3m: Final[_eedi3m._Core_bound.Plugin]
    """Enhanced Edge Directed Interpolation 3"""
# </attribute/Core_bound/eedi3m>
# <attribute/Core_bound/ffms2>
    ffms2: Final[_ffms2._Core_bound.Plugin]
    """FFmpegSource 2 for VapourSynth"""
# </attribute/Core_bound/ffms2>
# <attribute/Core_bound/fmtc>
    fmtc: Final[_fmtc._Core_bound.Plugin]
    """Format converter"""
# </attribute/Core_bound/fmtc>
# <attribute/Core_bound/hysteresis>
    hysteresis: Final[_hysteresis._Core_bound.Plugin]
    """Hysteresis filter."""
# </attribute/Core_bound/hysteresis>
# <attribute/Core_bound/imwri>
    imwri: Final[_imwri._Core_bound.Plugin]
    """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
# </attribute/Core_bound/imwri>
# <attribute/Core_bound/knlm>
    knlm: Final[_knlm._Core_bound.Plugin]
    """KNLMeansCL for VapourSynth"""
# </attribute/Core_bound/knlm>
# <attribute/Core_bound/lsmas>
    lsmas: Final[_lsmas._Core_bound.Plugin]
    """LSMASHSource for VapourSynth"""
# </attribute/Core_bound/lsmas>
# <attribute/Core_bound/manipmv>
    manipmv: Final[_manipmv._Core_bound.Plugin]
    """Manipulate Motion Vectors"""
# </attribute/Core_bound/manipmv>
# <attribute/Core_bound/mv>
    mv: Final[_mv._Core_bound.Plugin]
    """MVTools v24"""
# </attribute/Core_bound/mv>
# <attribute/Core_bound/neo_f3kdb>
    neo_f3kdb: Final[_neo_f3kdb._Core_bound.Plugin]
    """Neo F3KDB Deband Filter r10"""
# </attribute/Core_bound/neo_f3kdb>
# <attribute/Core_bound/nlm_cuda>
    nlm_cuda: Final[_nlm_cuda._Core_bound.Plugin]
    """Non-local means denoise filter implemented in CUDA"""
# </attribute/Core_bound/nlm_cuda>
# <attribute/Core_bound/nlm_ispc>
    nlm_ispc: Final[_nlm_ispc._Core_bound.Plugin]
    """Non-local means denoise filter implemented in ISPC"""
# </attribute/Core_bound/nlm_ispc>
# <attribute/Core_bound/noise>
    noise: Final[_noise._Core_bound.Plugin]
    """Noise generator"""
# </attribute/Core_bound/noise>
# <attribute/Core_bound/placebo>
    placebo: Final[_placebo._Core_bound.Plugin]
    """libplacebo plugin for VapourSynth"""
# </attribute/Core_bound/placebo>
# <attribute/Core_bound/resize>
    resize: Final[_resize._Core_bound.Plugin]
    """VapourSynth Resize"""
# </attribute/Core_bound/resize>
# <attribute/Core_bound/resize2>
    resize2: Final[_resize2._Core_bound.Plugin]
    """Built-in VapourSynth resizer based on zimg with some modifications."""
# </attribute/Core_bound/resize2>
# <attribute/Core_bound/sangnom>
    sangnom: Final[_sangnom._Core_bound.Plugin]
    """VapourSynth Single Field Deinterlacer"""
# </attribute/Core_bound/sangnom>
# <attribute/Core_bound/scxvid>
    scxvid: Final[_scxvid._Core_bound.Plugin]
    """VapourSynth Scxvid Plugin"""
# </attribute/Core_bound/scxvid>
# <attribute/Core_bound/sneedif>
    sneedif: Final[_sneedif._Core_bound.Plugin]
    """Setsugen No Ensemble of Edge Directed Interpolation Functions"""
# </attribute/Core_bound/sneedif>
# <attribute/Core_bound/std>
    std: Final[_std._Core_bound.Plugin]
    """VapourSynth Core Functions"""
# </attribute/Core_bound/std>
# <attribute/Core_bound/sub>
    sub: Final[_sub._Core_bound.Plugin]
    """A subtitling filter based on libass and FFmpeg."""
# </attribute/Core_bound/sub>
# <attribute/Core_bound/tcanny>
    tcanny: Final[_tcanny._Core_bound.Plugin]
    """Build an edge map using canny edge detection"""
# </attribute/Core_bound/tcanny>
# <attribute/Core_bound/text>
    text: Final[_text._Core_bound.Plugin]
    """VapourSynth Text"""
# </attribute/Core_bound/text>
# <attribute/Core_bound/vivtc>
    vivtc: Final[_vivtc._Core_bound.Plugin]
    """VFM"""
# </attribute/Core_bound/vivtc>
# <attribute/Core_bound/vszip>
    vszip: Final[_vszip._Core_bound.Plugin]
    """VapourSynth Zig Image Process"""
# </attribute/Core_bound/vszip>
# <attribute/Core_bound/wnnm>
    wnnm: Final[_wnnm._Core_bound.Plugin]
    """Weighted Nuclear Norm Minimization Denoiser"""
# </attribute/Core_bound/wnnm>
# <attribute/Core_bound/wwxd>
    wwxd: Final[_wwxd._Core_bound.Plugin]
    """Scene change detection approximately like Xvid's"""
# </attribute/Core_bound/wwxd>
# <attribute/Core_bound/znedi3>
    znedi3: Final[_znedi3._Core_bound.Plugin]
    """Neural network edge directed interpolation (3rd gen.)"""
# </attribute/Core_bound/znedi3>
# <attribute/Core_bound/zsmooth>
    zsmooth: Final[_zsmooth._Core_bound.Plugin]
    """Smoothing functions in Zig"""
# </attribute/Core_bound/zsmooth>
# </plugins/bound/Core>

# _CoreProxy doesn't inherit from Core but __getattr__ returns the attribute from the actual core
class _CoreProxy(Core):
    def __setattr__(self, name: str, value: Any) -> None: ...
    @property
    def core(self) -> Core: ...

core: _CoreProxy

# <plugins/implementations>
# <implementation/adg>
class _adg:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Mask(self, clip: VideoNode, luma_scaling: _FloatLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Mask(self, luma_scaling: _FloatLike | None = None) -> VideoNode: ...

# </implementation/adg>

# <implementation/akarin>
_ReturnDict_akarin_Version = TypedDict("_ReturnDict_akarin_Version", {"version": _AnyStr, "expr_backend": _AnyStr, "expr_features": _AnyStr | list[_AnyStr], "select_features": _AnyStr | list[_AnyStr], "text_features": _AnyStr | list[_AnyStr], "tmpl_features": _AnyStr | list[_AnyStr]})


class _akarin:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cambi(self, clip: VideoNode, window_size: _IntLike | None = None, topk: _FloatLike | None = None, tvi_threshold: _FloatLike | None = None, scores: _IntLike | None = None, scaling: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLISR(self, clip: VideoNode, scale: _IntLike | None = None, device_id: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLVFX(self, clip: VideoNode, op: _IntLike, scale: _FloatLike | None = None, strength: _FloatLike | None = None, output_depth: _IntLike | None = None, num_streams: _IntLike | None = None, model_dir: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, clips: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr], format: _IntLike | None = None, opt: _IntLike | None = None, boundary: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropExpr(self, clips: VideoNode | _SequenceLike[VideoNode], dict: _VSCallback_akarin_PropExpr_dict) -> VideoNode: ...
            @_Wrapper.Function
            def Select(self, clip_src: VideoNode | _SequenceLike[VideoNode], prop_src: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, clips: VideoNode | _SequenceLike[VideoNode], text: _AnyStr, alignment: _IntLike | None = None, scale: _IntLike | None = None, prop: _AnyStr | None = None, strict: _IntLike | None = None, vspipe: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tmpl(self, clips: VideoNode | _SequenceLike[VideoNode], prop: _AnyStr | _SequenceLike[_AnyStr], text: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> _ReturnDict_akarin_Version: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cambi(self, window_size: _IntLike | None = None, topk: _FloatLike | None = None, tvi_threshold: _FloatLike | None = None, scores: _IntLike | None = None, scaling: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLISR(self, scale: _IntLike | None = None, device_id: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLVFX(self, op: _IntLike, scale: _FloatLike | None = None, strength: _FloatLike | None = None, output_depth: _IntLike | None = None, num_streams: _IntLike | None = None, model_dir: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, expr: _AnyStr | _SequenceLike[_AnyStr], format: _IntLike | None = None, opt: _IntLike | None = None, boundary: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropExpr(self, dict: _VSCallback_akarin_PropExpr_dict) -> VideoNode: ...
            @_Wrapper.Function
            def Select(self, prop_src: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, text: _AnyStr, alignment: _IntLike | None = None, scale: _IntLike | None = None, prop: _AnyStr | None = None, strict: _IntLike | None = None, vspipe: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tmpl(self, prop: _AnyStr | _SequenceLike[_AnyStr], text: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...

# </implementation/akarin>

# <implementation/awarp>
class _awarp:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AWarp(self, clip: VideoNode, mask: VideoNode, depth_h: _IntLike | _SequenceLike[_IntLike] | None = None, depth_v: _IntLike | _SequenceLike[_IntLike] | None = None, mask_first_plane: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AWarp(self, mask: VideoNode, depth_h: _IntLike | _SequenceLike[_IntLike] | None = None, depth_v: _IntLike | _SequenceLike[_IntLike] | None = None, mask_first_plane: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

# </implementation/awarp>

# <implementation/bilateralgpu>
class _bilateralgpu:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, clip: VideoNode, sigma_spatial: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigma_color: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | _SequenceLike[_IntLike] | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None, use_shared_memory: _IntLike | None = None, ref: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, sigma_spatial: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigma_color: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | _SequenceLike[_IntLike] | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None, use_shared_memory: _IntLike | None = None, ref: VideoNode | None = None) -> VideoNode: ...

# </implementation/bilateralgpu>

# <implementation/bilateralgpu_rtc>
class _bilateralgpu_rtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, clip: VideoNode, sigma_spatial: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigma_color: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | _SequenceLike[_IntLike] | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None, use_shared_memory: _IntLike | None = None, block_x: _IntLike | None = None, block_y: _IntLike | None = None, ref: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, sigma_spatial: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigma_color: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | _SequenceLike[_IntLike] | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None, use_shared_memory: _IntLike | None = None, block_x: _IntLike | None = None, block_y: _IntLike | None = None, ref: VideoNode | None = None) -> VideoNode: ...

# </implementation/bilateralgpu_rtc>

# <implementation/bm3d>
class _bm3d:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Basic(self, input: VideoNode, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, th_mse: _FloatLike | None = None, hard_thr: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Final(self, input: VideoNode, ref: VideoNode, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, th_mse: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def OPP2RGB(self, input: VideoNode, sample: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RGB2OPP(self, input: VideoNode, sample: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, input: VideoNode, radius: _IntLike | None = None, sample: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VBasic(self, input: VideoNode, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, ps_step: _IntLike | None = None, th_mse: _FloatLike | None = None, hard_thr: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFinal(self, input: VideoNode, ref: VideoNode, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, ps_step: _IntLike | None = None, th_mse: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Basic(self, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, th_mse: _FloatLike | None = None, hard_thr: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Final(self, ref: VideoNode, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, th_mse: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def OPP2RGB(self, sample: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RGB2OPP(self, sample: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, radius: _IntLike | None = None, sample: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VBasic(self, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, ps_step: _IntLike | None = None, th_mse: _FloatLike | None = None, hard_thr: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFinal(self, ref: VideoNode, profile: _AnyStr | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, bm_step: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, ps_step: _IntLike | None = None, th_mse: _FloatLike | None = None, matrix: _IntLike | None = None) -> VideoNode: ...

# </implementation/bm3d>

# <implementation/bm3dcpu>
class _bm3dcpu:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, clip: VideoNode, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, chroma: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, clip: VideoNode, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, chroma: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, chroma: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, chroma: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

# </implementation/bm3dcpu>

# <implementation/bm3dcuda>
class _bm3dcuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, clip: VideoNode, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, clip: VideoNode, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

# </implementation/bm3dcuda>

# <implementation/bm3dcuda_rtc>
class _bm3dcuda_rtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, clip: VideoNode, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, clip: VideoNode, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, ref: VideoNode | None = None, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_step: _IntLike | _SequenceLike[_IntLike] | None = None, bm_range: _IntLike | _SequenceLike[_IntLike] | None = None, radius: _IntLike | None = None, ps_num: _IntLike | _SequenceLike[_IntLike] | None = None, ps_range: _IntLike | _SequenceLike[_IntLike] | None = None, chroma: _IntLike | None = None, device_id: _IntLike | None = None, fast: _IntLike | None = None, extractor_exp: _IntLike | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

# </implementation/bm3dcuda_rtc>

# <implementation/bs>
_ReturnDict_bs_TrackInfo = TypedDict("_ReturnDict_bs_TrackInfo", {"mediatype": int, "mediatypestr": _AnyStr, "codec": int, "codecstr": _AnyStr, "disposition": int, "dispositionstr": _AnyStr})


class _bs:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AudioSource(self, source: _AnyStr, track: _IntLike | None = None, adjustdelay: _IntLike | None = None, threads: _IntLike | None = None, enable_drefs: _IntLike | None = None, use_absolute_path: _IntLike | None = None, drc_scale: _FloatLike | None = None, cachemode: _IntLike | None = None, cachepath: _AnyStr | None = None, cachesize: _IntLike | None = None, showprogress: _IntLike | None = None, maxdecoders: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def Metadata(self, source: _AnyStr, track: _IntLike | None = None, enable_drefs: _IntLike | None = None, use_absolute_path: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetDebugOutput(self, enable: _IntLike) -> None: ...
            @_Wrapper.Function
            def SetFFmpegLogLevel(self, level: _IntLike) -> _IntLike: ...
            @_Wrapper.Function
            def TrackInfo(self, source: _AnyStr, enable_drefs: _IntLike | None = None, use_absolute_path: _IntLike | None = None) -> _ReturnDict_bs_TrackInfo: ...
            @_Wrapper.Function
            def VideoSource(self, source: _AnyStr, track: _IntLike | None = None, variableformat: _IntLike | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None, rff: _IntLike | None = None, threads: _IntLike | None = None, seekpreroll: _IntLike | None = None, enable_drefs: _IntLike | None = None, use_absolute_path: _IntLike | None = None, cachemode: _IntLike | None = None, cachepath: _AnyStr | None = None, cachesize: _IntLike | None = None, hwdevice: _AnyStr | None = None, extrahwframes: _IntLike | None = None, timecodes: _AnyStr | None = None, start_number: _IntLike | None = None, viewid: _IntLike | None = None, showprogress: _IntLike | None = None, maxdecoders: _IntLike | None = None, hwfallback: _IntLike | None = None, exporttimestamps: _IntLike | None = None) -> VideoNode: ...

# </implementation/bs>

# <implementation/bwdif>
class _bwdif:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bwdif(self, clip: VideoNode, field: _IntLike, edeint: VideoNode | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bwdif(self, field: _IntLike, edeint: VideoNode | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/bwdif>

# <implementation/cs>
class _cs:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ConvertColor(self, clip: VideoNode, output_profile: _AnyStr, input_profile: _AnyStr | None = None, float_output: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ImageSource(self, source: _AnyStr, subsampling_pad: _IntLike | None = None, jpeg_rgb: _IntLike | None = None, jpeg_fancy_upsampling: _IntLike | None = None, jpeg_cmyk_profile: _AnyStr | None = None, jpeg_cmyk_target_profile: _AnyStr | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ConvertColor(self, output_profile: _AnyStr, input_profile: _AnyStr | None = None, float_output: _IntLike | None = None) -> VideoNode: ...

# </implementation/cs>

# <implementation/dctf>
class _dctf:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DCTFilter(self, clip: VideoNode, factors: _FloatLike | _SequenceLike[_FloatLike], planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DCTFilter(self, factors: _FloatLike | _SequenceLike[_FloatLike], planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

# </implementation/dctf>

# <implementation/deblock>
class _deblock:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deblock(self, clip: VideoNode, quant: _IntLike | None = None, aoffset: _IntLike | None = None, boffset: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deblock(self, quant: _IntLike | None = None, aoffset: _IntLike | None = None, boffset: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

# </implementation/deblock>

# <implementation/descale>
class _descale:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, src: VideoNode, width: _IntLike, height: _IntLike, b: _FloatLike | None = None, c: _FloatLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debicubic(self, src: VideoNode, width: _IntLike, height: _IntLike, b: _FloatLike | None = None, c: _FloatLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debilinear(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Decustom(self, src: VideoNode, width: _IntLike, height: _IntLike, custom_kernel: _VSCallback_descale_Decustom_custom_kernel, taps: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Delanczos(self, src: VideoNode, width: _IntLike, height: _IntLike, taps: _IntLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Depoint(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline16(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline36(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline64(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, src: VideoNode, width: _IntLike, height: _IntLike, taps: _IntLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleCustom(self, src: VideoNode, width: _IntLike, height: _IntLike, custom_kernel: _VSCallback_descale_ScaleCustom_custom_kernel, taps: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, src: VideoNode, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, width: _IntLike, height: _IntLike, b: _FloatLike | None = None, c: _FloatLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debicubic(self, width: _IntLike, height: _IntLike, b: _FloatLike | None = None, c: _FloatLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debilinear(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Decustom(self, width: _IntLike, height: _IntLike, custom_kernel: _VSCallback_descale_Decustom_custom_kernel, taps: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Delanczos(self, width: _IntLike, height: _IntLike, taps: _IntLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Depoint(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline16(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline36(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline64(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, width: _IntLike, height: _IntLike, taps: _IntLike | None = None, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleCustom(self, width: _IntLike, height: _IntLike, custom_kernel: _VSCallback_descale_ScaleCustom_custom_kernel, taps: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, width: _IntLike, height: _IntLike, blur: _FloatLike | None = None, post_conv: _FloatLike | _SequenceLike[_FloatLike] | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, border_handling: _IntLike | None = None, ignore_mask: VideoNode | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/descale>

# <implementation/dfttest>
class _dfttest:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, ftype: _IntLike | None = None, sigma: _FloatLike | None = None, sigma2: _FloatLike | None = None, pmin: _FloatLike | None = None, pmax: _FloatLike | None = None, sbsize: _IntLike | None = None, smode: _IntLike | None = None, sosize: _IntLike | None = None, tbsize: _IntLike | None = None, tmode: _IntLike | None = None, tosize: _IntLike | None = None, swin: _IntLike | None = None, twin: _IntLike | None = None, sbeta: _FloatLike | None = None, tbeta: _FloatLike | None = None, zmean: _IntLike | None = None, f0beta: _FloatLike | None = None, nlocation: _IntLike | _SequenceLike[_IntLike] | None = None, alpha: _FloatLike | None = None, slocation: _FloatLike | _SequenceLike[_FloatLike] | None = None, ssx: _FloatLike | _SequenceLike[_FloatLike] | None = None, ssy: _FloatLike | _SequenceLike[_FloatLike] | None = None, sst: _FloatLike | _SequenceLike[_FloatLike] | None = None, ssystem: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, ftype: _IntLike | None = None, sigma: _FloatLike | None = None, sigma2: _FloatLike | None = None, pmin: _FloatLike | None = None, pmax: _FloatLike | None = None, sbsize: _IntLike | None = None, smode: _IntLike | None = None, sosize: _IntLike | None = None, tbsize: _IntLike | None = None, tmode: _IntLike | None = None, tosize: _IntLike | None = None, swin: _IntLike | None = None, twin: _IntLike | None = None, sbeta: _FloatLike | None = None, tbeta: _FloatLike | None = None, zmean: _IntLike | None = None, f0beta: _FloatLike | None = None, nlocation: _IntLike | _SequenceLike[_IntLike] | None = None, alpha: _FloatLike | None = None, slocation: _FloatLike | _SequenceLike[_FloatLike] | None = None, ssx: _FloatLike | _SequenceLike[_FloatLike] | None = None, ssy: _FloatLike | _SequenceLike[_FloatLike] | None = None, sst: _FloatLike | _SequenceLike[_FloatLike] | None = None, ssystem: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/dfttest>

# <implementation/dfttest2_cpu>
class _dfttest2_cpu:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, window: _FloatLike | _SequenceLike[_FloatLike], sigma: _FloatLike | _SequenceLike[_FloatLike], sigma2: _FloatLike, pmin: _FloatLike, pmax: _FloatLike, filter_type: _IntLike, radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, zero_mean: _IntLike | None = None, window_freq: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RDFT(self, data: _FloatLike | _SequenceLike[_FloatLike], shape: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, window: _FloatLike | _SequenceLike[_FloatLike], sigma: _FloatLike | _SequenceLike[_FloatLike], sigma2: _FloatLike, pmin: _FloatLike, pmax: _FloatLike, filter_type: _IntLike, radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, zero_mean: _IntLike | None = None, window_freq: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/dfttest2_cpu>

# <implementation/dfttest2_cuda>
class _dfttest2_cuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, in_place: _IntLike | None = None, device_id: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RDFT(self, data: _FloatLike | _SequenceLike[_FloatLike], shape: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def ToSingle(self, data: _FloatLike | _SequenceLike[_FloatLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, in_place: _IntLike | None = None, device_id: _IntLike | None = None) -> VideoNode: ...

# </implementation/dfttest2_cuda>

# <implementation/dfttest2_nvrtc>
class _dfttest2_nvrtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, in_place: _IntLike | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RDFT(self, data: _FloatLike | _SequenceLike[_FloatLike], shape: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def ToSingle(self, data: _FloatLike | _SequenceLike[_FloatLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: _IntLike | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, in_place: _IntLike | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None) -> VideoNode: ...

# </implementation/dfttest2_nvrtc>

# <implementation/dvdsrc2>
class _dvdsrc2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def FullVts(self, path: _AnyStr, vts: _IntLike, ranges: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FullVtsAc3(self, path: _AnyStr, vts: _IntLike, audio: _IntLike, ranges: _IntLike | _SequenceLike[_IntLike] | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def FullVtsLpcm(self, path: _AnyStr, vts: _IntLike, audio: _IntLike, ranges: _IntLike | _SequenceLike[_IntLike] | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def Ifo(self, path: _AnyStr, ifo: _IntLike) -> _AnyStr: ...
            @_Wrapper.Function
            def RawAc3(self, path: _AnyStr, vts: _IntLike, audio: _IntLike, ranges: _IntLike | _SequenceLike[_IntLike] | None = None) -> AudioNode: ...

# </implementation/dvdsrc2>

# <implementation/edgemasks>
class _edgemasks:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cross(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExKirsch(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExPrewitt(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExSobel(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FDoG(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kirsch(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kroon(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RScharr(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson3(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson5(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Scharr(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tritical(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cross(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExKirsch(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExPrewitt(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExSobel(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FDoG(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kirsch(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kroon(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RScharr(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson3(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson5(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Scharr(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tritical(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | _SequenceLike[_FloatLike] | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/edgemasks>

# <implementation/eedi2>
class _eedi2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI2(self, clip: VideoNode, field: _IntLike, mthresh: _IntLike | None = None, lthresh: _IntLike | None = None, vthresh: _IntLike | None = None, estr: _IntLike | None = None, dstr: _IntLike | None = None, maxd: _IntLike | None = None, map: _IntLike | None = None, nt: _IntLike | None = None, pp: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI2(self, field: _IntLike, mthresh: _IntLike | None = None, lthresh: _IntLike | None = None, vthresh: _IntLike | None = None, estr: _IntLike | None = None, dstr: _IntLike | None = None, maxd: _IntLike | None = None, map: _IntLike | None = None, nt: _IntLike | None = None, pp: _IntLike | None = None) -> VideoNode: ...

# </implementation/eedi2>

# <implementation/eedi3m>
class _eedi3m:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI3(self, clip: VideoNode, field: _IntLike, dh: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, alpha: _FloatLike | None = None, beta: _FloatLike | None = None, gamma: _FloatLike | None = None, nrad: _IntLike | None = None, mdis: _IntLike | None = None, hp: _IntLike | None = None, ucubic: _IntLike | None = None, cost3: _IntLike | None = None, vcheck: _IntLike | None = None, vthresh0: _FloatLike | None = None, vthresh1: _FloatLike | None = None, vthresh2: _FloatLike | None = None, sclip: VideoNode | None = None, mclip: VideoNode | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI3(self, field: _IntLike, dh: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, alpha: _FloatLike | None = None, beta: _FloatLike | None = None, gamma: _FloatLike | None = None, nrad: _IntLike | None = None, mdis: _IntLike | None = None, hp: _IntLike | None = None, ucubic: _IntLike | None = None, cost3: _IntLike | None = None, vcheck: _IntLike | None = None, vthresh0: _FloatLike | None = None, vthresh1: _FloatLike | None = None, vthresh2: _FloatLike | None = None, sclip: VideoNode | None = None, mclip: VideoNode | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/eedi3m>

# <implementation/ffms2>
class _ffms2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def GetLogLevel(self) -> _IntLike: ...
            @_Wrapper.Function
            def Index(self, source: _AnyStr, cachefile: _AnyStr | None = None, indextracks: _IntLike | _SequenceLike[_IntLike] | None = None, errorhandling: _IntLike | None = None, overwrite: _IntLike | None = None, enable_drefs: _IntLike | None = None, use_absolute_path: _IntLike | None = None) -> _AnyStr: ...
            @_Wrapper.Function
            def SetLogLevel(self, level: _IntLike) -> _IntLike: ...
            @_Wrapper.Function
            def Source(self, source: _AnyStr, track: _IntLike | None = None, cache: _IntLike | None = None, cachefile: _AnyStr | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None, threads: _IntLike | None = None, timecodes: _AnyStr | None = None, seekmode: _IntLike | None = None, width: _IntLike | None = None, height: _IntLike | None = None, resizer: _AnyStr | None = None, format: _IntLike | None = None, alpha: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> _AnyStr: ...

# </implementation/ffms2>

# <implementation/fmtc>
class _fmtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def bitdepth(self, clip: VideoNode, csp: _IntLike | None = None, bits: _IntLike | None = None, flt: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, dmode: _IntLike | None = None, ampo: _FloatLike | None = None, ampn: _FloatLike | None = None, dyn: _IntLike | None = None, staticnoise: _IntLike | None = None, cpuopt: _IntLike | None = None, patsize: _IntLike | None = None, tpdfo: _IntLike | None = None, tpdfn: _IntLike | None = None, corplane: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def histluma(self, clip: VideoNode, full: _IntLike | None = None, amp: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix(self, clip: VideoNode, mat: _AnyStr | None = None, mats: _AnyStr | None = None, matd: _AnyStr | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, coef: _FloatLike | _SequenceLike[_FloatLike] | None = None, csp: _IntLike | None = None, col_fam: _IntLike | None = None, bits: _IntLike | None = None, singleout: _IntLike | None = None, cpuopt: _IntLike | None = None, planes: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix2020cl(self, clip: VideoNode, full: _IntLike | None = None, csp: _IntLike | None = None, bits: _IntLike | None = None, cpuopt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def nativetostack16(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def primaries(self, clip: VideoNode, rs: _FloatLike | _SequenceLike[_FloatLike] | None = None, gs: _FloatLike | _SequenceLike[_FloatLike] | None = None, bs: _FloatLike | _SequenceLike[_FloatLike] | None = None, ws: _FloatLike | _SequenceLike[_FloatLike] | None = None, rd: _FloatLike | _SequenceLike[_FloatLike] | None = None, gd: _FloatLike | _SequenceLike[_FloatLike] | None = None, bd: _FloatLike | _SequenceLike[_FloatLike] | None = None, wd: _FloatLike | _SequenceLike[_FloatLike] | None = None, prims: _AnyStr | None = None, primd: _AnyStr | None = None, wconv: _IntLike | None = None, cpuopt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def resample(self, clip: VideoNode, w: _IntLike | None = None, h: _IntLike | None = None, sx: _FloatLike | _SequenceLike[_FloatLike] | None = None, sy: _FloatLike | _SequenceLike[_FloatLike] | None = None, sw: _FloatLike | _SequenceLike[_FloatLike] | None = None, sh: _FloatLike | _SequenceLike[_FloatLike] | None = None, scale: _FloatLike | None = None, scaleh: _FloatLike | None = None, scalev: _FloatLike | None = None, kernel: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelh: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelv: _AnyStr | _SequenceLike[_AnyStr] | None = None, impulse: _FloatLike | _SequenceLike[_FloatLike] | None = None, impulseh: _FloatLike | _SequenceLike[_FloatLike] | None = None, impulsev: _FloatLike | _SequenceLike[_FloatLike] | None = None, taps: _IntLike | _SequenceLike[_IntLike] | None = None, tapsh: _IntLike | _SequenceLike[_IntLike] | None = None, tapsv: _IntLike | _SequenceLike[_IntLike] | None = None, a1: _FloatLike | _SequenceLike[_FloatLike] | None = None, a2: _FloatLike | _SequenceLike[_FloatLike] | None = None, a3: _FloatLike | _SequenceLike[_FloatLike] | None = None, a1h: _FloatLike | _SequenceLike[_FloatLike] | None = None, a2h: _FloatLike | _SequenceLike[_FloatLike] | None = None, a3h: _FloatLike | _SequenceLike[_FloatLike] | None = None, a1v: _FloatLike | _SequenceLike[_FloatLike] | None = None, a2v: _FloatLike | _SequenceLike[_FloatLike] | None = None, a3v: _FloatLike | _SequenceLike[_FloatLike] | None = None, kovrspl: _IntLike | _SequenceLike[_IntLike] | None = None, fh: _FloatLike | _SequenceLike[_FloatLike] | None = None, fv: _FloatLike | _SequenceLike[_FloatLike] | None = None, cnorm: _IntLike | _SequenceLike[_IntLike] | None = None, total: _FloatLike | _SequenceLike[_FloatLike] | None = None, totalh: _FloatLike | _SequenceLike[_FloatLike] | None = None, totalv: _FloatLike | _SequenceLike[_FloatLike] | None = None, invks: _IntLike | _SequenceLike[_IntLike] | None = None, invksh: _IntLike | _SequenceLike[_IntLike] | None = None, invksv: _IntLike | _SequenceLike[_IntLike] | None = None, invkstaps: _IntLike | _SequenceLike[_IntLike] | None = None, invkstapsh: _IntLike | _SequenceLike[_IntLike] | None = None, invkstapsv: _IntLike | _SequenceLike[_IntLike] | None = None, csp: _IntLike | None = None, css: _AnyStr | None = None, planes: _FloatLike | _SequenceLike[_FloatLike] | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, center: _IntLike | _SequenceLike[_IntLike] | None = None, cplace: _AnyStr | None = None, cplaces: _AnyStr | None = None, cplaced: _AnyStr | None = None, interlaced: _IntLike | None = None, interlacedd: _IntLike | None = None, tff: _IntLike | None = None, tffd: _IntLike | None = None, flt: _IntLike | None = None, cpuopt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def stack16tonative(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def transfer(self, clip: VideoNode, transs: _AnyStr | _SequenceLike[_AnyStr] | None = None, transd: _AnyStr | _SequenceLike[_AnyStr] | None = None, cont: _FloatLike | None = None, gcor: _FloatLike | None = None, bits: _IntLike | None = None, flt: _IntLike | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, logceis: _IntLike | None = None, logceid: _IntLike | None = None, cpuopt: _IntLike | None = None, blacklvl: _FloatLike | None = None, sceneref: _IntLike | None = None, lb: _FloatLike | None = None, lw: _FloatLike | None = None, lws: _FloatLike | None = None, lwd: _FloatLike | None = None, ambient: _FloatLike | None = None, match: _IntLike | None = None, gy: _IntLike | None = None, debug: _IntLike | None = None, sig_c: _FloatLike | None = None, sig_t: _FloatLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def bitdepth(self, csp: _IntLike | None = None, bits: _IntLike | None = None, flt: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, dmode: _IntLike | None = None, ampo: _FloatLike | None = None, ampn: _FloatLike | None = None, dyn: _IntLike | None = None, staticnoise: _IntLike | None = None, cpuopt: _IntLike | None = None, patsize: _IntLike | None = None, tpdfo: _IntLike | None = None, tpdfn: _IntLike | None = None, corplane: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def histluma(self, full: _IntLike | None = None, amp: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix(self, mat: _AnyStr | None = None, mats: _AnyStr | None = None, matd: _AnyStr | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, coef: _FloatLike | _SequenceLike[_FloatLike] | None = None, csp: _IntLike | None = None, col_fam: _IntLike | None = None, bits: _IntLike | None = None, singleout: _IntLike | None = None, cpuopt: _IntLike | None = None, planes: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix2020cl(self, full: _IntLike | None = None, csp: _IntLike | None = None, bits: _IntLike | None = None, cpuopt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def nativetostack16(self) -> VideoNode: ...
            @_Wrapper.Function
            def primaries(self, rs: _FloatLike | _SequenceLike[_FloatLike] | None = None, gs: _FloatLike | _SequenceLike[_FloatLike] | None = None, bs: _FloatLike | _SequenceLike[_FloatLike] | None = None, ws: _FloatLike | _SequenceLike[_FloatLike] | None = None, rd: _FloatLike | _SequenceLike[_FloatLike] | None = None, gd: _FloatLike | _SequenceLike[_FloatLike] | None = None, bd: _FloatLike | _SequenceLike[_FloatLike] | None = None, wd: _FloatLike | _SequenceLike[_FloatLike] | None = None, prims: _AnyStr | None = None, primd: _AnyStr | None = None, wconv: _IntLike | None = None, cpuopt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def resample(self, w: _IntLike | None = None, h: _IntLike | None = None, sx: _FloatLike | _SequenceLike[_FloatLike] | None = None, sy: _FloatLike | _SequenceLike[_FloatLike] | None = None, sw: _FloatLike | _SequenceLike[_FloatLike] | None = None, sh: _FloatLike | _SequenceLike[_FloatLike] | None = None, scale: _FloatLike | None = None, scaleh: _FloatLike | None = None, scalev: _FloatLike | None = None, kernel: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelh: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelv: _AnyStr | _SequenceLike[_AnyStr] | None = None, impulse: _FloatLike | _SequenceLike[_FloatLike] | None = None, impulseh: _FloatLike | _SequenceLike[_FloatLike] | None = None, impulsev: _FloatLike | _SequenceLike[_FloatLike] | None = None, taps: _IntLike | _SequenceLike[_IntLike] | None = None, tapsh: _IntLike | _SequenceLike[_IntLike] | None = None, tapsv: _IntLike | _SequenceLike[_IntLike] | None = None, a1: _FloatLike | _SequenceLike[_FloatLike] | None = None, a2: _FloatLike | _SequenceLike[_FloatLike] | None = None, a3: _FloatLike | _SequenceLike[_FloatLike] | None = None, a1h: _FloatLike | _SequenceLike[_FloatLike] | None = None, a2h: _FloatLike | _SequenceLike[_FloatLike] | None = None, a3h: _FloatLike | _SequenceLike[_FloatLike] | None = None, a1v: _FloatLike | _SequenceLike[_FloatLike] | None = None, a2v: _FloatLike | _SequenceLike[_FloatLike] | None = None, a3v: _FloatLike | _SequenceLike[_FloatLike] | None = None, kovrspl: _IntLike | _SequenceLike[_IntLike] | None = None, fh: _FloatLike | _SequenceLike[_FloatLike] | None = None, fv: _FloatLike | _SequenceLike[_FloatLike] | None = None, cnorm: _IntLike | _SequenceLike[_IntLike] | None = None, total: _FloatLike | _SequenceLike[_FloatLike] | None = None, totalh: _FloatLike | _SequenceLike[_FloatLike] | None = None, totalv: _FloatLike | _SequenceLike[_FloatLike] | None = None, invks: _IntLike | _SequenceLike[_IntLike] | None = None, invksh: _IntLike | _SequenceLike[_IntLike] | None = None, invksv: _IntLike | _SequenceLike[_IntLike] | None = None, invkstaps: _IntLike | _SequenceLike[_IntLike] | None = None, invkstapsh: _IntLike | _SequenceLike[_IntLike] | None = None, invkstapsv: _IntLike | _SequenceLike[_IntLike] | None = None, csp: _IntLike | None = None, css: _AnyStr | None = None, planes: _FloatLike | _SequenceLike[_FloatLike] | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, center: _IntLike | _SequenceLike[_IntLike] | None = None, cplace: _AnyStr | None = None, cplaces: _AnyStr | None = None, cplaced: _AnyStr | None = None, interlaced: _IntLike | None = None, interlacedd: _IntLike | None = None, tff: _IntLike | None = None, tffd: _IntLike | None = None, flt: _IntLike | None = None, cpuopt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def stack16tonative(self) -> VideoNode: ...
            @_Wrapper.Function
            def transfer(self, transs: _AnyStr | _SequenceLike[_AnyStr] | None = None, transd: _AnyStr | _SequenceLike[_AnyStr] | None = None, cont: _FloatLike | None = None, gcor: _FloatLike | None = None, bits: _IntLike | None = None, flt: _IntLike | None = None, fulls: _IntLike | None = None, fulld: _IntLike | None = None, logceis: _IntLike | None = None, logceid: _IntLike | None = None, cpuopt: _IntLike | None = None, blacklvl: _FloatLike | None = None, sceneref: _IntLike | None = None, lb: _FloatLike | None = None, lw: _FloatLike | None = None, lws: _FloatLike | None = None, lwd: _FloatLike | None = None, ambient: _FloatLike | None = None, match: _IntLike | None = None, gy: _IntLike | None = None, debug: _IntLike | None = None, sig_c: _FloatLike | None = None, sig_t: _FloatLike | None = None) -> VideoNode: ...

# </implementation/fmtc>

# <implementation/hysteresis>
class _hysteresis:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Hysteresis(self, clipa: VideoNode, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Hysteresis(self, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

# </implementation/hysteresis>

# <implementation/imwri>
class _imwri:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Read(self, filename: _AnyStr | _SequenceLike[_AnyStr], firstnum: _IntLike | None = None, mismatch: _IntLike | None = None, alpha: _IntLike | None = None, float_output: _IntLike | None = None, embed_icc: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Write(self, clip: VideoNode, imgformat: _AnyStr, filename: _AnyStr, firstnum: _IntLike | None = None, quality: _IntLike | None = None, dither: _IntLike | None = None, compression_type: _AnyStr | None = None, overwrite: _IntLike | None = None, alpha: VideoNode | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Write(self, imgformat: _AnyStr, filename: _AnyStr, firstnum: _IntLike | None = None, quality: _IntLike | None = None, dither: _IntLike | None = None, compression_type: _AnyStr | None = None, overwrite: _IntLike | None = None, alpha: VideoNode | None = None) -> VideoNode: ...

# </implementation/imwri>

# <implementation/knlm>
class _knlm:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def KNLMeansCL(self, clip: VideoNode, d: _IntLike | None = None, a: _IntLike | None = None, s: _IntLike | None = None, h: _FloatLike | None = None, channels: _AnyStr | None = None, wmode: _IntLike | None = None, wref: _FloatLike | None = None, rclip: VideoNode | None = None, device_type: _AnyStr | None = None, device_id: _IntLike | None = None, ocl_x: _IntLike | None = None, ocl_y: _IntLike | None = None, ocl_r: _IntLike | None = None, info: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def KNLMeansCL(self, d: _IntLike | None = None, a: _IntLike | None = None, s: _IntLike | None = None, h: _FloatLike | None = None, channels: _AnyStr | None = None, wmode: _IntLike | None = None, wref: _FloatLike | None = None, rclip: VideoNode | None = None, device_type: _AnyStr | None = None, device_id: _IntLike | None = None, ocl_x: _IntLike | None = None, ocl_y: _IntLike | None = None, ocl_r: _IntLike | None = None, info: _IntLike | None = None) -> VideoNode: ...

# </implementation/knlm>

# <implementation/lsmas>
class _lsmas:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def LWLibavSource(self, source: _AnyStr, stream_index: _IntLike | None = None, cache: _IntLike | None = None, cachefile: _AnyStr | None = None, threads: _IntLike | None = None, seek_mode: _IntLike | None = None, seek_threshold: _IntLike | None = None, dr: _IntLike | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None, variable: _IntLike | None = None, format: _AnyStr | None = None, decoder: _AnyStr | None = None, prefer_hw: _IntLike | None = None, repeat: _IntLike | None = None, dominance: _IntLike | None = None, ff_loglevel: _IntLike | None = None, cachedir: _AnyStr | None = None, ff_options: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LibavSMASHSource(self, source: _AnyStr, track: _IntLike | None = None, threads: _IntLike | None = None, seek_mode: _IntLike | None = None, seek_threshold: _IntLike | None = None, dr: _IntLike | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None, variable: _IntLike | None = None, format: _AnyStr | None = None, decoder: _AnyStr | None = None, prefer_hw: _IntLike | None = None, ff_loglevel: _IntLike | None = None, ff_options: _AnyStr | None = None) -> VideoNode: ...

# </implementation/lsmas>

# <implementation/manipmv>
class _manipmv:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ExpandAnalysisData(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleVect(self, clip: VideoNode, scaleX: _IntLike | None = None, scaleY: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ShowVect(self, clip: VideoNode, vectors: VideoNode, useSceneChangeProps: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ExpandAnalysisData(self) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleVect(self, scaleX: _IntLike | None = None, scaleY: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ShowVect(self, vectors: VideoNode, useSceneChangeProps: _IntLike | None = None) -> VideoNode: ...

# </implementation/manipmv>

# <implementation/mv>
class _mv:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Analyse(self, super: VideoNode, blksize: _IntLike | None = None, blksizev: _IntLike | None = None, levels: _IntLike | None = None, search: _IntLike | None = None, searchparam: _IntLike | None = None, pelsearch: _IntLike | None = None, isb: _IntLike | None = None, lambda_: _IntLike | None = None, chroma: _IntLike | None = None, delta: _IntLike | None = None, truemotion: _IntLike | None = None, lsad: _IntLike | None = None, plevel: _IntLike | None = None, global_: _IntLike | None = None, pnew: _IntLike | None = None, pzero: _IntLike | None = None, pglobal: _IntLike | None = None, overlap: _IntLike | None = None, overlapv: _IntLike | None = None, divide: _IntLike | None = None, badsad: _IntLike | None = None, badrange: _IntLike | None = None, opt: _IntLike | None = None, meander: _IntLike | None = None, trymany: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None, search_coarse: _IntLike | None = None, dct: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlockFPS(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: _IntLike | None = None, den: _IntLike | None = None, mode: _IntLike | None = None, ml: _FloatLike | None = None, blend: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Compensate(self, clip: VideoNode, super: VideoNode, vectors: VideoNode, scbehavior: _IntLike | None = None, thsad: _IntLike | None = None, fields: _IntLike | None = None, time: _FloatLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain1(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain2(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain3(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain4(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain5(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain6(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, mvbw6: VideoNode, mvfw6: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanAnalyse(self, clip: VideoNode, vectors: VideoNode, mask: VideoNode | None = None, zoom: _IntLike | None = None, rot: _IntLike | None = None, pixaspect: _FloatLike | None = None, error: _FloatLike | None = None, info: _IntLike | None = None, wrong: _FloatLike | None = None, zerow: _FloatLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanCompensate(self, clip: VideoNode, data: VideoNode, offset: _FloatLike | None = None, subpixel: _IntLike | None = None, pixaspect: _FloatLike | None = None, matchfields: _IntLike | None = None, mirror: _IntLike | None = None, blur: _IntLike | None = None, info: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanEstimate(self, clip: VideoNode, trust: _FloatLike | None = None, winx: _IntLike | None = None, winy: _IntLike | None = None, wleft: _IntLike | None = None, wtop: _IntLike | None = None, dxmax: _IntLike | None = None, dymax: _IntLike | None = None, zoommax: _FloatLike | None = None, stab: _FloatLike | None = None, pixaspect: _FloatLike | None = None, info: _IntLike | None = None, show: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanStabilise(self, clip: VideoNode, data: VideoNode, cutoff: _FloatLike | None = None, damping: _FloatLike | None = None, initzoom: _FloatLike | None = None, addzoom: _IntLike | None = None, prev: _IntLike | None = None, next: _IntLike | None = None, mirror: _IntLike | None = None, blur: _IntLike | None = None, dxmax: _FloatLike | None = None, dymax: _FloatLike | None = None, zoommax: _FloatLike | None = None, rotmax: _FloatLike | None = None, subpixel: _IntLike | None = None, pixaspect: _FloatLike | None = None, fitlast: _IntLike | None = None, tzoom: _FloatLike | None = None, info: _IntLike | None = None, method: _IntLike | None = None, fields: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Finest(self, super: VideoNode, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Flow(self, clip: VideoNode, super: VideoNode, vectors: VideoNode, time: _FloatLike | None = None, mode: _IntLike | None = None, fields: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowBlur(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, blur: _FloatLike | None = None, prec: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowFPS(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: _IntLike | None = None, den: _IntLike | None = None, mask: _IntLike | None = None, ml: _FloatLike | None = None, blend: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowInter(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, time: _FloatLike | None = None, ml: _FloatLike | None = None, blend: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Mask(self, clip: VideoNode, vectors: VideoNode, ml: _FloatLike | None = None, gamma: _FloatLike | None = None, kind: _IntLike | None = None, time: _FloatLike | None = None, ysc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Recalculate(self, super: VideoNode, vectors: VideoNode, thsad: _IntLike | None = None, smooth: _IntLike | None = None, blksize: _IntLike | None = None, blksizev: _IntLike | None = None, search: _IntLike | None = None, searchparam: _IntLike | None = None, lambda_: _IntLike | None = None, chroma: _IntLike | None = None, truemotion: _IntLike | None = None, pnew: _IntLike | None = None, overlap: _IntLike | None = None, overlapv: _IntLike | None = None, divide: _IntLike | None = None, opt: _IntLike | None = None, meander: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None, dct: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SCDetection(self, clip: VideoNode, vectors: VideoNode, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Super(self, clip: VideoNode, hpad: _IntLike | None = None, vpad: _IntLike | None = None, pel: _IntLike | None = None, levels: _IntLike | None = None, chroma: _IntLike | None = None, sharp: _IntLike | None = None, rfilter: _IntLike | None = None, pelclip: VideoNode | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Analyse(self, blksize: _IntLike | None = None, blksizev: _IntLike | None = None, levels: _IntLike | None = None, search: _IntLike | None = None, searchparam: _IntLike | None = None, pelsearch: _IntLike | None = None, isb: _IntLike | None = None, lambda_: _IntLike | None = None, chroma: _IntLike | None = None, delta: _IntLike | None = None, truemotion: _IntLike | None = None, lsad: _IntLike | None = None, plevel: _IntLike | None = None, global_: _IntLike | None = None, pnew: _IntLike | None = None, pzero: _IntLike | None = None, pglobal: _IntLike | None = None, overlap: _IntLike | None = None, overlapv: _IntLike | None = None, divide: _IntLike | None = None, badsad: _IntLike | None = None, badrange: _IntLike | None = None, opt: _IntLike | None = None, meander: _IntLike | None = None, trymany: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None, search_coarse: _IntLike | None = None, dct: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlockFPS(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: _IntLike | None = None, den: _IntLike | None = None, mode: _IntLike | None = None, ml: _FloatLike | None = None, blend: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Compensate(self, super: VideoNode, vectors: VideoNode, scbehavior: _IntLike | None = None, thsad: _IntLike | None = None, fields: _IntLike | None = None, time: _FloatLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain1(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain2(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain3(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain4(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain5(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain6(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, mvbw6: VideoNode, mvfw6: VideoNode, thsad: _IntLike | None = None, thsadc: _IntLike | None = None, plane: _IntLike | None = None, limit: _IntLike | None = None, limitc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, weights: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanAnalyse(self, vectors: VideoNode, mask: VideoNode | None = None, zoom: _IntLike | None = None, rot: _IntLike | None = None, pixaspect: _FloatLike | None = None, error: _FloatLike | None = None, info: _IntLike | None = None, wrong: _FloatLike | None = None, zerow: _FloatLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanCompensate(self, data: VideoNode, offset: _FloatLike | None = None, subpixel: _IntLike | None = None, pixaspect: _FloatLike | None = None, matchfields: _IntLike | None = None, mirror: _IntLike | None = None, blur: _IntLike | None = None, info: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanEstimate(self, trust: _FloatLike | None = None, winx: _IntLike | None = None, winy: _IntLike | None = None, wleft: _IntLike | None = None, wtop: _IntLike | None = None, dxmax: _IntLike | None = None, dymax: _IntLike | None = None, zoommax: _FloatLike | None = None, stab: _FloatLike | None = None, pixaspect: _FloatLike | None = None, info: _IntLike | None = None, show: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanStabilise(self, data: VideoNode, cutoff: _FloatLike | None = None, damping: _FloatLike | None = None, initzoom: _FloatLike | None = None, addzoom: _IntLike | None = None, prev: _IntLike | None = None, next: _IntLike | None = None, mirror: _IntLike | None = None, blur: _IntLike | None = None, dxmax: _FloatLike | None = None, dymax: _FloatLike | None = None, zoommax: _FloatLike | None = None, rotmax: _FloatLike | None = None, subpixel: _IntLike | None = None, pixaspect: _FloatLike | None = None, fitlast: _IntLike | None = None, tzoom: _FloatLike | None = None, info: _IntLike | None = None, method: _IntLike | None = None, fields: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Finest(self, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Flow(self, super: VideoNode, vectors: VideoNode, time: _FloatLike | None = None, mode: _IntLike | None = None, fields: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowBlur(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, blur: _FloatLike | None = None, prec: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowFPS(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: _IntLike | None = None, den: _IntLike | None = None, mask: _IntLike | None = None, ml: _FloatLike | None = None, blend: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowInter(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, time: _FloatLike | None = None, ml: _FloatLike | None = None, blend: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Mask(self, vectors: VideoNode, ml: _FloatLike | None = None, gamma: _FloatLike | None = None, kind: _IntLike | None = None, time: _FloatLike | None = None, ysc: _IntLike | None = None, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Recalculate(self, vectors: VideoNode, thsad: _IntLike | None = None, smooth: _IntLike | None = None, blksize: _IntLike | None = None, blksizev: _IntLike | None = None, search: _IntLike | None = None, searchparam: _IntLike | None = None, lambda_: _IntLike | None = None, chroma: _IntLike | None = None, truemotion: _IntLike | None = None, pnew: _IntLike | None = None, overlap: _IntLike | None = None, overlapv: _IntLike | None = None, divide: _IntLike | None = None, opt: _IntLike | None = None, meander: _IntLike | None = None, fields: _IntLike | None = None, tff: _IntLike | None = None, dct: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SCDetection(self, vectors: VideoNode, thscd1: _IntLike | None = None, thscd2: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Super(self, hpad: _IntLike | None = None, vpad: _IntLike | None = None, pel: _IntLike | None = None, levels: _IntLike | None = None, chroma: _IntLike | None = None, sharp: _IntLike | None = None, rfilter: _IntLike | None = None, pelclip: VideoNode | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/mv>

# <implementation/neo_f3kdb>
class _neo_f3kdb:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, clip: VideoNode, range: _IntLike | None = None, y: _IntLike | None = None, cb: _IntLike | None = None, cr: _IntLike | None = None, grainy: _IntLike | None = None, grainc: _IntLike | None = None, sample_mode: _IntLike | None = None, seed: _IntLike | None = None, blur_first: _IntLike | None = None, dynamic_grain: _IntLike | None = None, opt: _IntLike | None = None, mt: _IntLike | None = None, dither_algo: _IntLike | None = None, keep_tv_range: _IntLike | None = None, output_depth: _IntLike | None = None, random_algo_ref: _IntLike | None = None, random_algo_grain: _IntLike | None = None, random_param_ref: _FloatLike | None = None, random_param_grain: _FloatLike | None = None, preset: _AnyStr | None = None, y_1: _IntLike | None = None, cb_1: _IntLike | None = None, cr_1: _IntLike | None = None, y_2: _IntLike | None = None, cb_2: _IntLike | None = None, cr_2: _IntLike | None = None, scale: _IntLike | None = None, angle_boost: _FloatLike | None = None, max_angle: _FloatLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, range: _IntLike | None = None, y: _IntLike | None = None, cb: _IntLike | None = None, cr: _IntLike | None = None, grainy: _IntLike | None = None, grainc: _IntLike | None = None, sample_mode: _IntLike | None = None, seed: _IntLike | None = None, blur_first: _IntLike | None = None, dynamic_grain: _IntLike | None = None, opt: _IntLike | None = None, mt: _IntLike | None = None, dither_algo: _IntLike | None = None, keep_tv_range: _IntLike | None = None, output_depth: _IntLike | None = None, random_algo_ref: _IntLike | None = None, random_algo_grain: _IntLike | None = None, random_param_ref: _FloatLike | None = None, random_param_grain: _FloatLike | None = None, preset: _AnyStr | None = None, y_1: _IntLike | None = None, cb_1: _IntLike | None = None, cr_1: _IntLike | None = None, y_2: _IntLike | None = None, cb_2: _IntLike | None = None, cr_2: _IntLike | None = None, scale: _IntLike | None = None, angle_boost: _FloatLike | None = None, max_angle: _FloatLike | None = None) -> VideoNode: ...

# </implementation/neo_f3kdb>

# <implementation/nlm_cuda>
class _nlm_cuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, clip: VideoNode, d: _IntLike | None = None, a: _IntLike | None = None, s: _IntLike | None = None, h: _FloatLike | None = None, channels: _AnyStr | None = None, wmode: _IntLike | None = None, wref: _FloatLike | None = None, rclip: VideoNode | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, d: _IntLike | None = None, a: _IntLike | None = None, s: _IntLike | None = None, h: _FloatLike | None = None, channels: _AnyStr | None = None, wmode: _IntLike | None = None, wref: _FloatLike | None = None, rclip: VideoNode | None = None, device_id: _IntLike | None = None, num_streams: _IntLike | None = None) -> VideoNode: ...

# </implementation/nlm_cuda>

# <implementation/nlm_ispc>
class _nlm_ispc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, clip: VideoNode, d: _IntLike | None = None, a: _IntLike | None = None, s: _IntLike | None = None, h: _FloatLike | None = None, channels: _AnyStr | None = None, wmode: _IntLike | None = None, wref: _FloatLike | None = None, rclip: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, d: _IntLike | None = None, a: _IntLike | None = None, s: _IntLike | None = None, h: _FloatLike | None = None, channels: _AnyStr | None = None, wmode: _IntLike | None = None, wref: _FloatLike | None = None, rclip: VideoNode | None = None) -> VideoNode: ...

# </implementation/nlm_ispc>

# <implementation/noise>
class _noise:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Add(self, clip: VideoNode, var: _FloatLike | None = None, uvar: _FloatLike | None = None, type: _IntLike | None = None, hcorr: _FloatLike | None = None, vcorr: _FloatLike | None = None, xsize: _FloatLike | None = None, ysize: _FloatLike | None = None, scale: _FloatLike | None = None, seed: _IntLike | None = None, constant: _IntLike | None = None, every: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Add(self, var: _FloatLike | None = None, uvar: _FloatLike | None = None, type: _IntLike | None = None, hcorr: _FloatLike | None = None, vcorr: _FloatLike | None = None, xsize: _FloatLike | None = None, ysize: _FloatLike | None = None, scale: _FloatLike | None = None, seed: _IntLike | None = None, constant: _IntLike | None = None, every: _IntLike | None = None, opt: _IntLike | None = None) -> VideoNode: ...

# </implementation/noise>

# <implementation/placebo>
class _placebo:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, clip: VideoNode, planes: _IntLike | None = None, iterations: _IntLike | None = None, threshold: _FloatLike | None = None, radius: _FloatLike | None = None, grain: _FloatLike | None = None, dither: _IntLike | None = None, dither_algo: _IntLike | None = None, log_level: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Resample(self, clip: VideoNode, width: _IntLike, height: _IntLike, filter: _AnyStr | None = None, clamp: _FloatLike | None = None, blur: _FloatLike | None = None, taper: _FloatLike | None = None, radius: _FloatLike | None = None, param1: _FloatLike | None = None, param2: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, sx: _FloatLike | None = None, sy: _FloatLike | None = None, antiring: _FloatLike | None = None, sigmoidize: _IntLike | None = None, sigmoid_center: _FloatLike | None = None, sigmoid_slope: _FloatLike | None = None, linearize: _IntLike | None = None, trc: _IntLike | None = None, min_luma: _FloatLike | None = None, log_level: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Shader(self, clip: VideoNode, shader: _AnyStr | None = None, width: _IntLike | None = None, height: _IntLike | None = None, chroma_loc: _IntLike | None = None, matrix: _IntLike | None = None, trc: _IntLike | None = None, linearize: _IntLike | None = None, sigmoidize: _IntLike | None = None, sigmoid_center: _FloatLike | None = None, sigmoid_slope: _FloatLike | None = None, antiring: _FloatLike | None = None, filter: _AnyStr | None = None, clamp: _FloatLike | None = None, blur: _FloatLike | None = None, taper: _FloatLike | None = None, radius: _FloatLike | None = None, param1: _FloatLike | None = None, param2: _FloatLike | None = None, shader_s: _AnyStr | None = None, log_level: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tonemap(self, clip: VideoNode, src_csp: _IntLike | None = None, dst_csp: _IntLike | None = None, dst_prim: _IntLike | None = None, src_max: _FloatLike | None = None, src_min: _FloatLike | None = None, dst_max: _FloatLike | None = None, dst_min: _FloatLike | None = None, dynamic_peak_detection: _IntLike | None = None, smoothing_period: _FloatLike | None = None, scene_threshold_low: _FloatLike | None = None, scene_threshold_high: _FloatLike | None = None, percentile: _FloatLike | None = None, gamut_mapping: _IntLike | None = None, tone_mapping_function: _IntLike | None = None, tone_mapping_function_s: _AnyStr | None = None, tone_mapping_param: _FloatLike | None = None, metadata: _IntLike | None = None, use_dovi: _IntLike | None = None, visualize_lut: _IntLike | None = None, show_clipping: _IntLike | None = None, contrast_recovery: _FloatLike | None = None, log_level: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, planes: _IntLike | None = None, iterations: _IntLike | None = None, threshold: _FloatLike | None = None, radius: _FloatLike | None = None, grain: _FloatLike | None = None, dither: _IntLike | None = None, dither_algo: _IntLike | None = None, log_level: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Resample(self, width: _IntLike, height: _IntLike, filter: _AnyStr | None = None, clamp: _FloatLike | None = None, blur: _FloatLike | None = None, taper: _FloatLike | None = None, radius: _FloatLike | None = None, param1: _FloatLike | None = None, param2: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, sx: _FloatLike | None = None, sy: _FloatLike | None = None, antiring: _FloatLike | None = None, sigmoidize: _IntLike | None = None, sigmoid_center: _FloatLike | None = None, sigmoid_slope: _FloatLike | None = None, linearize: _IntLike | None = None, trc: _IntLike | None = None, min_luma: _FloatLike | None = None, log_level: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Shader(self, shader: _AnyStr | None = None, width: _IntLike | None = None, height: _IntLike | None = None, chroma_loc: _IntLike | None = None, matrix: _IntLike | None = None, trc: _IntLike | None = None, linearize: _IntLike | None = None, sigmoidize: _IntLike | None = None, sigmoid_center: _FloatLike | None = None, sigmoid_slope: _FloatLike | None = None, antiring: _FloatLike | None = None, filter: _AnyStr | None = None, clamp: _FloatLike | None = None, blur: _FloatLike | None = None, taper: _FloatLike | None = None, radius: _FloatLike | None = None, param1: _FloatLike | None = None, param2: _FloatLike | None = None, shader_s: _AnyStr | None = None, log_level: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tonemap(self, src_csp: _IntLike | None = None, dst_csp: _IntLike | None = None, dst_prim: _IntLike | None = None, src_max: _FloatLike | None = None, src_min: _FloatLike | None = None, dst_max: _FloatLike | None = None, dst_min: _FloatLike | None = None, dynamic_peak_detection: _IntLike | None = None, smoothing_period: _FloatLike | None = None, scene_threshold_low: _FloatLike | None = None, scene_threshold_high: _FloatLike | None = None, percentile: _FloatLike | None = None, gamut_mapping: _IntLike | None = None, tone_mapping_function: _IntLike | None = None, tone_mapping_function_s: _AnyStr | None = None, tone_mapping_param: _FloatLike | None = None, metadata: _IntLike | None = None, use_dovi: _IntLike | None = None, visualize_lut: _IntLike | None = None, show_clipping: _IntLike | None = None, contrast_recovery: _FloatLike | None = None, log_level: _IntLike | None = None) -> VideoNode: ...

# </implementation/placebo>

# <implementation/resize>
class _resize:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, clip: VideoNode, filter: _AnyStr | None = None, tff: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, filter: _AnyStr | None = None, tff: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, approximate_gamma: _IntLike | None = None) -> VideoNode: ...

# </implementation/resize>

# <implementation/resize2>
class _resize2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, clip: VideoNode, filter: _AnyStr | None = None, tff: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Custom(self, clip: VideoNode, custom_kernel: _VSCallback_resize2_Custom_custom_kernel, taps: _IntLike, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, clip: VideoNode, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, filter: _AnyStr | None = None, tff: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Custom(self, custom_kernel: _VSCallback_resize2_Custom_custom_kernel, taps: _IntLike, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None, range_s: _AnyStr | None = None, chromaloc: _IntLike | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: _IntLike | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: _IntLike | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: _IntLike | None = None, primaries_in_s: _AnyStr | None = None, range_in: _IntLike | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: _IntLike | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: _FloatLike | None = None, filter_param_b: _FloatLike | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: _FloatLike | None = None, filter_param_b_uv: _FloatLike | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: _IntLike | None = None, src_left: _FloatLike | None = None, src_top: _FloatLike | None = None, src_width: _FloatLike | None = None, src_height: _FloatLike | None = None, nominal_luminance: _FloatLike | None = None, force: _IntLike | None = None, force_h: _IntLike | None = None, force_v: _IntLike | None = None, blur: _FloatLike | None = None) -> VideoNode: ...

# </implementation/resize2>

# <implementation/sangnom>
class _sangnom:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def SangNom(self, clip: VideoNode, order: _IntLike | None = None, dh: _IntLike | None = None, aa: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def SangNom(self, order: _IntLike | None = None, dh: _IntLike | None = None, aa: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

# </implementation/sangnom>

# <implementation/scxvid>
class _scxvid:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Scxvid(self, clip: VideoNode, log: _AnyStr | None = None, use_slices: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Scxvid(self, log: _AnyStr | None = None, use_slices: _IntLike | None = None) -> VideoNode: ...

# </implementation/scxvid>

# <implementation/sneedif>
_ReturnDict_sneedif_DeviceInfo = TypedDict("_ReturnDict_sneedif_DeviceInfo", {"name": _AnyStr, "vendor": _AnyStr, "profile": _AnyStr, "version": _AnyStr, "max_compute_units": int, "max_work_group_size": int, "max_work_item_sizes": _IntLike | list[_IntLike], "image2D_max_width": int, "image2D_max_height": int, "image_support": int, "global_memory_cache_type": _AnyStr, "global_memory_cache": int, "global_memory_size": int, "max_constant_buffer_size": int, "max_constant_arguments": int, "local_memory_type": _AnyStr, "local_memory_size": int, "available": int, "compiler_available": int, "linker_available": int, "opencl_c_version": _AnyStr, "image_max_buffer_size": int})
_ReturnDict_sneedif_ListDevices = TypedDict("_ReturnDict_sneedif_ListDevices", {"numDevices": int, "deviceNames": _AnyStr | list[_AnyStr], "platformNames": _AnyStr | list[_AnyStr]})
_ReturnDict_sneedif_PlatformInfo = TypedDict("_ReturnDict_sneedif_PlatformInfo", {"profile": _AnyStr, "version": _AnyStr, "name": _AnyStr, "vendor": _AnyStr})

class _sneedif:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DeviceInfo(self, device: _IntLike | None = None) -> _ReturnDict_sneedif_DeviceInfo: ...
            @_Wrapper.Function
            def ListDevices(self) -> _ReturnDict_sneedif_ListDevices: ...
            @_Wrapper.Function
            def NNEDI3(self, clip: VideoNode, field: _IntLike, dh: _IntLike | None = None, dw: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, nsize: _IntLike | None = None, nns: _IntLike | None = None, qual: _IntLike | None = None, etype: _IntLike | None = None, pscrn: _IntLike | None = None, transpose_first: _IntLike | None = None, device: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlatformInfo(self, device: _IntLike | None = None) -> _ReturnDict_sneedif_PlatformInfo: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NNEDI3(self, field: _IntLike, dh: _IntLike | None = None, dw: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, nsize: _IntLike | None = None, nns: _IntLike | None = None, qual: _IntLike | None = None, etype: _IntLike | None = None, pscrn: _IntLike | None = None, transpose_first: _IntLike | None = None, device: _IntLike | None = None) -> VideoNode: ...

# </implementation/sneedif>

# <implementation/std>
class _std:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AddBorders(self, clip: VideoNode, left: _IntLike | None = None, right: _IntLike | None = None, top: _IntLike | None = None, bottom: _IntLike | None = None, color: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AssumeFPS(self, clip: VideoNode, src: VideoNode | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AssumeSampleRate(self, clip: AudioNode, src: AudioNode | None = None, samplerate: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioGain(self, clip: AudioNode, gain: _FloatLike | _SequenceLike[_FloatLike] | None = None, overflow_error: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioLoop(self, clip: AudioNode, times: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioMix(self, clips: AudioNode | _SequenceLike[AudioNode], matrix: _FloatLike | _SequenceLike[_FloatLike], channels_out: _IntLike | _SequenceLike[_IntLike], overflow_error: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioReverse(self, clip: AudioNode) -> AudioNode: ...
            @_Wrapper.Function
            def AudioSplice(self, clips: AudioNode | _SequenceLike[AudioNode]) -> AudioNode: ...
            @_Wrapper.Function
            def AudioTrim(self, clip: AudioNode, first: _IntLike | None = None, last: _IntLike | None = None, length: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AverageFrames(self, clips: VideoNode | _SequenceLike[VideoNode], weights: _FloatLike | _SequenceLike[_FloatLike], scale: _FloatLike | None = None, scenechange: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Binarize(self, clip: VideoNode, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, v0: _FloatLike | _SequenceLike[_FloatLike] | None = None, v1: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BinarizeMask(self, clip: VideoNode, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, v0: _FloatLike | _SequenceLike[_FloatLike] | None = None, v1: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlankAudio(self, clip: AudioNode | None = None, channels: _IntLike | _SequenceLike[_IntLike] | None = None, bits: _IntLike | None = None, sampletype: _IntLike | None = None, samplerate: _IntLike | None = None, length: _IntLike | None = None, keep: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def BlankClip(self, clip: VideoNode | None = None, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, length: _IntLike | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None, color: _FloatLike | _SequenceLike[_FloatLike] | None = None, keep: _IntLike | None = None, varsize: _IntLike | None = None, varformat: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, hradius: _IntLike | None = None, hpasses: _IntLike | None = None, vradius: _IntLike | None = None, vpasses: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Cache(self, clip: VideoNode, size: _IntLike | None = None, fixed: _IntLike | None = None, make_linear: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ClipToProp(self, clip: VideoNode, mclip: VideoNode, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Convolution(self, clip: VideoNode, matrix: _FloatLike | _SequenceLike[_FloatLike], bias: _FloatLike | None = None, divisor: _FloatLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, saturate: _IntLike | None = None, mode: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CopyFrameProps(self, clip: VideoNode, prop_src: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Crop(self, clip: VideoNode, left: _IntLike | None = None, right: _IntLike | None = None, top: _IntLike | None = None, bottom: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropAbs(self, clip: VideoNode, width: _IntLike, height: _IntLike, left: _IntLike | None = None, top: _IntLike | None = None, x: _IntLike | None = None, y: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropRel(self, clip: VideoNode, left: _IntLike | None = None, right: _IntLike | None = None, top: _IntLike | None = None, bottom: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Deflate(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DeleteFrames(self, clip: VideoNode, frames: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def DoubleWeave(self, clip: VideoNode, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DuplicateFrames(self, clip: VideoNode, frames: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, clips: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr], format: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlipHorizontal(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def FlipVertical(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper_Core_bound_FrameEval.Function
            def FrameEval(self, clip: VideoNode, eval: _VSCallback_std_FrameEval_eval, prop_src: VideoNode | _SequenceLike[VideoNode] | None = None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FreezeFrames(self, clip: VideoNode, first: _IntLike | _SequenceLike[_IntLike] | None = None, last: _IntLike | _SequenceLike[_IntLike] | None = None, replacement: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Inflate(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Interleave(self, clips: VideoNode | _SequenceLike[VideoNode], extend: _IntLike | None = None, mismatch: _IntLike | None = None, modify_duration: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Invert(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InvertMask(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Levels(self, clip: VideoNode, min_in: _FloatLike | _SequenceLike[_FloatLike] | None = None, max_in: _FloatLike | _SequenceLike[_FloatLike] | None = None, gamma: _FloatLike | _SequenceLike[_FloatLike] | None = None, min_out: _FloatLike | _SequenceLike[_FloatLike] | None = None, max_out: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, clip: VideoNode, min: _FloatLike | _SequenceLike[_FloatLike] | None = None, max: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LoadAllPlugins(self, path: _AnyStr) -> None: ...
            @_Wrapper.Function
            def LoadPlugin(self, path: _AnyStr, altsearchpath: _IntLike | None = None, forcens: _AnyStr | None = None, forceid: _AnyStr | None = None) -> None: ...
            @_Wrapper.Function
            def Loop(self, clip: VideoNode, times: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, lut: _IntLike | _SequenceLike[_IntLike] | None = None, lutf: _FloatLike | _SequenceLike[_FloatLike] | None = None, function: _VSCallback_std_Lut_function | None = None, bits: _IntLike | None = None, floatout: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut2(self, clipa: VideoNode, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, lut: _IntLike | _SequenceLike[_IntLike] | None = None, lutf: _FloatLike | _SequenceLike[_FloatLike] | None = None, function: _VSCallback_std_Lut2_function | None = None, bits: _IntLike | None = None, floatout: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeDiff(self, clipa: VideoNode, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def MaskedMerge(self, clipa: VideoNode, clipb: VideoNode, mask: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, first_plane: _IntLike | None = None, premultiplied: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Maximum(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None, coordinates: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Merge(self, clipa: VideoNode, clipb: VideoNode, weight: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeDiff(self, clipa: VideoNode, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Minimum(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None, coordinates: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper_Core_bound_ModifyFrame.Function
            def ModifyFrame(self, clip: VideoNode, clips: VideoNode | _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector) -> VideoNode: ...
            @_Wrapper.Function
            def PEMVerifier(self, clip: VideoNode, upper: _FloatLike | _SequenceLike[_FloatLike] | None = None, lower: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneStats(self, clipa: VideoNode, clipb: VideoNode | None = None, plane: _IntLike | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PreMultiply(self, clip: VideoNode, alpha: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropToClip(self, clip: VideoNode, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveFrameProps(self, clip: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Reverse(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def SelectEvery(self, clip: VideoNode, cycle: _IntLike, offsets: _IntLike | _SequenceLike[_IntLike], modify_duration: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SeparateFields(self, clip: VideoNode, tff: _IntLike | None = None, modify_duration: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetAudioCache(self, clip: AudioNode, mode: _IntLike | None = None, fixedsize: _IntLike | None = None, maxsize: _IntLike | None = None, maxhistory: _IntLike | None = None) -> None: ...
            @_Wrapper.Function
            def SetFieldBased(self, clip: VideoNode, value: _IntLike) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProp(self, clip: VideoNode, prop: _AnyStr, intval: _IntLike | _SequenceLike[_IntLike] | None = None, floatval: _FloatLike | _SequenceLike[_FloatLike] | None = None, data: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProps(self, clip: VideoNode, **kwargs: Any) -> VideoNode: ...
            @_Wrapper.Function
            def SetMaxCPU(self, cpu: _AnyStr) -> _AnyStr: ...
            @_Wrapper.Function
            def SetVideoCache(self, clip: VideoNode, mode: _IntLike | None = None, fixedsize: _IntLike | None = None, maxsize: _IntLike | None = None, maxhistory: _IntLike | None = None) -> None: ...
            @_Wrapper.Function
            def ShuffleChannels(self, clips: AudioNode | _SequenceLike[AudioNode], channels_in: _IntLike | _SequenceLike[_IntLike], channels_out: _IntLike | _SequenceLike[_IntLike]) -> AudioNode: ...
            @_Wrapper.Function
            def ShufflePlanes(self, clips: VideoNode | _SequenceLike[VideoNode], planes: _IntLike | _SequenceLike[_IntLike], colorfamily: _IntLike, prop_src: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Splice(self, clips: VideoNode | _SequenceLike[VideoNode], mismatch: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SplitChannels(self, clip: AudioNode) -> AudioNode | list[AudioNode]: ...
            @_Wrapper.Function
            def SplitPlanes(self, clip: VideoNode) -> VideoNode | list[VideoNode]: ...
            @_Wrapper.Function
            def StackHorizontal(self, clips: VideoNode | _SequenceLike[VideoNode]) -> VideoNode: ...
            @_Wrapper.Function
            def StackVertical(self, clips: VideoNode | _SequenceLike[VideoNode]) -> VideoNode: ...
            @_Wrapper.Function
            def TestAudio(self, channels: _IntLike | _SequenceLike[_IntLike] | None = None, bits: _IntLike | None = None, isfloat: _IntLike | None = None, samplerate: _IntLike | None = None, length: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def Transpose(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Trim(self, clip: VideoNode, first: _IntLike | None = None, last: _IntLike | None = None, length: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Turn180(self, clip: VideoNode) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AddBorders(self, left: _IntLike | None = None, right: _IntLike | None = None, top: _IntLike | None = None, bottom: _IntLike | None = None, color: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AssumeFPS(self, src: VideoNode | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AverageFrames(self, weights: _FloatLike | _SequenceLike[_FloatLike], scale: _FloatLike | None = None, scenechange: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Binarize(self, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, v0: _FloatLike | _SequenceLike[_FloatLike] | None = None, v1: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BinarizeMask(self, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, v0: _FloatLike | _SequenceLike[_FloatLike] | None = None, v1: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlankClip(self, width: _IntLike | None = None, height: _IntLike | None = None, format: _IntLike | None = None, length: _IntLike | None = None, fpsnum: _IntLike | None = None, fpsden: _IntLike | None = None, color: _FloatLike | _SequenceLike[_FloatLike] | None = None, keep: _IntLike | None = None, varsize: _IntLike | None = None, varformat: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, hradius: _IntLike | None = None, hpasses: _IntLike | None = None, vradius: _IntLike | None = None, vpasses: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Cache(self, size: _IntLike | None = None, fixed: _IntLike | None = None, make_linear: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ClipToProp(self, mclip: VideoNode, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Convolution(self, matrix: _FloatLike | _SequenceLike[_FloatLike], bias: _FloatLike | None = None, divisor: _FloatLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, saturate: _IntLike | None = None, mode: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CopyFrameProps(self, prop_src: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Crop(self, left: _IntLike | None = None, right: _IntLike | None = None, top: _IntLike | None = None, bottom: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropAbs(self, width: _IntLike, height: _IntLike, left: _IntLike | None = None, top: _IntLike | None = None, x: _IntLike | None = None, y: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropRel(self, left: _IntLike | None = None, right: _IntLike | None = None, top: _IntLike | None = None, bottom: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Deflate(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DeleteFrames(self, frames: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def DoubleWeave(self, tff: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DuplicateFrames(self, frames: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, expr: _AnyStr | _SequenceLike[_AnyStr], format: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlipHorizontal(self) -> VideoNode: ...
            @_Wrapper.Function
            def FlipVertical(self) -> VideoNode: ...
            @_Wrapper_VideoNode_bound_FrameEval.Function
            def FrameEval(self, eval: _VSCallback_std_FrameEval_eval, prop_src: VideoNode | _SequenceLike[VideoNode] | None = None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FreezeFrames(self, first: _IntLike | _SequenceLike[_IntLike] | None = None, last: _IntLike | _SequenceLike[_IntLike] | None = None, replacement: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Inflate(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Interleave(self, extend: _IntLike | None = None, mismatch: _IntLike | None = None, modify_duration: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Invert(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InvertMask(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Levels(self, min_in: _FloatLike | _SequenceLike[_FloatLike] | None = None, max_in: _FloatLike | _SequenceLike[_FloatLike] | None = None, gamma: _FloatLike | _SequenceLike[_FloatLike] | None = None, min_out: _FloatLike | _SequenceLike[_FloatLike] | None = None, max_out: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, min: _FloatLike | _SequenceLike[_FloatLike] | None = None, max: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Loop(self, times: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, lut: _IntLike | _SequenceLike[_IntLike] | None = None, lutf: _FloatLike | _SequenceLike[_FloatLike] | None = None, function: _VSCallback_std_Lut_function | None = None, bits: _IntLike | None = None, floatout: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut2(self, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, lut: _IntLike | _SequenceLike[_IntLike] | None = None, lutf: _FloatLike | _SequenceLike[_FloatLike] | None = None, function: _VSCallback_std_Lut2_function | None = None, bits: _IntLike | None = None, floatout: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeDiff(self, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeFullDiff(self, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def MaskedMerge(self, clipb: VideoNode, mask: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, first_plane: _IntLike | None = None, premultiplied: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Maximum(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None, coordinates: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Merge(self, clipb: VideoNode, weight: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeDiff(self, clipb: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeFullDiff(self, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Minimum(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | None = None, coordinates: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper_VideoNode_bound_ModifyFrame.Function
            def ModifyFrame(self, clips: VideoNode | _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector) -> VideoNode: ...
            @_Wrapper.Function
            def PEMVerifier(self, upper: _FloatLike | _SequenceLike[_FloatLike] | None = None, lower: _FloatLike | _SequenceLike[_FloatLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneStats(self, clipb: VideoNode | None = None, plane: _IntLike | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PreMultiply(self, alpha: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropToClip(self, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveFrameProps(self, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Reverse(self) -> VideoNode: ...
            @_Wrapper.Function
            def SelectEvery(self, cycle: _IntLike, offsets: _IntLike | _SequenceLike[_IntLike], modify_duration: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SeparateFields(self, tff: _IntLike | None = None, modify_duration: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetFieldBased(self, value: _IntLike) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProp(self, prop: _AnyStr, intval: _IntLike | _SequenceLike[_IntLike] | None = None, floatval: _FloatLike | _SequenceLike[_FloatLike] | None = None, data: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProps(self, **kwargs: Any) -> VideoNode: ...
            @_Wrapper.Function
            def SetVideoCache(self, mode: _IntLike | None = None, fixedsize: _IntLike | None = None, maxsize: _IntLike | None = None, maxhistory: _IntLike | None = None) -> None: ...
            @_Wrapper.Function
            def ShufflePlanes(self, planes: _IntLike | _SequenceLike[_IntLike], colorfamily: _IntLike, prop_src: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Splice(self, mismatch: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SplitPlanes(self) -> VideoNode | list[VideoNode]: ...
            @_Wrapper.Function
            def StackHorizontal(self) -> VideoNode: ...
            @_Wrapper.Function
            def StackVertical(self) -> VideoNode: ...
            @_Wrapper.Function
            def Transpose(self) -> VideoNode: ...
            @_Wrapper.Function
            def Trim(self, first: _IntLike | None = None, last: _IntLike | None = None, length: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Turn180(self) -> VideoNode: ...

    class _AudioNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AssumeSampleRate(self, src: AudioNode | None = None, samplerate: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioGain(self, gain: _FloatLike | _SequenceLike[_FloatLike] | None = None, overflow_error: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioLoop(self, times: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioMix(self, matrix: _FloatLike | _SequenceLike[_FloatLike], channels_out: _IntLike | _SequenceLike[_IntLike], overflow_error: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioReverse(self) -> AudioNode: ...
            @_Wrapper.Function
            def AudioSplice(self) -> AudioNode: ...
            @_Wrapper.Function
            def AudioTrim(self, first: _IntLike | None = None, last: _IntLike | None = None, length: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def BlankAudio(self, channels: _IntLike | _SequenceLike[_IntLike] | None = None, bits: _IntLike | None = None, sampletype: _IntLike | None = None, samplerate: _IntLike | None = None, length: _IntLike | None = None, keep: _IntLike | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def SetAudioCache(self, mode: _IntLike | None = None, fixedsize: _IntLike | None = None, maxsize: _IntLike | None = None, maxhistory: _IntLike | None = None) -> None: ...
            @_Wrapper.Function
            def ShuffleChannels(self, channels_in: _IntLike | _SequenceLike[_IntLike], channels_out: _IntLike | _SequenceLike[_IntLike]) -> AudioNode: ...
            @_Wrapper.Function
            def SplitChannels(self) -> AudioNode | list[AudioNode]: ...

# </implementation/std>

# <implementation/sub>
class _sub:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ImageFile(self, clip: VideoNode, file: _AnyStr, id: _IntLike | None = None, palette: _IntLike | _SequenceLike[_IntLike] | None = None, gray: _IntLike | None = None, info: _IntLike | None = None, flatten: _IntLike | None = None, blend: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Subtitle(self, clip: VideoNode, text: _AnyStr, start: _IntLike | None = None, end: _IntLike | None = None, debuglevel: _IntLike | None = None, fontdir: _AnyStr | None = None, linespacing: _FloatLike | None = None, margins: _IntLike | _SequenceLike[_IntLike] | None = None, sar: _FloatLike | None = None, style: _AnyStr | None = None, blend: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TextFile(self, clip: VideoNode, file: _AnyStr, charset: _AnyStr | None = None, scale: _FloatLike | None = None, debuglevel: _IntLike | None = None, fontdir: _AnyStr | None = None, linespacing: _FloatLike | None = None, margins: _IntLike | _SequenceLike[_IntLike] | None = None, sar: _FloatLike | None = None, style: _AnyStr | None = None, blend: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ImageFile(self, file: _AnyStr, id: _IntLike | None = None, palette: _IntLike | _SequenceLike[_IntLike] | None = None, gray: _IntLike | None = None, info: _IntLike | None = None, flatten: _IntLike | None = None, blend: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Subtitle(self, text: _AnyStr, start: _IntLike | None = None, end: _IntLike | None = None, debuglevel: _IntLike | None = None, fontdir: _AnyStr | None = None, linespacing: _FloatLike | None = None, margins: _IntLike | _SequenceLike[_IntLike] | None = None, sar: _FloatLike | None = None, style: _AnyStr | None = None, blend: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TextFile(self, file: _AnyStr, charset: _AnyStr | None = None, scale: _FloatLike | None = None, debuglevel: _IntLike | None = None, fontdir: _AnyStr | None = None, linespacing: _FloatLike | None = None, margins: _IntLike | _SequenceLike[_IntLike] | None = None, sar: _FloatLike | None = None, style: _AnyStr | None = None, blend: _IntLike | None = None, matrix: _IntLike | None = None, matrix_s: _AnyStr | None = None, transfer: _IntLike | None = None, transfer_s: _AnyStr | None = None, primaries: _IntLike | None = None, primaries_s: _AnyStr | None = None, range: _IntLike | None = None) -> VideoNode: ...

# </implementation/sub>

# <implementation/tcanny>
class _tcanny:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def TCanny(self, clip: VideoNode, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigma_v: _FloatLike | _SequenceLike[_FloatLike] | None = None, t_h: _FloatLike | None = None, t_l: _FloatLike | None = None, mode: _IntLike | None = None, op: _IntLike | None = None, scale: _FloatLike | None = None, opt: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def TCanny(self, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigma_v: _FloatLike | _SequenceLike[_FloatLike] | None = None, t_h: _FloatLike | None = None, t_l: _FloatLike | None = None, mode: _IntLike | None = None, op: _IntLike | None = None, scale: _FloatLike | None = None, opt: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...

# </implementation/tcanny>

# <implementation/text>
class _text:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ClipInfo(self, clip: VideoNode, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CoreInfo(self, clip: VideoNode | None = None, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameNum(self, clip: VideoNode, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameProps(self, clip: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, clip: VideoNode, text: _AnyStr, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ClipInfo(self, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CoreInfo(self, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameNum(self, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameProps(self, props: _AnyStr | _SequenceLike[_AnyStr] | None = None, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, text: _AnyStr, alignment: _IntLike | None = None, scale: _IntLike | None = None) -> VideoNode: ...

# </implementation/text>

# <implementation/vivtc>
class _vivtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VDecimate(self, clip: VideoNode, cycle: _IntLike | None = None, chroma: _IntLike | None = None, dupthresh: _FloatLike | None = None, scthresh: _FloatLike | None = None, blockx: _IntLike | None = None, blocky: _IntLike | None = None, clip2: VideoNode | None = None, ovr: _AnyStr | None = None, dryrun: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFM(self, clip: VideoNode, order: _IntLike, field: _IntLike | None = None, mode: _IntLike | None = None, mchroma: _IntLike | None = None, cthresh: _IntLike | None = None, mi: _IntLike | None = None, chroma: _IntLike | None = None, blockx: _IntLike | None = None, blocky: _IntLike | None = None, y0: _IntLike | None = None, y1: _IntLike | None = None, scthresh: _FloatLike | None = None, micmatch: _IntLike | None = None, micout: _IntLike | None = None, clip2: VideoNode | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VDecimate(self, cycle: _IntLike | None = None, chroma: _IntLike | None = None, dupthresh: _FloatLike | None = None, scthresh: _FloatLike | None = None, blockx: _IntLike | None = None, blocky: _IntLike | None = None, clip2: VideoNode | None = None, ovr: _AnyStr | None = None, dryrun: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFM(self, order: _IntLike, field: _IntLike | None = None, mode: _IntLike | None = None, mchroma: _IntLike | None = None, cthresh: _IntLike | None = None, mi: _IntLike | None = None, chroma: _IntLike | None = None, blockx: _IntLike | None = None, blocky: _IntLike | None = None, y0: _IntLike | None = None, y1: _IntLike | None = None, scthresh: _FloatLike | None = None, micmatch: _IntLike | None = None, micout: _IntLike | None = None, clip2: VideoNode | None = None) -> VideoNode: ...

# </implementation/vivtc>

# <implementation/vszip>
class _vszip:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AdaptiveBinarize(self, clip: VideoNode, clip2: VideoNode, c: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilateral(self, clip: VideoNode, ref: VideoNode | None = None, sigmaS: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigmaR: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, algorithm: _IntLike | _SequenceLike[_IntLike] | None = None, PBFICnum: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None, hradius: _IntLike | None = None, hpasses: _IntLike | None = None, vradius: _IntLike | None = None, vpasses: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CLAHE(self, clip: VideoNode, limit: _IntLike | None = None, tiles: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Checkmate(self, clip: VideoNode, thr: _IntLike | None = None, tmax: _IntLike | None = None, tthr2: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ColorMap(self, clip: VideoNode, color: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CombMask(self, clip: VideoNode, cthresh: _IntLike | None = None, mthresh: _IntLike | None = None, expand: _IntLike | None = None, metric: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CombMaskMT(self, clip: VideoNode, thY1: _IntLike | None = None, thY2: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ImageRead(self, path: _AnyStr | _SequenceLike[_AnyStr], validate: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LimitFilter(self, flt: VideoNode, src: VideoNode, ref: VideoNode | None = None, dark_thr: _FloatLike | _SequenceLike[_FloatLike] | None = None, bright_thr: _FloatLike | _SequenceLike[_FloatLike] | None = None, elast: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, clip: VideoNode, min: _FloatLike | _SequenceLike[_FloatLike] | None = None, max: _FloatLike | _SequenceLike[_FloatLike] | None = None, tv_range: _IntLike | None = None, mask: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Metrics(self, reference: VideoNode, distorted: VideoNode, mode: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PackRGB(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneAverage(self, clipa: VideoNode, exclude: _IntLike | _SequenceLike[_IntLike], clipb: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneMinMax(self, clipa: VideoNode, minthr: _FloatLike | None = None, maxthr: _FloatLike | None = None, clipb: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RFS(self, clipa: VideoNode, clipb: VideoNode, frames: _IntLike | _SequenceLike[_IntLike], mismatch: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SSIMULACRA2(self, reference: VideoNode, distorted: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def XPSNR(self, reference: VideoNode, distorted: VideoNode, temporal: _IntLike | None = None, verbose: _IntLike | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AdaptiveBinarize(self, clip2: VideoNode, c: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilateral(self, ref: VideoNode | None = None, sigmaS: _FloatLike | _SequenceLike[_FloatLike] | None = None, sigmaR: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, algorithm: _IntLike | _SequenceLike[_IntLike] | None = None, PBFICnum: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None, hradius: _IntLike | None = None, hpasses: _IntLike | None = None, vradius: _IntLike | None = None, vpasses: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CLAHE(self, limit: _IntLike | None = None, tiles: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Checkmate(self, thr: _IntLike | None = None, tmax: _IntLike | None = None, tthr2: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ColorMap(self, color: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CombMask(self, cthresh: _IntLike | None = None, mthresh: _IntLike | None = None, expand: _IntLike | None = None, metric: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CombMaskMT(self, thY1: _IntLike | None = None, thY2: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LimitFilter(self, src: VideoNode, ref: VideoNode | None = None, dark_thr: _FloatLike | _SequenceLike[_FloatLike] | None = None, bright_thr: _FloatLike | _SequenceLike[_FloatLike] | None = None, elast: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, min: _FloatLike | _SequenceLike[_FloatLike] | None = None, max: _FloatLike | _SequenceLike[_FloatLike] | None = None, tv_range: _IntLike | None = None, mask: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Metrics(self, distorted: VideoNode, mode: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PackRGB(self) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneAverage(self, exclude: _IntLike | _SequenceLike[_IntLike], clipb: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneMinMax(self, minthr: _FloatLike | None = None, maxthr: _FloatLike | None = None, clipb: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RFS(self, clipb: VideoNode, frames: _IntLike | _SequenceLike[_IntLike], mismatch: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SSIMULACRA2(self, distorted: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def XPSNR(self, distorted: VideoNode, temporal: _IntLike | None = None, verbose: _IntLike | None = None) -> VideoNode: ...

# </implementation/vszip>

# <implementation/wnnm>
class _wnnm:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike], internal: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...
            @_Wrapper.Function
            def WNNM(self, clip: VideoNode, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, residual: _IntLike | None = None, adaptive_aggregation: _IntLike | None = None, rclip: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def WNNMRaw(self, clip: VideoNode, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, residual: _IntLike | None = None, adaptive_aggregation: _IntLike | None = None, rclip: VideoNode | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: _IntLike | _SequenceLike[_IntLike], internal: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def WNNM(self, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, residual: _IntLike | None = None, adaptive_aggregation: _IntLike | None = None, rclip: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def WNNMRaw(self, sigma: _FloatLike | _SequenceLike[_FloatLike] | None = None, block_size: _IntLike | None = None, block_step: _IntLike | None = None, group_size: _IntLike | None = None, bm_range: _IntLike | None = None, radius: _IntLike | None = None, ps_num: _IntLike | None = None, ps_range: _IntLike | None = None, residual: _IntLike | None = None, adaptive_aggregation: _IntLike | None = None, rclip: VideoNode | None = None) -> VideoNode: ...

# </implementation/wnnm>

# <implementation/wwxd>
class _wwxd:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def WWXD(self, clip: VideoNode) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def WWXD(self) -> VideoNode: ...

# </implementation/wwxd>

# <implementation/znedi3>
class _znedi3:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def nnedi3(self, clip: VideoNode, field: _IntLike, dh: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, nsize: _IntLike | None = None, nns: _IntLike | None = None, qual: _IntLike | None = None, etype: _IntLike | None = None, pscrn: _IntLike | None = None, opt: _IntLike | None = None, int16_prescreener: _IntLike | None = None, int16_predictor: _IntLike | None = None, exp: _IntLike | None = None, show_mask: _IntLike | None = None, x_nnedi3_weights_bin: _AnyStr | None = None, x_cpu: _AnyStr | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def nnedi3(self, field: _IntLike, dh: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, nsize: _IntLike | None = None, nns: _IntLike | None = None, qual: _IntLike | None = None, etype: _IntLike | None = None, pscrn: _IntLike | None = None, opt: _IntLike | None = None, int16_prescreener: _IntLike | None = None, int16_predictor: _IntLike | None = None, exp: _IntLike | None = None, show_mask: _IntLike | None = None, x_nnedi3_weights_bin: _AnyStr | None = None, x_cpu: _AnyStr | None = None) -> VideoNode: ...

# </implementation/znedi3>

# <implementation/zsmooth>
class _zsmooth:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BackwardClense(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CCD(self, clip: VideoNode, threshold: _FloatLike | None = None, temporal_radius: _IntLike | None = None, points: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Clense(self, clip: VideoNode, previous: VideoNode | None = None, next: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DegrainMedian(self, clip: VideoNode, limit: _FloatLike | _SequenceLike[_FloatLike] | None = None, mode: _IntLike | _SequenceLike[_IntLike] | None = None, interlaced: _IntLike | None = None, norow: _IntLike | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothST(self, clip: VideoNode, temporal_threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, spatial_threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothT(self, clip: VideoNode, temporal_threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ForwardClense(self, clip: VideoNode, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InterQuartileMean(self, clip: VideoNode, radius: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, clip: VideoNode, radius: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveGrain(self, clip: VideoNode, mode: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Repair(self, clip: VideoNode, repairclip: VideoNode, mode: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def SmartMedian(self, clip: VideoNode, radius: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, scalep: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TTempSmooth(self, clip: VideoNode, maxr: _IntLike | None = None, thresh: _IntLike | _SequenceLike[_IntLike] | None = None, mdiff: _IntLike | _SequenceLike[_IntLike] | None = None, strength: _IntLike | None = None, scthresh: _FloatLike | None = None, fp: _IntLike | None = None, pfclip: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalMedian(self, clip: VideoNode, radius: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scenechange: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalRepair(self, clip: VideoNode, repairclip: VideoNode, mode: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalSoften(self, clip: VideoNode, radius: _IntLike | None = None, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, scenechange: _IntLike | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VerticalCleaner(self, clip: VideoNode, mode: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BackwardClense(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CCD(self, threshold: _FloatLike | None = None, temporal_radius: _IntLike | None = None, points: _IntLike | _SequenceLike[_IntLike] | None = None, scale: _FloatLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Clense(self, previous: VideoNode | None = None, next: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DegrainMedian(self, limit: _FloatLike | _SequenceLike[_FloatLike] | None = None, mode: _IntLike | _SequenceLike[_IntLike] | None = None, interlaced: _IntLike | None = None, norow: _IntLike | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothST(self, temporal_threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, spatial_threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothT(self, temporal_threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ForwardClense(self, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InterQuartileMean(self, radius: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, radius: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveGrain(self, mode: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def Repair(self, repairclip: VideoNode, mode: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...
            @_Wrapper.Function
            def SmartMedian(self, radius: _IntLike | _SequenceLike[_IntLike] | None = None, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, scalep: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TTempSmooth(self, maxr: _IntLike | None = None, thresh: _IntLike | _SequenceLike[_IntLike] | None = None, mdiff: _IntLike | _SequenceLike[_IntLike] | None = None, strength: _IntLike | None = None, scthresh: _FloatLike | None = None, fp: _IntLike | None = None, pfclip: VideoNode | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalMedian(self, radius: _IntLike | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None, scenechange: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalRepair(self, repairclip: VideoNode, mode: _IntLike | _SequenceLike[_IntLike] | None = None, planes: _IntLike | _SequenceLike[_IntLike] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalSoften(self, radius: _IntLike | None = None, threshold: _FloatLike | _SequenceLike[_FloatLike] | None = None, scenechange: _IntLike | None = None, scalep: _IntLike | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VerticalCleaner(self, mode: _IntLike | _SequenceLike[_IntLike]) -> VideoNode: ...

# </implementation/zsmooth>

# </plugins/implementations>

class VideoOutputTuple(NamedTuple):
    clip: VideoNode
    alpha: VideoNode | None
    alt_output: Literal[0, 1, 2]

def clear_output(index: _IntLike = 0) -> None: ...
def clear_outputs() -> None: ...
def get_outputs() -> MappingProxyType[int, VideoOutputTuple | AudioNode]: ...
def get_output(index: _IntLike = 0) -> VideoOutputTuple | AudioNode: ...

def construct_signature(
    signature: str | Function, return_signature: str, injected: bool | None = None, name: str | None = None
) -> Signature: ...
def _construct_type(signature: str) -> Any: ...
def _construct_parameter(signature: str) -> Any: ...
def _construct_repr_wrap(value: str | Enum | VideoFormat | Iterator[str]) -> str: ...
def _construct_repr(obj: Any, **kwargs: Any) -> str: ...
