# ruff: noqa: RUF100, E501, PYI002, PYI029, PYI046, PYI047, N801, N802, N803, N805, I001
from concurrent.futures import Future
from ctypes import c_void_p
from enum import Enum, IntEnum, IntFlag
from fractions import Fraction
from inspect import Signature
from logging import Handler, LogRecord, StreamHandler
from types import MappingProxyType, TracebackType
from typing import Any, Callable, Concatenate, Final, IO, Iterable, Iterator, Literal, Mapping, MutableMapping, NamedTuple, Protocol, Self, TextIO, TypedDict, final, overload
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

type _VSValueSingle = (
    int | float | _AnyStr | RawFrame | VideoFrame | AudioFrame | RawNode | VideoNode | AudioNode | Callable[..., Any]
)

type _VSValueIterable = (
    _SupportsIter[int]
    | _SupportsIter[_AnyStr]
    | _SupportsIter[float]
    | _SupportsIter[RawFrame]
    | _SupportsIter[VideoFrame]
    | _SupportsIter[AudioFrame]
    | _SupportsIter[RawNode]
    | _SupportsIter[VideoNode]
    | _SupportsIter[AudioNode]
    | _SupportsIter[Callable[..., Any]]
    | _GetItemIterable[int]
    | _GetItemIterable[float]
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

class _SupportsIter[_T](Protocol):
    def __iter__(self) -> Iterator[_T]: ...

class _SequenceLike[_T](Protocol):
    def __iter__(self) -> Iterator[_T]: ...
    def __len__(self) -> int: ...

class _GetItemIterable[_T](Protocol):
    def __getitem__(self, i: int, /) -> _T: ...

class _SupportsKeysAndGetItem[_KT, _VT](Protocol):
    def __getitem__(self, key: _KT, /) -> _VT: ...
    def keys(self) -> Iterable[_KT]: ...

class _SupportsInt(Protocol):
    def __int__(self) -> int: ...

class _VSCallback(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> _VSValue: ...

# Known callback signatures
# _VSCallback_{plugin_namespace}_{Function_name}_{parameter_name}
class _VSCallback_akarin_PropExpr_dict(Protocol):
    def __call__(
        self,
    ) -> Mapping[
        str,
        int
        | float
        | _AnyStr
        | _SupportsIter[int]
        | _SupportsIter[_AnyStr]
        | _SupportsIter[float]
        | _GetItemIterable[int]
        | _GetItemIterable[float]
        | _GetItemIterable[_AnyStr],
    ]: ...

class _VSCallback_descale_Decustom_custom_kernel(Protocol):
    def __call__(self, *, x: float) -> float: ...

class _VSCallback_descale_ScaleCustom_custom_kernel(Protocol):
    def __call__(self, *, x: float) -> float: ...

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
    def __call__(self, *, x: int) -> int: ...

class _VSCallback_std_Lut_function_1(Protocol):
    def __call__(self, *, x: float) -> float: ...

type _VSCallback_std_Lut_function = _VSCallback_std_Lut_function_0 | _VSCallback_std_Lut_function_1  # noqa: PYI047

class _VSCallback_std_Lut2_function_0(Protocol):
    def __call__(self, *, x: int, y: int) -> int: ...

class _VSCallback_std_Lut2_function_1(Protocol):
    def __call__(self, *, x: float, y: float) -> float: ...

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
    def __call__(self, *, x: float) -> float: ...

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
    def create_environment(self, flags: int = 0) -> EnvironmentData: ...
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
        bits_per_sample: int = ...,
        subsampling_w: int = ...,
        subsampling_h: int = ...,
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
    class Function[**_P, _R](_VSFunction):
        def __init__[_PluginT: Plugin](self, function: Callable[Concatenate[_PluginT, _P], _R]) -> None: ...
        def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R: ...

class _Wrapper_Core_bound_FrameEval:
    class Function(_VSFunction):
        def __init__[_PluginT: Plugin](self, function: Callable[Concatenate[_PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(self, clip: VideoNode, eval: _VSCallback_std_FrameEval_eval_0, prop_src: None = None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, clip: VideoNode, eval: _VSCallback_std_FrameEval_eval_1, prop_src: VideoNode, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, clip: VideoNode, eval: _VSCallback_std_FrameEval_eval_2, prop_src: _SequenceLike[VideoNode], clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, clip: VideoNode, eval: _VSCallback_std_FrameEval_eval_3, prop_src: VideoNode | _SequenceLike[VideoNode], clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, clip: VideoNode, eval: Func | _VSCallback_std_FrameEval_eval, prop_src: VideoNode | _SequenceLike[VideoNode] | None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...

class _Wrapper_VideoNode_bound_FrameEval:
    class Function(_VSFunction):
        def __init__[_PluginT: Plugin](self, function: Callable[Concatenate[_PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(self, eval: _VSCallback_std_FrameEval_eval_0, prop_src: None = None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, eval: _VSCallback_std_FrameEval_eval_1, prop_src: VideoNode, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, eval: _VSCallback_std_FrameEval_eval_2, prop_src: _SequenceLike[VideoNode], clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, eval: _VSCallback_std_FrameEval_eval_3, prop_src: VideoNode | _SequenceLike[VideoNode], clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
        @overload
        def __call__(self, eval: Func | _VSCallback_std_FrameEval_eval, prop_src: VideoNode | _SequenceLike[VideoNode] | None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...

class _Wrapper_Core_bound_ModifyFrame:
    class Function(_VSFunction):
        def __init__[_PluginT: Plugin](self, function: Callable[Concatenate[_PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(self, clip: VideoNode, clips: VideoNode, selector: _VSCallback_std_ModifyFrame_selector_0) -> VideoNode: ...
        @overload
        def __call__(self, clip: VideoNode, clips: _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector_1) -> VideoNode: ...
        @overload
        def __call__(self, clip: VideoNode, clips: VideoNode | _SequenceLike[VideoNode], selector: Func | _VSCallback_std_ModifyFrame_selector) -> VideoNode: ...

class _Wrapper_VideoNode_bound_ModifyFrame:
    class Function(_VSFunction):
        def __init__[_PluginT: Plugin](self, function: Callable[Concatenate[_PluginT, ...], VideoNode]) -> None: ...
        @overload
        def __call__(self, clips: VideoNode, selector: _VSCallback_std_ModifyFrame_selector_0) -> VideoNode: ...
        @overload
        def __call__(self, clips: _SequenceLike[VideoNode], selector: _VSCallback_std_ModifyFrame_selector_1) -> VideoNode: ...
        @overload
        def __call__(self, clips: VideoNode | _SequenceLike[VideoNode], selector: Func | _VSCallback_std_ModifyFrame_selector) -> VideoNode: ...

class FramePtr: ...

# These memoryview-likes don't exist at runtime.
class _video_view(memoryview):  # type: ignore[misc]
    def __getitem__(self, index: tuple[int, int]) -> int | float: ...  # type: ignore[override]
    def __setitem__(self, index: tuple[int, int], other: int | float) -> None: ...  # type: ignore[override]
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def strides(self) -> tuple[int, int]: ...
    @property
    def ndim(self) -> Literal[2]: ...
    @property
    def obj(self) -> FramePtr: ...  # type: ignore[override]
    def tolist(self) -> list[int | float]: ...  # type: ignore[override]

class _audio_view(memoryview):  # type: ignore[misc]
    def __getitem__(self, index: int) -> int | float: ...  # type: ignore[override]
    def __setitem__(self, index: int, other: int | float) -> None: ...  # type: ignore[override]
    @property
    def shape(self) -> tuple[int]: ...
    @property
    def strides(self) -> tuple[int]: ...
    @property
    def ndim(self) -> Literal[1]: ...
    @property
    def obj(self) -> FramePtr: ...  # type: ignore[override]
    def tolist(self) -> list[int | float]: ...  # type: ignore[override]

class RawFrame:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self, exc: type[BaseException] | None = None, val: BaseException | None = None, tb: TracebackType | None = None
    ) -> bool | None: ...
    def __getitem__(self, index: int) -> memoryview: ...
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
    def get_write_ptr(self, plane: int) -> c_void_p: ...
    def get_read_ptr(self, plane: int) -> c_void_p: ...
    def get_stride(self, plane: int) -> int: ...

# Behave like a Sequence
class VideoFrame(RawFrame):
    format: Final[VideoFormat]
    width: Final[int]
    height: Final[int]

    def __getitem__(self, index: int) -> _video_view: ...
    def readchunks(self) -> Iterator[_video_view]: ...

# Behave like a Sequence
class AudioFrame(RawFrame):
    sample_type: Final[SampleType]
    bits_per_sample: Final[int]
    bytes_per_sample: Final[int]
    channel_layout: Final[int]
    num_channels: Final[int]

    def __getitem__(self, index: int) -> _audio_view: ...
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
    def __mul__(self, other: int | Self) -> Self: ...
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
    def get_frame(self, n: int) -> RawFrame: ...
    @overload
    def get_frame_async(self, n: int) -> Future[RawFrame]: ...
    @overload
    def get_frame_async(self, n: int, cb: Callable[[RawFrame | None, Exception | None], None]) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[RawFrame]: ...
    def set_output(self, index: int = 0) -> None: ...
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
    def get_frame(self, n: int) -> VideoFrame: ...
    @overload  # type: ignore[override]
    def get_frame_async(self, n: int) -> Future[VideoFrame]: ...
    @overload
    def get_frame_async(self, n: int, cb: Callable[[VideoFrame | None, Exception | None], None]) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[VideoFrame]: ...
    def set_output(self, index: int = 0, alpha: Self | None = None, alt_output: Literal[0, 1, 2] = 0) -> None: ...
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
    """Sharpen images by warping"""
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
# <attribute/VideoNode_bound/eedi2cuda>
    eedi2cuda: Final[_eedi2cuda._VideoNode_bound.Plugin]
    """EEDI2 filter using CUDA"""
# </attribute/VideoNode_bound/eedi2cuda>
# <attribute/VideoNode_bound/eedi3m>
    eedi3m: Final[_eedi3m._VideoNode_bound.Plugin]
    """Enhanced Edge Directed Interpolation 3"""
# </attribute/VideoNode_bound/eedi3m>
# <attribute/VideoNode_bound/fft3dfilter>
    fft3dfilter: Final[_fft3dfilter._VideoNode_bound.Plugin]
    """systems"""
# </attribute/VideoNode_bound/fft3dfilter>
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
    def get_frame(self, n: int) -> AudioFrame: ...
    @overload  # type: ignore[override]
    def get_frame_async(self, n: int) -> Future[AudioFrame]: ...
    @overload
    def get_frame_async(self, n: int, cb: Callable[[AudioFrame | None, Exception | None], None]) -> None: ...  # pyright: ignore[reportIncompatibleMethodOverride]
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
    def num_threads(self, value: int) -> None: ...
    @property
    def max_cache_size(self) -> int: ...
    @max_cache_size.setter
    def max_cache_size(self, mb: int) -> None: ...
    @property
    def used_cache_size(self) -> int: ...
    @property
    def flags(self) -> int: ...
    def plugins(self) -> Iterator[Plugin]: ...
    def query_video_format(
        self,
        color_family: int,
        sample_type: int,
        bits_per_sample: int,
        subsampling_w: int = 0,
        subsampling_h: int = 0,
    ) -> VideoFormat: ...
    def get_video_format(self, id: _SupportsInt) -> VideoFormat: ...
    def create_video_frame(self, format: VideoFormat, width: int, height: int) -> VideoFrame: ...
    def log_message(self, message_type: int, message: str) -> None: ...
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
    """Sharpen images by warping"""
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
# <attribute/Core_bound/eedi2cuda>
    eedi2cuda: Final[_eedi2cuda._Core_bound.Plugin]
    """EEDI2 filter using CUDA"""
# </attribute/Core_bound/eedi2cuda>
# <attribute/Core_bound/eedi3m>
    eedi3m: Final[_eedi3m._Core_bound.Plugin]
    """Enhanced Edge Directed Interpolation 3"""
# </attribute/Core_bound/eedi3m>
# <attribute/Core_bound/ffms2>
    ffms2: Final[_ffms2._Core_bound.Plugin]
    """FFmpegSource 2 for VapourSynth"""
# </attribute/Core_bound/ffms2>
# <attribute/Core_bound/fft3dfilter>
    fft3dfilter: Final[_fft3dfilter._Core_bound.Plugin]
    """systems"""
# </attribute/Core_bound/fft3dfilter>
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
            def Mask(self, clip: VideoNode, luma_scaling: float | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Mask(self, luma_scaling: float | None = None) -> VideoNode: ...

# </implementation/adg>

# <implementation/akarin>
_ReturnDict_akarin_Version = TypedDict("_ReturnDict_akarin_Version", {"version": _AnyStr, "expr_backend": _AnyStr, "expr_features": list[_AnyStr], "select_features": list[_AnyStr], "text_features": list[_AnyStr], "tmpl_features": list[_AnyStr]})

class _akarin:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cambi(self, clip: VideoNode, window_size: int | None = None, topk: float | None = None, tvi_threshold: float | None = None, scores: int | None = None, scaling: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLISR(self, clip: VideoNode, scale: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLVFX(self, clip: VideoNode, op: int, scale: float | None = None, strength: float | None = None, output_depth: int | None = None, num_streams: int | None = None, model_dir: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, clips: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr], format: int | None = None, opt: int | None = None, boundary: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropExpr(self, clips: VideoNode | _SequenceLike[VideoNode], dict: Func | _VSCallback_akarin_PropExpr_dict) -> VideoNode: ...
            @_Wrapper.Function
            def Select(self, clip_src: VideoNode | _SequenceLike[VideoNode], prop_src: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, clips: VideoNode | _SequenceLike[VideoNode], text: _AnyStr, alignment: int | None = None, scale: int | None = None, prop: _AnyStr | None = None, strict: int | None = None, vspipe: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tmpl(self, clips: VideoNode | _SequenceLike[VideoNode], prop: _AnyStr | _SequenceLike[_AnyStr], text: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> _ReturnDict_akarin_Version: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cambi(self, window_size: int | None = None, topk: float | None = None, tvi_threshold: float | None = None, scores: int | None = None, scaling: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLISR(self, scale: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DLVFX(self, op: int, scale: float | None = None, strength: float | None = None, output_depth: int | None = None, num_streams: int | None = None, model_dir: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, expr: _AnyStr | _SequenceLike[_AnyStr], format: int | None = None, opt: int | None = None, boundary: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropExpr(self, dict: Func | _VSCallback_akarin_PropExpr_dict) -> VideoNode: ...
            @_Wrapper.Function
            def Select(self, prop_src: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, text: _AnyStr, alignment: int | None = None, scale: int | None = None, prop: _AnyStr | None = None, strict: int | None = None, vspipe: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tmpl(self, prop: _AnyStr | _SequenceLike[_AnyStr], text: _AnyStr | _SequenceLike[_AnyStr]) -> VideoNode: ...

# </implementation/akarin>

# <implementation/awarp>
class _awarp:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AWarp(self, clip: VideoNode, mask: VideoNode, depth_h: int | _SequenceLike[int] | None = None, depth_v: int | _SequenceLike[int] | None = None, mask_first_plane: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AWarp(self, mask: VideoNode, depth_h: int | _SequenceLike[int] | None = None, depth_v: int | _SequenceLike[int] | None = None, mask_first_plane: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

# </implementation/awarp>

# <implementation/bilateralgpu>
class _bilateralgpu:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, clip: VideoNode, sigma_spatial: float | _SequenceLike[float] | None = None, sigma_color: float | _SequenceLike[float] | None = None, radius: int | _SequenceLike[int] | None = None, device_id: int | None = None, num_streams: int | None = None, use_shared_memory: int | None = None, ref: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, sigma_spatial: float | _SequenceLike[float] | None = None, sigma_color: float | _SequenceLike[float] | None = None, radius: int | _SequenceLike[int] | None = None, device_id: int | None = None, num_streams: int | None = None, use_shared_memory: int | None = None, ref: VideoNode | None = None) -> VideoNode: ...

# </implementation/bilateralgpu>

# <implementation/bilateralgpu_rtc>
class _bilateralgpu_rtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, clip: VideoNode, sigma_spatial: float | _SequenceLike[float] | None = None, sigma_color: float | _SequenceLike[float] | None = None, radius: int | _SequenceLike[int] | None = None, device_id: int | None = None, num_streams: int | None = None, use_shared_memory: int | None = None, block_x: int | None = None, block_y: int | None = None, ref: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bilateral(self, sigma_spatial: float | _SequenceLike[float] | None = None, sigma_color: float | _SequenceLike[float] | None = None, radius: int | _SequenceLike[int] | None = None, device_id: int | None = None, num_streams: int | None = None, use_shared_memory: int | None = None, block_x: int | None = None, block_y: int | None = None, ref: VideoNode | None = None) -> VideoNode: ...

# </implementation/bilateralgpu_rtc>

# <implementation/bm3d>
class _bm3d:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Basic(self, input: VideoNode, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, th_mse: float | None = None, hard_thr: float | None = None, matrix: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Final(self, input: VideoNode, ref: VideoNode, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, th_mse: float | None = None, matrix: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def OPP2RGB(self, input: VideoNode, sample: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RGB2OPP(self, input: VideoNode, sample: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, input: VideoNode, radius: int | None = None, sample: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VBasic(self, input: VideoNode, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, radius: int | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, ps_num: int | None = None, ps_range: int | None = None, ps_step: int | None = None, th_mse: float | None = None, hard_thr: float | None = None, matrix: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFinal(self, input: VideoNode, ref: VideoNode, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, radius: int | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, ps_num: int | None = None, ps_range: int | None = None, ps_step: int | None = None, th_mse: float | None = None, matrix: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Basic(self, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, th_mse: float | None = None, hard_thr: float | None = None, matrix: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Final(self, ref: VideoNode, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, th_mse: float | None = None, matrix: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def OPP2RGB(self, sample: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RGB2OPP(self, sample: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, radius: int | None = None, sample: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VBasic(self, ref: VideoNode | None = None, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, radius: int | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, ps_num: int | None = None, ps_range: int | None = None, ps_step: int | None = None, th_mse: float | None = None, hard_thr: float | None = None, matrix: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFinal(self, ref: VideoNode, profile: _AnyStr | None = None, sigma: float | _SequenceLike[float] | None = None, radius: int | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, bm_step: int | None = None, ps_num: int | None = None, ps_range: int | None = None, ps_step: int | None = None, th_mse: float | None = None, matrix: int | None = None) -> VideoNode: ...

# </implementation/bm3d>

# <implementation/bm3dcpu>
class _bm3dcpu:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, clip: VideoNode, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, chroma: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, clip: VideoNode, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, chroma: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: int | _SequenceLike[int]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, chroma: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, chroma: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: int | _SequenceLike[int]) -> VideoNode: ...

# </implementation/bm3dcpu>

# <implementation/bm3dcuda>
class _bm3dcuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, clip: VideoNode, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, clip: VideoNode, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: int | _SequenceLike[int]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: int | _SequenceLike[int]) -> VideoNode: ...

# </implementation/bm3dcuda>

# <implementation/bm3dcuda_rtc>
class _bm3dcuda_rtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, clip: VideoNode, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, clip: VideoNode, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: int | _SequenceLike[int]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BM3D(self, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BM3Dv2(self, ref: VideoNode | None = None, sigma: float | _SequenceLike[float] | None = None, block_step: int | _SequenceLike[int] | None = None, bm_range: int | _SequenceLike[int] | None = None, radius: int | None = None, ps_num: int | _SequenceLike[int] | None = None, ps_range: int | _SequenceLike[int] | None = None, chroma: int | None = None, device_id: int | None = None, fast: int | None = None, extractor_exp: int | None = None, bm_error_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_2d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, transform_1d_s: _AnyStr | _SequenceLike[_AnyStr] | None = None, zero_init: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: int | _SequenceLike[int]) -> VideoNode: ...

# </implementation/bm3dcuda_rtc>

# <implementation/bs>
_ReturnDict_bs_TrackInfo = TypedDict("_ReturnDict_bs_TrackInfo", {"mediatype": int, "mediatypestr": _AnyStr, "codec": int, "codecstr": _AnyStr, "disposition": int, "dispositionstr": _AnyStr})

class _bs:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AudioSource(self, source: _AnyStr, track: int | None = None, adjustdelay: int | None = None, threads: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None, drc_scale: float | None = None, cachemode: int | None = None, cachepath: _AnyStr | None = None, cachesize: int | None = None, showprogress: int | None = None, maxdecoders: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def Metadata(self, source: _AnyStr, track: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetDebugOutput(self, enable: int) -> None: ...
            @_Wrapper.Function
            def SetFFmpegLogLevel(self, level: int) -> int: ...
            @_Wrapper.Function
            def TrackInfo(self, source: _AnyStr, enable_drefs: int | None = None, use_absolute_path: int | None = None) -> _ReturnDict_bs_TrackInfo: ...
            @_Wrapper.Function
            def VideoSource(self, source: _AnyStr, track: int | None = None, variableformat: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, rff: int | None = None, threads: int | None = None, seekpreroll: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None, cachemode: int | None = None, cachepath: _AnyStr | None = None, cachesize: int | None = None, hwdevice: _AnyStr | None = None, extrahwframes: int | None = None, timecodes: _AnyStr | None = None, start_number: int | None = None, viewid: int | None = None, showprogress: int | None = None, maxdecoders: int | None = None, hwfallback: int | None = None) -> VideoNode: ...

# </implementation/bs>

# <implementation/bwdif>
class _bwdif:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bwdif(self, clip: VideoNode, field: int, edeint: VideoNode | None = None, opt: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bwdif(self, field: int, edeint: VideoNode | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/bwdif>

# <implementation/cs>
class _cs:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ConvertColor(self, clip: VideoNode, output_profile: _AnyStr, input_profile: _AnyStr | None = None, float_output: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ImageSource(self, source: _AnyStr, subsampling_pad: int | None = None, jpeg_rgb: int | None = None, jpeg_fancy_upsampling: int | None = None, jpeg_cmyk_profile: _AnyStr | None = None, jpeg_cmyk_target_profile: _AnyStr | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ConvertColor(self, output_profile: _AnyStr, input_profile: _AnyStr | None = None, float_output: int | None = None) -> VideoNode: ...

# </implementation/cs>

# <implementation/dctf>
class _dctf:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DCTFilter(self, clip: VideoNode, factors: float | _SequenceLike[float], planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DCTFilter(self, factors: float | _SequenceLike[float], planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

# </implementation/dctf>

# <implementation/deblock>
class _deblock:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deblock(self, clip: VideoNode, quant: int | None = None, aoffset: int | None = None, boffset: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deblock(self, quant: int | None = None, aoffset: int | None = None, boffset: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

# </implementation/deblock>

# <implementation/descale>
class _descale:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, src: VideoNode, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debicubic(self, src: VideoNode, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debilinear(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Decustom(self, src: VideoNode, width: int, height: int, custom_kernel: Func | _VSCallback_descale_Decustom_custom_kernel, taps: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Delanczos(self, src: VideoNode, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Depoint(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline16(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline36(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline64(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, src: VideoNode, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleCustom(self, src: VideoNode, width: int, height: int, custom_kernel: Func | _VSCallback_descale_ScaleCustom_custom_kernel, taps: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debicubic(self, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Debilinear(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Decustom(self, width: int, height: int, custom_kernel: Func | _VSCallback_descale_Decustom_custom_kernel, taps: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Delanczos(self, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Depoint(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline16(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline36(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Despline64(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleCustom(self, width: int, height: int, custom_kernel: Func | _VSCallback_descale_ScaleCustom_custom_kernel, taps: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, width: int, height: int, blur: float | None = None, post_conv: float | _SequenceLike[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/descale>

# <implementation/dfttest>
class _dfttest:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, ftype: int | None = None, sigma: float | None = None, sigma2: float | None = None, pmin: float | None = None, pmax: float | None = None, sbsize: int | None = None, smode: int | None = None, sosize: int | None = None, tbsize: int | None = None, tmode: int | None = None, tosize: int | None = None, swin: int | None = None, twin: int | None = None, sbeta: float | None = None, tbeta: float | None = None, zmean: int | None = None, f0beta: float | None = None, nlocation: int | _SequenceLike[int] | None = None, alpha: float | None = None, slocation: float | _SequenceLike[float] | None = None, ssx: float | _SequenceLike[float] | None = None, ssy: float | _SequenceLike[float] | None = None, sst: float | _SequenceLike[float] | None = None, ssystem: int | None = None, planes: int | _SequenceLike[int] | None = None, opt: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, ftype: int | None = None, sigma: float | None = None, sigma2: float | None = None, pmin: float | None = None, pmax: float | None = None, sbsize: int | None = None, smode: int | None = None, sosize: int | None = None, tbsize: int | None = None, tmode: int | None = None, tosize: int | None = None, swin: int | None = None, twin: int | None = None, sbeta: float | None = None, tbeta: float | None = None, zmean: int | None = None, f0beta: float | None = None, nlocation: int | _SequenceLike[int] | None = None, alpha: float | None = None, slocation: float | _SequenceLike[float] | None = None, ssx: float | _SequenceLike[float] | None = None, ssy: float | _SequenceLike[float] | None = None, sst: float | _SequenceLike[float] | None = None, ssystem: int | None = None, planes: int | _SequenceLike[int] | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/dfttest>

# <implementation/dfttest2_cpu>
class _dfttest2_cpu:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, window: float | _SequenceLike[float], sigma: float | _SequenceLike[float], sigma2: float, pmin: float, pmax: float, filter_type: int, radius: int | None = None, block_size: int | None = None, block_step: int | None = None, zero_mean: int | None = None, window_freq: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RDFT(self, data: float | _SequenceLike[float], shape: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, window: float | _SequenceLike[float], sigma: float | _SequenceLike[float], sigma2: float, pmin: float, pmax: float, filter_type: int, radius: int | None = None, block_size: int | None = None, block_step: int | None = None, zero_mean: int | None = None, window_freq: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/dfttest2_cpu>

# <implementation/dfttest2_cuda>
class _dfttest2_cuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: int | None = None, block_size: int | None = None, block_step: int | None = None, planes: int | _SequenceLike[int] | None = None, in_place: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RDFT(self, data: float | _SequenceLike[float], shape: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def ToSingle(self, data: float | _SequenceLike[float]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: int | None = None, block_size: int | None = None, block_step: int | None = None, planes: int | _SequenceLike[int] | None = None, in_place: int | None = None, device_id: int | None = None) -> VideoNode: ...

# </implementation/dfttest2_cuda>

# <implementation/dfttest2_nvrtc>
class _dfttest2_nvrtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, clip: VideoNode, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: int | None = None, block_size: int | None = None, block_step: int | None = None, planes: int | _SequenceLike[int] | None = None, in_place: int | None = None, device_id: int | None = None, num_streams: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RDFT(self, data: float | _SequenceLike[float], shape: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def ToSingle(self, data: float | _SequenceLike[float]) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DFTTest(self, kernel: _AnyStr | _SequenceLike[_AnyStr], radius: int | None = None, block_size: int | None = None, block_step: int | None = None, planes: int | _SequenceLike[int] | None = None, in_place: int | None = None, device_id: int | None = None, num_streams: int | None = None) -> VideoNode: ...

# </implementation/dfttest2_nvrtc>

# <implementation/dvdsrc2>
class _dvdsrc2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def FullVts(self, path: _AnyStr, vts: int, ranges: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FullVtsAc3(self, path: _AnyStr, vts: int, audio: int, ranges: int | _SequenceLike[int] | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def FullVtsLpcm(self, path: _AnyStr, vts: int, audio: int, ranges: int | _SequenceLike[int] | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def Ifo(self, path: _AnyStr, ifo: int) -> _AnyStr: ...
            @_Wrapper.Function
            def RawAc3(self, path: _AnyStr, vts: int, audio: int, ranges: int | _SequenceLike[int] | None = None) -> AudioNode: ...

# </implementation/dvdsrc2>

# <implementation/edgemasks>
class _edgemasks:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cross(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExKirsch(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExPrewitt(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExSobel(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FDoG(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kirsch(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kroon(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RScharr(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson3(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson5(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Scharr(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tritical(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Cross(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExKirsch(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExPrewitt(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ExSobel(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FDoG(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kirsch(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Kroon(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RScharr(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson3(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Robinson5(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Scharr(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tritical(self, planes: int | _SequenceLike[int] | None = None, scale: float | _SequenceLike[float] | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/edgemasks>

# <implementation/eedi2>
class _eedi2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI2(self, clip: VideoNode, field: int, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI2(self, field: int, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None) -> VideoNode: ...

# </implementation/eedi2>

# <implementation/eedi2cuda>
class _eedi2cuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AA2(self, clip: VideoNode, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None, planes: int | _SequenceLike[int] | None = None, num_streams: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BuildConfig(self) -> _AnyStr: ...
            @_Wrapper.Function
            def EEDI2(self, clip: VideoNode, field: int, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None, planes: int | _SequenceLike[int] | None = None, num_streams: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Enlarge2(self, clip: VideoNode, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None, planes: int | _SequenceLike[int] | None = None, num_streams: int | None = None, device_id: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AA2(self, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None, planes: int | _SequenceLike[int] | None = None, num_streams: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def EEDI2(self, field: int, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None, planes: int | _SequenceLike[int] | None = None, num_streams: int | None = None, device_id: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Enlarge2(self, mthresh: int | None = None, lthresh: int | None = None, vthresh: int | None = None, estr: int | None = None, dstr: int | None = None, maxd: int | None = None, map: int | None = None, nt: int | None = None, pp: int | None = None, planes: int | _SequenceLike[int] | None = None, num_streams: int | None = None, device_id: int | None = None) -> VideoNode: ...

# </implementation/eedi2cuda>

# <implementation/eedi3m>
class _eedi3m:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI3(self, clip: VideoNode, field: int, dh: int | None = None, planes: int | _SequenceLike[int] | None = None, alpha: float | None = None, beta: float | None = None, gamma: float | None = None, nrad: int | None = None, mdis: int | None = None, hp: int | None = None, ucubic: int | None = None, cost3: int | None = None, vcheck: int | None = None, vthresh0: float | None = None, vthresh1: float | None = None, vthresh2: float | None = None, sclip: VideoNode | None = None, mclip: VideoNode | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def EEDI3CL(self, clip: VideoNode, field: int, dh: int | None = None, planes: int | _SequenceLike[int] | None = None, alpha: float | None = None, beta: float | None = None, gamma: float | None = None, nrad: int | None = None, mdis: int | None = None, hp: int | None = None, ucubic: int | None = None, cost3: int | None = None, vcheck: int | None = None, vthresh0: float | None = None, vthresh1: float | None = None, vthresh2: float | None = None, sclip: VideoNode | None = None, opt: int | None = None, device: int | None = None, list_device: int | None = None, info: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def EEDI3(self, field: int, dh: int | None = None, planes: int | _SequenceLike[int] | None = None, alpha: float | None = None, beta: float | None = None, gamma: float | None = None, nrad: int | None = None, mdis: int | None = None, hp: int | None = None, ucubic: int | None = None, cost3: int | None = None, vcheck: int | None = None, vthresh0: float | None = None, vthresh1: float | None = None, vthresh2: float | None = None, sclip: VideoNode | None = None, mclip: VideoNode | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def EEDI3CL(self, field: int, dh: int | None = None, planes: int | _SequenceLike[int] | None = None, alpha: float | None = None, beta: float | None = None, gamma: float | None = None, nrad: int | None = None, mdis: int | None = None, hp: int | None = None, ucubic: int | None = None, cost3: int | None = None, vcheck: int | None = None, vthresh0: float | None = None, vthresh1: float | None = None, vthresh2: float | None = None, sclip: VideoNode | None = None, opt: int | None = None, device: int | None = None, list_device: int | None = None, info: int | None = None) -> VideoNode: ...

# </implementation/eedi3m>

# <implementation/ffms2>
class _ffms2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def GetLogLevel(self) -> int: ...
            @_Wrapper.Function
            def Index(self, source: _AnyStr, cachefile: _AnyStr | None = None, indextracks: int | _SequenceLike[int] | None = None, errorhandling: int | None = None, overwrite: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None) -> _AnyStr: ...
            @_Wrapper.Function
            def SetLogLevel(self, level: int) -> int: ...
            @_Wrapper.Function
            def Source(self, source: _AnyStr, track: int | None = None, cache: int | None = None, cachefile: _AnyStr | None = None, fpsnum: int | None = None, fpsden: int | None = None, threads: int | None = None, timecodes: _AnyStr | None = None, seekmode: int | None = None, width: int | None = None, height: int | None = None, resizer: _AnyStr | None = None, format: int | None = None, alpha: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> _AnyStr: ...

# </implementation/ffms2>

# <implementation/fft3dfilter>
class _fft3dfilter:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def FFT3DFilter(self, clip: VideoNode, sigma: float | None = None, beta: float | None = None, planes: int | _SequenceLike[int] | None = None, bw: int | None = None, bh: int | None = None, bt: int | None = None, ow: int | None = None, oh: int | None = None, kratio: float | None = None, sharpen: float | None = None, scutoff: float | None = None, svr: float | None = None, smin: float | None = None, smax: float | None = None, measure: int | None = None, interlaced: int | None = None, wintype: int | None = None, pframe: int | None = None, px: int | None = None, py: int | None = None, pshow: int | None = None, pcutoff: float | None = None, pfactor: float | None = None, sigma2: float | None = None, sigma3: float | None = None, sigma4: float | None = None, degrid: float | None = None, dehalo: float | None = None, hr: float | None = None, ht: float | None = None, ncpu: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def FFT3DFilter(self, sigma: float | None = None, beta: float | None = None, planes: int | _SequenceLike[int] | None = None, bw: int | None = None, bh: int | None = None, bt: int | None = None, ow: int | None = None, oh: int | None = None, kratio: float | None = None, sharpen: float | None = None, scutoff: float | None = None, svr: float | None = None, smin: float | None = None, smax: float | None = None, measure: int | None = None, interlaced: int | None = None, wintype: int | None = None, pframe: int | None = None, px: int | None = None, py: int | None = None, pshow: int | None = None, pcutoff: float | None = None, pfactor: float | None = None, sigma2: float | None = None, sigma3: float | None = None, sigma4: float | None = None, degrid: float | None = None, dehalo: float | None = None, hr: float | None = None, ht: float | None = None, ncpu: int | None = None) -> VideoNode: ...

# </implementation/fft3dfilter>

# <implementation/fmtc>
class _fmtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def bitdepth(self, clip: VideoNode, csp: int | None = None, bits: int | None = None, flt: int | None = None, planes: int | _SequenceLike[int] | None = None, fulls: int | None = None, fulld: int | None = None, dmode: int | None = None, ampo: float | None = None, ampn: float | None = None, dyn: int | None = None, staticnoise: int | None = None, cpuopt: int | None = None, patsize: int | None = None, tpdfo: int | None = None, tpdfn: int | None = None, corplane: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def histluma(self, clip: VideoNode, full: int | None = None, amp: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix(self, clip: VideoNode, mat: _AnyStr | None = None, mats: _AnyStr | None = None, matd: _AnyStr | None = None, fulls: int | None = None, fulld: int | None = None, coef: float | _SequenceLike[float] | None = None, csp: int | None = None, col_fam: int | None = None, bits: int | None = None, singleout: int | None = None, cpuopt: int | None = None, planes: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix2020cl(self, clip: VideoNode, full: int | None = None, csp: int | None = None, bits: int | None = None, cpuopt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def nativetostack16(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def primaries(self, clip: VideoNode, rs: float | _SequenceLike[float] | None = None, gs: float | _SequenceLike[float] | None = None, bs: float | _SequenceLike[float] | None = None, ws: float | _SequenceLike[float] | None = None, rd: float | _SequenceLike[float] | None = None, gd: float | _SequenceLike[float] | None = None, bd: float | _SequenceLike[float] | None = None, wd: float | _SequenceLike[float] | None = None, prims: _AnyStr | None = None, primd: _AnyStr | None = None, wconv: int | None = None, cpuopt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def resample(self, clip: VideoNode, w: int | None = None, h: int | None = None, sx: float | _SequenceLike[float] | None = None, sy: float | _SequenceLike[float] | None = None, sw: float | _SequenceLike[float] | None = None, sh: float | _SequenceLike[float] | None = None, scale: float | None = None, scaleh: float | None = None, scalev: float | None = None, kernel: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelh: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelv: _AnyStr | _SequenceLike[_AnyStr] | None = None, impulse: float | _SequenceLike[float] | None = None, impulseh: float | _SequenceLike[float] | None = None, impulsev: float | _SequenceLike[float] | None = None, taps: int | _SequenceLike[int] | None = None, tapsh: int | _SequenceLike[int] | None = None, tapsv: int | _SequenceLike[int] | None = None, a1: float | _SequenceLike[float] | None = None, a2: float | _SequenceLike[float] | None = None, a3: float | _SequenceLike[float] | None = None, a1h: float | _SequenceLike[float] | None = None, a2h: float | _SequenceLike[float] | None = None, a3h: float | _SequenceLike[float] | None = None, a1v: float | _SequenceLike[float] | None = None, a2v: float | _SequenceLike[float] | None = None, a3v: float | _SequenceLike[float] | None = None, kovrspl: int | _SequenceLike[int] | None = None, fh: float | _SequenceLike[float] | None = None, fv: float | _SequenceLike[float] | None = None, cnorm: int | _SequenceLike[int] | None = None, total: float | _SequenceLike[float] | None = None, totalh: float | _SequenceLike[float] | None = None, totalv: float | _SequenceLike[float] | None = None, invks: int | _SequenceLike[int] | None = None, invksh: int | _SequenceLike[int] | None = None, invksv: int | _SequenceLike[int] | None = None, invkstaps: int | _SequenceLike[int] | None = None, invkstapsh: int | _SequenceLike[int] | None = None, invkstapsv: int | _SequenceLike[int] | None = None, csp: int | None = None, css: _AnyStr | None = None, planes: float | _SequenceLike[float] | None = None, fulls: int | None = None, fulld: int | None = None, center: int | _SequenceLike[int] | None = None, cplace: _AnyStr | None = None, cplaces: _AnyStr | None = None, cplaced: _AnyStr | None = None, interlaced: int | None = None, interlacedd: int | None = None, tff: int | None = None, tffd: int | None = None, flt: int | None = None, cpuopt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def stack16tonative(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def transfer(self, clip: VideoNode, transs: _AnyStr | _SequenceLike[_AnyStr] | None = None, transd: _AnyStr | _SequenceLike[_AnyStr] | None = None, cont: float | None = None, gcor: float | None = None, bits: int | None = None, flt: int | None = None, fulls: int | None = None, fulld: int | None = None, logceis: int | None = None, logceid: int | None = None, cpuopt: int | None = None, blacklvl: float | None = None, sceneref: int | None = None, lb: float | None = None, lw: float | None = None, lws: float | None = None, lwd: float | None = None, ambient: float | None = None, match: int | None = None, gy: int | None = None, debug: int | None = None, sig_c: float | None = None, sig_t: float | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def bitdepth(self, csp: int | None = None, bits: int | None = None, flt: int | None = None, planes: int | _SequenceLike[int] | None = None, fulls: int | None = None, fulld: int | None = None, dmode: int | None = None, ampo: float | None = None, ampn: float | None = None, dyn: int | None = None, staticnoise: int | None = None, cpuopt: int | None = None, patsize: int | None = None, tpdfo: int | None = None, tpdfn: int | None = None, corplane: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def histluma(self, full: int | None = None, amp: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix(self, mat: _AnyStr | None = None, mats: _AnyStr | None = None, matd: _AnyStr | None = None, fulls: int | None = None, fulld: int | None = None, coef: float | _SequenceLike[float] | None = None, csp: int | None = None, col_fam: int | None = None, bits: int | None = None, singleout: int | None = None, cpuopt: int | None = None, planes: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def matrix2020cl(self, full: int | None = None, csp: int | None = None, bits: int | None = None, cpuopt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def nativetostack16(self) -> VideoNode: ...
            @_Wrapper.Function
            def primaries(self, rs: float | _SequenceLike[float] | None = None, gs: float | _SequenceLike[float] | None = None, bs: float | _SequenceLike[float] | None = None, ws: float | _SequenceLike[float] | None = None, rd: float | _SequenceLike[float] | None = None, gd: float | _SequenceLike[float] | None = None, bd: float | _SequenceLike[float] | None = None, wd: float | _SequenceLike[float] | None = None, prims: _AnyStr | None = None, primd: _AnyStr | None = None, wconv: int | None = None, cpuopt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def resample(self, w: int | None = None, h: int | None = None, sx: float | _SequenceLike[float] | None = None, sy: float | _SequenceLike[float] | None = None, sw: float | _SequenceLike[float] | None = None, sh: float | _SequenceLike[float] | None = None, scale: float | None = None, scaleh: float | None = None, scalev: float | None = None, kernel: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelh: _AnyStr | _SequenceLike[_AnyStr] | None = None, kernelv: _AnyStr | _SequenceLike[_AnyStr] | None = None, impulse: float | _SequenceLike[float] | None = None, impulseh: float | _SequenceLike[float] | None = None, impulsev: float | _SequenceLike[float] | None = None, taps: int | _SequenceLike[int] | None = None, tapsh: int | _SequenceLike[int] | None = None, tapsv: int | _SequenceLike[int] | None = None, a1: float | _SequenceLike[float] | None = None, a2: float | _SequenceLike[float] | None = None, a3: float | _SequenceLike[float] | None = None, a1h: float | _SequenceLike[float] | None = None, a2h: float | _SequenceLike[float] | None = None, a3h: float | _SequenceLike[float] | None = None, a1v: float | _SequenceLike[float] | None = None, a2v: float | _SequenceLike[float] | None = None, a3v: float | _SequenceLike[float] | None = None, kovrspl: int | _SequenceLike[int] | None = None, fh: float | _SequenceLike[float] | None = None, fv: float | _SequenceLike[float] | None = None, cnorm: int | _SequenceLike[int] | None = None, total: float | _SequenceLike[float] | None = None, totalh: float | _SequenceLike[float] | None = None, totalv: float | _SequenceLike[float] | None = None, invks: int | _SequenceLike[int] | None = None, invksh: int | _SequenceLike[int] | None = None, invksv: int | _SequenceLike[int] | None = None, invkstaps: int | _SequenceLike[int] | None = None, invkstapsh: int | _SequenceLike[int] | None = None, invkstapsv: int | _SequenceLike[int] | None = None, csp: int | None = None, css: _AnyStr | None = None, planes: float | _SequenceLike[float] | None = None, fulls: int | None = None, fulld: int | None = None, center: int | _SequenceLike[int] | None = None, cplace: _AnyStr | None = None, cplaces: _AnyStr | None = None, cplaced: _AnyStr | None = None, interlaced: int | None = None, interlacedd: int | None = None, tff: int | None = None, tffd: int | None = None, flt: int | None = None, cpuopt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def stack16tonative(self) -> VideoNode: ...
            @_Wrapper.Function
            def transfer(self, transs: _AnyStr | _SequenceLike[_AnyStr] | None = None, transd: _AnyStr | _SequenceLike[_AnyStr] | None = None, cont: float | None = None, gcor: float | None = None, bits: int | None = None, flt: int | None = None, fulls: int | None = None, fulld: int | None = None, logceis: int | None = None, logceid: int | None = None, cpuopt: int | None = None, blacklvl: float | None = None, sceneref: int | None = None, lb: float | None = None, lw: float | None = None, lws: float | None = None, lwd: float | None = None, ambient: float | None = None, match: int | None = None, gy: int | None = None, debug: int | None = None, sig_c: float | None = None, sig_t: float | None = None) -> VideoNode: ...

# </implementation/fmtc>

# <implementation/hysteresis>
class _hysteresis:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Hysteresis(self, clipa: VideoNode, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Hysteresis(self, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

# </implementation/hysteresis>

# <implementation/imwri>
class _imwri:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Read(self, filename: _AnyStr | _SequenceLike[_AnyStr], firstnum: int | None = None, mismatch: int | None = None, alpha: int | None = None, float_output: int | None = None, embed_icc: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Write(self, clip: VideoNode, imgformat: _AnyStr, filename: _AnyStr, firstnum: int | None = None, quality: int | None = None, dither: int | None = None, compression_type: _AnyStr | None = None, overwrite: int | None = None, alpha: VideoNode | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Write(self, imgformat: _AnyStr, filename: _AnyStr, firstnum: int | None = None, quality: int | None = None, dither: int | None = None, compression_type: _AnyStr | None = None, overwrite: int | None = None, alpha: VideoNode | None = None) -> VideoNode: ...

# </implementation/imwri>

# <implementation/knlm>
class _knlm:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def KNLMeansCL(self, clip: VideoNode, d: int | None = None, a: int | None = None, s: int | None = None, h: float | None = None, channels: _AnyStr | None = None, wmode: int | None = None, wref: float | None = None, rclip: VideoNode | None = None, device_type: _AnyStr | None = None, device_id: int | None = None, ocl_x: int | None = None, ocl_y: int | None = None, ocl_r: int | None = None, info: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def KNLMeansCL(self, d: int | None = None, a: int | None = None, s: int | None = None, h: float | None = None, channels: _AnyStr | None = None, wmode: int | None = None, wref: float | None = None, rclip: VideoNode | None = None, device_type: _AnyStr | None = None, device_id: int | None = None, ocl_x: int | None = None, ocl_y: int | None = None, ocl_r: int | None = None, info: int | None = None) -> VideoNode: ...

# </implementation/knlm>

# <implementation/lsmas>
class _lsmas:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def LWLibavSource(self, source: _AnyStr, stream_index: int | None = None, cache: int | None = None, cachefile: _AnyStr | None = None, threads: int | None = None, seek_mode: int | None = None, seek_threshold: int | None = None, dr: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, variable: int | None = None, format: _AnyStr | None = None, decoder: _AnyStr | None = None, prefer_hw: int | None = None, repeat: int | None = None, dominance: int | None = None, ff_loglevel: int | None = None, cachedir: _AnyStr | None = None, ff_options: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LibavSMASHSource(self, source: _AnyStr, track: int | None = None, threads: int | None = None, seek_mode: int | None = None, seek_threshold: int | None = None, dr: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, variable: int | None = None, format: _AnyStr | None = None, decoder: _AnyStr | None = None, prefer_hw: int | None = None, ff_loglevel: int | None = None, ff_options: _AnyStr | None = None) -> VideoNode: ...

# </implementation/lsmas>

# <implementation/manipmv>
class _manipmv:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ExpandAnalysisData(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleVect(self, clip: VideoNode, scaleX: int | None = None, scaleY: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ShowVect(self, clip: VideoNode, vectors: VideoNode, useSceneChangeProps: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ExpandAnalysisData(self) -> VideoNode: ...
            @_Wrapper.Function
            def ScaleVect(self, scaleX: int | None = None, scaleY: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ShowVect(self, vectors: VideoNode, useSceneChangeProps: int | None = None) -> VideoNode: ...

# </implementation/manipmv>

# <implementation/mv>
class _mv:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Analyse(self, super: VideoNode, blksize: int | None = None, blksizev: int | None = None, levels: int | None = None, search: int | None = None, searchparam: int | None = None, pelsearch: int | None = None, isb: int | None = None, lambda_: int | None = None, chroma: int | None = None, delta: int | None = None, truemotion: int | None = None, lsad: int | None = None, plevel: int | None = None, global_: int | None = None, pnew: int | None = None, pzero: int | None = None, pglobal: int | None = None, overlap: int | None = None, overlapv: int | None = None, divide: int | None = None, badsad: int | None = None, badrange: int | None = None, opt: int | None = None, meander: int | None = None, trymany: int | None = None, fields: int | None = None, tff: int | None = None, search_coarse: int | None = None, dct: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlockFPS(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: int | None = None, den: int | None = None, mode: int | None = None, ml: float | None = None, blend: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Compensate(self, clip: VideoNode, super: VideoNode, vectors: VideoNode, scbehavior: int | None = None, thsad: int | None = None, fields: int | None = None, time: float | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain1(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain2(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain3(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain4(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain5(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain6(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, mvbw6: VideoNode, mvfw6: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanAnalyse(self, clip: VideoNode, vectors: VideoNode, mask: VideoNode | None = None, zoom: int | None = None, rot: int | None = None, pixaspect: float | None = None, error: float | None = None, info: int | None = None, wrong: float | None = None, zerow: float | None = None, thscd1: int | None = None, thscd2: int | None = None, fields: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanCompensate(self, clip: VideoNode, data: VideoNode, offset: float | None = None, subpixel: int | None = None, pixaspect: float | None = None, matchfields: int | None = None, mirror: int | None = None, blur: int | None = None, info: int | None = None, fields: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanEstimate(self, clip: VideoNode, trust: float | None = None, winx: int | None = None, winy: int | None = None, wleft: int | None = None, wtop: int | None = None, dxmax: int | None = None, dymax: int | None = None, zoommax: float | None = None, stab: float | None = None, pixaspect: float | None = None, info: int | None = None, show: int | None = None, fields: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanStabilise(self, clip: VideoNode, data: VideoNode, cutoff: float | None = None, damping: float | None = None, initzoom: float | None = None, addzoom: int | None = None, prev: int | None = None, next: int | None = None, mirror: int | None = None, blur: int | None = None, dxmax: float | None = None, dymax: float | None = None, zoommax: float | None = None, rotmax: float | None = None, subpixel: int | None = None, pixaspect: float | None = None, fitlast: int | None = None, tzoom: float | None = None, info: int | None = None, method: int | None = None, fields: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Finest(self, super: VideoNode, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Flow(self, clip: VideoNode, super: VideoNode, vectors: VideoNode, time: float | None = None, mode: int | None = None, fields: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowBlur(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, blur: float | None = None, prec: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowFPS(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: int | None = None, den: int | None = None, mask: int | None = None, ml: float | None = None, blend: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowInter(self, clip: VideoNode, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, time: float | None = None, ml: float | None = None, blend: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Mask(self, clip: VideoNode, vectors: VideoNode, ml: float | None = None, gamma: float | None = None, kind: int | None = None, time: float | None = None, ysc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Recalculate(self, super: VideoNode, vectors: VideoNode, thsad: int | None = None, smooth: int | None = None, blksize: int | None = None, blksizev: int | None = None, search: int | None = None, searchparam: int | None = None, lambda_: int | None = None, chroma: int | None = None, truemotion: int | None = None, pnew: int | None = None, overlap: int | None = None, overlapv: int | None = None, divide: int | None = None, opt: int | None = None, meander: int | None = None, fields: int | None = None, tff: int | None = None, dct: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SCDetection(self, clip: VideoNode, vectors: VideoNode, thscd1: int | None = None, thscd2: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Super(self, clip: VideoNode, hpad: int | None = None, vpad: int | None = None, pel: int | None = None, levels: int | None = None, chroma: int | None = None, sharp: int | None = None, rfilter: int | None = None, pelclip: VideoNode | None = None, opt: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Analyse(self, blksize: int | None = None, blksizev: int | None = None, levels: int | None = None, search: int | None = None, searchparam: int | None = None, pelsearch: int | None = None, isb: int | None = None, lambda_: int | None = None, chroma: int | None = None, delta: int | None = None, truemotion: int | None = None, lsad: int | None = None, plevel: int | None = None, global_: int | None = None, pnew: int | None = None, pzero: int | None = None, pglobal: int | None = None, overlap: int | None = None, overlapv: int | None = None, divide: int | None = None, badsad: int | None = None, badrange: int | None = None, opt: int | None = None, meander: int | None = None, trymany: int | None = None, fields: int | None = None, tff: int | None = None, search_coarse: int | None = None, dct: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlockFPS(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: int | None = None, den: int | None = None, mode: int | None = None, ml: float | None = None, blend: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Compensate(self, super: VideoNode, vectors: VideoNode, scbehavior: int | None = None, thsad: int | None = None, fields: int | None = None, time: float | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain1(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain2(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain3(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain4(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain5(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Degrain6(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, mvbw2: VideoNode, mvfw2: VideoNode, mvbw3: VideoNode, mvfw3: VideoNode, mvbw4: VideoNode, mvfw4: VideoNode, mvbw5: VideoNode, mvfw5: VideoNode, mvbw6: VideoNode, mvfw6: VideoNode, thsad: int | None = None, thsadc: int | None = None, plane: int | None = None, limit: int | None = None, limitc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, weights: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanAnalyse(self, vectors: VideoNode, mask: VideoNode | None = None, zoom: int | None = None, rot: int | None = None, pixaspect: float | None = None, error: float | None = None, info: int | None = None, wrong: float | None = None, zerow: float | None = None, thscd1: int | None = None, thscd2: int | None = None, fields: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanCompensate(self, data: VideoNode, offset: float | None = None, subpixel: int | None = None, pixaspect: float | None = None, matchfields: int | None = None, mirror: int | None = None, blur: int | None = None, info: int | None = None, fields: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanEstimate(self, trust: float | None = None, winx: int | None = None, winy: int | None = None, wleft: int | None = None, wtop: int | None = None, dxmax: int | None = None, dymax: int | None = None, zoommax: float | None = None, stab: float | None = None, pixaspect: float | None = None, info: int | None = None, show: int | None = None, fields: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DepanStabilise(self, data: VideoNode, cutoff: float | None = None, damping: float | None = None, initzoom: float | None = None, addzoom: int | None = None, prev: int | None = None, next: int | None = None, mirror: int | None = None, blur: int | None = None, dxmax: float | None = None, dymax: float | None = None, zoommax: float | None = None, rotmax: float | None = None, subpixel: int | None = None, pixaspect: float | None = None, fitlast: int | None = None, tzoom: float | None = None, info: int | None = None, method: int | None = None, fields: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Finest(self, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Flow(self, super: VideoNode, vectors: VideoNode, time: float | None = None, mode: int | None = None, fields: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowBlur(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, blur: float | None = None, prec: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowFPS(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, num: int | None = None, den: int | None = None, mask: int | None = None, ml: float | None = None, blend: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlowInter(self, super: VideoNode, mvbw: VideoNode, mvfw: VideoNode, time: float | None = None, ml: float | None = None, blend: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Mask(self, vectors: VideoNode, ml: float | None = None, gamma: float | None = None, kind: int | None = None, time: float | None = None, ysc: int | None = None, thscd1: int | None = None, thscd2: int | None = None, opt: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Recalculate(self, vectors: VideoNode, thsad: int | None = None, smooth: int | None = None, blksize: int | None = None, blksizev: int | None = None, search: int | None = None, searchparam: int | None = None, lambda_: int | None = None, chroma: int | None = None, truemotion: int | None = None, pnew: int | None = None, overlap: int | None = None, overlapv: int | None = None, divide: int | None = None, opt: int | None = None, meander: int | None = None, fields: int | None = None, tff: int | None = None, dct: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SCDetection(self, vectors: VideoNode, thscd1: int | None = None, thscd2: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Super(self, hpad: int | None = None, vpad: int | None = None, pel: int | None = None, levels: int | None = None, chroma: int | None = None, sharp: int | None = None, rfilter: int | None = None, pelclip: VideoNode | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/mv>

# <implementation/neo_f3kdb>
class _neo_f3kdb:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, clip: VideoNode, range: int | None = None, y: int | None = None, cb: int | None = None, cr: int | None = None, grainy: int | None = None, grainc: int | None = None, sample_mode: int | None = None, seed: int | None = None, blur_first: int | None = None, dynamic_grain: int | None = None, opt: int | None = None, mt: int | None = None, dither_algo: int | None = None, keep_tv_range: int | None = None, output_depth: int | None = None, random_algo_ref: int | None = None, random_algo_grain: int | None = None, random_param_ref: float | None = None, random_param_grain: float | None = None, preset: _AnyStr | None = None, y_1: int | None = None, cb_1: int | None = None, cr_1: int | None = None, y_2: int | None = None, cb_2: int | None = None, cr_2: int | None = None, scale: int | None = None, angle_boost: float | None = None, max_angle: float | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, range: int | None = None, y: int | None = None, cb: int | None = None, cr: int | None = None, grainy: int | None = None, grainc: int | None = None, sample_mode: int | None = None, seed: int | None = None, blur_first: int | None = None, dynamic_grain: int | None = None, opt: int | None = None, mt: int | None = None, dither_algo: int | None = None, keep_tv_range: int | None = None, output_depth: int | None = None, random_algo_ref: int | None = None, random_algo_grain: int | None = None, random_param_ref: float | None = None, random_param_grain: float | None = None, preset: _AnyStr | None = None, y_1: int | None = None, cb_1: int | None = None, cr_1: int | None = None, y_2: int | None = None, cb_2: int | None = None, cr_2: int | None = None, scale: int | None = None, angle_boost: float | None = None, max_angle: float | None = None) -> VideoNode: ...

# </implementation/neo_f3kdb>

# <implementation/nlm_cuda>
class _nlm_cuda:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, clip: VideoNode, d: int | None = None, a: int | None = None, s: int | None = None, h: float | None = None, channels: _AnyStr | None = None, wmode: int | None = None, wref: float | None = None, rclip: VideoNode | None = None, device_id: int | None = None, num_streams: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, d: int | None = None, a: int | None = None, s: int | None = None, h: float | None = None, channels: _AnyStr | None = None, wmode: int | None = None, wref: float | None = None, rclip: VideoNode | None = None, device_id: int | None = None, num_streams: int | None = None) -> VideoNode: ...

# </implementation/nlm_cuda>

# <implementation/nlm_ispc>
class _nlm_ispc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, clip: VideoNode, d: int | None = None, a: int | None = None, s: int | None = None, h: float | None = None, channels: _AnyStr | None = None, wmode: int | None = None, wref: float | None = None, rclip: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NLMeans(self, d: int | None = None, a: int | None = None, s: int | None = None, h: float | None = None, channels: _AnyStr | None = None, wmode: int | None = None, wref: float | None = None, rclip: VideoNode | None = None) -> VideoNode: ...

# </implementation/nlm_ispc>

# <implementation/noise>
class _noise:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Add(self, clip: VideoNode, var: float | None = None, uvar: float | None = None, type: int | None = None, hcorr: float | None = None, vcorr: float | None = None, xsize: float | None = None, ysize: float | None = None, scale: float | None = None, seed: int | None = None, constant: int | None = None, every: int | None = None, opt: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Add(self, var: float | None = None, uvar: float | None = None, type: int | None = None, hcorr: float | None = None, vcorr: float | None = None, xsize: float | None = None, ysize: float | None = None, scale: float | None = None, seed: int | None = None, constant: int | None = None, every: int | None = None, opt: int | None = None) -> VideoNode: ...

# </implementation/noise>

# <implementation/placebo>
class _placebo:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, clip: VideoNode, planes: int | None = None, iterations: int | None = None, threshold: float | None = None, radius: float | None = None, grain: float | None = None, dither: int | None = None, dither_algo: int | None = None, log_level: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Resample(self, clip: VideoNode, width: int, height: int, filter: _AnyStr | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, src_width: float | None = None, src_height: float | None = None, sx: float | None = None, sy: float | None = None, antiring: float | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, linearize: int | None = None, trc: int | None = None, min_luma: float | None = None, log_level: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Shader(self, clip: VideoNode, shader: _AnyStr | None = None, width: int | None = None, height: int | None = None, chroma_loc: int | None = None, matrix: int | None = None, trc: int | None = None, linearize: int | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, antiring: float | None = None, filter: _AnyStr | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, shader_s: _AnyStr | None = None, log_level: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tonemap(self, clip: VideoNode, src_csp: int | None = None, dst_csp: int | None = None, dst_prim: int | None = None, src_max: float | None = None, src_min: float | None = None, dst_max: float | None = None, dst_min: float | None = None, dynamic_peak_detection: int | None = None, smoothing_period: float | None = None, scene_threshold_low: float | None = None, scene_threshold_high: float | None = None, percentile: float | None = None, gamut_mapping: int | None = None, tone_mapping_function: int | None = None, tone_mapping_function_s: _AnyStr | None = None, tone_mapping_param: float | None = None, metadata: int | None = None, use_dovi: int | None = None, visualize_lut: int | None = None, show_clipping: int | None = None, contrast_recovery: float | None = None, log_level: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Deband(self, planes: int | None = None, iterations: int | None = None, threshold: float | None = None, radius: float | None = None, grain: float | None = None, dither: int | None = None, dither_algo: int | None = None, log_level: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Resample(self, width: int, height: int, filter: _AnyStr | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, src_width: float | None = None, src_height: float | None = None, sx: float | None = None, sy: float | None = None, antiring: float | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, linearize: int | None = None, trc: int | None = None, min_luma: float | None = None, log_level: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Shader(self, shader: _AnyStr | None = None, width: int | None = None, height: int | None = None, chroma_loc: int | None = None, matrix: int | None = None, trc: int | None = None, linearize: int | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, antiring: float | None = None, filter: _AnyStr | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, shader_s: _AnyStr | None = None, log_level: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Tonemap(self, src_csp: int | None = None, dst_csp: int | None = None, dst_prim: int | None = None, src_max: float | None = None, src_min: float | None = None, dst_max: float | None = None, dst_min: float | None = None, dynamic_peak_detection: int | None = None, smoothing_period: float | None = None, scene_threshold_low: float | None = None, scene_threshold_high: float | None = None, percentile: float | None = None, gamut_mapping: int | None = None, tone_mapping_function: int | None = None, tone_mapping_function_s: _AnyStr | None = None, tone_mapping_param: float | None = None, metadata: int | None = None, use_dovi: int | None = None, visualize_lut: int | None = None, show_clipping: int | None = None, contrast_recovery: float | None = None, log_level: int | None = None) -> VideoNode: ...

# </implementation/placebo>

# <implementation/resize>
class _resize:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, clip: VideoNode, filter: _AnyStr | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, filter: _AnyStr | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...

# </implementation/resize>

# <implementation/resize2>
class _resize2:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, clip: VideoNode, filter: _AnyStr | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Custom(self, clip: VideoNode, custom_kernel: Func | _VSCallback_resize2_Custom_custom_kernel, taps: int, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Bicubic(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilinear(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bob(self, filter: _AnyStr | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Custom(self, custom_kernel: Func | _VSCallback_resize2_Custom_custom_kernel, taps: int, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lanczos(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Point(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline16(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline36(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Spline64(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None, range_s: _AnyStr | None = None, chromaloc: int | None = None, chromaloc_s: _AnyStr | None = None, matrix_in: int | None = None, matrix_in_s: _AnyStr | None = None, transfer_in: int | None = None, transfer_in_s: _AnyStr | None = None, primaries_in: int | None = None, primaries_in_s: _AnyStr | None = None, range_in: int | None = None, range_in_s: _AnyStr | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _AnyStr | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _AnyStr | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _AnyStr | None = None, cpu_type: _AnyStr | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, blur: float | None = None) -> VideoNode: ...

# </implementation/resize2>

# <implementation/sangnom>
class _sangnom:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def SangNom(self, clip: VideoNode, order: int | None = None, dh: int | None = None, aa: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def SangNom(self, order: int | None = None, dh: int | None = None, aa: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

# </implementation/sangnom>

# <implementation/scxvid>
class _scxvid:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Scxvid(self, clip: VideoNode, log: _AnyStr | None = None, use_slices: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def Scxvid(self, log: _AnyStr | None = None, use_slices: int | None = None) -> VideoNode: ...

# </implementation/scxvid>

# <implementation/sneedif>
_ReturnDict_sneedif_DeviceInfo = TypedDict("_ReturnDict_sneedif_DeviceInfo", {"name": _AnyStr, "vendor": _AnyStr, "profile": _AnyStr, "version": _AnyStr, "max_compute_units": int, "max_work_group_size": int, "max_work_item_sizes": int | _SequenceLike[int], "image2D_max_width": int, "image2D_max_height": int, "image_support": int, "global_memory_cache_type": _AnyStr, "global_memory_cache": int, "global_memory_size": int, "max_constant_buffer_size": int, "max_constant_arguments": int, "local_memory_type": _AnyStr, "local_memory_size": int, "available": int, "compiler_available": int, "linker_available": int, "opencl_c_version": _AnyStr, "image_max_buffer_size": int})
_ReturnDict_sneedif_ListDevices = TypedDict("_ReturnDict_sneedif_ListDevices", {"numDevices": int, "deviceNames": _AnyStr | _SequenceLike[_AnyStr], "platformNames": _AnyStr | _SequenceLike[_AnyStr]})
_ReturnDict_sneedif_PlatformInfo = TypedDict("_ReturnDict_sneedif_PlatformInfo", {"profile": _AnyStr, "version": _AnyStr, "name": _AnyStr, "vendor": _AnyStr})

class _sneedif:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def DeviceInfo(self, device: int | None = None) -> _ReturnDict_sneedif_DeviceInfo: ...
            @_Wrapper.Function
            def ListDevices(self) -> _ReturnDict_sneedif_ListDevices: ...
            @_Wrapper.Function
            def NNEDI3(self, clip: VideoNode, field: int, dh: int | None = None, dw: int | None = None, planes: int | _SequenceLike[int] | None = None, nsize: int | None = None, nns: int | None = None, qual: int | None = None, etype: int | None = None, pscrn: int | None = None, transpose_first: int | None = None, device: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlatformInfo(self, device: int | None = None) -> _ReturnDict_sneedif_PlatformInfo: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def NNEDI3(self, field: int, dh: int | None = None, dw: int | None = None, planes: int | _SequenceLike[int] | None = None, nsize: int | None = None, nns: int | None = None, qual: int | None = None, etype: int | None = None, pscrn: int | None = None, transpose_first: int | None = None, device: int | None = None) -> VideoNode: ...

# </implementation/sneedif>

# <implementation/std>
class _std:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AddBorders(self, clip: VideoNode, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None, color: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AssumeFPS(self, clip: VideoNode, src: VideoNode | None = None, fpsnum: int | None = None, fpsden: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AssumeSampleRate(self, clip: AudioNode, src: AudioNode | None = None, samplerate: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioGain(self, clip: AudioNode, gain: float | _SequenceLike[float] | None = None, overflow_error: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioLoop(self, clip: AudioNode, times: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioMix(self, clips: AudioNode | _SequenceLike[AudioNode], matrix: float | _SequenceLike[float], channels_out: int | _SequenceLike[int], overflow_error: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioReverse(self, clip: AudioNode) -> AudioNode: ...
            @_Wrapper.Function
            def AudioSplice(self, clips: AudioNode | _SequenceLike[AudioNode]) -> AudioNode: ...
            @_Wrapper.Function
            def AudioTrim(self, clip: AudioNode, first: int | None = None, last: int | None = None, length: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AverageFrames(self, clips: VideoNode | _SequenceLike[VideoNode], weights: float | _SequenceLike[float], scale: float | None = None, scenechange: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Binarize(self, clip: VideoNode, threshold: float | _SequenceLike[float] | None = None, v0: float | _SequenceLike[float] | None = None, v1: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BinarizeMask(self, clip: VideoNode, threshold: float | _SequenceLike[float] | None = None, v0: float | _SequenceLike[float] | None = None, v1: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlankAudio(self, clip: AudioNode | None = None, channels: int | _SequenceLike[int] | None = None, bits: int | None = None, sampletype: int | None = None, samplerate: int | None = None, length: int | None = None, keep: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def BlankClip(self, clip: VideoNode | None = None, width: int | None = None, height: int | None = None, format: int | None = None, length: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, color: float | _SequenceLike[float] | None = None, keep: int | None = None, varsize: int | None = None, varformat: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Cache(self, clip: VideoNode, size: int | None = None, fixed: int | None = None, make_linear: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ClipToProp(self, clip: VideoNode, mclip: VideoNode, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Convolution(self, clip: VideoNode, matrix: float | _SequenceLike[float], bias: float | None = None, divisor: float | None = None, planes: int | _SequenceLike[int] | None = None, saturate: int | None = None, mode: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CopyFrameProps(self, clip: VideoNode, prop_src: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Crop(self, clip: VideoNode, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropAbs(self, clip: VideoNode, width: int, height: int, left: int | None = None, top: int | None = None, x: int | None = None, y: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropRel(self, clip: VideoNode, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Deflate(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DeleteFrames(self, clip: VideoNode, frames: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def DoubleWeave(self, clip: VideoNode, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DuplicateFrames(self, clip: VideoNode, frames: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, clips: VideoNode | _SequenceLike[VideoNode], expr: _AnyStr | _SequenceLike[_AnyStr], format: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlipHorizontal(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def FlipVertical(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper_Core_bound_FrameEval.Function
            def FrameEval(self, clip: VideoNode, eval: Func | _VSCallback_std_FrameEval_eval, prop_src: VideoNode | _SequenceLike[VideoNode] | None = None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FreezeFrames(self, clip: VideoNode, first: int | _SequenceLike[int] | None = None, last: int | _SequenceLike[int] | None = None, replacement: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Inflate(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Interleave(self, clips: VideoNode | _SequenceLike[VideoNode], extend: int | None = None, mismatch: int | None = None, modify_duration: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Invert(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InvertMask(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Levels(self, clip: VideoNode, min_in: float | _SequenceLike[float] | None = None, max_in: float | _SequenceLike[float] | None = None, gamma: float | _SequenceLike[float] | None = None, min_out: float | _SequenceLike[float] | None = None, max_out: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, clip: VideoNode, min: float | _SequenceLike[float] | None = None, max: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LoadAllPlugins(self, path: _AnyStr) -> None: ...
            @_Wrapper.Function
            def LoadPlugin(self, path: _AnyStr, altsearchpath: int | None = None, forcens: _AnyStr | None = None, forceid: _AnyStr | None = None) -> None: ...
            @_Wrapper.Function
            def Loop(self, clip: VideoNode, times: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, lut: int | _SequenceLike[int] | None = None, lutf: float | _SequenceLike[float] | None = None, function: Func | _VSCallback_std_Lut_function | None = None, bits: int | None = None, floatout: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut2(self, clipa: VideoNode, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None, lut: int | _SequenceLike[int] | None = None, lutf: float | _SequenceLike[float] | None = None, function: Func | _VSCallback_std_Lut2_function | None = None, bits: int | None = None, floatout: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeDiff(self, clipa: VideoNode, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def MaskedMerge(self, clipa: VideoNode, clipb: VideoNode, mask: VideoNode, planes: int | _SequenceLike[int] | None = None, first_plane: int | None = None, premultiplied: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Maximum(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None, coordinates: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Merge(self, clipa: VideoNode, clipb: VideoNode, weight: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeDiff(self, clipa: VideoNode, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Minimum(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None, coordinates: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper_Core_bound_ModifyFrame.Function
            def ModifyFrame(self, clip: VideoNode, clips: VideoNode | _SequenceLike[VideoNode], selector: Func | _VSCallback_std_ModifyFrame_selector) -> VideoNode: ...
            @_Wrapper.Function
            def PEMVerifier(self, clip: VideoNode, upper: float | _SequenceLike[float] | None = None, lower: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneStats(self, clipa: VideoNode, clipb: VideoNode | None = None, plane: int | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PreMultiply(self, clip: VideoNode, alpha: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropToClip(self, clip: VideoNode, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveFrameProps(self, clip: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Reverse(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def SelectEvery(self, clip: VideoNode, cycle: int, offsets: int | _SequenceLike[int], modify_duration: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SeparateFields(self, clip: VideoNode, tff: int | None = None, modify_duration: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetAudioCache(self, clip: AudioNode, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> None: ...
            @_Wrapper.Function
            def SetFieldBased(self, clip: VideoNode, value: int) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProp(self, clip: VideoNode, prop: _AnyStr, intval: int | _SequenceLike[int] | None = None, floatval: float | _SequenceLike[float] | None = None, data: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProps(self, clip: VideoNode, **kwargs: Any) -> VideoNode: ...
            @_Wrapper.Function
            def SetMaxCPU(self, cpu: _AnyStr) -> _AnyStr: ...
            @_Wrapper.Function
            def SetVideoCache(self, clip: VideoNode, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> None: ...
            @_Wrapper.Function
            def ShuffleChannels(self, clips: AudioNode | _SequenceLike[AudioNode], channels_in: int | _SequenceLike[int], channels_out: int | _SequenceLike[int]) -> AudioNode: ...
            @_Wrapper.Function
            def ShufflePlanes(self, clips: VideoNode | _SequenceLike[VideoNode], planes: int | _SequenceLike[int], colorfamily: int, prop_src: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, scale: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Splice(self, clips: VideoNode | _SequenceLike[VideoNode], mismatch: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SplitChannels(self, clip: AudioNode) -> AudioNode | list[AudioNode]: ...
            @_Wrapper.Function
            def SplitPlanes(self, clip: VideoNode) -> VideoNode | list[VideoNode]: ...
            @_Wrapper.Function
            def StackHorizontal(self, clips: VideoNode | _SequenceLike[VideoNode]) -> VideoNode: ...
            @_Wrapper.Function
            def StackVertical(self, clips: VideoNode | _SequenceLike[VideoNode]) -> VideoNode: ...
            @_Wrapper.Function
            def TestAudio(self, channels: int | _SequenceLike[int] | None = None, bits: int | None = None, isfloat: int | None = None, samplerate: int | None = None, length: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def Transpose(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Trim(self, clip: VideoNode, first: int | None = None, last: int | None = None, length: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Turn180(self, clip: VideoNode) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AddBorders(self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None, color: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AssumeFPS(self, src: VideoNode | None = None, fpsnum: int | None = None, fpsden: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def AverageFrames(self, weights: float | _SequenceLike[float], scale: float | None = None, scenechange: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Binarize(self, threshold: float | _SequenceLike[float] | None = None, v0: float | _SequenceLike[float] | None = None, v1: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BinarizeMask(self, threshold: float | _SequenceLike[float] | None = None, v0: float | _SequenceLike[float] | None = None, v1: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BlankClip(self, width: int | None = None, height: int | None = None, format: int | None = None, length: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, color: float | _SequenceLike[float] | None = None, keep: int | None = None, varsize: int | None = None, varformat: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, planes: int | _SequenceLike[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Cache(self, size: int | None = None, fixed: int | None = None, make_linear: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ClipToProp(self, mclip: VideoNode, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Convolution(self, matrix: float | _SequenceLike[float], bias: float | None = None, divisor: float | None = None, planes: int | _SequenceLike[int] | None = None, saturate: int | None = None, mode: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CopyFrameProps(self, prop_src: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Crop(self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropAbs(self, width: int, height: int, left: int | None = None, top: int | None = None, x: int | None = None, y: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CropRel(self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Deflate(self, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DeleteFrames(self, frames: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def DoubleWeave(self, tff: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DuplicateFrames(self, frames: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def Expr(self, expr: _AnyStr | _SequenceLike[_AnyStr], format: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FlipHorizontal(self) -> VideoNode: ...
            @_Wrapper.Function
            def FlipVertical(self) -> VideoNode: ...
            @_Wrapper_VideoNode_bound_FrameEval.Function
            def FrameEval(self, eval: Func | _VSCallback_std_FrameEval_eval, prop_src: VideoNode | _SequenceLike[VideoNode] | None = None, clip_src: VideoNode | _SequenceLike[VideoNode] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FreezeFrames(self, first: int | _SequenceLike[int] | None = None, last: int | _SequenceLike[int] | None = None, replacement: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Inflate(self, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Interleave(self, extend: int | None = None, mismatch: int | None = None, modify_duration: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Invert(self, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InvertMask(self, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Levels(self, min_in: float | _SequenceLike[float] | None = None, max_in: float | _SequenceLike[float] | None = None, gamma: float | _SequenceLike[float] | None = None, min_out: float | _SequenceLike[float] | None = None, max_out: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, min: float | _SequenceLike[float] | None = None, max: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Loop(self, times: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut(self, planes: int | _SequenceLike[int] | None = None, lut: int | _SequenceLike[int] | None = None, lutf: float | _SequenceLike[float] | None = None, function: Func | _VSCallback_std_Lut_function | None = None, bits: int | None = None, floatout: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Lut2(self, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None, lut: int | _SequenceLike[int] | None = None, lutf: float | _SequenceLike[float] | None = None, function: Func | _VSCallback_std_Lut2_function | None = None, bits: int | None = None, floatout: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeDiff(self, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MakeFullDiff(self, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def MaskedMerge(self, clipb: VideoNode, mask: VideoNode, planes: int | _SequenceLike[int] | None = None, first_plane: int | None = None, premultiplied: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Maximum(self, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None, coordinates: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Merge(self, clipb: VideoNode, weight: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeDiff(self, clipb: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def MergeFullDiff(self, clipb: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Minimum(self, planes: int | _SequenceLike[int] | None = None, threshold: float | None = None, coordinates: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper_VideoNode_bound_ModifyFrame.Function
            def ModifyFrame(self, clips: VideoNode | _SequenceLike[VideoNode], selector: Func | _VSCallback_std_ModifyFrame_selector) -> VideoNode: ...
            @_Wrapper.Function
            def PEMVerifier(self, upper: float | _SequenceLike[float] | None = None, lower: float | _SequenceLike[float] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneStats(self, clipb: VideoNode | None = None, plane: int | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PreMultiply(self, alpha: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def Prewitt(self, planes: int | _SequenceLike[int] | None = None, scale: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PropToClip(self, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveFrameProps(self, props: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Reverse(self) -> VideoNode: ...
            @_Wrapper.Function
            def SelectEvery(self, cycle: int, offsets: int | _SequenceLike[int], modify_duration: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SeparateFields(self, tff: int | None = None, modify_duration: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetFieldBased(self, value: int) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProp(self, prop: _AnyStr, intval: int | _SequenceLike[int] | None = None, floatval: float | _SequenceLike[float] | None = None, data: _AnyStr | _SequenceLike[_AnyStr] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SetFrameProps(self, **kwargs: Any) -> VideoNode: ...
            @_Wrapper.Function
            def SetVideoCache(self, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> None: ...
            @_Wrapper.Function
            def ShufflePlanes(self, planes: int | _SequenceLike[int], colorfamily: int, prop_src: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Sobel(self, planes: int | _SequenceLike[int] | None = None, scale: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Splice(self, mismatch: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SplitPlanes(self) -> VideoNode | list[VideoNode]: ...
            @_Wrapper.Function
            def StackHorizontal(self) -> VideoNode: ...
            @_Wrapper.Function
            def StackVertical(self) -> VideoNode: ...
            @_Wrapper.Function
            def Transpose(self) -> VideoNode: ...
            @_Wrapper.Function
            def Trim(self, first: int | None = None, last: int | None = None, length: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Turn180(self) -> VideoNode: ...

    class _AudioNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AssumeSampleRate(self, src: AudioNode | None = None, samplerate: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioGain(self, gain: float | _SequenceLike[float] | None = None, overflow_error: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioLoop(self, times: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioMix(self, matrix: float | _SequenceLike[float], channels_out: int | _SequenceLike[int], overflow_error: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def AudioReverse(self) -> AudioNode: ...
            @_Wrapper.Function
            def AudioSplice(self) -> AudioNode: ...
            @_Wrapper.Function
            def AudioTrim(self, first: int | None = None, last: int | None = None, length: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def BlankAudio(self, channels: int | _SequenceLike[int] | None = None, bits: int | None = None, sampletype: int | None = None, samplerate: int | None = None, length: int | None = None, keep: int | None = None) -> AudioNode: ...
            @_Wrapper.Function
            def SetAudioCache(self, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> None: ...
            @_Wrapper.Function
            def ShuffleChannels(self, channels_in: int | _SequenceLike[int], channels_out: int | _SequenceLike[int]) -> AudioNode: ...
            @_Wrapper.Function
            def SplitChannels(self) -> AudioNode | list[AudioNode]: ...

# </implementation/std>

# <implementation/sub>
class _sub:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ImageFile(self, clip: VideoNode, file: _AnyStr, id: int | None = None, palette: int | _SequenceLike[int] | None = None, gray: int | None = None, info: int | None = None, flatten: int | None = None, blend: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Subtitle(self, clip: VideoNode, text: _AnyStr, start: int | None = None, end: int | None = None, debuglevel: int | None = None, fontdir: _AnyStr | None = None, linespacing: float | None = None, margins: int | _SequenceLike[int] | None = None, sar: float | None = None, style: _AnyStr | None = None, blend: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TextFile(self, clip: VideoNode, file: _AnyStr, charset: _AnyStr | None = None, scale: float | None = None, debuglevel: int | None = None, fontdir: _AnyStr | None = None, linespacing: float | None = None, margins: int | _SequenceLike[int] | None = None, sar: float | None = None, style: _AnyStr | None = None, blend: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ImageFile(self, file: _AnyStr, id: int | None = None, palette: int | _SequenceLike[int] | None = None, gray: int | None = None, info: int | None = None, flatten: int | None = None, blend: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Subtitle(self, text: _AnyStr, start: int | None = None, end: int | None = None, debuglevel: int | None = None, fontdir: _AnyStr | None = None, linespacing: float | None = None, margins: int | _SequenceLike[int] | None = None, sar: float | None = None, style: _AnyStr | None = None, blend: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TextFile(self, file: _AnyStr, charset: _AnyStr | None = None, scale: float | None = None, debuglevel: int | None = None, fontdir: _AnyStr | None = None, linespacing: float | None = None, margins: int | _SequenceLike[int] | None = None, sar: float | None = None, style: _AnyStr | None = None, blend: int | None = None, matrix: int | None = None, matrix_s: _AnyStr | None = None, transfer: int | None = None, transfer_s: _AnyStr | None = None, primaries: int | None = None, primaries_s: _AnyStr | None = None, range: int | None = None) -> VideoNode: ...

# </implementation/sub>

# <implementation/tcanny>
class _tcanny:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def TCanny(self, clip: VideoNode, sigma: float | _SequenceLike[float] | None = None, sigma_v: float | _SequenceLike[float] | None = None, t_h: float | None = None, t_l: float | None = None, mode: int | None = None, op: int | None = None, scale: float | None = None, opt: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def TCanny(self, sigma: float | _SequenceLike[float] | None = None, sigma_v: float | _SequenceLike[float] | None = None, t_h: float | None = None, t_l: float | None = None, mode: int | None = None, op: int | None = None, scale: float | None = None, opt: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...

# </implementation/tcanny>

# <implementation/text>
class _text:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ClipInfo(self, clip: VideoNode, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CoreInfo(self, clip: VideoNode | None = None, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameNum(self, clip: VideoNode, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameProps(self, clip: VideoNode, props: _AnyStr | _SequenceLike[_AnyStr] | None = None, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, clip: VideoNode, text: _AnyStr, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def ClipInfo(self, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CoreInfo(self, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameNum(self, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FrameProps(self, props: _AnyStr | _SequenceLike[_AnyStr] | None = None, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Text(self, text: _AnyStr, alignment: int | None = None, scale: int | None = None) -> VideoNode: ...

# </implementation/text>

# <implementation/vivtc>
class _vivtc:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VDecimate(self, clip: VideoNode, cycle: int | None = None, chroma: int | None = None, dupthresh: float | None = None, scthresh: float | None = None, blockx: int | None = None, blocky: int | None = None, clip2: VideoNode | None = None, ovr: _AnyStr | None = None, dryrun: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFM(self, clip: VideoNode, order: int, field: int | None = None, mode: int | None = None, mchroma: int | None = None, cthresh: int | None = None, mi: int | None = None, chroma: int | None = None, blockx: int | None = None, blocky: int | None = None, y0: int | None = None, y1: int | None = None, scthresh: float | None = None, micmatch: int | None = None, micout: int | None = None, clip2: VideoNode | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VDecimate(self, cycle: int | None = None, chroma: int | None = None, dupthresh: float | None = None, scthresh: float | None = None, blockx: int | None = None, blocky: int | None = None, clip2: VideoNode | None = None, ovr: _AnyStr | None = None, dryrun: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VFM(self, order: int, field: int | None = None, mode: int | None = None, mchroma: int | None = None, cthresh: int | None = None, mi: int | None = None, chroma: int | None = None, blockx: int | None = None, blocky: int | None = None, y0: int | None = None, y1: int | None = None, scthresh: float | None = None, micmatch: int | None = None, micout: int | None = None, clip2: VideoNode | None = None) -> VideoNode: ...

# </implementation/vivtc>

# <implementation/vszip>
class _vszip:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AdaptiveBinarize(self, clip: VideoNode, clip2: VideoNode, c: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilateral(self, clip: VideoNode, ref: VideoNode | None = None, sigmaS: float | _SequenceLike[float] | None = None, sigmaR: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, algorithm: int | _SequenceLike[int] | None = None, PBFICnum: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CLAHE(self, clip: VideoNode, limit: int | None = None, tiles: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Checkmate(self, clip: VideoNode, thr: int | None = None, tmax: int | None = None, tthr2: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ColorMap(self, clip: VideoNode, color: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CombMaskMT(self, clip: VideoNode, thY1: int | None = None, thY2: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ImageRead(self, path: _AnyStr | _SequenceLike[_AnyStr], validate: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LimitFilter(self, flt: VideoNode, src: VideoNode, ref: VideoNode | None = None, dark_thr: float | _SequenceLike[float] | None = None, bright_thr: float | _SequenceLike[float] | None = None, elast: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, clip: VideoNode, min: float | _SequenceLike[float] | None = None, max: float | _SequenceLike[float] | None = None, tv_range: int | None = None, mask: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Metrics(self, reference: VideoNode, distorted: VideoNode, mode: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PackRGB(self, clip: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneAverage(self, clipa: VideoNode, exclude: int | _SequenceLike[int], clipb: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneMinMax(self, clipa: VideoNode, minthr: float | None = None, maxthr: float | None = None, clipb: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RFS(self, clipa: VideoNode, clipb: VideoNode, frames: int | _SequenceLike[int], mismatch: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SSIMULACRA2(self, reference: VideoNode, distorted: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def XPSNR(self, reference: VideoNode, distorted: VideoNode, temporal: int | None = None, verbose: int | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def AdaptiveBinarize(self, clip2: VideoNode, c: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Bilateral(self, ref: VideoNode | None = None, sigmaS: float | _SequenceLike[float] | None = None, sigmaR: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, algorithm: int | _SequenceLike[int] | None = None, PBFICnum: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def BoxBlur(self, planes: int | _SequenceLike[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CLAHE(self, limit: int | None = None, tiles: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Checkmate(self, thr: int | None = None, tmax: int | None = None, tthr2: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ColorMap(self, color: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CombMaskMT(self, thY1: int | None = None, thY2: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def LimitFilter(self, src: VideoNode, ref: VideoNode | None = None, dark_thr: float | _SequenceLike[float] | None = None, bright_thr: float | _SequenceLike[float] | None = None, elast: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Limiter(self, min: float | _SequenceLike[float] | None = None, max: float | _SequenceLike[float] | None = None, tv_range: int | None = None, mask: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Metrics(self, distorted: VideoNode, mode: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PackRGB(self) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneAverage(self, exclude: int | _SequenceLike[int], clipb: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def PlaneMinMax(self, minthr: float | None = None, maxthr: float | None = None, clipb: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None, prop: _AnyStr | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RFS(self, clipb: VideoNode, frames: int | _SequenceLike[int], mismatch: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def SSIMULACRA2(self, distorted: VideoNode) -> VideoNode: ...
            @_Wrapper.Function
            def XPSNR(self, distorted: VideoNode, temporal: int | None = None, verbose: int | None = None) -> VideoNode: ...

# </implementation/vszip>

# <implementation/wnnm>
class _wnnm:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VAggregate(self, clip: VideoNode, src: VideoNode, planes: int | _SequenceLike[int], internal: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Version(self) -> VideoNode: ...
            @_Wrapper.Function
            def WNNM(self, clip: VideoNode, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, residual: int | None = None, adaptive_aggregation: int | None = None, rclip: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def WNNMRaw(self, clip: VideoNode, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, residual: int | None = None, adaptive_aggregation: int | None = None, rclip: VideoNode | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def VAggregate(self, src: VideoNode, planes: int | _SequenceLike[int], internal: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def WNNM(self, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, residual: int | None = None, adaptive_aggregation: int | None = None, rclip: VideoNode | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def WNNMRaw(self, sigma: float | _SequenceLike[float] | None = None, block_size: int | None = None, block_step: int | None = None, group_size: int | None = None, bm_range: int | None = None, radius: int | None = None, ps_num: int | None = None, ps_range: int | None = None, residual: int | None = None, adaptive_aggregation: int | None = None, rclip: VideoNode | None = None) -> VideoNode: ...

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
            def nnedi3(self, clip: VideoNode, field: int, dh: int | None = None, planes: int | _SequenceLike[int] | None = None, nsize: int | None = None, nns: int | None = None, qual: int | None = None, etype: int | None = None, pscrn: int | None = None, opt: int | None = None, int16_prescreener: int | None = None, int16_predictor: int | None = None, exp: int | None = None, show_mask: int | None = None, x_nnedi3_weights_bin: _AnyStr | None = None, x_cpu: _AnyStr | None = None) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def nnedi3(self, field: int, dh: int | None = None, planes: int | _SequenceLike[int] | None = None, nsize: int | None = None, nns: int | None = None, qual: int | None = None, etype: int | None = None, pscrn: int | None = None, opt: int | None = None, int16_prescreener: int | None = None, int16_predictor: int | None = None, exp: int | None = None, show_mask: int | None = None, x_nnedi3_weights_bin: _AnyStr | None = None, x_cpu: _AnyStr | None = None) -> VideoNode: ...

# </implementation/znedi3>

# <implementation/zsmooth>
class _zsmooth:
    class _Core_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BackwardClense(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CCD(self, clip: VideoNode, threshold: float | None = None, temporal_radius: int | None = None, points: int | _SequenceLike[int] | None = None, scale: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Clense(self, clip: VideoNode, previous: VideoNode | None = None, next: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DegrainMedian(self, clip: VideoNode, limit: float | _SequenceLike[float] | None = None, mode: int | _SequenceLike[int] | None = None, interlaced: int | None = None, norow: int | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothST(self, clip: VideoNode, temporal_threshold: float | _SequenceLike[float] | None = None, spatial_threshold: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothT(self, clip: VideoNode, temporal_threshold: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ForwardClense(self, clip: VideoNode, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InterQuartileMean(self, clip: VideoNode, radius: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, clip: VideoNode, radius: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveGrain(self, clip: VideoNode, mode: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def Repair(self, clip: VideoNode, repairclip: VideoNode, mode: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def SmartMedian(self, clip: VideoNode, radius: int | _SequenceLike[int] | None = None, threshold: float | _SequenceLike[float] | None = None, scalep: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TTempSmooth(self, clip: VideoNode, maxr: int | None = None, thresh: int | _SequenceLike[int] | None = None, mdiff: int | _SequenceLike[int] | None = None, strength: int | None = None, scthresh: float | None = None, fp: int | None = None, pfclip: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalMedian(self, clip: VideoNode, radius: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalRepair(self, clip: VideoNode, repairclip: VideoNode, mode: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalSoften(self, clip: VideoNode, radius: int | None = None, threshold: float | _SequenceLike[float] | None = None, scenechange: int | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VerticalCleaner(self, clip: VideoNode, mode: int | _SequenceLike[int]) -> VideoNode: ...

    class _VideoNode_bound:
        class Plugin(_VSPlugin):
            @_Wrapper.Function
            def BackwardClense(self, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def CCD(self, threshold: float | None = None, temporal_radius: int | None = None, points: int | _SequenceLike[int] | None = None, scale: float | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Clense(self, previous: VideoNode | None = None, next: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def DegrainMedian(self, limit: float | _SequenceLike[float] | None = None, mode: int | _SequenceLike[int] | None = None, interlaced: int | None = None, norow: int | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothST(self, temporal_threshold: float | _SequenceLike[float] | None = None, spatial_threshold: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def FluxSmoothT(self, temporal_threshold: float | _SequenceLike[float] | None = None, planes: int | _SequenceLike[int] | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def ForwardClense(self, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def InterQuartileMean(self, radius: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def Median(self, radius: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def RemoveGrain(self, mode: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def Repair(self, repairclip: VideoNode, mode: int | _SequenceLike[int]) -> VideoNode: ...
            @_Wrapper.Function
            def SmartMedian(self, radius: int | _SequenceLike[int] | None = None, threshold: float | _SequenceLike[float] | None = None, scalep: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TTempSmooth(self, maxr: int | None = None, thresh: int | _SequenceLike[int] | None = None, mdiff: int | _SequenceLike[int] | None = None, strength: int | None = None, scthresh: float | None = None, fp: int | None = None, pfclip: VideoNode | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalMedian(self, radius: int | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalRepair(self, repairclip: VideoNode, mode: int | _SequenceLike[int] | None = None, planes: int | _SequenceLike[int] | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def TemporalSoften(self, radius: int | None = None, threshold: float | _SequenceLike[float] | None = None, scenechange: int | None = None, scalep: int | None = None) -> VideoNode: ...
            @_Wrapper.Function
            def VerticalCleaner(self, mode: int | _SequenceLike[int]) -> VideoNode: ...

# </implementation/zsmooth>

# </plugins/implementations>

class VideoOutputTuple(NamedTuple):
    clip: VideoNode
    alpha: VideoNode | None
    alt_output: Literal[0, 1, 2]

def clear_output(index: int = 0) -> None: ...
def clear_outputs() -> None: ...
def get_outputs() -> MappingProxyType[int, VideoOutputTuple | AudioNode]: ...
def get_output(index: int = 0) -> VideoOutputTuple | AudioNode: ...

def construct_signature(
    signature: str | Function, return_signature: str, injected: bool | None = None, name: str | None = None
) -> Signature: ...
def _construct_type(signature: str) -> Any: ...
def _construct_parameter(signature: str) -> Any: ...
def _construct_repr_wrap(value: str | Enum | VideoFormat | Iterator[str]) -> str: ...
def _construct_repr(obj: Any, **kwargs: Any) -> str: ...
