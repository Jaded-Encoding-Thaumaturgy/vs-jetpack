from abc import abstractmethod
from ctypes import c_void_p
from enum import IntEnum, IntFlag
from fractions import Fraction
from inspect import Signature
from types import MappingProxyType, TracebackType
from typing import (
    TYPE_CHECKING, Any, BinaryIO, Callable, ContextManager, Dict, Generic, Iterator, Literal,
    MutableMapping, NamedTuple, NoReturn, Protocol, Sequence, Tuple, TypedDict,
    TypeVar, Union, cast, overload, runtime_checkable
)
from weakref import ReferenceType

__all__ = [
    # Versioning
    '__version__', '__api_version__', 'PluginVersion',

    # Enums and constants
    'MessageType',
        'MESSAGE_TYPE_DEBUG', 'MESSAGE_TYPE_INFORMATION', 'MESSAGE_TYPE_WARNING',
        'MESSAGE_TYPE_CRITICAL', 'MESSAGE_TYPE_FATAL',

    'FilterMode',
        'fmParallel', 'fmParallelRequests', 'fmUnordered', 'fmFrameState',

    'CoreCreationFlags',
        'ccfEnableGraphInspection', 'ccfDisableAutoLoading', 'ccfDisableLibraryUnloading',

    'MediaType',
        'VIDEO', 'AUDIO',

    'ColorFamily',
        'UNDEFINED', 'GRAY', 'RGB', 'YUV',

    'ColorRange',
        'RANGE_FULL', 'RANGE_LIMITED',

    'SampleType',
        'INTEGER', 'FLOAT',

    'PresetVideoFormat',
        'GRAY',
        'GRAY8', 'GRAY9', 'GRAY10', 'GRAY12', 'GRAY14', 'GRAY16', 'GRAY32', 'GRAYH', 'GRAYS',
        'RGB',
        'RGB24', 'RGB27', 'RGB30', 'RGB36', 'RGB42', 'RGB48', 'RGBH', 'RGBS',
        'YUV',
        'YUV410P8',
        'YUV411P8',
        'YUV420P8', 'YUV420P9', 'YUV420P10', 'YUV420P12', 'YUV420P14', 'YUV420P16',
        'YUV422P8', 'YUV422P9', 'YUV422P10', 'YUV422P12', 'YUV422P14', 'YUV422P16',
        'YUV440P8',
        'YUV444P8', 'YUV444P9', 'YUV444P10', 'YUV444P12', 'YUV444P14', 'YUV444P16',
        'YUV420PH', 'YUV422PH', 'YUV444PH',
        'YUV420PS', 'YUV422PS', 'YUV444PS',
        'NONE',

    'AudioChannels',
        'FRONT_LEFT', 'FRONT_RIGHT', 'FRONT_CENTER',
        'BACK_LEFT', 'BACK_RIGHT', 'BACK_CENTER',
        'SIDE_LEFT', 'SIDE_RIGHT',
        'TOP_CENTER',

        'TOP_FRONT_LEFT', 'TOP_FRONT_RIGHT', 'TOP_FRONT_CENTER',
        'TOP_BACK_LEFT', 'TOP_BACK_RIGHT', 'TOP_BACK_CENTER',

        'WIDE_LEFT', 'WIDE_RIGHT',

        'SURROUND_DIRECT_LEFT', 'SURROUND_DIRECT_RIGHT',

        'FRONT_LEFT_OF_CENTER', 'FRONT_RIGHT_OF_CENTER',

        'STEREO_LEFT', 'STEREO_RIGHT',

        'LOW_FREQUENCY', 'LOW_FREQUENCY2',

    'ChromaLocation',
        'CHROMA_TOP_LEFT', 'CHROMA_TOP',
        'CHROMA_LEFT', 'CHROMA_CENTER',
        'CHROMA_BOTTOM_LEFT', 'CHROMA_BOTTOM',

    'FieldBased',
        'FIELD_PROGRESSIVE', 'FIELD_TOP', 'FIELD_BOTTOM',

    'MatrixCoefficients',
        'MATRIX_RGB', 'MATRIX_BT709', 'MATRIX_UNSPECIFIED', 'MATRIX_FCC',
        'MATRIX_BT470_BG', 'MATRIX_ST170_M', 'MATRIX_ST240_M', 'MATRIX_YCGCO', 'MATRIX_BT2020_NCL', 'MATRIX_BT2020_CL',
        'MATRIX_CHROMATICITY_DERIVED_NCL', 'MATRIX_CHROMATICITY_DERIVED_CL', 'MATRIX_ICTCP',

    'TransferCharacteristics',
        'TRANSFER_BT709', 'TRANSFER_UNSPECIFIED', 'TRANSFER_BT470_M', 'TRANSFER_BT470_BG', 'TRANSFER_BT601',
        'TRANSFER_ST240_M', 'TRANSFER_LINEAR', 'TRANSFER_LOG_100', 'TRANSFER_LOG_316', 'TRANSFER_IEC_61966_2_4',
        'TRANSFER_IEC_61966_2_1', 'TRANSFER_BT2020_10', 'TRANSFER_BT2020_12', 'TRANSFER_ST2084', 'TRANSFER_ST428',
        'TRANSFER_ARIB_B67',

    'ColorPrimaries', 'PRIMARIES_BT709', 'PRIMARIES_UNSPECIFIED',
        'PRIMARIES_BT470_M', 'PRIMARIES_BT470_BG', 'PRIMARIES_ST170_M', 'PRIMARIES_ST240_M', 'PRIMARIES_FILM',
        'PRIMARIES_BT2020', 'PRIMARIES_ST428', 'PRIMARIES_ST431_2', 'PRIMARIES_ST432_1', 'PRIMARIES_EBU3213_E',

    # Environment SubSystem
    'Environment', 'EnvironmentData',

    'EnvironmentPolicy',

    'EnvironmentPolicyAPI',
    'register_policy', 'has_policy',
    'register_on_destroy', 'unregister_on_destroy',

    'get_current_environment',

    'VideoOutputTuple',
    'clear_output', 'clear_outputs', 'get_outputs', 'get_output',

    # Logging
    'LogHandle', 'Error',

    # Functions
    'FuncData', 'Func', 'FramePtr',
    'Plugin', 'Function',

    # Formats
    'VideoFormat', 'ChannelLayout',

    # Frames
    'RawFrame', 'VideoFrame', 'AudioFrame',
    'FrameProps',

    # Nodes
    'RawNode', 'VideoNode', 'AudioNode',

    'Core', '_CoreProxy', 'core',

    # Inspection API [UNSTABLE API]
    # '_try_enable_introspection'
]


###
# Typing

_T = TypeVar('_T')
_S = TypeVar('_S')

_SingleAndSequence = _T | Sequence[_T]


@runtime_checkable
class _SupportsString(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        ...


_DataType = str | bytes | bytearray | _SupportsString

_VapourSynthMapValue = Union[
    _SingleAndSequence[int],
    _SingleAndSequence[float],
    _SingleAndSequence[_DataType],
    _SingleAndSequence['VideoNode'],
    _SingleAndSequence['VideoFrame'],
    _SingleAndSequence['AudioNode'],
    _SingleAndSequence['AudioFrame'],
    _SingleAndSequence['_VSMapValueCallback[Any]']
]

_BoundVSMapValue = TypeVar('_BoundVSMapValue', bound=_VapourSynthMapValue)

_VSMapValueCallback = Callable[..., _BoundVSMapValue]


class _Future(Generic[_T]):
    def set_result(self, value: _T) -> None: ...

    def set_exception(self, exception: BaseException) -> None: ...

    def result(self) -> _T: ...

    def exception(self) -> NoReturn | None: ...

###
# Typed dicts


class _VideoFormatInfo(TypedDict):
    id: int
    name: str
    color_family: 'ColorFamily'
    sample_type: 'SampleType'
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int


###
# VapourSynth Versioning


class VapourSynthVersion(NamedTuple):
    release_major: int
    release_minor: int


class VapourSynthAPIVersion(NamedTuple):
    api_major: int
    api_minor: int


__version__: VapourSynthVersion
__api_version__: VapourSynthAPIVersion


###
# Plugin Versioning


class PluginVersion(NamedTuple):
    major: int
    minor: int


###
# VapourSynth Enums and Constants


class MessageType(IntFlag):
    MESSAGE_TYPE_DEBUG = cast(MessageType, ...)
    MESSAGE_TYPE_INFORMATION = cast(MessageType, ...)
    MESSAGE_TYPE_WARNING = cast(MessageType, ...)
    MESSAGE_TYPE_CRITICAL = cast(MessageType, ...)
    MESSAGE_TYPE_FATAL = cast(MessageType, ...)


MESSAGE_TYPE_DEBUG: Literal[MessageType.MESSAGE_TYPE_DEBUG]
MESSAGE_TYPE_INFORMATION: Literal[MessageType.MESSAGE_TYPE_INFORMATION]
MESSAGE_TYPE_WARNING: Literal[MessageType.MESSAGE_TYPE_WARNING]
MESSAGE_TYPE_CRITICAL: Literal[MessageType.MESSAGE_TYPE_CRITICAL]
MESSAGE_TYPE_FATAL: Literal[MessageType.MESSAGE_TYPE_FATAL]


class FilterMode(IntEnum):
    PARALLEL = cast(FilterMode, ...)
    PARALLEL_REQUESTS = cast(FilterMode, ...)
    UNORDERED = cast(FilterMode, ...)
    FRAME_STATE = cast(FilterMode, ...)


PARALLEL: Literal[FilterMode.PARALLEL]
PARALLEL_REQUESTS: Literal[FilterMode.PARALLEL_REQUESTS]
UNORDERED: Literal[FilterMode.UNORDERED]
FRAME_STATE: Literal[FilterMode.FRAME_STATE]


class CoreCreationFlags(IntFlag):
    ENABLE_GRAPH_INSPECTION = cast(CoreCreationFlags, ...)
    DISABLE_AUTO_LOADING = cast(CoreCreationFlags, ...)
    DISABLE_LIBRARY_UNLOADING = cast(CoreCreationFlags, ...)


ENABLE_GRAPH_INSPECTION: Literal[CoreCreationFlags.ENABLE_GRAPH_INSPECTION]
DISABLE_AUTO_LOADING: Literal[CoreCreationFlags.DISABLE_AUTO_LOADING]
DISABLE_LIBRARY_UNLOADING: Literal[CoreCreationFlags.DISABLE_LIBRARY_UNLOADING]


class MediaType(IntEnum):
    VIDEO = cast(MediaType, ...)
    AUDIO = cast(MediaType, ...)


VIDEO: Literal[MediaType.VIDEO]
AUDIO: Literal[MediaType.AUDIO]


class ColorFamily(IntEnum):
    UNDEFINED = cast(ColorFamily, ...)
    GRAY = cast(ColorFamily, ...)
    RGB = cast(ColorFamily, ...)
    YUV = cast(ColorFamily, ...)


UNDEFINED: Literal[ColorFamily.UNDEFINED]
GRAY: Literal[ColorFamily.GRAY]
RGB: Literal[ColorFamily.RGB]
YUV: Literal[ColorFamily.YUV]


class ColorRange(IntEnum):
    RANGE_FULL = cast(ColorRange, ...)
    RANGE_LIMITED = cast(ColorRange, ...)


RANGE_FULL: Literal[ColorRange.RANGE_FULL]
RANGE_LIMITED: Literal[ColorRange.RANGE_LIMITED]


class SampleType(IntEnum):
    INTEGER = cast(SampleType, ...)
    FLOAT = cast(SampleType, ...)


INTEGER: Literal[SampleType.INTEGER]
FLOAT: Literal[SampleType.FLOAT]


class PresetVideoFormat(IntEnum):
    NONE = cast(PresetVideoFormat, ...)

    GRAY8 = cast(PresetVideoFormat, ...)
    GRAY9 = cast(PresetVideoFormat, ...)
    GRAY10 = cast(PresetVideoFormat, ...)
    GRAY12 = cast(PresetVideoFormat, ...)
    GRAY14 = cast(PresetVideoFormat, ...)
    GRAY16 = cast(PresetVideoFormat, ...)
    GRAY32 = cast(PresetVideoFormat, ...)

    GRAYH = cast(PresetVideoFormat, ...)
    GRAYS = cast(PresetVideoFormat, ...)

    YUV420P8 = cast(PresetVideoFormat, ...)
    YUV422P8 = cast(PresetVideoFormat, ...)
    YUV444P8 = cast(PresetVideoFormat, ...)
    YUV410P8 = cast(PresetVideoFormat, ...)
    YUV411P8 = cast(PresetVideoFormat, ...)
    YUV440P8 = cast(PresetVideoFormat, ...)

    YUV420P9 = cast(PresetVideoFormat, ...)
    YUV422P9 = cast(PresetVideoFormat, ...)
    YUV444P9 = cast(PresetVideoFormat, ...)

    YUV420P10 = cast(PresetVideoFormat, ...)
    YUV422P10 = cast(PresetVideoFormat, ...)
    YUV444P10 = cast(PresetVideoFormat, ...)

    YUV420P12 = cast(PresetVideoFormat, ...)
    YUV422P12 = cast(PresetVideoFormat, ...)
    YUV444P12 = cast(PresetVideoFormat, ...)

    YUV420P14 = cast(PresetVideoFormat, ...)
    YUV422P14 = cast(PresetVideoFormat, ...)
    YUV444P14 = cast(PresetVideoFormat, ...)

    YUV420P16 = cast(PresetVideoFormat, ...)
    YUV422P16 = cast(PresetVideoFormat, ...)
    YUV444P16 = cast(PresetVideoFormat, ...)

    YUV420PH = cast(PresetVideoFormat, ...)
    YUV420PS = cast(PresetVideoFormat, ...)

    YUV422PH = cast(PresetVideoFormat, ...)
    YUV422PS = cast(PresetVideoFormat, ...)

    YUV444PH = cast(PresetVideoFormat, ...)
    YUV444PS = cast(PresetVideoFormat, ...)

    RGB24 = cast(PresetVideoFormat, ...)
    RGB27 = cast(PresetVideoFormat, ...)
    RGB30 = cast(PresetVideoFormat, ...)
    RGB36 = cast(PresetVideoFormat, ...)
    RGB42 = cast(PresetVideoFormat, ...)
    RGB48 = cast(PresetVideoFormat, ...)

    RGBH = cast(PresetVideoFormat, ...)
    RGBS = cast(PresetVideoFormat, ...)


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


class AudioChannels(IntEnum):
    FRONT_LEFT = cast(AudioChannels, ...)
    FRONT_RIGHT = cast(AudioChannels, ...)
    FRONT_CENTER = cast(AudioChannels, ...)
    LOW_FREQUENCY = cast(AudioChannels, ...)
    BACK_LEFT = cast(AudioChannels, ...)
    BACK_RIGHT = cast(AudioChannels, ...)
    FRONT_LEFT_OF_CENTER = cast(AudioChannels, ...)
    FRONT_RIGHT_OF_CENTER = cast(AudioChannels, ...)
    BACK_CENTER = cast(AudioChannels, ...)
    SIDE_LEFT = cast(AudioChannels, ...)
    SIDE_RIGHT = cast(AudioChannels, ...)
    TOP_CENTER = cast(AudioChannels, ...)
    TOP_FRONT_LEFT = cast(AudioChannels, ...)
    TOP_FRONT_CENTER = cast(AudioChannels, ...)
    TOP_FRONT_RIGHT = cast(AudioChannels, ...)
    TOP_BACK_LEFT = cast(AudioChannels, ...)
    TOP_BACK_CENTER = cast(AudioChannels, ...)
    TOP_BACK_RIGHT = cast(AudioChannels, ...)
    STEREO_LEFT = cast(AudioChannels, ...)
    STEREO_RIGHT = cast(AudioChannels, ...)
    WIDE_LEFT = cast(AudioChannels, ...)
    WIDE_RIGHT = cast(AudioChannels, ...)
    SURROUND_DIRECT_LEFT = cast(AudioChannels, ...)
    SURROUND_DIRECT_RIGHT = cast(AudioChannels, ...)
    LOW_FREQUENCY2 = cast(AudioChannels, ...)


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


class ChromaLocation(IntEnum):
    CHROMA_LEFT = cast(ChromaLocation, ...)
    CHROMA_CENTER = cast(ChromaLocation, ...)
    CHROMA_TOP_LEFT = cast(ChromaLocation, ...)
    CHROMA_TOP = cast(ChromaLocation, ...)
    CHROMA_BOTTOM_LEFT = cast(ChromaLocation, ...)
    CHROMA_BOTTOM = cast(ChromaLocation, ...)


CHROMA_LEFT: Literal[ChromaLocation.CHROMA_LEFT]
CHROMA_CENTER: Literal[ChromaLocation.CHROMA_CENTER]
CHROMA_TOP_LEFT: Literal[ChromaLocation.CHROMA_TOP_LEFT]
CHROMA_TOP: Literal[ChromaLocation.CHROMA_TOP]
CHROMA_BOTTOM_LEFT: Literal[ChromaLocation.CHROMA_BOTTOM_LEFT]
CHROMA_BOTTOM: Literal[ChromaLocation.CHROMA_BOTTOM]


class FieldBased(IntEnum):
    FIELD_PROGRESSIVE = cast(FieldBased, ...)
    FIELD_TOP = cast(FieldBased, ...)
    FIELD_BOTTOM = cast(FieldBased, ...)


FIELD_PROGRESSIVE: Literal[FieldBased.FIELD_PROGRESSIVE]
FIELD_TOP: Literal[FieldBased.FIELD_TOP]
FIELD_BOTTOM: Literal[FieldBased.FIELD_BOTTOM]


class MatrixCoefficients(IntEnum):
    MATRIX_RGB = cast(MatrixCoefficients, ...)
    MATRIX_BT709 = cast(MatrixCoefficients, ...)
    MATRIX_UNSPECIFIED = cast(MatrixCoefficients, ...)
    MATRIX_FCC = cast(MatrixCoefficients, ...)
    MATRIX_BT470_BG = cast(MatrixCoefficients, ...)
    MATRIX_ST170_M = cast(MatrixCoefficients, ...)
    MATRIX_ST240_M = cast(MatrixCoefficients, ...)
    MATRIX_YCGCO = cast(MatrixCoefficients, ...)
    MATRIX_BT2020_NCL = cast(MatrixCoefficients, ...)
    MATRIX_BT2020_CL = cast(MatrixCoefficients, ...)
    MATRIX_CHROMATICITY_DERIVED_NCL = cast(MatrixCoefficients, ...)
    MATRIX_CHROMATICITY_DERIVED_CL = cast(MatrixCoefficients, ...)
    MATRIX_ICTCP = cast(MatrixCoefficients, ...)


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
    TRANSFER_BT709 = cast(TransferCharacteristics, ...)
    TRANSFER_UNSPECIFIED = cast(TransferCharacteristics, ...)
    TRANSFER_BT470_M = cast(TransferCharacteristics, ...)
    TRANSFER_BT470_BG = cast(TransferCharacteristics, ...)
    TRANSFER_BT601 = cast(TransferCharacteristics, ...)
    TRANSFER_ST240_M = cast(TransferCharacteristics, ...)
    TRANSFER_LINEAR = cast(TransferCharacteristics, ...)
    TRANSFER_LOG_100 = cast(TransferCharacteristics, ...)
    TRANSFER_LOG_316 = cast(TransferCharacteristics, ...)
    TRANSFER_IEC_61966_2_4 = cast(TransferCharacteristics, ...)
    TRANSFER_IEC_61966_2_1 = cast(TransferCharacteristics, ...)
    TRANSFER_BT2020_10 = cast(TransferCharacteristics, ...)
    TRANSFER_BT2020_12 = cast(TransferCharacteristics, ...)
    TRANSFER_ST2084 = cast(TransferCharacteristics, ...)
    TRANSFER_ST428 = cast(TransferCharacteristics, ...)
    TRANSFER_ARIB_B67 = cast(TransferCharacteristics, ...)


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
    PRIMARIES_BT709 = cast(ColorPrimaries, ...)
    PRIMARIES_UNSPECIFIED = cast(ColorPrimaries, ...)
    PRIMARIES_BT470_M = cast(ColorPrimaries, ...)
    PRIMARIES_BT470_BG = cast(ColorPrimaries, ...)
    PRIMARIES_ST170_M = cast(ColorPrimaries, ...)
    PRIMARIES_ST240_M = cast(ColorPrimaries, ...)
    PRIMARIES_FILM = cast(ColorPrimaries, ...)
    PRIMARIES_BT2020 = cast(ColorPrimaries, ...)
    PRIMARIES_ST428 = cast(ColorPrimaries, ...)
    PRIMARIES_ST431_2 = cast(ColorPrimaries, ...)
    PRIMARIES_ST432_1 = cast(ColorPrimaries, ...)
    PRIMARIES_EBU3213_E = cast(ColorPrimaries, ...)


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


###
# VapourSynth Environment SubSystem


class EnvironmentData:
    def __init__(self) -> NoReturn: ...


class EnvironmentPolicy:
    def on_policy_registered(self, special_api: 'EnvironmentPolicyAPI') -> None: ...

    def on_policy_cleared(self) -> None: ...

    @abstractmethod
    def get_current_environment(self) -> EnvironmentData | None: ...

    @abstractmethod
    def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData | None: ...

    def is_alive(self, environment: EnvironmentData) -> bool: ...


class EnvironmentPolicyAPI:
    def __init__(self) -> NoReturn: ...

    def wrap_environment(self, environment_data: EnvironmentData) -> 'Environment': ...

    def create_environment(self, flags: int = 0) -> EnvironmentData: ...

    def set_logger(self, env: EnvironmentData, logger: Callable[[int, str], None]) -> None: ...

    def get_vapoursynth_api(self, version: int) -> c_void_p: ...

    def get_core_ptr(self, environment_data: EnvironmentData) -> c_void_p: ...

    def destroy_environment(self, env: EnvironmentData) -> None: ...

    def unregister_policy(self) -> None: ...


def register_policy(policy: EnvironmentPolicy) -> None:
    ...


if not TYPE_CHECKING:
    def _try_enable_introspection(version: int = None): ...


def has_policy() -> bool:
    ...


def register_on_destroy(callback: Callable[..., None]) -> None:
    ...


def unregister_on_destroy(callback: Callable[..., None]) -> None:
    ...


class Environment:
    env: ReferenceType[EnvironmentData]

    def __init__(self) -> NoReturn: ...

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

    def copy(self) -> 'Environment': ...

    def use(self) -> ContextManager[None]: ...

    def __eq__(self, other: 'Environment') -> bool: ...  # type: ignore[override]

    def __repr__(self) -> str: ...


def get_current_environment() -> Environment:
    ...


class Local:
    def __getattr__(self, key: str) -> Any: ...
    
    # Even though object does have set/del methods, typecheckers will treat them differently
    # when they are not explicit; for example by raising a member not found warning.

    def __setattr__(self, key: str, value: Any) -> None: ...
    
    def __delattr__(self, key: str) -> None: ...


class VideoOutputTuple(NamedTuple):
    clip: 'VideoNode'
    alpha: 'VideoNode' | None
    alt_output: Literal[0, 1, 2]


class Error(Exception):
    ...


def clear_output(index: int = 0) -> None:
    ...


def clear_outputs() -> None:
    ...


def get_outputs() -> MappingProxyType[int, VideoOutputTuple | 'AudioNode']:
    ...


def get_output(index: int = 0) -> VideoOutputTuple | 'AudioNode':
    ...


class FuncData:
    def __init__(self) -> NoReturn: ...

    def __call__(self, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...


class Func:
    def __init__(self) -> NoReturn: ...

    def __call__(self, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...


class FramePtr:
    def __init__(self) -> NoReturn: ...


class VideoFormat:
    id: int
    name: str
    color_family: ColorFamily
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int

    def __init__(self) -> NoReturn: ...

    def _as_dict(self) -> _VideoFormatInfo: ...

    def replace(
        self, *,
        color_family: ColorFamily | None = None,
        sample_type: SampleType | None = None,
        bits_per_sample: int | None = None,
        subsampling_w: int | None = None,
        subsampling_h: int | None = None
    ) -> 'VideoFormat': ...

    @overload
    def __eq__(self, other: 'VideoFormat') -> bool: ...

    @overload
    def __eq__(self, other: Any) -> Literal[False]: ...


class FrameProps(MutableMapping[str, _VapourSynthMapValue]):
    def __init__(self) -> NoReturn: ...

    def setdefault(
        self, key: str, default: _VapourSynthMapValue = 0
    ) -> _VapourSynthMapValue: ...

    def copy(self) -> MutableMapping[str, _VapourSynthMapValue]: ...

    # Since we're inheriting from the MutableMapping abstract class,
    # we *have* to specify that we have indeed created these methods.
    # If we don't, mypy will complain that we're working with abstract methods.

    def __setattr__(self, name: str, value: _VapourSynthMapValue) -> None: ...

    def __getattr__(self, name: str) -> _VapourSynthMapValue: ...

    def __delattr__(self, name: str) -> None: ...

    def __setitem__(self, name: str, value: _VapourSynthMapValue) -> None: ...

    def __getitem__(self, name: str) -> _VapourSynthMapValue: ...

    def __delitem__(self, name: str) -> None: ...

    def __iter__(self) -> Iterator[str]: ...

    def __len__(self) -> int: ...


class ChannelLayout(int):
    def __init__(self) -> NoReturn: ...

    def __contains__(self, layout: AudioChannels) -> bool: ...

    def __iter__(self) -> Iterator[AudioChannels]: ...

    @overload
    def __eq__(self, other: 'ChannelLayout') -> bool: ...

    @overload
    def __eq__(self, other: Any) -> Literal[False]: ...

    def __len__(self) -> int: ...


class audio_view(memoryview):  # type: ignore[misc]
    @property
    def shape(self) -> tuple[int]: ...

    @property
    def strides(self) -> tuple[int]: ...

    @property
    def ndim(self) -> Literal[1]: ...

    @property
    def obj(self) -> FramePtr: ...  # type: ignore[override]

    def __getitem__(self, index: int) -> int | float: ...  # type: ignore[override]

    def __setitem__(self, index: int, other: int | float) -> None: ...  # type: ignore[override]

    def tolist(self) -> list[int | float]: ...  # type: ignore[override]


class video_view(memoryview):  # type: ignore[misc]
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def strides(self) -> tuple[int, int]: ...

    @property
    def ndim(self) -> Literal[2]: ...

    @property
    def obj(self) -> FramePtr: ...  # type: ignore[override]

    def __getitem__(self, index: Tuple[int, int]) -> int | float: ...  # type: ignore[override]

    def __setitem__(self, index: Tuple[int, int], other: int | float) -> None: ...  # type: ignore[override]

    def tolist(self) -> list[int | float]: ...  # type: ignore[override]


class RawFrame:
    def __init__(self) -> None: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...

    def copy(self: 'SelfFrame') -> 'SelfFrame': ...

    @property
    def props(self) -> FrameProps: ...

    @props.setter
    def props(self, new_props: MappingProxyType[str, _VapourSynthMapValue]) -> None: ...

    def get_write_ptr(self, plane: int) -> c_void_p: ...

    def get_read_ptr(self, plane: int) -> c_void_p: ...

    def get_stride(self, plane: int) -> int: ...

    @property
    def readonly(self) -> bool: ...

    def __enter__(self: 'SelfFrame') -> 'SelfFrame': ...

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None, /,
    ) -> bool | None: ...

    def __getitem__(self, index: int) -> memoryview: ...

    def __len__(self) -> int: ...


SelfFrame = TypeVar('SelfFrame', bound=RawFrame)


class VideoFrame(RawFrame):
    format: VideoFormat
    width: int
    height: int

    def readchunks(self) -> Iterator[video_view]: ...

    def __getitem__(self, index: int) -> video_view: ...


class AudioFrame(RawFrame):
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    channel_layout: int
    num_channels: int

    @property
    def channels(self) -> ChannelLayout: ...

    def __getitem__(self, index: int) -> audio_view: ...

    
# implementation: akarin

class _Plugin_akarin_Core_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cambi(self, clip: VideoNode, window_size: int | None = None, topk: float | None = None, tvi_threshold: float | None = None, scores: int | None = None, scaling: float | None = None) -> _ConstantFormatVideoNode: ...
    def DLISR(self, clip: VideoNode, scale: int | None = None, device_id: int | None = None) -> _ConstantFormatVideoNode: ...
    def DLVFX(self, clip: VideoNode, op: int, scale: float | None = None, strength: float | None = None, output_depth: int | None = None, num_streams: int | None = None, model_dir: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Expr(self, clips: _SingleAndSequence[VideoNode], expr: _SingleAndSequence[_DataType], format: int | None = None, opt: int | None = None, boundary: int | None = None) -> _ConstantFormatVideoNode: ...
    def ExprTest(self, clips: _SingleAndSequence[float], expr: _DataType, props: _VSMapValueCallback[_VapourSynthMapValue] | None = None, ref: VideoNode | None = None, vars: int | None = None) -> _ConstantFormatVideoNode: ...
    def PickFrames(self, clip: VideoNode, indices: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def PropExpr(self, clips: _SingleAndSequence[VideoNode], dict: _VSMapValueCallback[_VapourSynthMapValue]) -> _ConstantFormatVideoNode: ...
    def Select(self, clip_src: _SingleAndSequence[VideoNode], prop_src: _SingleAndSequence[VideoNode], expr: _SingleAndSequence[_DataType]) -> _ConstantFormatVideoNode: ...
    def Text(self, clips: _SingleAndSequence[VideoNode], text: _DataType, alignment: int | None = None, scale: int | None = None, prop: _DataType | None = None, strict: int | None = None, vspipe: int | None = None) -> _ConstantFormatVideoNode: ...
    def Tmpl(self, clips: _SingleAndSequence[VideoNode], prop: _SingleAndSequence[_DataType], text: _SingleAndSequence[_DataType]) -> _ConstantFormatVideoNode: ...
    def Version(self) -> _ConstantFormatVideoNode: ...

class _Plugin_akarin_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cambi(self, window_size: int | None = None, topk: float | None = None, tvi_threshold: float | None = None, scores: int | None = None, scaling: float | None = None) -> _ConstantFormatVideoNode: ...
    def DLISR(self, scale: int | None = None, device_id: int | None = None) -> _ConstantFormatVideoNode: ...
    def DLVFX(self, op: int, scale: float | None = None, strength: float | None = None, output_depth: int | None = None, num_streams: int | None = None, model_dir: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Expr(self, expr: _SingleAndSequence[_DataType], format: int | None = None, opt: int | None = None, boundary: int | None = None) -> _ConstantFormatVideoNode: ...
    def PickFrames(self, indices: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def PropExpr(self, dict: _VSMapValueCallback[_VapourSynthMapValue]) -> _ConstantFormatVideoNode: ...
    def Select(self, prop_src: _SingleAndSequence[VideoNode], expr: _SingleAndSequence[_DataType]) -> _ConstantFormatVideoNode: ...
    def Text(self, text: _DataType, alignment: int | None = None, scale: int | None = None, prop: _DataType | None = None, strict: int | None = None, vspipe: int | None = None) -> _ConstantFormatVideoNode: ...
    def Tmpl(self, prop: _SingleAndSequence[_DataType], text: _SingleAndSequence[_DataType]) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: bs

_ReturnDict_bs_TrackInfo = TypedDict("_ReturnDict_bs_TrackInfo", {"mediatype": int, "mediatypestr": _DataType, "codec": int, "codecstr": _DataType, "disposition": int, "dispositionstr": _DataType})


class _Plugin_bs_Core_Bound(Plugin):
    """This class implements the module definitions for the "bs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AudioSource(self, source: _DataType, track: int | None = None, adjustdelay: int | None = None, threads: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None, drc_scale: float | None = None, cachemode: int | None = None, cachepath: _DataType | None = None, cachesize: int | None = None, showprogress: int | None = None) -> _ConstantFormatVideoNode: ...
    def Metadata(self, source: _DataType, track: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None) -> _ConstantFormatVideoNode: ...
    def SetDebugOutput(self, enable: int) -> _ConstantFormatVideoNode: ...
    def SetFFmpegLogLevel(self, level: int) -> _ConstantFormatVideoNode: ...
    def TrackInfo(self, source: _DataType, enable_drefs: int | None = None, use_absolute_path: int | None = None) -> _ConstantFormatVideoNode: ...
    def VideoSource(self, source: _DataType, track: int | None = None, variableformat: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, rff: int | None = None, threads: int | None = None, seekpreroll: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None, cachemode: int | None = None, cachepath: _DataType | None = None, cachesize: int | None = None, hwdevice: _DataType | None = None, extrahwframes: int | None = None, timecodes: _DataType | None = None, start_number: int | None = None, viewid: int | None = None, showprogress: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: cs

class _Plugin_cs_Core_Bound(Plugin):
    """This class implements the module definitions for the "cs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ConvertColor(self, clip: VideoNode, output_profile: _DataType, input_profile: _DataType | None = None, float_output: int | None = None) -> _ConstantFormatVideoNode: ...
    def ImageSource(self, source: _DataType, subsampling_pad: int | None = None, jpeg_rgb: int | None = None, jpeg_fancy_upsampling: int | None = None, jpeg_cmyk_profile: _DataType | None = None, jpeg_cmyk_target_profile: _DataType | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_cs_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "cs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ConvertColor(self, output_profile: _DataType, input_profile: _DataType | None = None, float_output: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: d2v

class _Plugin_d2v_Core_Bound(Plugin):
    """This class implements the module definitions for the "d2v" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Source(self, input: _DataType, threads: int | None = None, nocrop: int | None = None, rff: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: descale

class _Plugin_descale_Core_Bound(Plugin):
    """This class implements the module definitions for the "descale" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, src: VideoNode, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Bilinear(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Debicubic(self, src: VideoNode, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Debilinear(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Decustom(self, src: VideoNode, width: int, height: int, custom_kernel: _VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Delanczos(self, src: VideoNode, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Despline16(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Despline36(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Despline64(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lanczos(self, src: VideoNode, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def ScaleCustom(self, src: VideoNode, width: int, height: int, custom_kernel: _VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Spline16(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Spline36(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Spline64(self, src: VideoNode, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_descale_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "descale" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Bilinear(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Debicubic(self, width: int, height: int, b: float | None = None, c: float | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Debilinear(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Decustom(self, width: int, height: int, custom_kernel: _VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Delanczos(self, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Despline16(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Despline36(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Despline64(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lanczos(self, width: int, height: int, taps: int | None = None, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def ScaleCustom(self, width: int, height: int, custom_kernel: _VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Spline16(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Spline36(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...
    def Spline64(self, width: int, height: int, blur: float | None = None, post_conv: _SingleAndSequence[float] | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, border_handling: int | None = None, ignore_mask: VideoNode | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None, opt: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: dgdecodenv

class _Plugin_dgdecodenv_Core_Bound(Plugin):
    """This class implements the module definitions for the "dgdecodenv" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DGSource(self, source: _DataType, i420: int | None = None, deinterlace: int | None = None, use_top_field: int | None = None, use_pf: int | None = None, ct: int | None = None, cb: int | None = None, cl: int | None = None, cr: int | None = None, rw: int | None = None, rh: int | None = None, fieldop: int | None = None, show: int | None = None, show2: _DataType | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: dvdsrc2

class _Plugin_dvdsrc2_Core_Bound(Plugin):
    """This class implements the module definitions for the "dvdsrc2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def FullVts(self, path: _DataType, vts: int, ranges: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def FullVtsAc3(self, path: _DataType, vts: int, audio: int, ranges: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def FullVtsLpcm(self, path: _DataType, vts: int, audio: int, ranges: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Ifo(self, path: _DataType, ifo: int) -> _ConstantFormatVideoNode: ...
    def RawAc3(self, path: _DataType, vts: int, audio: int, ranges: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: ffms2

class _Plugin_ffms2_Core_Bound(Plugin):
    """This class implements the module definitions for the "ffms2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def GetLogLevel(self) -> _ConstantFormatVideoNode: ...
    def Index(self, source: _DataType, cachefile: _DataType | None = None, indextracks: _SingleAndSequence[int] | None = None, errorhandling: int | None = None, overwrite: int | None = None, enable_drefs: int | None = None, use_absolute_path: int | None = None) -> _ConstantFormatVideoNode: ...
    def SetLogLevel(self, level: int) -> _ConstantFormatVideoNode: ...
    def Source(self, source: _DataType, track: int | None = None, cache: int | None = None, cachefile: _DataType | None = None, fpsnum: int | None = None, fpsden: int | None = None, threads: int | None = None, timecodes: _DataType | None = None, seekmode: int | None = None, width: int | None = None, height: int | None = None, resizer: _DataType | None = None, format: int | None = None, alpha: int | None = None) -> _ConstantFormatVideoNode: ...
    def Version(self) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: fmtc

class _Plugin_fmtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "fmtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def bitdepth(self, clip: VideoNode, csp: int | None = None, bits: int | None = None, flt: int | None = None, planes: _SingleAndSequence[int] | None = None, fulls: int | None = None, fulld: int | None = None, dmode: int | None = None, ampo: float | None = None, ampn: float | None = None, dyn: int | None = None, staticnoise: int | None = None, cpuopt: int | None = None, patsize: int | None = None, tpdfo: int | None = None, tpdfn: int | None = None, corplane: int | None = None) -> _ConstantFormatVideoNode: ...
    def histluma(self, clip: VideoNode, full: int | None = None, amp: int | None = None) -> _ConstantFormatVideoNode: ...
    def matrix(self, clip: VideoNode, mat: _DataType | None = None, mats: _DataType | None = None, matd: _DataType | None = None, fulls: int | None = None, fulld: int | None = None, coef: _SingleAndSequence[float] | None = None, csp: int | None = None, col_fam: int | None = None, bits: int | None = None, singleout: int | None = None, cpuopt: int | None = None, planes: _SingleAndSequence[float] | None = None) -> _ConstantFormatVideoNode: ...
    def matrix2020cl(self, clip: VideoNode, full: int | None = None, csp: int | None = None, bits: int | None = None, cpuopt: int | None = None) -> _ConstantFormatVideoNode: ...
    def nativetostack16(self, clip: VideoNode) -> _ConstantFormatVideoNode: ...
    def primaries(self, clip: VideoNode, rs: _SingleAndSequence[float] | None = None, gs: _SingleAndSequence[float] | None = None, bs: _SingleAndSequence[float] | None = None, ws: _SingleAndSequence[float] | None = None, rd: _SingleAndSequence[float] | None = None, gd: _SingleAndSequence[float] | None = None, bd: _SingleAndSequence[float] | None = None, wd: _SingleAndSequence[float] | None = None, prims: _DataType | None = None, primd: _DataType | None = None, wconv: int | None = None, cpuopt: int | None = None) -> _ConstantFormatVideoNode: ...
    def resample(self, clip: VideoNode, w: int | None = None, h: int | None = None, sx: _SingleAndSequence[float] | None = None, sy: _SingleAndSequence[float] | None = None, sw: _SingleAndSequence[float] | None = None, sh: _SingleAndSequence[float] | None = None, scale: float | None = None, scaleh: float | None = None, scalev: float | None = None, kernel: _SingleAndSequence[_DataType] | None = None, kernelh: _SingleAndSequence[_DataType] | None = None, kernelv: _SingleAndSequence[_DataType] | None = None, impulse: _SingleAndSequence[float] | None = None, impulseh: _SingleAndSequence[float] | None = None, impulsev: _SingleAndSequence[float] | None = None, taps: _SingleAndSequence[int] | None = None, tapsh: _SingleAndSequence[int] | None = None, tapsv: _SingleAndSequence[int] | None = None, a1: _SingleAndSequence[float] | None = None, a2: _SingleAndSequence[float] | None = None, a3: _SingleAndSequence[float] | None = None, a1h: _SingleAndSequence[float] | None = None, a2h: _SingleAndSequence[float] | None = None, a3h: _SingleAndSequence[float] | None = None, a1v: _SingleAndSequence[float] | None = None, a2v: _SingleAndSequence[float] | None = None, a3v: _SingleAndSequence[float] | None = None, kovrspl: _SingleAndSequence[int] | None = None, fh: _SingleAndSequence[float] | None = None, fv: _SingleAndSequence[float] | None = None, cnorm: _SingleAndSequence[int] | None = None, total: _SingleAndSequence[float] | None = None, totalh: _SingleAndSequence[float] | None = None, totalv: _SingleAndSequence[float] | None = None, invks: _SingleAndSequence[int] | None = None, invksh: _SingleAndSequence[int] | None = None, invksv: _SingleAndSequence[int] | None = None, invkstaps: _SingleAndSequence[int] | None = None, invkstapsh: _SingleAndSequence[int] | None = None, invkstapsv: _SingleAndSequence[int] | None = None, csp: int | None = None, css: _DataType | None = None, planes: _SingleAndSequence[float] | None = None, fulls: int | None = None, fulld: int | None = None, center: _SingleAndSequence[int] | None = None, cplace: _DataType | None = None, cplaces: _DataType | None = None, cplaced: _DataType | None = None, interlaced: int | None = None, interlacedd: int | None = None, tff: int | None = None, tffd: int | None = None, flt: int | None = None, cpuopt: int | None = None) -> _ConstantFormatVideoNode: ...
    def stack16tonative(self, clip: VideoNode) -> _ConstantFormatVideoNode: ...
    def transfer(self, clip: VideoNode, transs: _SingleAndSequence[_DataType] | None = None, transd: _SingleAndSequence[_DataType] | None = None, cont: float | None = None, gcor: float | None = None, bits: int | None = None, flt: int | None = None, fulls: int | None = None, fulld: int | None = None, logceis: int | None = None, logceid: int | None = None, cpuopt: int | None = None, blacklvl: float | None = None, sceneref: int | None = None, lb: float | None = None, lw: float | None = None, lws: float | None = None, lwd: float | None = None, ambient: float | None = None, match: int | None = None, gy: int | None = None, debug: int | None = None, sig_c: float | None = None, sig_t: float | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_fmtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "fmtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def bitdepth(self, csp: int | None = None, bits: int | None = None, flt: int | None = None, planes: _SingleAndSequence[int] | None = None, fulls: int | None = None, fulld: int | None = None, dmode: int | None = None, ampo: float | None = None, ampn: float | None = None, dyn: int | None = None, staticnoise: int | None = None, cpuopt: int | None = None, patsize: int | None = None, tpdfo: int | None = None, tpdfn: int | None = None, corplane: int | None = None) -> _ConstantFormatVideoNode: ...
    def histluma(self, full: int | None = None, amp: int | None = None) -> _ConstantFormatVideoNode: ...
    def matrix(self, mat: _DataType | None = None, mats: _DataType | None = None, matd: _DataType | None = None, fulls: int | None = None, fulld: int | None = None, coef: _SingleAndSequence[float] | None = None, csp: int | None = None, col_fam: int | None = None, bits: int | None = None, singleout: int | None = None, cpuopt: int | None = None, planes: _SingleAndSequence[float] | None = None) -> _ConstantFormatVideoNode: ...
    def matrix2020cl(self, full: int | None = None, csp: int | None = None, bits: int | None = None, cpuopt: int | None = None) -> _ConstantFormatVideoNode: ...
    def nativetostack16(self) -> _ConstantFormatVideoNode: ...
    def primaries(self, rs: _SingleAndSequence[float] | None = None, gs: _SingleAndSequence[float] | None = None, bs: _SingleAndSequence[float] | None = None, ws: _SingleAndSequence[float] | None = None, rd: _SingleAndSequence[float] | None = None, gd: _SingleAndSequence[float] | None = None, bd: _SingleAndSequence[float] | None = None, wd: _SingleAndSequence[float] | None = None, prims: _DataType | None = None, primd: _DataType | None = None, wconv: int | None = None, cpuopt: int | None = None) -> _ConstantFormatVideoNode: ...
    def resample(self, w: int | None = None, h: int | None = None, sx: _SingleAndSequence[float] | None = None, sy: _SingleAndSequence[float] | None = None, sw: _SingleAndSequence[float] | None = None, sh: _SingleAndSequence[float] | None = None, scale: float | None = None, scaleh: float | None = None, scalev: float | None = None, kernel: _SingleAndSequence[_DataType] | None = None, kernelh: _SingleAndSequence[_DataType] | None = None, kernelv: _SingleAndSequence[_DataType] | None = None, impulse: _SingleAndSequence[float] | None = None, impulseh: _SingleAndSequence[float] | None = None, impulsev: _SingleAndSequence[float] | None = None, taps: _SingleAndSequence[int] | None = None, tapsh: _SingleAndSequence[int] | None = None, tapsv: _SingleAndSequence[int] | None = None, a1: _SingleAndSequence[float] | None = None, a2: _SingleAndSequence[float] | None = None, a3: _SingleAndSequence[float] | None = None, a1h: _SingleAndSequence[float] | None = None, a2h: _SingleAndSequence[float] | None = None, a3h: _SingleAndSequence[float] | None = None, a1v: _SingleAndSequence[float] | None = None, a2v: _SingleAndSequence[float] | None = None, a3v: _SingleAndSequence[float] | None = None, kovrspl: _SingleAndSequence[int] | None = None, fh: _SingleAndSequence[float] | None = None, fv: _SingleAndSequence[float] | None = None, cnorm: _SingleAndSequence[int] | None = None, total: _SingleAndSequence[float] | None = None, totalh: _SingleAndSequence[float] | None = None, totalv: _SingleAndSequence[float] | None = None, invks: _SingleAndSequence[int] | None = None, invksh: _SingleAndSequence[int] | None = None, invksv: _SingleAndSequence[int] | None = None, invkstaps: _SingleAndSequence[int] | None = None, invkstapsh: _SingleAndSequence[int] | None = None, invkstapsv: _SingleAndSequence[int] | None = None, csp: int | None = None, css: _DataType | None = None, planes: _SingleAndSequence[float] | None = None, fulls: int | None = None, fulld: int | None = None, center: _SingleAndSequence[int] | None = None, cplace: _DataType | None = None, cplaces: _DataType | None = None, cplaced: _DataType | None = None, interlaced: int | None = None, interlacedd: int | None = None, tff: int | None = None, tffd: int | None = None, flt: int | None = None, cpuopt: int | None = None) -> _ConstantFormatVideoNode: ...
    def stack16tonative(self) -> _ConstantFormatVideoNode: ...
    def transfer(self, transs: _SingleAndSequence[_DataType] | None = None, transd: _SingleAndSequence[_DataType] | None = None, cont: float | None = None, gcor: float | None = None, bits: int | None = None, flt: int | None = None, fulls: int | None = None, fulld: int | None = None, logceis: int | None = None, logceid: int | None = None, cpuopt: int | None = None, blacklvl: float | None = None, sceneref: int | None = None, lb: float | None = None, lw: float | None = None, lws: float | None = None, lwd: float | None = None, ambient: float | None = None, match: int | None = None, gy: int | None = None, debug: int | None = None, sig_c: float | None = None, sig_t: float | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: imwri

class _Plugin_imwri_Core_Bound(Plugin):
    """This class implements the module definitions for the "imwri" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Read(self, filename: _SingleAndSequence[_DataType], firstnum: int | None = None, mismatch: int | None = None, alpha: int | None = None, float_output: int | None = None, embed_icc: int | None = None) -> _ConstantFormatVideoNode: ...
    def Write(self, clip: VideoNode, imgformat: _DataType, filename: _DataType, firstnum: int | None = None, quality: int | None = None, dither: int | None = None, compression_type: _DataType | None = None, overwrite: int | None = None, alpha: VideoNode | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_imwri_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "imwri" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Write(self, imgformat: _DataType, filename: _DataType, firstnum: int | None = None, quality: int | None = None, dither: int | None = None, compression_type: _DataType | None = None, overwrite: int | None = None, alpha: VideoNode | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: lsmas

class _Plugin_lsmas_Core_Bound(Plugin):
    """This class implements the module definitions for the "lsmas" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def LibavSMASHSource(self, source: _DataType, track: int | None = None, threads: int | None = None, seek_mode: int | None = None, seek_threshold: int | None = None, dr: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, variable: int | None = None, format: _DataType | None = None, decoder: _DataType | None = None, prefer_hw: int | None = None, ff_loglevel: int | None = None, ff_options: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def LWLibavSource(self, source: _DataType, stream_index: int | None = None, cache: int | None = None, cachefile: _DataType | None = None, threads: int | None = None, seek_mode: int | None = None, seek_threshold: int | None = None, dr: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, variable: int | None = None, format: _DataType | None = None, decoder: _DataType | None = None, prefer_hw: int | None = None, repeat: int | None = None, dominance: int | None = None, ff_loglevel: int | None = None, cachedir: _DataType | None = None, ff_options: _DataType | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: placebo

class _Plugin_placebo_Core_Bound(Plugin):
    """This class implements the module definitions for the "placebo" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(self, clip: VideoNode, planes: int | None = None, iterations: int | None = None, threshold: float | None = None, radius: float | None = None, grain: float | None = None, dither: int | None = None, dither_algo: int | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...
    def Resample(self, clip: VideoNode, width: int, height: int, filter: _DataType | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, src_width: float | None = None, src_height: float | None = None, sx: float | None = None, sy: float | None = None, antiring: float | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, linearize: int | None = None, trc: int | None = None, min_luma: float | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...
    def Shader(self, clip: VideoNode, shader: _DataType | None = None, width: int | None = None, height: int | None = None, chroma_loc: int | None = None, matrix: int | None = None, trc: int | None = None, linearize: int | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, antiring: float | None = None, filter: _DataType | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, shader_s: _DataType | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...
    def Tonemap(self, clip: VideoNode, src_csp: int | None = None, dst_csp: int | None = None, dst_prim: int | None = None, src_max: float | None = None, src_min: float | None = None, dst_max: float | None = None, dst_min: float | None = None, dynamic_peak_detection: int | None = None, smoothing_period: float | None = None, scene_threshold_low: float | None = None, scene_threshold_high: float | None = None, percentile: float | None = None, gamut_mapping: int | None = None, tone_mapping_function: int | None = None, tone_mapping_function_s: _DataType | None = None, tone_mapping_param: float | None = None, metadata: int | None = None, use_dovi: int | None = None, visualize_lut: int | None = None, show_clipping: int | None = None, contrast_recovery: float | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_placebo_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "placebo" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(self, planes: int | None = None, iterations: int | None = None, threshold: float | None = None, radius: float | None = None, grain: float | None = None, dither: int | None = None, dither_algo: int | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...
    def Resample(self, width: int, height: int, filter: _DataType | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, src_width: float | None = None, src_height: float | None = None, sx: float | None = None, sy: float | None = None, antiring: float | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, linearize: int | None = None, trc: int | None = None, min_luma: float | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...
    def Shader(self, shader: _DataType | None = None, width: int | None = None, height: int | None = None, chroma_loc: int | None = None, matrix: int | None = None, trc: int | None = None, linearize: int | None = None, sigmoidize: int | None = None, sigmoid_center: float | None = None, sigmoid_slope: float | None = None, antiring: float | None = None, filter: _DataType | None = None, clamp: float | None = None, blur: float | None = None, taper: float | None = None, radius: float | None = None, param1: float | None = None, param2: float | None = None, shader_s: _DataType | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...
    def Tonemap(self, src_csp: int | None = None, dst_csp: int | None = None, dst_prim: int | None = None, src_max: float | None = None, src_min: float | None = None, dst_max: float | None = None, dst_min: float | None = None, dynamic_peak_detection: int | None = None, smoothing_period: float | None = None, scene_threshold_low: float | None = None, scene_threshold_high: float | None = None, percentile: float | None = None, gamut_mapping: int | None = None, tone_mapping_function: int | None = None, tone_mapping_function_s: _DataType | None = None, tone_mapping_param: float | None = None, metadata: int | None = None, use_dovi: int | None = None, visualize_lut: int | None = None, show_clipping: int | None = None, contrast_recovery: float | None = None, log_level: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: resize

class _Plugin_resize_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Bilinear(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Bob(self, clip: VideoNode, filter: _DataType | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lanczos(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Point(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Spline16(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Spline36(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Spline64(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...

class _Plugin_resize_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Bilinear(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Bob(self, filter: _DataType | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lanczos(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Point(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Spline16(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Spline36(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...
    def Spline64(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, approximate_gamma: int | None = None) -> VideoNode: ...

# end implementation

    
# implementation: resize2

class _Plugin_resize2_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Bilinear(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Bob(self, clip: VideoNode, filter: _DataType | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Custom(self, clip: VideoNode, custom_kernel: _VSMapValueCallback[_VapourSynthMapValue], taps: int, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lanczos(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Point(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Spline16(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Spline36(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Spline64(self, clip: VideoNode, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...

class _Plugin_resize2_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Bilinear(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Bob(self, filter: _DataType | None = None, tff: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Custom(self, custom_kernel: _VSMapValueCallback[_VapourSynthMapValue], taps: int, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lanczos(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Point(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Spline16(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Spline36(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...
    def Spline64(self, width: int | None = None, height: int | None = None, format: int | None = None, matrix: int | None = None, matrix_s: _DataType | None = None, transfer: int | None = None, transfer_s: _DataType | None = None, primaries: int | None = None, primaries_s: _DataType | None = None, range: int | None = None, range_s: _DataType | None = None, chromaloc: int | None = None, chromaloc_s: _DataType | None = None, matrix_in: int | None = None, matrix_in_s: _DataType | None = None, transfer_in: int | None = None, transfer_in_s: _DataType | None = None, primaries_in: int | None = None, primaries_in_s: _DataType | None = None, range_in: int | None = None, range_in_s: _DataType | None = None, chromaloc_in: int | None = None, chromaloc_in_s: _DataType | None = None, filter_param_a: float | None = None, filter_param_b: float | None = None, resample_filter_uv: _DataType | None = None, filter_param_a_uv: float | None = None, filter_param_b_uv: float | None = None, dither_type: _DataType | None = None, cpu_type: _DataType | None = None, prefer_props: int | None = None, src_left: float | None = None, src_top: float | None = None, src_width: float | None = None, src_height: float | None = None, nominal_luminance: float | None = None, force: int | None = None, force_h: int | None = None, force_v: int | None = None) -> VideoNode: ...

# end implementation

    
# implementation: rgvs

class _Plugin_rgvs_Core_Bound(Plugin):
    """This class implements the module definitions for the "rgvs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BackwardClense(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Clense(self, clip: VideoNode, previous: VideoNode | None = None, next: VideoNode | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def ForwardClense(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def RemoveGrain(self, clip: VideoNode, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def Repair(self, clip: VideoNode, repairclip: VideoNode, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def VerticalCleaner(self, clip: VideoNode, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...

class _Plugin_rgvs_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "rgvs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BackwardClense(self, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Clense(self, previous: VideoNode | None = None, next: VideoNode | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def ForwardClense(self, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def RemoveGrain(self, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def Repair(self, repairclip: VideoNode, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def VerticalCleaner(self, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: scxvid

class _Plugin_scxvid_Core_Bound(Plugin):
    """This class implements the module definitions for the "scxvid" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Scxvid(self, clip: VideoNode, log: _DataType | None = None, use_slices: int | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_scxvid_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "scxvid" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Scxvid(self, log: _DataType | None = None, use_slices: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: std

class _Plugin_std_Core_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(self, clip: VideoNode, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None, color: _SingleAndSequence[float] | None = None) -> VideoNode: ...
    def AssumeFPS(self, clip: VideoNode, src: VideoNode | None = None, fpsnum: int | None = None, fpsden: int | None = None) -> VideoNode: ...
    def AssumeSampleRate(self, clip: AudioNode, src: AudioNode | None = None, samplerate: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioGain(self, clip: AudioNode, gain: _SingleAndSequence[float] | None = None, overflow_error: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioLoop(self, clip: AudioNode, times: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioMix(self, clips: _SingleAndSequence[AudioNode], matrix: _SingleAndSequence[float], channels_out: _SingleAndSequence[int], overflow_error: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioReverse(self, clip: AudioNode) -> _ConstantFormatVideoNode: ...
    def AudioSplice(self, clips: _SingleAndSequence[AudioNode]) -> _ConstantFormatVideoNode: ...
    def AudioTrim(self, clip: AudioNode, first: int | None = None, last: int | None = None, length: int | None = None) -> _ConstantFormatVideoNode: ...
    def AverageFrames(self, clips: _SingleAndSequence[VideoNode], weights: _SingleAndSequence[float], scale: float | None = None, scenechange: int | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Binarize(self, clip: VideoNode, threshold: _SingleAndSequence[float] | None = None, v0: _SingleAndSequence[float] | None = None, v1: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def BinarizeMask(self, clip: VideoNode, threshold: _SingleAndSequence[float] | None = None, v0: _SingleAndSequence[float] | None = None, v1: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def BlankAudio(self, clip: AudioNode | None = None, channels: _SingleAndSequence[int] | None = None, bits: int | None = None, sampletype: int | None = None, samplerate: int | None = None, length: int | None = None, keep: int | None = None) -> _ConstantFormatVideoNode: ...
    def BlankClip(self, clip: VideoNode | None = None, width: int | None = None, height: int | None = None, format: int | None = None, length: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, color: _SingleAndSequence[float] | None = None, keep: int | None = None, varsize: int | None = None, varformat: int | None = None) -> VideoNode: ...
    def BoxBlur(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> _ConstantFormatVideoNode: ...
    def Cache(self, clip: VideoNode, size: int | None = None, fixed: int | None = None, make_linear: int | None = None) -> _ConstantFormatVideoNode: ...
    def ClipToProp(self, clip: VideoNode, mclip: VideoNode, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Convolution(self, clip: VideoNode, matrix: _SingleAndSequence[float], bias: float | None = None, divisor: float | None = None, planes: _SingleAndSequence[int] | None = None, saturate: int | None = None, mode: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def CopyFrameProps(self, clip: VideoNode, prop_src: VideoNode, props: _SingleAndSequence[_DataType] | None = None) -> VideoNode: ...
    def Crop(self, clip: VideoNode, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> _ConstantFormatVideoNode: ...
    def CropAbs(self, clip: VideoNode, width: int, height: int, left: int | None = None, top: int | None = None, x: int | None = None, y: int | None = None) -> VideoNode: ...
    def CropRel(self, clip: VideoNode, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> _ConstantFormatVideoNode: ...
    def Deflate(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None) -> _ConstantFormatVideoNode: ...
    def DeleteFrames(self, clip: VideoNode, frames: _SingleAndSequence[int]) -> VideoNode: ...
    def DoubleWeave(self, clip: VideoNode, tff: int | None = None) -> _ConstantFormatVideoNode: ...
    def DuplicateFrames(self, clip: VideoNode, frames: _SingleAndSequence[int]) -> VideoNode: ...
    def Expr(self, clips: _SingleAndSequence[VideoNode], expr: _SingleAndSequence[_DataType], format: int | None = None) -> _ConstantFormatVideoNode: ...
    def FlipHorizontal(self, clip: VideoNode) -> VideoNode: ...
    def FlipVertical(self, clip: VideoNode) -> VideoNode: ...
    def FrameEval(self, clip: VideoNode, eval: _VSMapValueCallback[_VapourSynthMapValue], prop_src: _SingleAndSequence[VideoNode] | None = None, clip_src: _SingleAndSequence[VideoNode] | None = None) -> VideoNode: ...
    def FreezeFrames(self, clip: VideoNode, first: _SingleAndSequence[int] | None = None, last: _SingleAndSequence[int] | None = None, replacement: _SingleAndSequence[int] | None = None) -> VideoNode: ...
    def Inflate(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None) -> _ConstantFormatVideoNode: ...
    def Interleave(self, clips: _SingleAndSequence[VideoNode], extend: int | None = None, mismatch: int | None = None, modify_duration: int | None = None) -> VideoNode: ...
    def Invert(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None) -> VideoNode: ...
    def InvertMask(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None) -> VideoNode: ...
    def Levels(self, clip: VideoNode, min_in: _SingleAndSequence[float] | None = None, max_in: _SingleAndSequence[float] | None = None, gamma: _SingleAndSequence[float] | None = None, min_out: _SingleAndSequence[float] | None = None, max_out: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Limiter(self, clip: VideoNode, min: _SingleAndSequence[float] | None = None, max: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def LoadAllPlugins(self, path: _DataType) -> _ConstantFormatVideoNode: ...
    def LoadPlugin(self, path: _DataType, altsearchpath: int | None = None, forcens: _DataType | None = None, forceid: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Loop(self, clip: VideoNode, times: int | None = None) -> VideoNode: ...
    def Lut(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, lut: _SingleAndSequence[int] | None = None, lutf: _SingleAndSequence[float] | None = None, function: _VSMapValueCallback[_VapourSynthMapValue] | None = None, bits: int | None = None, floatout: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lut2(self, clipa: VideoNode, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None, lut: _SingleAndSequence[int] | None = None, lutf: _SingleAndSequence[float] | None = None, function: _VSMapValueCallback[_VapourSynthMapValue] | None = None, bits: int | None = None, floatout: int | None = None) -> _ConstantFormatVideoNode: ...
    def MakeDiff(self, clipa: VideoNode, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def MakeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> _ConstantFormatVideoNode: ...
    def MaskedMerge(self, clipa: VideoNode, clipb: VideoNode, mask: VideoNode, planes: _SingleAndSequence[int] | None = None, first_plane: int | None = None, premultiplied: int | None = None) -> _ConstantFormatVideoNode: ...
    def Maximum(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None, coordinates: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Median(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Merge(self, clipa: VideoNode, clipb: VideoNode, weight: _SingleAndSequence[float] | None = None) -> _ConstantFormatVideoNode: ...
    def MergeDiff(self, clipa: VideoNode, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def MergeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> _ConstantFormatVideoNode: ...
    def Minimum(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None, coordinates: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def ModifyFrame(self, clip: VideoNode, clips: _SingleAndSequence[VideoNode], selector: _VSMapValueCallback[_VapourSynthMapValue]) -> VideoNode: ...
    def PEMVerifier(self, clip: VideoNode, upper: _SingleAndSequence[float] | None = None, lower: _SingleAndSequence[float] | None = None) -> _ConstantFormatVideoNode: ...
    def PlaneStats(self, clipa: VideoNode, clipb: VideoNode | None = None, plane: int | None = None, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def PreMultiply(self, clip: VideoNode, alpha: VideoNode) -> _ConstantFormatVideoNode: ...
    def Prewitt(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, scale: float | None = None) -> _ConstantFormatVideoNode: ...
    def PropToClip(self, clip: VideoNode, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def RemoveFrameProps(self, clip: VideoNode, props: _SingleAndSequence[_DataType] | None = None) -> VideoNode: ...
    def Reverse(self, clip: VideoNode) -> VideoNode: ...
    def SelectEvery(self, clip: VideoNode, cycle: int, offsets: _SingleAndSequence[int], modify_duration: int | None = None) -> VideoNode: ...
    def SeparateFields(self, clip: VideoNode, tff: int | None = None, modify_duration: int | None = None) -> _ConstantFormatVideoNode: ...
    def SetAudioCache(self, clip: AudioNode, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> _ConstantFormatVideoNode: ...
    def SetFieldBased(self, clip: VideoNode, value: int) -> VideoNode: ...
    def SetFrameProp(self, clip: VideoNode, prop: _DataType, intval: _SingleAndSequence[int] | None = None, floatval: _SingleAndSequence[float] | None = None, data: _SingleAndSequence[_DataType] | None = None) -> VideoNode: ...
    def SetFrameProps(self, clip: VideoNode, **kwargs: _VapourSynthMapValue) -> VideoNode: ...
    def SetMaxCPU(self, cpu: _DataType) -> _ConstantFormatVideoNode: ...
    def SetVideoCache(self, clip: VideoNode, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> VideoNode: ...
    def ShuffleChannels(self, clips: _SingleAndSequence[AudioNode], channels_in: _SingleAndSequence[int], channels_out: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def ShufflePlanes(self, clips: _SingleAndSequence[VideoNode], planes: _SingleAndSequence[int], colorfamily: int, prop_src: VideoNode | None = None) -> _ConstantFormatVideoNode: ...
    def Sobel(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, scale: float | None = None) -> _ConstantFormatVideoNode: ...
    def Splice(self, clips: _SingleAndSequence[VideoNode], mismatch: int | None = None) -> VideoNode: ...
    def SplitChannels(self, clip: AudioNode) -> _ConstantFormatVideoNode: ...
    def SplitPlanes(self, clip: VideoNode) -> _ConstantFormatVideoNode: ...
    def StackHorizontal(self, clips: _SingleAndSequence[VideoNode]) -> VideoNode: ...
    def StackVertical(self, clips: _SingleAndSequence[VideoNode]) -> VideoNode: ...
    def TestAudio(self, channels: _SingleAndSequence[int] | None = None, bits: int | None = None, isfloat: int | None = None, samplerate: int | None = None, length: int | None = None) -> _ConstantFormatVideoNode: ...
    def Transpose(self, clip: VideoNode) -> _ConstantFormatVideoNode: ...
    def Trim(self, clip: VideoNode, first: int | None = None, last: int | None = None, length: int | None = None) -> VideoNode: ...
    def Turn180(self, clip: VideoNode) -> VideoNode: ...

class _Plugin_std_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None, color: _SingleAndSequence[float] | None = None) -> VideoNode: ...
    def AssumeFPS(self, src: VideoNode | None = None, fpsnum: int | None = None, fpsden: int | None = None) -> VideoNode: ...
    def AverageFrames(self, weights: _SingleAndSequence[float], scale: float | None = None, scenechange: int | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Binarize(self, threshold: _SingleAndSequence[float] | None = None, v0: _SingleAndSequence[float] | None = None, v1: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def BinarizeMask(self, threshold: _SingleAndSequence[float] | None = None, v0: _SingleAndSequence[float] | None = None, v1: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def BlankClip(self, width: int | None = None, height: int | None = None, format: int | None = None, length: int | None = None, fpsnum: int | None = None, fpsden: int | None = None, color: _SingleAndSequence[float] | None = None, keep: int | None = None, varsize: int | None = None, varformat: int | None = None) -> VideoNode: ...
    def BoxBlur(self, planes: _SingleAndSequence[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> _ConstantFormatVideoNode: ...
    def Cache(self, size: int | None = None, fixed: int | None = None, make_linear: int | None = None) -> _ConstantFormatVideoNode: ...
    def ClipToProp(self, mclip: VideoNode, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def Convolution(self, matrix: _SingleAndSequence[float], bias: float | None = None, divisor: float | None = None, planes: _SingleAndSequence[int] | None = None, saturate: int | None = None, mode: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def CopyFrameProps(self, prop_src: VideoNode, props: _SingleAndSequence[_DataType] | None = None) -> VideoNode: ...
    def Crop(self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> _ConstantFormatVideoNode: ...
    def CropAbs(self, width: int, height: int, left: int | None = None, top: int | None = None, x: int | None = None, y: int | None = None) -> VideoNode: ...
    def CropRel(self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None) -> _ConstantFormatVideoNode: ...
    def Deflate(self, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None) -> _ConstantFormatVideoNode: ...
    def DeleteFrames(self, frames: _SingleAndSequence[int]) -> VideoNode: ...
    def DoubleWeave(self, tff: int | None = None) -> _ConstantFormatVideoNode: ...
    def DuplicateFrames(self, frames: _SingleAndSequence[int]) -> VideoNode: ...
    def Expr(self, expr: _SingleAndSequence[_DataType], format: int | None = None) -> _ConstantFormatVideoNode: ...
    def FlipHorizontal(self) -> VideoNode: ...
    def FlipVertical(self) -> VideoNode: ...
    def FrameEval(self, eval: _VSMapValueCallback[_VapourSynthMapValue], prop_src: _SingleAndSequence[VideoNode] | None = None, clip_src: _SingleAndSequence[VideoNode] | None = None) -> VideoNode: ...
    def FreezeFrames(self, first: _SingleAndSequence[int] | None = None, last: _SingleAndSequence[int] | None = None, replacement: _SingleAndSequence[int] | None = None) -> VideoNode: ...
    def Inflate(self, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None) -> _ConstantFormatVideoNode: ...
    def Interleave(self, extend: int | None = None, mismatch: int | None = None, modify_duration: int | None = None) -> VideoNode: ...
    def Invert(self, planes: _SingleAndSequence[int] | None = None) -> VideoNode: ...
    def InvertMask(self, planes: _SingleAndSequence[int] | None = None) -> VideoNode: ...
    def Levels(self, min_in: _SingleAndSequence[float] | None = None, max_in: _SingleAndSequence[float] | None = None, gamma: _SingleAndSequence[float] | None = None, min_out: _SingleAndSequence[float] | None = None, max_out: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Limiter(self, min: _SingleAndSequence[float] | None = None, max: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Loop(self, times: int | None = None) -> VideoNode: ...
    def Lut(self, planes: _SingleAndSequence[int] | None = None, lut: _SingleAndSequence[int] | None = None, lutf: _SingleAndSequence[float] | None = None, function: _VSMapValueCallback[_VapourSynthMapValue] | None = None, bits: int | None = None, floatout: int | None = None) -> _ConstantFormatVideoNode: ...
    def Lut2(self, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None, lut: _SingleAndSequence[int] | None = None, lutf: _SingleAndSequence[float] | None = None, function: _VSMapValueCallback[_VapourSynthMapValue] | None = None, bits: int | None = None, floatout: int | None = None) -> _ConstantFormatVideoNode: ...
    def MakeDiff(self, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def MakeFullDiff(self, clipb: VideoNode) -> _ConstantFormatVideoNode: ...
    def MaskedMerge(self, clipb: VideoNode, mask: VideoNode, planes: _SingleAndSequence[int] | None = None, first_plane: int | None = None, premultiplied: int | None = None) -> _ConstantFormatVideoNode: ...
    def Maximum(self, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None, coordinates: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Median(self, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def Merge(self, clipb: VideoNode, weight: _SingleAndSequence[float] | None = None) -> _ConstantFormatVideoNode: ...
    def MergeDiff(self, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def MergeFullDiff(self, clipb: VideoNode) -> _ConstantFormatVideoNode: ...
    def Minimum(self, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None, coordinates: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def ModifyFrame(self, clips: _SingleAndSequence[VideoNode], selector: _VSMapValueCallback[_VapourSynthMapValue]) -> VideoNode: ...
    def PEMVerifier(self, upper: _SingleAndSequence[float] | None = None, lower: _SingleAndSequence[float] | None = None) -> _ConstantFormatVideoNode: ...
    def PlaneStats(self, clipb: VideoNode | None = None, plane: int | None = None, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def PreMultiply(self, alpha: VideoNode) -> _ConstantFormatVideoNode: ...
    def Prewitt(self, planes: _SingleAndSequence[int] | None = None, scale: float | None = None) -> _ConstantFormatVideoNode: ...
    def PropToClip(self, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def RemoveFrameProps(self, props: _SingleAndSequence[_DataType] | None = None) -> VideoNode: ...
    def Reverse(self) -> VideoNode: ...
    def SelectEvery(self, cycle: int, offsets: _SingleAndSequence[int], modify_duration: int | None = None) -> VideoNode: ...
    def SeparateFields(self, tff: int | None = None, modify_duration: int | None = None) -> _ConstantFormatVideoNode: ...
    def SetFieldBased(self, value: int) -> VideoNode: ...
    def SetFrameProp(self, prop: _DataType, intval: _SingleAndSequence[int] | None = None, floatval: _SingleAndSequence[float] | None = None, data: _SingleAndSequence[_DataType] | None = None) -> VideoNode: ...
    def SetFrameProps(self, **kwargs: Any) -> VideoNode: ...
    def SetVideoCache(self, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> VideoNode: ...
    def ShufflePlanes(self, planes: _SingleAndSequence[int], colorfamily: int, prop_src: VideoNode | None = None) -> _ConstantFormatVideoNode: ...
    def Sobel(self, planes: _SingleAndSequence[int] | None = None, scale: float | None = None) -> _ConstantFormatVideoNode: ...
    def Splice(self, mismatch: int | None = None) -> VideoNode: ...
    def SplitPlanes(self) -> _ConstantFormatVideoNode: ...
    def StackHorizontal(self) -> VideoNode: ...
    def StackVertical(self) -> VideoNode: ...
    def Transpose(self) -> _ConstantFormatVideoNode: ...
    def Trim(self, first: int | None = None, last: int | None = None, length: int | None = None) -> VideoNode: ...
    def Turn180(self) -> VideoNode: ...

class _Plugin_std_AudioNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AssumeSampleRate(self, src: AudioNode | None = None, samplerate: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioGain(self, gain: _SingleAndSequence[float] | None = None, overflow_error: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioLoop(self, times: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioMix(self, matrix: _SingleAndSequence[float], channels_out: _SingleAndSequence[int], overflow_error: int | None = None) -> _ConstantFormatVideoNode: ...
    def AudioReverse(self) -> _ConstantFormatVideoNode: ...
    def AudioSplice(self) -> _ConstantFormatVideoNode: ...
    def AudioTrim(self, first: int | None = None, last: int | None = None, length: int | None = None) -> _ConstantFormatVideoNode: ...
    def BlankAudio(self, channels: _SingleAndSequence[int] | None = None, bits: int | None = None, sampletype: int | None = None, samplerate: int | None = None, length: int | None = None, keep: int | None = None) -> _ConstantFormatVideoNode: ...
    def SetAudioCache(self, mode: int | None = None, fixedsize: int | None = None, maxsize: int | None = None, maxhistory: int | None = None) -> _ConstantFormatVideoNode: ...
    def ShuffleChannels(self, channels_in: _SingleAndSequence[int], channels_out: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def SplitChannels(self) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: vszip

class _Plugin_vszip_Core_Bound(Plugin):
    """This class implements the module definitions for the "vszip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AdaptiveBinarize(self, clip: VideoNode, clip2: VideoNode, c: int | None = None) -> _ConstantFormatVideoNode: ...
    def Bilateral(self, clip: VideoNode, ref: VideoNode | None = None, sigmaS: _SingleAndSequence[float] | None = None, sigmaR: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None, algorithm: _SingleAndSequence[int] | None = None, PBFICnum: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def BoxBlur(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> _ConstantFormatVideoNode: ...
    def Checkmate(self, clip: VideoNode, thr: int | None = None, tmax: int | None = None, tthr2: int | None = None) -> _ConstantFormatVideoNode: ...
    def CLAHE(self, clip: VideoNode, limit: int | None = None, tiles: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def CombMaskMT(self, clip: VideoNode, thY1: int | None = None, thY2: int | None = None) -> _ConstantFormatVideoNode: ...
    def Limiter(self, clip: VideoNode, min: _SingleAndSequence[float] | None = None, max: _SingleAndSequence[float] | None = None, tv_range: int | None = None) -> _ConstantFormatVideoNode: ...
    def Metrics(self, reference: VideoNode, distorted: VideoNode, mode: int | None = None) -> VideoNode: ...
    def PlaneAverage(self, clipa: VideoNode, exclude: _SingleAndSequence[int], clipb: VideoNode | None = None, planes: _SingleAndSequence[int] | None = None, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def PlaneMinMax(self, clipa: VideoNode, minthr: float | None = None, maxthr: float | None = None, clipb: VideoNode | None = None, planes: _SingleAndSequence[int] | None = None, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def RFS(self, clipa: VideoNode, clipb: VideoNode, frames: _SingleAndSequence[int], mismatch: int | None = None, planes: _SingleAndSequence[int] | None = None) -> VideoNode: ...

class _Plugin_vszip_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "vszip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AdaptiveBinarize(self, clip2: VideoNode, c: int | None = None) -> _ConstantFormatVideoNode: ...
    def Bilateral(self, ref: VideoNode | None = None, sigmaS: _SingleAndSequence[float] | None = None, sigmaR: _SingleAndSequence[float] | None = None, planes: _SingleAndSequence[int] | None = None, algorithm: _SingleAndSequence[int] | None = None, PBFICnum: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def BoxBlur(self, planes: _SingleAndSequence[int] | None = None, hradius: int | None = None, hpasses: int | None = None, vradius: int | None = None, vpasses: int | None = None) -> _ConstantFormatVideoNode: ...
    def Checkmate(self, thr: int | None = None, tmax: int | None = None, tthr2: int | None = None) -> _ConstantFormatVideoNode: ...
    def CLAHE(self, limit: int | None = None, tiles: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def CombMaskMT(self, thY1: int | None = None, thY2: int | None = None) -> _ConstantFormatVideoNode: ...
    def Limiter(self, min: _SingleAndSequence[float] | None = None, max: _SingleAndSequence[float] | None = None, tv_range: int | None = None) -> _ConstantFormatVideoNode: ...
    def Metrics(self, distorted: VideoNode, mode: int | None = None) -> VideoNode: ...
    def PlaneAverage(self, exclude: _SingleAndSequence[int], clipb: VideoNode | None = None, planes: _SingleAndSequence[int] | None = None, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def PlaneMinMax(self, minthr: float | None = None, maxthr: float | None = None, clipb: VideoNode | None = None, planes: _SingleAndSequence[int] | None = None, prop: _DataType | None = None) -> _ConstantFormatVideoNode: ...
    def RFS(self, clipb: VideoNode, frames: _SingleAndSequence[int], mismatch: int | None = None, planes: _SingleAndSequence[int] | None = None) -> VideoNode: ...

# end implementation

    
# implementation: wwxd

class _Plugin_wwxd_Core_Bound(Plugin):
    """This class implements the module definitions for the "wwxd" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def WWXD(self, clip: VideoNode) -> _ConstantFormatVideoNode: ...

class _Plugin_wwxd_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "wwxd" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def WWXD(self) -> _ConstantFormatVideoNode: ...

# end implementation

    
# implementation: zsmooth

class _Plugin_zsmooth_Core_Bound(Plugin):
    """This class implements the module definitions for the "zsmooth" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DegrainMedian(self, clip: VideoNode, limit: _SingleAndSequence[float] | None = None, mode: _SingleAndSequence[int] | None = None, interlaced: int | None = None, norow: int | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...
    def FluxSmoothST(self, clip: VideoNode, temporal_threshold: _SingleAndSequence[float] | None = None, spatial_threshold: _SingleAndSequence[float] | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...
    def FluxSmoothT(self, clip: VideoNode, temporal_threshold: _SingleAndSequence[float] | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...
    def RemoveGrain(self, clip: VideoNode, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def TemporalMedian(self, clip: VideoNode, radius: int | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def TemporalSoften(self, clip: VideoNode, radius: int | None = None, threshold: _SingleAndSequence[float] | None = None, scenechange: int | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...

class _Plugin_zsmooth_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "zsmooth" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DegrainMedian(self, limit: _SingleAndSequence[float] | None = None, mode: _SingleAndSequence[int] | None = None, interlaced: int | None = None, norow: int | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...
    def FluxSmoothST(self, temporal_threshold: _SingleAndSequence[float] | None = None, spatial_threshold: _SingleAndSequence[float] | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...
    def FluxSmoothT(self, temporal_threshold: _SingleAndSequence[float] | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...
    def RemoveGrain(self, mode: _SingleAndSequence[int]) -> _ConstantFormatVideoNode: ...
    def TemporalMedian(self, radius: int | None = None, planes: _SingleAndSequence[int] | None = None) -> _ConstantFormatVideoNode: ...
    def TemporalSoften(self, radius: int | None = None, threshold: _SingleAndSequence[float] | None = None, scenechange: int | None = None, scalep: int | None = None) -> _ConstantFormatVideoNode: ...

# end implementation



class RawNode:
    def __init__(self) -> None: ...

    def get_frame(self, n: int) -> RawFrame: ...

    @overload
    def get_frame_async(self, n: int, cb: None = None) -> _Future[RawFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[RawFrame | None, Exception | None], None]) -> None: ...

    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[RawFrame]: ...

    def clear_cache(self) -> None: ...

    def set_output(self, index: int = 0) -> None: ...

    def is_inspectable(self, version: int | None = None) -> bool: ...

    if not TYPE_CHECKING:
        @property
        def _node_name(self) -> str: ...

        @property
        def _name(self) -> str: ...

        @property
        def _inputs(self) -> Dict[str, _VapourSynthMapValue]: ...

        @property
        def _timings(self) -> int: ...

        @property
        def _mode(self) -> FilterMode: ...

        @property
        def _dependencies(self): ...

    @overload
    def __eq__(self: 'SelfRawNode', other: 'SelfRawNode', /) -> bool: ...

    @overload
    def __eq__(self, other: Any, /) -> Literal[False]: ...

    def __add__(self: 'SelfRawNode', other: 'SelfRawNode', /) -> 'SelfRawNode': ...

    def __radd__(self: 'SelfRawNode', other: 'SelfRawNode', /) -> 'SelfRawNode': ...

    def __mul__(self: 'SelfRawNode', other: int) -> 'SelfRawNode': ...

    def __rmul__(self: 'SelfRawNode', other: int) -> 'SelfRawNode': ...

    def __getitem__(self: 'SelfRawNode', index: int | slice, /) -> 'SelfRawNode': ...

    def __len__(self) -> int: ...


SelfRawNode = TypeVar('SelfRawNode', bound=RawNode)


class VideoNode(RawNode):
    format: VideoFormat | None

    width: int
    height: int

    fps_num: int
    fps_den: int

    fps: Fraction

    num_frames: int

    def set_output(
        self, index: int = 0, alpha: 'VideoNode' | None = None, alt_output: Literal[0, 1, 2] = 0
    ) -> None: ...

    def output(
        self, fileobj: BinaryIO, y4m: bool = False, progress_update: Callable[[int, int], None] | None = None,
        prefetch: int = 0, backlog: int = -1
    ) -> None: ...

    def get_frame(self, n: int) -> VideoFrame: ...

    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[VideoFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[VideoFrame | None, Exception | None], None]) -> None: ...

    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[VideoFrame]: ...


    # instance_bound_VideoNode: akarin
    @property
    def akarin(self) -> _Plugin_akarin_VideoNode_Bound:
        """Akarin's Experimental Filters"""
    # end instance
    # instance_bound_VideoNode: cs
    @property
    def cs(self) -> _Plugin_cs_VideoNode_Bound:
        """carefulsource"""
    # end instance
    # instance_bound_VideoNode: descale
    @property
    def descale(self) -> _Plugin_descale_VideoNode_Bound:
        """Undo linear interpolation"""
    # end instance
    # instance_bound_VideoNode: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_VideoNode_Bound:
        """Format converter"""
    # end instance
    # instance_bound_VideoNode: imwri
    @property
    def imwri(self) -> _Plugin_imwri_VideoNode_Bound:
        """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
    # end instance
    # instance_bound_VideoNode: placebo
    @property
    def placebo(self) -> _Plugin_placebo_VideoNode_Bound:
        """libplacebo plugin for VapourSynth"""
    # end instance
    # instance_bound_VideoNode: resize
    @property
    def resize(self) -> _Plugin_resize_VideoNode_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_VideoNode: resize2
    @property
    def resize2(self) -> _Plugin_resize2_VideoNode_Bound:
        """Built-in VapourSynth resizer based on zimg with some modifications."""
    # end instance
    # instance_bound_VideoNode: rgvs
    @property
    def rgvs(self) -> _Plugin_rgvs_VideoNode_Bound:
        """RemoveGrain VapourSynth Port"""
    # end instance
    # instance_bound_VideoNode: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_VideoNode_Bound:
        """VapourSynth Scxvid Plugin"""
    # end instance
    # instance_bound_VideoNode: std
    @property
    def std(self) -> _Plugin_std_VideoNode_Bound:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_VideoNode: vszip
    @property
    def vszip(self) -> _Plugin_vszip_VideoNode_Bound:
        """VapourSynth Zig Image Process"""
    # end instance
    # instance_bound_VideoNode: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_VideoNode_Bound:
        """Scene change detection approximately like Xvid's"""
    # end instance
    # instance_bound_VideoNode: zsmooth
    @property
    def zsmooth(self) -> _Plugin_zsmooth_VideoNode_Bound:
        """Smoothing functions in Zig"""
    # end instance


class _ConstantFormatVideoNode(VideoNode):
    format: VideoFormat


    # instance_bound_VideoNode: akarin
    @property
    def akarin(self) -> _Plugin_akarin_VideoNode_Bound:
        """Akarin's Experimental Filters"""
    # end instance
    # instance_bound_VideoNode: cs
    @property
    def cs(self) -> _Plugin_cs_VideoNode_Bound:
        """carefulsource"""
    # end instance
    # instance_bound_VideoNode: descale
    @property
    def descale(self) -> _Plugin_descale_VideoNode_Bound:
        """Undo linear interpolation"""
    # end instance
    # instance_bound_VideoNode: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_VideoNode_Bound:
        """Format converter"""
    # end instance
    # instance_bound_VideoNode: imwri
    @property
    def imwri(self) -> _Plugin_imwri_VideoNode_Bound:
        """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
    # end instance
    # instance_bound_VideoNode: placebo
    @property
    def placebo(self) -> _Plugin_placebo_VideoNode_Bound:
        """libplacebo plugin for VapourSynth"""
    # end instance
    # instance_bound_VideoNode: resize
    @property
    def resize(self) -> _Plugin_resize_VideoNode_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_VideoNode: resize2
    @property
    def resize2(self) -> _Plugin_resize2_VideoNode_Bound:
        """Built-in VapourSynth resizer based on zimg with some modifications."""
    # end instance
    # instance_bound_VideoNode: rgvs
    @property
    def rgvs(self) -> _Plugin_rgvs_VideoNode_Bound:
        """RemoveGrain VapourSynth Port"""
    # end instance
    # instance_bound_VideoNode: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_VideoNode_Bound:
        """VapourSynth Scxvid Plugin"""
    # end instance
    # instance_bound_VideoNode: std
    @property
    def std(self) -> _Plugin_std_VideoNode_Bound:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_VideoNode: vszip
    @property
    def vszip(self) -> _Plugin_vszip_VideoNode_Bound:
        """VapourSynth Zig Image Process"""
    # end instance
    # instance_bound_VideoNode: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_VideoNode_Bound:
        """Scene change detection approximately like Xvid's"""
    # end instance
    # instance_bound_VideoNode: zsmooth
    @property
    def zsmooth(self) -> _Plugin_zsmooth_VideoNode_Bound:
        """Smoothing functions in Zig"""
    # end instance


class AudioNode(RawNode):
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int

    channel_layout: int
    num_channels: int

    sample_rate: int
    num_samples: int

    num_frames: int

    @property
    def channels(self) -> ChannelLayout: ...

    def get_frame(self, n: int) -> AudioFrame: ...

    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[AudioFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[AudioFrame | None, Exception | None], None]) -> None: ...

    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[AudioFrame]: ...

    # instance_bound_AudioNode: std
    @property
    def std(self) -> _Plugin_std_AudioNode_Bound:
        """VapourSynth Core Functions"""
    # end instance


class LogHandle:
    def __init__(self) -> NoReturn: ...


class Function:
    plugin: 'Plugin'
    name: str
    signature: str
    return_signature: str

    def __init__(self) -> NoReturn: ...

    def __call__(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...

    @property
    def __signature__(self) -> Signature: ...


class Plugin:
    identifier: str
    namespace: str
    name: str

    def __init__(self) -> NoReturn: ...

    def __getattr__(self, name: str) -> Function: ...

    def functions(self) -> Iterator[Function]: ...

    @property
    def version(self) -> PluginVersion: ...

    @property
    def plugin_path(self) -> str: ...


class Core:
    def __init__(self) -> NoReturn: ...

    @property
    def num_threads(self) -> int: ...

    @num_threads.setter
    def num_threads(self) -> None: ...

    @property
    def max_cache_size(self) -> int: ...

    @max_cache_size.setter
    def max_cache_size(self) -> None: ...

    @property
    def flags(self) -> int: ...

    def plugins(self) -> Iterator[Plugin]: ...

    def query_video_format(
        self, color_family: ColorFamily, sample_type: SampleType, bits_per_sample: int, subsampling_w: int = 0,
        subsampling_h: int = 0
    ) -> VideoFormat: ...

    def get_video_format(self, id: VideoFormat | int | PresetVideoFormat) -> VideoFormat: ...

    def create_video_frame(self, format: VideoFormat, width: int, height: int) -> VideoFrame: ...

    def log_message(self, message_type: MessageType, message: str) -> None: ...

    def add_log_handler(self, handler_func: Callable[[MessageType, str], None]) -> LogHandle: ...

    def remove_log_handler(self, handle: LogHandle) -> None: ...

    def clear_cache(self) -> None: ...

    def version(self) -> str: ...

    def version_number(self) -> int: ...

    # instance_bound_Core: akarin
    @property
    def akarin(self) -> _Plugin_akarin_Core_Bound:
        """Akarin's Experimental Filters"""
    # end instance
    # instance_bound_Core: bs
    @property
    def bs(self) -> _Plugin_bs_Core_Bound:
        """Best Source 2"""
    # end instance
    # instance_bound_Core: cs
    @property
    def cs(self) -> _Plugin_cs_Core_Bound:
        """carefulsource"""
    # end instance
    # instance_bound_Core: d2v
    @property
    def d2v(self) -> _Plugin_d2v_Core_Bound:
        """D2V Source"""
    # end instance
    # instance_bound_Core: descale
    @property
    def descale(self) -> _Plugin_descale_Core_Bound:
        """Undo linear interpolation"""
    # end instance
    # instance_bound_Core: dgdecodenv
    @property
    def dgdecodenv(self) -> _Plugin_dgdecodenv_Core_Bound:
        """DGDecodeNV for VapourSynth"""
    # end instance
    # instance_bound_Core: dvdsrc2
    @property
    def dvdsrc2(self) -> _Plugin_dvdsrc2_Core_Bound:
        """Dvdsrc 2nd tour"""
    # end instance
    # instance_bound_Core: ffms2
    @property
    def ffms2(self) -> _Plugin_ffms2_Core_Bound:
        """FFmpegSource 2 for VapourSynth"""
    # end instance
    # instance_bound_Core: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_Core_Bound:
        """Format converter"""
    # end instance
    # instance_bound_Core: imwri
    @property
    def imwri(self) -> _Plugin_imwri_Core_Bound:
        """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
    # end instance
    # instance_bound_Core: lsmas
    @property
    def lsmas(self) -> _Plugin_lsmas_Core_Bound:
        """LSMASHSource for VapourSynth"""
    # end instance
    # instance_bound_Core: placebo
    @property
    def placebo(self) -> _Plugin_placebo_Core_Bound:
        """libplacebo plugin for VapourSynth"""
    # end instance
    # instance_bound_Core: resize
    @property
    def resize(self) -> _Plugin_resize_Core_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_Core: resize2
    @property
    def resize2(self) -> _Plugin_resize2_Core_Bound:
        """Built-in VapourSynth resizer based on zimg with some modifications."""
    # end instance
    # instance_bound_Core: rgvs
    @property
    def rgvs(self) -> _Plugin_rgvs_Core_Bound:
        """RemoveGrain VapourSynth Port"""
    # end instance
    # instance_bound_Core: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_Core_Bound:
        """VapourSynth Scxvid Plugin"""
    # end instance
    # instance_bound_Core: std
    @property
    def std(self) -> _Plugin_std_Core_Bound:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_Core: vszip
    @property
    def vszip(self) -> _Plugin_vszip_Core_Bound:
        """VapourSynth Zig Image Process"""
    # end instance
    # instance_bound_Core: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_Core_Bound:
        """Scene change detection approximately like Xvid's"""
    # end instance
    # instance_bound_Core: zsmooth
    @property
    def zsmooth(self) -> _Plugin_zsmooth_Core_Bound:
        """Smoothing functions in Zig"""
    # end instance


class _CoreProxy(Core):
    @property
    def core(self) -> Core: ...


core: _CoreProxy
