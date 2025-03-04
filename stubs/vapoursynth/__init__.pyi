# Stop pep8 from complaining (hopefully)
# NOQA

# Ignore Flake Warnings
# flake8: noqa

# Ignore coverage
# (No coverage)

# From https://gist.github.com/pylover/7870c235867cf22817ac5b096defb768
# noinspection PyPep8
# noinspection PyPep8Naming
# noinspection PyTypeChecker
# noinspection PyAbstractClass
# noinspection PyArgumentEqualDefault
# noinspection PyArgumentList
# noinspection PyAssignmentToLoopOrWithParameter
# noinspection PyAttributeOutsideInit
# noinspection PyAugmentAssignment
# noinspection PyBroadException
# noinspection PyByteLiteral
# noinspection PyCallByClass
# noinspection PyChainedComparsons
# noinspection PyClassHasNoInit
# noinspection PyClassicStyleClass
# noinspection PyComparisonWithNone
# noinspection PyCompatibility
# noinspection PyDecorator
# noinspection PyDefaultArgument
# noinspection PyDictCreation
# noinspection PyDictDuplicateKeys
# noinspection PyDocstringTypes
# noinspection PyExceptClausesOrder
# noinspection PyExceptionInheritance
# noinspection PyFromFutureImport
# noinspection PyGlobalUndefined
# noinspection PyIncorrectDocstring
# noinspection PyInitNewSignature
# noinspection PyInterpreter
# noinspection PyListCreation
# noinspection PyMandatoryEncoding
# noinspection PyMethodFirstArgAssignment
# noinspection PyMethodMayBeStatic
# noinspection PyMethodOverriding
# noinspection PyMethodParameters
# noinspection PyMissingConstructor
# noinspection PyMissingOrEmptyDocstring
# noinspection PyNestedDecorators
# noinspection PynonAsciiChar
# noinspection PyNoneFunctionAssignment
# noinspection PyOldStyleClasses
# noinspection PyPackageRequirements
# noinspection PyPropertyAccess
# noinspection PyPropertyDefinition
# noinspection PyProtectedMember
# noinspection PyRaisingNewStyleClass
# noinspection PyRedeclaration
# noinspection PyRedundantParentheses
# noinspection PySetFunctionToLiteral
# noinspection PySimplifyBooleanCheck
# noinspection PySingleQuotedDocstring
# noinspection PyStatementEffect
# noinspection PyStringException
# noinspection PyStringFormat
# noinspection PySuperArguments
# noinspection PyTrailingSemicolon
# noinspection PyTupleAssignmentBalance
# noinspection PyTupleItemAssignment
# noinspection PyUnboundLocalVariable
# noinspection PyUnnecessaryBackslash
# noinspection PyUnreachableCode
# noinspection PyUnresolvedReferences
# noinspection PyUnusedLocal
# noinspection ReturnValueFromInit


from abc import abstractmethod
from ctypes import c_void_p
from enum import IntEnum, IntFlag
from fractions import Fraction
from inspect import Signature
from types import MappingProxyType, TracebackType
from typing import (
    TYPE_CHECKING, Any, BinaryIO, Callable, ContextManager, Dict, Generic, Iterator, Literal,
    MutableMapping, NamedTuple, NoReturn, Optional, Protocol, Sequence, Tuple, Type, TypedDict,
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

T = TypeVar('T')
S = TypeVar('S')

SingleAndSequence = Union[T, Sequence[T]]


@runtime_checkable
class SupportsString(Protocol):
    @abstractmethod
    def __str__(self) -> str:
        ...


DataType = Union[str, bytes, bytearray, SupportsString]

_VapourSynthMapValue = Union[
    SingleAndSequence[int],
    SingleAndSequence[float],
    SingleAndSequence[DataType],
    SingleAndSequence['VideoNode'],
    SingleAndSequence['VideoFrame'],
    SingleAndSequence['AudioNode'],
    SingleAndSequence['AudioFrame'],
    SingleAndSequence['VSMapValueCallback[Any]']
]

BoundVSMapValue = TypeVar('BoundVSMapValue', bound=_VapourSynthMapValue)

VSMapValueCallback = Callable[..., BoundVSMapValue]


class _Future(Generic[T]):
    def set_result(self, value: T) -> None: ...

    def set_exception(self, exception: BaseException) -> None: ...

    def result(self) -> T: ...

    def exception(self) -> Union[NoReturn, None]: ...

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
    def get_current_environment(self) -> Union[EnvironmentData, None]: ...

    @abstractmethod
    def set_environment(self, environment: Union[EnvironmentData, None]) -> Union[EnvironmentData, None]: ...

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
    alpha: Union['VideoNode', None]
    alt_output: Literal[0, 1, 2]


class Error(Exception):
    ...


def clear_output(index: int = 0) -> None:
    ...


def clear_outputs() -> None:
    ...


def get_outputs() -> MappingProxyType[int, Union[VideoOutputTuple, 'AudioNode']]:
    ...


def get_output(index: int = 0) -> Union[VideoOutputTuple, 'AudioNode']:
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
        color_family: Union[ColorFamily, None] = None,
        sample_type: Union[SampleType, None] = None,
        bits_per_sample: Union[int, None] = None,
        subsampling_w: Union[int, None] = None,
        subsampling_h: Union[int, None] = None
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
        self, exc_type: Union[Type[BaseException], None],
        exc_value: Union[BaseException, None],
        traceback: Union[TracebackType, None], /,
    ) -> Union[bool, None]: ...

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
    def Cambi(self, clip: 'VideoNode', window_size: Optional[int] = None, topk: Optional[float] = None, tvi_threshold: Optional[float] = None, scores: Optional[int] = None, scaling: Optional[float] = None) -> 'VideoNode': ...
    def DLISR(self, clip: 'VideoNode', scale: Optional[int] = None, device_id: Optional[int] = None) -> 'VideoNode': ...
    def DLVFX(self, clip: 'VideoNode', op: int, scale: Optional[float] = None, strength: Optional[float] = None, output_depth: Optional[int] = None, num_streams: Optional[int] = None, model_dir: Optional[DataType] = None) -> 'VideoNode': ...
    def Expr(self, clips: SingleAndSequence['VideoNode'], expr: SingleAndSequence[DataType], format: Optional[int] = None, opt: Optional[int] = None, boundary: Optional[int] = None) -> 'VideoNode': ...
    def ExprTest(self, clips: SingleAndSequence[float], expr: DataType, props: Optional[VSMapValueCallback[_VapourSynthMapValue]] = None, ref: Optional['VideoNode'] = None, vars: Optional[int] = None) -> 'VideoNode': ...
    def PickFrames(self, clip: 'VideoNode', indices: SingleAndSequence[int]) -> 'VideoNode': ...
    def PropExpr(self, clips: SingleAndSequence['VideoNode'], dict: VSMapValueCallback[_VapourSynthMapValue]) -> 'VideoNode': ...
    def Select(self, clip_src: SingleAndSequence['VideoNode'], prop_src: SingleAndSequence['VideoNode'], expr: SingleAndSequence[DataType]) -> 'VideoNode': ...
    def Text(self, clips: SingleAndSequence['VideoNode'], text: DataType, alignment: Optional[int] = None, scale: Optional[int] = None, prop: Optional[DataType] = None, strict: Optional[int] = None, vspipe: Optional[int] = None) -> 'VideoNode': ...
    def Tmpl(self, clips: SingleAndSequence['VideoNode'], prop: SingleAndSequence[DataType], text: SingleAndSequence[DataType]) -> 'VideoNode': ...
    def Version(self) -> 'VideoNode': ...

class _Plugin_akarin_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cambi(self, window_size: Optional[int] = None, topk: Optional[float] = None, tvi_threshold: Optional[float] = None, scores: Optional[int] = None, scaling: Optional[float] = None) -> 'VideoNode': ...
    def DLISR(self, scale: Optional[int] = None, device_id: Optional[int] = None) -> 'VideoNode': ...
    def DLVFX(self, op: int, scale: Optional[float] = None, strength: Optional[float] = None, output_depth: Optional[int] = None, num_streams: Optional[int] = None, model_dir: Optional[DataType] = None) -> 'VideoNode': ...
    def Expr(self, expr: SingleAndSequence[DataType], format: Optional[int] = None, opt: Optional[int] = None, boundary: Optional[int] = None) -> 'VideoNode': ...
    def PickFrames(self, indices: SingleAndSequence[int]) -> 'VideoNode': ...
    def PropExpr(self, dict: VSMapValueCallback[_VapourSynthMapValue]) -> 'VideoNode': ...
    def Select(self, prop_src: SingleAndSequence['VideoNode'], expr: SingleAndSequence[DataType]) -> 'VideoNode': ...
    def Text(self, text: DataType, alignment: Optional[int] = None, scale: Optional[int] = None, prop: Optional[DataType] = None, strict: Optional[int] = None, vspipe: Optional[int] = None) -> 'VideoNode': ...
    def Tmpl(self, prop: SingleAndSequence[DataType], text: SingleAndSequence[DataType]) -> 'VideoNode': ...

# end implementation

    
# implementation: bs

_ReturnDict_bs_TrackInfo = TypedDict("_ReturnDict_bs_TrackInfo", {"mediatype": int, "mediatypestr": DataType, "codec": int, "codecstr": DataType, "disposition": int, "dispositionstr": DataType})


class _Plugin_bs_Core_Bound(Plugin):
    """This class implements the module definitions for the "bs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AudioSource(self, source: DataType, track: Optional[int] = None, adjustdelay: Optional[int] = None, threads: Optional[int] = None, enable_drefs: Optional[int] = None, use_absolute_path: Optional[int] = None, drc_scale: Optional[float] = None, cachemode: Optional[int] = None, cachepath: Optional[DataType] = None, cachesize: Optional[int] = None, showprogress: Optional[int] = None) -> 'AudioNode': ...
    def Metadata(self, source: DataType, track: Optional[int] = None, enable_drefs: Optional[int] = None, use_absolute_path: Optional[int] = None) -> 'VideoNode': ...
    def SetDebugOutput(self, enable: int) -> None: ...
    def SetFFmpegLogLevel(self, level: int) -> int: ...
    def TrackInfo(self, source: DataType, enable_drefs: Optional[int] = None, use_absolute_path: Optional[int] = None) -> '_ReturnDict_bs_TrackInfo': ...
    def VideoSource(self, source: DataType, track: Optional[int] = None, variableformat: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, rff: Optional[int] = None, threads: Optional[int] = None, seekpreroll: Optional[int] = None, enable_drefs: Optional[int] = None, use_absolute_path: Optional[int] = None, cachemode: Optional[int] = None, cachepath: Optional[DataType] = None, cachesize: Optional[int] = None, hwdevice: Optional[DataType] = None, extrahwframes: Optional[int] = None, timecodes: Optional[DataType] = None, start_number: Optional[int] = None, viewid: Optional[int] = None, showprogress: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: cs

class _Plugin_cs_Core_Bound(Plugin):
    """This class implements the module definitions for the "cs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ConvertColor(self, clip: 'VideoNode', output_profile: DataType, input_profile: Optional[DataType] = None, float_output: Optional[int] = None) -> 'VideoNode': ...
    def ImageSource(self, source: DataType, subsampling_pad: Optional[int] = None, jpeg_rgb: Optional[int] = None, jpeg_fancy_upsampling: Optional[int] = None, jpeg_cmyk_profile: Optional[DataType] = None, jpeg_cmyk_target_profile: Optional[DataType] = None) -> 'VideoNode': ...

class _Plugin_cs_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "cs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ConvertColor(self, output_profile: DataType, input_profile: Optional[DataType] = None, float_output: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: d2v

class _Plugin_d2v_Core_Bound(Plugin):
    """This class implements the module definitions for the "d2v" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Source(self, input: DataType, threads: Optional[int] = None, nocrop: Optional[int] = None, rff: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: descale

class _Plugin_descale_Core_Bound(Plugin):
    """This class implements the module definitions for the "descale" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, src: 'VideoNode', width: int, height: int, b: Optional[float] = None, c: Optional[float] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Bilinear(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Debicubic(self, src: 'VideoNode', width: int, height: int, b: Optional[float] = None, c: Optional[float] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Debilinear(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Decustom(self, src: 'VideoNode', width: int, height: int, custom_kernel: VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Delanczos(self, src: 'VideoNode', width: int, height: int, taps: Optional[int] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Despline16(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Despline36(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Despline64(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Lanczos(self, src: 'VideoNode', width: int, height: int, taps: Optional[int] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def ScaleCustom(self, src: 'VideoNode', width: int, height: int, custom_kernel: VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Spline16(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Spline36(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Spline64(self, src: 'VideoNode', width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_descale_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "descale" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: int, height: int, b: Optional[float] = None, c: Optional[float] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Bilinear(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Debicubic(self, width: int, height: int, b: Optional[float] = None, c: Optional[float] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Debilinear(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Decustom(self, width: int, height: int, custom_kernel: VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Delanczos(self, width: int, height: int, taps: Optional[int] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Despline16(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Despline36(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Despline64(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Lanczos(self, width: int, height: int, taps: Optional[int] = None, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def ScaleCustom(self, width: int, height: int, custom_kernel: VSMapValueCallback[_VapourSynthMapValue], taps: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Spline16(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Spline36(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...
    def Spline64(self, width: int, height: int, blur: Optional[float] = None, post_conv: Optional[SingleAndSequence[float]] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, border_handling: Optional[int] = None, ignore_mask: Optional['VideoNode'] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None, opt: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: dgdecodenv

class _Plugin_dgdecodenv_Core_Bound(Plugin):
    """This class implements the module definitions for the "dgdecodenv" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DGSource(self, source: DataType, i420: Optional[int] = None, deinterlace: Optional[int] = None, use_top_field: Optional[int] = None, use_pf: Optional[int] = None, ct: Optional[int] = None, cb: Optional[int] = None, cl: Optional[int] = None, cr: Optional[int] = None, rw: Optional[int] = None, rh: Optional[int] = None, fieldop: Optional[int] = None, show: Optional[int] = None, show2: Optional[DataType] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: dvdsrc2

class _Plugin_dvdsrc2_Core_Bound(Plugin):
    """This class implements the module definitions for the "dvdsrc2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def FullVts(self, path: DataType, vts: int, ranges: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def FullVtsAc3(self, path: DataType, vts: int, audio: int, ranges: Optional[SingleAndSequence[int]] = None) -> 'AudioNode': ...
    def FullVtsLpcm(self, path: DataType, vts: int, audio: int, ranges: Optional[SingleAndSequence[int]] = None) -> 'AudioNode': ...
    def Ifo(self, path: DataType, ifo: int) -> DataType: ...
    def RawAc3(self, path: DataType, vts: int, audio: int, ranges: Optional[SingleAndSequence[int]] = None) -> 'AudioNode': ...

# end implementation

    
# implementation: ffms2

class _Plugin_ffms2_Core_Bound(Plugin):
    """This class implements the module definitions for the "ffms2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def GetLogLevel(self) -> int: ...
    def Index(self, source: DataType, cachefile: Optional[DataType] = None, indextracks: Optional[SingleAndSequence[int]] = None, errorhandling: Optional[int] = None, overwrite: Optional[int] = None, enable_drefs: Optional[int] = None, use_absolute_path: Optional[int] = None) -> DataType: ...
    def SetLogLevel(self, level: int) -> int: ...
    def Source(self, source: DataType, track: Optional[int] = None, cache: Optional[int] = None, cachefile: Optional[DataType] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, threads: Optional[int] = None, timecodes: Optional[DataType] = None, seekmode: Optional[int] = None, width: Optional[int] = None, height: Optional[int] = None, resizer: Optional[DataType] = None, format: Optional[int] = None, alpha: Optional[int] = None) -> 'VideoNode': ...
    def Version(self) -> DataType: ...

# end implementation

    
# implementation: fmtc

class _Plugin_fmtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "fmtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def bitdepth(self, clip: 'VideoNode', csp: Optional[int] = None, bits: Optional[int] = None, flt: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, dmode: Optional[int] = None, ampo: Optional[float] = None, ampn: Optional[float] = None, dyn: Optional[int] = None, staticnoise: Optional[int] = None, cpuopt: Optional[int] = None, patsize: Optional[int] = None, tpdfo: Optional[int] = None, tpdfn: Optional[int] = None, corplane: Optional[int] = None) -> 'VideoNode': ...
    def histluma(self, clip: 'VideoNode', full: Optional[int] = None, amp: Optional[int] = None) -> 'VideoNode': ...
    def matrix(self, clip: 'VideoNode', mat: Optional[DataType] = None, mats: Optional[DataType] = None, matd: Optional[DataType] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, coef: Optional[SingleAndSequence[float]] = None, csp: Optional[int] = None, col_fam: Optional[int] = None, bits: Optional[int] = None, singleout: Optional[int] = None, cpuopt: Optional[int] = None, planes: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def matrix2020cl(self, clip: 'VideoNode', full: Optional[int] = None, csp: Optional[int] = None, bits: Optional[int] = None, cpuopt: Optional[int] = None) -> 'VideoNode': ...
    def nativetostack16(self, clip: 'VideoNode') -> 'VideoNode': ...
    def primaries(self, clip: 'VideoNode', rs: Optional[SingleAndSequence[float]] = None, gs: Optional[SingleAndSequence[float]] = None, bs: Optional[SingleAndSequence[float]] = None, ws: Optional[SingleAndSequence[float]] = None, rd: Optional[SingleAndSequence[float]] = None, gd: Optional[SingleAndSequence[float]] = None, bd: Optional[SingleAndSequence[float]] = None, wd: Optional[SingleAndSequence[float]] = None, prims: Optional[DataType] = None, primd: Optional[DataType] = None, wconv: Optional[int] = None, cpuopt: Optional[int] = None) -> 'VideoNode': ...
    def resample(self, clip: 'VideoNode', w: Optional[int] = None, h: Optional[int] = None, sx: Optional[SingleAndSequence[float]] = None, sy: Optional[SingleAndSequence[float]] = None, sw: Optional[SingleAndSequence[float]] = None, sh: Optional[SingleAndSequence[float]] = None, scale: Optional[float] = None, scaleh: Optional[float] = None, scalev: Optional[float] = None, kernel: Optional[SingleAndSequence[DataType]] = None, kernelh: Optional[SingleAndSequence[DataType]] = None, kernelv: Optional[SingleAndSequence[DataType]] = None, impulse: Optional[SingleAndSequence[float]] = None, impulseh: Optional[SingleAndSequence[float]] = None, impulsev: Optional[SingleAndSequence[float]] = None, taps: Optional[SingleAndSequence[int]] = None, tapsh: Optional[SingleAndSequence[int]] = None, tapsv: Optional[SingleAndSequence[int]] = None, a1: Optional[SingleAndSequence[float]] = None, a2: Optional[SingleAndSequence[float]] = None, a3: Optional[SingleAndSequence[float]] = None, a1h: Optional[SingleAndSequence[float]] = None, a2h: Optional[SingleAndSequence[float]] = None, a3h: Optional[SingleAndSequence[float]] = None, a1v: Optional[SingleAndSequence[float]] = None, a2v: Optional[SingleAndSequence[float]] = None, a3v: Optional[SingleAndSequence[float]] = None, kovrspl: Optional[SingleAndSequence[int]] = None, fh: Optional[SingleAndSequence[float]] = None, fv: Optional[SingleAndSequence[float]] = None, cnorm: Optional[SingleAndSequence[int]] = None, total: Optional[SingleAndSequence[float]] = None, totalh: Optional[SingleAndSequence[float]] = None, totalv: Optional[SingleAndSequence[float]] = None, invks: Optional[SingleAndSequence[int]] = None, invksh: Optional[SingleAndSequence[int]] = None, invksv: Optional[SingleAndSequence[int]] = None, invkstaps: Optional[SingleAndSequence[int]] = None, invkstapsh: Optional[SingleAndSequence[int]] = None, invkstapsv: Optional[SingleAndSequence[int]] = None, csp: Optional[int] = None, css: Optional[DataType] = None, planes: Optional[SingleAndSequence[float]] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, center: Optional[SingleAndSequence[int]] = None, cplace: Optional[DataType] = None, cplaces: Optional[DataType] = None, cplaced: Optional[DataType] = None, interlaced: Optional[int] = None, interlacedd: Optional[int] = None, tff: Optional[int] = None, tffd: Optional[int] = None, flt: Optional[int] = None, cpuopt: Optional[int] = None) -> 'VideoNode': ...
    def stack16tonative(self, clip: 'VideoNode') -> 'VideoNode': ...
    def transfer(self, clip: 'VideoNode', transs: Optional[SingleAndSequence[DataType]] = None, transd: Optional[SingleAndSequence[DataType]] = None, cont: Optional[float] = None, gcor: Optional[float] = None, bits: Optional[int] = None, flt: Optional[int] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, logceis: Optional[int] = None, logceid: Optional[int] = None, cpuopt: Optional[int] = None, blacklvl: Optional[float] = None, sceneref: Optional[int] = None, lb: Optional[float] = None, lw: Optional[float] = None, lws: Optional[float] = None, lwd: Optional[float] = None, ambient: Optional[float] = None, match: Optional[int] = None, gy: Optional[int] = None, debug: Optional[int] = None, sig_c: Optional[float] = None, sig_t: Optional[float] = None) -> 'VideoNode': ...

class _Plugin_fmtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "fmtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def bitdepth(self, csp: Optional[int] = None, bits: Optional[int] = None, flt: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, dmode: Optional[int] = None, ampo: Optional[float] = None, ampn: Optional[float] = None, dyn: Optional[int] = None, staticnoise: Optional[int] = None, cpuopt: Optional[int] = None, patsize: Optional[int] = None, tpdfo: Optional[int] = None, tpdfn: Optional[int] = None, corplane: Optional[int] = None) -> 'VideoNode': ...
    def histluma(self, full: Optional[int] = None, amp: Optional[int] = None) -> 'VideoNode': ...
    def matrix(self, mat: Optional[DataType] = None, mats: Optional[DataType] = None, matd: Optional[DataType] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, coef: Optional[SingleAndSequence[float]] = None, csp: Optional[int] = None, col_fam: Optional[int] = None, bits: Optional[int] = None, singleout: Optional[int] = None, cpuopt: Optional[int] = None, planes: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def matrix2020cl(self, full: Optional[int] = None, csp: Optional[int] = None, bits: Optional[int] = None, cpuopt: Optional[int] = None) -> 'VideoNode': ...
    def nativetostack16(self) -> 'VideoNode': ...
    def primaries(self, rs: Optional[SingleAndSequence[float]] = None, gs: Optional[SingleAndSequence[float]] = None, bs: Optional[SingleAndSequence[float]] = None, ws: Optional[SingleAndSequence[float]] = None, rd: Optional[SingleAndSequence[float]] = None, gd: Optional[SingleAndSequence[float]] = None, bd: Optional[SingleAndSequence[float]] = None, wd: Optional[SingleAndSequence[float]] = None, prims: Optional[DataType] = None, primd: Optional[DataType] = None, wconv: Optional[int] = None, cpuopt: Optional[int] = None) -> 'VideoNode': ...
    def resample(self, w: Optional[int] = None, h: Optional[int] = None, sx: Optional[SingleAndSequence[float]] = None, sy: Optional[SingleAndSequence[float]] = None, sw: Optional[SingleAndSequence[float]] = None, sh: Optional[SingleAndSequence[float]] = None, scale: Optional[float] = None, scaleh: Optional[float] = None, scalev: Optional[float] = None, kernel: Optional[SingleAndSequence[DataType]] = None, kernelh: Optional[SingleAndSequence[DataType]] = None, kernelv: Optional[SingleAndSequence[DataType]] = None, impulse: Optional[SingleAndSequence[float]] = None, impulseh: Optional[SingleAndSequence[float]] = None, impulsev: Optional[SingleAndSequence[float]] = None, taps: Optional[SingleAndSequence[int]] = None, tapsh: Optional[SingleAndSequence[int]] = None, tapsv: Optional[SingleAndSequence[int]] = None, a1: Optional[SingleAndSequence[float]] = None, a2: Optional[SingleAndSequence[float]] = None, a3: Optional[SingleAndSequence[float]] = None, a1h: Optional[SingleAndSequence[float]] = None, a2h: Optional[SingleAndSequence[float]] = None, a3h: Optional[SingleAndSequence[float]] = None, a1v: Optional[SingleAndSequence[float]] = None, a2v: Optional[SingleAndSequence[float]] = None, a3v: Optional[SingleAndSequence[float]] = None, kovrspl: Optional[SingleAndSequence[int]] = None, fh: Optional[SingleAndSequence[float]] = None, fv: Optional[SingleAndSequence[float]] = None, cnorm: Optional[SingleAndSequence[int]] = None, total: Optional[SingleAndSequence[float]] = None, totalh: Optional[SingleAndSequence[float]] = None, totalv: Optional[SingleAndSequence[float]] = None, invks: Optional[SingleAndSequence[int]] = None, invksh: Optional[SingleAndSequence[int]] = None, invksv: Optional[SingleAndSequence[int]] = None, invkstaps: Optional[SingleAndSequence[int]] = None, invkstapsh: Optional[SingleAndSequence[int]] = None, invkstapsv: Optional[SingleAndSequence[int]] = None, csp: Optional[int] = None, css: Optional[DataType] = None, planes: Optional[SingleAndSequence[float]] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, center: Optional[SingleAndSequence[int]] = None, cplace: Optional[DataType] = None, cplaces: Optional[DataType] = None, cplaced: Optional[DataType] = None, interlaced: Optional[int] = None, interlacedd: Optional[int] = None, tff: Optional[int] = None, tffd: Optional[int] = None, flt: Optional[int] = None, cpuopt: Optional[int] = None) -> 'VideoNode': ...
    def stack16tonative(self) -> 'VideoNode': ...
    def transfer(self, transs: Optional[SingleAndSequence[DataType]] = None, transd: Optional[SingleAndSequence[DataType]] = None, cont: Optional[float] = None, gcor: Optional[float] = None, bits: Optional[int] = None, flt: Optional[int] = None, fulls: Optional[int] = None, fulld: Optional[int] = None, logceis: Optional[int] = None, logceid: Optional[int] = None, cpuopt: Optional[int] = None, blacklvl: Optional[float] = None, sceneref: Optional[int] = None, lb: Optional[float] = None, lw: Optional[float] = None, lws: Optional[float] = None, lwd: Optional[float] = None, ambient: Optional[float] = None, match: Optional[int] = None, gy: Optional[int] = None, debug: Optional[int] = None, sig_c: Optional[float] = None, sig_t: Optional[float] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: imwri

class _Plugin_imwri_Core_Bound(Plugin):
    """This class implements the module definitions for the "imwri" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Read(self, filename: SingleAndSequence[DataType], firstnum: Optional[int] = None, mismatch: Optional[int] = None, alpha: Optional[int] = None, float_output: Optional[int] = None, embed_icc: Optional[int] = None) -> 'VideoNode': ...
    def Write(self, clip: 'VideoNode', imgformat: DataType, filename: DataType, firstnum: Optional[int] = None, quality: Optional[int] = None, dither: Optional[int] = None, compression_type: Optional[DataType] = None, overwrite: Optional[int] = None, alpha: Optional['VideoNode'] = None) -> 'VideoNode': ...

class _Plugin_imwri_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "imwri" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Write(self, imgformat: DataType, filename: DataType, firstnum: Optional[int] = None, quality: Optional[int] = None, dither: Optional[int] = None, compression_type: Optional[DataType] = None, overwrite: Optional[int] = None, alpha: Optional['VideoNode'] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: lsmas

class _Plugin_lsmas_Core_Bound(Plugin):
    """This class implements the module definitions for the "lsmas" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def LibavSMASHSource(self, source: DataType, track: Optional[int] = None, threads: Optional[int] = None, seek_mode: Optional[int] = None, seek_threshold: Optional[int] = None, dr: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, variable: Optional[int] = None, format: Optional[DataType] = None, decoder: Optional[DataType] = None, prefer_hw: Optional[int] = None, ff_loglevel: Optional[int] = None, ff_options: Optional[DataType] = None) -> 'VideoNode': ...
    def LWLibavSource(self, source: DataType, stream_index: Optional[int] = None, cache: Optional[int] = None, cachefile: Optional[DataType] = None, threads: Optional[int] = None, seek_mode: Optional[int] = None, seek_threshold: Optional[int] = None, dr: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, variable: Optional[int] = None, format: Optional[DataType] = None, decoder: Optional[DataType] = None, prefer_hw: Optional[int] = None, repeat: Optional[int] = None, dominance: Optional[int] = None, ff_loglevel: Optional[int] = None, cachedir: Optional[DataType] = None, ff_options: Optional[DataType] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: placebo

class _Plugin_placebo_Core_Bound(Plugin):
    """This class implements the module definitions for the "placebo" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(self, clip: 'VideoNode', planes: Optional[int] = None, iterations: Optional[int] = None, threshold: Optional[float] = None, radius: Optional[float] = None, grain: Optional[float] = None, dither: Optional[int] = None, dither_algo: Optional[int] = None, log_level: Optional[int] = None) -> 'VideoNode': ...
    def Resample(self, clip: 'VideoNode', width: int, height: int, filter: Optional[DataType] = None, clamp: Optional[float] = None, blur: Optional[float] = None, taper: Optional[float] = None, radius: Optional[float] = None, param1: Optional[float] = None, param2: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, sx: Optional[float] = None, sy: Optional[float] = None, antiring: Optional[float] = None, sigmoidize: Optional[int] = None, sigmoid_center: Optional[float] = None, sigmoid_slope: Optional[float] = None, linearize: Optional[int] = None, trc: Optional[int] = None, min_luma: Optional[float] = None, log_level: Optional[int] = None) -> 'VideoNode': ...
    def Shader(self, clip: 'VideoNode', shader: Optional[DataType] = None, width: Optional[int] = None, height: Optional[int] = None, chroma_loc: Optional[int] = None, matrix: Optional[int] = None, trc: Optional[int] = None, linearize: Optional[int] = None, sigmoidize: Optional[int] = None, sigmoid_center: Optional[float] = None, sigmoid_slope: Optional[float] = None, antiring: Optional[float] = None, filter: Optional[DataType] = None, clamp: Optional[float] = None, blur: Optional[float] = None, taper: Optional[float] = None, radius: Optional[float] = None, param1: Optional[float] = None, param2: Optional[float] = None, shader_s: Optional[DataType] = None, log_level: Optional[int] = None) -> 'VideoNode': ...
    def Tonemap(self, clip: 'VideoNode', src_csp: Optional[int] = None, dst_csp: Optional[int] = None, dst_prim: Optional[int] = None, src_max: Optional[float] = None, src_min: Optional[float] = None, dst_max: Optional[float] = None, dst_min: Optional[float] = None, dynamic_peak_detection: Optional[int] = None, smoothing_period: Optional[float] = None, scene_threshold_low: Optional[float] = None, scene_threshold_high: Optional[float] = None, percentile: Optional[float] = None, gamut_mapping: Optional[int] = None, tone_mapping_function: Optional[int] = None, tone_mapping_function_s: Optional[DataType] = None, tone_mapping_param: Optional[float] = None, metadata: Optional[int] = None, use_dovi: Optional[int] = None, visualize_lut: Optional[int] = None, show_clipping: Optional[int] = None, contrast_recovery: Optional[float] = None, log_level: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_placebo_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "placebo" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(self, planes: Optional[int] = None, iterations: Optional[int] = None, threshold: Optional[float] = None, radius: Optional[float] = None, grain: Optional[float] = None, dither: Optional[int] = None, dither_algo: Optional[int] = None, log_level: Optional[int] = None) -> 'VideoNode': ...
    def Resample(self, width: int, height: int, filter: Optional[DataType] = None, clamp: Optional[float] = None, blur: Optional[float] = None, taper: Optional[float] = None, radius: Optional[float] = None, param1: Optional[float] = None, param2: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, sx: Optional[float] = None, sy: Optional[float] = None, antiring: Optional[float] = None, sigmoidize: Optional[int] = None, sigmoid_center: Optional[float] = None, sigmoid_slope: Optional[float] = None, linearize: Optional[int] = None, trc: Optional[int] = None, min_luma: Optional[float] = None, log_level: Optional[int] = None) -> 'VideoNode': ...
    def Shader(self, shader: Optional[DataType] = None, width: Optional[int] = None, height: Optional[int] = None, chroma_loc: Optional[int] = None, matrix: Optional[int] = None, trc: Optional[int] = None, linearize: Optional[int] = None, sigmoidize: Optional[int] = None, sigmoid_center: Optional[float] = None, sigmoid_slope: Optional[float] = None, antiring: Optional[float] = None, filter: Optional[DataType] = None, clamp: Optional[float] = None, blur: Optional[float] = None, taper: Optional[float] = None, radius: Optional[float] = None, param1: Optional[float] = None, param2: Optional[float] = None, shader_s: Optional[DataType] = None, log_level: Optional[int] = None) -> 'VideoNode': ...
    def Tonemap(self, src_csp: Optional[int] = None, dst_csp: Optional[int] = None, dst_prim: Optional[int] = None, src_max: Optional[float] = None, src_min: Optional[float] = None, dst_max: Optional[float] = None, dst_min: Optional[float] = None, dynamic_peak_detection: Optional[int] = None, smoothing_period: Optional[float] = None, scene_threshold_low: Optional[float] = None, scene_threshold_high: Optional[float] = None, percentile: Optional[float] = None, gamut_mapping: Optional[int] = None, tone_mapping_function: Optional[int] = None, tone_mapping_function_s: Optional[DataType] = None, tone_mapping_param: Optional[float] = None, metadata: Optional[int] = None, use_dovi: Optional[int] = None, visualize_lut: Optional[int] = None, show_clipping: Optional[int] = None, contrast_recovery: Optional[float] = None, log_level: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: resize

class _Plugin_resize_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Bilinear(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Bob(self, clip: 'VideoNode', filter: Optional[DataType] = None, tff: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Lanczos(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Point(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Spline16(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Spline36(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Spline64(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_resize_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Bilinear(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Bob(self, filter: Optional[DataType] = None, tff: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Lanczos(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Point(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Spline16(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Spline36(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...
    def Spline64(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, approximate_gamma: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: resize2

class _Plugin_resize2_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Bilinear(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Bob(self, clip: 'VideoNode', filter: Optional[DataType] = None, tff: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None) -> 'VideoNode': ...
    def Custom(self, clip: 'VideoNode', custom_kernel: VSMapValueCallback[_VapourSynthMapValue], taps: int, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Lanczos(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Point(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Spline16(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Spline36(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Spline64(self, clip: 'VideoNode', width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_resize2_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Bilinear(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Bob(self, filter: Optional[DataType] = None, tff: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None) -> 'VideoNode': ...
    def Custom(self, custom_kernel: VSMapValueCallback[_VapourSynthMapValue], taps: int, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Lanczos(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Point(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Spline16(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Spline36(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...
    def Spline64(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, matrix: Optional[int] = None, matrix_s: Optional[DataType] = None, transfer: Optional[int] = None, transfer_s: Optional[DataType] = None, primaries: Optional[int] = None, primaries_s: Optional[DataType] = None, range: Optional[int] = None, range_s: Optional[DataType] = None, chromaloc: Optional[int] = None, chromaloc_s: Optional[DataType] = None, matrix_in: Optional[int] = None, matrix_in_s: Optional[DataType] = None, transfer_in: Optional[int] = None, transfer_in_s: Optional[DataType] = None, primaries_in: Optional[int] = None, primaries_in_s: Optional[DataType] = None, range_in: Optional[int] = None, range_in_s: Optional[DataType] = None, chromaloc_in: Optional[int] = None, chromaloc_in_s: Optional[DataType] = None, filter_param_a: Optional[float] = None, filter_param_b: Optional[float] = None, resample_filter_uv: Optional[DataType] = None, filter_param_a_uv: Optional[float] = None, filter_param_b_uv: Optional[float] = None, dither_type: Optional[DataType] = None, cpu_type: Optional[DataType] = None, prefer_props: Optional[int] = None, src_left: Optional[float] = None, src_top: Optional[float] = None, src_width: Optional[float] = None, src_height: Optional[float] = None, nominal_luminance: Optional[float] = None, force: Optional[int] = None, force_h: Optional[int] = None, force_v: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: scxvid

class _Plugin_scxvid_Core_Bound(Plugin):
    """This class implements the module definitions for the "scxvid" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Scxvid(self, clip: 'VideoNode', log: Optional[DataType] = None, use_slices: Optional[int] = None) -> 'VideoNode': ...

class _Plugin_scxvid_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "scxvid" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Scxvid(self, log: Optional[DataType] = None, use_slices: Optional[int] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: std

class _Plugin_std_Core_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(self, clip: 'VideoNode', left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def AssumeFPS(self, clip: 'VideoNode', src: Optional['VideoNode'] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None) -> 'VideoNode': ...
    def AssumeSampleRate(self, clip: 'AudioNode', src: Optional['AudioNode'] = None, samplerate: Optional[int] = None) -> 'AudioNode': ...
    def AudioGain(self, clip: 'AudioNode', gain: Optional[SingleAndSequence[float]] = None, overflow_error: Optional[int] = None) -> 'AudioNode': ...
    def AudioLoop(self, clip: 'AudioNode', times: Optional[int] = None) -> 'AudioNode': ...
    def AudioMix(self, clips: SingleAndSequence['AudioNode'], matrix: SingleAndSequence[float], channels_out: SingleAndSequence[int], overflow_error: Optional[int] = None) -> 'AudioNode': ...
    def AudioReverse(self, clip: 'AudioNode') -> 'AudioNode': ...
    def AudioSplice(self, clips: SingleAndSequence['AudioNode']) -> 'AudioNode': ...
    def AudioTrim(self, clip: 'AudioNode', first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'AudioNode': ...
    def AverageFrames(self, clips: SingleAndSequence['VideoNode'], weights: SingleAndSequence[float], scale: Optional[float] = None, scenechange: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Binarize(self, clip: 'VideoNode', threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BinarizeMask(self, clip: 'VideoNode', threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BlankAudio(self, clip: Optional['AudioNode'] = None, channels: Optional[SingleAndSequence[int]] = None, bits: Optional[int] = None, sampletype: Optional[int] = None, samplerate: Optional[int] = None, length: Optional[int] = None, keep: Optional[int] = None) -> 'AudioNode': ...
    def BlankClip(self, clip: Optional['VideoNode'] = None, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, length: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None, keep: Optional[int] = None, varsize: Optional[int] = None, varformat: Optional[int] = None) -> 'VideoNode': ...
    def BoxBlur(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, hradius: Optional[int] = None, hpasses: Optional[int] = None, vradius: Optional[int] = None, vpasses: Optional[int] = None) -> 'VideoNode': ...
    def Cache(self, clip: 'VideoNode', size: Optional[int] = None, fixed: Optional[int] = None, make_linear: Optional[int] = None) -> 'VideoNode': ...
    def ClipToProp(self, clip: 'VideoNode', mclip: 'VideoNode', prop: Optional[DataType] = None) -> 'VideoNode': ...
    def Convolution(self, clip: 'VideoNode', matrix: SingleAndSequence[float], bias: Optional[float] = None, divisor: Optional[float] = None, planes: Optional[SingleAndSequence[int]] = None, saturate: Optional[int] = None, mode: Optional[DataType] = None) -> 'VideoNode': ...
    def CopyFrameProps(self, clip: 'VideoNode', prop_src: 'VideoNode', props: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def Crop(self, clip: 'VideoNode', left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def CropAbs(self, clip: 'VideoNode', width: int, height: int, left: Optional[int] = None, top: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None) -> 'VideoNode': ...
    def CropRel(self, clip: 'VideoNode', left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def Deflate(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def DeleteFrames(self, clip: 'VideoNode', frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def DoubleWeave(self, clip: 'VideoNode', tff: Optional[int] = None) -> 'VideoNode': ...
    def DuplicateFrames(self, clip: 'VideoNode', frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def Expr(self, clips: SingleAndSequence['VideoNode'], expr: SingleAndSequence[DataType], format: Optional[int] = None) -> 'VideoNode': ...
    def FlipHorizontal(self, clip: 'VideoNode') -> 'VideoNode': ...
    def FlipVertical(self, clip: 'VideoNode') -> 'VideoNode': ...
    def FrameEval(self, clip: 'VideoNode', eval: VSMapValueCallback[_VapourSynthMapValue], prop_src: Optional[SingleAndSequence[VideoNode]] = None, clip_src: Optional[SingleAndSequence[VideoNode]] = None) -> 'VideoNode': ...
    def FreezeFrames(self, clip: 'VideoNode', first: Optional[SingleAndSequence[int]] = None, last: Optional[SingleAndSequence[int]] = None, replacement: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Inflate(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def Interleave(self, clips: SingleAndSequence['VideoNode'], extend: Optional[int] = None, mismatch: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def Invert(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def InvertMask(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Levels(self, clip: 'VideoNode', min_in: Optional[SingleAndSequence[float]] = None, max_in: Optional[SingleAndSequence[float]] = None, gamma: Optional[SingleAndSequence[float]] = None, min_out: Optional[SingleAndSequence[float]] = None, max_out: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Limiter(self, clip: 'VideoNode', min: Optional[SingleAndSequence[float]] = None, max: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def LoadAllPlugins(self, path: DataType) -> None: ...
    def LoadPlugin(self, path: DataType, altsearchpath: Optional[int] = None, forcens: Optional[DataType] = None, forceid: Optional[DataType] = None) -> None: ...
    def Loop(self, clip: 'VideoNode', times: Optional[int] = None) -> 'VideoNode': ...
    def Lut(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[_VapourSynthMapValue]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def Lut2(self, clipa: 'VideoNode', clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[_VapourSynthMapValue]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def MakeDiff(self, clipa: 'VideoNode', clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def MakeFullDiff(self, clipa: 'VideoNode', clipb: 'VideoNode') -> 'VideoNode': ...
    def MaskedMerge(self, clipa: 'VideoNode', clipb: 'VideoNode', mask: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, first_plane: Optional[int] = None, premultiplied: Optional[int] = None) -> 'VideoNode': ...
    def Maximum(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Median(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Merge(self, clipa: 'VideoNode', clipb: 'VideoNode', weight: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def MergeDiff(self, clipa: 'VideoNode', clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def MergeFullDiff(self, clipa: 'VideoNode', clipb: 'VideoNode') -> 'VideoNode': ...
    def Minimum(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def ModifyFrame(self, clip: 'VideoNode', clips: SingleAndSequence['VideoNode'], selector: VSMapValueCallback[_VapourSynthMapValue]) -> 'VideoNode': ...
    def PEMVerifier(self, clip: 'VideoNode', upper: Optional[SingleAndSequence[float]] = None, lower: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def PlaneStats(self, clipa: 'VideoNode', clipb: Optional['VideoNode'] = None, plane: Optional[int] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def PreMultiply(self, clip: 'VideoNode', alpha: 'VideoNode') -> 'VideoNode': ...
    def Prewitt(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def PropToClip(self, clip: 'VideoNode', prop: Optional[DataType] = None) -> 'VideoNode': ...
    def RemoveFrameProps(self, clip: 'VideoNode', props: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def Reverse(self, clip: 'VideoNode') -> 'VideoNode': ...
    def SelectEvery(self, clip: 'VideoNode', cycle: int, offsets: SingleAndSequence[int], modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SeparateFields(self, clip: 'VideoNode', tff: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SetAudioCache(self, clip: 'AudioNode', mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def SetFieldBased(self, clip: 'VideoNode', value: int) -> 'VideoNode': ...
    def SetFrameProp(self, clip: 'VideoNode', prop: DataType, intval: Optional[SingleAndSequence[int]] = None, floatval: Optional[SingleAndSequence[float]] = None, data: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def SetFrameProps(self, clip: 'VideoNode', **kwargs: _VapourSynthMapValue) -> 'VideoNode': ...
    def SetMaxCPU(self, cpu: DataType) -> DataType: ...
    def SetVideoCache(self, clip: 'VideoNode', mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def ShuffleChannels(self, clips: SingleAndSequence['AudioNode'], channels_in: SingleAndSequence[int], channels_out: SingleAndSequence[int]) -> 'AudioNode': ...
    def ShufflePlanes(self, clips: SingleAndSequence['VideoNode'], planes: SingleAndSequence[int], colorfamily: int, prop_src: Optional['VideoNode'] = None) -> 'VideoNode': ...
    def Sobel(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def Splice(self, clips: SingleAndSequence['VideoNode'], mismatch: Optional[int] = None) -> 'VideoNode': ...
    def SplitChannels(self, clip: 'AudioNode') -> SingleAndSequence['AudioNode']: ...
    def SplitPlanes(self, clip: 'VideoNode') -> SingleAndSequence['VideoNode']: ...
    def StackHorizontal(self, clips: SingleAndSequence['VideoNode']) -> 'VideoNode': ...
    def StackVertical(self, clips: SingleAndSequence['VideoNode']) -> 'VideoNode': ...
    def TestAudio(self, channels: Optional[SingleAndSequence[int]] = None, bits: Optional[int] = None, isfloat: Optional[int] = None, samplerate: Optional[int] = None, length: Optional[int] = None) -> 'AudioNode': ...
    def Transpose(self, clip: 'VideoNode') -> 'VideoNode': ...
    def Trim(self, clip: 'VideoNode', first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'VideoNode': ...
    def Turn180(self, clip: 'VideoNode') -> 'VideoNode': ...

class _Plugin_std_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(self, left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def AssumeFPS(self, src: Optional['VideoNode'] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None) -> 'VideoNode': ...
    def AverageFrames(self, weights: SingleAndSequence[float], scale: Optional[float] = None, scenechange: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Binarize(self, threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BinarizeMask(self, threshold: Optional[SingleAndSequence[float]] = None, v0: Optional[SingleAndSequence[float]] = None, v1: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BlankClip(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[int] = None, length: Optional[int] = None, fpsnum: Optional[int] = None, fpsden: Optional[int] = None, color: Optional[SingleAndSequence[float]] = None, keep: Optional[int] = None, varsize: Optional[int] = None, varformat: Optional[int] = None) -> 'VideoNode': ...
    def BoxBlur(self, planes: Optional[SingleAndSequence[int]] = None, hradius: Optional[int] = None, hpasses: Optional[int] = None, vradius: Optional[int] = None, vpasses: Optional[int] = None) -> 'VideoNode': ...
    def Cache(self, size: Optional[int] = None, fixed: Optional[int] = None, make_linear: Optional[int] = None) -> 'VideoNode': ...
    def ClipToProp(self, mclip: 'VideoNode', prop: Optional[DataType] = None) -> 'VideoNode': ...
    def Convolution(self, matrix: SingleAndSequence[float], bias: Optional[float] = None, divisor: Optional[float] = None, planes: Optional[SingleAndSequence[int]] = None, saturate: Optional[int] = None, mode: Optional[DataType] = None) -> 'VideoNode': ...
    def CopyFrameProps(self, prop_src: 'VideoNode', props: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def Crop(self, left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def CropAbs(self, width: int, height: int, left: Optional[int] = None, top: Optional[int] = None, x: Optional[int] = None, y: Optional[int] = None) -> 'VideoNode': ...
    def CropRel(self, left: Optional[int] = None, right: Optional[int] = None, top: Optional[int] = None, bottom: Optional[int] = None) -> 'VideoNode': ...
    def Deflate(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def DeleteFrames(self, frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def DoubleWeave(self, tff: Optional[int] = None) -> 'VideoNode': ...
    def DuplicateFrames(self, frames: SingleAndSequence[int]) -> 'VideoNode': ...
    def Expr(self, expr: SingleAndSequence[DataType], format: Optional[int] = None) -> 'VideoNode': ...
    def FlipHorizontal(self) -> 'VideoNode': ...
    def FlipVertical(self) -> 'VideoNode': ...
    def FrameEval(self, eval: VSMapValueCallback[_VapourSynthMapValue], prop_src: Optional[SingleAndSequence[VideoNode]] = None, clip_src: Optional[SingleAndSequence[VideoNode]] = None) -> 'VideoNode': ...
    def FreezeFrames(self, first: Optional[SingleAndSequence[int]] = None, last: Optional[SingleAndSequence[int]] = None, replacement: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Inflate(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None) -> 'VideoNode': ...
    def Interleave(self, extend: Optional[int] = None, mismatch: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def Invert(self, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def InvertMask(self, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Levels(self, min_in: Optional[SingleAndSequence[float]] = None, max_in: Optional[SingleAndSequence[float]] = None, gamma: Optional[SingleAndSequence[float]] = None, min_out: Optional[SingleAndSequence[float]] = None, max_out: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Limiter(self, min: Optional[SingleAndSequence[float]] = None, max: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Loop(self, times: Optional[int] = None) -> 'VideoNode': ...
    def Lut(self, planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[_VapourSynthMapValue]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def Lut2(self, clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, lut: Optional[SingleAndSequence[int]] = None, lutf: Optional[SingleAndSequence[float]] = None, function: Optional[VSMapValueCallback[_VapourSynthMapValue]] = None, bits: Optional[int] = None, floatout: Optional[int] = None) -> 'VideoNode': ...
    def MakeDiff(self, clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def MakeFullDiff(self, clipb: 'VideoNode') -> 'VideoNode': ...
    def MaskedMerge(self, clipb: 'VideoNode', mask: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, first_plane: Optional[int] = None, premultiplied: Optional[int] = None) -> 'VideoNode': ...
    def Maximum(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Median(self, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def Merge(self, clipb: 'VideoNode', weight: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def MergeDiff(self, clipb: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def MergeFullDiff(self, clipb: 'VideoNode') -> 'VideoNode': ...
    def Minimum(self, planes: Optional[SingleAndSequence[int]] = None, threshold: Optional[float] = None, coordinates: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def ModifyFrame(self, clips: SingleAndSequence['VideoNode'], selector: VSMapValueCallback[_VapourSynthMapValue]) -> 'VideoNode': ...
    def PEMVerifier(self, upper: Optional[SingleAndSequence[float]] = None, lower: Optional[SingleAndSequence[float]] = None) -> 'VideoNode': ...
    def PlaneStats(self, clipb: Optional['VideoNode'] = None, plane: Optional[int] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def PreMultiply(self, alpha: 'VideoNode') -> 'VideoNode': ...
    def Prewitt(self, planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def PropToClip(self, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def RemoveFrameProps(self, props: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def Reverse(self) -> 'VideoNode': ...
    def SelectEvery(self, cycle: int, offsets: SingleAndSequence[int], modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SeparateFields(self, tff: Optional[int] = None, modify_duration: Optional[int] = None) -> 'VideoNode': ...
    def SetFieldBased(self, value: int) -> 'VideoNode': ...
    def SetFrameProp(self, prop: DataType, intval: Optional[SingleAndSequence[int]] = None, floatval: Optional[SingleAndSequence[float]] = None, data: Optional[SingleAndSequence[DataType]] = None) -> 'VideoNode': ...
    def SetFrameProps(self, **kwargs: Any) -> 'VideoNode': ...
    def SetVideoCache(self, mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def ShufflePlanes(self, planes: SingleAndSequence[int], colorfamily: int, prop_src: Optional['VideoNode'] = None) -> 'VideoNode': ...
    def Sobel(self, planes: Optional[SingleAndSequence[int]] = None, scale: Optional[float] = None) -> 'VideoNode': ...
    def Splice(self, mismatch: Optional[int] = None) -> 'VideoNode': ...
    def SplitPlanes(self) -> SingleAndSequence['VideoNode']: ...
    def StackHorizontal(self) -> 'VideoNode': ...
    def StackVertical(self) -> 'VideoNode': ...
    def Transpose(self) -> 'VideoNode': ...
    def Trim(self, first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'VideoNode': ...
    def Turn180(self) -> 'VideoNode': ...

class _Plugin_std_AudioNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AssumeSampleRate(self, src: Optional['AudioNode'] = None, samplerate: Optional[int] = None) -> 'AudioNode': ...
    def AudioGain(self, gain: Optional[SingleAndSequence[float]] = None, overflow_error: Optional[int] = None) -> 'AudioNode': ...
    def AudioLoop(self, times: Optional[int] = None) -> 'AudioNode': ...
    def AudioMix(self, matrix: SingleAndSequence[float], channels_out: SingleAndSequence[int], overflow_error: Optional[int] = None) -> 'AudioNode': ...
    def AudioReverse(self) -> 'AudioNode': ...
    def AudioSplice(self) -> 'AudioNode': ...
    def AudioTrim(self, first: Optional[int] = None, last: Optional[int] = None, length: Optional[int] = None) -> 'AudioNode': ...
    def BlankAudio(self, channels: Optional[SingleAndSequence[int]] = None, bits: Optional[int] = None, sampletype: Optional[int] = None, samplerate: Optional[int] = None, length: Optional[int] = None, keep: Optional[int] = None) -> 'AudioNode': ...
    def SetAudioCache(self, mode: Optional[int] = None, fixedsize: Optional[int] = None, maxsize: Optional[int] = None, maxhistory: Optional[int] = None) -> None: ...
    def ShuffleChannels(self, channels_in: SingleAndSequence[int], channels_out: SingleAndSequence[int]) -> 'AudioNode': ...
    def SplitChannels(self) -> SingleAndSequence['AudioNode']: ...

# end implementation

    
# implementation: vszip

class _Plugin_vszip_Core_Bound(Plugin):
    """This class implements the module definitions for the "vszip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AdaptiveBinarize(self, clip: 'VideoNode', clip2: 'VideoNode', c: Optional[int] = None) -> 'VideoNode': ...
    def Bilateral(self, clip: 'VideoNode', ref: Optional['VideoNode'] = None, sigmaS: Optional[SingleAndSequence[float]] = None, sigmaR: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None, algorithm: Optional[SingleAndSequence[int]] = None, PBFICnum: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BoxBlur(self, clip: 'VideoNode', planes: Optional[SingleAndSequence[int]] = None, hradius: Optional[int] = None, hpasses: Optional[int] = None, vradius: Optional[int] = None, vpasses: Optional[int] = None) -> 'VideoNode': ...
    def Checkmate(self, clip: 'VideoNode', thr: Optional[int] = None, tmax: Optional[int] = None, tthr2: Optional[int] = None) -> 'VideoNode': ...
    def CLAHE(self, clip: 'VideoNode', limit: Optional[int] = None, tiles: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def CombMaskMT(self, clip: 'VideoNode', thY1: Optional[int] = None, thY2: Optional[int] = None) -> 'VideoNode': ...
    def Limiter(self, clip: 'VideoNode', min: Optional[SingleAndSequence[float]] = None, max: Optional[SingleAndSequence[float]] = None, tv_range: Optional[int] = None) -> 'VideoNode': ...
    def Metrics(self, reference: 'VideoNode', distorted: 'VideoNode', mode: Optional[int] = None) -> 'VideoNode': ...
    def PlaneAverage(self, clipa: 'VideoNode', exclude: SingleAndSequence[int], clipb: Optional['VideoNode'] = None, planes: Optional[SingleAndSequence[int]] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def PlaneMinMax(self, clipa: 'VideoNode', minthr: Optional[float] = None, maxthr: Optional[float] = None, clipb: Optional['VideoNode'] = None, planes: Optional[SingleAndSequence[int]] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def RFS(self, clipa: 'VideoNode', clipb: 'VideoNode', frames: SingleAndSequence[int], mismatch: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...

class _Plugin_vszip_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "vszip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AdaptiveBinarize(self, clip2: 'VideoNode', c: Optional[int] = None) -> 'VideoNode': ...
    def Bilateral(self, ref: Optional['VideoNode'] = None, sigmaS: Optional[SingleAndSequence[float]] = None, sigmaR: Optional[SingleAndSequence[float]] = None, planes: Optional[SingleAndSequence[int]] = None, algorithm: Optional[SingleAndSequence[int]] = None, PBFICnum: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def BoxBlur(self, planes: Optional[SingleAndSequence[int]] = None, hradius: Optional[int] = None, hpasses: Optional[int] = None, vradius: Optional[int] = None, vpasses: Optional[int] = None) -> 'VideoNode': ...
    def Checkmate(self, thr: Optional[int] = None, tmax: Optional[int] = None, tthr2: Optional[int] = None) -> 'VideoNode': ...
    def CLAHE(self, limit: Optional[int] = None, tiles: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...
    def CombMaskMT(self, thY1: Optional[int] = None, thY2: Optional[int] = None) -> 'VideoNode': ...
    def Limiter(self, min: Optional[SingleAndSequence[float]] = None, max: Optional[SingleAndSequence[float]] = None, tv_range: Optional[int] = None) -> 'VideoNode': ...
    def Metrics(self, distorted: 'VideoNode', mode: Optional[int] = None) -> 'VideoNode': ...
    def PlaneAverage(self, exclude: SingleAndSequence[int], clipb: Optional['VideoNode'] = None, planes: Optional[SingleAndSequence[int]] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def PlaneMinMax(self, minthr: Optional[float] = None, maxthr: Optional[float] = None, clipb: Optional['VideoNode'] = None, planes: Optional[SingleAndSequence[int]] = None, prop: Optional[DataType] = None) -> 'VideoNode': ...
    def RFS(self, clipb: 'VideoNode', frames: SingleAndSequence[int], mismatch: Optional[int] = None, planes: Optional[SingleAndSequence[int]] = None) -> 'VideoNode': ...

# end implementation

    
# implementation: wwxd

class _Plugin_wwxd_Core_Bound(Plugin):
    """This class implements the module definitions for the "wwxd" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def WWXD(self, clip: 'VideoNode') -> 'VideoNode': ...

class _Plugin_wwxd_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "wwxd" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def WWXD(self) -> 'VideoNode': ...

# end implementation



class RawNode:
    def __init__(self) -> None: ...

    def get_frame(self, n: int) -> RawFrame: ...

    @overload
    def get_frame_async(self, n: int, cb: None = None) -> _Future[RawFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[Union[RawFrame, None], Union[Exception, None]], None]) -> None: ...

    def frames(
        self, prefetch: Union[int, None] = None, backlog: Union[int, None] = None, close: bool = False
    ) -> Iterator[RawFrame]: ...

    def clear_cache(self) -> None: ...

    def set_output(self, index: int = 0) -> None: ...

    def is_inspectable(self, version: Union[int, None] = None) -> bool: ...

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

    def __getitem__(self: 'SelfRawNode', index: Union[int, slice], /) -> 'SelfRawNode': ...

    def __len__(self) -> int: ...


SelfRawNode = TypeVar('SelfRawNode', bound=RawNode)


class VideoNode(RawNode):
    format: Union[VideoFormat, None]

    width: int
    height: int

    fps_num: int
    fps_den: int

    fps: Fraction

    num_frames: int

    def set_output(
        self, index: int = 0, alpha: Union['VideoNode', None] = None, alt_output: Literal[0, 1, 2] = 0
    ) -> None: ...

    def output(
        self, fileobj: BinaryIO, y4m: bool = False, progress_update: Callable[[int, int], None] | None = None,
        prefetch: int = 0, backlog: int = -1
    ) -> None: ...

    def get_frame(self, n: int) -> VideoFrame: ...

    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[VideoFrame]: ...

    @overload
    def get_frame_async(self, n: int, cb: Callable[[Union[VideoFrame, None], Union[Exception, None]], None]) -> None: ...

    def frames(
        self, prefetch: Union[int, None] = None, backlog: Union[int, None] = None, close: bool = False
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
    def get_frame_async(self, n: int, cb: Callable[[Union[AudioFrame, None], Union[Exception, None]], None]) -> None: ...

    def frames(
        self, prefetch: Union[int, None] = None, backlog: Union[int, None] = None, close: bool = False
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

    def get_video_format(self, id: Union[VideoFormat, int, PresetVideoFormat]) -> VideoFormat: ...

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


class _CoreProxy(Core):
    @property
    def core(self) -> Core: ...


core: _CoreProxy
