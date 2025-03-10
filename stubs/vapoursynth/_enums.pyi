from enum import IntEnum, IntFlag
from typing import Literal, cast


__all__ = [
    'MessageType',
        'MESSAGE_TYPE_DEBUG', 'MESSAGE_TYPE_INFORMATION', 'MESSAGE_TYPE_WARNING',
        'MESSAGE_TYPE_CRITICAL', 'MESSAGE_TYPE_FATAL',

    'FilterMode',
        'PARALLEL', 'PARALLEL_REQUESTS', 'UNORDERED', 'FRAME_STATE',

    'CoreCreationFlags',
        'ENABLE_GRAPH_INSPECTION', 'DISABLE_AUTO_LOADING', 'DISABLE_LIBRARY_UNLOADING',

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
]


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