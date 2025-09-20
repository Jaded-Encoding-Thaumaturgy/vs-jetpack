from __future__ import annotations

from enum import IntEnum

from vapoursynth import FLOAT, GRAY, INTEGER, RGB, YUV
from vapoursynth import PresetVideoFormat as VSPresetVideoFormat

__all__ = [
    "GRAY8",
    "GRAY9",
    "GRAY10",
    "GRAY11",
    "GRAY12",
    "GRAY13",
    "GRAY14",
    "GRAY15",
    "GRAY16",
    "GRAY17",
    "GRAY18",
    "GRAY19",
    "GRAY20",
    "GRAY21",
    "GRAY22",
    "GRAY23",
    "GRAY24",
    "GRAY25",
    "GRAY26",
    "GRAY27",
    "GRAY28",
    "GRAY29",
    "GRAY30",
    "GRAY31",
    "GRAY32",
    "GRAYH",
    "GRAYS",
    "RGB24",
    "RGB27",
    "RGB30",
    "RGB33",
    "RGB36",
    "RGB39",
    "RGB42",
    "RGB45",
    "RGB48",
    "RGB51",
    "RGB54",
    "RGB57",
    "RGB60",
    "RGB63",
    "RGB66",
    "RGB69",
    "RGB72",
    "RGB75",
    "RGB78",
    "RGB81",
    "RGB84",
    "RGB87",
    "RGB90",
    "RGB93",
    "RGB96",
    "RGBH",
    "RGBS",
    "YUV410P8",
    "YUV410P9",
    "YUV410P10",
    "YUV410P11",
    "YUV410P12",
    "YUV410P13",
    "YUV410P14",
    "YUV410P15",
    "YUV410P16",
    "YUV410P17",
    "YUV410P18",
    "YUV410P19",
    "YUV410P20",
    "YUV410P21",
    "YUV410P22",
    "YUV410P23",
    "YUV410P24",
    "YUV410P25",
    "YUV410P26",
    "YUV410P27",
    "YUV410P28",
    "YUV410P29",
    "YUV410P30",
    "YUV410P31",
    "YUV410P32",
    "YUV410PH",
    "YUV410PS",
    "YUV411P8",
    "YUV411P9",
    "YUV411P10",
    "YUV411P11",
    "YUV411P12",
    "YUV411P13",
    "YUV411P14",
    "YUV411P15",
    "YUV411P16",
    "YUV411P17",
    "YUV411P18",
    "YUV411P19",
    "YUV411P20",
    "YUV411P21",
    "YUV411P22",
    "YUV411P23",
    "YUV411P24",
    "YUV411P25",
    "YUV411P26",
    "YUV411P27",
    "YUV411P28",
    "YUV411P29",
    "YUV411P30",
    "YUV411P31",
    "YUV411P32",
    "YUV411PH",
    "YUV411PS",
    "YUV420P8",
    "YUV420P9",
    "YUV420P10",
    "YUV420P11",
    "YUV420P12",
    "YUV420P13",
    "YUV420P14",
    "YUV420P15",
    "YUV420P16",
    "YUV420P17",
    "YUV420P18",
    "YUV420P19",
    "YUV420P20",
    "YUV420P21",
    "YUV420P22",
    "YUV420P23",
    "YUV420P24",
    "YUV420P25",
    "YUV420P26",
    "YUV420P27",
    "YUV420P28",
    "YUV420P29",
    "YUV420P30",
    "YUV420P31",
    "YUV420P32",
    "YUV420PH",
    "YUV420PS",
    "YUV422P8",
    "YUV422P9",
    "YUV422P10",
    "YUV422P11",
    "YUV422P12",
    "YUV422P13",
    "YUV422P14",
    "YUV422P15",
    "YUV422P16",
    "YUV422P17",
    "YUV422P18",
    "YUV422P19",
    "YUV422P20",
    "YUV422P21",
    "YUV422P22",
    "YUV422P23",
    "YUV422P24",
    "YUV422P25",
    "YUV422P26",
    "YUV422P27",
    "YUV422P28",
    "YUV422P29",
    "YUV422P30",
    "YUV422P31",
    "YUV422P32",
    "YUV422PH",
    "YUV422PS",
    "YUV440P8",
    "YUV440P9",
    "YUV440P10",
    "YUV440P11",
    "YUV440P12",
    "YUV440P13",
    "YUV440P14",
    "YUV440P15",
    "YUV440P16",
    "YUV440P17",
    "YUV440P18",
    "YUV440P19",
    "YUV440P20",
    "YUV440P21",
    "YUV440P22",
    "YUV440P23",
    "YUV440P24",
    "YUV440P25",
    "YUV440P26",
    "YUV440P27",
    "YUV440P28",
    "YUV440P29",
    "YUV440P30",
    "YUV440P31",
    "YUV440P32",
    "YUV440PH",
    "YUV440PS",
    "YUV444P8",
    "YUV444P9",
    "YUV444P10",
    "YUV444P11",
    "YUV444P12",
    "YUV444P13",
    "YUV444P14",
    "YUV444P15",
    "YUV444P16",
    "YUV444P17",
    "YUV444P18",
    "YUV444P19",
    "YUV444P20",
    "YUV444P21",
    "YUV444P22",
    "YUV444P23",
    "YUV444P24",
    "YUV444P25",
    "YUV444P26",
    "YUV444P27",
    "YUV444P28",
    "YUV444P29",
    "YUV444P30",
    "YUV444P31",
    "YUV444P32",
    "YUV444PH",
    "YUV444PS",
    "PresetVideoFormat",
]


def _make_video_id(
    color_family: int, sample_type: int, bits_per_sample: int, subsampling_w: int, subsampling_h: int
) -> int:
    return color_family << 28 | sample_type << 24 | bits_per_sample << 16 | subsampling_w << 8 | subsampling_h << 0


################################################


class PresetVideoFormat(IntEnum):
    NONE = VSPresetVideoFormat.NONE
    GRAY8 = VSPresetVideoFormat.GRAY8
    GRAY9 = VSPresetVideoFormat.GRAY9
    GRAY10 = VSPresetVideoFormat.GRAY10
    GRAY11 = _make_video_id(GRAY, INTEGER, 11, 0, 0)
    GRAY12 = VSPresetVideoFormat.GRAY12
    GRAY13 = _make_video_id(GRAY, INTEGER, 13, 0, 0)
    GRAY14 = VSPresetVideoFormat.GRAY14
    GRAY15 = _make_video_id(GRAY, INTEGER, 15, 0, 0)
    GRAY16 = VSPresetVideoFormat.GRAY16
    GRAY17 = _make_video_id(GRAY, INTEGER, 17, 0, 0)
    GRAY18 = _make_video_id(GRAY, INTEGER, 18, 0, 0)
    GRAY19 = _make_video_id(GRAY, INTEGER, 19, 0, 0)
    GRAY20 = _make_video_id(GRAY, INTEGER, 20, 0, 0)
    GRAY21 = _make_video_id(GRAY, INTEGER, 21, 0, 0)
    GRAY22 = _make_video_id(GRAY, INTEGER, 22, 0, 0)
    GRAY23 = _make_video_id(GRAY, INTEGER, 23, 0, 0)
    GRAY24 = _make_video_id(GRAY, INTEGER, 24, 0, 0)
    GRAY25 = _make_video_id(GRAY, INTEGER, 25, 0, 0)
    GRAY26 = _make_video_id(GRAY, INTEGER, 26, 0, 0)
    GRAY27 = _make_video_id(GRAY, INTEGER, 27, 0, 0)
    GRAY28 = _make_video_id(GRAY, INTEGER, 28, 0, 0)
    GRAY29 = _make_video_id(GRAY, INTEGER, 29, 0, 0)
    GRAY30 = _make_video_id(GRAY, INTEGER, 30, 0, 0)
    GRAY31 = _make_video_id(GRAY, INTEGER, 31, 0, 0)
    GRAY32 = VSPresetVideoFormat.GRAY32
    GRAYH = VSPresetVideoFormat.GRAYH
    GRAYS = VSPresetVideoFormat.GRAYS

    ################################################

    YUV420P8 = VSPresetVideoFormat.YUV420P8
    YUV420P9 = VSPresetVideoFormat.YUV420P9
    YUV420P10 = VSPresetVideoFormat.YUV420P10
    YUV420P11 = _make_video_id(YUV, INTEGER, 11, 1, 1)
    YUV420P12 = VSPresetVideoFormat.YUV420P12
    YUV420P13 = _make_video_id(YUV, INTEGER, 13, 1, 1)
    YUV420P14 = VSPresetVideoFormat.YUV420P14
    YUV420P15 = _make_video_id(YUV, INTEGER, 15, 1, 1)
    YUV420P16 = VSPresetVideoFormat.YUV420P16
    YUV420P17 = _make_video_id(YUV, INTEGER, 17, 1, 1)
    YUV420P18 = _make_video_id(YUV, INTEGER, 18, 1, 1)
    YUV420P19 = _make_video_id(YUV, INTEGER, 19, 1, 1)
    YUV420P20 = _make_video_id(YUV, INTEGER, 20, 1, 1)
    YUV420P21 = _make_video_id(YUV, INTEGER, 21, 1, 1)
    YUV420P22 = _make_video_id(YUV, INTEGER, 22, 1, 1)
    YUV420P23 = _make_video_id(YUV, INTEGER, 23, 1, 1)
    YUV420P24 = _make_video_id(YUV, INTEGER, 24, 1, 1)
    YUV420P25 = _make_video_id(YUV, INTEGER, 25, 1, 1)
    YUV420P26 = _make_video_id(YUV, INTEGER, 26, 1, 1)
    YUV420P27 = _make_video_id(YUV, INTEGER, 27, 1, 1)
    YUV420P28 = _make_video_id(YUV, INTEGER, 28, 1, 1)
    YUV420P29 = _make_video_id(YUV, INTEGER, 29, 1, 1)
    YUV420P30 = _make_video_id(YUV, INTEGER, 30, 1, 1)
    YUV420P31 = _make_video_id(YUV, INTEGER, 31, 1, 1)
    YUV420P32 = _make_video_id(YUV, INTEGER, 32, 1, 1)

    YUV420PH = VSPresetVideoFormat.YUV420PH
    YUV420PS = VSPresetVideoFormat.YUV420PS

    ################################################

    YUV444P8 = VSPresetVideoFormat.YUV444P8
    YUV444P9 = VSPresetVideoFormat.YUV444P9
    YUV444P10 = VSPresetVideoFormat.YUV444P10
    YUV444P11 = _make_video_id(YUV, INTEGER, 11, 0, 0)
    YUV444P12 = VSPresetVideoFormat.YUV444P12
    YUV444P13 = _make_video_id(YUV, INTEGER, 13, 0, 0)
    YUV444P14 = VSPresetVideoFormat.YUV444P14
    YUV444P15 = _make_video_id(YUV, INTEGER, 15, 0, 0)
    YUV444P16 = VSPresetVideoFormat.YUV444P16
    YUV444P17 = _make_video_id(YUV, INTEGER, 17, 0, 0)
    YUV444P18 = _make_video_id(YUV, INTEGER, 18, 0, 0)
    YUV444P19 = _make_video_id(YUV, INTEGER, 19, 0, 0)
    YUV444P20 = _make_video_id(YUV, INTEGER, 20, 0, 0)
    YUV444P21 = _make_video_id(YUV, INTEGER, 21, 0, 0)
    YUV444P22 = _make_video_id(YUV, INTEGER, 22, 0, 0)
    YUV444P23 = _make_video_id(YUV, INTEGER, 23, 0, 0)
    YUV444P24 = _make_video_id(YUV, INTEGER, 24, 0, 0)
    YUV444P25 = _make_video_id(YUV, INTEGER, 25, 0, 0)
    YUV444P26 = _make_video_id(YUV, INTEGER, 26, 0, 0)
    YUV444P27 = _make_video_id(YUV, INTEGER, 27, 0, 0)
    YUV444P28 = _make_video_id(YUV, INTEGER, 28, 0, 0)
    YUV444P29 = _make_video_id(YUV, INTEGER, 29, 0, 0)
    YUV444P30 = _make_video_id(YUV, INTEGER, 30, 0, 0)
    YUV444P31 = _make_video_id(YUV, INTEGER, 31, 0, 0)
    YUV444P32 = _make_video_id(YUV, INTEGER, 32, 0, 0)

    YUV444PH = VSPresetVideoFormat.YUV444PH
    YUV444PS = VSPresetVideoFormat.YUV444PS

    ################################################

    YUV422P8 = VSPresetVideoFormat.YUV422P8
    YUV422P9 = VSPresetVideoFormat.YUV422P9
    YUV422P10 = VSPresetVideoFormat.YUV422P10
    YUV422P11 = _make_video_id(YUV, INTEGER, 11, 1, 0)
    YUV422P12 = VSPresetVideoFormat.YUV422P12
    YUV422P13 = _make_video_id(YUV, INTEGER, 13, 1, 0)
    YUV422P14 = VSPresetVideoFormat.YUV422P14
    YUV422P15 = _make_video_id(YUV, INTEGER, 15, 1, 0)
    YUV422P16 = VSPresetVideoFormat.YUV422P16
    YUV422P17 = _make_video_id(YUV, INTEGER, 17, 1, 0)
    YUV422P18 = _make_video_id(YUV, INTEGER, 18, 1, 0)
    YUV422P19 = _make_video_id(YUV, INTEGER, 19, 1, 0)
    YUV422P20 = _make_video_id(YUV, INTEGER, 20, 1, 0)
    YUV422P21 = _make_video_id(YUV, INTEGER, 21, 1, 0)
    YUV422P22 = _make_video_id(YUV, INTEGER, 22, 1, 0)
    YUV422P23 = _make_video_id(YUV, INTEGER, 23, 1, 0)
    YUV422P24 = _make_video_id(YUV, INTEGER, 24, 1, 0)
    YUV422P25 = _make_video_id(YUV, INTEGER, 25, 1, 0)
    YUV422P26 = _make_video_id(YUV, INTEGER, 26, 1, 0)
    YUV422P27 = _make_video_id(YUV, INTEGER, 27, 1, 0)
    YUV422P28 = _make_video_id(YUV, INTEGER, 28, 1, 0)
    YUV422P29 = _make_video_id(YUV, INTEGER, 29, 1, 0)
    YUV422P30 = _make_video_id(YUV, INTEGER, 30, 1, 0)
    YUV422P31 = _make_video_id(YUV, INTEGER, 31, 1, 0)
    YUV422P32 = _make_video_id(YUV, INTEGER, 32, 1, 0)

    YUV422PH = VSPresetVideoFormat.YUV422PH
    YUV422PS = VSPresetVideoFormat.YUV422PS

    ################################################

    YUV411P8 = VSPresetVideoFormat.YUV411P8
    YUV411P9 = _make_video_id(YUV, INTEGER, 9, 2, 0)
    YUV411P10 = _make_video_id(YUV, INTEGER, 10, 2, 0)
    YUV411P11 = _make_video_id(YUV, INTEGER, 11, 2, 0)
    YUV411P12 = _make_video_id(YUV, INTEGER, 12, 2, 0)
    YUV411P13 = _make_video_id(YUV, INTEGER, 13, 2, 0)
    YUV411P14 = _make_video_id(YUV, INTEGER, 14, 2, 0)
    YUV411P15 = _make_video_id(YUV, INTEGER, 15, 2, 0)
    YUV411P16 = _make_video_id(YUV, INTEGER, 16, 2, 0)
    YUV411P17 = _make_video_id(YUV, INTEGER, 17, 2, 0)
    YUV411P18 = _make_video_id(YUV, INTEGER, 18, 2, 0)
    YUV411P19 = _make_video_id(YUV, INTEGER, 19, 2, 0)
    YUV411P20 = _make_video_id(YUV, INTEGER, 20, 2, 0)
    YUV411P21 = _make_video_id(YUV, INTEGER, 21, 2, 0)
    YUV411P22 = _make_video_id(YUV, INTEGER, 22, 2, 0)
    YUV411P23 = _make_video_id(YUV, INTEGER, 23, 2, 0)
    YUV411P24 = _make_video_id(YUV, INTEGER, 24, 2, 0)
    YUV411P25 = _make_video_id(YUV, INTEGER, 25, 2, 0)
    YUV411P26 = _make_video_id(YUV, INTEGER, 26, 2, 0)
    YUV411P27 = _make_video_id(YUV, INTEGER, 27, 2, 0)
    YUV411P28 = _make_video_id(YUV, INTEGER, 28, 2, 0)
    YUV411P29 = _make_video_id(YUV, INTEGER, 29, 2, 0)
    YUV411P30 = _make_video_id(YUV, INTEGER, 30, 2, 0)
    YUV411P31 = _make_video_id(YUV, INTEGER, 31, 2, 0)
    YUV411P32 = _make_video_id(YUV, INTEGER, 32, 2, 0)

    YUV411PH = _make_video_id(YUV, FLOAT, 16, 2, 0)
    YUV411PS = _make_video_id(YUV, FLOAT, 32, 2, 0)

    ################################################

    YUV440P8 = VSPresetVideoFormat.YUV440P8
    YUV440P9 = _make_video_id(YUV, INTEGER, 9, 0, 1)
    YUV440P10 = _make_video_id(YUV, INTEGER, 10, 0, 1)
    YUV440P11 = _make_video_id(YUV, INTEGER, 11, 0, 1)
    YUV440P12 = _make_video_id(YUV, INTEGER, 12, 0, 1)
    YUV440P13 = _make_video_id(YUV, INTEGER, 13, 0, 1)
    YUV440P14 = _make_video_id(YUV, INTEGER, 14, 0, 1)
    YUV440P15 = _make_video_id(YUV, INTEGER, 15, 0, 1)
    YUV440P16 = _make_video_id(YUV, INTEGER, 16, 0, 1)
    YUV440P17 = _make_video_id(YUV, INTEGER, 17, 0, 1)
    YUV440P18 = _make_video_id(YUV, INTEGER, 18, 0, 1)
    YUV440P19 = _make_video_id(YUV, INTEGER, 19, 0, 1)
    YUV440P20 = _make_video_id(YUV, INTEGER, 20, 0, 1)
    YUV440P21 = _make_video_id(YUV, INTEGER, 21, 0, 1)
    YUV440P22 = _make_video_id(YUV, INTEGER, 22, 0, 1)
    YUV440P23 = _make_video_id(YUV, INTEGER, 23, 0, 1)
    YUV440P24 = _make_video_id(YUV, INTEGER, 24, 0, 1)
    YUV440P25 = _make_video_id(YUV, INTEGER, 25, 0, 1)
    YUV440P26 = _make_video_id(YUV, INTEGER, 26, 0, 1)
    YUV440P27 = _make_video_id(YUV, INTEGER, 27, 0, 1)
    YUV440P28 = _make_video_id(YUV, INTEGER, 28, 0, 1)
    YUV440P29 = _make_video_id(YUV, INTEGER, 29, 0, 1)
    YUV440P30 = _make_video_id(YUV, INTEGER, 30, 0, 1)
    YUV440P31 = _make_video_id(YUV, INTEGER, 31, 0, 1)
    YUV440P32 = _make_video_id(YUV, INTEGER, 32, 0, 1)

    YUV440PH = _make_video_id(YUV, FLOAT, 16, 0, 1)
    YUV440PS = _make_video_id(YUV, FLOAT, 32, 0, 1)

    ################################################

    YUV410P8 = VSPresetVideoFormat.YUV410P8
    YUV410P9 = _make_video_id(YUV, INTEGER, 9, 2, 2)
    YUV410P10 = _make_video_id(YUV, INTEGER, 10, 2, 2)
    YUV410P11 = _make_video_id(YUV, INTEGER, 11, 2, 2)
    YUV410P12 = _make_video_id(YUV, INTEGER, 12, 2, 2)
    YUV410P13 = _make_video_id(YUV, INTEGER, 13, 2, 2)
    YUV410P14 = _make_video_id(YUV, INTEGER, 14, 2, 2)
    YUV410P15 = _make_video_id(YUV, INTEGER, 15, 2, 2)
    YUV410P16 = _make_video_id(YUV, INTEGER, 16, 2, 2)
    YUV410P17 = _make_video_id(YUV, INTEGER, 17, 2, 2)
    YUV410P18 = _make_video_id(YUV, INTEGER, 18, 2, 2)
    YUV410P19 = _make_video_id(YUV, INTEGER, 19, 2, 2)
    YUV410P20 = _make_video_id(YUV, INTEGER, 20, 2, 2)
    YUV410P21 = _make_video_id(YUV, INTEGER, 21, 2, 2)
    YUV410P22 = _make_video_id(YUV, INTEGER, 22, 2, 2)
    YUV410P23 = _make_video_id(YUV, INTEGER, 23, 2, 2)
    YUV410P24 = _make_video_id(YUV, INTEGER, 24, 2, 2)
    YUV410P25 = _make_video_id(YUV, INTEGER, 25, 2, 2)
    YUV410P26 = _make_video_id(YUV, INTEGER, 26, 2, 2)
    YUV410P27 = _make_video_id(YUV, INTEGER, 27, 2, 2)
    YUV410P28 = _make_video_id(YUV, INTEGER, 28, 2, 2)
    YUV410P29 = _make_video_id(YUV, INTEGER, 29, 2, 2)
    YUV410P30 = _make_video_id(YUV, INTEGER, 30, 2, 2)
    YUV410P31 = _make_video_id(YUV, INTEGER, 31, 2, 2)
    YUV410P32 = _make_video_id(YUV, INTEGER, 32, 2, 2)

    YUV410PH = _make_video_id(YUV, FLOAT, 16, 2, 2)
    YUV410PS = _make_video_id(YUV, FLOAT, 32, 2, 2)

    ################################################

    RGB24 = VSPresetVideoFormat.RGB24
    RGB27 = VSPresetVideoFormat.RGB27
    RGB30 = VSPresetVideoFormat.RGB30
    RGB33 = _make_video_id(RGB, INTEGER, 11, 0, 0)
    RGB36 = VSPresetVideoFormat.RGB36
    RGB39 = _make_video_id(RGB, INTEGER, 13, 0, 0)
    RGB42 = VSPresetVideoFormat.RGB42
    RGB45 = _make_video_id(RGB, INTEGER, 15, 0, 0)
    RGB48 = VSPresetVideoFormat.RGB48
    RGB51 = _make_video_id(RGB, INTEGER, 17, 0, 0)
    RGB54 = _make_video_id(RGB, INTEGER, 18, 0, 0)
    RGB57 = _make_video_id(RGB, INTEGER, 19, 0, 0)
    RGB60 = _make_video_id(RGB, INTEGER, 20, 0, 0)
    RGB63 = _make_video_id(RGB, INTEGER, 21, 0, 0)
    RGB66 = _make_video_id(RGB, INTEGER, 22, 0, 0)
    RGB69 = _make_video_id(RGB, INTEGER, 23, 0, 0)
    RGB72 = _make_video_id(RGB, INTEGER, 24, 0, 0)
    RGB75 = _make_video_id(RGB, INTEGER, 25, 0, 0)
    RGB78 = _make_video_id(RGB, INTEGER, 26, 0, 0)
    RGB81 = _make_video_id(RGB, INTEGER, 27, 0, 0)
    RGB84 = _make_video_id(RGB, INTEGER, 28, 0, 0)
    RGB87 = _make_video_id(RGB, INTEGER, 29, 0, 0)
    RGB90 = _make_video_id(RGB, INTEGER, 30, 0, 0)
    RGB93 = _make_video_id(RGB, INTEGER, 31, 0, 0)
    RGB96 = _make_video_id(RGB, INTEGER, 32, 0, 0)

    RGBH = VSPresetVideoFormat.RGBH
    RGBS = VSPresetVideoFormat.RGBS


GRAY8 = PresetVideoFormat.GRAY8
GRAY9 = PresetVideoFormat.GRAY9
GRAY10 = PresetVideoFormat.GRAY10
GRAY11 = PresetVideoFormat.GRAY11
GRAY12 = PresetVideoFormat.GRAY12
GRAY13 = PresetVideoFormat.GRAY13
GRAY14 = PresetVideoFormat.GRAY14
GRAY15 = PresetVideoFormat.GRAY15
GRAY16 = PresetVideoFormat.GRAY16
GRAY17 = PresetVideoFormat.GRAY17
GRAY18 = PresetVideoFormat.GRAY18
GRAY19 = PresetVideoFormat.GRAY19
GRAY20 = PresetVideoFormat.GRAY20
GRAY21 = PresetVideoFormat.GRAY21
GRAY22 = PresetVideoFormat.GRAY22
GRAY23 = PresetVideoFormat.GRAY23
GRAY24 = PresetVideoFormat.GRAY24
GRAY25 = PresetVideoFormat.GRAY25
GRAY26 = PresetVideoFormat.GRAY26
GRAY27 = PresetVideoFormat.GRAY27
GRAY28 = PresetVideoFormat.GRAY28
GRAY29 = PresetVideoFormat.GRAY29
GRAY30 = PresetVideoFormat.GRAY30
GRAY31 = PresetVideoFormat.GRAY31
GRAY32 = PresetVideoFormat.GRAY32
GRAYH = PresetVideoFormat.GRAYH
GRAYS = PresetVideoFormat.GRAYS
YUV420P8 = PresetVideoFormat.YUV420P8
YUV420P9 = PresetVideoFormat.YUV420P9
YUV420P10 = PresetVideoFormat.YUV420P10
YUV420P11 = PresetVideoFormat.YUV420P11
YUV420P12 = PresetVideoFormat.YUV420P12
YUV420P13 = PresetVideoFormat.YUV420P13
YUV420P14 = PresetVideoFormat.YUV420P14
YUV420P15 = PresetVideoFormat.YUV420P15
YUV420P16 = PresetVideoFormat.YUV420P16
YUV420P17 = PresetVideoFormat.YUV420P17
YUV420P18 = PresetVideoFormat.YUV420P18
YUV420P19 = PresetVideoFormat.YUV420P19
YUV420P20 = PresetVideoFormat.YUV420P20
YUV420P21 = PresetVideoFormat.YUV420P21
YUV420P22 = PresetVideoFormat.YUV420P22
YUV420P23 = PresetVideoFormat.YUV420P23
YUV420P24 = PresetVideoFormat.YUV420P24
YUV420P25 = PresetVideoFormat.YUV420P25
YUV420P26 = PresetVideoFormat.YUV420P26
YUV420P27 = PresetVideoFormat.YUV420P27
YUV420P28 = PresetVideoFormat.YUV420P28
YUV420P29 = PresetVideoFormat.YUV420P29
YUV420P30 = PresetVideoFormat.YUV420P30
YUV420P31 = PresetVideoFormat.YUV420P31
YUV420P32 = PresetVideoFormat.YUV420P32
YUV420PH = PresetVideoFormat.YUV420PH
YUV420PS = PresetVideoFormat.YUV420PS
YUV444P8 = PresetVideoFormat.YUV444P8
YUV444P9 = PresetVideoFormat.YUV444P9
YUV444P10 = PresetVideoFormat.YUV444P10
YUV444P11 = PresetVideoFormat.YUV444P11
YUV444P12 = PresetVideoFormat.YUV444P12
YUV444P13 = PresetVideoFormat.YUV444P13
YUV444P14 = PresetVideoFormat.YUV444P14
YUV444P15 = PresetVideoFormat.YUV444P15
YUV444P16 = PresetVideoFormat.YUV444P16
YUV444P17 = PresetVideoFormat.YUV444P17
YUV444P18 = PresetVideoFormat.YUV444P18
YUV444P19 = PresetVideoFormat.YUV444P19
YUV444P20 = PresetVideoFormat.YUV444P20
YUV444P21 = PresetVideoFormat.YUV444P21
YUV444P22 = PresetVideoFormat.YUV444P22
YUV444P23 = PresetVideoFormat.YUV444P23
YUV444P24 = PresetVideoFormat.YUV444P24
YUV444P25 = PresetVideoFormat.YUV444P25
YUV444P26 = PresetVideoFormat.YUV444P26
YUV444P27 = PresetVideoFormat.YUV444P27
YUV444P28 = PresetVideoFormat.YUV444P28
YUV444P29 = PresetVideoFormat.YUV444P29
YUV444P30 = PresetVideoFormat.YUV444P30
YUV444P31 = PresetVideoFormat.YUV444P31
YUV444P32 = PresetVideoFormat.YUV444P32
YUV444PH = PresetVideoFormat.YUV444PH
YUV444PS = PresetVideoFormat.YUV444PS
YUV422P8 = PresetVideoFormat.YUV422P8
YUV422P9 = PresetVideoFormat.YUV422P9
YUV422P10 = PresetVideoFormat.YUV422P10
YUV422P11 = PresetVideoFormat.YUV422P11
YUV422P12 = PresetVideoFormat.YUV422P12
YUV422P13 = PresetVideoFormat.YUV422P13
YUV422P14 = PresetVideoFormat.YUV422P14
YUV422P15 = PresetVideoFormat.YUV422P15
YUV422P16 = PresetVideoFormat.YUV422P16
YUV422P17 = PresetVideoFormat.YUV422P17
YUV422P18 = PresetVideoFormat.YUV422P18
YUV422P19 = PresetVideoFormat.YUV422P19
YUV422P20 = PresetVideoFormat.YUV422P20
YUV422P21 = PresetVideoFormat.YUV422P21
YUV422P22 = PresetVideoFormat.YUV422P22
YUV422P23 = PresetVideoFormat.YUV422P23
YUV422P24 = PresetVideoFormat.YUV422P24
YUV422P25 = PresetVideoFormat.YUV422P25
YUV422P26 = PresetVideoFormat.YUV422P26
YUV422P27 = PresetVideoFormat.YUV422P27
YUV422P28 = PresetVideoFormat.YUV422P28
YUV422P29 = PresetVideoFormat.YUV422P29
YUV422P30 = PresetVideoFormat.YUV422P30
YUV422P31 = PresetVideoFormat.YUV422P31
YUV422P32 = PresetVideoFormat.YUV422P32
YUV422PH = PresetVideoFormat.YUV422PH
YUV422PS = PresetVideoFormat.YUV422PS
YUV411P8 = PresetVideoFormat.YUV411P8
YUV411P9 = PresetVideoFormat.YUV411P9
YUV411P10 = PresetVideoFormat.YUV411P10
YUV411P11 = PresetVideoFormat.YUV411P11
YUV411P12 = PresetVideoFormat.YUV411P12
YUV411P13 = PresetVideoFormat.YUV411P13
YUV411P14 = PresetVideoFormat.YUV411P14
YUV411P15 = PresetVideoFormat.YUV411P15
YUV411P16 = PresetVideoFormat.YUV411P16
YUV411P17 = PresetVideoFormat.YUV411P17
YUV411P18 = PresetVideoFormat.YUV411P18
YUV411P19 = PresetVideoFormat.YUV411P19
YUV411P20 = PresetVideoFormat.YUV411P20
YUV411P21 = PresetVideoFormat.YUV411P21
YUV411P22 = PresetVideoFormat.YUV411P22
YUV411P23 = PresetVideoFormat.YUV411P23
YUV411P24 = PresetVideoFormat.YUV411P24
YUV411P25 = PresetVideoFormat.YUV411P25
YUV411P26 = PresetVideoFormat.YUV411P26
YUV411P27 = PresetVideoFormat.YUV411P27
YUV411P28 = PresetVideoFormat.YUV411P28
YUV411P29 = PresetVideoFormat.YUV411P29
YUV411P30 = PresetVideoFormat.YUV411P30
YUV411P31 = PresetVideoFormat.YUV411P31
YUV411P32 = PresetVideoFormat.YUV411P32
YUV411PH = PresetVideoFormat.YUV411PH
YUV411PS = PresetVideoFormat.YUV411PS
YUV440P8 = PresetVideoFormat.YUV440P8
YUV440P9 = PresetVideoFormat.YUV440P9
YUV440P10 = PresetVideoFormat.YUV440P10
YUV440P11 = PresetVideoFormat.YUV440P11
YUV440P12 = PresetVideoFormat.YUV440P12
YUV440P13 = PresetVideoFormat.YUV440P13
YUV440P14 = PresetVideoFormat.YUV440P14
YUV440P15 = PresetVideoFormat.YUV440P15
YUV440P16 = PresetVideoFormat.YUV440P16
YUV440P17 = PresetVideoFormat.YUV440P17
YUV440P18 = PresetVideoFormat.YUV440P18
YUV440P19 = PresetVideoFormat.YUV440P19
YUV440P20 = PresetVideoFormat.YUV440P20
YUV440P21 = PresetVideoFormat.YUV440P21
YUV440P22 = PresetVideoFormat.YUV440P22
YUV440P23 = PresetVideoFormat.YUV440P23
YUV440P24 = PresetVideoFormat.YUV440P24
YUV440P25 = PresetVideoFormat.YUV440P25
YUV440P26 = PresetVideoFormat.YUV440P26
YUV440P27 = PresetVideoFormat.YUV440P27
YUV440P28 = PresetVideoFormat.YUV440P28
YUV440P29 = PresetVideoFormat.YUV440P29
YUV440P30 = PresetVideoFormat.YUV440P30
YUV440P31 = PresetVideoFormat.YUV440P31
YUV440P32 = PresetVideoFormat.YUV440P32
YUV440PH = PresetVideoFormat.YUV440PH
YUV440PS = PresetVideoFormat.YUV440PS
YUV410P8 = PresetVideoFormat.YUV410P8
YUV410P9 = PresetVideoFormat.YUV410P9
YUV410P10 = PresetVideoFormat.YUV410P10
YUV410P11 = PresetVideoFormat.YUV410P11
YUV410P12 = PresetVideoFormat.YUV410P12
YUV410P13 = PresetVideoFormat.YUV410P13
YUV410P14 = PresetVideoFormat.YUV410P14
YUV410P15 = PresetVideoFormat.YUV410P15
YUV410P16 = PresetVideoFormat.YUV410P16
YUV410P17 = PresetVideoFormat.YUV410P17
YUV410P18 = PresetVideoFormat.YUV410P18
YUV410P19 = PresetVideoFormat.YUV410P19
YUV410P20 = PresetVideoFormat.YUV410P20
YUV410P21 = PresetVideoFormat.YUV410P21
YUV410P22 = PresetVideoFormat.YUV410P22
YUV410P23 = PresetVideoFormat.YUV410P23
YUV410P24 = PresetVideoFormat.YUV410P24
YUV410P25 = PresetVideoFormat.YUV410P25
YUV410P26 = PresetVideoFormat.YUV410P26
YUV410P27 = PresetVideoFormat.YUV410P27
YUV410P28 = PresetVideoFormat.YUV410P28
YUV410P29 = PresetVideoFormat.YUV410P29
YUV410P30 = PresetVideoFormat.YUV410P30
YUV410P31 = PresetVideoFormat.YUV410P31
YUV410P32 = PresetVideoFormat.YUV410P32
YUV410PH = PresetVideoFormat.YUV410PH
YUV410PS = PresetVideoFormat.YUV410PS
RGB24 = PresetVideoFormat.RGB24
RGB27 = PresetVideoFormat.RGB27
RGB30 = PresetVideoFormat.RGB30
RGB33 = PresetVideoFormat.RGB33
RGB36 = PresetVideoFormat.RGB36
RGB39 = PresetVideoFormat.RGB39
RGB42 = PresetVideoFormat.RGB42
RGB45 = PresetVideoFormat.RGB45
RGB48 = PresetVideoFormat.RGB48
RGB51 = PresetVideoFormat.RGB51
RGB54 = PresetVideoFormat.RGB54
RGB57 = PresetVideoFormat.RGB57
RGB60 = PresetVideoFormat.RGB60
RGB63 = PresetVideoFormat.RGB63
RGB66 = PresetVideoFormat.RGB66
RGB69 = PresetVideoFormat.RGB69
RGB72 = PresetVideoFormat.RGB72
RGB75 = PresetVideoFormat.RGB75
RGB78 = PresetVideoFormat.RGB78
RGB81 = PresetVideoFormat.RGB81
RGB84 = PresetVideoFormat.RGB84
RGB87 = PresetVideoFormat.RGB87
RGB90 = PresetVideoFormat.RGB90
RGB93 = PresetVideoFormat.RGB93
RGB96 = PresetVideoFormat.RGB96
RGBH = PresetVideoFormat.RGBH
RGBS = PresetVideoFormat.RGBS
