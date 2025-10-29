from __future__ import annotations

from typing import Any, Mapping

from jetpytools import FuncExcept

from ..exceptions import UndefinedColorRangeError, UndefinedMatrixError, UndefinedPrimariesError, UndefinedTransferError
from ..types import HoldsPropValue
from ..vs_proxy import vs
from .base import PropEnum, _base_from_video

__all__ = [
    "ColorRange",
    "ColorRangeLike",
    "ColorRangeT",  # Deprecated alias
    "Matrix",
    "MatrixLike",
    "MatrixT",  # Deprecated alias
    "Primaries",
    "PrimariesLike",
    "PrimariesT",  # Deprecated alias
    "Transfer",
    "TransferLike",
    "TransferT",  # Deprecated alias
]


class Matrix(PropEnum):
    """
    Matrix coefficients ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Table E.5).
    """

    RGB = 0, "gbr", "RGB"
    """
    The identity matrix.

    Typically used for GBR (often referred to as RGB); however, may also be used for YZX (often referred to as XYZ)

    IEC 61966-2-1 sRGB

    SMPTE ST 428-1 (2006)

    See ITU-T H.265 Equations E-31 to E-33
    """

    GBR = RGB

    BT709 = 1, "bt709", "BT.709"
    """
    Kr = 0.2126; Kb = 0.0722

    Rec. ITU-R BT.709-6

    Rec. ITU-R BT.1361-0 conventional colour gamut system and extended colour gamut system (historical)

    IEC 61966-2-4 xvYCC709

    SMPTE RP 177 (1993) Annex B
    """

    UNKNOWN = 2, "unknown"
    """
    Image characteristics are unknown or are determined by the application.
    """

    FCC = 4, "fcc", "FCC"
    """
    KR = 0.30; KB = 0.11

    FCC Title 47 Code of Federal Regulations (2003) 73.682 (a) (20)

    See ITU-T H.265 Equations E-28 to E-30
    """

    BT470BG = 5, "bt470bg", "BT.470bg"
    """
    KR = 0.299; KB = 0.114

    (Functionally the same as [Matrix.SMPTE170M][vstools.Matrix.SMPTE170M])

    Rec. ITU-R BT.470-6 System B, G (historical)

    Rec. ITU-R BT.601-7 625

    Rec. ITU-R BT.1358-0 625 (historical)

    Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM

    IEC 61966-2-1 sYCC

    IEC 61966-2-4 xvYCC601

    See ITU-T H.265 Equations E-28 to E-30
    """

    BT601_625 = BT470BG

    SMPTE170M = 6, "smpte170m", "SMPTE ST 170m"
    """
    Kr = 0.299; Kb = 0.114

    (Functionally the same as [Matrix.BT470BG][vstools.Matrix.BT470BG])

    Rec. ITU-R BT.601-7 525

    Rec. ITU-R BT.1358-1 525 or 625 (historical)

    Rec. ITU-R BT.1700-0 NTSC

    SMPTE ST 170 (2004)

    See ITU-T H.265 Equations E-28 to E-30

    """

    BT601_525 = SMPTE170M

    SMPTE240M = 7, "smpte240m", "SMPTE ST 240m"
    """
    KR = 0.212; KB = 0.087

    SMPTE ST 240 (1999, historical)

    See ITU-T H.265 Equations E-28 to E-30
    """

    YCGCO = 8, "ycgco", "YCgCo"
    """
    KR = 0.2126; KB = 0.0722

    See Implementation And Evaluation Of Residual Color Transform For 4:4:4 RGB Lossless Coding
    """

    BT2020NCL = 9, "bt2020nc", "BT.2020 non-constant luminance"
    """
    KR = 0.2627; KB = 0.0593

    Rec. ITU-R BT.2020-2 non-constant luminance system

    Rec. ITU-R BT.2100-2 Y'CbCr

    See ITU-T H.265 Equations E-28 to E-30

    """

    BT2020CL = 10, "bt2020c", "BT.2020 constant luminance"
    """
    KR = 0.2627; KB = 0.0593

    Rec. ITU-R BT.2020-2 constant luminance system

    See ITU-T H.265 Equations E-49 to E-58
    """

    CHROMANCL = 12, "chroma-derived-nc", "Chromaticity derived non-constant luminance"
    """
    Chromaticity-derived non-constant luminance system

    See ITU-T H.265 Equations E-22 to E-27
    See ITU-T H.265 Equations E-28 to E-30
    """

    CHROMACL = 13, "chroma-derived-c", "Chromaticity derived constant luminance"
    """
    Chromaticity-derived constant luminance system

    See ITU-T H.265 Equations E-22 to E-27

    See ITU-T H.265 Equations E-49 to E-58
    """

    ICTCP = 14, "ictcp", "ICtCp"
    """
    ICtCp

    Rec. ITU-R BT.2100-2 ICTCP

    See ITU-T H.265 Equations E-62 to E-64 for `transfer_characteristics` value 16 (PQ)

    See ITU-T H.265 Equations E-65 to E-67 for `transfer_characteristics` value 18 (HLG)

    """

    @classmethod
    def _missing_(cls, value: object) -> Matrix:
        return Matrix.UNKNOWN if (v := super()._missing_(value)) is None else v

    def is_unknown(self) -> bool:
        """
        Check if the matrix is Matrix.UNKNOWN.
        """
        return self is Matrix.UNKNOWN

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> Matrix:
        """
        Guess the matrix based on the clip's resolution.

        Args:
            frame: Input clip or frame.

        Returns:
            Matrix object.
        """
        from ..utils import get_var_infos

        fmt, width, height = get_var_infos(frame)

        if fmt.color_family == vs.RGB:
            return Matrix.RGB

        if (width, height) <= (1024, 576):
            return Matrix.SMPTE170M if height <= 486 else Matrix.BT470BG

        return Matrix.BT709

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> Matrix:
        """
        Try to obtain the matrix of a clip from the frame props or fallback to clip's resolution
        if the prop is undefined, strict=False and src is a clip.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the frame props. Will ALWAYS error with Matrix.UNKNOWN.
            func: Function returned for custom error handling.

        Returns:
            Matrix object.

        Raises:
            UndefinedMatrixError: If the matrix is undefined or can not be determined from the frameprops.
        """
        return _base_from_video(cls, src, UndefinedMatrixError, strict, func)


class Transfer(PropEnum):
    """
    Transfer characteristics ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Table E.4).
    """

    BT709 = 1, "bt709", "BT.709"
    """
    (Functionally the same as [Transfer.BT601][vstools.Transfer.BT601],
    [Transfer.BT2020_10][vstools.Transfer.BT2020_10],
    and [Transfer.BT2020_12][vstools.Transfer.BT2020_12])
    Rec. ITU-R BT.709-6
    Rec. ITU-R BT.1361-0 conventional
    Colour gamut system (historical)
    """

    BT1886 = BT709

    GAMMA24 = BT709  # Not exactly, but since zimg assumes infinite contrast BT1886 is effectively GAMMA24 here.

    UNKNOWN = 2, "unknown"
    """
    Image characteristics are unknown or are determined by the application.
    """

    BT470M = 4, "bt470m", "BT.470m"
    """
    Rec. ITU-R BT.470-6 System M (historical)
    NTSC Recommendation for transmission standards for colour television (1953)
    FCC, Title 47 Code of Federal Regulations (2003) 73.682 (a) (20)
    """

    GAMMA22 = BT470M

    BT470BG = 5, "bt470bg", "BT.470bg"
    """
    Rec. ITU-R BT.470-6 System B, G (historical)
    Rec. ITU-R BT.1700-0 625 PAL and
    625 SECAM
    """

    GAMMA28 = BT470BG

    BT601 = 6, "smpte170m", "BT.601"
    """
    (Functionally the same as [Transfer.BT709][vstools.Transfer.BT709],
    [Transfer.BT2020_10][vstools.Transfer.BT2020_10],
    and [Transfer.BT2020_12][vstools.Transfer.BT2020_12])
    Rec. ITU-R BT.601-7 525 or 625
    Rec. ITU-R BT.1358-1 525 or 625 (historical)
    Rec. ITU-R BT.1700-0 NTSC
    SMPTE ST 170 (2004)
    """

    SMPTE240M = 7, "smpte240m", "SMPTE ST 240m"
    """
    SMPTE ST 240 (1999, historical).
    """

    LINEAR = 8, "linear", "Linear"
    """
    Linear transfer characteristics.
    """

    LOG100 = 9, "log100", "Log 1:100 contrast"
    """
    Logarithmic transfer characteristic (100:1 range).
    """

    LOG316 = 10, "log316", "Log 1:316 contrast"
    """
    Logarithmic transfer characteristic (100 * sqrt(10):1 range).
    """

    XVYCC = 11, "iec61966-2-4", "xvYCC"
    """
    IEC 61966-2-4.
    """

    SRGB = 13, "iec61966-2-1", "sRGB"
    """
    IEC 61966-2-1 sRGB when matrix is equal to [Matrix.RGB][vstools.Matrix.RGB]
    IEC 61966-2-1 sYCC when matrix is equal to [Matrix.BT470BG][vstools.Matrix.BT470BG]
    """

    BT2020_10 = 14, "bt2020-10", "BT.2020 10 bits"
    """
    (Functionally the same as [Transfer.BT709][vstools.Transfer.BT709], [Transfer.BT601][vstools.Transfer.BT601],
    and [Transfer.BT2020_12][vstools.Transfer.BT2020_12])
    Rec. ITU-R BT.2020-2
    """

    BT2020_12 = 15, "bt2020-12", "BT.2020 12 bits"
    """
    (Functionally the same as [Transfer.BT709][vstools.Transfer.BT709], [Transfer.BT601][vstools.Transfer.BT601],
    and [Transfer.BT2020_10][vstools.Transfer.BT2020_10])
    Rec. ITU-R BT.2020-2
    """

    ST2084 = 16, "smpte2084", "SMPTE ST 2084 (PQ)"
    """
    SMPTE ST 2084 (2014) for 10, 12, 14, and 16-bit systems
    Rec. ITU-R BT.2100-2 perceptual quantization (PQ) system
    """

    PQ = ST2084

    STD_B67 = 18, "arib-std-b67", "ARIB std-b67 (HLG)"
    """
    Association of Radio Industries and Businesses (ARIB) STD-B67
    Rec. ITU-R BT.2100-2 hybrid loggamma (HLG) system
    """

    HLG = STD_B67

    @classmethod
    def _missing_(cls, value: Any) -> Transfer | None:
        return Transfer.UNKNOWN if (v := super()._missing_(value)) is None else v

    def is_unknown(self) -> bool:
        """
        Check if the transfer is Transfer.UNKNOWN.
        """
        return self is Transfer.UNKNOWN

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> Transfer:
        """
        Guess the transfer based on the clip's resolution.

        Args:
            frame: Input clip or frame.

        Returns:
            Transfer object.
        """
        from ..utils import get_var_infos

        fmt, width, height = get_var_infos(frame)

        if fmt.color_family == vs.RGB:
            return Transfer.SRGB

        return Transfer.BT601 if (width, height) <= (1024, 576) else Transfer.BT709

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> Transfer:
        """
        Try to obtain the transfer of a clip from the frame props or fallback to clip's resolution
        if the prop is undefined, strict=False and src is a clip.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the properties. The result may NOT be Transfer.UNKNOWN.
            func: Function returned for custom error handling.

        Returns:
            Transfer object.

        Raises:
            UndefinedTransferError: If the transfer is undefined or can not be determined from the frameprops.
        """
        return _base_from_video(cls, src, UndefinedTransferError, strict, func)


class Primaries(PropEnum):
    """
    Color primaries ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Table E.3).
    """

    BT709 = 1, "bt709", "BT.709"
    """
    Rec. ITU-R BT.709-6

    ```
    Primary      x      y
    Green     0.3000 0.6000
    Blue      0.1500 0.0600
    Red       0.6400 0.3300
    White D65 0.3127 0.3290
    ```

    Rec. ITU-R BT.709-6

    Rec. ITU-R BT.1361-0 conventional colour gamutsystem and extended colour gamut system (historical)

    IEC 61966-2-1 sRGB or sYCC

    IEC 61966-2-4

    SMPTE RP 177 (1993) Annex B
    """

    UNKNOWN = 2, "unknown"
    """
    Unspecified image characteristics are unknown or are determined by the application.
    """

    BT470M = 4, "bt470m", "BT.470m"
    """
    Rec. ITU-R BT.470-6 System M (historical)

    ```
    Primary     x      y
    Green    0.2100 0.7100
    Blue     0.1400 0.0800
    Red      0.6700 0.3300
    White C  0.3100 0.3160
    ```

    Rec. ITU-R BT.470-6 System M (historical)
    NTSC Recommendation for transmission standards for colour television (1953)
    FCC Title 47 Code of Federal Regulations (2003)
    73.682 (a) (20)
    """

    BT470BG = 5, "bt470bg", "BT.470bg"
    """
    Rec. ITU-R BT.470-6 System B, G (historical)

    ```
    Primary      x      y
    Green     0.2900 0.6000
    Blue      0.1500 0.0600
    Red       0.6400 0.3300
    White D65 0.3127 0.3290
    ```

    Rec. ITU-R BT.470-6 System B, G (historical)
    Rec. ITU-R BT.601-7 625
    Rec. ITU-R BT.1358-0 625 (historical)
    Rec. ITU-R BT.1700-0 625 PAL and 625
    SECAM
    """

    BT601_625 = BT470BG

    SMPTE170M = 6, "smpte170m", "SMPTE ST 170m"
    """
    (Functionally the same as [Primaries.SMPTE240M][vstools.Primaries.SMPTE240M])

    ```
    Primary      x      y
    Green     0.3100 0.5950
    Blue      0.1550 0.0700
    Red       0.6300 0.3400
    White D65 0.3127 0.3290
    ```

    Rec. ITU-R BT.601-7 525

    Rec. ITU-R BT.1358-1 525 or 625 (historical)

    Rec. ITU-R BT.1700-0 NTSC

    SMPTE ST 170 (2004)
    """

    BT601_525 = SMPTE170M

    SMPTE240M = 7, "smpte240m", "SMPTE ST 240m"
    """
    SMPTE ST 240 (1999, historical)

    (Functionally the same as [Primaries.SMPTE170M][vstools.Primaries.SMPTE170M])

    ```
    Primary      x      y
    Green     0.3100 0.5950
    Blue      0.1550 0.0700
    Red       0.6300 0.3400
    White D65 0.3127 0.3290
    ```

    SMPTE ST 240 (1999, historical)
    """

    FILM = 8, "film", "Film"
    """
    Generic film (colour filters using Illuminant C)

    ```
    Primary    x      y
    Green   0.2430 0.6920 #(Wratten 58)
    Blue    0.1450 0.0490 #(Wratten 47)
    Red     0.6810 0.3190 #(Wratten 25)
    White C 0.3100 0.3160
    ```
    """

    BT2020 = 9, "bt2020", "BT.2020"
    """
    Rec. ITU-R BT.2020-2

    ```
    Primary       x      y
    Green     0.1700 0.7970
    Blue      0.1310 0.0460
    Red       0.7080 0.2920
    White D65 0.3127 0.3290
    ```

    Rec. ITU-R BT.2020-2

    Rec. ITU-R BT.2100-2
    """

    ST428 = 10, "smpte428", "SMPTE ST 428 (XYZ)"
    """
    SMPTE ST 428-1 (2006)

    ```
    Primary        x   y
    Green    (Y)  0.0 1.0
    Blue     (Z)  0.0 0.0
    Red      (X)  1.0 0.0
    Centre White  1/3 1/3
    ```

    (CIE 1931 XYZ)
    """

    XYZ = ST428

    CIE1931 = ST428

    ST431_2 = 11, "smpte431", "DCI-P3, DCI white point"
    """
    SMPTE RP 431-2 (2011)

    ```
    Primary    x      y
    Green   0.2650 0.6900
    Blue    0.1500 0.0600
    Red     0.6800 0.3200
    White   0.3140 0.3510
    ```

    SMPTE ST 2113 (2019) "P3DCI"
    """

    DCI_P3 = ST431_2

    ST432_1 = 12, "smpte432", "DCI-P3 D65 white point"
    """
    SMPTE EG 432-1 (2010)

    ```
    Primary      x      y
    Green     0.2650 0.6900
    Blue      0.1500 0.0600
    Red       0.6800 0.3200
    White D65 0.3127 0.3290
    ```

    SMPTE EG 432-1 (2010)

    SMPTE ST 2113 (2019) "P3D65"
    """
    DISPLAY_P3 = ST432_1

    JEDEC_P22 = 22, "jedec-p22", "JEDEC P22 (EBU 3213-E)"
    """
    EBU Tech. 3213-E (1975)

    ```
    Primary      x      y
    Green     0.2950 0.6050
    Blue      0.1550 0.0770
    Red       0.6300 0.3400
    White D65 0.3127 0.3290
    ```
    """

    EBU3213 = JEDEC_P22

    @classmethod
    def _missing_(cls, value: Any) -> Primaries | None:
        return Primaries.UNKNOWN if (v := super()._missing_(value)) is None else v

    def is_unknown(self) -> bool:
        """
        Check if the primaries are Primaries.UNKNOWN.
        """
        return self is Primaries.UNKNOWN

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> Primaries:
        """
        Guess the primaries based on the clip's resolution.

        Args:
            frame: Input clip or frame.

        Returns:
            Primaries object.
        """
        from ..utils import get_var_infos

        fmt, width, height = get_var_infos(frame)

        if fmt.color_family == vs.RGB:
            return Primaries.BT709

        if (width, height) <= (1024, 576):
            return Primaries.SMPTE170M if height <= 486 else Primaries.BT470BG

        return Primaries.BT709

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> Primaries:
        """
        Try to obtain the primaries of a clip from the frame props or fallback to clip's resolution
        if the prop is undefined, strict=False and src is a clip.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the frame props. Will ALWAYS error with Primaries.UNKNOWN.
            func: Function returned for custom error handling.

        Returns:
            Primaries object.

        Raises:
            UndefinedPrimariesError: If the primaries are undefined or can not be determined from the frameprops.
        """
        return _base_from_video(cls, src, UndefinedPrimariesError, strict, func)


class ColorRange(PropEnum):
    """
    Pixel Range ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Equations E-10 through E-20.
    """

    LIMITED = 1
    """
    Studio (TV) legal range, 16-235 in 8 bits.

    This is primarily used with YUV integer formats.
    """
    TV = LIMITED

    FULL = 0
    """
    Full (PC) dynamic range, 0-255 in 8 bits.

    Note that float clips should ALWAYS be FULL range!
    RGB clips will ALWAYS be FULL range!
    """
    PC = FULL

    @classmethod
    def _missing_(cls, value: object) -> ColorRange:
        return ColorRange.LIMITED if (v := super()._missing_(value)) is None else v

    @property
    def value_vs(self) -> int:
        """
        VapourSynth (props) value.
        """
        return self.value

    @property
    def value_zimg(self) -> int:
        """
        zimg (resize plugin) value.
        """
        return ~self.value + 2

    def is_limited(self) -> bool:
        """
        Check if ColorRange is limited.
        """
        return bool(self.value)

    def is_full(self) -> bool:
        """
        Check if ColorRange is full.
        """
        return not self.value

    @classmethod
    def from_res(cls, frame: vs.VideoNode | vs.VideoFrame) -> ColorRange:
        """
        Guess the color range from the frame resolution.
        """
        from ..utils import get_var_infos

        fmt, _, _ = get_var_infos(frame)

        if fmt.color_family == vs.RGB:
            return cls.FULL

        return cls.LIMITED

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | Mapping[str, Any], strict: bool = False, func: FuncExcept | None = None
    ) -> ColorRange:
        """
        Try to obtain the color range of a clip from the frame props or fallback to clip's resolution
        if the prop is undefined, strict=False and src is a clip.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the frame props. Sets the ColorRange as MISSING if prop is not there.
            func: Function returned for custom error handling.

        Returns:
            ColorRange object.

        Raises:
            UndefinedColorRangeError: If the color range is undefined or can not be determined from the frameprops.
        """
        return _base_from_video(cls, src, UndefinedColorRangeError, strict, func)


type MatrixLike = int | vs.MatrixCoefficients | Matrix | HoldsPropValue
"""Type alias for values that can be used to initialize a [Matrix][vstools.Matrix]."""

type TransferLike = int | vs.TransferCharacteristics | Transfer | HoldsPropValue
"""Type alias for values that can be used to initialize a [Transfer][vstools.Transfer]."""

type PrimariesLike = int | vs.ColorPrimaries | Primaries | HoldsPropValue
"""Type alias for values that can be used to initialize a [Primaries][vstools.Primaries]."""

type ColorRangeLike = int | vs.ColorRange | ColorRange | HoldsPropValue
"""Type alias for values that can be used to initialize a [ColorRange][vstools.ColorRange]."""

MatrixT = MatrixLike
"""Deprecated alias of MatrixLike"""

TransferT = TransferLike
"""Deprecated alias of TransferLike"""

PrimariesT = PrimariesLike
"""Deprecated alias of PrimariesLike"""

ColorRangeT = ColorRangeLike
"""Deprecated alias of ColorRangeLike"""


def _norm_props_enums(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: (value.value_zimg if isinstance(value, ColorRange) else value) for key, value in kwargs.items()}
