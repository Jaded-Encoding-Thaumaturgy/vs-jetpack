from __future__ import annotations

from typing import Any, TypeAlias, Union

import vapoursynth as vs
from jetpytools import FuncExceptT

from ..exceptions import (
    ReservedMatrixError,
    ReservedPrimariesError,
    ReservedTransferError,
    UndefinedMatrixError,
    UndefinedPrimariesError,
    UndefinedTransferError,
    UnsupportedColorRangeError,
    UnsupportedMatrixError,
    UnsupportedPrimariesError,
    UnsupportedTransferError,
)
from ..types import HoldsPropValueT, KwargsT
from .stubs import PropEnum, _base_from_video, _ColorRangeMeta, _MatrixMeta, _PrimariesMeta, _TransferMeta

__all__ = [
    "ColorRange",
    "ColorRangeT",
    "Matrix",
    "MatrixT",
    "Primaries",
    "PrimariesT",
    "PropEnum",
    "Transfer",
    "TransferT",
]


class Matrix(_MatrixMeta):  # type: ignore[misc]
    """
    Matrix coefficients ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Table E.5).
    """

    _value_: int

    @classmethod
    def _missing_(cls: type[Matrix], value: Any) -> Matrix | None:
        value = super()._missing_(value)

        if value is None:
            return Matrix.UNKNOWN
        elif isinstance(value, cls):
            return value

        if Matrix.RGB < value < Matrix.ICTCP:
            raise ReservedMatrixError(f"Matrix({value}) is reserved.", cls)

        if value > Matrix.ICTCP:
            raise UnsupportedMatrixError(
                f"Matrix({value}) is current unsupported. "
                "If you believe this to be in error, please leave an issue "
                "in the vs-tools GitHub repository.",
                cls,
            )

        return None

    RGB = 0
    """
    ```
    # Identity
    ```
    The identity matrix.
    Typically used for GBR (often referred to as RGB); however, may also be
    used for YZX (often referred to as XYZ)
    IEC 61966-2-1 sRGB
    SMPTE ST 428-1 (2006)
    See ITU-T H.265 Equations E-31 to E-33
    """
    GBR = RGB

    BT709 = 1
    """
    ```
    Kr = 0.2126; Kb = 0.0722
    ```
    Rec. ITU-R BT.709-6
    Rec. ITU-R BT.1361-0 conventional colour gamut system and extended
    colour gamut system (historical)
    IEC 61966-2-4 xvYCC709
    SMPTE RP 177 (1993) Annex B
    """

    UNKNOWN = 2
    """
    Image characteristics are unknown or are determined by the application.
    """

    FCC = 4
    """
    ```
    KR = 0.30; KB = 0.11
    ```
    FCC Title 47 Code of Federal Regulations (2003) 73.682 (a) (20)
    See ITU-T H.265 Equations E-28 to E-30
    """

    BT470BG = 5
    """
    ```
    KR = 0.299; KB = 0.114
    ```
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

    SMPTE170M = 6
    """
    ```
    Kr = 0.299; Kb = 0.114
    ```
    (Functionally the same as [Matrix.BT470BG][vstools.Matrix.BT470BG])
    Rec. ITU-R BT.601-7 525
    Rec. ITU-R BT.1358-1 525 or 625 (historical)
    Rec. ITU-R BT.1700-0 NTSC
    SMPTE ST 170 (2004)
    See ITU-T H.265 Equations E-28 to E-30
    """
    BT601_525 = SMPTE170M

    SMPTE240M = 7
    """
    ```
    KR = 0.212; KB = 0.087
    ```
    SMPTE ST 240 (1999, historical)
    See ITU-T H.265 Equations E-28 to E-30
    """

    YCGCO = 8
    """
    ```
    KR = 0.2126; KB = 0.0722
    ```
    See Implementation And Evaluation Of Residual Color Transform For 4:4:4 RGB Lossless Coding
    """

    BT2020NCL = 9
    """
    ```
    KR = 0.2627; KB = 0.0593
    ```
    Rec. ITU-R BT.2020-2 non-constant luminance system
    Rec. ITU-R BT.2100-2 Y'CbCr
    See ITU-T H.265 Equations E-28 to E-30
    """

    BT2020CL = 10
    """
    ```
    KR = 0.2627; KB = 0.0593
    ```
    Rec. ITU-R BT.2020-2 constant luminance system
    See ITU-T H.265 Equations E-49 to E-58
    """

    CHROMANCL = 12
    """
    ```
    # See ITU-T H.265 Equations E-22 to E-27
    ```
    Chromaticity-derived non-constant luminance system
    See ITU-T H.265 Equations E-28 to E-30
    """

    CHROMACL = 13
    """
    ```
    # See ITU-T H.265 Equations E-22 to E-27
    ```
    Chromaticity-derived constant luminance system
    See ITU-T H.265 Equations E-49 to E-58
    """

    ICTCP = 14
    """
    ```
    ICtCp
    ```
    Rec. ITU-R BT.2100-2 ICTCP
    See ITU-T H.265 Equations E-62 to E-64 for `transfer_characteristics` value 16 (PQ)
    See ITU-T H.265 Equations E-65 to E-67 for `transfer_characteristics` value 18 (HLG)
    """

    @classmethod
    def is_unknown(cls, value: int | Matrix) -> bool:
        """
        Check if Matrix is Matrix.UNKNOWN.
        """

        return value == cls.UNKNOWN

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

        if width <= 1024 and height <= 576:
            if height > 486:
                return Matrix.BT470BG

            return Matrix.SMPTE170M

        return Matrix.BT709

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | vs.FrameProps, strict: bool = False, func: FuncExceptT | None = None
    ) -> Matrix:
        """
        Obtain the matrix of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the frame properties. Will ALWAYS error with Matrix.UNKNOWN.

        Returns:
            Matrix object.

        Raises:
            UndefinedMatrixError: Matrix is undefined.
            UndefinedMatrixError: Matrix can not be determined from the frameprops.
        """

        return _base_from_video(cls, src, UndefinedMatrixError, strict, func)

    @classmethod
    def from_transfer(cls, transfer: Transfer, strict: bool = False) -> Matrix:
        """
        Obtain the matrix from a Transfer object.

        Args:
            transfer: Transfer object.
            strict: Be strict about the transfer-matrix mapping. Will ALWAYS error with Transfer.UNKNOWN.

        Returns:
            Matrix object.

        Raises:
            UnsupportedTransferError: Transfer is not supported.
        """

        if transfer not in _transfer_matrix_map:
            if strict:
                raise UnsupportedTransferError(f"{transfer} is not supported!", cls.from_transfer)

            return cls(transfer.value)

        return _transfer_matrix_map[transfer]

    @classmethod
    def from_primaries(cls, primaries: Primaries, strict: bool = False) -> Matrix:
        """
        Obtain the matrix from a Primaries object.

        Args:
            primaries: Primaries object.
            strict: Be strict about the primaries-matrix mapping. Will ALWAYS error with Primaries.UNKNOWN.

        Returns:
            Matrix object.

        Raises:
            UnsupportedPrimariesError: Primaries is not supported.
        """

        if primaries not in _primaries_matrix_map:
            if strict:
                raise UnsupportedPrimariesError(f"{primaries} is not supported!", cls.from_primaries)

            return cls(primaries.value)

        return _primaries_matrix_map[primaries]

    @property
    def pretty_string(self) -> str:
        return _matrix_pretty_name_map.get(self, super().pretty_string)

    @property
    def string(self) -> str:
        return _matrix_name_map.get(self, super().string)


class Transfer(_TransferMeta):  # type: ignore[misc]
    """
    Transfer characteristics ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Table E.4).
    """

    _value_: int

    @classmethod
    def _missing_(cls: type[Transfer], value: Any) -> Transfer | None:
        value = super()._missing_(value)

        if value is None:
            return Transfer.UNKNOWN
        elif isinstance(value, cls):
            return value

        if Transfer.BT709 < value < Transfer.STD_B67 or value == 0:
            raise ReservedTransferError(f"Transfer({value}) is reserved.", cls)

        if value > Transfer.STD_B67:
            raise UnsupportedTransferError(
                f"Transfer({value}) is current unsupported. "
                "If you believe this to be in error, please leave an issue "
                "in the vs-tools GitHub repository.",
                cls,
            )

        return None

    BT709 = 1
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

    UNKNOWN = 2
    """
    Image characteristics are unknown or are determined by the application.
    """

    BT470M = 4
    """
    Rec. ITU-R BT.470-6 System M (historical)
    NTSC Recommendation for transmission standards for colour television (1953)
    FCC, Title 47 Code of Federal Regulations (2003) 73.682 (a) (20)
    """
    GAMMA22 = BT470M

    BT470BG = 5
    """
    Rec. ITU-R BT.470-6 System B, G (historical)
    Rec. ITU-R BT.1700-0 625 PAL and
    625 SECAM
    """
    GAMMA28 = BT470BG

    BT601 = 6
    """
    (Functionally the same as [Transfer.BT709][vstools.Transfer.BT709],
    [Transfer.BT2020_10][vstools.Transfer.BT2020_10],
    and [Transfer.BT2020_12][vstools.Transfer.BT2020_12])
    Rec. ITU-R BT.601-7 525 or 625
    Rec. ITU-R BT.1358-1 525 or 625 (historical)
    Rec. ITU-R BT.1700-0 NTSC
    SMPTE ST 170 (2004)
    """

    SMPTE240M = 7
    """
    SMPTE ST 240 (1999, historical).
    """

    LINEAR = 8
    """
    Linear transfer characteristics.
    """

    LOG100 = 9
    """
    Logarithmic transfer characteristic (100:1 range).
    """

    LOG316 = 10
    """
    Logarithmic transfer characteristic (100 * sqrt(10):1 range).
    """

    XVYCC = 11
    """
    IEC 61966-2-4.
    """

    SRGB = 13
    """
    IEC 61966-2-1 sRGB when matrix is equal to [Matrix.RGB][vstools.Matrix.RGB]
    IEC 61966-2-1 sYCC when matrix is equal to [Matrix.BT470BG][vstools.Matrix.BT470BG]
    """

    BT2020_10 = 14
    """
    (Functionally the same as [Transfer.BT709][vstools.Transfer.BT709], [Transfer.BT601][vstools.Transfer.BT601],
    and [Transfer.BT2020_12][vstools.Transfer.BT2020_12])
    Rec. ITU-R BT.2020-2
    """

    BT2020_12 = 15
    """
    (Functionally the same as [Transfer.BT709][vstools.Transfer.BT709], [Transfer.BT601][vstools.Transfer.BT601],
    and [Transfer.BT2020_10][vstools.Transfer.BT2020_10])
    Rec. ITU-R BT.2020-2
    """

    ST2084 = 16
    """
    SMPTE ST 2084 (2014) for 10, 12, 14, and 16-bit systems
    Rec. ITU-R BT.2100-2 perceptual quantization (PQ) system
    """
    PQ = ST2084

    STD_B67 = 18
    """
    Association of Radio Industries and Businesses (ARIB) STD-B67
    Rec. ITU-R BT.2100-2 hybrid loggamma (HLG) system
    """
    HLG = STD_B67

    # Transfer characteristics from libplacebo

    GAMMA18 = 104
    """
    Pure power gamma 1.8
    """

    GAMMA20 = 105
    """
    Pure power gamma 2.0
    """

    GAMMA26 = 108
    """
    Pure power gamma 2.6
    """

    PROPHOTO = 110
    """
    ProPhoto RGB (ROMM)
    """
    ROMM = PROPHOTO

    ST428 = 111
    """
    Digital Cinema Distribution Master (XYZ)
    """
    XYZ = ST428

    VLOG = 114
    """
    Panasonic V-Log (VARICAM)
    """
    VARICAM = VLOG

    SLOG_1 = 115
    """
    Sony S-Log1
    """

    SLOG_2 = 116
    """
    Sony S-Log2
    """

    @classmethod
    def is_unknown(cls, value: int | Transfer) -> bool:
        """
        Check if Transfer is unknown.
        """

        return value == cls.UNKNOWN

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

        if width <= 1024 and height <= 576:
            return Transfer.BT601

        return Transfer.BT709

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | vs.FrameProps, strict: bool = False, func: FuncExceptT | None = None
    ) -> Transfer:
        """
        Obtain the transfer of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the properties. The result may NOT be Transfer.UNKNOWN.

        Returns:
            Transfer object.

        Raises:
            UndefinedTransferError: Transfer is undefined.
            UndefinedTransferError: Transfer can not be determined from the frameprops.
        """

        return _base_from_video(cls, src, UndefinedTransferError, strict, func)

    @classmethod
    def from_matrix(cls, matrix: Matrix, strict: bool = False) -> Transfer:
        """
        Obtain the transfer from a Matrix object.

        Args:
            matrix: Matrix object.
            strict: Be strict about the matrix-transfer mapping. Will ALWAYS error with Matrix.UNKNOWN.

        Returns:
            Transfer object.

        Raises:
            UnsupportedMatrixError: Matrix is not supported.
        """

        if matrix not in _matrix_transfer_map:
            if strict:
                raise UnsupportedMatrixError(f"{matrix} is not supported!", cls.from_matrix)

            return cls(matrix.value)

        return _matrix_transfer_map[matrix]

    @classmethod
    def from_primaries(cls, primaries: Primaries, strict: bool = False) -> Transfer:
        """
        Obtain the transfer from a Primaries object.

        Args:
            primaries: Primaries object.
            strict: Be strict about the primaries-transfer mapping. Will ALWAYS error with Primaries.UNKNOWN.

        Returns:
            Transfer object.

        Raises:
            UnsupportedPrimariesError: Primaries is not supported.
        """

        if primaries not in _primaries_transfer_map:
            if strict:
                raise UnsupportedPrimariesError(f"{primaries} is not supported!", cls.from_primaries)

            return cls(primaries.value)

        return _primaries_transfer_map[primaries]

    @classmethod
    def from_libplacebo(cls, val: int) -> int:
        """
        Obtain the transfer from libplacebo.
        """

        return _placebo_transfer_map[val]

    @property
    def value_vs(self) -> int:
        """
        VapourSynth value.

        Raises:
            ReservedTransferError: Transfer is not an internal transfer, but a libplacebo one.
        """

        if self >= self.GAMMA18:
            raise ReservedTransferError(
                "This transfer isn't a VapourSynth internal transfer, but a libplacebo one!",
                f"{self.__class__.__name__}.value_vs",
            )

        return self.value

    @property
    def value_libplacebo(self) -> int:
        """
        libplacebo value.
        """

        return _transfer_placebo_map[self]

    @property
    def pretty_string(self) -> str:
        return _transfer_pretty_name_map.get(self, super().pretty_string)

    @property
    def string(self) -> str:
        return _transfer_name_map.get(self, super().string)


class Primaries(_PrimariesMeta):  # type: ignore[misc]
    """
    Color primaries ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Table E.3).
    """

    _value_: int

    @classmethod
    def _missing_(cls: type[Primaries], value: Any) -> Primaries | None:
        value = super()._missing_(value)

        if value is None:
            return Primaries.UNKNOWN
        elif isinstance(value, cls):
            return value

        if cls.BT709 < value < cls.JEDEC_P22:
            raise ReservedPrimariesError(f"Primaries({value}) is reserved.", cls)

        if value > cls.JEDEC_P22:
            raise UnsupportedPrimariesError(
                f"Primaries({value}) is current unsupported. "
                "If you believe this to be in error, please leave an issue "
                "in the vs-tools GitHub repository.",
                cls,
            )

        return None

    BT709 = 1
    """
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

    UNKNOWN = 2
    """
    Unspecified Image characteristics are unknown or are determined by the application.
    """

    BT470M = 4
    """
    ```
    Primary     x      y
    Green    0.2100 0.7100
    Blue     0.1400 0.0800
    Red      0.6700 0.3300
    White C  0.3100 0.3160
    ```

    Rec. ITU-R BT.470-6 System M (historical)
    NTSC Recommendation for transmission
    standards for colour television (1953)
    FCC Title 47 Code of Federal Regulations (2003)
    73.682 (a) (20)
    """

    BT470BG = 5
    """
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

    SMPTE170M = 6
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

    SMPTE240M = 7
    """
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

    FILM = 8
    """
    ```
    Primary    x      y
    Green   0.2430 0.6920 #(Wratten 58)
    Blue    0.1450 0.0490 #(Wratten 47)
    Red     0.6810 0.3190 #(Wratten 25)
    White C 0.3100 0.3160
    ```

    Generic film (colour filters using Illuminant C)
    """

    BT2020 = 9
    """
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

    ST428 = 10
    """
    ```
    Primary        x   y
    Green    (Y)  0.0 1.0
    Blue     (Z)  0.0 0.0
    Red      (X)  1.0 0.0
    Centre White  1/3 1/3
    ```

    SMPTE ST 428-1 (2006)
    (CIE 1931 XYZ)
    """
    XYZ = ST428
    CIE1931 = ST428

    ST431_2 = 11
    """
    ```
    Primary    x      y
    Green   0.2650 0.6900
    Blue    0.1500 0.0600
    Red     0.6800 0.3200
    White   0.3140 0.3510
    ```

    SMPTE RP 431-2 (2011)
    SMPTE ST 2113 (2019) "P3DCI"
    """
    DCI_P3 = ST431_2

    ST432_1 = 12
    """
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

    JEDEC_P22 = 22
    """
    ```
    Primary      x      y
    Green     0.2950 0.6050
    Blue      0.1550 0.0770
    Red       0.6300 0.3400
    White D65 0.3127 0.3290
    ```

    EBU Tech. 3213-E (1975)
    """
    EBU3213 = JEDEC_P22

    # Primary characteristics from libplacebo

    APPLE = 107
    """
    Apple RGB.
    """

    ADOBE = 108
    """
    Adobe RGB (1998).
    """

    PROPHOTO = 109
    """
    ProPhoto RGB (ROMM).
    """
    ROMM = PROPHOTO

    VGAMUT = 113
    """
    Panasonic V-Gamut (VARICAM).
    """
    VARICAM = VGAMUT

    SGAMUT = 114
    """
    Sony S-Gamut.
    """

    ACES_0 = 116
    """
    ACES Primaries #0 (ultra wide)
    """

    ACES_1 = 117
    """
    ACES Primaries #1
    """

    @classmethod
    def is_unknown(cls, value: int | Primaries) -> bool:
        """
        Check if Primaries is unknown.
        """

        return value == cls.UNKNOWN

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

        if width <= 1024 and height <= 576:
            if height > 486:
                return Primaries.BT470BG

            return Primaries.SMPTE170M

        return Primaries.BT709

    @classmethod
    def from_video(
        cls, src: vs.VideoNode | vs.VideoFrame | vs.FrameProps, strict: bool = False, func: FuncExceptT | None = None
    ) -> Primaries:
        """
        Obtain the primaries of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the frame properties. Will ALWAYS error with Primaries.UNKNOWN.

        Returns:
            Primaries object.

        Raises:
            UndefinedPrimariesError: Primaries is undefined.
            UndefinedPrimariesError: Primaries can not be determined from the frame properties.
        """

        return _base_from_video(cls, src, UndefinedPrimariesError, strict, func)

    @classmethod
    def from_matrix(cls, matrix: Matrix, strict: bool = False) -> Primaries:
        """
        Obtain the primaries from a Matrix object.

        Args:
            matrix: Matrix object.
            strict: Be strict about the matrix-primaries mapping. Will ALWAYS error with Matrix.UNKNOWN.

        Returns:
            Primaries object.

        Raises:
            UnsupportedMatrixError: Matrix is not supported.
        """

        if matrix not in _matrix_primaries_map:
            if strict:
                raise UnsupportedMatrixError(f"{matrix} is not supported!", cls.from_matrix)

            return cls(matrix.value)

        return _matrix_primaries_map[matrix]

    @classmethod
    def from_transfer(cls, transfer: Transfer, strict: bool = False) -> Primaries:
        """
        Obtain the primaries from a Transfer object.

        Args:
            transfer: Transfer object.
            strict: Be strict about the transfer-primaries mapping. Will ALWAYS error with Transfer.UNKNOWN.

        Returns:
            Matrix object.

        Raises:
            UnsupportedTransferError: Transfer is not supported.
        """

        if transfer not in _transfer_primaries_map:
            if strict:
                raise UnsupportedTransferError(f"{transfer} is not supported!", cls.from_transfer)

            return cls(transfer.value)

        return _transfer_primaries_map[transfer]

    @classmethod
    def from_libplacebo(cls, val: int) -> int:
        """
        Obtain the primaries from libplacebo.
        """

        return _placebo_primaries_map[val]

    @property
    def value_vs(self) -> int:
        """
        VapourSynth value.

        Raises:
            ReservedPrimariesError: Primaries are not an internal primaries, but a libplacebo one.
        """

        if self >= self.APPLE:
            raise ReservedPrimariesError(
                "This primaries isn't a VapourSynth internal primaries, but a libplacebo one!",
                f"{self.__class__.__name__}.value_vs",
            )

        return self.value

    @property
    def value_libplacebo(self) -> int:
        """
        libplacebo value.
        """

        return _primaries_placebo_map[self]

    @property
    def pretty_string(self) -> str:
        return _primaries_pretty_name_map.get(self, super().pretty_string)

    @property
    def string(self) -> str:
        return _primaries_name_map.get(self, super().string)


class ColorRange(_ColorRangeMeta):  # type: ignore[misc]
    """
    Pixel Range ([ITU-T H.265](https://www.itu.int/rec/T-REC-H.265) Equations E-10 through E-20.
    """

    _value_: int

    @classmethod
    def _missing_(cls: type[ColorRange], value: Any) -> ColorRange | None:
        value = super()._missing_(value)

        if value is None:
            return ColorRange.LIMITED
        elif isinstance(value, cls):
            return value

        if value > ColorRange.LIMITED:
            raise UnsupportedPrimariesError(f"ColorRange({value}) is unsupported.", cls)

        return None

    LIMITED = 1
    """
    Studio (TV) legal range, 16-235 in 8 bits.

    | This is primarily used with YUV integer formats.
    """
    TV = LIMITED

    FULL = 0
    """
    Full (PC) dynamic range, 0-255 in 8 bits.

    | Note that float clips should ALWAYS be FULL range!
    | RGB clips will ALWAYS be FULL range!
    """
    PC = FULL

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
        cls, src: vs.VideoNode | vs.VideoFrame | vs.FrameProps, strict: bool = False, func: FuncExceptT | None = None
    ) -> ColorRange:
        """
        Obtain the color range of a clip from the frame properties.

        Args:
            src: Input clip, frame, or props.
            strict: Be strict about the frame properties. Sets the ColorRange as MISSING if prop is not there.

        Returns:
            ColorRange object.
        """

        return _base_from_video(cls, src, UnsupportedColorRangeError, strict, func)

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

    @property
    def is_limited(self) -> bool:
        """
        Check if ColorRange is limited.
        """

        return bool(self.value)

    @property
    def is_full(self) -> bool:
        """
        Check if ColorRange is full.
        """

        return not self.value


_transfer_matrix_map: dict[Transfer, Matrix] = {
    Transfer.SRGB: Matrix.RGB,
    Transfer.BT709: Matrix.BT709,
    Transfer.BT470BG: Matrix.BT470BG,
    Transfer.ST2084: Matrix.BT2020NCL,
    Transfer.BT2020_10: Matrix.BT2020NCL,
    Transfer.BT2020_12: Matrix.BT2020NCL,
    Transfer.STD_B67: Matrix.BT2020NCL,
}

_primaries_matrix_map: dict[Primaries, Matrix] = {}

_matrix_transfer_map = {
    Matrix.RGB: Transfer.SRGB,
    Matrix.BT709: Transfer.BT709,
    Matrix.SMPTE170M: Transfer.BT601,
    Matrix.SMPTE240M: Transfer.SMPTE240M,
    Matrix.ICTCP: Transfer.BT2020_10,
}

_primaries_transfer_map: dict[Primaries, Transfer] = {}

_matrix_primaries_map: dict[Matrix, Primaries] = {}

_transfer_primaries_map: dict[Transfer, Primaries] = {}

_transfer_placebo_map = {
    Transfer.UNKNOWN: 0,
    Transfer.BT601: 1,
    Transfer.BT709: 1,
    Transfer.SRGB: 2,
    Transfer.LINEAR: 3,
    Transfer.GAMMA18: 4,
    Transfer.GAMMA20: 5,
    Transfer.BT470M: 6,
    Transfer.GAMMA26: 8,
    Transfer.BT470BG: 9,
    Transfer.PROPHOTO: 10,
    Transfer.ST428: 11,
    Transfer.ST2084: 12,
    Transfer.STD_B67: 13,
    Transfer.VLOG: 14,
    Transfer.SLOG_1: 15,
    Transfer.SLOG_2: 16,
}

_primaries_placebo_map = {
    Primaries.UNKNOWN: 0,
    Primaries.SMPTE170M: 1,
    Primaries.BT470BG: 2,
    Primaries.BT709: 3,
    Primaries.BT470M: 4,
    Primaries.JEDEC_P22: 5,
    Primaries.BT2020: 6,
    Primaries.APPLE: 7,
    Primaries.ADOBE: 8,
    Primaries.PROPHOTO: 9,
    Primaries.ST428: 10,
    Primaries.ST431_2: 11,
    Primaries.ST432_1: 12,
    Primaries.VGAMUT: 13,
    Primaries.SGAMUT: 14,
    Primaries.FILM: 15,
    Primaries.ACES_0: 16,
    Primaries.ACES_1: 17,
}

_placebo_transfer_map = {value: key for key, value in _transfer_placebo_map.items()}

_placebo_primaries_map = {value: key for key, value in _primaries_placebo_map.items()}

_matrix_name_map = {
    Matrix.RGB: "gbr",
    Matrix.BT709: "bt709",
    Matrix.UNKNOWN: "unknown",
    Matrix.FCC: "fcc",
    Matrix.BT470BG: "bt470bg",
    Matrix.SMPTE170M: "smpte170m",
    Matrix.SMPTE240M: "smpte240m",
    Matrix.YCGCO: "ycgco",
    Matrix.BT2020NCL: "bt2020nc",
    Matrix.BT2020CL: "bt2020c",
    Matrix.CHROMANCL: "chroma-derived-nc",
    Matrix.CHROMACL: "chroma-derived-c",
    Matrix.ICTCP: "ictcp",
}

_transfer_name_map = {
    Transfer.BT709: "bt709",
    Transfer.UNKNOWN: "unknown",
    Transfer.BT470M: "bt470m",
    Transfer.BT470BG: "bt470bg",
    Transfer.BT601: "smpte170m",
    Transfer.SMPTE240M: "smpte240m",
    Transfer.LINEAR: "linear",
    Transfer.LOG100: "log100",
    Transfer.LOG316: "log316",
    Transfer.XVYCC: "iec61966-2-4",
    Transfer.SRGB: "iec61966-2-1",
    Transfer.BT2020_10: "bt2020-10",
    Transfer.BT2020_12: "bt2020-12",
    Transfer.ST2084: "smpte2084",
    Transfer.STD_B67: "arib-std-b67",
}

_primaries_name_map = {
    Primaries.BT709: "bt709",
    Primaries.UNKNOWN: "unknown",
    Primaries.BT470M: "bt470m",
    Primaries.BT470BG: "bt470bg",
    Primaries.SMPTE170M: "smpte170m",
    Primaries.SMPTE240M: "smpte240m",
    Primaries.FILM: "film",
    Primaries.BT2020: "bt2020",
    Primaries.ST428: "smpte428",
    Primaries.ST431_2: "smpte431",
    Primaries.ST432_1: "smpte432",
    Primaries.JEDEC_P22: "jedec-p22",
}

_matrix_pretty_name_map = {
    Matrix.RGB: "RGB",
    Matrix.BT709: "BT.709",
    Matrix.FCC: "FCC",
    Matrix.BT470BG: "BT.470bg",
    Matrix.SMPTE170M: "SMPTE ST 170m",
    Matrix.SMPTE240M: "SMPTE ST 240m",
    Matrix.YCGCO: "YCgCo",
    Matrix.BT2020NCL: "BT.2020 non-constant luminance",
    Matrix.BT2020CL: "BT.2020 constant luminance",
    Matrix.CHROMANCL: "Chromaticity derived non-constant luminance",
    Matrix.CHROMACL: "Chromaticity derived constant luminance",
    Matrix.ICTCP: "ICtCp",
}

_transfer_pretty_name_map = {
    Transfer.BT709: "BT.709",
    Transfer.BT470M: "BT.470m",
    Transfer.BT470BG: "BT.470bg",
    Transfer.BT601: "BT.601",
    Transfer.SMPTE240M: "SMPTE ST 240m",
    Transfer.LINEAR: "Linear",
    Transfer.LOG100: "Log 1:100 contrast",
    Transfer.LOG316: "Log 1:316 contrast",
    Transfer.XVYCC: "xvYCC",
    Transfer.SRGB: "sRGB",
    Transfer.BT2020_10: "BT.2020 10 bits",
    Transfer.BT2020_12: "BT.2020 12 bits",
    Transfer.ST2084: "SMPTE ST 2084 (PQ)",
    Transfer.STD_B67: "ARIB std-b67 (HLG)",
}

_primaries_pretty_name_map = {
    Primaries.BT709: "BT.709",
    Primaries.BT470M: "BT.470m",
    Primaries.BT470BG: "BT.470bg",
    Primaries.SMPTE170M: "SMPTE ST 170m",
    Primaries.SMPTE240M: "SMPTE ST 240m",
    Primaries.FILM: "Film",
    Primaries.BT2020: "BT.2020",
    Primaries.ST428: "SMPTE ST 428 (XYZ)",
    Primaries.ST431_2: "DCI-P3, DCI white point",
    Primaries.ST432_1: "DCI-P3 D65 white point",
    Primaries.JEDEC_P22: "JEDEC P22 (EBU 3213-E)",
}


MatrixT: TypeAlias = Union[int, vs.MatrixCoefficients, Matrix, HoldsPropValueT]
"""Type alias for values that can be used to initialize a [Matrix][vstools.Matrix]."""

TransferT: TypeAlias = Union[int, vs.TransferCharacteristics, Transfer, HoldsPropValueT]
"""Type alias for values that can be used to initialize a [Transfer][vstools.Transfer]."""

PrimariesT: TypeAlias = Union[int, vs.ColorPrimaries, Primaries, HoldsPropValueT]
"""Type alias for values that can be used to initialize a [Primaries][vstools.Primaries]."""

ColorRangeT: TypeAlias = Union[int, vs.ColorRange, ColorRange, HoldsPropValueT]
"""Type alias for values that can be used to initialize a [ColorRange][vstools.ColorRange]."""


def _norm_props_enums(kwargs: KwargsT) -> KwargsT:
    return {
        key: (
            (value.value_zimg if hasattr(value, "value_zimg") else int(value))  # pyright: ignore[reportAttributeAccessIssue]
            if isinstance(value, PropEnum)
            else value
        )
        for key, value in kwargs.items()
    }
