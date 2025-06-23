from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Union

from vstools import SPath

__all__ = [
    "AllNeddedDvdFrameData",
    "D2VIndexFileInfo",
    "D2VIndexFrameData",
    "D2VIndexHeader",
    "DGIndexFileInfo",
    "DGIndexFooter",
    "DGIndexFrameData",
    "DGIndexHeader",
    "IndexFileFrameData",
    "IndexFileInfo",
]


class _SetItemMeta:
    def __setitem__(self, key: str, value: float | int) -> None:
        return self.__setattr__(key, value)


@dataclass
class IndexFileFrameData(_SetItemMeta):
    matrix: int
    pic_type: str


@dataclass
class _IndexFileInfoBase(_SetItemMeta):
    path: SPath
    file_idx: int


@dataclass
class IndexFileInfo(_IndexFileInfoBase):
    frame_data: list[IndexFileFrameData]


@dataclass
class D2VIndexHeader(_SetItemMeta):
    stream_type: int = 0
    MPEG_type: int = 0
    iDCT_algorithm: int = 0  # noqa: N815
    YUVRGB_scale: int = 1
    luminance_filter: tuple[int, ...] = (0, 0)
    clipping: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    aspect: Fraction = Fraction(16, 9)
    pic_size: str = ""
    field_op: int = 0
    frame_rate: Fraction = Fraction(30000, 1001)
    location: list[int] = field(default_factory=lambda: [0, 0, 0, 0])


@dataclass
class D2VIndexFrameData(IndexFileFrameData):
    vob: int
    cell: int
    info: int
    skip: int
    position: int
    frameflags: list[int]


@dataclass
class DGIndexHeader(_SetItemMeta):
    device: int = 0
    decode_modes: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    stream: tuple[int, ...] = (1, 0)
    ranges: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    depth: int = 8
    aspect: Fraction = Fraction(16, 9)
    colorimetry: tuple[int, ...] = (2, 2, 2)
    packet_size: int | None = None
    vpid: int | None = None


@dataclass
class DGIndexFrameData(IndexFileFrameData):
    vob: int | None
    cell: int | None


@dataclass
class DGIndexFooter(_SetItemMeta):
    film: float = 0.0
    frames_coded: int = 0
    frames_playback: int = 0
    order: int = 0


@dataclass
class D2VIndexFileInfo(_IndexFileInfoBase):
    header: D2VIndexHeader
    frame_data: list[D2VIndexFrameData]


@dataclass
class DGIndexFileInfo(_IndexFileInfoBase):
    header: DGIndexHeader
    frame_data: list[DGIndexFrameData]
    footer: DGIndexFooter


IndexFileType = Union[D2VIndexFileInfo, DGIndexFileInfo]


@dataclass
class AllNeddedDvdFrameData:
    vobids: list[tuple[int, int]]
    tff: list[int]
    rff: list[int]
    prog: list[int]
    progseq: list[int]
