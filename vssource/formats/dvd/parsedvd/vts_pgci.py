from dataclasses import dataclass
from os import SEEK_SET

from .sector import SectorReadHelper
from .timespan import TimeSpan

__all__ = [
    "BLOCK_MODE_FIRST_CELL",
    "BLOCK_MODE_IN_BLOCK",
    "BLOCK_MODE_LAST_CELL",
    "PGC",
    "AudioControl",
    "CellPlayback",
    "CellPosition",
    "VTSPgci",
]

BLOCK_MODE_FIRST_CELL = 1
BLOCK_MODE_IN_BLOCK = 2
BLOCK_MODE_LAST_CELL = 3


@dataclass
class CellPlayback:
    interleaved: bool
    seamless_play: bool
    seamless_angle: bool
    block_mode: int
    block_type: int
    playback_time: TimeSpan
    first_sector: int
    last_sector: int
    first_ilvu_end_sector: int
    last_vobu_start_sector: int


@dataclass
class CellPosition:
    cell_nr: int
    vob_id_nr: int


@dataclass
class AudioControl:
    available: bool
    number: int


@dataclass
class PGC:
    program_map: list[int]
    cell_playback: list[CellPlayback]
    cell_position: list[CellPosition]

    nr_of_cells: int
    nr_of_programs: int
    next_pgc_nr: int
    prev_pgc_nr: int
    goup_pgc_nr: int
    audio_control: list[AudioControl]


@dataclass
class VTSPgci:
    pgcs: list[PGC]

    def __init__(self, reader: SectorReadHelper):
        reader._goto_sector_ptr(0x00CC)

        posn = reader.ifo.tell()

        nr_pgcs, *_ = reader._unpack_byte(2, 2, 4)

        self.pgcs = list[PGC]()

        for _ in range(nr_pgcs):
            _, offset = reader._unpack_byte(4, 4)
            bk = reader.ifo.tell()

            audio_control = list[AudioControl]()

            pgc_base = posn + offset

            reader.ifo.seek(pgc_base, SEEK_SET)

            _, num_programs, num_cells = reader._unpack_byte(2, 1, 1)
            reader._unpack_byte(4, 4)

            for _ in range(8):
                ac, _ = reader._unpack_byte(1, 1)

                available = (ac & 0x80) != 0
                number = ac & 7

                audio_control.append(AudioControl(available=available, number=number))

            reader._unpack_byte(4, repeat=32)

            next_pgcn, prev_pgcn, group_pgcn = reader._unpack_byte(2, 2, 2)

            reader._unpack_byte(1, 1)

            reader._unpack_byte(4, repeat=16)

            _, offset_program, offset_playback, offset_position = reader._unpack_byte(2, 2, 2, 2)

            reader.ifo.seek(pgc_base + offset_program, SEEK_SET)

            program_map = list(reader._unpack_byte(1, repeat=num_programs))

            reader.ifo.seek(pgc_base + offset_position, SEEK_SET)

            cell_position_bytes = [reader._unpack_byte(2, 1, 1) for _ in range(num_cells)]
            cell_position = [CellPosition(cell_nr=a[2], vob_id_nr=a[0]) for a in cell_position_bytes]

            reader.ifo.seek(pgc_base + offset_playback, SEEK_SET)

            cell_playback_bytes = [reader._unpack_byte(1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4) for _ in range(num_cells)]

            cell_playback = [
                CellPlayback(
                    interleaved=(a[0] & 0b100) != 0,
                    seamless_play=(a[0] & 0b1000) != 0,
                    seamless_angle=(a[0] & 0b1) != 0,
                    block_mode=((a[0] & 0b11000000) >> 6),
                    block_type=((a[0] & 0b00110000) >> 4),
                    playback_time=TimeSpan(*a[4:8]),
                    first_sector=a[5 + 3],
                    last_sector=a[8 + 3],
                    first_ilvu_end_sector=a[6 + 3],
                    last_vobu_start_sector=a[7 + 3],
                )
                for a in cell_playback_bytes
            ]

            reader.ifo.seek(bk, SEEK_SET)

            self.pgcs.append(
                PGC(
                    nr_of_cells=num_cells,
                    nr_of_programs=num_programs,
                    next_pgc_nr=next_pgcn,
                    prev_pgc_nr=prev_pgcn,
                    goup_pgc_nr=group_pgcn,
                    program_map=program_map,
                    cell_position=cell_position,
                    cell_playback=cell_playback,
                    audio_control=audio_control,
                )
            )
