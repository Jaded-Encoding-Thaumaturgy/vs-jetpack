from dataclasses import dataclass

from .sector import SectorReadHelper

__all__ = ["CADT", "CellAdr"]


@dataclass
class CellAdr:
    vob_id: int
    cell_id: int
    start_sector: int
    last_sector: int


@dataclass
class CADT:
    vob_count: int
    cell_adr_table: list[CellAdr]

    def __init__(self, reader: SectorReadHelper):
        reader._goto_sector_ptr(0x00E0)
        self.vob_count, _, end = reader._unpack_byte(2, 2, 4)

        self.cell_adr_table = list[CellAdr]()
        cnt = (end + 1 - 6) // 12

        for _ in range(cnt):
            vob_id, cell_id, __, start_sector, last_sector = reader._unpack_byte(2, 1, 1, 4, 4)
            self.cell_adr_table.append(CellAdr(vob_id, cell_id, start_sector, last_sector))
