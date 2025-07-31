from dataclasses import asdict, dataclass
from os import SEEK_SET
from typing import Any

from .c_adt import CADT
from .sector import SectorReadHelper
from .vts_pgci import VTSPgci
from .vtsi_mat import VTSIMat

__all__ = ["IFO0", "IFOX", "IFO0Title", "PTTInfo", "to_json"]


@dataclass
class IFO0Title:
    title_set_nr: int
    title_set_sector: int
    nr_of_angles: int
    nr_of_ptts: int
    vts_ttn: int


@dataclass
class IFO0:
    num_vts: int
    tt_srpt: list[IFO0Title]

    def __init__(self, reader: SectorReadHelper):
        reader.ifo.seek(0x3E, SEEK_SET)
        (self.num_vts,) = reader._unpack_byte(2)

        # tt_srpt
        reader._goto_sector_ptr(0x00C4)
        num_title, *_ = reader._unpack_byte(2, 2, 4)

        tt_srpt = []
        for _ in range(num_title):
            _, nr_of_angles, nr_of_ptts, _, title_set_nr, vts_ttn, sector = reader._unpack_byte(1, 1, 2, 2, 1, 1, 4)

            tt_srpt += [IFO0Title(title_set_nr, sector, nr_of_angles, nr_of_ptts, vts_ttn)]
        self.tt_srpt = tt_srpt


@dataclass
class PTTInfo:
    pgn: int
    pgcn: int


@dataclass
class IFOX:
    vtsi_mat: VTSIMat
    vts_c_adt: CADT
    vts_vobu_admap: list[int]
    vts_ptt_srpt: list[list[PTTInfo]]
    vts_pgci: VTSPgci

    def __init__(self, reader: SectorReadHelper) -> None:
        self._vobu_admap(reader)
        self._vts_ptt_srpt(reader)
        self.vts_c_adt = CADT(reader)
        self.vtsi_mat = VTSIMat(reader)
        self.vts_pgci = VTSPgci(reader)

    def _vobu_admap(self, reader: SectorReadHelper) -> None:
        reader._goto_sector_ptr(0x00E4)
        (end,) = reader._unpack_byte(4)

        self.vts_vobu_admap = []
        cnt = (end + 1 - 4) // 4
        for _ in range(cnt):
            self.vts_vobu_admap += [reader._unpack_byte(4)[0]]

    def _vts_ptt_srpt(self, reader: SectorReadHelper) -> None:
        reader._goto_sector_ptr(0x00C8)
        num, _res, end = reader._unpack_byte(2, 2, 4)

        # not really sure with this
        correction = num * 4 + 8
        offsets = [x - correction for x in reader._unpack_byte(4, repeat=num)]

        total_ptts = (end - correction + 4 + 1 - 4) // 4

        all_ptts_x = list(reader._unpack_byte(2, 2, repeat=total_ptts))
        all_ptts = [(all_ptts_x[i * 2 + 0], all_ptts_x[i * 2 + 1]) for i in range(len(all_ptts_x) // 2)]

        offsets = [a // 4 for a in offsets] + [len(all_ptts)]

        titles = [[PTTInfo(p[1], p[0]) for p in all_ptts[offsets[a] : offsets[a + 1]]] for a in range(num)]

        self.vts_ptt_srpt = titles


def to_json(ifo0: IFO0, vts: list[IFOX]) -> dict[str, list[dict[str, Any]]]:
    crnt = dict[str, list[dict[str, Any]]]()

    # ifo0
    i0 = dict[str, list[dict[str, Any]]]()
    i0["tt_srpt"] = [asdict(a) for a in ifo0.tt_srpt]
    i0["pgci_ut"] = []
    i0["vts_c_adt"] = []
    i0["vts_pgcit"] = []
    i0["vts_ptt_srpt"] = []
    jj = [asdict(a) for a in vts]
    for ad in jj:
        cadt = ad["vts_c_adt"]
        ad["vts_c_adt"] = cadt["cell_adr_table"]
        ad["vts_pgcit"] = ad["vts_pgci"]["pgcs"]
        del ad["vts_pgci"]
        ad["pgci_ut"] = []
        ad["tt_srpt"] = []

    crnt["ifos"] = [i0, *jj]

    return crnt
