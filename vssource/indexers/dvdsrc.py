from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from vstools import SPath, core, get_prop, vs

from ..dataclasses import AllNeddedDvdFrameData
from ..rff import apply_rff_array, apply_rff_video
from .base import DVDIndexer

if TYPE_CHECKING:
    from ..formats.dvd.parsedvd import IFOX, IFO0Title


__all__ = ["DVDSRCIndexer"]


def get_sectorranges_for_vobcellpair(current_vts: IFOX, pair_id: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        (e.start_sector, e.last_sector)
        for e in current_vts.vts_c_adt.cell_adr_table
        if (e.vob_id, e.cell_id) == pair_id
    ]


class DVDSRCIndexer(DVDIndexer):
    def parse_vts(
        self,
        title: IFO0Title,
        disable_rff: bool,
        vobidcellids_to_take: list[tuple[int, int]],
        target_vts: IFOX,
        output_folder: SPath,
        vob_input_files: Sequence[SPath],
    ) -> tuple[vs.VideoNode, list[int], list[tuple[int, int]], list[int]]:
        admap = target_vts.vts_vobu_admap

        all_ranges = [x for a in vobidcellids_to_take for x in get_sectorranges_for_vobcellpair(target_vts, a)]

        vts_indices = list[int]()
        for a in all_ranges:
            start_index = admap.index(a[0])

            try:
                end_index = admap.index(a[1] + 1) - 1
            except ValueError:
                end_index = len(admap) - 1

            vts_indices.extend([start_index, end_index])

        rawnode = core.dvdsrc2.FullVts(str(self.iso_path), vts=title.title_set_nr, ranges=vts_indices)
        staff = self._extract_data(rawnode)

        if not disable_rff:
            rnode = apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)
            _vobids = apply_rff_array(staff.vobids, staff.rff, staff.tff, staff.progseq)
        else:
            rnode = rawnode
            _vobids = staff.vobids

        return rnode, staff.rff, _vobids, vts_indices

    def _extract_data(self, rawnode: vs.VideoNode) -> AllNeddedDvdFrameData:
        dd = bytes(get_prop(rawnode, "InfoFrame", vs.VideoFrame)[0])

        assert len(dd) == len(rawnode) * 4

        vobids = list[tuple[int, int]]()
        tff = list[int]()
        rff = list[int]()
        prog = list[int]()
        progseq = list[int]()

        for i in range(len(rawnode)):
            sb = dd[i * 4 + 0]

            vobids += [((dd[i * 4 + 1] << 8) + dd[i * 4 + 2], dd[i * 4 + 3])]
            tff += [(sb & (1 << 0)) >> 0]
            rff += [(sb & (1 << 1)) >> 1]
            prog += [(sb & (1 << 2)) >> 2]
            progseq += [(sb & (1 << 3)) >> 3]

        return AllNeddedDvdFrameData(vobids, tff, rff, prog, progseq)
