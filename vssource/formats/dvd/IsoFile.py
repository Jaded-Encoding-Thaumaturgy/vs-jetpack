# noqa: N999

import warnings
from datetime import timedelta
from fractions import Fraction
from itertools import count
from typing import cast

from jetpytools import CustomValueError, DependencyNotFoundError, SPath

from vstools import Region, core, get_prop, vs

from ...dataclasses import AllNeddedDvdFrameData
from ...rff import apply_rff_array, apply_rff_video
from .parsedvd import (
    AUDIO_FORMAT_AC3,
    AUDIO_FORMAT_LPCM,
    BLOCK_MODE_FIRST_CELL,
    BLOCK_MODE_IN_BLOCK,
    BLOCK_MODE_LAST_CELL,
    IFO0,
    IFOX,
    IFO0Title,
    SectorReadHelper,
)
from .title import Title
from .utils import absolute_time_from_timecode

__all__ = ["IsoFile"]


class IsoFile:
    ifo0: IFO0
    vts: list[IFOX]
    title_count: int

    def __init__(
        self,
        path: SPath | str,
    ):
        if not hasattr(core, "dvdsrc2"):
            raise DependencyNotFoundError(
                self.__class__, "", "dvdsrc2 is needed for {cfunc} to work!", cfunc=self.__class__
            )

        self.iso_path = SPath(path).absolute()

        if not self.iso_path.exists():
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD!', str(path), self.__class__)

        def _getifo(i: int) -> bytes:
            return cast(bytes, core.dvdsrc2.Ifo(str(self.iso_path), i))

        self.ifo0 = IFO0(SectorReadHelper(_getifo(0)))
        self.vts = [IFOX(SectorReadHelper(_getifo(i))) for i in range(1, self.ifo0.num_vts + 1)]

        self.title_count = len(self.ifo0.tt_srpt)

    def get_vts(self, title_set_nr: int = 1, apply_rff: bool = True) -> vs.VideoNode:
        fullvts = core.dvdsrc2.FullVts(str(self.iso_path), vts=title_set_nr)
        if apply_rff:
            return fullvts
        else:
            staff = dvdsrc_extract_data(fullvts)

            return apply_rff_video(fullvts, staff.rff, staff.tff, staff.prog, staff.progseq)

    def get_title(self, title_nr: int = 1, angle_nr: int | None = None, rff_mode: int = 0) -> Title:
        """
        Gets a title.

        Args:
            title_nr: Title index, 1-index based.
            angle_nr: Angle index, 1-index based.
            rff_mode: 0 Apply rff soft telecine (default); 1 Calculate per frame durations based on rff; 2 Set average
                fps on global clip;
        """
        # TODO: assert angle_nr range
        disable_rff = rff_mode >= 1

        tt_srpt = self.ifo0.tt_srpt
        title_idx = title_nr - 1

        if title_idx < 0 or title_idx >= len(tt_srpt):
            raise CustomValueError('"title_idx" out of range', self.get_title)

        tt = tt_srpt[title_idx]

        if tt.nr_of_angles != 1 and angle_nr is None:
            raise CustomValueError("No angle_nr given for multi angle title", self.get_title)

        target_vts = self.vts[tt.title_set_nr - 1]
        target_title = target_vts.vts_ptt_srpt[tt.vts_ttn - 1]

        assert len(target_title) == tt.nr_of_ptts

        for ptt in target_title[1:]:
            if ptt.pgcn != target_title[0].pgcn:
                warnings.warn("Title is not one program chain (currently untested)")

        vobidcellids_to_take = list[tuple[int, int]]()
        is_chapter = list[bool]()

        i = 0
        while i < len(target_title):
            ptt_to_take_for_pgc = len([ppt for ppt in target_title[i:] if target_title[i].pgcn == ppt.pgcn])

            assert ptt_to_take_for_pgc >= 1

            title_programs = [a.pgn for a in target_title[i : i + ptt_to_take_for_pgc]]
            target_pgc = target_vts.vts_pgci.pgcs[target_title[i].pgcn - 1]
            pgc_programs = target_pgc.program_map

            if title_programs[0] != 1 or pgc_programs[0] != 1:
                warnings.warn("Open Title does not start at the first cell\n")

            target_programs = [
                a[1] for a in list(filter(lambda x: (x[0] + 1) in title_programs, enumerate(pgc_programs)))
            ]

            if target_programs != pgc_programs:
                warnings.warn("The program chain does not include all ptts\n")

            current_angle = 1

            for cell_i in range(len(target_pgc.cell_position)):
                cell_position = target_pgc.cell_position[cell_i]
                cell_playback = target_pgc.cell_playback[cell_i]

                block_mode = cell_playback.block_mode

                if block_mode == BLOCK_MODE_FIRST_CELL:
                    current_angle = 1
                    angle_start_cell_i = cell_i
                elif block_mode in (BLOCK_MODE_IN_BLOCK, BLOCK_MODE_LAST_CELL):
                    current_angle += 1

                if block_mode == 0:
                    take_cell = True
                    angle_start_cell_i = cell_i
                else:
                    take_cell = current_angle == angle_nr

                if take_cell:
                    vobidcellids_to_take += [(cell_position.vob_id_nr, cell_position.cell_nr)]
                    is_chapter += [(angle_start_cell_i + 1) in target_programs]  # pyright: ignore[reportPossiblyUnboundVariable]

            i += ptt_to_take_for_pgc

        assert len(is_chapter) == len(vobidcellids_to_take)

        rnode, rff, vobids, dvdsrc_ranges = dvdsrc_parse_vts(
            self.iso_path,
            tt,
            disable_rff,
            vobidcellids_to_take,
            target_vts,
        )

        region = Region.from_framerate(rnode.fps)
        rfps = region.framerate

        if not disable_rff:
            rnode = core.std.AssumeFPS(rnode, fpsnum=rfps.numerator, fpsden=rfps.denominator)
            durationcodes = [Fraction(rfps.denominator, rfps.numerator)] * len(rnode)
            absolutetime = [a * (rfps.denominator / rfps.numerator) for a in range(len(rnode))]
        else:
            if rff_mode == 1:
                durationcodes = timecodes = [Fraction(rfps.denominator * (a + 2), rfps.numerator * 2) for a in rff]
                absolutetime = absolute_time_from_timecode(timecodes)

                def _apply_timecodes(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
                    f = f.copy()

                    f.props._DurationNum = timecodes[n].numerator
                    f.props._DurationDen = timecodes[n].denominator
                    f.props._AbsoluteTime = absolutetime[n]

                    return f

                rnode = rnode.std.ModifyFrame(rnode, _apply_timecodes)
            else:
                rffcnt = len([a for a in rff if a])

                asd = (rffcnt * 3 + 2 * (len(rff) - rffcnt)) / len(rff)

                fcc = len(rnode) * 5
                new_fps = Fraction(
                    rfps.numerator * fcc * 2,
                    int(fcc * rfps.denominator * asd),
                )

                rnode = core.std.AssumeFPS(rnode, fpsnum=new_fps.numerator, fpsden=new_fps.denominator)

                durationcodes = timecodes = [Fraction(rfps.denominator * (a + 2), rfps.numerator * 2) for a in rff]
                absolutetime = absolute_time_from_timecode(timecodes)

        changes = [*(i for i, pvob, nvob in zip(count(1), vobids[:-1], vobids[1:]) if nvob != pvob), len(rnode)]

        assert len(changes) == len(is_chapter)

        last_chapter_i = next((i for i, c in reversed(list(enumerate(is_chapter))) if c), 0)

        output_chapters = list[int]()
        for i in range(len(is_chapter)):
            if not is_chapter[i]:
                continue

            for j in range(i + 1, len(is_chapter)):
                if is_chapter[j]:
                    output_chapters.append(changes[j - 1])
                    break
            else:
                output_chapters.append(changes[last_chapter_i])

        patched_end_chapter = None
        # only the chapter | are defined by dvd
        # (the splitting logic assumes though that there is a chapter at the start and end)
        # TODO: verify these claims and check the splitting logic and figure out what the best solution is
        # you could either always add the end as chapter or stretch the last chapter till the end
        # Guess #1: We only need to handle the case where last is not acually a chapter so stretching
        # is the only correct solution there, adding would be wrong
        output_chapters = [0, *output_chapters]
        lastchpt = len(rnode)
        if output_chapters[-1] != lastchpt:
            patched_end_chapter = output_chapters[-1]
            output_chapters[-1] = lastchpt
        #            output_chapters += [lastchpt]

        audios = list[str]()
        for i, ac in enumerate(target_pgc.audio_control):  # pyright: ignore[reportPossiblyUnboundVariable]
            if ac.available:
                audio = target_vts.vtsi_mat.vts_audio_attr[i]

                if audio.audio_format == AUDIO_FORMAT_AC3:
                    aformat = "ac3"
                elif audio.audio_format == AUDIO_FORMAT_LPCM:
                    aformat = "lpcm"
                else:
                    aformat = "unk"

                audios += [f"{aformat}({audio.language})"]
            else:
                audios += ["none"]

        durationcodesf = list(map(float, durationcodes))

        assert output_chapters[0] == 0
        assert output_chapters[-1] == len(rnode)

        return Title(
            rnode,
            output_chapters,
            changes,
            self,
            title_idx,
            tt.title_set_nr,
            vobidcellids_to_take,
            dvdsrc_ranges,
            absolutetime,
            durationcodesf,
            audios,
            patched_end_chapter,
        )

    def __repr__(self) -> str:
        to_print = f"Path: {self.iso_path}\n"
        for i, tt in enumerate(self.ifo0.tt_srpt):
            target_vts = self.vts[tt.title_set_nr - 1]
            ptts = target_vts.vts_ptt_srpt[tt.vts_ttn - 1]

            current_time = 0.0
            seconds = list[float]()
            vobids = list[tuple[int, int]]()
            for a in ptts:
                target_pgc = target_vts.vts_pgci.pgcs[a.pgcn - 1]
                cell_n = target_pgc.program_map[a.pgn - 1]

                chap_time = target_pgc.cell_playback[cell_n - 1].playback_time.get_seconds_float()
                vobid = target_pgc.cell_position[cell_n - 1]

                current_time += chap_time
                seconds += [chap_time]
                vobids += [(vobid.vob_id_nr, vobid.cell_nr)]
            to_print += f"Title: {i + 1:02}\n"
            lastv = None

            crnt = 0.0
            crnt_glbl = 0.0
            to_print += "  nbr vobid start localstart localend duration\n"
            for i, v in enumerate(vobids):
                if lastv != v[0]:
                    if i != 0:
                        to_print += "\n"
                    crnt = 0
                    lastv = v[0]
                sta = str(timedelta(seconds=crnt))
                sta_g = str(timedelta(seconds=crnt_glbl))
                end = str(timedelta(seconds=crnt + seconds[i]))
                dur = str(timedelta(seconds=seconds[i]))

                to_print += f"  {i + 1:02} {v} start={sta_g} local={sta} end={end} duration={dur}\n"
                crnt += seconds[i]
                crnt_glbl += seconds[i]

        return to_print.strip()


def get_sectorranges_for_vobcellpair(current_vts: IFOX, pair_id: tuple[int, int]) -> list[tuple[int, int]]:
    return [
        (e.start_sector, e.last_sector)
        for e in current_vts.vts_c_adt.cell_adr_table
        if (e.vob_id, e.cell_id) == pair_id
    ]


def dvdsrc_parse_vts(
    iso_path: SPath,
    title: IFO0Title,
    disable_rff: bool,
    vobidcellids_to_take: list[tuple[int, int]],
    target_vts: IFOX,
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

    rawnode = core.dvdsrc2.FullVts(str(iso_path), vts=title.title_set_nr, ranges=vts_indices)
    staff = dvdsrc_extract_data(rawnode)

    if not disable_rff:
        rnode = apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)
        _vobids = apply_rff_array(staff.vobids, staff.rff, staff.tff, staff.progseq)
    else:
        rnode = rawnode
        _vobids = staff.vobids

    return rnode, staff.rff, _vobids, vts_indices


def dvdsrc_extract_data(rawnode: vs.VideoNode) -> AllNeddedDvdFrameData:
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
