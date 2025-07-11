# noqa: N999

from __future__ import annotations

import datetime
import json
import os
import warnings
from abc import abstractmethod
from fractions import Fraction
from itertools import count
from tempfile import gettempdir
from typing import Sequence, cast

from vstools import CustomValueError, DependencyNotFoundError, Region, SPath, SPathLike, core, vs

from ...indexers import DVDExtIndexer, DVDSRCIndexer, ExternalIndexer
from ...rff import apply_rff_video
from ...utils import debug_print
from .parsedvd import (
    AUDIO_FORMAT_AC3,
    AUDIO_FORMAT_LPCM,
    BLOCK_MODE_FIRST_CELL,
    BLOCK_MODE_IN_BLOCK,
    BLOCK_MODE_LAST_CELL,
    IFO0,
    IFOX,
    SectorReadHelper,
    to_json,
)
from .title import Title
from .utils import absolute_time_from_timecode, double_check_dvdnav

__all__ = ["IsoFileCore"]


class IsoFileCore:
    _subfolder = "VIDEO_TS"

    ifo0: IFO0
    vts: list[IFOX]
    indexer: DVDSRCIndexer | DVDExtIndexer
    title_count: int

    def __init__(
        self,
        path: SPath | str,
        indexer: DVDExtIndexer | type[DVDExtIndexer] | None = None,
    ):
        """
        Only external indexer supported D2VWitch and DGIndex

        If the indexer is None, dvdsrc is used
        """
        self.force_root = False
        self.output_folder = SPath(gettempdir())

        self._mount_path: SPath | None = None
        self._vob_files: list[SPath] | None = None
        self._ifo_files: list[SPath] | None = None

        if not hasattr(core, "dvdsrc2"):
            raise DependencyNotFoundError(
                self.__class__, "", "dvdsrc2 is needed for {cfunc} to work!", cfunc=self.__class__
            )

        self.iso_path = SPath(path).absolute()

        if not self.iso_path.exists():
            raise CustomValueError('"path" needs to point to a .ISO or a dir root of DVD!', str(path), self.__class__)

        ifo0: IFO0 | None = None
        ifos: Sequence[SPathLike | bytes] = []
        if indexer is None:

            def _getifo(i: int) -> bytes:
                return cast(bytes, core.dvdsrc2.Ifo(str(self.iso_path), i))

            _ifo0b = _getifo(0)

            # remove in 2025
            if len(_ifo0b) <= 30:
                warnings.warn("Newer VapourSynth is required for dvdsrc2 information gathering without mounting!")
            else:
                ifo0 = IFO0(SectorReadHelper(_ifo0b))
                ifos = [_getifo(i) for i in range(1, ifo0.num_vts + 1)]

        if not ifo0:
            _ifo0p, *ifos = self.ifo_files
            ifo0 = IFO0(SectorReadHelper(_ifo0p))

        self.ifo0 = ifo0
        self.vts = [IFOX(SectorReadHelper(ifo)) for ifo in ifos]

        self._double_check_json()

        self.dvdsrc = DVDSRCIndexer()
        self.dvdsrc.iso_path = self.iso_path

        if indexer is None:
            self.indexer = self.dvdsrc
        else:
            self.indexer = indexer() if isinstance(indexer, type) else indexer
            self.indexer.iso_path = self.iso_path

        self.title_count = len(self.ifo0.tt_srpt)

    def get_vts(self, title_set_nr: int = 1, d2v_our_rff: bool = False) -> vs.VideoNode:
        """
        Gets a full vts.
        Only works with dvdsrc2 and with d2vsource.

        It uses our rff for dvdsrc and d2source rff for d2vsource.

        Mainly useful for debugging and checking if our rff algorithm is good.
        """

        if isinstance(self.indexer, DVDSRCIndexer) or d2v_our_rff:
            fullvts = core.dvdsrc2.FullVts(str(self.iso_path), vts=title_set_nr)

        if isinstance(self.indexer, ExternalIndexer):
            vob_input_files = self._get_title_vob_files_for_vts(title_set_nr)
            index_file = self.indexer.index(vob_input_files, output_folder=self.output_folder)[0]
            rawnode = self.indexer._source_func(index_file, rff=not d2v_our_rff)

            if not d2v_our_rff:
                return rawnode
        else:
            rawnode = fullvts

        staff = self.dvdsrc._extract_data(fullvts)

        return apply_rff_video(rawnode, staff.rff, staff.tff, staff.prog, staff.progseq)

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
                warnings.warn("Open Title does not start at the first cell (open issue in github with sample)\n")

            target_programs = [
                a[1] for a in list(filter(lambda x: (x[0] + 1) in title_programs, enumerate(pgc_programs)))
            ]

            if target_programs != pgc_programs:
                warnings.warn("Open the program chain does not include all ptts\n")

            current_angle = 1
            angle_start_cell_i: int

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
                    is_chapter += [(angle_start_cell_i + 1) in target_programs]

            i += ptt_to_take_for_pgc

        assert len(is_chapter) == len(vobidcellids_to_take)

        rnode, rff, vobids, dvdsrc_ranges = self.indexer.parse_vts(
            tt,
            disable_rff,
            vobidcellids_to_take,
            target_vts,
            self.output_folder,
            [] if isinstance(self.indexer, DVDSRCIndexer) else self._get_title_vob_files_for_vts(tt.title_set_nr),
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

        dvnavchapters = double_check_dvdnav(self.iso_path, title_idx + 1)

        if dvnavchapters is not None:  # and (rff_mode == 0 or rff_mode == 2):
            # ???????
            if rfps.denominator == 1001:
                dvnavchapters = [a * 1.001 for a in dvnavchapters]

            adjusted = [
                absolutetime[i] if i != len(absolutetime) else absolutetime[i - 1] + durationcodes[i - 1]
                for i in output_chapters
            ]
            if len(adjusted) != len(dvnavchapters):
                warnings.warn(
                    "dvdnavchapters length do not match our chapters "
                    f"{len(adjusted)} {len(dvnavchapters)} (open an issue in github)"
                )
                print(adjusted, "\n\n\n", dvnavchapters)
            else:
                framelen = rfps.denominator / rfps.numerator
                for i in range(len(adjusted)):
                    # tolerance no idea why so big
                    # on hard telecine ntcs it matches up almost perfectly
                    # but on ~24p pal rffd it does not lol
                    if abs(adjusted[i] - dvnavchapters[i]) > framelen * 20:
                        warnings.warn(
                            f"dvdnavchapters do not match our chapters {len(adjusted)} {len(dvnavchapters)}"
                            " (open an issue in github)\n"
                            f" index: {i} {adjusted[i]}"
                        )
                        print(adjusted, "\n\n\n", dvnavchapters)
                        break
        else:
            debug_print("Skipping sanity check with dvdnav")

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
        for i, ac in enumerate(target_pgc.audio_control):
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

    def _get_title_vob_files_for_vts(self, vts: int) -> Sequence[SPath]:
        return [
            vob for vob in self.vob_files if f"VTS_{vts:02}_" in (s := vob.to_str()) and not s.upper().endswith("0.VOB")
        ]

    def __repr__(self) -> str:
        to_print = f"Path: {self.iso_path}\n"
        to_print += f"Mount: {self._mount_path}\n"
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
                sta = str(datetime.timedelta(seconds=crnt))
                sta_g = str(datetime.timedelta(seconds=crnt_glbl))
                end = str(datetime.timedelta(seconds=crnt + seconds[i]))
                dur = str(datetime.timedelta(seconds=seconds[i]))

                to_print += f"  {i + 1:02} {v} start={sta_g} local={sta} end={end} duration={dur}\n"
                crnt += seconds[i]
                crnt_glbl += seconds[i]

        return to_print.strip()

    def _mount_folder_path(self) -> SPath:
        if self.force_root:
            return self.iso_path

        if self.iso_path.name.upper() == self._subfolder:
            self.iso_path = self.iso_path.parent

        return self.iso_path / self._subfolder

    @property
    def mount_path(self) -> SPath:
        if self._mount_path is not None:
            return self._mount_path

        if self.iso_path.is_dir():
            return self._mount_folder_path()

        disc = self._get_mounted_disc() or self._mount()
        if not disc:
            raise RuntimeError("IsoFile: Couldn't mount ISO file!")

        self._mount_path = disc / self._subfolder

        return self._mount_path

    @property
    def vob_files(self) -> list[SPath]:
        if self._vob_files is not None:
            return self._vob_files

        vob_files = [f for f in sorted(self.mount_path.glob("*.[vV][oO][bB]")) if f.stem != "VIDEO_TS"]

        if not len(vob_files):
            raise FileNotFoundError("IsoFile: No VOBs found!")

        self._vob_files = vob_files

        return self._vob_files

    @property
    def ifo_files(self) -> list[SPath]:
        if self._ifo_files is not None:
            return self._ifo_files

        ifo_files = sorted(self.mount_path.glob("*.[iI][fF][oO]"))

        if not len(ifo_files):
            raise FileNotFoundError("IsoFile: No IFOs found!")

        self._ifo_files = ifo_files

        return self._ifo_files

    @abstractmethod
    def _get_mounted_disc(self) -> SPath | None:
        raise NotImplementedError()

    @abstractmethod
    def _mount(self) -> SPath | None:
        raise NotImplementedError()

    def _double_check_json(self) -> None:
        if hasattr(core, "dvdsrc"):
            our_json = to_json(self.ifo0, self.vts)

            dvdsrc_json = json.loads(cast(str, core.dvdsrc.Json(str(self.iso_path))))

            for key in ("dvdpath", "current_vts", "current_domain"):
                dvdsrc_json.pop(key, None)

            for ifo in dvdsrc_json.get("ifos", []):
                ifo["pgci_ut"] = []

            ja = json.dumps(dvdsrc_json, sort_keys=True)
            jb = json.dumps(our_json, sort_keys=True)

            if ja != jb:
                warnings.warn(
                    f"libdvdread json does not match python implentation\n"
                    f"json a,b have been written to {self.output_folder}"
                )

                for k, v in [("a", ja), ("b", jb)]:
                    with open(os.path.join(self.output_folder, f"{k}.json"), "wt") as file:
                        file.write(v)
        else:
            debug_print("We don't have dvdsrc and can't double check the json output with libdvdread.")
