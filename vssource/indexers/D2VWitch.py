# noqa: N999

from __future__ import annotations

import re
from fractions import Fraction
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Callable, ClassVar, Sequence

from vstools import CustomValueError, SPath, core, remap_frames, vs

from ..dataclasses import D2VIndexFileInfo, D2VIndexFrameData, D2VIndexHeader
from ..rff import apply_rff_array, apply_rff_video, cut_array_on_ranges
from .base import DVDExtIndexer

if TYPE_CHECKING:
    from ..formats.dvd.parsedvd import IFOX, IFO0Title

__all__ = ["D2VWitch"]


class D2VWitch(DVDExtIndexer):
    _bin_path = "d2vwitch"
    _ext = "d2v"
    _source_func: ClassVar[Callable[..., vs.VideoNode]] = core.lazy.d2v.Source

    _default_args = ("--single-input",)

    def get_cmd(self, files: list[SPath], output: SPath) -> list[str]:
        return list(map(str, [self._get_bin_path(), *files, "--output", output]))

    def update_video_filenames(self, index_path: SPath, filepaths: list[SPath]) -> None:
        with open(index_path, "r") as file:
            file_content = file.read()

        lines = file_content.split("\n")

        str_filepaths = list(map(str, filepaths))

        if "DGIndex" not in lines[0]:
            self.file_corrupted(index_path)

        if not (n_files := int(lines[1])) or n_files != len(str_filepaths):
            self.file_corrupted(index_path)

        end_videos = lines.index("")

        if lines[2:end_videos] == str_filepaths:
            return

        lines[2:end_videos] = str_filepaths

        with open(index_path, "w") as file:
            file.write("\n".join(lines))

    @lru_cache
    def get_info(self, index_path: SPath, file_idx: int = -1) -> D2VIndexFileInfo:
        with open(index_path, "r") as f:
            file_content = f.read()

        lines = file_content.split("\n")

        head, lines = lines[:2], lines[2:]

        if "DGIndex" not in head[0]:
            self.file_corrupted(index_path)

        raw_header, lines = self._split_lines(self._split_lines(lines)[1])

        header = D2VIndexHeader()

        for rlin in raw_header:
            if split_val := rlin.rstrip().split("="):
                key: str = split_val[0].upper()
                values: list[str] = ",".join(split_val[1:]).split(",")
            else:
                continue

            if key == "STREAM_TYPE":
                header.stream_type = int(values[0])
            elif key == "MPEG_TYPE":
                header.MPEG_type = int(values[0])
            elif key == "IDCT_ALGORITHM":
                header.iDCT_algorithm = int(values[0])
            elif key == "YUVRGB_SCALE":
                header.YUVRGB_scale = int(values[0])
            elif key == "LUMINANCE_FILTER":
                header.luminance_filter = tuple(map(int, values))
            elif key == "CLIPPING":
                header.clipping = list(map(int, values))
            elif key == "ASPECT_RATIO":
                header.aspect = Fraction(*list(map(int, values[0].split(":"))))
            elif key == "PICTURE_SIZE":
                header.pic_size = str(values[0])
            elif key == "FIELD_OPERATION":
                header.field_op = int(values[0])
            elif key == "FRAME_RATE":
                if matches := re.search(r".*\((\d+\/\d+)", values[0]):
                    header.frame_rate = Fraction(matches.group(1))
            elif key == "LOCATION":
                header.location = list(map(partial(int, base=16), values))

        frame_data = list[D2VIndexFrameData]()

        if file_idx >= 0:
            for rawline in lines:
                if len(rawline) == 0:
                    break

                line = rawline.split(" ", maxsplit=7)

                ffile_idx = int(line[2])

                if ffile_idx < file_idx:
                    continue
                elif ffile_idx > file_idx:
                    break

                frame_data.append(
                    D2VIndexFrameData(
                        int(line[1]),
                        "I",
                        int(line[5]),
                        int(line[6]),
                        int(line[0], 16),
                        int(line[4]),
                        int(line[3]),
                        [int(a, 16) for a in line[7:]],
                    )
                )
        elif file_idx == -1:
            for rawline in lines:
                if len(rawline) == 0:
                    break

                line = rawline.split(" ")

                frame_data.append(
                    D2VIndexFrameData(
                        int(line[1]),
                        "I",
                        int(line[5]),
                        int(line[6]),
                        int(line[0], 16),
                        int(line[4]),
                        int(line[3]),
                        [int(a, 16) for a in line[7:]],
                    )
                )

        return D2VIndexFileInfo(index_path, file_idx, header, frame_data)

    def parse_vts(
        self,
        title: IFO0Title,
        disable_rff: bool,
        vobidcellids_to_take: list[tuple[int, int]],
        target_vts: IFOX,
        output_folder: SPath,
        vob_input_files: Sequence[SPath],
    ) -> tuple[vs.VideoNode, list[int], list[tuple[int, int]], list[int]]:
        dvddd = self._d2v_vobid_frameset(vob_input_files, output_folder)

        if len(dvddd.keys()) == 1 and (0, 0) in dvddd:
            raise CustomValueError(
                "Youre indexer created a d2v file with only zeros for vobid cellid; "
                "This usually means outdated/unpatched D2Vwitch",
                self.parse_vts,
            )

        frameranges = [x for y in [dvddd[a] for a in vobidcellids_to_take] for x in y]

        fflags, vobids, progseq = self._d2v_collect_all_frameflags(vob_input_files, output_folder)

        index_file = self.index(vob_input_files, output_folder=output_folder)[0]
        node = self._source_func(index_file, rff=False)

        assert len(node) == len(fflags) == len(vobids) == len(progseq)

        progseq = cut_array_on_ranges(progseq, frameranges)
        fflags = cut_array_on_ranges(fflags, frameranges)
        vobids = cut_array_on_ranges(vobids, frameranges)
        node = remap_frames(node, frameranges)

        rff = [(a & 1) for a in fflags]

        if not disable_rff:
            tff = [int((a & 2) >> 1) for a in fflags]
            prog = [int((a & 0b01000000) != 0) for a in fflags]

            node = apply_rff_video(node, rff, tff, prog, progseq)
            vobids = apply_rff_array(vobids, rff, tff, progseq)

        return node, rff, vobids, []

    def _d2v_collect_all_frameflags(
        self, files: Sequence[SPath], output_folder: SPath
    ) -> tuple[list[int], list[tuple[int, int]], list[int]]:
        index_file = self.index(files, output_folder=output_folder)[0]
        index_info = self.get_info(index_file)

        frameflagslst = list[int]()
        vobidlst = list[tuple[int, int]]()
        progseqlst = list[int]()

        for iframe in index_info.frame_data:
            assert isinstance(iframe, D2VIndexFrameData)

            vobcell = (iframe.vob, iframe.cell)

            progseq = int((iframe.info & 0b1000000000) != 0)

            for a in iframe.frameflags:
                if a != 0xFF:
                    frameflagslst.append(a)
                    vobidlst.append(vobcell)
                    progseqlst.append(progseq)

        return frameflagslst, vobidlst, progseqlst

    def _d2v_vobid_frameset(
        self, files: Sequence[SPath], output_folder: SPath
    ) -> dict[tuple[int, int], list[tuple[int, int]]]:
        _, vobids, _ = self._d2v_collect_all_frameflags(files, output_folder)

        vobidset = dict[tuple[int, int], list[tuple[int, int]]]()
        for i, a in enumerate(vobids):
            if a not in vobidset:
                vobidset[a] = [(i, i - 1)]

            last = vobidset[a][-1]

            if last[1] + 1 == i:
                vobidset[a][-1] = (last[0], last[1] + 1)
                continue

            vobidset[a].append((i, i))

        return vobidset
