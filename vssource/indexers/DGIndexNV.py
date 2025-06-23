# noqa: N999

from __future__ import annotations

import os
from fractions import Fraction
from functools import lru_cache
from typing import Any, Sequence

from vstools import SPath, core

from ..dataclasses import DGIndexFileInfo, DGIndexFooter, DGIndexFrameData, DGIndexHeader
from ..utils import opt_int, opt_ints
from .base import ExternalIndexer

__all__ = ["DGIndexNV"]


class DGIndexNV(ExternalIndexer):
    _bin_path = "DGIndexNV"
    _ext = "dgi"
    _source_func = core.lazy.dgdecodenv.DGSource

    def __init__(self, **kwargs: Any) -> None:
        import warnings

        # Would prefer to do a version check, but I don't think it uses proper versioning...
        warnings.filterwarnings("always", category=DeprecationWarning)  # why does it require this now...?

        warnings.warn(
            f"{self.__class__.__name__}: This source filter is deprecated due to a number of regressions "
            "in recent versions and will be removed in a future release. Use FFMS2 or BestSource instead.",
            DeprecationWarning,
        )

        super().__init__(**kwargs)

    def get_cmd(self, files: list[SPath], output: SPath) -> list[str]:
        return list(
            map(
                str,
                [
                    self._get_bin_path(),
                    "-i",
                    ",".join(str(path.absolute()) for path in files),
                    "-h",
                    "-o",
                    output,
                    "-e",
                ],
            )
        )

    def update_video_filenames(self, index_path: SPath, filepaths: list[SPath]) -> None:
        lines = index_path.read_lines()

        str_filepaths = list(map(str, filepaths))

        if "DGIndexNV" not in lines[0]:
            self.file_corrupted(index_path)

        start_videos = lines.index("") + 1
        end_videos = lines.index("", start_videos)

        if end_videos - start_videos != len(str_filepaths):
            self.file_corrupted(index_path)

        split_lines = [line.split(" ") for line in lines[start_videos:end_videos]]

        current_paths = [line[:-1][0] for line in split_lines]

        if current_paths == str_filepaths:
            return

        video_args = [line[-1:] for line in split_lines]

        lines[start_videos:end_videos] = [" ".join([path, *args]) for path, args in zip(str_filepaths, video_args)]

        index_path.write_lines(lines)

    @lru_cache
    def get_info(self, index_path: SPath, file_idx: int = -1) -> DGIndexFileInfo:
        with open(index_path, "r") as file:
            file_content = file.read()

        lines = file_content.split("\n")

        head, lines = self._split_lines(lines)

        if "DGIndexNV" not in head[0]:
            self.file_corrupted(index_path)

        vid_lines, lines = self._split_lines(lines)
        raw_header, lines = self._split_lines(lines)

        header = DGIndexHeader()

        for rlin in raw_header:
            if split_val := rlin.rstrip().split(" "):
                key: str = split_val[0].upper()
                values: list[str] = split_val[1:]
            else:
                continue

            if key == "DEVICE":
                header.device = int(values[0])
            elif key == "DECODE_MODES":
                header.decode_modes = list(map(int, values[0].split(",")))
            elif key == "STREAM":
                header.stream = tuple(map(int, values))
            elif key == "RANGE":
                header.ranges = list(map(int, values))
            elif key == "DEMUX":
                continue
            elif key == "DEPTH":
                header.depth = int(values[0])
            elif key == "ASPECT":
                try:
                    header.aspect = Fraction(*list(map(int, values)))
                except ZeroDivisionError:
                    header.aspect = Fraction(1, 1)
                    if os.environ.get("VSSOURCE_DEBUG", False):
                        print(ResourceWarning("Encountered video with 0/0 aspect ratio!"))
            elif key == "COLORIMETRY":
                header.colorimetry = tuple(map(int, values))
            elif key == "PKTSIZ":
                header.packet_size = int(values[0])
            elif key == "VPID":
                header.vpid = int(values[0])

        video_sizes = [int(line[-1]) for line in [line.split(" ") for line in vid_lines]]

        max_sector = sum([0, *video_sizes[: file_idx + 1]])

        idx_file_sector = [max_sector - video_sizes[file_idx], max_sector]

        curr_seq, frame_data = 0, []

        for rawline in lines:
            if len(rawline) == 0:
                break

            line: Sequence[str | None] = [*rawline.split(" ", maxsplit=6), *([None] * 6)]

            name = str(line[0])

            if name == "SEQ":
                curr_seq = opt_int(line[1]) or 0

            if curr_seq < idx_file_sector[0]:
                continue
            elif curr_seq > idx_file_sector[1]:
                break

            try:
                int(name.split(":")[0])
            except ValueError:
                continue

            frame_data.append(DGIndexFrameData(int(line[2] or 0) + 2, str(line[1]), *opt_ints(line[4:6])))

        footer = DGIndexFooter()

        for rlin in lines[-10:]:
            if split_val := rlin.rstrip().split(" "):
                values = [split_val[0], " ".join(split_val[1:])]
            else:
                continue

            for key in footer.__dict__:
                if key.split("_")[-1].upper() in values:
                    if key == "film":
                        try:
                            value = next(float(v.replace("%", "")) for v in values if "%" in v)
                        except IndexError:
                            value = 0
                    else:
                        value = int(values[1])

                    footer[key] = value

        return DGIndexFileInfo(index_path, file_idx, header, frame_data, footer)
