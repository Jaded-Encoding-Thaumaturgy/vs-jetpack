# noqa: N999

from __future__ import annotations

import re
from collections.abc import Callable
from fractions import Fraction
from functools import partial
from typing import ClassVar

from jetpytools import SPath

from vstools import core, vs

from ..dataclasses import D2VIndexFileInfo, D2VIndexFrameData, D2VIndexHeader
from .base import ExternalIndexer

__all__ = ["D2VWitch"]


class D2VWitch(ExternalIndexer):
    _bin_path = "d2vwitch"
    _ext = "d2v"
    _source_func: ClassVar[Callable[..., vs.VideoNode]] = core.lazy.d2v.Source

    _default_args = ("--single-input",)

    def get_cmd(self, files: list[SPath], output: SPath) -> list[str]:
        return list(map(str, [self._get_bin_path(), *files, "--output", output]))

    def update_video_filenames(self, index_path: SPath, filepaths: list[SPath]) -> None:
        with open(index_path) as file:
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

    def get_info(self, index_path: SPath, file_idx: int = -1) -> D2VIndexFileInfo:
        with open(index_path) as f:
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
