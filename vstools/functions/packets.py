from __future__ import annotations

import io
import itertools
import json
import shutil
import warnings
from functools import cache
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile
from typing import TypedDict

import vapoursynth as vs
from jetpytools import CustomValueError, DependencyNotFoundError, FuncExceptT, SPath, SPathLike
from typing_extensions import Self

from ..types import VideoNodeT
from .file import PackageStorage
from .timecodes import Keyframes

__all__ = [
    "ScenePacketStats",
    "VideoPackets",
]


@cache
def _get_packet_storage() -> PackageStorage:
    return PackageStorage(package_name="packets")


class ScenePacketStats(TypedDict):
    """
    A class representing the packet size statistics for a scene in a video.
    """

    PktSceneAvgSize: float
    """The average packet size for the scene."""

    PktSceneMaxSize: float
    """The maximum packet size for the scene."""

    PktSceneMinSize: float
    """The minimum packet size for the scene."""


class VideoPackets(list[int]):
    """
    A class representing video packet sizes for each frame in a video.

    Packet sizes are useful for analyzing video encoding characteristics such as bitrate,
    allowing you to process frames and/or scenes based on packet sizes.
    """

    @classmethod
    def from_video(
        cls, src_file: SPathLike, out_file: SPathLike | None = None, offset: int = 0, *, func: FuncExceptT | None = None
    ) -> Self:
        """
        Obtain packet sizes from a video file.

        If the packet sizes are already calculated, they will be read from the output file.
        Otherwise, this method will use `ffprobe` to calculate the packet sizes and save them to the output file.

        `offset` can be used to remove or add frames from the start of the list. This is useful for applying
        the packet sizes to a trimmed clip. Positive values will trim the start of the list, and negative values
        will duplicate packets at the start of the list.

        Args:
            src_file: The path to the source video file.
            out_file: The path to the output file where packet sizes will be saved. If None, output file will be placed
                alongside the source file. Default: None.
            offset: An optional integer offset to trim the packet sizes. This is useful for applying the packet sizes to
                a trimmed clip. Positive values will trim the start of the list, and negative values will duplicate
                packets at the start of the list. Default: 0.
            func: An optional function to use for error handling. This should only be set by package developers.

        Returns:
            A VideoPackets object containing the packet sizes.
        """

        func = func or cls.from_video

        src_file = SPath(src_file)

        if not src_file.exists():
            raise CustomValueError("Source file not found!", func, src_file.absolute())

        if out_file is None:
            out_file = src_file.with_stem(src_file.stem + "_packets").with_suffix(".txt")

        if video_packets := cls.from_file(out_file, func=func):
            return video_packets

        out_file = _get_packet_storage().get_file(out_file, ext=".txt")

        if not shutil.which("ffprobe"):
            raise DependencyNotFoundError(func, "ffprobe", "Could not find {package}! Make sure it's in your PATH!")

        proc = Popen(
            [
                "ffprobe",
                "-hide_banner",
                "-show_frames",
                "-show_streams",
                "-threads",
                str(vs.core.num_threads),
                "-loglevel",
                "quiet",
                "-print_format",
                "json",
                "-select_streams",
                "v:0",
                src_file.to_str(),
            ],
            stdout=PIPE,
        )

        with NamedTemporaryFile("a+", delete=False) as tempfile:
            assert proc.stdout

            for line in io.TextIOWrapper(proc.stdout, "utf-8"):
                tempfile.write(line)

            tempfile.flush()

        try:
            with open(tempfile.name, "r") as f:
                data = dict(json.load(f))
        finally:
            SPath(tempfile.name).unlink()

        if not (frames := data.get("frames", {})):
            raise CustomValueError(f"No frames found in file, '{src_file}'! Your file may be corrupted!", func)

        pkt_sizes = [int(dict(frame).get("pkt_size", -1)) for frame in frames]

        print(f"Writing packet sizes to '{out_file.absolute()}'...")

        out_file.write_text("\n".join(map(str, pkt_sizes)), "utf-8", newline="\n")

        if offset < 0:
            pkt_sizes = [-1] * -offset + pkt_sizes
        elif offset > 0:
            pkt_sizes = pkt_sizes[offset:]

        return cls(pkt_sizes)

    @classmethod
    def from_file(cls, file: SPathLike, *, func: FuncExceptT | None = None) -> Self | None:
        """
        Obtain packet sizes from a given file.

        Args:
            file: The path to the file containing the packet sizes.
            func: An optional function to use for error handling. This should only be set by package developers.

        Returns:
            A VideoPackets object containing the packet sizes.
        """

        if file is not None:
            file = _get_packet_storage().get_file(file, ext=".txt")

            if file.exists() and not file.stat().st_size:
                file.unlink()

        if file is not None and file.exists():
            with file.open("r+") as f:
                return cls(map(int, f.readlines()))

        return None

    @classmethod
    def from_clip(
        cls,
        clip: vs.VideoNode,
        out_file: SPathLike,
        src_file: SPathLike | None = None,
        offset: int = 0,
        *,
        func: FuncExceptT | None = None,
    ) -> Self:
        """
        Obtain packet sizes from a given clip.

        Args:
            clip: The clip to obtain packet sizes from. Must have the `IdxFilePath` frame property.
            out_file: The path to the output file where packet sizes will be saved.
            src_file: The path to the source video file. If None, the source file will be obtained from the clip.
                Default: None.
        """

        from ..utils import get_clip_filepath

        func = func or cls.from_video

        out_file = SPath(str(out_file)).stem + f"_{clip.num_frames}_{clip.fps_num}_{clip.fps_den}"

        if video_packets := cls.from_file(out_file, func=func):
            return video_packets

        if (src_file := get_clip_filepath(clip, src_file, func=func)) is None:
            raise CustomValueError("You must provide a source file!", func)

        return cls.from_video(src_file, out_file, offset, func=func)

    def get_scenestats(self, keyframes: Keyframes) -> list[ScenePacketStats]:
        """
        Calculate scene-based packet size statistics by referencing Keyframes.

        Args:
            keyframes: The keyframe list to get scene packet statistics for.

        Returns:
            A list of ScenePacketStats objects.
        """

        stats = list[ScenePacketStats]()

        try:
            for start, end in itertools.pairwise(keyframes):
                pkt_scenes = self[start:end]

                stats.append(
                    ScenePacketStats(
                        PktSceneAvgSize=sum(pkt_scenes) / len(pkt_scenes),
                        PktSceneMaxSize=max(pkt_scenes),
                        PktSceneMinSize=min(pkt_scenes),
                    )
                )
        except ValueError as e:
            raise CustomValueError("Some kind of error occurred!", self.get_scenestats, str(e))

        return stats

    def apply_props(
        self, clip: VideoNodeT, keyframes: Keyframes | None = None, *, func: FuncExceptT | None = None
    ) -> VideoNodeT:
        """
        Apply packet size properties to a clip.

        Args:
            clip: The clip to apply the packet size properties to.
            keyframes: The keyframe list to get scene packet statistics for. If None, the packet size properties will be
                applied to each frame. Default: None.
            func: An optional function to use for error handling. This should only be set by package developers.

        Returns:
            A clip with the packet size properties applied.
        """

        func = func or self.apply_props

        def _set_sizes_props(n: int) -> vs.VideoNode:
            if (pkt_size := self[n]) < 0:
                warnings.warn(f"{func}: 'Frame {n} bitrate could not be determined!'")

            return clip.std.SetFrameProps(PktSize=pkt_size)

        if not keyframes:
            return vs.core.std.FrameEval(clip, _set_sizes_props)

        def _set_scene_stats(n: int, keyframes: Keyframes) -> vs.VideoNode:
            if (pkt_size := self[n]) < 0:
                warnings.warn(f"{func}: 'Frame {n} bitrate could not be determined!'")

            try:
                return clip.std.SetFrameProps(PktSize=pkt_size, **scenestats[keyframes.scenes.indices[n]])
            except Exception:
                warnings.warn(f"{func}: 'Could not find stats for a section... (Frame: {n})'")

                return clip.std.SetFrameProps(PktSize=-1, PktSceneAvgSize=-1, PktSceneMaxSize=-1, PktSceneMinSize=-1)

        scenestats = self.get_scenestats(keyframes)

        return vs.core.std.FrameEval(clip, lambda n: _set_scene_stats(n, keyframes))
