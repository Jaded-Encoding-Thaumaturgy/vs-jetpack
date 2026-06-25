from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, Literal, Self, cast, overload

from jetpytools import (
    CustomRuntimeError,
    CustomValueError,
    FilePathType,
    FuncExcept,
    LinearRangeLut,
    SPath,
    check_perms,
    fallback,
)

from ..exceptions import FramesLengthError, UnsupportedTimecodeVersionError
from ..utils import DynamicClipsCache, PackageStorage
from ..vs_proxy import VSObject, vs
from .ranges import replace_ranges
from .render import clip_async_render, clip_data_gather

__all__ = ["Keyframes", "Timecodes"]


@cache
def _get_keyframes_storage() -> PackageStorage:
    return PackageStorage(package_name="keyframes")


@dataclass
class FrameDur:
    """
    A fraction representing the duration of a specific frame.
    """

    frame: int
    """The frame number."""

    numerator: int
    """The frame duration's numerator."""

    denominator: int
    """The frame duration's denominator."""

    def to_fraction(self) -> Fraction:
        """
        Convert the FrameDur to a Fraction that represents the frame duration.
        """

        return Fraction(self.numerator, self.denominator)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FrameDur):
            return False

        return (self.numerator, self.denominator) == (other.numerator, other.denominator)

    def __int__(self) -> float:
        return self.frame

    def __float__(self) -> float:
        return float(self.to_fraction())


class Timecodes(list[FrameDur]):
    """
    A list of frame durations, together representing a (possibly variable) frame rate.
    """

    V1 = 1
    """
    V1 timecode format, containing a list of frame ranges with associated frame rates. For example:
    ```
    # timecodes format v1
    Assume 23.976023976024
    544,548,29.97002997003
    721,725,29.97002997003
    770,772,17.982017982018
    ```
    """

    V2 = 2
    """
    V2 timecode format, containing a timestamp for each frame, including possibly a final timestamp after the last frame
    to specify the final frame's duration. For example:
    ```
    # timecode format v2
    0.000000
    41.708333
    83.416667
    125.125000
    166.833333
    ```
    """

    def to_fractions(self) -> list[Fraction]:
        """
        Convert to a list of frame lengths, representing the individual framerates.
        """

        return [t.to_fraction() for t in self]

    def to_normalized_ranges(self) -> dict[tuple[int, int], Fraction]:
        """
        Convert to a list of normalized frame ranges and their assigned framerate.
        """

        timecodes_ranges = dict[tuple[int, int], Fraction]()

        last_i = len(self) - 1
        last_tcode: tuple[int, FrameDur] = (0, self[0])

        for tcode in self[1:]:
            start, ltcode = last_tcode

            if tcode != ltcode:
                timecodes_ranges[start, tcode.frame - 1] = 1 / ltcode.to_fraction()
                last_tcode = (tcode.frame, tcode)
            elif tcode.frame == last_i:
                timecodes_ranges[start, tcode.frame] = 1 / tcode.to_fraction()

        return timecodes_ranges

    @classmethod
    def normalize_range_timecodes(
        cls, timecodes: dict[tuple[int | None, int | None], Fraction], length: int, assume: Fraction | None = None
    ) -> list[Fraction]:
        """
        Convert from normalized ranges to a list of frame duration.
        """
        norm_timecodes = [assume] * length if assume else list[Fraction]()

        for (startn, endn), fps in timecodes.items():
            start = max(fallback(startn, 0), 0)
            end = fallback(endn, length - 1)

            if end + 1 > len(norm_timecodes):
                norm_timecodes += [1 / fps] * (end + 1 - len(norm_timecodes))

            norm_timecodes[start : end + 1] = [1 / fps] * (end + 1 - start)

        return norm_timecodes

    @classmethod
    def separate_norm_timecodes(
        cls, timecodes: Timecodes | dict[tuple[int, int], Fraction]
    ) -> tuple[Fraction, dict[tuple[int, int], Fraction]]:
        if isinstance(timecodes, Timecodes):
            timecodes = timecodes.to_normalized_ranges()

        times_count = dict.fromkeys(timecodes.values(), 0)

        for v in timecodes.values():
            times_count[v] += 1

        major_count = max(times_count.values())
        major_time = next(t for t, c in times_count.items() if c == major_count)
        minor_fps = {r: v for r, v in timecodes.items() if v != major_time}

        return major_time, minor_fps

    @classmethod
    def accumulate_norm_timecodes(
        cls, timecodes: Timecodes | dict[tuple[int, int], Fraction]
    ) -> tuple[Fraction, dict[Fraction, list[tuple[int, int]]]]:
        if isinstance(timecodes, Timecodes):
            timecodes = timecodes.to_normalized_ranges()

        major_time, minor_fps = cls.separate_norm_timecodes(timecodes)

        acc_ranges = dict[Fraction, list[tuple[int, int]]]()

        for k, v in minor_fps.items():
            if v not in acc_ranges:
                acc_ranges[v] = []

            acc_ranges[v].append(k)

        return major_time, acc_ranges

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, **kwargs: Any) -> Self:
        """
        Get the timecodes from a given clip.

        Args:
            clip: Clip to gather metrics from.
            **kwargs: Keyword arguments to pass on to `clip_async_render`.
        """
        from ..utils import get_prop

        def _get_timecode(n: int, f: vs.VideoFrame) -> FrameDur:
            return FrameDur(n, get_prop(f, "_DurationNum", int), get_prop(f, "_DurationDen", int))

        return cls(clip_async_render(clip, None, "Fetching timecodes...", _get_timecode, **kwargs))

    @overload
    @classmethod
    def from_file(cls, file: FilePathType, ref: vs.VideoNode, /, *, func: FuncExcept | None = None) -> Self:
        """
        Read the timecodes from a given file.

        Args:
            file: File to read.
            ref: Reference clip to get the total number of frames from.
            func: Function returned for custom error handling. This should only be set by VS package developers.
        """

    @overload
    @classmethod
    def from_file(
        cls, file: FilePathType, length: int, den: int | None = None, /, func: FuncExcept | None = None
    ) -> Self:
        """
        Read the timecodes from a given file.

        Args:
            file: File to read.
            length: Total number of frames.
            den: The frame rate denominator. If None, try to obtain it from the ref if possible, else fall back to 1001.
            func: Function returned for custom error handling. This should only be set by VS package developers.
        """

    @classmethod
    def from_file(
        cls,
        file: FilePathType,
        ref_or_length: int | vs.VideoNode,
        den: int | None = None,
        /,
        func: FuncExcept | None = None,
    ) -> Self:
        func = func or cls.from_file

        file = Path(str(file)).resolve()

        length = ref_or_length if isinstance(ref_or_length, int) else ref_or_length.num_frames

        fb_den = (
            (None if ref_or_length.fps_den in {0, 1} else ref_or_length.fps_den)
            if isinstance(ref_or_length, vs.VideoNode)
            else None
        )

        denominator = den or fb_den or 1001

        version, *timecodes = file.read_text().splitlines()

        if "v1" in version:

            def _norm(xd: str) -> Fraction:
                return Fraction(round(denominator / float(xd)), denominator)

            assume = None

            timecodes_d = dict[tuple[int | None, int | None], Fraction]()

            for line in timecodes:
                if line.startswith("#"):
                    continue

                if line.startswith("Assume"):
                    assume = _norm(line[7:])
                    continue

                starts, ends, fps = line.split(",")
                timecodes_d[int(starts), int(ends) + 1] = _norm(fps)

            norm_timecodes = cls.normalize_range_timecodes(timecodes_d, length, assume)
        elif "v2" in version:
            timecodes_l = [float(t) for t in timecodes if not t.startswith("#")]
            norm_timecodes = [
                Fraction(denominator, int(denominator / float(f"{round((x - y) * 100, 4) / 100000:.08f}"[:-1])))
                for x, y in zip(timecodes_l[1:], timecodes_l[:-1])
            ]
        else:
            raise CustomValueError("timecodes file not supported!", func, file)

        if len(norm_timecodes) != length:
            raise FramesLengthError(
                func,
                "",
                "timecodes file length mismatch with specified length!",
                reason={"timecodes": len(norm_timecodes), "clip": length},
            )

        return cls(FrameDur(i, f.numerator, f.denominator) for i, f in enumerate(norm_timecodes))

    def assume_vfr(self, clip: vs.VideoNode, func: FuncExcept | None = None) -> vs.VideoNode:
        """
        Force the given clip to be assumed to be vfr by other applications.

        Args:
            clip: Clip to process.
            func: Function returned for custom error handling. This should only be set by VS package developers.

        Returns:
            Clip that should always be assumed to be vfr by other applications.
        """
        func = func or self.assume_vfr

        major_time, minor_fps = self.accumulate_norm_timecodes(self)

        assumed_clip = vs.core.std.AssumeFPS(clip, None, major_time.numerator, major_time.denominator)

        for other_fps, fps_ranges in minor_fps.items():
            assumed_clip = replace_ranges(
                assumed_clip,
                vs.core.std.AssumeFPS(clip, None, other_fps.numerator, other_fps.denominator),
                fps_ranges,
                mismatch=True,
            )

        return assumed_clip

    def to_file(self, out: FilePathType, format: int = V2, func: FuncExcept | None = None) -> None:
        """
        Write timecodes to a file.

        This file should always be muxed into the video container when working with Variable Frame Rate video.

        Args:
            out: Path to write the file to.
            format: Format to write the file to.
        """
        func = func or self.to_file

        out_path = Path(str(out)).resolve()

        check_perms(out_path, "w+", func=func)

        UnsupportedTimecodeVersionError.check(self.to_file, format)

        out_text = [f"# timecode format v{format}"]

        if format == Timecodes.V1:
            major_time, minor_fps = self.separate_norm_timecodes(self)

            out_text.append(f"Assume {round(float(major_time), 12)}")

            out_text.extend([",".join(map(str, [*frange, round(float(fps), 12)])) for frange, fps in minor_fps.items()])
        elif format == Timecodes.V2:
            acc = Fraction()  # in milliseconds

            for time in [*self, Fraction()]:
                ns = round(acc * 10**6)
                ms, dec = divmod(ns, 10**6)
                out_text.append(f"{ms}.{dec:06}")
                acc += Fraction(time.numerator * 1000, time.denominator)

        out_path.unlink(True)
        out_path.touch()
        out_path.write_text("\n".join([*out_text, ""]))


class Keyframes(list[int]):
    """
    Class representing keyframes, or scenechanges.

    They follow the convention of signaling the start of the new scene.
    """

    class _Scenes(dict[int, range]):
        __slots__ = ("indices",)

        def __init__(self, kf: Keyframes) -> None:
            if kf:
                super().__init__({i: range(x, y) for i, (x, y) in enumerate(zip(kf, [*kf[1:], 1 << 32]))})

            self.indices = LinearRangeLut(self)

    def __init__(self, iterable: Iterable[int] = []) -> None:
        super().__init__(sorted(iterable))
        self.scenes = self.__class__._Scenes(self)

    def to_file(
        self,
        out: str | os.PathLike[str],
        fmt: Literal["v1", "xvid"] = "v1",
        func: FuncExcept | None = None,
        header: bool = True,
        force: bool = False,
    ) -> None:
        func = func or self.to_file

        if not self:
            raise CustomValueError("Keyframes must be non-empty", func)

        out_path = Path(out).resolve()

        if out_path.exists():
            if not force and out_path.stat().st_size > 0:
                return

            out_path.unlink()

        out_path.parent.mkdir(parents=True, exist_ok=True)

        check_perms(out_path, "w+", func=func)

        match fmt:
            case "v1":
                out_text = list[str]()
                if header:
                    out_text.extend(["# keyframe format v1", "fps 0", ""])
                out_text.extend(f"{n} I -1" for n in self)
                out_text.append("")
            case "xvid":
                lut_self = set(self)
                out_text = list[str]()

                if header:
                    out_text.extend(["# XviD 2pass stat file", ""])

                for i in range(max(self) + 1):
                    if i in lut_self:
                        out_text.append("i")
                        lut_self.remove(i)
                    else:
                        out_text.append("b")

        out_path.unlink(True)
        out_path.touch()
        out_path.write_text("\n".join(out_text))

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, **kwargs: Any) -> Self:
        """
        Create a Keyframes object from a clip by checking frame props.

        Assumes that the clip has already been processed by scxvid.Scxvid or similar.

        Args:
            clip: Clip to get keyframes from.
            **kwargs: Additional keyword arguments to pass to clip_async_render.

        Returns:
            Keyframes from the clip.
        """

        def check_props(n: int, f: vs.VideoFrame) -> int:
            sc_next = f.props.get("_SceneChangeNext")
            sc_prev = f.props.get("_SceneChangePrev")

            if sc_next is None and sc_prev is None:
                raise CustomRuntimeError("No scenechange props are present!", cls)

            if sc_next:
                return n + 1
            if sc_prev:
                return n

            return -1

        frames = clip_async_render(clip, None, "Detecting scene changes...", check_props, **kwargs)
        return cls(f for f in set(frames) if f >= 0)

    @classmethod
    def from_file(cls, file: str | os.PathLike[str]) -> Self:
        file = SPath(file).resolve()

        if not file.exists():
            raise FileNotFoundError

        lines = [line.strip() for line in file.read_lines("utf-8") if line and not line.startswith("#")]

        if not lines:
            raise ValueError("No keyframe could be found!")

        match lines[0].lower():
            # XVID
            case line if line.startswith("fps"):
                split_lines = [line.split(" ") for line in lines]
                return cls(int(n) for n, t, *_ in split_lines if t.lower() == "i")
            # V1
            case line if line.startswith(("i", "b", "p", "n")):
                return cls(i for i, line in enumerate(lines) if line.startswith("i"))
            case _:
                raise ValueError("Could not determine keyframe file type!")

    @classmethod
    def from_param(cls, clip: vs.VideoNode, param: Self | str) -> Self:
        if isinstance(param, str):
            return cls.unique(clip, param)

        if isinstance(param, cls):
            return param

        return cls(param)

    @classmethod
    def unique(cls, clip: vs.VideoNode, key: str, **kwargs: Any) -> Self:
        """
        Get the keyframes from a clip and write them to a file.

        Assumes that the clip has already been processed by scxvid.Scxvid or similar.

        This method tries to generate a unique filename based on the clip's
        properties and the `key` prefix. If a file with that name exists and is
        not empty, the keyframes are loaded from the file. Otherwise, they are
        detected from the clip and then written to the file.

        Example:
            When working on a TV series, the episode number can be a convenient key
            (e.g. `"01"` for episode 1, `"02"` for episode 2, etc.):
            ```py
            keyframes = Keyframes.unique(clip, "01")
            ```

        Args:
            clip: The clip to get keyframes from.
            key: A prefix for the filename.
            **kwargs: Additional keyword arguments passed to
                [vstools.Keyframes.from_file][] or [vstools.Keyframes.from_clip][].

        Returns:
            An instance of [vstools.Keyframes][] containing the keyframes.
        """
        file = cls._get_unique_path(clip, key)

        if file.exists():
            if file.stat().st_size > 0:
                return cls.from_file(file, **kwargs)

            file.unlink()

        keyframes = cls.from_clip(clip, **kwargs)
        keyframes.to_file(file, force=True)

        return keyframes

    @staticmethod
    def _get_unique_path(clip: vs.VideoNode, key: str) -> SPath:
        key = SPath(key).stem + f"_{clip.num_frames}_{clip.fps_num}_{clip.fps_den}"

        return _get_keyframes_storage().get_file(key, ext=".txt")


class SceneBasedDynamicCache(DynamicClipsCache[int]):
    def __init__(self, clip: vs.VideoNode, keyframes: Keyframes | str, cache_size: int = 5) -> None:
        super().__init__(cache_size)

        self.clip = clip
        self.keyframes = Keyframes.from_param(clip, keyframes)

    @abstractmethod
    def get_clip(self, key: int) -> vs.VideoNode: ...

    def get_eval(self) -> vs.VideoNode:
        return self.clip.std.FrameEval(lambda n: self[self.keyframes.scenes.indices[n]])

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, keyframes: Keyframes | str, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return cls(clip, keyframes, *args, **kwargs).get_eval()


class SceneAverageStats(SceneBasedDynamicCache):
    _props_keys = ("Min", "Max", "Average")

    class _Cache(VSObject, dict[int, tuple[float, float, float]]):
        def __init__(self, clip: vs.VideoNode, keyframes: Keyframes, plane: int) -> None:
            self.props = clip.std.PlaneStats(plane=plane)
            self.keyframes = keyframes

        def __getitem__(self, idx: int) -> tuple[float, float, float]:
            if idx not in self:
                frame_range = self.keyframes.scenes[idx]
                cut_clip = self.props[frame_range.start : frame_range.stop]

                frames_min_max_avg = clip_data_gather(
                    cut_clip,
                    None,
                    lambda n, f: tuple(cast(float, f.props[f"PlaneStats{p}"]) for p in SceneAverageStats._props_keys),
                )

                frames_min, frames_max, frames_avgs = [[x[i] for x in frames_min_max_avg] for i in (0, 1, 2)]

                self[idx] = (min(frames_min), max(frames_max), sum(frames_avgs) / len(frames_avgs))

            return super().__getitem__(idx)

    def __init__(
        self,
        clip: vs.VideoNode,
        keyframes: Keyframes | str,
        prop: str = "SceneStats",
        plane: int = 0,
        cache_size: int = 5,
    ) -> None:
        super().__init__(clip, keyframes, cache_size)

        self.prop_keys = tuple(f"{prop}{x}" for x in self._props_keys)
        self.scene_avgs = self._Cache(self.clip, self.keyframes, plane)

    def get_clip(self, key: int) -> vs.VideoNode:
        return self.clip.std.SetFrameProps(**dict(zip(self.prop_keys, self.scene_avgs[key])))
