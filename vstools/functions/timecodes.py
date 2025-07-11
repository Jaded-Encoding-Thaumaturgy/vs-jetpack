from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, Iterable, Literal, NamedTuple, TypeVar, overload

import vapoursynth as vs
from jetpytools import CustomValueError, FilePathType, FuncExceptT, LinearRangeLut, Sentinel, SPath, inject_self
from typing_extensions import Self

from ..enums import Matrix, SceneChangeMode
from ..exceptions import FramesLengthError, InvalidTimecodeVersionError
from ..types import VideoNodeT
from .file import PackageStorage
from .render import clip_async_render

__all__ = ["Keyframes", "LWIndex", "Timecodes"]


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

        from .funcs import fallback

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
    def from_file(cls, file: FilePathType, ref: vs.VideoNode, /, *, func: FuncExceptT | None = None) -> Self:
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
        cls, file: FilePathType, length: int, den: int | None = None, /, func: FuncExceptT | None = None
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
        func: FuncExceptT | None = None,
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

        version, *_timecodes = file.read_text().splitlines()

        if "v1" in version:

            def _norm(xd: str) -> Fraction:
                return Fraction(round(denominator / float(xd)), denominator)

            assume = None

            timecodes_d = dict[tuple[int | None, int | None], Fraction]()

            for line in _timecodes:
                if line.startswith("#"):
                    continue

                if line.startswith("Assume"):
                    assume = _norm(_timecodes[0][7:])
                    continue

                starts, ends, _fps = line.split(",")
                timecodes_d[(int(starts), int(ends) + 1)] = _norm(_fps)

            norm_timecodes = cls.normalize_range_timecodes(timecodes_d, length, assume)
        elif "v2" in version:
            timecodes_l = [float(t) for t in _timecodes if not t.startswith("#")]
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

    def assume_vfr(self, clip: VideoNodeT, func: FuncExceptT | None = None) -> VideoNodeT:
        """
        Force the given clip to be assumed to be vfr by other applications.

        Args:
            clip: Clip to process.
            func: Function returned for custom error handling. This should only be set by VS package developers.

        Returns:
            Clip that should always be assumed to be vfr by other applications.
        """
        from ..utils import replace_ranges

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

    def to_file(self, out: FilePathType, format: int = V2, func: FuncExceptT | None = None) -> None:
        """
        Write timecodes to a file.

        This file should always be muxed into the video container when working with Variable Frame Rate video.

        Args:
            out: Path to write the file to.
            format: Format to write the file to.
        """
        from ..utils import check_perms

        func = func or self.to_file

        out_path = Path(str(out)).resolve()

        check_perms(out_path, "w+", func=func)

        InvalidTimecodeVersionError.check(self.to_file, format)

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

    V1 = 1
    XVID = -1

    WWXD: ClassVar = SceneChangeMode.WWXD
    SCXVID: ClassVar = SceneChangeMode.SCXVID

    class _Scenes(dict[int, range]):
        __slots__ = ("indices",)

        def __init__(self, kf: Keyframes) -> None:
            if kf:
                super().__init__({i: range(x, y) for i, (x, y) in enumerate(zip(kf, [*kf[1:], 1 << 32]))})

            self.indices = LinearRangeLut(self)

    def __init__(self, iterable: Iterable[int] = [], *, _dummy: bool = False) -> None:
        super().__init__(sorted(iterable))

        self._dummy = _dummy

        self.scenes = self.__class__._Scenes(self)

    @staticmethod
    def _get_unique_path(clip: vs.VideoNode, key: str) -> SPath:
        key = SPath(str(key)).stem + f"_{clip.num_frames}_{clip.fps_num}_{clip.fps_den}"

        return _get_keyframes_storage().get_file(key, ext=".txt")

    @classmethod
    def unique(cls, clip: vs.VideoNode, key: str, **kwargs: Any) -> Self:
        file = cls._get_unique_path(clip, key)

        if file.exists():
            if file.stat().st_size > 0:
                return cls.from_file(file, **kwargs)

            file.unlink()

        keyframes = cls.from_clip(clip, **kwargs)
        keyframes.to_file(file, force=True)

        return keyframes

    @classmethod
    def from_clip(
        cls,
        clip: vs.VideoNode,
        mode: SceneChangeMode | int = WWXD,
        height: int | Literal[False] = 360,
        **kwargs: Any,
    ) -> Self:
        mode = SceneChangeMode(mode)

        clip = mode.prepare_clip(clip, height)

        frames = clip_async_render(clip, None, "Detecting scene changes...", mode.lambda_cb(), **kwargs)

        return cls(Sentinel.filter(frames))

    @inject_self.with_args(_dummy=True)
    def to_clip(
        self,
        clip: vs.VideoNode,
        *,
        mode: SceneChangeMode | int = WWXD,
        height: int | Literal[False] = 360,
        prop_key: str = next(iter(SceneChangeMode.SCXVID.prop_keys)),
        scene_idx_prop: bool = False,
    ) -> vs.VideoNode:
        from ..utils import replace_ranges

        propset_clip = clip.std.SetFrameProp(prop_key, True)

        if self._dummy:
            mode = SceneChangeMode(mode)

            prop_clip, callback = mode.prepare_clip(clip, height), mode.check_cb()

            out = replace_ranges(clip, propset_clip, callback, prop_src=prop_clip)
        else:
            out = replace_ranges(clip, propset_clip, self)

        if not scene_idx_prop:
            return out

        def _add_scene_idx(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            f = f.copy()

            f.props._SceneIdx = self.scenes.indices[n]

            return f

        return out.std.ModifyFrame(out, _add_scene_idx)

    @classmethod
    def from_file(cls, file: FilePathType, **kwargs: Any) -> Self:
        file = SPath(str(file)).resolve()

        if not file.exists():
            raise FileNotFoundError

        if file.stat().st_size <= 0:
            raise OSError("File is empty!")

        lines = [line.strip() for line in file.read_lines("utf-8") if line and not line.startswith("#")]

        if not lines:
            raise ValueError("No keyframe could be found!")

        kf_type: int | None = None

        line = lines[0].lower()

        if line.startswith("fps"):
            kf_type = Keyframes.XVID
        elif line.startswith(("i", "b", "p", "n")):
            kf_type = Keyframes.V1

        if kf_type is None:
            raise ValueError("Could not determine keyframe file type!")

        if kf_type == Keyframes.V1:
            return cls(i for i, line in enumerate(lines) if line.startswith("i"))

        if kf_type == Keyframes.XVID:
            split_lines = [line.split(" ") for line in lines]

            return cls(int(n) for n, t, *_ in split_lines if t.lower() == "i")

        raise ValueError("Invalid keyframe file type!")

    def to_file(
        self,
        out: FilePathType,
        format: int = V1,
        func: FuncExceptT | None = None,
        header: bool = True,
        force: bool = False,
    ) -> None:
        from ..utils import check_perms

        func = func or self.to_file

        out_path = Path(str(out)).resolve()

        if out_path.exists():
            if not force and out_path.stat().st_size > 0:
                return

            out_path.unlink()

        out_path.parent.mkdir(parents=True, exist_ok=True)

        check_perms(out_path, "w+", func=func)

        if format == Keyframes.V1:
            out_text = [*(["# keyframe format v1", "fps 0", ""] if header else []), *(f"{n} I -1" for n in self), ""]
        elif format == Keyframes.XVID:
            lut_self = set(self)
            out_text = list[str]()

            if header:
                out_text.extend(["# XviD 2pass stat file", ""])

            for i in range(max(self)):
                if i in lut_self:
                    out_text.append("i")
                    lut_self.remove(i)
                else:
                    out_text.append("b")

        out_path.unlink(True)
        out_path.touch()
        out_path.write_text("\n".join(out_text))

    @classmethod
    def from_param(cls, clip: vs.VideoNode, param: Self | str) -> Self:
        if isinstance(param, str):
            return cls.unique(clip, param)

        if isinstance(param, cls):
            return param

        return cls(param)


KeyframesBoundT = TypeVar("KeyframesBoundT", bound=Keyframes)


@dataclass
class LWIndex:
    stream_info: StreamInfo
    frame_data: list[Frame]
    keyframes: Keyframes

    class Regex:
        frame_first = re.compile(
            r"Index=(?P<Index>-?[0-9]+),POS=(?P<POS>-?[0-9]+),PTS=(?P<PTS>-?[0-9]+),"
            r"DTS=(?P<DTS>-?[0-9]+),EDI=(?P<EDI>-?[0-9]+)"
        )

        frame_second = re.compile(
            r"Key=(?P<Key>-?[0-9]+),Pic=(?P<Pic>-?[0-9]+),POC=(?P<POC>-?[0-9]+),"
            r"Repeat=(?P<Repeat>-?[0-9]+),Field=(?P<Field>-?[0-9]+)"
        )

        streaminfo = re.compile(
            r"Codec=(?P<Codec>[0-9]+),TimeBase=(?P<TimeBase>[0-9\/]+),Width=(?P<Width>[0-9]+),"
            r"Height=(?P<Height>[0-9]+),Format=(?P<Format>[0-9a-zA-Z]+),ColorSpace=(?P<ColorSpace>[0-9]+)"
        )

    class StreamInfo(NamedTuple):
        codec: int
        timebase: Fraction
        width: int
        height: int
        format: str
        colorspace: Matrix

    class Frame(NamedTuple):
        idx: int
        pos: int
        pts: int
        dts: int
        edi: int
        key: int
        pic: int
        poc: int
        repeat: int
        field: int

    @classmethod
    def from_file(
        cls, file: FilePathType, ref_or_length: int | vs.VideoNode | None = None, *, func: FuncExceptT | None = None
    ) -> LWIndex:
        func = func or cls.from_file

        file = Path(str(file)).resolve()

        length = ref_or_length.num_frames if isinstance(ref_or_length, vs.VideoNode) else ref_or_length

        data = file.read_text("latin1").splitlines()

        indexstart, indexend = data.index("</StreamInfo>") + 1, data.index("</LibavReaderIndex>")

        if length and (idxlen := ((indexend - indexstart) // 2)) != length:
            raise FramesLengthError(
                func, "", "index file length mismatch with specified length!", reason={"index": idxlen, "clip": length}
            )

        sinfomatch = LWIndex.Regex.streaminfo.match(data[indexstart - 2])

        assert sinfomatch

        timebase_num, timebase_den = [int(i) for i in sinfomatch.group("TimeBase").split("/")]

        streaminfo = LWIndex.StreamInfo(
            int(sinfomatch.group("Codec")),
            Fraction(timebase_num, timebase_den),
            int(sinfomatch.group("Width")),
            int(sinfomatch.group("Height")),
            sinfomatch.group("Format"),
            Matrix(int(sinfomatch.group("ColorSpace"))),
        )

        frames = list[LWIndex.Frame]()

        for i in range(indexstart, indexend, 2):
            match_first = LWIndex.Regex.frame_first.match(data[i])
            match_second = LWIndex.Regex.frame_second.match(data[i + 1])

            for match, keys in [
                (match_first, ["Index", "POS", "PTS", "DTS", "EDI"]),
                (match_second, ["Key", "Pic", "POC", "Repeat", "Field"]),
            ]:
                assert match

                frames.append(LWIndex.Frame(*(int(match.group(x)) for x in keys)))

        frames = sorted(frames, key=lambda x: x.pts)

        keyframes = Keyframes(i for i, f in enumerate(frames) if f.key)

        return LWIndex(streaminfo, frames, keyframes)
