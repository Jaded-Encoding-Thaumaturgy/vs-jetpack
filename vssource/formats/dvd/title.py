from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, SupportsIndex, overload

from jetpytools import CustomValueError, to_arr

from vstools import VSObjectABC, get_prop, set_output, vs

from .utils import AC3_FRAME_LENGTH, PCR_CLOCK, absolute_time_from_timecode

if TYPE_CHECKING:
    from .IsoFile import IsoFile

__all__ = ["Title"]


@dataclass
class SplitTitle:
    # maybe just return None instead of a SplitTitle with video None
    video: vs.VideoNode | None
    audios: list[vs.AudioNode | None]
    chapters: list[int]

    _title: Title
    _split_chpts: tuple[int, int]  # inclusive inclusive

    def ac3(self, outfile: str, audio_i: int = 0) -> float:
        return SplitHelper.split_range_ac3(self._title, *self._split_chpts, audio_i, outfile)

    def __repr__(self) -> str:
        if self.video is None:
            return "None"
        # TODO: use absolutetime from title
        duration_times = [1 / float(self.video.fps)] * len(self.video)
        absolute_time = absolute_time_from_timecode(duration_times)

        chapters = self.chapters

        chapter_lengths = [
            (absolute_time[chapters[i + 1] - 1] + duration_times[chapters[i + 1] - 1]) - absolute_time[chapters[i]]
            for i in range(len(self.chapters) - 1)
        ]

        chapter_lengths_str = [str(datetime.timedelta(seconds=x)) for x in chapter_lengths]

        timestrings = [str(datetime.timedelta(seconds=absolute_time[x])) for x in self.chapters]

        to_print = ["Chapters:"]

        to_print.extend(
            [
                f"{i:02} {tms:015} {cptls:015} {cpt}"
                for i, tms, cptls, cpt in zip(count(1), timestrings, chapter_lengths_str, self.chapters)
            ]
        )

        to_print.append("Audios: (fz)")

        if len(self.audios) >= 1:
            to_print.extend([f"{i} {a}" for i, a in enumerate(self.audios)])

        return "\n".join(to_print)


class TitleAudios(VSObjectABC, Sequence[vs.AudioNode]):
    def __init__(self, title: Title) -> None:
        self.title = title

        self.cache = dict[int, vs.AudioNode | None](dict.fromkeys(range(len(self.title._audios))))

    @overload
    def __getitem__(self, key: SupportsIndex) -> vs.AudioNode: ...

    @overload
    def __getitem__(self, key: slice) -> list[vs.AudioNode]: ...

    def __getitem__(self, key: SupportsIndex | slice) -> vs.AudioNode | list[vs.AudioNode]:
        if isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]

        i = int(key)

        if i not in self.cache:
            raise KeyError

        if _anode := self.cache[i]:
            return _anode

        asd = self.title._audios[i]

        anode: vs.AudioNode
        args = (str(self.title._core.iso_path), self.title._vts, i, self.title._dvdsrc_ranges)
        if asd.startswith("ac3"):
            anode = vs.core.dvdsrc2.FullVtsAc3(*args)
        elif asd.startswith("lpcm"):
            anode = vs.core.dvdsrc2.FullVtsLpcm(*args)
        else:
            raise CustomValueError("Invalid index for audio node!", self.__class__)

        strt = (get_prop(anode, "Stuff_Start_PTS", int) * anode.sample_rate) / PCR_CLOCK
        endd = (get_prop(anode, "Stuff_End_PTS", int) * anode.sample_rate) / PCR_CLOCK

        start, end = int(strt), int(endd)

        if start >= 0:
            anode = anode[start : len(anode) - end]
        else:
            anode = vs.core.std.BlankAudio(anode, length=-start) + anode[: len(anode) - end]

        self.cache[i] = anode

        return anode

    def __len__(self) -> int:
        return len(self.cache)


@dataclass
class Title:
    video: vs.VideoNode
    chapters: list[int]

    # only for reference for gui or sth
    cell_changes: list[int]

    _core: IsoFile
    _title: int
    _vts: int
    _vobidcellids_to_take: list[tuple[int, int]]
    _dvdsrc_ranges: list[int]
    _absolute_time: list[float]
    _duration_times: list[float]
    _audios: list[str]
    _patched_end_chapter: int | None

    def __post_init__(self) -> None:
        self.audios = TitleAudios(self)

    @property
    def audio(self) -> vs.AudioNode:
        if not self.audios:
            raise CustomValueError(f"No main audio found in this {self.__class__.__name__}!")
        return self.audios[0]

    def split_at(self, splits: list[int], audio: int | list[int] | None = None) -> tuple[SplitTitle, ...]:
        # Check if chapters are still valid, user is allowed to manipulated them
        last_chpt = -1
        for a in self.chapters:
            if a < 0:
                raise CustomValueError(f"Negative chapter point {a}", self.split_at)
            if a <= last_chpt:
                raise CustomValueError(f"Chapter must be monotonly increasing {a} before {last_chpt}", self.split_at)
            if a > len(self.video):
                raise CustomValueError("Chapter must not be higher than video length", self.split_at)
            last_chpt = a
        output_cnt = SplitHelper._sanitize_splits(self, splits)
        video = SplitHelper.split_video(self, splits)
        chapters = SplitHelper.split_chapters(self, splits)

        audios: list[list[vs.AudioNode | None]]

        if audio is not None and (audio := to_arr(audio)):
            audio_per_output_cnt = len(audio)

            auds = [SplitHelper.split_audio(self, splits, a) for a in audio]

            audios = [[auds[j][i] for j in range(audio_per_output_cnt)] for i in range(output_cnt)]
        else:
            audios = [[]] * output_cnt

        fromy = 1
        from_to_s = list[tuple[int, int]]()

        for j in splits:
            from_to_s.append((fromy, j - 1))
            fromy = j

        from_to_s.append((fromy, len(self.chapters) - 1))

        return tuple(SplitTitle(v, a, c, self, f) for v, a, c, f in zip(video, audios, chapters, from_to_s))

    def split_ranges(
        self, split: list[tuple[int, int]], audio: list[int] | int | None = None
    ) -> tuple[SplitTitle, ...]:
        return tuple(self.split_range(start, end, audio) for start, end in split)

    def split_range(self, start: int, end: int, audio: list[int] | int | None = None) -> SplitTitle:
        if start < 0:
            start = len(self.chapters) + start

        if end < 0:
            end = len(self.chapters) + end

        if start == 1 and end == len(self.chapters) - 1:
            return self.split_at([], audio)[0]

        if start == 1:
            return self.split_at([end + 1], audio)[0]

        if end == len(self.chapters) - 1:
            return self.split_at([start], audio)[1]

        return self.split_at([start, end + 1], audio)[1]

    def preview(self, split: SplitTitle | Sequence[SplitTitle] | None = None) -> None:
        set_output(self.video, f"title v{self._title}")

        if split is not None:
            split = to_arr(split)

            for i, s in enumerate(split):
                if s.video:
                    set_output(s.video, f"split {i}")

            for i, s in enumerate(split):
                if len(s.audios) >= 1:
                    for j, audio in enumerate(s.audios):
                        if audio:
                            set_output(audio, f"split {i} - {j}")

    def dump_ac3(self, a: str, audio_i: int = 0, only_calc_delay: bool = False) -> float:
        if not self._audios[audio_i].startswith("ac3"):
            raise CustomValueError(f"Audio at {audio_i} is not ac3", self.dump_ac3)

        nd = vs.core.dvdsrc2.RawAc3(str(self._core.iso_path), self._vts, audio_i, self._dvdsrc_ranges)

        if not only_calc_delay:
            with open(a, "wb") as wrt:
                wrt.writelines(bytes(f[0]) for f in nd.frames())

        return float(get_prop(nd, "Stuff_Start_PTS", int)) / PCR_CLOCK

    def __repr__(self) -> str:
        chapters = [*self.chapters]
        chapter_lengths = [
            (self._absolute_time[chapters[i + 1] - 1] + self._duration_times[chapters[i + 1] - 1])
            - self._absolute_time[chapters[i]]
            for i in range(len(self.chapters) - 1)
        ]

        chapter_lengths_str = [str(datetime.timedelta(seconds=x)) for x in chapter_lengths]
        timestrings = [str(datetime.timedelta(seconds=self._absolute_time[x])) for x in self.chapters[:-1]]

        to_print = "Chapters:\n"
        for i in range(len(timestrings)):
            to_print += f"{i + 1:02} {timestrings[i]:015} {chapter_lengths_str[i]:015} {self.chapters[i]}"

            if i == 0:
                to_print += " (faked)"

            if self._patched_end_chapter is not None and i == len(timestrings) - 1:
                delta = self.chapters[i] - self._patched_end_chapter
                to_print += f" (originally {self._patched_end_chapter} delta {delta})"

            to_print += "\n"

        to_print += f"\ncellchange: {self.cell_changes}\n"
        to_print += "\nAudios: (fz)\n"
        for i, a in enumerate(self._audios):
            to_print += f"{i} {a}\n"

        return to_print.strip()


class SplitHelper:
    @staticmethod
    def split_range_ac3(title: Title, f: int, t: int, audio_i: int, outfile: str) -> float:
        nd = vs.core.dvdsrc2.RawAc3(str(title._core.iso_path), title._vts, audio_i, title._dvdsrc_ranges)

        start, _ = (get_prop(nd, f"Stuff_{x}_PTS", int) for x in ("Start", "End"))

        raw_start = title._absolute_time[title.chapters[f - 1]] * PCR_CLOCK
        raw_end = (title._absolute_time[title.chapters[t]] + title._duration_times[title.chapters[t]]) * PCR_CLOCK

        start_pts = raw_start + start
        end_pts = start_pts + (raw_end - raw_start)

        audio_offset_pts = 0.0

        with open(outfile, "wb") as outf:
            start = int(start_pts / AC3_FRAME_LENGTH)

            for j, frame in enumerate(nd[start:].frames(close=True)):
                i = j + start
                pkt_start_pts = i * AC3_FRAME_LENGTH
                pkt_end_pts = (i + 1) * AC3_FRAME_LENGTH

                assert pkt_end_pts > start_pts

                if pkt_start_pts < start_pts:
                    audio_offset_pts = start_pts - pkt_start_pts

                outf.write(bytes(frame[0]))

                if pkt_end_pts > end_pts:
                    break

        return audio_offset_pts / PCR_CLOCK

    @staticmethod
    def split_chapters(title: Title, splits: list[int]) -> list[list[int]]:
        out = list[list[int]]()

        rebase = title.chapters[0]  # normally 0
        chaps = list[int]()

        for i, a in enumerate(title.chapters):
            chaps.append(a - rebase)

            if (i + 1) in splits:
                rebase = a

                out.append(chaps)
                chaps = [0]

        if len(chaps) >= 1:
            out.append(chaps)

        assert len(out) == len(splits) + 1

        return out

    @staticmethod
    def split_video(title: Title, splits: list[int]) -> tuple[vs.VideoNode | None, ...]:
        reta = SplitHelper._cut_split(title, splits, title.video, SplitHelper._cut_fz_v)
        assert len(reta) == len(splits) + 1
        return reta

    @staticmethod
    def split_audio(title: Title, splits: list[int], i: int = 0) -> tuple[vs.AudioNode | None, ...]:
        reta = SplitHelper._cut_split(title, splits, title.audios[i], SplitHelper._cut_fz_a)
        assert len(reta) == len(splits) + 1
        return reta

    @staticmethod
    def _sanitize_splits(title: Title, splits: list[int]) -> int:
        assert isinstance(splits, list)

        lasta = -1

        for a in splits:
            assert isinstance(a, int)
            if not (a > lasta):
                raise CustomValueError("Chapter splits are not ordered correctly!", SplitHelper._sanitize_splits)
            if not (a <= len(title.chapters)):
                raise CustomValueError("Chapter split is out of bounds!", SplitHelper._sanitize_splits)

            lasta = a

        return len(splits) + 1

    @staticmethod
    def _cut_split[T](
        title: Title, splits: list[int], a: T, b: Callable[[Title, T, int, int], T | None]
    ) -> tuple[T | None, ...]:
        out, last = list[T | None](), 0

        for s in splits:
            index = s - 1
            out.append(b(title, a, last, index))
            last = index

        out.append(b(title, a, last, len(title.chapters) - 1))

        return tuple(out)

    @staticmethod
    def _cut_fz_v(title: Title, vnode: vs.VideoNode, f: int, t: int) -> vs.VideoNode | None:
        f = title.chapters[f]
        t = title.chapters[t]
        assert f >= 0
        assert t <= len(vnode)
        assert f <= t

        if f == t:
            return None
        else:
            return vnode[f:t]

    @staticmethod
    def _cut_fz_a(title: Title, anode: vs.AudioNode, start: int, end: int) -> vs.AudioNode | None:
        chapter_idxs = [title.chapters[i] for i in (start, end)]
        timecodes = [
            title._absolute_time[i]
            if i != len(title._absolute_time)
            else title._absolute_time[i - 1] + title._duration_times[i - 1]
            for i in chapter_idxs
        ]

        samples_start, samples_end, *_ = [min(round(i * anode.sample_rate), anode.num_samples) for i in timecodes]

        if samples_start == samples_end:
            return None
        else:
            return anode[samples_start:samples_end]
