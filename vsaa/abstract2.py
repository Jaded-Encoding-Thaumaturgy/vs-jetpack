from abc import ABC
from dataclasses import KW_ONLY, dataclass, replace
from enum import IntFlag, auto
from typing import Any, ClassVar

from jetpytools import inject_self
from typing_extensions import Self

from vskernels import LeftShift, Scaler, TopShift
from vstools import ChromaLocation, ConstantFormatVideoNode, VSFunctionAllArgs, check_variable, core, normalize_seq, vs


@dataclass
class Deinterlacer(ABC):
    _deinterlacer_function: ClassVar[VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]]

    _: KW_ONLY
    tff: bool = False
    double_rate: bool = False
    transpose_first: bool = False

    class AADirection(IntFlag):
        VERTICAL = auto()
        HORIZONTAL = auto()
        BOTH = VERTICAL | HORIZONTAL

    def __post_init__(self) -> None:
        self._field_order = int(self.tff) + (int(self.double_rate) * 2)

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def deinterlace(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return self._deinterlacer_function(clip, self._field_order, **self.get_deint_args(**kwargs))

    def aa(self, clip: vs.VideoNode, direction: AADirection = AADirection.BOTH, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.aa)

        for y in sorted((aa_dir for aa_dir in self.AADirection), key=lambda x: x.value, reverse=self.transpose_first):
            if direction in (y, self.AADirection.BOTH):
                if y == self.AADirection.HORIZONTAL:
                    clip = clip.std.Transpose()

                clip = self.deinterlace(clip, **kwargs)

                if self.double_rate:
                    clip = core.std.Merge(clip[::2], clip[1::2])  # ?????

                if y == self.AADirection.HORIZONTAL:
                    clip = clip.std.Transpose()

        return clip

    def copy(self, **kwargs: Any) -> Self:
        """Returns a new Antialiaser class replacing specified fields with new values"""
        return replace(self, **kwargs)


class SuperSampler(Deinterlacer, Scaler, ABC):
    def get_ss_args(self, **kwargs: Any) -> dict[str, Any]:
        return dict(dh=True) | kwargs

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.scale)

        dest_dimensions = list(self._wh_norm(clip, width, height))
        sy, sx = shift

        cloc = list(ChromaLocation.get_offsets(clip, ChromaLocation.from_video(clip)))
        nshift: list[list[float]] = [
            normalize_seq(sx, clip.format.num_planes),
            normalize_seq(sy, clip.format.num_planes)
        ]

        if not self.transpose_first:
            dest_dimensions.reverse()
            cloc.reverse()
            nshift.reverse()

        for x, dim in enumerate(dest_dimensions):
            is_width = not x and self.transpose_first or not self.transpose_first and x

            while (clip.width if is_width else clip.height) < dim:

                field = int(self.tff) if nshift[x][0] == 0 else 1 if nshift[x][0] > 0 else 0

                for y in range(clip.format.num_planes):
                    nshift[x][y] = (nshift[x][y] + -0.25 if field else 0.25) * 2

                if is_width:
                    clip = clip.std.Transpose()

                clip = self.deinterlace(clip, **self.get_deint_args() | self.get_ss_args(**kwargs))

                if is_width:
                    clip = clip.std.Transpose()

        if not self.transpose_first:
            nshift.reverse()

        return clip.fmtc.resample(width, height, nshift[0], nshift[1])


@dataclass
class NNEDI3(SuperSampler, Deinterlacer):
    nsize: int | None = None
    nns: int | None = None
    qual: int | None = None
    etype: int | None = None
    pscrn: int | None = None

    _deinterlacer_function = core.lazy.znedi3.nnedi3

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return dict(
            nsize=self.nsize,
            nns=self.nns,
            qual=self.qual,
            etype=self.etype,
            pscrn=self.pscrn
        ) | kwargs
