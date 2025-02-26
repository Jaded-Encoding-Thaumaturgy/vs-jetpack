from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vstools import inject_self, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser, _FullInterpolate

__all__ = [
    'Nnedi3', 'Nnedi3DR'
]


@dataclass
class NNEDI3(_FullInterpolate, _Antialiaser):
    nsize: int = 0
    nns: int = 4
    qual: int = 2
    etype: int = 0
    pscrn: int = 1

    opencl: bool = False

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.opencl and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        assert clip.format
        pscrn = min(self.pscrn, 1) if clip.format.sample_type == vs.FLOAT else self.pscrn
        return dict(nsize=self.nsize, nns=self.nns, qual=self.qual, etype=self.etype, pscrn=pscrn)

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> vs.VideoNode:
        interpolated = clip.znedi3.nnedi3(self.field, double_y or not self.drop_fields, **kwargs)
        return self.shift_interpolate(clip, interpolated, double_y, **kwargs)

    def full_interpolate(self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any) -> vs.VideoNode:
        return clip.sneedif.NNEDI3(self.field, double_y, double_x, transpose_first=self.transpose_first, **kwargs)

    _shift = 0.5

    @inject_self.property
    def kernel_radius(self) -> int:
        match self.nsize:
            case 1 | 5:
                return 16
            case 2 | 6:
                return 32
            case 3:
                return 48
            case _:
                return 8


class Nnedi3SS(NNEDI3, SuperSampler):
    ...


class Nnedi3SR(NNEDI3, SingleRater):
    ...


class Nnedi3DR(NNEDI3, DoubleRater):
    ...


class Nnedi3(NNEDI3, Antialiaser):
    ...
