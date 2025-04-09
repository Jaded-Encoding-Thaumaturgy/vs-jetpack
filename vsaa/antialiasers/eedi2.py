from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vstools import ConstantFormatVideoNode, check_variable_format, core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser, _FullInterpolate

__all__ = [
    'Eedi2', 'Eedi2DR'
]


@dataclass
class EEDI2(_FullInterpolate, _Antialiaser):
    mthresh: int = 10
    lthresh: int = 20
    vthresh: int = 20
    estr: int = 2
    dstr: int = 4
    maxd: int = 24
    pp: int = 1

    cuda: bool = False

    # Class Variable
    _shift = -0.5

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.cuda and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return dict(
            mthresh=self.mthresh,
            lthresh=self.lthresh,
            vthresh=self.vthresh,
            estr=self.estr,
            dstr=self.dstr,
            maxd=self.maxd,
            pp=self.pp
        ) | kwargs

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable_format(clip, self.__class__)

        kwargs = self.get_aa_args(clip) | kwargs

        if self.cuda:
            inter = core.eedi2cuda.EEDI2(clip, self.field, **kwargs)
        else:
            inter = core.eedi2.EEDI2(clip, self.field, **kwargs)

        if not double_y:
            if self.drop_fields:
                inter = inter.std.SeparateFields(not self.field)[::2]

                inter = self._shifter.shift(inter, (0.5 - 0.75 * self.field, 0))
            else:
                inter = self._scaler.scale(  # type: ignore[assignment]
                    inter, clip.width, clip.height, (self._shift * int(not self.field), 0)
                )

        return self._post_interpolate(clip, inter, double_y)  # pyright: ignore[reportArgumentType]

    def full_interpolate(
        self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return core.eedi2cuda.Enlarge2(clip, **kwargs)


class Eedi2SS(EEDI2, SuperSampler):
    _static_kernel_radius = 2


class Eedi2SR(EEDI2, SingleRater):
    ...


class Eedi2DR(EEDI2, DoubleRater):
    ...


class Eedi2(Eedi2SS, Antialiaser):
    ...
