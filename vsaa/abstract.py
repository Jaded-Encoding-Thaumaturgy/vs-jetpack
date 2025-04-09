from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from dataclasses import replace
from itertools import zip_longest
from math import ceil, log2
from typing import Any, Callable, ClassVar, overload

from typing_extensions import Self

from vsexprtools import norm_expr
from vskernels import Catrom, Kernel, KernelT, NoShift, Scaler, ScalerT
from vskernels.types import LeftShift, TopShift
from vstools import ConstantFormatVideoNode, check_progressive, check_variable, core, inject_self, vs, vs_object

from .enums import AADirection

__all__ = [
    'SuperSampler',
    'SingleRater', 'DoubleRater',
    'Antialiaser'
]


class _SingleInterpolate(ABC):
    _shift: ClassVar[float]

    def _post_interpolate(
        self,
        clip: ConstantFormatVideoNode,
        aa_clip: ConstantFormatVideoNode,
        double_y: bool,
        mclip: ConstantFormatVideoNode | None = None,
    ) -> ConstantFormatVideoNode:
        if not double_y and mclip:
            return norm_expr([clip, aa_clip, mclip], 'z y x ?', func=self.__class__._post_interpolate)

        return aa_clip

    @abstractmethod
    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        ...


@dataclass
class _Antialiaser(_SingleInterpolate, ABC):
    _: KW_ONLY
    field: int = 0
    drop_fields: bool = True
    transpose_first: bool = False
    shifter: KernelT = Catrom
    scaler: ScalerT | None = None

    def __post_init__(self) -> None:
        self._shifter = Kernel.ensure_obj(self.shifter)
        self._scaler = self._shifter.ensure_obj(self.scaler, self.__class__)

    def _preprocess_clip(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        return clip

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def shift_interpolate(
        self,
        clip: vs.VideoNode,
        inter: vs.VideoNode,
        double_y: bool,
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)
        assert check_variable(inter, self.__class__)

        if not double_y and not self.drop_fields:
            shift = (self._shift * int(not self.field), 0)

            inter = self._scaler.scale(inter, clip.width, clip.height, shift)

            return self._post_interpolate(clip, inter, double_y)  # type: ignore[arg-type]

        return inter

    def copy(self, **kwargs: Any) -> Self:
        return replace(self, **kwargs)


class _FullInterpolate(_SingleInterpolate, ABC):
    @abstractmethod
    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        ...

    @abstractmethod
    def full_interpolate(
        self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        ...


class SuperSampler(_Antialiaser, Scaler, ABC):
    def get_ss_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any
    ) -> vs.VideoNode:

        assert check_progressive(clip, self.scale)

        clip = self._preprocess_clip(clip)
        width, height = self._wh_norm(clip, width, height)

        if (clip.width, clip.height) == (width, height):
            return clip

        kwargs = self.get_aa_args(clip, **kwargs) | self.get_ss_args(clip, **kwargs) | kwargs

        divw, divh = (ceil(size) for size in (width / clip.width, height / clip.height))

        mult_x, mult_y = (int(log2(divs)) for divs in (divw, divh))

        cdivw, cdivh = 1 << clip.format.subsampling_w, 1 << clip.format.subsampling_h

        upscaled = clip

        def _transpose(before: bool, is_width: int, y: int, x: int) -> None:
            nonlocal upscaled

            before = self.transpose_first if before else not self.transpose_first

            if ((before or not y) if is_width else (before and x)):
                upscaled = upscaled.std.Transpose()

        for (y, x) in zip_longest([True] * mult_y, [True] * mult_x, fillvalue=False):
            if isinstance(self, _FullInterpolate) and self.is_full_interpolate_enabled(x, y):
                upscaled = self.full_interpolate(upscaled, y, x, **kwargs)
            else:
                for isx, val in enumerate([y, x]):
                    if val:
                        _transpose(True, isx, y, x)

                        upscaled = self.interpolate(upscaled, True, **kwargs)

                        _transpose(False, isx, y, x)

            topshift = leftshift = cleftshift = ctopshift = 0.0

            if y and self._shift:
                topshift = ctopshift = self._shift

                if cdivw == 2 and cdivh == 2:
                    ctopshift -= 0.125
                elif cdivw == 1 and cdivh == 2:
                    ctopshift += 0.125

            cresshift = 0.0

            if x and self._shift:
                leftshift = cleftshift = self._shift

                if cdivw in {4, 2} and cdivh in {4, 2, 1}:
                    cleftshift = self._shift + 0.5

                    if cdivw == 4 and cdivh == 1:
                        cresshift = 0.125 * 1
                    elif cdivw == 2 and cdivh == 2:
                        cresshift = 0.125 * 2
                    elif cdivw == 2 and cdivh == 1:
                        cresshift = 0.125 * 3

                    cleftshift -= cresshift

            if isinstance(self._shifter, NoShift):
                if upscaled.format.subsampling_h or upscaled.format.subsampling_w:
                    upscaled = Catrom.shift(upscaled, 0, [0, cleftshift + cresshift])
            else:
                upscaled = self._shifter.shift(
                    upscaled, [topshift, ctopshift], [leftshift, cleftshift]
                )

        return self._scaler.scale(upscaled, width, height, shift)


class SingleRater(_Antialiaser, ABC):
    def get_sr_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        return {}

    @overload
    def aa(
        self, clip: vs.VideoNode, y: bool = True, x: bool = True, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def aa(
        self, clip: vs.VideoNode, dir: AADirection = AADirection.BOTH, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        ...

    def aa(
        self, clip: vs.VideoNode, y_or_dir: bool | AADirection = True, x: bool = True, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        if isinstance(y_or_dir, AADirection):
            y, x = y_or_dir.to_yx()
        else:
            y = y_or_dir

        clip = self._preprocess_clip(clip)

        return self._do_aa(clip, y, x, **kwargs)

    def _do_aa(
        self, clip: ConstantFormatVideoNode, y: bool = True, x: bool = False, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        kwargs = self.get_aa_args(clip, **kwargs) | self.get_sr_args(clip, **kwargs) | kwargs

        upscaled = clip

        def _transpose(before: bool, is_width: int) -> None:
            nonlocal upscaled

            before = self.transpose_first if before else not self.transpose_first

            if ((before or not y) if is_width else (before and x)):
                upscaled = upscaled.std.Transpose()

                if 'mclip' in kwargs:
                    kwargs["mclip"] = kwargs.pop('mclip').std.Transpose()

                if 'sclip' in kwargs:
                    kwargs["sclip"] = kwargs.pop('sclip').std.Transpose()

        for isx, val in enumerate([y, x]):
            if val:
                _transpose(True, isx)

                if isinstance(self, _FullInterpolate) and self.is_full_interpolate_enabled(x, y):
                    upscaled = self.full_interpolate(upscaled, False, False, **kwargs)
                else:
                    upscaled = self.interpolate(upscaled, False, **kwargs)

                _transpose(False, isx)

        return upscaled


@dataclass(kw_only=True)
class DoubleRater(SingleRater, ABC):
    merge_func: Callable[[vs.VideoNode, vs.VideoNode], ConstantFormatVideoNode] = core.proxied.std.Merge

    @overload
    def draa(
        self, clip: vs.VideoNode, y: bool = True, x: bool = True, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        ...

    @overload
    def draa(
        self, clip: vs.VideoNode, dir: AADirection = AADirection.BOTH, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        ...

    def draa(
        self, clip: vs.VideoNode, y_or_dir: bool | AADirection = True, x: bool = True, /, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        if isinstance(y_or_dir, AADirection):
            y, x = y_or_dir.to_yx()
        else:
            y = y_or_dir

        clip = self._preprocess_clip(clip)

        original_field = int(self.field)

        self.field = 0
        aa0 = super()._do_aa(clip, y, x, **kwargs)

        self.field = 1
        aa1 = super()._do_aa(clip, y, x, **kwargs)

        self.field = original_field

        return self.merge_func(aa0, aa1)


@dataclass
class Antialiaser(DoubleRater, SuperSampler, ABC):
    ...
