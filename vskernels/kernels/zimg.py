from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jetpytools import inject_kwargs_params

from vstools import ConstantFormatVideoNode, Dar, FieldBased, Sar, inject_self, vs

from ..types import BorderHandling, Center, LeftShift, SampleGridModel, ShiftT, Slope, TopShift
from .abstract import Descaler
from .complex import ComplexKernel

__all__ = [
    'ZimgDescaler',
    'ZimgComplexKernel'
]


class ZimgDescaler(Descaler):
    if TYPE_CHECKING:
        @inject_self.cached
        @inject_kwargs_params
        def descale(
            self, clip: vs.VideoNode, width: int | None, height: int | None,
            shift: ShiftT = (0, 0),
            *,
            # `border_handling`, `sample_grid_model` and `field_based` from Descaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            field_based: FieldBased | None = None,
            # ZimgDescaler adds `blur` and `ignore_mask` parameters
            blur: float = 1.0, ignore_mask: vs.VideoNode | None = None,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...


class ZimgComplexKernel(ComplexKernel, ZimgDescaler):
    if TYPE_CHECKING:
        # Override signature to add `blur`
        @inject_self.cached
        @inject_kwargs_params
        def scale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: tuple[TopShift, LeftShift] = (0, 0),
            *,
            # `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar` from KeepArScaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            sar: Sar | float | bool | None = None, dar: Dar | float | bool | None = None,
            dar_in: Dar | bool | float | None = None, keep_ar: bool | None = None,
            # `linear` and `sigmoid` from LinearScaler
            linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False,
            # ZimgComplexKernel adds blur parameter
            blur: float = 1.0,
            **kwargs: Any
        ) -> vs.VideoNode:
            ...

        @inject_self.cached
        @inject_kwargs_params
        def descale(
            self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
            shift: ShiftT = (0, 0),
            *,
            # `border_handling`, `sample_grid_model` and `field_based` from Descaler
            border_handling: BorderHandling = BorderHandling.MIRROR,
            sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
            field_based: FieldBased | None = None,
            # `linear` and `sigmoid` from LinearDescaler
            linear: bool = False, sigmoid: bool | tuple[Slope, Center] = False,
            # `blur` and `ignore_mask` parameters from ZimgDescaler
            blur: float = 1.0, ignore_mask: vs.VideoNode | None = None,
            **kwargs: Any
        ) -> ConstantFormatVideoNode:
            ...

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)

        if not is_descale:
            for key in ('border_handling', 'ignore_mask'):
                args.pop(key, None)

        return args
