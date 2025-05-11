from __future__ import annotations

from math import ceil
from typing import Any

from vstools import core, vs

from .zimg import ZimgComplexKernel

__all__ = [
    'Point',
    'Bilinear',
    'Lanczos',
]


class Point(ZimgComplexKernel):
    """Point resizer."""

    scale_function = resample_function = core.lazy.resize2.Point
    descale_function = core.lazy.descale.Depoint
    _static_kernel_radius = 1


class Bilinear(ZimgComplexKernel):
    """Bilinear resizer."""

    scale_function = resample_function = core.lazy.resize2.Bilinear
    descale_function = core.lazy.descale.Debilinear
    _static_kernel_radius = 1


class Lanczos(ZimgComplexKernel):
    """Lanczos resizer."""

    scale_function = resample_function = core.lazy.resize2.Lanczos
    descale_function = core.lazy.descale.Delanczos

    def __init__(self, taps: int = 3, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific number of taps.

        :param taps:    Determines the radius of the kernel.
        :param kwargs:  Additional keyword arguments passed to the superclass.
        """
        self.taps = taps
        super().__init__(**kwargs)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)
        if is_descale:
            return args | dict(taps=self.taps)
        return args | dict(filter_param_a=self.taps)

    @ZimgComplexKernel.cached_property
    def kernel_radius(self) -> int:
        return ceil(self.taps)
