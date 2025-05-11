from __future__ import annotations

from math import ceil
from typing import Any

from vstools import core, vs

from .complex import ComplexKernel

__all__ = [
    'Point',
    'Bilinear',
    'Lanczos',
]


class Point(ComplexKernel):
    """Point resizer."""

    scale_function = resample_function = core.lazy.resize2.Point
    descale_function = core.lazy.descale.Depoint
    rescale_function = core.lazy.descale.Point
    _static_kernel_radius = 1


class Bilinear(ComplexKernel):
    """Bilinear resizer."""

    scale_function = resample_function = core.lazy.resize2.Bilinear
    descale_function = core.lazy.descale.Debilinear
    rescale_function = core.lazy.descale.Bilinear
    _static_kernel_radius = 1


class Lanczos(ComplexKernel):
    """Lanczos resizer."""

    scale_function = resample_function = core.lazy.resize2.Lanczos
    descale_function = core.lazy.descale.Delanczos
    rescale_function = core.lazy.descale.Lanczos

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

    @ComplexKernel.cached_property
    def kernel_radius(self) -> int:
        return ceil(self.taps)
