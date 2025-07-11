from __future__ import annotations

from vstools import core

from .abstract import ZimgComplexKernel

__all__ = [
    "Spline16",
    "Spline36",
    "Spline64",
]


class Spline16(ZimgComplexKernel):
    """
    Spline16 resizer.
    """

    scale_function = resample_function = core.lazy.resize2.Spline16
    descale_function = core.lazy.descale.Despline16
    rescale_function = core.lazy.descale.Spline16
    _static_kernel_radius = 2


class Spline36(ZimgComplexKernel):
    """
    Spline36 resizer.
    """

    scale_function = resample_function = core.lazy.resize2.Spline36
    descale_function = core.lazy.descale.Despline36
    rescale_function = core.lazy.descale.Spline36
    _static_kernel_radius = 3


class Spline64(ZimgComplexKernel):
    """
    Spline64 resizer.
    """

    scale_function = resample_function = core.lazy.resize2.Spline64
    descale_function = core.lazy.descale.Despline64
    rescale_function = core.lazy.descale.Spline64
    _static_kernel_radius = 4
