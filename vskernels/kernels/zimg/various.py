from __future__ import annotations

from math import ceil
from typing import Any, Callable

from vstools import ConstantFormatVideoNode, core, vs

from ...types import LeftShift, TopShift
from .abstract import ZimgComplexKernel

__all__ = [
    "Bilinear",
    "Lanczos",
    "Point",
]


class Point(ZimgComplexKernel):
    """
    Point resizer.
    """

    scale_function: Callable[..., vs.VideoNode] = core.lazy.resize2.Point
    resample_function: Callable[..., ConstantFormatVideoNode] = core.lazy.resize2.Point
    descale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Depoint
    rescale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Point
    _static_kernel_radius = 1


class Bilinear(ZimgComplexKernel):
    """
    Bilinear resizer.
    """

    scale_function: Callable[..., vs.VideoNode] = core.lazy.resize2.Bilinear
    resample_function: Callable[..., ConstantFormatVideoNode] = core.lazy.resize2.Bilinear
    descale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Debilinear
    rescale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Bilinear
    _static_kernel_radius = 1


class Lanczos(ZimgComplexKernel):
    """
    Lanczos resizer.
    """

    scale_function: Callable[..., vs.VideoNode] = core.lazy.resize2.Lanczos
    resample_function: Callable[..., ConstantFormatVideoNode] = core.lazy.resize2.Lanczos
    descale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Delanczos
    rescale_function: Callable[..., ConstantFormatVideoNode] = core.lazy.descale.Lanczos

    def __init__(self, taps: float = 3, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific number of taps.

        Args:
            taps: Determines the radius of the kernel.
            **kwargs: Additional keyword arguments passed to the superclass.
        """
        self.taps = taps
        super().__init__(**kwargs)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)
        if is_descale:
            return args | {"taps": self.kernel_radius}
        return args | {"filter_param_a": self.kernel_radius}

    def get_bob_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().get_bob_args(clip, shift, filter_param_a=self.kernel_radius, **kwargs)

    @ZimgComplexKernel.cached_property
    def kernel_radius(self) -> int:
        return ceil(self.taps)

    def _pretty_string(self, **attrs: Any) -> str:
        return super()._pretty_string(**{"taps": self.taps} | attrs)
