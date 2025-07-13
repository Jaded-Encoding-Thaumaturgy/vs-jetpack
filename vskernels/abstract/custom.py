"""
This module defines the abstract classes for scaling, descaling and resampling operations
based on a custom kernel.
"""

from __future__ import annotations

from abc import abstractmethod
from math import ceil
from typing import Any, Union

from jetpytools import CustomError, CustomValueError

from vstools import ConstantFormatVideoNode, core, vs

from .base import Kernel
from .complex import ComplexKernel

__all__ = [
    "CustomComplexKernel",
    "CustomComplexKernelLike",
    "CustomComplexTapsKernel",
    "CustomKernel",
]


class CustomKernel(Kernel):
    """
    Abstract base class for defining custom kernel-based scaling and descaling operations.

    This class allows users to implement their own kernel function by overriding the
    `kernel()` method.

    Subclasses must implement the `kernel()` method to specify the mathematical shape of the kernel.
    """

    @abstractmethod
    def kernel(self, *, x: float) -> float:
        """
        Define the kernel function at a given position.

        This method must be implemented by subclasses to provide the actual kernel logic.

        Args:
            x: The input position.

        Returns:
            The evaluated kernel value at position `x`.
        """

    def scale_function(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None, *args: Any, **kwargs: Any
    ) -> vs.VideoNode:
        try:
            return core.resize2.Custom(
                clip, self.kernel, ceil(kwargs.pop("taps", self.kernel_radius)), width, height, *args, **kwargs
            )
        except vs.Error as e:
            raise CustomError(e, self.__class__) from e

    def resample_function(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return self.scale_function(clip, width, height, *args, **kwargs)  # type: ignore[return-value]

    def descale_function(
        self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        try:
            return core.descale.Decustom(
                clip, width, height, self.kernel, ceil(kwargs.pop("taps", self.kernel_radius)), *args, **kwargs
            )
        except vs.Error as e:
            if "Output dimension must be" in str(e):
                raise CustomValueError(
                    f"Output dimension ({width}x{height}) must be less than or equal to "
                    f"input dimension ({clip.width}x{clip.height}).",
                    self.__class__,
                )

            raise CustomError(e, self.__class__) from e

    def rescale_function(
        self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        try:
            return core.descale.ScaleCustom(
                clip, width, height, self.kernel, ceil(kwargs.pop("taps", self.kernel_radius)), *args, **kwargs
            )
        except vs.Error as e:
            raise CustomError(e, self.__class__) from e


class CustomComplexKernel(CustomKernel, ComplexKernel):
    """
    Abstract kernel class that combines custom kernel behavior with advanced scaling and descaling capabilities.

    This class extends both `CustomKernel` and `ComplexKernel`, enabling the definition
    of custom mathematical kernels with the advanced rescaling logic provided by
    linear and aspect-ratio-aware components.
    """


CustomComplexKernelLike = Union[str, type[CustomComplexKernel], CustomComplexKernel]
"""
Type alias for anything that can resolve to a CustomComplexKernel.

This includes:
- A string identifier.
- A class type subclassing `CustomComplexKernel`.
- An instance of a `CustomComplexKernel`.
"""


class CustomComplexTapsKernel(CustomComplexKernel):
    """
    Extension of `CustomComplexKernel` that introduces configurable kernel taps.
    """

    def __init__(self, taps: float, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific number of taps and optional keyword arguments.

        These keyword arguments are automatically forwarded to the `_implemented_funcs` methods
        but only if the method explicitly accepts them as named parameters.
        If the same keyword is passed to both `__init__` and one of the `_implemented_funcs`,
        the one passed to `func` takes precedence.

        Args:
            taps: Determines the radius of the kernel.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        self.taps = taps
        super().__init__(**kwargs)

    @Kernel.cached_property
    def kernel_radius(self) -> int:
        """
        Compute the effective kernel radius based on the number of taps.

        Returns:
            Radius as the ceiling of `taps`.
        """
        return ceil(self.taps)

    def _pretty_string(self, **attrs: Any) -> str:
        return super()._pretty_string(**{"taps": self.taps} | attrs)
