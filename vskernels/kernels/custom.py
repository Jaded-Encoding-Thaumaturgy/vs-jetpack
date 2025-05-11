from __future__ import annotations

from math import ceil
from typing import Any

from jetpytools import CustomNotImplementedError, CustomValueError

from vstools import ConstantFormatVideoNode, core, vs

from .abstract import Kernel

__all__ = [
    'CustomKernel'
]


class CustomKernel(Kernel):
    """
    An abstract base class for defining custom kernel-based scaling and descaling operations.

    This class allows users to implement their own kernel function by overriding the
    `kernel()` method. It provides flexible support for parameters like `blur` and `taps`,
    enabling dynamic modification of the kernel's behavior at runtime.

    Subclasses must implement the `kernel()` method to specify the mathematical shape of the kernel.
    """

    def kernel(self, *, x: float) -> float:
        """
        Define the kernel function at a given position.

        This method must be implemented by subclasses to provide the actual kernel logic.

        :param x:                           The input position.
        :return:                            The evaluated kernel value at position `x`.
        :raises CustomNotImplementedError:  If not overridden by subclass.
        """
        raise CustomNotImplementedError

    def scale_function(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None, *args: Any, **kwargs: Any
    ) -> vs.VideoNode:
        return core.resize2.Custom(
            clip, self.kernel, ceil(kwargs.pop('taps', self.kernel_radius)), width, height, *args, **kwargs
        )

    def resample_function(
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return self.scale_function(clip, width, height, *args, **kwargs)  # type: ignore[return-value]

    def descale_function(
        self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        try:
            return core.descale.Decustom(
                clip, width, height, self.kernel, ceil(kwargs.pop('taps', self.kernel_radius)), *args, **kwargs
            )
        except vs.Error as e:
            if 'Output dimension must be' in str(e):
                raise CustomValueError(
                    f'Output dimension ({width}x{height}) must be less than or equal to '
                    f'input dimension ({clip.width}x{clip.height}).', self.__class__
                )

            raise CustomValueError(e, self.__class__)

    def rescale_function(
        self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return core.descale.ScaleCustom(
            clip, width, height, self.kernel, ceil(kwargs.pop('taps', self.kernel_radius)), *args, **kwargs
        )
