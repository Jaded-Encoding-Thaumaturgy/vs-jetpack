from typing import Any, Union

from vstools import core, vs

from ...abstract import Bobber, ComplexKernel
from ...types import LeftShift, TopShift

__all__ = [
    "ZimgBobber",
    "ZimgComplexKernel",
    "ZimgComplexKernelLike",
]


class ZimgBobber(Bobber):
    """
    Abstract scaler class that applies bob deinterlacing using a zimg-based resizer.
    """

    bob_function = core.lazy.resize2.Bob
    """Bob function called internally when performing bobbing operations."""

    def get_bob_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate the keyword arguments used for bobbing.

        Args:
            clip: The source clip.
            shift: Subpixel shift (top, left).
            **kwargs: Extra parameters to merge.

        Returns:
            Final dictionary of keyword arguments for the bob function.
        """
        return (
            {"filter": self.__class__.__name__.lower(), "src_top": shift[0], "src_left": shift[1]}
            | self.kwargs
            | kwargs
        )


class ZimgComplexKernel(ComplexKernel, ZimgBobber):
    """
    Combined kernel class that supports complex scaling operations and zimg-based bob deinterlacing.

    This class integrates the full functionality of `ComplexKernel`—including scaling, descaling,
    resampling, and linear light/aspect ratio handling—with the bobbing capabilities of `ZimgBobber`.
    """


ZimgComplexKernelLike = Union[str, type[ZimgComplexKernel], ZimgComplexKernel]
"""
Type alias for anything that can resolve to a ZimgComplexKernel.

This includes:
- A string identifier.
- A class type subclassing `ZimgComplexKernel`.
- An instance of a `ZimgComplexKernel`.
"""
