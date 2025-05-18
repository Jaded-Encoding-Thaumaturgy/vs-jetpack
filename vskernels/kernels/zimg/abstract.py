from typing import Any, Callable, ClassVar, Union

from vstools import FieldBased, FieldBasedT, core, vs

from ...abstract import ComplexKernel
from ...abstract.base import BaseScaler
from ...types import LeftShift, TopShift


__all__ = [
    "ZimgBobber",
    "ZimgComplexKernel",
    "ZimgComplexKernelLike",
]


class ZimgBobber(BaseScaler):
    """
    Abstract scaler class that applies bob deinterlacing using a zimg-based resizer.
    """

    bob_function: Callable[..., vs.VideoNode] = core.lazy.resize.Bob
    """Bob function called internally when performing bobbing operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("bob",)

    def bob(
        self, clip: vs.VideoNode, tff: FieldBasedT | bool | None = None, double_rate: bool = True, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Apply bob deinterlacing to a given clip using the selected resizer.

        :param clip:        The source clip
        :param tff:         Field order of the clip.
        :param double_rate: Wether to double the frame rate (True) of retain the original rate (False).
        :return:            The bobbed clip.
        """
        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        bobbed = self.bob_function(clip, **self.get_bob_args(clip, tff=clip_fieldbased.is_tff, **kwargs))

        if not double_rate:
            return bobbed[::2]

        return bobbed

    def get_bob_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate the keyword arguments used for bobbing.

        :param clip:    The source clip.
        :param shift:   Subpixel shift (top, left).
        :param kwargs:  Extra parameters to merge.
        :return:        Final dictionary of keyword arguments for the bob function.
        """
        return dict(
            filter=self.__class__.__name__.lower(), src_top=shift[0], src_left=shift[1]
        ) | self.kwargs | kwargs


class ZimgComplexKernel(ComplexKernel, ZimgBobber):
    """
    Combined kernel class that supports complex scaling operations and zimg-based bob deinterlacing.

    This class integrates the full functionality of `ComplexKernel`—including scaling, descaling,
    resampling, and linear light/aspect ratio handling—with the bobbing capabilities of `ZimgBobber`.
    """

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("scale", "descale", "rescale", "resample", "shift", "bob")


ZimgComplexKernelLike = Union[str, type[ZimgComplexKernel], ZimgComplexKernel]
"""
Type alias for anything that can resolve to a ZimgComplexKernel.

This includes:
- A string identifier.
- A class type subclassing `ZimgComplexKernel`.
- An instance of a `ZimgComplexKernel`.
"""
