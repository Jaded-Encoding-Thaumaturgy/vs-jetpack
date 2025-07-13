from typing import Any, Callable, ClassVar, Union

from vstools import ConstantFormatVideoNode, FieldBased, FieldBasedT, check_variable, core, vs

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

    bob_function: Callable[..., ConstantFormatVideoNode] = core.lazy.resize2.Bob
    """Bob function called internally when performing bobbing operations."""

    _implemented_funcs: ClassVar[tuple[str, ...]] = ("bob", "deinterlace")

    def bob(
        self, clip: vs.VideoNode, *, tff: FieldBasedT | bool | None = None, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Apply bob deinterlacing to a given clip using the selected resizer.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip
            tff: Field order of the clip.

        Returns:
            The bobbed clip.
        """
        clip_fieldbased = FieldBased.from_param_or_video(tff, clip, True, self.__class__)

        assert check_variable(clip, self.__class__)

        return self.bob_function(clip, **self.get_bob_args(clip, tff=clip_fieldbased.is_tff, **kwargs))

    def deinterlace(
        self, clip: vs.VideoNode, *, tff: FieldBasedT | bool | None = None, double_rate: bool = True, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Apply deinterlacing to a given clip using the selected resizer.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            clip: The source clip
            tff: Field order of the clip.
            double_rate: Wether to double the frame rate (True) of retain the original rate (False).

        Returns:
            The bobbed clip.
        """
        bobbed = self.bob(clip, tff=tff, **kwargs)

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

    _implemented_funcs: ClassVar[tuple[str, ...]] = (
        "scale",
        "descale",
        "rescale",
        "resample",
        "shift",
        "bob",
        "deinterlace",
    )


ZimgComplexKernelLike = Union[str, type[ZimgComplexKernel], ZimgComplexKernel]
"""
Type alias for anything that can resolve to a ZimgComplexKernel.

This includes:
- A string identifier.
- A class type subclassing `ZimgComplexKernel`.
- An instance of a `ZimgComplexKernel`.
"""
