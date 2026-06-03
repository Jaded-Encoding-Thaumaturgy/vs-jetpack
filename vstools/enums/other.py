from __future__ import annotations

from fractions import Fraction
from typing import Literal, Self

from ..types import HoldsPropValue
from ..vs_proxy import vs

__all__ = ["Dar", "Sar"]


class Dar(Fraction):
    """
    A Fraction representing the Display Aspect Ratio.

    This represents the dimensions of the physical display used to view the image.
    For more information, see <https://en.wikipedia.org/wiki/Display_aspect_ratio>.
    """

    @classmethod
    def from_res(cls, width: int, height: int, sar: Fraction | Literal[False] = False) -> Self:
        """
        Get the DAR from the specified dimensions and SAR.

        Args:
            width: The width of the image.
            height: The height of the image.
            sar: The SAR object. Optional.

        Returns:
            A DAR object created using the specified dimensions and SAR.
        """

        dar = Fraction(width, height)

        if sar is not False:
            dar /= sar

        return cls(dar)

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, sar: bool = True) -> Self:
        """
        Get the DAR from the specified clip and SAR.

        Args:
            clip: Clip or frame that holds the frame properties.
            sar: Whether to use SAR metadata.

        Returns:
            A DAR object created using the specified clip and SAR.
        """

        return cls.from_res(clip.width, clip.height, Sar.from_clip(clip) if sar else sar)

    def to_sar(self, active_area: int | Fraction, height: int) -> Sar:
        """
        Convert the DAR to a SAR object.

        Args:
            active_area: The active image area. For more information, see ``Sar.from_ar``.
            height: The height of the image.

        Returns:
            A SAR object created using the DAR.
        """

        return Sar.from_ar(active_area, height, self)


class Sar(Fraction):
    """
    A Fraction representing the Sample Aspect Ratio.

    This represents the aspect ratio of the pixels or samples of an image.
    It may also be known as the Pixel Aspect Ratio in certain scenarios.
    For more information, see <https://en.wikipedia.org/wiki/Pixel_aspect_ratio>.
    """

    @classmethod
    def from_clip(cls, clip: HoldsPropValue) -> Self:
        """
        Get the SAR from the clip's frame properties.

        Args:
            clip: Clip or frame that holds the frame properties.

        Returns:
            A SAR object of the SAR properties from the given clip.
        """

        from ..utils import get_prop

        return cls(get_prop(clip, "_SARNum", int, default=1), get_prop(clip, "_SARDen", int, default=1))

    @classmethod
    def from_ar(cls, active_area: int | Fraction, height: int, dar: Fraction) -> Self:
        """
        Calculate the SAR using a DAR object & active area. See ``Dar.to_sar`` for more information.

        For a list of known standards, refer to the following tables:
        `<https://docs.google.com/spreadsheets/d/1pzVHFusLCI7kys2GzK9BTk3w7G8zcLxgHs3DMsurF7g>`_

        Args:
            active_area: The active image area.
            height: The height of the image.
            dar: The DAR object.

        Returns:
            A SAR object created using DAR and active image area information.
        """

        return cls(dar / (Fraction(active_area) / height))

    def apply(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Apply the SAR values as _SARNum and _SARDen frame properties to a clip.
        """

        return vs.core.std.SetFrameProps(clip, _SARNum=self.numerator, _SARDen=self.denominator)
