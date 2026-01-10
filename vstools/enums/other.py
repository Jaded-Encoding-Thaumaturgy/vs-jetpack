from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from typing import Literal, Self

from jetpytools import CustomIntEnum, CustomRuntimeError, Sentinel, SentinelT

from ..types import HoldsPropValue
from ..vs_proxy import vs

__all__ = ["Dar", "Sar", "SceneChangeMode"]


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


class SceneChangeMode(CustomIntEnum):
    """
    Enum for various scene change modes.
    """

    WWXD = 1
    """
    Get the scene changes using the vapoursynth-wwxd plugin <https://github.com/dubhater/vapoursynth-wwxd>.
    """

    SCXVID = 2
    """
    Get the scene changes using the vapoursynth-scxvid plugin <https://github.com/dubhater/vapoursynth-scxvid>.
    """

    WWXD_SCXVID_UNION = 3  # WWXD | SCXVID
    """
    Get every scene change detected by both wwxd or scxvid.
    """

    WWXD_SCXVID_INTERSECTION = 0  # WWXD & SCXVID
    """
    Only get the scene changes if both wwxd and scxvid mark a frame as being a scene change.
    """

    @property
    def is_WWXD(self) -> bool:  # noqa: N802
        """
        Check whether a mode that uses wwxd is used.
        """
        return self in (
            SceneChangeMode.WWXD,
            SceneChangeMode.WWXD_SCXVID_UNION,
            SceneChangeMode.WWXD_SCXVID_INTERSECTION,
        )

    @property
    def is_SCXVID(self) -> bool:  # noqa: N802
        """
        Check whether a mode that uses scxvid is used.
        """
        return self in (
            SceneChangeMode.SCXVID,
            SceneChangeMode.WWXD_SCXVID_UNION,
            SceneChangeMode.WWXD_SCXVID_INTERSECTION,
        )

    def ensure_presence(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Ensures all the frame properties necessary for scene change detection are created.
        """
        from ..utils import merge_clip_props

        stats_clip = list[vs.VideoNode]()

        if self.is_SCXVID:
            if not hasattr(vs.core, "scxvid"):
                raise CustomRuntimeError(
                    "You are missing scxvid!\n\tDownload it from https://github.com/dubhater/vapoursynth-scxvid",
                    self.ensure_presence,
                )
            stats_clip.append(clip.scxvid.Scxvid())

        if self.is_WWXD:
            if not hasattr(vs.core, "wwxd"):
                raise CustomRuntimeError(
                    "You are missing wwxd!\n\tDownload it from https://github.com/dubhater/vapoursynth-wwxd",
                    self.ensure_presence,
                )
            stats_clip.append(clip.wwxd.WWXD())

        keys = tuple(self.prop_keys)

        expr = " ".join([f"x.{k}" for k in keys]) + (" and" * (len(keys) - 1))

        blank = clip.std.BlankClip(1, 1, vs.GRAY8, keep=True)

        if len(stats_clip) > 1:
            return merge_clip_props(blank, *stats_clip).akarin.Expr(expr)

        return blank.std.CopyFrameProps(stats_clip[0]).akarin.Expr(expr)

    @property
    def prop_keys(self) -> Iterator[str]:
        if self.is_WWXD:
            yield "Scenechange"

        if self.is_SCXVID:
            yield "_SceneChangePrev"

    def lambda_cb(self) -> Callable[[int, vs.VideoFrame], SentinelT | int]:
        return lambda n, f: Sentinel.check(n, bool(f[0][0, 0]))

    def prepare_clip(self, clip: vs.VideoNode, height: int | Literal[False] = 360) -> vs.VideoNode:
        """
        Prepare a clip for scene change metric calculations.

        The clip will always be resampled to YUV420 8bit if it's not already,
        as that's what the plugins support.

        Args:
            clip: Clip to process.
            height: Output height of the clip. Smaller frame sizes are faster to process, but may miss more scene
                changes or introduce more false positives. Width is automatically calculated. `False` means no resizing
                operation is performed. Default: 360.

        Returns:
            A prepared clip for performing scene change metric calculations on.
        """
        from ..utils import get_w

        if height:
            clip = clip.resize.Bilinear(get_w(height, clip), height, vs.YUV420P8)
        elif not clip.format or (clip.format and clip.format.id != vs.YUV420P8):
            clip = clip.resize.Bilinear(format=vs.YUV420P8)

        return self.ensure_presence(clip)
