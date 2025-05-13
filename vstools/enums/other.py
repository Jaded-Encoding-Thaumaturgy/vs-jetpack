from __future__ import annotations

from fractions import Fraction
from typing import Callable, Iterator, Literal, NamedTuple

import vapoursynth as vs

from jetpytools import Coordinate, CustomIntEnum, CustomStrEnum, Position, Sentinel, SentinelT, Size
from typing_extensions import Self

from ..types import ConstantFormatVideoNode, HoldsPropValueT, VideoNodeT

__all__ = [
    'Direction',
    'Dar', 'Sar',
    'Region',
    'Resolution',
    'Coordinate',
    'Position',
    'Size',
    'SceneChangeMode'
]


class Direction(CustomIntEnum):
    """Enum to simplify the direction argument."""

    HORIZONTAL = 0
    VERTICAL = 1

    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5

    @property
    def is_axis(self) -> bool:
        """Whether the Direction represents an axis (horizontal/vertical)."""

        return self <= self.VERTICAL

    @property
    def is_way(self) -> bool:
        """Whether the Direction is one of the 4 arrow directions."""

        return self > self.VERTICAL

    @property
    def string(self) -> str:
        """A string representation of the Direction."""

        return self._name_.lower()


class _Xar(Fraction):
    @classmethod
    def from_param(cls, value: Self | bool | float | None, fallback: Self | float) -> Self | None:
        """
        Get the Xar from a Xar, a boolean, a float or a None object.

        :param value:       Value identifier.
        :param fallback:    Fallback value.
        :return:            Xar object or None.
        """
        if value is False:
            return cls(fallback)

        if value is True:
            return None

        if isinstance(value, cls):
            return value

        if value is not None:
            return cls.from_float(float(value))

        return None


class Dar(_Xar):
    """
    A Fraction representing the Display Aspect Ratio.

    This represents the dimensions of the physical display used to view the image.
    For more information, see <https://en.wikipedia.org/wiki/Display_aspect_ratio>.
    """

    @classmethod
    def from_res(cls, width: int, height: int, sar: Sar | None = None) -> Self:
        """
        Get the DAR from the specified dimensions and SAR.

        :param width:      The width of the image.
        :param height:     The height of the image.
        :param sar:        The SAR object. Optional.

        :return:           A DAR object created using the specified dimensions and SAR.
        """

        dar = Fraction(width, height)

        if sar:
            if sar.denominator > sar.numerator:
                dar /= sar
            else:
                dar *= sar

        return cls(dar)

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, sar: bool = True) -> Self:
        """
        Get the DAR from the specified clip and SAR.

        :param clip:     Clip or frame that holds the frame properties.
        :param sar:      Whether to use SAR metadata.

        :return:         A DAR object created using the specified clip and SAR.
        """

        return cls.from_res(clip.width, clip.height, Sar.from_clip(clip) if sar else None)

    def to_sar(self, active_area: int | Fraction, height: int) -> Sar:
        """
        Convert the DAR to a SAR object.

        :param active_area:     The active image area. For more information, see ``Sar.from_ar``.
        :param height:          The height of the image.

        :return:                A SAR object created using the DAR.
        """

        assert isinstance(active_area, int | Fraction)

        return Sar.from_ar(active_area, height, self)


class Sar(_Xar):
    """
    A Fraction representing the Sample Aspect Ratio.

    This represents the aspect ratio of the pixels or samples of an image.
    It may also be known as the Pixel Aspect Ratio in certain scenarios.
    For more information, see <https://en.wikipedia.org/wiki/Pixel_aspect_ratio>.
    """

    @classmethod
    def from_clip(cls, clip: HoldsPropValueT) -> Self:
        """
        Get the SAR from the clip's frame properties.

        :param clip:        Clip or frame that holds the frame properties.

        :return:            A SAR object of the SAR properties from the given clip.
        """

        from ..utils import get_prop

        return cls(get_prop(clip, '_SARNum', int, None, 1), get_prop(clip, '_SARDen', int, None, 1))

    @classmethod
    def from_ar(cls, active_area: int | Fraction, height: int, dar: Dar) -> Self:
        """
        Calculate the SAR using a DAR object & active area. See ``Dar.to_sar`` for more information.

        For a list of known standards, refer to the following tables:
        `<https://docs.google.com/spreadsheets/d/1pzVHFusLCI7kys2GzK9BTk3w7G8zcLxgHs3DMsurF7g>`_

        :param active_area:     The active image area.
        :param height:          The height of the image.
        :param dar:             The DAR object.

        :return:                A SAR object created using DAR and active image area information.
        """

        assert isinstance(active_area, int | Fraction)

        return cls(dar / (Fraction(active_area) / height))

    def apply(self, clip: VideoNodeT) -> VideoNodeT:
        """Apply the SAR values as _SARNum and _SARDen frame properties to a clip."""

        return vs.core.std.SetFrameProps(clip, _SARNum=self.numerator, _SARDen=self.denominator)


class Region(CustomStrEnum):
    """StrEnum signifying an analog television region."""

    UNKNOWN = 'unknown'
    """Unknown region."""

    NTSC = 'NTSC'
    """
    The first American standard for analog television broadcast was developed by
    National Television System Committee (NTSC) in 1941.

    For more information see `this <https://en.wikipedia.org/wiki/NTSC>`_.
    """

    NTSCi = 'NTSCi'
    """Interlaced NTSC."""

    PAL = 'PAL'
    """
    Phase Alternating Line (PAL) colour encoding system.

    For more information see `this <https://en.wikipedia.org/wiki/PAL>`_.
    """

    PALi = 'PALi'
    """Interlaced PAL."""

    FILM = 'FILM'
    """True 24fps content."""

    NTSC_FILM = 'NTSC (FILM)'
    """NTSC 23.976fps content."""

    @property
    def framerate(self) -> Fraction:
        """Obtain the Region's framerate."""

        return _region_framerate_map[self]

    @classmethod
    def from_framerate(cls, framerate: float | Fraction, strict: bool = False) -> Self:
        """Determine the Region using a given framerate."""

        key = Fraction(framerate)

        if strict:
            return cls(_framerate_region_map[key])

        if key not in _framerate_region_map:
            diffs = [(k, abs(float(key) - float(v))) for k, v in _region_framerate_map.items()]

            diffs.sort(key=lambda x: x[1])

            return cls(diffs[0][0])

        return cls(_framerate_region_map[key])


_region_framerate_map = {
    Region.UNKNOWN: Fraction(0),
    Region.NTSC: Fraction(30000, 1001),
    Region.NTSCi: Fraction(60000, 1001),
    Region.PAL: Fraction(25, 1),
    Region.PALi: Fraction(50, 1),
    Region.FILM: Fraction(24, 1),
    Region.NTSC_FILM: Fraction(24000, 1001),
}

_framerate_region_map = {r.framerate: r for r in Region}


class Resolution(NamedTuple):
    """Tuple representing a resolution."""

    width: int
    height: int

    @classmethod
    def from_video(cls, clip: vs.VideoNode) -> Self:
        """Create a Resolution object using a given clip's dimensions."""

        from ..functions import check_variable_resolution

        assert check_variable_resolution(clip, cls.from_video)

        return cls(clip.width, clip.height)

    def transpose(self) -> Self:
        """Flip the Resolution matrix over its diagonal."""

        return self.__class__(self.height, self.width)

    def __str__(self) -> str:
        return f'{self.width}x{self.height}'


class SceneChangeMode(CustomIntEnum):
    """Enum for various scene change modes."""

    WWXD = 1
    """Get the scene changes using the vapoursynth-wwxd plugin <https://github.com/dubhater/vapoursynth-wwxd>."""

    SCXVID = 2
    """Get the scene changes using the vapoursynth-scxvid plugin <https://github.com/dubhater/vapoursynth-scxvid>."""

    WWXD_SCXVID_UNION = 3  # WWXD | SCXVID
    """Get every scene change detected by both wwxd or scxvid."""

    WWXD_SCXVID_INTERSECTION = 0  # WWXD & SCXVID
    """Only get the scene changes if both wwxd and scxvid mark a frame as being a scene change."""

    @property
    def is_WWXD(self) -> bool:
        """Check whether a mode that uses wwxd is used."""

        return self in (
            SceneChangeMode.WWXD, SceneChangeMode.WWXD_SCXVID_UNION,
            SceneChangeMode.WWXD_SCXVID_INTERSECTION
        )

    @property
    def is_SCXVID(self) -> bool:
        """Check whether a mode that uses scxvid is used."""

        return self in (
            SceneChangeMode.SCXVID, SceneChangeMode.WWXD_SCXVID_UNION,
            SceneChangeMode.WWXD_SCXVID_INTERSECTION
        )

    def ensure_presence(self, clip: vs.VideoNode, akarin: bool | None = None) -> ConstantFormatVideoNode:
        """Ensures all the frame properties necessary for scene change detection are created."""

        from ..exceptions import CustomRuntimeError
        from ..functions import check_variable_format

        assert check_variable_format(clip, self.ensure_presence)

        stats_clip = list[ConstantFormatVideoNode]()

        if self.is_SCXVID:
            if not hasattr(vs.core, 'scxvid'):
                raise CustomRuntimeError(
                    'You are missing scxvid!\n\tDownload it from https://github.com/dubhater/vapoursynth-scxvid',
                    self.ensure_presence
                )
            stats_clip.append(clip.scxvid.Scxvid())

        if self.is_WWXD:
            if not hasattr(vs.core, 'wwxd'):
                raise CustomRuntimeError(
                    'You are missing wwxd!\n\tDownload it from https://github.com/dubhater/vapoursynth-wwxd',
                    self.ensure_presence
                )
            stats_clip.append(clip.wwxd.WWXD())

        return self._prepare_akarin(clip, stats_clip, akarin)

    @property
    def prop_keys(self) -> Iterator[str]:
        if self.is_WWXD:
            yield 'Scenechange'

        if self.is_SCXVID:
            yield '_SceneChangePrev'

    def check_cb(self, akarin: bool | None = None) -> Callable[[vs.VideoFrame], bool]:
        if akarin is None:
            akarin = hasattr(vs.core, 'akarin')

        if akarin:
            return (lambda f: bool(f[0][0, 0]))

        keys = set(self.prop_keys)
        prop_key = next(iter(keys))

        if self is SceneChangeMode.WWXD_SCXVID_UNION:
            return (lambda f: any(f.props[key] == 1 for key in keys))

        if self is SceneChangeMode.WWXD_SCXVID_INTERSECTION:
            return (lambda f: all(f.props[key] == 1 for key in keys))

        return (lambda f: f.props[prop_key] == 1)

    def lambda_cb(self, akarin: bool | None = None) -> Callable[[int, vs.VideoFrame], SentinelT | int]:
        callback = self.check_cb(akarin)
        return (lambda n, f: Sentinel.check(n, callback(f)))

    def prepare_clip(self, clip: vs.VideoNode, height: int | Literal[False] = 360, akarin: bool | None = None) -> ConstantFormatVideoNode:
        """
        Prepare a clip for scene change metric calculations.

        The clip will always be resampled to YUV420 8bit if it's not already,
        as that's what the plugins support.

        :param clip:        Clip to process.
        :param height:      Output height of the clip. Smaller frame sizes are faster to process,
                            but may miss more scene changes or introduce more false positives.
                            Width is automatically calculated. `False` means no resizing operation is performed.
                            Default: 360.
        :param akarin:      Use the akarin plugin for speed optimizations. `None` means it will check if its available,
                            and if it is, use it. Default: None.

        :return:            A prepared clip for performing scene change metric calculations on.
        """
        from ..utils import get_w

        if height:
            clip = clip.resize.Bilinear(get_w(height, clip), height, vs.YUV420P8)
        elif not clip.format or (clip.format and clip.format.id != vs.YUV420P8):
            clip = clip.resize.Bilinear(format=vs.YUV420P8)

        return self.ensure_presence(clip, akarin)

    def _prepare_akarin(
        self, clip: ConstantFormatVideoNode, stats_clip: list[ConstantFormatVideoNode], akarin: bool | None = None
    ) -> ConstantFormatVideoNode:
        from ..utils import merge_clip_props

        if akarin is None:
            akarin = hasattr(vs.core, 'akarin')

        if akarin:

            keys = list(self.prop_keys)

            expr = ' '.join([f'x.{k}' for k in keys]) + (' and' * (len(keys) - 1))

            blank = clip.std.BlankClip(1, 1, vs.GRAY8, keep=True)

            if len(stats_clip) > 1:
                return merge_clip_props(blank, *stats_clip).akarin.Expr(expr)

            return blank.std.CopyFrameProps(stats_clip[0]).akarin.Expr(expr)

        if len(stats_clip) > 1:
            return merge_clip_props(clip, *stats_clip)

        return stats_clip[0]
