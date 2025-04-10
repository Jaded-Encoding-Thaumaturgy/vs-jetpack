"""
This module implements wrappers for the Enhanced Edge Directed Interpolation (2nd gen.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jetpytools import CustomRuntimeError

from vstools import ConstantFormatVideoNode, check_variable_format, core, vs

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, Interpolater, _FullInterpolate

__all__ = [
    "EEDI2",

    "Eedi2SS", "Eedi2DR", "Eedi2DR",

    'Eedi2',
]


@dataclass
class EEDI2(_FullInterpolate, Interpolater):
    """Base class for EEDI2 interpolating methods."""

    mthresh: int = 10
    """
    Controls the edge magnitude threshold used in edge detection for building the initial edge map.
    Its range is from 0 to 255, with lower values detecting weaker edges.
    """

    lthresh: int = 20
    """
    Controls the Laplacian threshold used in edge detection. 
    Its range is from 0 to 510, with lower values detecting weaker lines.
    """

    vthresh: int = 20
    """
    Controls the variance threshold used in edge detection. 
    Its range is from 0 to a large number, with lower values detecting weaker edges.
    """

    estr: int = 2
    """
    Defines the required number of edge pixels (<=) in a 3x3 area, in which the center pixel
    has been detected as an edge pixel, for the center pixel to be removed from the edge map.
    """

    dstr: int = 4
    """
    Defines the required number of edge pixels (>=) in a 3x3 area, in which the center pixel
    has not been detected as an edge pixel, for the center pixel to be added to the edge map.
    """

    maxd: int = 24
    """
    Sets the maximum pixel search distance for determining the interpolation direction.
    Larger values allow the algorithm to connect edges and lines with smaller slopes but may introduce artifacts.
    In some cases, using a smaller `maxd` value can yield better results than a larger one.
    The maximum possible value for `maxd` is 29.
    """

    pp: int = 1
    """
    Enables two optional post-processing modes designed to reduce artifacts by identifying problem areas
    and applying simple vertical linear interpolation in those areas.
    While these modes can improve results, they may slow down processing and slightly reduce edge sharpness.
    - 0 = No post-processing
    - 1 = Check for spatial consistency of final interpolation directions
    - 2 = Check for junctions and corners
    - 3 = Apply both checks from 1 and 2

    Only `pp=0` and `pp=1` is implemented for the CUDA variant.
    """

    cuda: bool = False
    """
    Enables the use of the CUDA variant for processing.
    Note that full interpolating is only supported by CUDA.
    """

    num_streams: int = 1
    """
    Specifies the number of CUDA streams.
    """

    device_id: int = -1
    """
    Specifies the target CUDA device.
    The default value (-1) triggers auto-detection of the available device.
    """

    # Class Variable
    _shift = -0.5

    def is_full_interpolate_enabled(self, x: bool, y: bool) -> bool:
        return self.cuda and x and y

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        if (pp := self.pp) > 1 and self.cuda:
            from warnings import warn
            warn(
                f"{self.__class__.__name__}: Only `pp=0` and `pp=1` is implemented for the CUDA variant. "
                "Falling back to `pp=1`...",
                Warning
            )
            pp = 1

        args = dict(
            mthresh=self.mthresh,
            lthresh=self.lthresh,
            vthresh=self.vthresh,
            estr=self.estr,
            dstr=self.dstr,
            maxd=self.maxd,
            pp=pp
        )

        if self.cuda:
            args.update(num_streams=self.num_streams, device_id=self.device_id)

        return args | kwargs

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        assert check_variable_format(clip, self.__class__)

        kwargs = self.get_aa_args(clip) | kwargs

        if self.cuda:
            inter = core.eedi2cuda.EEDI2(clip, self.field, **kwargs)
        else:
            inter = core.eedi2.EEDI2(clip, self.field, **kwargs)

        if not double_y:
            if self.drop_fields:
                inter = inter.std.SeparateFields(not self.field)[::2]

                inter = self._shifter.shift(inter, (0.5 - 0.75 * self.field, 0))
            else:
                inter = self._scaler.scale(  # type: ignore[assignment]
                    inter, clip.width, clip.height, (self._shift * int(not self.field), 0)
                )

        return self._post_interpolate(clip, inter, double_y)  # pyright: ignore[reportArgumentType]

    def full_interpolate(
        self, clip: vs.VideoNode, double_y: bool, double_x: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        if not all([double_y, double_x]):
            raise CustomRuntimeError(
                "`double_y` and `double_x` should be set to True to use full_interpolate!",
                self.full_interpolate,
                (double_y, double_x)
            )

        return core.eedi2cuda.Enlarge2(clip, **self.get_aa_args(clip) | kwargs)


class Eedi2SS(EEDI2, SuperSampler):
    """Concrete implementation of EEDI2 used as a supersampler."""

    _static_kernel_radius = 2


class Eedi2SR(EEDI2, SingleRater):
    """Concrete implementation of EEDI2 used as a single-rater."""


class Eedi2DR(Eedi2SR, DoubleRater):
    """Concrete implementation of EEDI2 used as a double-rater."""


class Eedi2(Eedi2DR, Eedi2SS, Antialiaser):
    """Full implementation of the EEDI2 anti-aliaser"""
