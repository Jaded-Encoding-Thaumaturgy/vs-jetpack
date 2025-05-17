"""
This module implements wrappers for the Enhanced Edge Directed Interpolation (3rd gen.)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

from jetpytools import fallback

from vstools import ConstantFormatVideoNode, CustomValueError, core, vs, vs_object

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, Interpolater
from . import nnedi3

__all__ = [
    "EEDI3",

    "Eedi3SS", "Eedi3DR", "Eedi3DR",

    'Eedi3',
]


@dataclass
class EEDI3(Interpolater):
    """Base class for EEDI3 interpolating methods."""

    alpha: float = 0.25
    """
    Controls the weight given to connecting similar neighborhoods.
    It must be in the range [0, 1]. 
    A larger value for alpha will connect more lines and edges.
    Increasing alpha prioritizes connecting similar regions,
    which can reduce artifacts but may lead to excessive connections.
    """

    beta: float = 0.5
    """
    Controls the weight given to the vertical difference created by the interpolation.
    It must also be in the range [0, 1], and the sum of alpha and beta must not exceed 1.
    A larger value for beta will reduce the number of connected lines and edges,
    making the result less directed by edges.
    At a value of 1.0, there will be no edge-directed interpolation at all. 
    """

    gamma: float = 40
    """
    Penalizes changes in interpolation direction.
    The larger the value of gamma, the smoother the interpolation field will be between two lines.
    The range for gamma is [0, ∞].
    Increasing gamma results in a smoother interpolation between lines but may reduce the sharpness of edges.

    If lines are not connecting properly, try increasing alpha and possibly decreasing beta/gamma.
    If unwanted artifacts occur, reduce alpha and consider increasing beta or gamma.
    """

    nrad: int = 2
    """
    Sets the radius used for computing neighborhood similarity. The valid range is [0, 3]. 
    A larger value for `nrad` will consider a wider neighborhood for similarity,
    which can improve edge connections but may also increase processing time.
    """

    mdis: int = 20
    """
    Sets the maximum connection radius. The valid range is [1, 40].
    For example, with `mdis=20`, when interpolating the pixel at (50, 10) (x, y),
    the farthest connections allowed would be between (30, 9)/(70, 11) and (70, 9)/(30, 11). 
    Larger values for `mdis` will allow connecting lines with smaller slopes,
    but this can also increase the chance of artifacts and slow down processing.
    """

    ucubic: bool = True
    """
    Determines the type of interpolation used.
    - When `ucubic=True`, cubic 4-point interpolation is applied.
    - When `ucubic=False`, 2-point linear interpolation is used.
    """

    cost3: bool = True
    """
    Defines the neighborhood cost function used to measure similarity.
    - When `cost3=True`, a 3-neighborhood cost function is used.
    - When `cost3=False`, a 1-neighborhood cost function is applied.
    """

    vcheck: int = 2
    """
    Defines the reliability check level for the resulting interpolation. The possible values are:
    - 0: No reliability check
    - 1: Weak reliability check
    - 2: Medium reliability check
    - 3: Strong reliability check
    """

    vthresh0: float = 32.0
    """Threshold used to calculate the reliability for the first difference."""

    vthresh1: float = 64.0
    """Threshold used for the second difference."""

    vthresh2: float = 4.0
    """Threshold used to control the weighting of the interpolation direction."""

    opt: int | None = None
    """
    Specifies the CPU optimizations to use during processing.
    The possible values are:

    - None = Auto-adjust based on whether `mclip` is used.
    - 0 = Auto-detect the optimal optimization based on the CPU.
    - 1 = Use standard C implementation.
    - 2 = Use SSE2.
    - 3 = Use SSE4.1.
    - 4 = Use AVX.
    - 5 = Use AVX512.
    """

    device: int = -1
    """
    Specifies the target OpenCL device.
    The default value (-1) triggers auto-detection of the available device.
    """

    opencl: bool = False
    """
    Enables the use of the OpenCL variant for processing.
    Note that in most cases, OpenCL may be slower than the CPU version.
    """

    mclip: vs.VideoNode | None = None
    """
    A mask used to apply edge-directed interpolation only to specified pixels. 
    Pixels where the mask value is 0 will be interpolated using cubic linear
    or bicubic methods instead. 
    The primary purpose of the mask is to reduce computational overhead
    by limiting edge-directed interpolation to certain pixels.
    """

    sclip_aa: type[Antialiaser] | Antialiaser | Literal[True] | vs.VideoNode | None = dc_field(
        default_factory=lambda: nnedi3.Nnedi3
    )
    """
    Provides additional control over the interpolation by using a reference clip.
    If set to None, vertical cubic interpolation is used as a fallback method instead.
    """

    # Class Variable
    _shift = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()

        self._sclip_aa: Antialiaser | Literal[True] | vs.VideoNode | None

        if (
            self.sclip_aa is not None
            and self.sclip_aa is not True
            and not isinstance(self.sclip_aa, (Antialiaser, vs.VideoNode))
        ):
            self._sclip_aa = self.sclip_aa()
        else:
            self._sclip_aa = self.sclip_aa

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        args = dict(
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            nrad=self.nrad, mdis=self.mdis,
            ucubic=self.ucubic, cost3=self.cost3,
            vcheck=self.vcheck,
            vthresh0=self.vthresh0, vthresh1=self.vthresh1, vthresh2=self.vthresh2
        )

        if self.opencl:
            args.update(device=self.device)
        elif self.mclip is not None or kwargs.get('mclip'):
            # opt=3 appears to always give reliable speed boosts if mclip is used.
            args.update(opt=fallback(kwargs.pop('opt', None), self.opt, 3))

        return args | kwargs

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        aa_kwargs = self.get_aa_args(clip, **kwargs)
        aa_kwargs = self._handle_sclip(clip, double_y, aa_kwargs, **kwargs)

        if self.opencl:
            interpolated = core.eedi3m.EEDI3CL(clip, self.field, double_y or not self.drop_fields, **aa_kwargs)
        else:
            interpolated = core.eedi3m.EEDI3(clip, self.field, double_y or not self.drop_fields, **aa_kwargs)

        return self.shift_interpolate(clip, interpolated, double_y)

    def _handle_sclip(
        self, clip: vs.VideoNode, double_y: bool, aa_kwargs: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        if not self._sclip_aa or ('sclip' in kwargs and kwargs['sclip']):
            return aa_kwargs

        if self._sclip_aa is True or isinstance(self._sclip_aa, vs.VideoNode):
            if double_y:
                if self._sclip_aa is True:
                    raise CustomValueError("You can't pass sclip_aa=True when supersampling!", self.__class__)

                if (clip.width, clip.height) != (self._sclip_aa.width, self._sclip_aa.height):
                    raise CustomValueError(
                        f'The dimensions of sclip_aa ({self._sclip_aa.width}x{self._sclip_aa.height}) '
                        f'don\'t match the expected dimensions ({clip.width}x{clip.height})!',
                        self.__class__
                    )

            aa_kwargs.update(sclip=clip)

            return aa_kwargs

        sclip_args = self._sclip_aa.get_aa_args(clip, mclip=kwargs.get('mclip'))
        sclip_args.update(self._sclip_aa.get_ss_args(clip) if double_y else self._sclip_aa.get_sr_args(clip))

        aa_kwargs.update(sclip=self._sclip_aa.interpolate(clip, double_y or not self.drop_fields, **sclip_args))

        return aa_kwargs

    def __del__(self) -> None:
        self.mclip = None


class Eedi3SS(EEDI3, SuperSampler):
    """Concrete implementation of EEDI3 used as a supersampler."""

    @SuperSampler.cached_property
    def kernel_radius(self) -> int:
        return self.nrad


class Eedi3SR(EEDI3, SingleRater, vs_object):
    """Concrete implementation of EEDI3 used as a single-rater."""

    _mclips: tuple[vs.VideoNode, vs.VideoNode] | None = None

    def get_sr_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        if not self.mclip:
            return {}

        if not self._mclips:
            self._mclips = (self.mclip, self.mclip.std.Transpose())

        if self.mclip.width == clip.width and self.mclip.height == clip.height:
            return dict(mclip=self._mclips[0]) | kwargs

        return dict(mclip=self._mclips[1]) | kwargs

    def __del__(self) -> None:
        self._mclips = None
        self.mclip = None

    def __vs_del__(self, core_id: int) -> None:
        self._mclips = None
        self.mclip = None


class Eedi3DR(Eedi3SR, DoubleRater):
    """Concrete implementation of EEDI3 used as a double-rater."""


class Eedi3(Eedi3DR, Eedi3SS, Antialiaser):
    """Full implementation of the EEDI3 anti-aliaser"""
