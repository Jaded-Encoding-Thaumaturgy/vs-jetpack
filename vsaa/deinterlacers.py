from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass, replace
from enum import IntFlag, auto
from fractions import Fraction
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Never, Protocol, Self, cast, runtime_checkable

from jetpytools import MISSING, CustomNotImplementedError, CustomStrEnum, CustomValueError, fallback, normalize_seq

from vsjetpack import TypeVar
from vskernels import (
    Bobber,
    Catrom,
    ComplexScaler,
    ComplexScalerLike,
    LeftShift,
    MixedScalerProcess,
    Point,
    Scaler,
    TopShift,
)
from vstools import ChromaLocation, FieldBased, FieldBasedLike, VSFunctionAllArgs, VSFunctionNoArgs, core, vs

__all__ = [
    "BWDIF",
    "EEDI3",
    "NNEDI3",
    "AntiAliaser",
    "Deinterlacer",
    "SangNom",
    "SuperSampler",
    "SuperSamplerProcess",
    "SupportsBobDeinterlace",
]


@runtime_checkable
class SupportsBobDeinterlace(Protocol):
    """
    Protocol for classes that support bob deinterlacing.
    """

    __slots__ = ()

    def deinterlace(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode: ...

    def bob(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode: ...


class DeinterlacerKwargs(UserDict[str, Any]):
    """
    A dict-like wrapper that syncs keys with a `Deinterlacer` instance.

    - If a key matches an attribute of `deinterlacer`, the value is set on
      the object instead of stored in the dict.
    - Otherwise, the pair is stored normally.
    """

    def __init__(self, deinterlacer: Deinterlacer) -> None:
        self.deinterlacer = deinterlacer
        super().__init__()

    def __setitem__(self, key: str, value: Any) -> None:
        if not hasattr(self.deinterlacer, key):
            return super().__setitem__(key, value)
        setattr(self.deinterlacer, key, value)


@dataclass(kw_only=True)
class Deinterlacer(Bobber, ABC):
    """
    Abstract base class for deinterlacing operations.
    """

    tff: bool | None = None
    """The field order."""

    double_rate: bool = True
    """Whether to double the FPS."""

    def __post_init__(self) -> None:
        # UserDict inherits from MutableMapping but they share the same methods
        self.kwargs = cast(dict[str, Any], DeinterlacerKwargs(self))

    def deinterlace(
        self,
        clip: vs.VideoNode,
        *,
        tff: FieldBasedLike | bool | None = None,
        double_rate: bool | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Apply deinterlacing to the given clip.

        Args:
            clip: The input clip.
            tff: Field order of the clip.
            double_rate: Whether to double the frame rate (True) or retain the original rate (False).
            **kwargs: Additional arguments passed to the plugin function.

        Returns:
            Deinterlaced clip.
        """
        field_based = FieldBased.from_param_or_video(fallback(tff, self.tff, default=None), clip, True, self.__class__)

        return self._interpolate(clip, field_based.is_tff, fallback(double_rate, self.double_rate), False, **kwargs)

    def bob(self, clip: vs.VideoNode, *, tff: FieldBasedLike | bool | None = None, **kwargs: Any) -> vs.VideoNode:
        """
        Apply bob deinterlacing to the given clip.

        Args:
            clip: The input clip.
            tff: Field order of the clip.
            **kwargs: Additional arguments passed to the plugin function.

        Returns:
            Deinterlaced clip.
        """
        field_based = FieldBased.from_param_or_video(fallback(tff, self.tff, default=None), clip, True, self.__class__)

        return self._interpolate(clip, field_based.is_tff, True, False, **kwargs)

    def copy(self, **kwargs: Any) -> Self:
        """
        Returns a new Antialiaser class replacing specified fields with new values
        """
        return replace(self, **kwargs)

    @abstractmethod
    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        """
        Retrieves arguments for deinterlacing processing.

        Args:
            **kwargs: Additional arguments.

        Returns:
            Passed keyword arguments.
        """
        return kwargs

    @property
    @abstractmethod
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        """
        Get the plugin function.
        """

    @abstractmethod
    def _interpolate(self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any) -> vs.VideoNode:
        """
        Performs deinterlacing if dh is False or doubling if dh is True.

        Subclasses should handle tff to field if needed and add the kwargs from `get_deint_args`

        Args:
            clip: The input clip.
            tff: The field order of the input clip.
            double_rate: Whether to double the FPS.
            dh: If True, doubles the height of the input by copying each line to every other line of the output, with
                the missing lines interpolated.

        Returns:
            Interpolated clip.
        """


@dataclass(kw_only=True)
class AntiAliaser(Deinterlacer, ABC):
    """
    Abstract base class for anti-aliasing operations.
    """

    transpose_first: bool = False
    """Transpose the clip before any operation."""

    class AADirection(IntFlag):
        """
        Enum representing the direction(s) in which anti-aliasing should be applied.
        """

        VERTICAL = auto()
        """
        Apply anti-aliasing in the vertical direction.
        """

        HORIZONTAL = auto()
        """
        Apply anti-aliasing in the horizontal direction.
        """

        BOTH = VERTICAL | HORIZONTAL
        """
        Apply anti-aliasing in both horizontal and vertical directions.
        """

    def antialias(self, clip: vs.VideoNode, direction: AADirection = AADirection.BOTH, **kwargs: Any) -> vs.VideoNode:
        """
        Apply anti-aliasing to the given clip.

        Args:
            clip: The input clip.
            direction: Direction in which to apply anti-aliasing. Defaults to AADirection.BOTH.
            **kwargs: Additional arguments passed to the plugin function.

        Returns:
            Anti-aliased clip.
        """
        tff = fallback(kwargs.pop("tff", self.tff), True)

        for y in sorted(self.AADirection, key=lambda x: x.value, reverse=self.transpose_first):
            if direction in (y, self.AADirection.BOTH):
                if y == self.AADirection.HORIZONTAL:
                    clip, tclips = self.transpose(clip, **kwargs)
                    kwargs |= tclips

                clip = self._interpolate(clip, tff, self.double_rate, False, **kwargs)

                if self.double_rate:
                    clip = core.std.Merge(clip[::2], clip[1::2])

                if y == self.AADirection.HORIZONTAL:
                    clip, tclips = self.transpose(clip, **kwargs)
                    kwargs |= tclips

        return clip

    def transpose(
        self,
        clip: vs.VideoNode,
        **kwargs: Any,
    ) -> tuple[vs.VideoNode, MutableMapping[str, vs.VideoNode | None]]:
        """
        Transpose the input clip by swapping its horizontal and vertical axes.

        Args:
            clip: The input clip.

        Returns:
            The transposed clip.
        """
        return clip.std.Transpose(), {}


@dataclass(kw_only=True)
class SuperSampler(Scaler, AntiAliaser, ABC):
    """
    Abstract base class for supersampling operations.
    """

    scaler: ComplexScalerLike = Catrom
    """Scaler used for downscaling and shifting after supersampling."""

    noshift: bool | Sequence[bool] = False
    """
    Disables sub-pixel shifting after supersampling.

    - `bool`: Applies to both luma and chroma.
    - `Sequence[bool]`: First for luma, second for chroma.
    """

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale the given clip using super sampling method.

        Note: Setting `tff=True` results in less chroma shift for non-centered chroma locations.

        Args:
            clip: The source clip.
            width: Target width (defaults to clip width if None).
            height: Target height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling.
            **kwargs: Additional arguments forwarded to the deinterlacing function.

        Returns:
            The scaled clip.
        """
        tff_fallback = fallback(kwargs.pop("tff", self.tff), True)

        dims = self._wh_norm(clip, width, height)
        dest_dimensions = list(dims)
        sy, sx = shift
        in_width, in_height = clip.width, clip.height

        cloc = list(ChromaLocation.from_video(clip).get_offsets(clip))
        subsampling = [2**clip.format.subsampling_w, 2**clip.format.subsampling_h]

        nshift: list[list[float]] = [
            normalize_seq(sx, clip.format.num_planes),
            normalize_seq(sy, clip.format.num_planes),
        ]

        if not self.transpose_first:
            dest_dimensions.reverse()
            cloc.reverse()
            subsampling.reverse()
            nshift.reverse()

        for x, dim in enumerate(dest_dimensions):
            is_width = (not x and self.transpose_first) or (not self.transpose_first and x)

            if is_width:
                clip, _ = self.transpose(clip)

            while clip.height < dim:
                delta = max(nshift[x], key=lambda y: abs(y))
                tff = False if delta < 0 else True if delta > 0 else tff_fallback
                offset = -0.25 if tff else 0.25

                for y in range(clip.format.num_planes):
                    if not y:
                        nshift[x][y] = (nshift[x][y] + offset) * 2
                    else:
                        nshift[x][y] = (nshift[x][y] + offset) * 2 - cloc[x] / subsampling[x]

                clip = self._interpolate(clip, tff, False, True, **kwargs)

            if is_width:
                clip, _ = self.transpose(clip)

        if not self.transpose_first:
            nshift.reverse()

        self._ss_shifts = nshift

        if self.noshift:
            noshift = normalize_seq(self.noshift, clip.format.num_planes)

            if all(noshift) and dims == (clip.width, clip.height):
                return clip

            for ns in nshift:
                for i in range(len(ns)):
                    ns[i] *= not noshift[i]

        sar_scale = Fraction(clip.height // in_height, clip.width // in_width)

        return ComplexScaler.ensure_obj(self.scaler, self.__class__).scale(
            clip, width, height, (nshift[1], nshift[0]), _sar_scale=sar_scale
        )

    if TYPE_CHECKING:

        def supersample(
            self, clip: vs.VideoNode, rfactor: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
        ) -> vs.VideoNode:
            """
            Supersample a clip by a given scaling factor.

            Note: Setting `tff=True` results in less chroma shift for non-centered chroma locations.

            Args:
                clip: The source clip.
                rfactor: Scaling factor for supersampling.
                shift: Subpixel shift (top, left) applied during scaling.
                **kwargs: Additional arguments forwarded to the scale function.

            Raises:
                CustomValueError: If resulting resolution is non-positive.

            Returns:
                The supersampled clip.
            """
            ...


@dataclass
class NNEDI3(SuperSampler):
    """
    Neural Network Edge Directed Interpolation (3rd gen.)

    More information:
        - https://github.com/sekrit-twc/znedi3
        - https://github.com/HolyWu/VapourSynth-nnedi3vk
    """

    nsize: int = 0
    """
    Size of the local neighbourhood around each pixel used by the predictor neural network.
    Possible settings:
        - 0: 8x6
        - 1: 16x6
        - 2: 32x6
        - 3: 48x6
        - 4: 8x4
        - 5: 16x4
        - 6: 32x4

    Wrapper default is 0, plugin default is 6.
    """

    nns: int = 4
    """
    Number of neurons in the predictor neural network. Possible values:
        - 0: 16
        - 1: 32
        - 2: 64
        - 3: 128
        - 4: 256

    Wrapper default is 4, plugin default is 1.
    """

    qual: int = 2
    """
    The number of different neural network predictions that are blended together to compute the final output value.
    Each neural network was trained on a different set of training data.
    Blending the results of these different networks improves generalisation to unseen data.
    Possible values are 1 and 2.

    Wrapper default is 2, plugin default is 1.
    """

    etype: int = 0
    """
    The set of weights used in the predictor neural network. Possible values:
    - 0: Weights trained to minimise absolute error.
    - 1: Weights trained to minimise squared error.
    """

    pscrn: int = 4
    """
    The prescreener used to decide which pixels should be processed by the predictor neural network,
    and which can be handled by simple cubic interpolation.
    Since most pixels can be handled by cubic interpolation, using the prescreener
    generally results in much faster processing. Possible values:
    - 0: No prescreening. No pixels will be processed with cubic interpolation. This is really slow.
    - 1: Old prescreener.
    - 2: New prescreener level 0.
    - 3: New prescreener level 1.
    - 4: New prescreener level 2.

    Wrapper default is 4, plugin default is 2.
    """

    gpu: bool = False
    """Enables the use of the Vulkan variant."""

    opencl: Never = cast(Never, MISSING)
    """Unused deprecated parameter."""

    def __post_init__(self) -> None:
        if self.opencl is not cast(Never, MISSING):
            warnings.warn("The 'opencl' argument has been removed and is deprecated.", RuntimeWarning)
        return super().__post_init__()

    @Scaler.cachedproperty
    def kernel_radius(self) -> int:
        match self.nsize:
            case 0 | 4:
                return 8
            case 1 | 5:
                return 16
            case 3:
                return 48
            case _:
                return 32

    def get_deint_args(self, *, clip: vs.VideoNode | None = None, **kwargs: Any) -> dict[str, Any]:
        return {
            "nsize": self.nsize,
            "nns": self.nns,
            "qual": self.qual,
            "etype": self.etype,
            "pscrn": self.pscrn,
        } | kwargs

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return core.lazy.nnedi3vk.NNEDI3 if self.gpu else core.lazy.znedi3.nnedi3

    def _interpolate(self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any) -> vs.VideoNode:
        field = tff + double_rate * 2

        return self._deinterlacer_function(clip, field, dh, **self.get_deint_args(clip=clip, **kwargs))


@dataclass
class EEDI3(SuperSampler):
    """
    Enhanced Edge Directed Interpolation (3rd gen.)
    """

    class Backend(CustomStrEnum):
        """Enum representing available backends on which to run the plugin."""

        CPU = "vszip"
        """Pure CPU implementation"""

        OPENCL = "vszipcl"
        """An OpenCL device"""

        VULKAN = "eedi3vk2"
        """A Vulkan device"""

        @property
        def supports_mclip(self) -> bool:
            """Whether the backend supports mclip."""
            return "mclip" in getattr(core, self.value).EEDI3.__signature__.parameters

        @property
        def supports_h(self) -> bool:
            """Whether the backend supports columns interpolating."""
            return hasattr(getattr(core, self.value), "EEDI3H")

        @property
        def should_h(self) -> bool:
            """Whether the backend should run columns interpolating."""
            return self.supports_h and self != EEDI3.Backend.CPU

        def EEDI3(  # noqa: N802
            self,
            clip: vs.VideoNode,
            field: int,
            *args: Any,
            sclip: vs.VideoNode | None = None,
            mclip: vs.VideoNode | None = None,
            **kwargs: Any,
        ) -> vs.VideoNode:
            """Applies the EEDI3 filter using the plugin associated with the selected backend."""
            if self.supports_mclip:
                kwargs.update(sclip=sclip, mclip=mclip)
            else:
                kwargs.update(sclip=sclip)

            return getattr(core, self.value).EEDI3(clip, field, *args, **kwargs)

        def EEDI3H(  # noqa: N802
            self,
            clip: vs.VideoNode,
            field: int,
            *args: Any,
            sclip: vs.VideoNode | None = None,
            mclip: vs.VideoNode | None = None,
            **kwargs: Any,
        ) -> vs.VideoNode:
            """Applies the EEDI3H filter using the plugin associated with the selected backend."""
            if self.should_h:
                attr = "EEDI3H"
                tclips = {"sclip": sclip, "mclip": mclip}
            else:
                attr = "EEDI3"
                clip, tclips = self.transpose(clip, sclip=sclip, mclip=mclip, **kwargs)

            if self.supports_mclip:
                kwargs.update(sclip=tclips["sclip"], mclip=tclips["mclip"])
            else:
                kwargs.update(sclip=tclips["sclip"])

            result = attrgetter(f"{self.value}.{attr}")(core)(clip, field, *args, **kwargs)
            return result if self.should_h else result.std.Transpose()

        @staticmethod
        def transpose(
            clip: vs.VideoNode,
            *,
            sclip: vs.VideoNode | None = None,
            mclip: vs.VideoNode | None = None,
            **kwargs: Any,
        ) -> tuple[vs.VideoNode, MutableMapping[str, vs.VideoNode | None]]:
            """Transpose the input clip by swapping its horizontal and vertical axes."""
            if isinstance(sclip, vs.VideoNode):
                sclip = sclip.std.Transpose()

            if isinstance(mclip, vs.VideoNode):
                mclip = mclip.std.Transpose()

            return clip.std.Transpose(), kwargs | {"sclip": sclip, "mclip": mclip}

    alpha: float = 0.2
    """
    Controls the weight given to connecting similar neighborhoods.
    It must be in the range [0, 1].
    A larger value for alpha will connect more lines and edges.
    Increasing alpha prioritizes connecting similar regions,
    which can reduce artifacts but may lead to excessive connections.
    """

    beta: float = 0.25
    """
    Controls the weight given to the vertical difference created by the interpolation.
    It must also be in the range [0, 1], and the sum of alpha and beta must not exceed 1.
    A larger value for beta will reduce the number of connected lines and edges,
    making the result less directed by edges.
    At a value of 1.0, there will be no edge-directed interpolation at all.
    """

    gamma: float = 20.0
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

    hp: bool = False
    """
    Determines the type of search steps:
    - When hp=True, it uses half-pel search steps.
    - When hp=False, it uses full-pel search steps.
    """

    ucubic: Never = cast(Never, MISSING)
    """Unused deprecated parameter."""

    cost3: Never = cast(Never, MISSING)
    """Unused deprecated parameter."""

    vcheck: int = 2
    """
    Defines the reliability check level for the resulting interpolation. The possible values are:
    - 0: No reliability check
    - 1: Weak reliability check
    - 2: Medium reliability check
    - 3: Strong reliability check
    """

    vthresh: tuple[float | None, float | None, float | None] | None = (32.0, 64.0, 4.0)
    """
    Sequence of three thresholds:
    - vthresh[0]: Used to calculate the reliability for the first difference.
    - vthresh[1]: Used for the second difference.
    - vthresh[2]: Controls the weighting of the interpolation direction.
    """

    sclip: vs.VideoNode | Deinterlacer | VSFunctionNoArgs | None = None
    """
    Provides additional control over the interpolation by using a reference clip.
    If set to None, vertical cubic interpolation is used as a fallback method instead.

    Passing a Deinterlacer object is only supported for pure deinterlacing.
    """

    mclip: vs.VideoNode | VSFunctionNoArgs | None = None
    """
    A mask used to apply edge-directed interpolation only to specified pixels.
    Pixels where the mask value is 0 will be interpolated using cubic linear
    or bicubic methods instead.
    The primary purpose of the mask is to reduce computational overhead
    by limiting edge-directed interpolation to certain pixels.

    Only the VULKAN and CPU backends supports it.
    """

    backend: Backend = Backend.CPU
    """
    Set the backend to use for processing.
    """

    @Scaler.cachedproperty
    def kernel_radius(self) -> int:
        return self.mdis

    def deinterlace(
        self,
        clip: vs.VideoNode,
        *,
        tff: FieldBasedLike | bool | None = None,
        double_rate: bool | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        kwargs = self.get_deint_args(**kwargs)

        sclip, mclip = kwargs.pop("sclip"), kwargs.pop("mclip")

        if isinstance(sclip, Deinterlacer):
            sclip = sclip.deinterlace(clip, tff=tff, double_rate=double_rate)

        if callable(sclip):
            sclip = sclip(clip)

        if callable(mclip):
            mclip = mclip(clip)

        return super().deinterlace(clip, tff=tff, double_rate=double_rate, sclip=sclip, mclip=mclip, **kwargs)

    def bob(self, clip: vs.VideoNode, *, tff: FieldBasedLike | bool | None = None, **kwargs: Any) -> vs.VideoNode:
        kwargs = self.get_deint_args(**kwargs)

        sclip, mclip = kwargs.pop("sclip"), kwargs.pop("mclip")

        if isinstance(sclip, Deinterlacer):
            sclip = sclip.bob(clip, tff=tff)

        if callable(sclip):
            sclip = sclip(clip)

        if callable(mclip):
            mclip = mclip(clip)

        return super().bob(clip, tff=tff, sclip=sclip, mclip=mclip, **kwargs)

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        vthresh = (None, None, None) if self.vthresh is None else self.vthresh

        kwargs = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "nrad": self.nrad,
            "mdis": self.mdis,
            "ucubic": self.ucubic,
            "cost3": self.cost3,
            "vcheck": self.vcheck,
            "vthresh0": vthresh[0],
            "vthresh1": vthresh[1],
            "vthresh2": vthresh[2],
            "sclip": self.sclip,
            "mclip": self.mclip,
        } | kwargs

        # TODO: Remove that
        for k, v in ((k, v) for k, v in kwargs.copy().items() if k in ["ucubic", "cost3"]):
            if v is not cast(Never, MISSING):
                warnings.warn(f"'{k}' argument has been removed and is deprecated.", RuntimeWarning)
            del kwargs[k]

        return kwargs

    def antialias(
        self,
        clip: vs.VideoNode,
        direction: AntiAliaser.AADirection = AntiAliaser.AADirection.BOTH,
        **kwargs: Any,
    ) -> vs.VideoNode:
        kwargs = self.get_deint_args(**kwargs)

        sclip, mclip = kwargs.pop("sclip", None), kwargs.pop("mclip", None)

        if isinstance(sclip, Deinterlacer):
            raise CustomValueError("sclip must be a callable or VideoNode", self.antialias)

        if sclip:
            if isinstance(sclip, VSFunctionNoArgs):
                sclip = sclip(clip)

            if self.double_rate:
                sclip = core.std.Interleave([sclip, sclip])

        if isinstance(mclip, VSFunctionNoArgs):
            mclip = mclip(clip)

        tff = fallback(kwargs.pop("tff", self.tff), True)

        for y in sorted(self.AADirection, key=lambda x: x.value, reverse=self.transpose_first):
            if direction in (y, self.AADirection.BOTH):
                horizontal = y == self.AADirection.HORIZONTAL
                clip = self._interpolate(
                    clip,
                    tff,
                    self.double_rate,
                    False,
                    horizontal=horizontal,
                    sclip=sclip,
                    mclip=mclip,
                    **kwargs,
                )

                if self.double_rate:
                    clip = core.std.Merge(clip[::2], clip[1::2])

        return clip

    def transpose(
        self,
        clip: vs.VideoNode,
        *,
        sclip: vs.VideoNode | None = None,
        mclip: vs.VideoNode | None = None,
        **kwargs: Any,
    ) -> tuple[vs.VideoNode, MutableMapping[str, vs.VideoNode | None]]:
        if isinstance(sclip, vs.VideoNode):
            sclip = sclip.std.Transpose()

        if isinstance(mclip, vs.VideoNode):
            mclip = mclip.std.Transpose()

        return clip.std.Transpose(), kwargs | {"sclip": sclip, "mclip": mclip}

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        kwargs = self.get_deint_args(**kwargs)

        if kwargs["sclip"] or kwargs["mclip"]:
            raise CustomNotImplementedError("sclip and mclip are currently not supported.", self.scale)

        return super().scale(clip, width, height, shift, **kwargs)

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return self.backend.EEDI3

    @property
    def _deinterlacer_function_h(self) -> VSFunctionAllArgs:
        return self.backend.EEDI3H

    def _interpolate(
        self,
        clip: vs.VideoNode,
        tff: bool,
        double_rate: bool,
        dh: bool,
        *,
        horizontal: bool = False,
        **kwargs: Any,
    ) -> vs.VideoNode:
        func = self._deinterlacer_function_h if horizontal else self._deinterlacer_function
        return func(clip, tff + double_rate * 2, dh, **self.get_deint_args(**kwargs))


@dataclass
class SangNom(SuperSampler):
    """
    SangNom single field deinterlacer using edge-directed interpolation
    """

    aa: int | Sequence[int] | None = None
    """
    The strength of luma anti-aliasing, applied to an 8-bit clip.
    Must be an integer between 0 and 128, inclusive.
    """

    _static_kernel_radius = 3

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return {"aa": self.aa} | kwargs

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return core.lazy.sangnom.SangNom

    def _interpolate(self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any) -> vs.VideoNode:
        if double_rate:
            order = 0
            clip = clip.std.SeparateFields(tff).std.DoubleWeave(tff)
        else:
            order = 1 if tff else 2

        return self._deinterlacer_function(clip, order, dh, **self.get_deint_args(**kwargs))


@dataclass
class BWDIF(Deinterlacer):
    """
    Motion adaptive deinterlacing based on yadif with the use of w3fdif and cubic interpolation algorithms.
    """

    edeint: vs.VideoNode | Deinterlacer | VSFunctionNoArgs | None = None
    """
    Allows the specification of an external clip from which to take spatial predictions
    instead of having Bwdif use cubic interpolation.

    This clip must be the same width, height, and colorspace as the input clip.

    If using same rate output, this clip should have the same number of frames as the input.
    If using double rate output, this clip should have twice as many frames as the input.
    """

    _static_kernel_radius = 2

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return {"edeint": self.edeint} | kwargs

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs:
        return core.lazy.bwdif.Bwdif

    def _interpolate(self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any) -> vs.VideoNode:
        field = tff + double_rate * 2

        kwargs = self.get_deint_args(**kwargs)

        edeint = kwargs.pop("edeint")

        if isinstance(edeint, Deinterlacer):
            kwargs["edeint"] = edeint._interpolate(clip, tff, double_rate, False)

        if callable(edeint):
            kwargs["edeint"] = edeint(clip)

        return self._deinterlacer_function(clip, field, **kwargs)


if TYPE_CHECKING:
    # Let's assume the specialized SuperSampler isn't abstract
    class _ConcreteSuperSampler(SuperSampler):
        def get_deint_args(self, **kwargs: Any) -> dict[str, Any]: ...
        @property
        def _deinterlacer_function(self) -> VSFunctionAllArgs: ...
        def _interpolate(
            self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
        ) -> vs.VideoNode: ...
else:
    _ConcreteSuperSampler = SuperSampler

_SuperSamplerWithNNEDI3DefaultT = TypeVar("_SuperSamplerWithNNEDI3DefaultT", bound=SuperSampler, default=NNEDI3)


class SuperSamplerProcess(MixedScalerProcess[_SuperSamplerWithNNEDI3DefaultT, Point], _ConcreteSuperSampler):
    """
    A utility SuperSampler class that applies a given function to a supersampled clip,
    then downsamples it back using Point.

    If used without a specified scaler, it defaults to inheriting from `NNEDI3`.
    """

    default_scaler = NNEDI3

    def __init__(self, *, function: VSFunctionNoArgs, noshift: bool | Sequence[bool] = True, **kwargs: Any) -> None:
        """
        Initialize the SuperSamplerProcess.

        Note:
            Chroma planes will not align properly during processing.
            Avoid using this class if accurate chroma placement relative to luma is required.

        Example:
        ```py
        processed = SuperSamplerProcess[NNEDI3](function=lambda clip: cool_function(clip, ...)).supersample(
            src, rfactor=2
        )
        ```

        Args:
            function: A function to apply on the supersampled clip.
            noshift: Disables sub-pixel shifting after supersampling.

                   - `bool`: Applies to both luma and chroma.
                   - `Sequence[bool]`: First for luma, second for chroma.

            **kwargs: Additional arguments to the specialized SuperSampler.
        """
        super().__init__(function=function, noshift=noshift, **kwargs)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> vs.VideoNode:
        ss_clip = super().scale(clip, width, height, shift, **kwargs)

        processed = self.function(ss_clip)

        return (
            self._others[0]
            .scale(
                processed,
                clip.width,
                clip.height,
                tuple([round(s - 1e-6) for s in dim_shifts] for dim_shifts in reversed(self._ss_shifts)),  # type: ignore[arg-type]
            )
            .std.CopyFrameProps(processed)
        )
