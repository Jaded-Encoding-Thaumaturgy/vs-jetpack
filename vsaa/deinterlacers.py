from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import IntFlag, auto
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from jetpytools import MISSING
from typing_extensions import Self

from vskernels import Catrom, ComplexScaler, ComplexScalerLike, LeftShift, Scaler, TopShift
from vstools import (
    ChromaLocation,
    ConstantFormatVideoNode,
    VideoNodeT,
    VSFunctionAllArgs,
    VSFunctionNoArgs,
    check_variable,
    core,
    normalize_seq,
    vs,
    vs_object,
)

__all__ = [
    "BWDIF",
    "EEDI2",
    "EEDI3",
    "NNEDI3",
    "AntiAliaser",
    "Deinterlacer",
    "SangNom",
    "SuperSampler",
    "SupportsBobDeinterlace",
]


@dataclass(kw_only=True)
class Deinterlacer(vs_object, ABC):
    """
    Abstract base class for deinterlacing operations.
    """

    tff: bool = False
    """The field order."""

    double_rate: bool = True
    """Whether to double the FPS."""

    @property
    @abstractmethod
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """
        Get the plugin function.
        """

    @abstractmethod
    def _interpolate(
        self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
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

    def deinterlace(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        """
        Apply deinterlacing to the given clip.

        Args:
            clip: The input clip.
            **kwargs: Additional arguments passed to the plugin function.

        Returns:
            Deinterlaced clip.
        """
        return self._interpolate(clip, self.tff, self.double_rate, False, **kwargs)

    def bob(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        """
        Apply bob deinterlacing to the given clip.

        Args:
            clip: The input clip.
            **kwargs: Additional arguments passed to the plugin function.

        Returns:
            Deinterlaced clip.
        """
        return self._interpolate(clip, self.tff, True, False, **kwargs)

    def copy(self, **kwargs: Any) -> Self:
        """
        Returns a new Antialiaser class replacing specified fields with new values
        """
        return replace(self, **kwargs)


@runtime_checkable
class SupportsBobDeinterlace(Protocol):
    """
    Protocol for classes that support bob deinterlacing.
    """

    __slots__ = ()

    def bob(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode: ...

    def deinterlace(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode: ...


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

    def antialias(
        self, clip: vs.VideoNode, direction: AADirection = AADirection.BOTH, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        """
        Apply anti-aliasing to the given clip.

        Args:
            clip: The input clip.
            direction: Direction in which to apply anti-aliasing. Defaults to AADirection.BOTH.
            **kwargs: Additional arguments passed to the plugin function.

        Returns:
            Anti-aliased clip.
        """
        assert check_variable(clip, self.antialias)

        for y in sorted((aa_dir for aa_dir in self.AADirection), key=lambda x: x.value, reverse=self.transpose_first):
            if direction in (y, self.AADirection.BOTH):
                if y == self.AADirection.HORIZONTAL:
                    clip = self.transpose(clip)

                clip = self.deinterlace(clip, **kwargs)

                if self.double_rate:
                    clip = core.std.Merge(clip[::2], clip[1::2])

                if y == self.AADirection.HORIZONTAL:
                    clip = self.transpose(clip)

        return clip

    def transpose(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        """
        Transpose the input clip by swapping its horizontal and vertical axes.

        Args:
            clip: The input clip.

        Returns:
            The transposed clip.
        """
        return clip.std.Transpose()


@dataclass(kw_only=True)
class SuperSampler(AntiAliaser, Scaler, ABC):
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
    ) -> ConstantFormatVideoNode:
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
        assert check_variable(clip, self.scale)

        dest_dimensions = list(self._wh_norm(clip, width, height))
        sy, sx = shift

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

            while (clip.width if is_width else clip.height) < dim:
                delta = max(nshift[x], key=lambda y: abs(y))
                tff = False if delta < 0 else True if delta > 0 else self.tff
                offset = -0.25 if tff else 0.25

                for y in range(clip.format.num_planes):
                    if not y:
                        nshift[x][y] = (nshift[x][y] + offset) * 2
                    else:
                        nshift[x][y] = (nshift[x][y] + offset) * 2 - cloc[x] / subsampling[x]

                if is_width:
                    clip = self.transpose(clip)

                clip = self._interpolate(clip, tff, False, True, **kwargs)

                if is_width:
                    clip = self.transpose(clip)

        if not self.transpose_first:
            nshift.reverse()

        if self.noshift:
            noshift = normalize_seq(self.noshift, clip.format.num_planes)

            for ns in nshift:
                for i in range(len(ns)):
                    ns[i] *= not noshift[i]

        return ComplexScaler.ensure_obj(self.scaler, self.__class__).scale(  # type: ignore[return-value]
            clip, width, height, (nshift[1], nshift[0])
        )

    if TYPE_CHECKING:

        def supersample(
            self, clip: VideoNodeT, rfactor: float = 2.0, shift: tuple[TopShift, LeftShift] = (0, 0), **kwargs: Any
        ) -> VideoNodeT:
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

    More informations: https://github.com/sekrit-twc/znedi3
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

    pscrn: int | None = 1
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

    The new prescreener is not available with float input.

    Wrapper default is 1, plugin default is 2 for integer input and 1 for float input.
    """

    opencl: bool = False
    """
    Enables the use of the OpenCL variant.
    """

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.sneedif.NNEDI3 if self.opencl else core.lazy.znedi3.nnedi3

    def _interpolate(
        self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        field = int(tff) + int(double_rate) * 2

        return self._deinterlacer_function(clip, field, dh, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "nsize": self.nsize,
            "nns": self.nns,
            "qual": self.qual,
            "etype": self.etype,
            "pscrn": self.pscrn,
        } | kwargs

    @Scaler.cached_property
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


@dataclass
class EEDI2(SuperSampler):
    """
    Enhanced Edge Directed Interpolation (2nd gen.)
    """

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

    map: int = 0
    """
    Allows one of three possible maps to be shown:
    - 0 = no map
    - 1 = edge map (Edge pixels will be set to 255 and non-edge pixels will be set to 0)
    - 2 = original scale direction map
    - 3 = 2x scale direction map
    """

    nt: int = 50
    """
    Defines the noise threshold between pixels in the sliding vectors.
    This value is used to determine initial starting conditions.
    Lower values typically reduce artifacts but may degrade edge reconstruction,
    while higher values can enhance edge reconstruction at the cost of introducing more artifacts.
    The valid range is from 0 to 256.
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
    """Enables the use of the CUDA variant for processing."""

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.eedi2cuda.EEDI2 if self.cuda else core.lazy.eedi2.EEDI2

    def _interpolate(
        self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        field = int(tff) + int(double_rate) * 2

        if not dh:
            clip = clip.std.SeparateFields(tff)

            if not double_rate:
                clip = clip[::2]

        return self._deinterlacer_function(clip, field, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "mthresh": self.mthresh,
            "lthresh": self.lthresh,
            "vthresh": self.vthresh,
            "estr": self.estr,
            "dstr": self.dstr,
            "maxd": self.maxd,
            "map": self.map,
            "nt": self.nt,
            "pp": self.pp,
        } | kwargs

    @Scaler.cached_property
    def kernel_radius(self) -> int:
        return self.maxd


@dataclass
class EEDI3(SuperSampler):
    """
    Enhanced Edge Directed Interpolation (3rd gen.)
    """

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

    vthresh: tuple[float | None, float | None, float | None] | None = (32.0, 64.0, 4.0)
    """
    Sequence of three thresholds:
    - vthresh[0]: Used to calculate the reliability for the first difference.
    - vthresh[1]: Used for the second difference.
    - vthresh[2]: Controls the weighting of the interpolation direction.
    """

    sclip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None
    """
    Provides additional control over the interpolation by using a reference clip.
    If set to None, vertical cubic interpolation is used as a fallback method instead.
    """

    mclip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None
    """
    A mask used to apply edge-directed interpolation only to specified pixels.
    Pixels where the mask value is 0 will be interpolated using cubic linear
    or bicubic methods instead.
    The primary purpose of the mask is to reduce computational overhead
    by limiting edge-directed interpolation to certain pixels.
    """

    opencl: bool = False
    """Enables the use of the OpenCL variant for processing."""

    def _set_sclip_mclip(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        sclip, mclip = kwargs.pop("sclip", MISSING), kwargs.pop("mclip", MISSING)

        if sclip is not MISSING:
            self.sclip = sclip

        if mclip is not MISSING:
            self.mclip = mclip

        return kwargs

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.eedi3m.EEDI3CL if self.opencl else core.lazy.eedi3m.EEDI3

    def _interpolate(
        self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        field = int(tff) + int(double_rate) * 2

        kwargs = self.get_deint_args(**kwargs)

        if callable(self.sclip):
            kwargs.update(sclip=self.sclip(clip))

        if callable(self.mclip):
            kwargs.update(mclip=self.mclip(clip))

        return self._deinterlacer_function(clip, field, dh, **kwargs)

    def antialias(
        self, clip: vs.VideoNode, direction: AntiAliaser.AADirection = AntiAliaser.AADirection.BOTH, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        kwargs = self._set_sclip_mclip(kwargs)

        if self.sclip and self.double_rate:
            if callable(self.sclip):
                self.sclip = self.sclip(clip)

            self.sclip = core.std.Interleave([self.sclip, self.sclip])

        return super().antialias(clip, direction, **kwargs)

    def transpose(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        if isinstance(self.sclip, vs.VideoNode):
            self.sclip = self.sclip.std.Transpose()

        if isinstance(self.mclip, vs.VideoNode):
            self.mclip = self.mclip.std.Transpose()

        return super().transpose(clip)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        return super().scale(clip, width, height, shift, **self._set_sclip_mclip(kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        if self.vthresh is None:
            self.vthresh = (None, None, None)

        eedi3_kwargs = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "nrad": self.nrad,
            "mdis": self.mdis,
            "ucubic": self.ucubic,
            "cost3": self.cost3,
            "vcheck": self.vcheck,
            "vthresh0": self.vthresh[0],
            "vthresh1": self.vthresh[1],
            "vthresh2": self.vthresh[2],
            "sclip": self.sclip,
        }

        if not self.opencl:
            eedi3_kwargs.update(mclip=self.mclip)

        return eedi3_kwargs | kwargs

    @Scaler.cached_property
    def kernel_radius(self) -> int:
        return self.mdis

    def __vs_del__(self, core_id: int) -> None:
        self.sclip = None
        self.mclip = None


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

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.sangnom.SangNom

    def _interpolate(
        self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        if double_rate:
            order = 0
            clip = clip.std.SeparateFields(tff).std.DoubleWeave(tff)
        else:
            order = 1 if tff else 2

        return self._deinterlacer_function(clip, order, dh, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return {"aa": self.aa} | kwargs

    _static_kernel_radius = 3


@dataclass
class BWDIF(Deinterlacer):
    """
    Motion adaptive deinterlacing based on yadif with the use of w3fdif and cubic interpolation algorithms.
    """

    edeint: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None
    """
    Allows the specification of an external clip from which to take spatial predictions
    instead of having Bwdif use cubic interpolation.
    This clip must be the same width, height, and colorspace as the input clip.
    If using same rate output, this clip should have the same number of frames as the input.
    If using double rate output, this clip should have twice as many frames as the input.
    """

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.bwdif.Bwdif

    def _interpolate(
        self, clip: vs.VideoNode, tff: bool, double_rate: bool, dh: bool, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        field = int(tff) + int(double_rate) * 2

        if callable(self.edeint):
            kwargs.update(edeint=self.edeint(clip))

        return self._deinterlacer_function(clip, field, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return {"edeint": self.edeint} | kwargs

    def __vs_del__(self, core_id: int) -> None:
        self.edeint = None
