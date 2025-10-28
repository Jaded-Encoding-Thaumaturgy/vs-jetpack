"""
This module provides utilities for chroma reconstruction using local linear regression.

Most of this is based on the work of `doop`,
originally written for [Irozuku Sekai no Ashita kara](https://myanimelist.net/anime/37497/Irozuku_Sekai_no_Ashita_kara).
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from math import sqrt
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Sequence

from jetpytools import CustomRuntimeError, CustomValueError, FuncExcept, MismatchRefError, cachedproperty, clamp

from vsexprtools import norm_expr
from vskernels import LeftShift, TopShift
from vstools import (
    ChromaLocation,
    ConstantFormatVideoNode,
    ConvMode,
    FormatsRefClipMismatchError,
    ResolutionsRefClipMismatchError,
    UnsupportedSampleTypeError,
    UnsupportedVideoFormatError,
    VSObject,
    VSObjectABC,
    core,
    vs,
)

from .blur import box_blur
from .freqs import MeanMode

__all__ = ["ChromaReconstruct", "reconstruct", "regression"]

type Width = int
"""
Type alias for width in pixels.
"""

type Height = int
"""
Type alias for height in pixels.
"""


class MeanFunction(Protocol):
    """
    Protocol for mean-filtering functions.
    """

    def __call__(self, clip: vs.VideoNode, /, radius: int) -> vs.VideoNode: ...


class _Shift(NamedTuple):
    top: float
    left: float


class RegressClips(NamedTuple):
    """
    Results of per-pixel linear regression between one clip (x) and another (y).
    """

    slope: vs.VideoNode
    """Regression slope."""

    intercept: vs.VideoNode | None
    """Regression intercept."""

    correlation: vs.VideoNode
    """Correlation coefficient."""


def t_based_weight(n: int, alpha: float) -> float:
    """
    Compute a reliability weight based on the t-distribution critical correlation.

    The weight decreases with larger critical correlation thresholds (harder to reach significance)
    and increases with sample size.

    Args:
        n: Sample size (must be > 3 for a meaningful result).
        alpha: Significance level for the two-tailed test.

    Returns:
        A weight in [0.0, 1.0], where larger values indicate higher reliability.
    """
    from scipy import stats

    if n <= 3:
        return 0.0

    df = n - 2
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    rcrit = tcrit / sqrt(tcrit**2 + df)

    # Smaller critical r (easier to detect significance) = higher reliability.
    # Map rcrit in [0,1] to weight in [0,1]: weight = 1 - rcrit
    return clamp(1.0 - float(rcrit), 0.0, 1.0)


def fisher_weight(n: int, alpha: float) -> float:
    """
    Compute a reliability weight using Fisher's z-transformation.

    The weight is inversely related to the confidence interval width in z-space:
    smaller confidence intervals indicate more reliable correlations.

    Args:
        n: Sample size (must be > 3 for a meaningful result).
        alpha: Significance level for the two-tailed confidence interval.

    Returns:
        A weight in [0.0, 1.0], where larger values indicate higher reliability.
    """
    from scipy import stats

    if n <= 3:
        return 0.0

    se = 1.0 / sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    width = 2 * zcrit * se  # total CI width in z-space

    # Map width -> weight: smaller width = higher weight
    # 1 / (1 + width) ensures it stays in (0,1] smoothly
    return clamp(1.0 / (1.0 + float(width)), 0.0, 1.0)


def get_weight(radius: int, alpha: float, method: str) -> float:
    """
    Select a reliability weight computation method.

    Args:
        radius: Neighborhood radius used for regression. Determines sample size.
        alpha: Significance/confidence level for the statistical method.
        method: Weighting method, either "threshold" (t-based) or "fisher".

    Returns:
        A weight in [0.0, 1.0].

    Raises:
        CustomValueError: If `method` is not recognized.
    """
    n = (2 * radius + 1) ** 2  # window size

    if method == "threshold":
        return t_based_weight(n, alpha)

    if method == "fisher":
        return fisher_weight(n, alpha)

    raise CustomValueError("Unknown weight_method", get_weight, method)


def regression(
    x: vs.VideoNode,
    *ys: vs.VideoNode,
    radius: int = 1,
    mean: MeanMode | MeanFunction = lambda c, radius: box_blur(c, radius, mode=ConvMode.HV),
    alpha: float = 0.20,
    weight_method: Literal["threshold", "fisher"] = "threshold",
    intercept: float = False,
    eps: float = 1e-7,
    func: FuncExcept | None = None,
) -> list[RegressClips]:
    """
    Perform per-pixel linear regression between a reference clip `x` and one or more target clips `ys`,
    using local neighborhoods for mean/variance estimates.

    More information:
        - [Simple_linear_regression](https://en.wikipedia.org/wiki/Simple_linear_regression)

    Each regression fits the model:

        y ≈ x * slope + intercept

    on a per-pixel basis, with slope, intercept (optional),
    and correlation computed over neighborhoods of size `(2 * radius + 1)^2`.

    Args:
        x: Reference clip (independent variable).
        *ys: One or more clips to regress on `x` (dependent variables).
        radius: Neighborhood radius for mean/variance estimates. Must be > 0.
        mean: Function or mode used to compute local means. Defaults to box blur.
        alpha: Significance/confidence level for reliability weighting. Defaults to 0.20.
        weight_method: Method for computing reliability weight, "threshold" or "fisher".
        intercept: If True, compute regression intercepts. If False, intercepts are None.
        eps: Small constant to avoid division by zero. Defaults to 1e-7.
        func: An optional function to use for error handling.

    Raises:
        CustomValueError: If `radius` or `alpha` are out of bounds, or if an unknown `weight_method` is provided.

    Returns:
        A list of `RegressClips`, one for each input clip in `ys`, containing slope, intercept (or None),
        and correlation.
    """
    func = func or regression

    UnsupportedSampleTypeError.check([x, *ys], vs.FLOAT, func, "All input clips must have a FLOAT sample type.")

    if not radius > 0:
        raise CustomValueError('"radius" must be greater than zero.', func, radius)

    if not 0 < alpha <= 1.0:
        raise CustomValueError('"alpha" must be greater than zero and lower than 1.0 (inclusive).', func, alpha)

    if isinstance(mean, MeanMode):
        mean = mean.single

    mean_x, *mean_ys = (mean(c, radius) for c in (x, *ys))
    mean_xx, *mean_yys = (mean(norm_expr(c, "src0 dup *", func=func), radius) for c in (x, *ys))
    mean_xys = [mean(norm_expr([x, y], "src0 src1 *", func=func), radius) for y in ys]

    var_x, *var_ys = (
        norm_expr([mean_zz, mean_z], "src0 src1 dup * - 0 max", func=func)
        for mean_zz, mean_z in zip([mean_xx, *mean_yys], [mean_x, *mean_ys])
    )

    cov_xys = [norm_expr([xy, mean_x, y], "src0 src1 src2 * -", func=func) for xy, y in zip(mean_xys, mean_ys)]

    slopes = [norm_expr([cov, var_x], f"src0 src1 {eps} + /", func=func) for cov in cov_xys]

    if intercept:
        intercepts = [
            norm_expr([y, slope, mean_x], f"src0 src1 src2 * - {float(intercept)} /", func=func)
            for y, slope in zip(mean_ys, slopes)
        ]
    else:
        intercepts = [None] * len(slopes)

    w = get_weight(radius, alpha, weight_method)

    corrs = [
        norm_expr(
            [cov_xy, var_x, var_y],
            [f"src0 dup * src1 src2 * {eps} + / sqrt", f"{1 - w} - {w} / 0 max" if w != 1.0 else ""],
            func=func,
        ).std.SetFrameProp("RegressWeight", floatval=w)
        for cov_xy, var_y in zip(cov_xys, var_ys)
    ]

    return [RegressClips(s, i, c) for s, i, c in zip(slopes, intercepts, corrs)]


def reconstruct(
    clip: vs.VideoNode,
    r: RegressClips,
    radius: int = 1,
    mean: MeanMode | MeanFunction = lambda c, radius: box_blur(c, radius, mode=ConvMode.HV),
    eps: float = 1e-7,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Reconstruct a predicted clip from regression results.

    Uses the regression slope, intercept (if available), and correlation to reconstruct
    the dependent variable from the input clip.

    Args:
        clip: Input clip (independent variable).
        r: Regression results (slope, intercept, correlation).
        radius: Neighborhood radius for smoothing regression coefficients. Defaults to 1.
        mean: Function or mode used to compute local means. Defaults to box blur.
        eps: Small constant to avoid division by zero. Defaults to 1e-7.
        func: Optional function reference for error handling context.

    Returns:
        A new clip representing the reconstructed dependent variable.
    """
    func = func or reconstruct

    UnsupportedSampleTypeError.check(
        [clip, *filter(None, r)], vs.FLOAT, func, "All input clips must have a FLOAT sample type."
    )

    if isinstance(mean, MeanMode):
        mean = mean.single

    corr_sum = mean(r.correlation, radius)

    slope_pm = norm_expr([r.slope, r.correlation], "x y *", func=func)
    slope_pm_sum = mean(slope_pm, radius)

    if r.intercept:
        intercept_pm = norm_expr([r.intercept, r.correlation], "x y *", func=func)
        intercept_pm_sum = mean(intercept_pm, radius)
    else:
        intercept_pm_sum = clip.std.BlankClip(keep=True, color=[0] * clip.format.num_planes)

    return norm_expr([clip, slope_pm_sum, intercept_pm_sum, corr_sum], f"x y * z + a {eps} + /", func=func)


class SubsampledShift(VSObject):
    """
    Utility class for handling chroma subsampling shifts.
    """

    def __init__(self, chroma_location: ChromaLocation, fmt: vs.VideoFormat) -> None:
        """
        Initializes the class.

        Args:
            chroma_location: Chroma siting/positioning information.
            fmt: Video format of the clip.
        """
        self._chroma_location = chroma_location
        self._fmt = fmt

        left, top = self._chroma_location.get_offsets(fmt)
        self._off_top = top
        self._off_left = left

    def to_full(self) -> _Shift:
        """
        Convert subsampled offsets to full-resolution (4:4:4) shift.

        Assumes the source is subsampled (e.g. 4:2:0 or 4:2:2).

        Returns:
            Vertical and horizontal shift scaled to full resolution.
        """
        return _Shift(self._off_top * -1 / 2**self._fmt.subsampling_h, self._off_left * -1 / 2**self._fmt.subsampling_w)

    def to_subsampled(
        self, src_dim: tuple[Width, Height] | vs.VideoNode, dst_dim: tuple[Width, Height] | vs.VideoNode
    ) -> _Shift:
        """
        Convert full-resolution offsets to subsampled coordinates.

        Assumes the source is 4:4:4.

        Args:
            src_dim: Source dimensions or clip (treated as 4:4:4).
            dst_dim: Destination dimensions or clip (subsampled).

        Returns:
            Vertical and horizontal shift in subsampled coordinates.
        """
        if isinstance(src_dim, vs.VideoNode):
            src_dim = (src_dim.width, src_dim.height)
        if isinstance(dst_dim, vs.VideoNode):
            dst_dim = (dst_dim.width, dst_dim.height)

        return _Shift(
            self._off_top * src_dim[1] / (dst_dim[1] * 2**self._fmt.subsampling_h),
            self._off_left * src_dim[0] / (dst_dim[0] * 2**self._fmt.subsampling_w),
        )

    @property
    def offsets(self) -> tuple[TopShift, LeftShift]:
        """The raw vertical (top) and horizontal (left) chroma offsets."""
        return (self._off_top, self._off_left)


@dataclass(init=False, repr=False, eq=False, match_args=False)
class ChromaReconstruct(VSObjectABC):
    """
    Abstract base class for chroma reconstruction via regression.

    Provides an interface for reconstructing chroma planes from luma using linear regression,
    with support for chroma demangling/mangling depending on chroma siting.

    Subclasses must at least implement:
        - A `mangle_luma` method
        - Either `demangle_luma` and `demangle_chroma`, or a single `demangle` method handling all planes.

    Example:
        Consider the show [*Sirius the Jaeger*](https://myanimelist.net/anime/37569/Sirius), produced by P.A. Works,
        known for chroma mangling in some titles.
        Several steps are required to reconstruct the chroma planes:

           - Choose the appropriate luma base plane (usually descaled to the production resolution).
           - Determine how the chroma planes were mangled during production.

        For *Sirius*, originally produced at 720p, the chroma planes underwent:

           - Produced at 1280x720 4:4:4.
           - Downscaled to 640x720 4:2:2 with point resize.
           - Resized to 960x540 with point resize (width) and neutral downscaling (height, e.g. Catrom).

        A custom subclass may then be defined:
        ```py
        from vskernels import BorderHandling, Catrom, Lanczos
        from vstools import depth, split


        class SiriusReconstruct(ChromaReconstruct):
            catrom = Catrom()

            def mangle_luma(self, luma_base: vs.VideoNode) -> vs.VideoNode:
                # Input: descaled luma plane (1280x720).
                # Align as if it were subsampled, matching the chroma path.
                return core.resize.Point(luma_base, 640, 720, src_left=-1).resize.Point(960, 720, src_left=-0.25)

            def demangle(self, clip: vs.VideoNode) -> vs.VideoNode:
                # Input: luma 960x720 or chroma 960x540.
                # Undo the width point-resize.
                clip = core.resize.Point(clip, 640, clip.height)
                # Rescale to 1280x540. Since only vertical downscaling is needed,
                # this avoids blurring chroma by upscaling vertically.
                return self.catrom.scale(clip, 1280, 540, self.ss_shift.to_full())

            def base_for_reconstruct(self, luma_base: vs.VideoNode, demangled_luma: vs.VideoNode) -> vs.VideoNode:
                # `demangle` outputs 1280x540, so downscale luma_base accordingly.
                # Better to downscale luma here than to upscale chroma vertically.
                return self.catrom.scale(luma_base, height=540)


        clip = depth(clip, 32, vs.FLOAT)

        y, u, v = split(clip)
        descaled_y = Lanczos(3).descale(y, 1280, 720, border_handling=BorderHandling.ZERO)

        recon = SiriusReconstruct(descaled_y, [u, v], clip)
        recon.regression(radius=4)
        chroma_recon = recon.reconstruct(radius=4)
        ```

        The `reconstruct` method outputs reconstructed chroma planes at the same resolution returned by `demangle`.
        From here, you may:

           - Downscale to obtain a 4:2:0 clip (recommended).
           - Upscale to produce a 1080p 4:4:4 clip.
           - Combine with the descaled luma for a 720p 4:4:4 clip (requires upscaling chroma vertically).

        Example: downscale reconstructed chroma to 4:2:0:

        ```py
        from vskernels import Hermite
        from vsscale import ArtCNN
        from vstools import join

        rescaled_y = ArtCNN.R8F64(scaler=Hermite(linear=True)).scale(descaled_y, 1920, 1080)
        merged = join(
            [
                rescaled_y,
                *(Hermite().scale(c, 960, 540, recon.ss_shift.to_subsampled(c, (960, 540))) for c in chroma_recon),
            ]
        )
        ```
    """

    luma_base: vs.VideoNode
    """The luma base plane."""

    chroma_bases: Sequence[vs.VideoNode]
    """The chroma base planes."""

    clip_src: ConstantFormatVideoNode
    """The original source clip."""

    chroma_loc: ChromaLocation = field(init=False)
    """Chroma location derived from the source clip."""

    def __init__(self, luma_base: vs.VideoNode, chroma_bases: Sequence[vs.VideoNode], clip_src: vs.VideoNode) -> None:
        """
        Initialize chroma reconstruction.

        Args:
            luma_base: Luma base plane (float GRAY).
            chroma_bases: Chroma base planes (float GRAY).
            clip_src: Source clip from which to derive chroma location.

        Raises:
            UnsupportedVideoFormatError: If any base clip is not GRAY float format.
        """
        UnsupportedVideoFormatError.check(
            [luma_base, *chroma_bases], vs.GRAYS, self.__class__, "All base clips must be of GRAYS format."
        )

        self.luma_base = luma_base
        self.chroma_bases = chroma_bases
        self.clip_src = clip_src
        self.chroma_loc = ChromaLocation.from_video(clip_src, True, self.__class__)

    def __post_init__(self) -> None:
        ChromaReconstruct.__init__(self, self.luma_base, self.chroma_bases, self.clip_src)

    def __init_subclass__(cls) -> None:
        if hasattr(cls, "demangle_luma") ^ hasattr(cls, "demangle_chroma"):
            raise CustomRuntimeError(
                "You must implement both `demangle_luma` and `demangle_chroma` or neither of them.", cls
            )

    @abstractmethod
    def mangle_luma(self, luma_base: vs.VideoNode, /) -> vs.VideoNode:
        """
        Mangle the luma base plane before regression.

        This may adjust resolution or siting. The resolution does not need to match the chroma planes.

        Args:
            luma_base: Luma base plane.

        Returns:
            Mangled luma plane.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    def demangle(self, clip: vs.VideoNode, /) -> vs.VideoNode:
        """
        Demangle a plane (luma or chroma).

        Applies the inverse of mangling to recover aligned planes for regression.

        Args:
            clip: Plane to demangle.

        Returns:
            Demangled plane.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError

    if TYPE_CHECKING:

        def demangle_luma(self, luma_mangled: vs.VideoNode, /) -> vs.VideoNode:
            """
            Demangle a luma plane.

            Applies the inverse of mangling to recover aligned luma for regression.

            Args:
                luma_mangled: Luma plane to demangle.

            Returns:
                Demangled luma plane.
            """
            ...

        def demangle_chroma(self, chroma_mangled: vs.VideoNode, /) -> vs.VideoNode:
            """
            Demangle a chroma plane.

            Applies the inverse of mangling to recover aligned chroma for regression.

            Args:
                chroma_mangled: Chroma plane to demangle.

            Returns:
                Demangled chroma plane.
            """
            ...

    def base_for_reconstruct(self, luma_base: vs.VideoNode, demangled_luma: vs.VideoNode, /) -> vs.VideoNode:
        """
        Optionally resize or adjust luma before reconstructing.

        By default, returns the luma base unchanged.

        Args:
            luma_base: Original luma base plane.
            demangled_luma: Demangled luma plane.

        Returns:
            Adjusted luma base plane.
        """
        return luma_base

    def regression(
        self,
        radius: int = 1,
        mean: MeanMode | MeanFunction = lambda c, radius: box_blur(c, radius, mode=ConvMode.HV),
        alpha: float = 0.20,
        **kwargs: Any,
    ) -> Sequence[RegressClips]:
        """
        Perform per-pixel regression between demangled luma and chroma planes.

        Uses local neighborhoods to estimate slopes, intercepts, and correlations.

        Args:
            radius: Neighborhood radius for mean/variance estimates. Must be > 0.
            mean: Function or mode used to compute local means. Defaults to box blur.
            alpha: Significance/confidence level for reliability weighting. Defaults to 0.20.
            **kwargs: Forwarded to [regression][vsrgtools.regress.regression].

        Raises:
            FormatsRefClipMismatchError: If luma and chroma demangled planes differ in format.
            ResolutionsRefClipMismatchError: If luma and chroma demangled planes differ in resolution.

        Returns:
            Sequence of RegressClips, one for each chroma plane.
        """
        func = kwargs.pop("func", self.regression)

        mangled_luma = self.mangle_luma(self.luma_base)

        if hasattr(self, "demangle_luma"):
            self._demangled_luma = self.demangle_luma(mangled_luma)
            self._demangled_chroma = [self.demangle_chroma(c) for c in self.chroma_bases]
        else:
            demangled_luma, *demangled_chroma = [self.demangle(c) for c in (mangled_luma, *self.chroma_bases)]
            self._demangled_luma = demangled_luma
            self._demangled_chroma = demangled_chroma

        errors = list[MismatchRefError]()
        errors_to_check = list[type[MismatchRefError]]([FormatsRefClipMismatchError, ResolutionsRefClipMismatchError])

        for uv_dm in self._demangled_chroma:
            for e in errors_to_check:
                with e.catch() as catcher:
                    e.check(func, self._demangled_luma, uv_dm)

                if catcher.error:
                    errors.append(catcher.error)

        if errors:
            raise ExceptionGroup("Demangled luma and chroma planes must have the same resolution and format.", errors)

        self._rchroma = regression(
            self._demangled_luma, *self._demangled_chroma, radius=radius, mean=mean, alpha=alpha, func=func, **kwargs
        )

        return self._rchroma

    def reconstruct(
        self,
        radius: int = 1,
        mean: MeanMode | MeanFunction = lambda c, radius: box_blur(c, radius, mode=ConvMode.HV),
        **kwargs: Any,
    ) -> Sequence[vs.VideoNode]:
        """
        Reconstruct chroma planes using regression results.

        Must be called after [regression][vsrgtools.ChromaReconstruct.regression].

        Args:
            radius: Neighborhood radius for smoothing regression coefficients.
            mean: Function or mode used to compute local means. Defaults to box blur.
            **kwargs: Forwarded to [reconstruct][vsrgtools.regress.reconstruct].

        Raises:
            CustomRuntimeError: If [regression][vsrgtools.ChromaReconstruct.regression] has not been called first.
            FormatsRefClipMismatchError: If luma formats mismatch.
            ResolutionsRefClipMismatchError: If luma resolutions mismatch.

        Returns:
            Sequence of reconstructed chroma planes.
        """
        func = kwargs.pop("func", self.reconstruct)

        if not (hasattr(self, "_demangled_luma") and hasattr(self, "_rchroma")):
            raise CustomRuntimeError("You must call `regression` before `reconstruct`.", func)

        luma_base = self.base_for_reconstruct(self.luma_base, self._demangled_luma)

        FormatsRefClipMismatchError.check(
            func,
            self._demangled_luma,
            luma_base,
            message="The formats of luma_base and the demangled luma must be the same.",
        )
        ResolutionsRefClipMismatchError.check(
            func,
            self._demangled_luma,
            luma_base,
            message="The dimensions of luma_base and the demangled luma must be the same.",
        )

        luma_fixup = core.std.MakeDiff(luma_base, self._demangled_luma)

        chroma_fixup = [reconstruct(luma_fixup, r, radius, mean, func=func, **kwargs) for r in self._rchroma]
        recons = [
            core.std.MergeDiff(chroma_dm, fixup) for chroma_dm, fixup in zip(self._demangled_chroma, chroma_fixup)
        ]

        return recons

    @cachedproperty
    def ss_shift(self) -> SubsampledShift:
        """
        Helper for handling chroma subsampling shifts.

        Returns:
            SubsampledShift: A helper object for chroma siting/offsets.
        """
        return SubsampledShift(self.chroma_loc, self.clip_src.format)
