from __future__ import annotations

from collections.abc import Callable, Sequence
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, Protocol, overload

from jetpytools import CustomIntEnum, normalize_seq, to_arr

if TYPE_CHECKING:
    from vsdenoise import PrefilterLike

from vsexprtools import norm_expr
from vsrgtools import gauss_blur
from vstools import Planes, core, depth, expect_bits, join, normalize_param_planes, split, vs

__all__ = ["f3k_deband", "mdb_bilateral", "pfdeband", "placebo_deband"]

logger = getLogger(__name__)


class F3KDeband[**P, R]:
    """
    Class decorator that wraps the [f3k_deband][vsdeband.debanders.f3k_deband] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, f3k_deband: Callable[P, R]) -> None:
        self._func = f3k_deband

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class SampleMode(CustomIntEnum):
        """
        Enum that determines how reference pixels are sampled for debanding.
        """

        kwargs: dict[str, Any]
        """Additional keyword arguments."""

        def __init__(self, value: int, **kwargs: Any) -> None:
            self._value_ = value
            self.kwargs = kwargs

        COLUMN = 1
        """Column references (vertical pair)."""

        SQUARE = 2
        """Square references (four points)."""

        ROW = 3
        """Row references (horizontal pair)."""

        COL_ROW_MEAN = 4
        """Average of modes COLUMN and ROW."""

        MEAN_DIFF_INT = 5
        """Detail-preserving mode using additional thresholds (`thr1`, `thr2`)."""

        MEAN_DIFF_FLOAT = 6
        """Similar to COL_ROW_MEAN but uses multiple thresholds for detail preservation."""

        MEAN_DIFF_GRADIENT = 7
        """An extension of MEAN_DIFF_FLOAT that adds a gradient angle check for more intelligent detail preservation."""

        @overload
        def __call__[
            MeanDiffT: (Literal[F3KDeband.SampleMode.MEAN_DIFF_INT], Literal[F3KDeband.SampleMode.MEAN_DIFF_FLOAT])
        ](
            self: MeanDiffT,
            thr_max: float | Sequence[float] | None = None,
            thr_mid: float | Sequence[float] | None = None,
        ) -> MeanDiffT:
            """
            Configure MEAN_DIFF_INT or MEAN_DIFF_FLOAT using threshold sequences.

            Args:
                thr_max: Detail protection threshold (`thr1` in vszip) for respective planes.
                    Applies to the max difference check.
                thr_mid: Gradient/Texture protection threshold (`thr2` in vszip) for respective planes.
                    Applies to the midpoint difference check.

            Returns:
                The configured enum.
            """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_GRADIENT],
            thr_max: float | Sequence[float] | None = None,
            thr_mid: float | Sequence[float] | None = None,
            angle_boost: float | None = None,
            max_angle: float | None = None,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_GRADIENT]:
            """
            Configure MEAN_DIFF_GRADIENT using threshold sequences.

            Args:
                thr_max: Detail protection threshold (`thr1` in vszip) for respective planes.
                    Applies to the max difference check.
                thr_mid: Gradient/Texture protection threshold (`thr2` in vszip) for respective planes.
                    Applies to the midpoint difference check.
                angle_boost: Multiplier to increase the debanding strength on consistent gradients.
                max_angle: Threshold for the gradient angle check.

            Returns:
                The configured enum.
            """

        def __call__(
            self,
            thr_max: float | Sequence[float] | None = None,
            thr_mid: float | Sequence[float] | None = None,
            angle_boost: float | None = None,
            max_angle: float | None = None,
        ) -> Any:
            """
            Configure `MEAN_DIFF` with either individual values or sequences.

            ## `thr_max` or `thr1` in vszip
            These are detail protection thresholds (max difference).

            This threshold applies to the maxDif check.
            maxDif is the largest absolute difference found between the current pixel
            and any of its four individual cross-shaped reference pixels.
            If this maxDif is greater than or equal to any of the thresholds, the pixel is considered detail.

            It helps protect sharp edges and fine details from being blurred by the debanding process.

            ## `thr_mid` or `thr2` in vszip
            These are gradient / texture protection thresholds (mid-pair difference).

            This threshold applies to the midDif checks.
            midDif measures how much the current pixel deviates from the midpoint of a pair of opposing reference pixels
            (one check for the vertical pair, one for the horizontal pair).
            If the current pixel is far from this midpoint (i.e., midDif is greater than or equal
            to any of the thresholds), it might indicate a texture.

            This helps distinguish true banding in gradients from textured areas or complex details.

            Args:
                thr_max: Detail protection threshold (`thr1` in vszip) for respective planes.
                    Applies to the max difference check.
                thr_mid: Gradient/Texture protection threshold (`thr2` in vszip) for respective planes.
                    Applies to the midpoint difference check.
                angle_boost: Multiplier to increase the debanding strength on consistent gradients.
                max_angle: Threshold for the gradient angle check.

            Returns:
                The configured enum.
            """
            assert self >= 5

            new_enum = CustomIntEnum(self.__class__.__name__, F3KDeband.SampleMode.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.kwargs = dict[str, Any](thr1=thr_max, thr2=thr_mid)

            if self is F3KDeband.SampleMode.MEAN_DIFF_GRADIENT:
                member.kwargs.update(angle_boost=angle_boost, max_angle=max_angle)

            return member

    class RandomAlgo(CustomIntEnum):
        """
        Random number generation algorithm used for reference positions or grain patterns.
        """

        sigma: float | None
        """Standard deviation value used only for GAUSSIAN."""

        def __init__(self, value: int, sigma: float | None = None) -> None:
            self._value_ = value
            self.sigma = sigma

        OLD = 0
        """
        Legacy algorithm from older versions.
        """

        UNIFORM = 1
        """
        Uniform distribution for randomization.
        """

        GAUSSIAN = 2
        """
        Gaussian distribution. Supports custom standard deviation via `__call__`.
        """

        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.RandomAlgo.GAUSSIAN],  # pyright: ignore[reportGeneralTypeIssues]
            sigma: float,
            /,
        ) -> Literal[F3KDeband.RandomAlgo.GAUSSIAN]:
            """
            Configure the standard deviation for the GAUSSIAN algorithm.

            Only values in the range `[-1.0, 1.0]` are considered valid and used as multipliers.
            Values outside this range are ignored.

            Args:
                sigma: Standard deviation used for the Gaussian distribution.

            Returns:
                A new instance of GAUSSIAN with the given `sigma`.
            """
            assert self is F3KDeband.RandomAlgo.GAUSSIAN

            new_enum = CustomIntEnum(self.__class__.__name__, F3KDeband.RandomAlgo.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.sigma = sigma

            return member


@F3KDeband
def f3k_deband(
    clip: vs.VideoNode,
    radius: int = 16,
    thr: float | Sequence[float] = 96,
    grain: float | Sequence[float] = 0,
    planes: Planes = None,
    *,
    sample_mode: F3KDeband.SampleMode = F3KDeband.SampleMode.SQUARE,
    dynamic_grain: bool = False,
    blur_first: bool = True,
    seed: int | None = None,
    random: F3KDeband.RandomAlgo | tuple[F3KDeband.RandomAlgo, F3KDeband.RandomAlgo] = F3KDeband.RandomAlgo.UNIFORM,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Debanding filter wrapper using the `vszip.Deband` plugin.

    More information:
        <https://github.com/dnjulek/vapoursynth-zip/wiki/Deband>
        <https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb>

    Args:
        clip: Input clip.
        radius: Radius used for banding detection. Valid range is [1, 255].
        thr: Banding detection threshold(s) for each plane. A pixel is considered banded if the difference
            with its reference pixel(s) is less than the corresponding threshold.
        grain: Amount of grain to add after debanding.
        planes: Specifies which planes to process. Default is all planes.
        sample_mode: Strategy used to sample reference pixels.
            See [SampleMode][vsdeband.debanders.F3KDeband.SampleMode].
        dynamic_grain: If True, generates a unique grain pattern for each frame.
        blur_first: If True, compares current pixel to the mean of surrounding pixels. If False, compares directly to
            all reference pixels. A pixel is marked as banded only if all pixel-wise differences are below threshold.
        seed: Random seed for grain generation.
        random: Random number generation strategy. Can be a single value for both reference and grain, or a tuple
            specifying separate algorithms. See [RandomAlgo][vsdeband.debanders.F3KDeband.RandomAlgo].
        **kwargs: Additional keyword arguments passed directly to the `vszip.Deband` plugin.

    Returns:
        Debanded and optionally grained clip.
    """

    # Simulate scale=True like the default we had when wrapping neo_f3kdb
    scale = kwargs.pop("scale", True)
    thr = [t * 255 / ((1 << (16 if scale else 14)) - 1) for t in to_arr(thr)]
    grain = [g * 255 / ((1 << 14) - 1) for g in to_arr(grain)]

    thr = normalize_param_planes(clip, thr, planes, 0)
    grain = normalize_param_planes(clip, grain, planes, 0)

    random_ref, random_grain = normalize_seq(random, 2)

    kwargs = (
        sample_mode.kwargs
        | {
            "random_algo_ref": random_ref,
            "random_algo_grain": random_ref.sigma,
            "random_param_ref": random_grain,
            "random_param_grain": random_grain.sigma,
        }
        | kwargs
    )

    logger.debug("vszip.Deband params: thr=%s, grain=%s %s", thr, grain, kwargs)

    return core.vszip.Deband(
        clip,
        radius,
        thr,
        grain[:2],
        sample_mode,
        seed,
        blur_first,
        dynamic_grain,
        **kwargs,
    )


def placebo_deband(
    clip: vs.VideoNode,
    radius: float = 16.0,
    thr: float | Sequence[float] = 3.0,
    grain: float | Sequence[float] = 0.0,
    planes: Planes = None,
    *,
    iterations: int = 4,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Debanding wrapper around the `placebo.Deband` filter from the VapourSynth `vs-placebo` plugin.

    For full plugin documentation, see:
    <https://github.com/sgt0/vs-placebo?tab=readme-ov-file#deband>

    Args:
        clip: Input clip.
        radius: Initial debanding radius. The radius increases linearly with each iteration. A higher radius will find
            more gradients, but a lower radius will smooth more aggressively.
        thr: Cut-off threshold(s) for each plane. Higher values increase debanding strength but may remove fine details.
            Accepts a single float or a sequence per plane.
        grain: Amount of grain/noise to add after debanding. Helps mask residual artifacts. Accepts a float or a
            sequence per plane. Note: For HDR content, grain can significantly affect brightness. Consider reducing or
            disabling.
        planes: Which planes to process. Defaults to all planes.
        iterations: Number of debanding steps to perform per sample. More iterations yield stronger effect but quickly
            lose efficiency beyond 4.

    Returns:
        Debanded and optionally grained clip.
    """
    thr = normalize_param_planes(clip, thr, planes, 0)
    ngrain = normalize_param_planes(clip, grain, planes, 0)

    def _placebo(clip: vs.VideoNode, threshold: float, grain_val: float, planes: Sequence[int]) -> vs.VideoNode:
        plane = 0

        if threshold == grain_val == 0:
            return clip

        for p in planes:
            plane |= pow(2, p)

        return clip.placebo.Deband(plane, iterations, threshold, radius, grain_val, **kwargs)

    set_grn = set(ngrain)

    if set_grn == {0} or clip.format.num_planes == 1:
        debs = [_placebo(p, t, ngrain[0], [0]) for p, t in zip(split(clip), thr)]

        if len(debs) == 1:
            return debs[0]

        return join(debs, clip.format.color_family)

    plane_map = {tuple(i for i in range(clip.format.num_planes) if ngrain[i] == x): x for x in set_grn - {0}}

    debanded = clip

    for planes, grain_val in plane_map.items():
        if len({thr[p] for p in planes}) == 1:
            debanded = _placebo(debanded, thr[planes[0]], grain_val, planes)
        else:
            for p in planes:
                debanded = _placebo(debanded, thr[p], grain_val, planes)

    return debanded


class _DebanderFunc[Nb: (float, int)](Protocol):
    """
    Protocol for debanding functions.
    """

    def __call__(
        self,
        clip: vs.VideoNode,
        radius: int = ...,
        thr: Nb | Sequence[Nb] = ...,
        grain: Nb | Sequence[Nb] = ...,
        planes: Planes = ...,
    ) -> vs.VideoNode: ...


def mdb_bilateral(
    clip: vs.VideoNode,
    radius: int = 16,
    thr: float = 260,
    debander: _DebanderFunc[Any] = f3k_deband,
    dark_thr: float | Sequence[float] = 0.6,
    bright_thr: float | Sequence[float] = 0.6,
    elast: float | Sequence[float] = 3.0,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    Multi stage debanding, bilateral-esque filter.

    This function is more of a last resort for extreme banding.

    Example usage:
        ```py
        from vsdeband import mdb_bilateral

        debanded = mdb_bilateral(clip, 22, 320)
        ```

    Args:
        clip: Input clip.
        radius: Banding detection range.
        thr: Banding detection thr(s) for planes.
        debander: Specifies what debander callable to use.
        dark_thr: LimitFilter parameter. Threshold (8-bit scale) to limit dark filtering diff.
        bright_thr: LimitFilter parameter. Threshold (8-bit scale) to limit bright filtering diff.
        elast: LimitFilter parameter. Elasticity of the soft threshold.
        planes: Which planes to process.

    Returns:
        Debanded clip.
    """
    clip, bits = expect_bits(clip, 16)

    rad1, rad2, rad3 = round(radius * 4 / 3), round(radius * 2 / 3), round(radius / 3)

    db1 = debander(clip, rad1, [max(1, th // 2) for th in to_arr(thr)], 0, planes)
    db2 = debander(db1, rad2, thr, 0, planes)
    db3 = debander(db2, rad3, thr, 0, planes)

    limit = core.vszip.LimitFilter(db3, clip, db1, dark_thr, bright_thr, elast, planes)

    return depth(limit, bits)


class _SupportPlanesParam(Protocol):
    """
    Protocol for functions that support planes parameter.
    """

    def __call__(self, clip: vs.VideoNode, *, planes: Planes = ..., **kwargs: Any) -> vs.VideoNode: ...


def pfdeband(
    clip: vs.VideoNode,
    radius: int = 16,
    thr: float | Sequence[float] = 96,
    prefilter: PrefilterLike | _SupportPlanesParam = gauss_blur,
    debander: _DebanderFunc[Any] = f3k_deband,
    ref: vs.VideoNode | None = None,
    dark_thr: float | Sequence[float] = 0.3,
    bright_thr: float | Sequence[float] = 0.3,
    elast: float | Sequence[float] = 2.5,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    Prefilter and deband a clip.

    Args:
        clip: Input clip.
        radius: Banding detection range.
        thr: Banding detection thr(s) for planes.
        prefilter: Prefilter used to blur the clip before debanding.
        debander: Specifies what debander callable to use.
        planes: Planes to process
        ref: LimitFilter parameter. Reference clip, to compute the weight to be applied on filtering diff.
        dark_thr: LimitFilter parameter. Threshold (8-bit scale) to limit dark filtering diff.
        bright_thr: LimitFilter parameter. Threshold (8-bit scale) to limit bright filtering diff.
        elast: LimitFilter parameter. Elasticity of the soft threshold.

    Returns:
        Debanded clip.
    """
    clip, bits = expect_bits(clip, 16)

    blur = prefilter(clip, planes=planes)
    smooth = debander(blur, radius, thr, planes=planes)
    limit = core.vszip.LimitFilter(smooth, blur, ref, dark_thr, bright_thr, elast, planes)
    merge = norm_expr([clip, blur, limit], "z x y - +", planes, func=pfdeband)

    return depth(merge, bits)
