from __future__ import annotations

from types import NoneType
from typing import Any, Callable, Generic, Literal, Protocol, Sequence, TypeVar, overload

from jetpytools import CustomValueError, P, R, to_arr

from vsdenoise import PrefilterLike
from vsrgtools import gauss_blur, limit_filter
from vstools import (
    ConstantFormatVideoNode, CustomIntEnum, InvalidColorFamilyError, PlanesT, check_variable, core, depth, expect_bits,
    join, normalize_param_planes, normalize_seq, split, vs
)

__all__ = [
    "f3k_deband",
    "placebo_deband",
    "mdb_bilateral",
    "pfdeband",
]


class F3KDeband(Generic[P, R]):
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
        """
        Take 2 reference pixels from the same column as the current pixel.
        """

        SQUARE = 2
        """
        Take 4 reference pixels forming a square pattern around the current pixel.
        """

        ROW = 3
        """
        Take 2 reference pixels from the same row as the current pixel.
        """

        COL_ROW_MEAN = 4
        """
        Take the arithmetic mean of COLUMN and ROW sampling patterns.
        Reference pixels are randomly selected from both configurations.
        """

        MEAN_DIFF_INT = 5
        """
        Similar to COL_ROW_MEAN, but includes configurable maximum and median 
        difference thresholds for finer control and detail preservation.
        """

        MEAN_DIFF_FLOAT = 6
        """
        Similar to COL_ROW_MEAN, but includes configurable maximum and median 
        difference thresholds for finer control and detail preservation.
        """

        MEAN_DIFF_GRADIENT = 7
        """
        An extension of MEAN_DIFF_FLOAT that adds a gradient angle check for smarter detail preservation.
        """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_INT],
            y1: int | None = None, cb1: int | None = None, cr1: int | None = None,
            y2: int | None = None, cb2: int | None = None, cr2: int | None = None,
            /,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_INT]:
            """
            Configure MEAN_DIFF_INT using direct values.

            :param y1:      maxDif threshold for luma (Y).
            :param cb1:     maxDif threshold for blue-difference chroma (U).
            :param cr1:     maxDif threshold for red-difference chroma (V).
            :param y2:      midDif threshold for luma (Y).
            :param cb2:     midDif threshold for blue-difference chroma (U).
            :param cr2:     midDif threshold for red-difference chroma (V).
            :return:        The configured enum.
            """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_INT],
            thr_max: Sequence[int | None] | None = None,
            thr_mid: Sequence[int | None] | None = None,
            /,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_INT]:
            """
            Configure MEAN_DIFF_INT using threshold sequences.

            :param thr_max:     maxDif thresholds for Y, Cb, and Cr.
            :param thr_mid:     midDif thresholds for Y, Cb, and Cr.
            :return:            The configured enum.
            """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_FLOAT],
            y1: int | None = None, cb1: int | None = None, cr1: int | None = None,
            y2: int | None = None, cb2: int | None = None, cr2: int | None = None,
            /,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_FLOAT]:
            """
            Configure MEAN_DIFF_FLOAT using direct values.

            :param y1:      maxDif threshold for luma (Y).
            :param cb1:     maxDif threshold for blue-difference chroma (U).
            :param cr1:     maxDif threshold for red-difference chroma (V).
            :param y2:      midDif threshold for luma (Y).
            :param cb2:     midDif threshold for blue-difference chroma (U).
            :param cr2:     midDif threshold for red-difference chroma (V).
            :return:        The configured enum.
            """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_FLOAT],
            thr_max: Sequence[int | None] | None = None,
            thr_mid: Sequence[int | None] | None = None,
            /,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_FLOAT]:
            """
            Configure MEAN_DIFF_FLOAT using threshold sequences.

            :param thr_max:     maxDif thresholds for Y, Cb, and Cr.
            :param thr_mid:     midDif thresholds for Y, Cb, and Cr.
            :return:            The configured enum.
            """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_GRADIENT],
            y1: int | None = None, cb1: int | None = None, cr1: int | None = None,
            y2: int | None = None, cb2: int | None = None, cr2: int | None = None,
            /,
            angle_boost: float | None = None,
            max_angle: float | None = None,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_GRADIENT]:
            """
            Configure MEAN_DIFF_GRADIENT using direct values.

            :param y1:          maxDif threshold for luma (Y).
            :param cb1:         maxDif threshold for blue-difference chroma (U).
            :param cr1:         maxDif threshold for red-difference chroma (V).
            :param y2:          midDif threshold for luma (Y).
            :param cb2:         midDif threshold for blue-difference chroma (U).
            :param cr2:         midDif threshold for red-difference chroma (V).
            :param angle_boost: Multiplier to increase the debanding strength on consistent gradients.
            :param max_angle:   Threshold for the gradient angle check.
            :return:            The configured enum.
            """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF_GRADIENT],
            thr_max: Sequence[int | None] | None = None,
            thr_mid: Sequence[int | None] | None = None,
            /,
            *,
            angle_boost: float | None = None,
            max_angle: float | None = None,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF_GRADIENT]:
            """
            Configure MEAN_DIFF_GRADIENT using threshold sequences.

            :param thr_max:     maxDif thresholds for Y, Cb, and Cr.
            :param thr_mid:     midDif thresholds for Y, Cb, and Cr.
            :param angle_boost: Multiplier to increase the debanding strength on consistent gradients.
            :param max_angle:   Threshold for the gradient angle check.
            :return:            The configured enum.
            """

        def __call__(
            self,
            y1_or_thr_max: int | None | Sequence[int | None] = None,
            cb1_or_thr_mid: int | None | Sequence[int | None] = None,
            cr1: int | None = None,
            y2: int | None = None, cb2: int | None = None, cr2: int | None = None,
            angle_boost: float | None = None,
            max_angle: float | None = None,
        ) -> Any:
            """
            Configure `MEAN_DIFF` with either individual values or sequences.

            You can use either:
            - Individual thresholds: ``y1, cb1, cr1, y2, cb2, cr2``
            - Sequences: ``thr_max`` and ``thr_mid``

            ### y1 / cb1 / cr1
            These are detail protection thresholds (max difference).

            This threshold applies to the maxDif check.
            maxDif is the largest absolute difference found between the current pixel
            and any of its four individual cross-shaped reference pixels.
            If this maxDif is greater than or equal to y1 / cb1 / cr1, the pixel is considered detail.

            It helps protect sharp edges and fine details from being blurred by the debanding process.

            ### y2 / cb2 / cr2
            These are gradient / texture protection threshold (mid-pair difference).

            This threshold applies to the midDif checks.
            midDif measures how much the current pixel deviates from the midpoint of a pair of opposing reference pixels
            (one check for the vertical pair, one for the horizontal pair).
            If the current pixel is far from this midpoint (i.e., midDif is greater than or equal to y2 / cb2 / cr2), it might indicate a texture.

            This helps distinguish true banding in gradients from textured areas or complex details.

            :param y1_or_thr_max:   maxDif threshold for luma (Y) or sequence of max thresholds.
            :param cb1_or_thr_mid:  maxDif threshold for blue-difference chroma (U) or sequence of mid thresholds.
            :param cr1:             maxDif threshold for red-difference chroma (V).
            :param y2:              midDif threshold for luma (Y).
            :param cb2:             midDif threshold for blue-difference chroma (U).
            :param cr2:             midDif threshold for red-difference chroma (V).
            :param angle_boost:     Multiplier to increase the debanding strength on consistent gradients.
            :param max_angle:       Threshold for the gradient angle check.
            :return:                The configured enum.
            """
            assert self >= 5

            if isinstance(y1_or_thr_max, Sequence) and isinstance(cb1_or_thr_mid, Sequence):
                y1, cb1, cr1 = normalize_seq(y1_or_thr_max, 3)
                y2, cb2, cr2 = normalize_seq(cb1_or_thr_mid, 3)
            elif isinstance(y1_or_thr_max, (int, NoneType)) and isinstance(cb1_or_thr_mid, (int, NoneType)):
                y1, cb1 = y1_or_thr_max, cb1_or_thr_mid
            else:
                raise TypeError

            new_enum = CustomIntEnum(self.__class__.__name__, F3KDeband.SampleMode.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.kwargs = dict[str, Any](y1=y1, cb1=cb1, cr1=cr1, y2=y2, cb2=cb2, cr2=cr2)

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
        """Legacy algorithm from older versions."""

        UNIFORM = 1
        """Uniform distribution for randomization."""

        GAUSSIAN = 2
        """Gaussian distribution. Supports custom standard deviation via `__call__`."""

        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.RandomAlgo.GAUSSIAN], sigma: float, /,  # pyright: ignore[reportGeneralTypeIssues]
        ) -> Literal[F3KDeband.RandomAlgo.GAUSSIAN]:
            """
            Configure the standard deviation for the GAUSSIAN algorithm.

            Only values in the range `[-1.0, 1.0]` are considered valid and used as multipliers.
            Values outside this range are ignored.

            :param sigma:   Standard deviation used for the Gaussian distribution.
            :return:        A new instance of GAUSSIAN with the given `sigma`.
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
    thr: int | Sequence[int] = 96,
    grain: float | Sequence[float] = 0.0,
    planes: PlanesT = None,
    *,
    sample_mode: F3KDeband.SampleMode = F3KDeband.SampleMode.SQUARE,
    dynamic_grain: bool = False,
    blur_first: bool = True,
    seed: int | None = None,
    random: F3KDeband.RandomAlgo | tuple[F3KDeband.RandomAlgo, F3KDeband.RandomAlgo] = F3KDeband.RandomAlgo.UNIFORM,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Debanding filter wrapper using the `neo_f3kdb` plugin.

    More informations can be found in the plugin documentation: https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb

    :param clip:            Input clip.
    :param radius:          Radius used for banding detection. Valid range is [1, 255].
    :param thr:             Banding detection threshold(s) for each plane (Y, Cb, Cr).
                            A pixel is considered banded if the difference with its reference pixel(s)
                            is less than the corresponding threshold.
    :param grain:           Amount of grain to add after debanding. Accepts a float or sequence of floats.
    :param planes:          Specifies which planes to process. Default is all planes.
    :param sample_mode:     Strategy used to sample reference pixels. See [SampleMode][vsdeband.debanders.F3KDeband.SampleMode].
    :param dynamic_grain:   If True, generates a unique grain pattern for each frame.
    :param blur_first:      If True, compares current pixel to the mean of surrounding pixels.
                            If False, compares directly to all reference pixels. A pixel is marked as banded
                            only if all pixel-wise differences are below threshold.
    :param seed:            Random seed for grain generation.
    :param random:          Random number generation strategy.
                            Can be a single value for both reference and grain, or a tuple specifying separate algorithms.
                            See [RandomAlgo][vsdeband.debanders.F3KDeband.RandomAlgo].
    :param kwargs:          Additional keyword arguments passed directly to the `neo_f3kdb.Deband` plugin.

    :raises CustomValueError:           If inconsistent grain parameters are provided across chroma planes.
    :raises InvalidColorFamilyError:    If input clip is not YUV or GRAY.

    :return:                Debanded and optionally grained clip.
    """

    InvalidColorFamilyError.check(clip, (vs.GRAY, vs.YUV), f3k_deband)

    clip, bits = expect_bits(clip, 16)

    y, cb, cr = normalize_param_planes(clip, normalize_seq(thr, 3), planes, 0, f3k_deband)
    grainy, *ngrainc = normalize_param_planes(clip, normalize_seq(grain, 2), planes, 0, f3k_deband)

    if len(set(ngrainc)) > 1:
        raise CustomValueError("Inconsistent grain parameters across chroma planes.", f3k_deband)

    random_ref, random_grain = normalize_seq(random, 2)

    kwargs = (
        sample_mode.kwargs
        | dict(
            scale=True,
            random_algo_ref=random_ref,
            random_algo_grain=random_ref.sigma,
            random_param_ref=random_grain,
            random_param_grain=random_grain.sigma
        )
        | kwargs
    )

    debanded = core.neo_f3kdb.Deband(
        clip,
        radius,
        y,
        cb,
        cr,
        round(grainy * 255 * 0.8),
        round(ngrainc[0] * 255 * 0.8),
        sample_mode,
        seed,
        blur_first,
        dynamic_grain,
        **kwargs
    )

    return depth(debanded, bits)


def placebo_deband(
    clip: vs.VideoNode,
    radius: float = 16.0,
    thr: float | Sequence[float] = 3.0,
    grain: float | Sequence[float] = 0.0,
    planes: PlanesT = None,
    *,
    iterations: int = 4,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Debanding wrapper around the `placebo.Deband` filter from the VapourSynth `vs-placebo` plugin.

    For full plugin documentation, see:
    https://github.com/sgt0/vs-placebo?tab=readme-ov-file#deband

    :param clip:        Input clip.
    :param radius:      Initial debanding radius.
                        The radius increases linearly with each iteration.
                        A higher radius will find more gradients, but a lower radius will smooth more aggressively.
    :param thr:         Cut-off threshold(s) for each plane. Higher values increase debanding strength
                        but may remove fine details. Accepts a single float or a sequence per plane.
    :param grain:       Amount of grain/noise to add after debanding. Helps mask residual artifacts.
                        Accepts a float or a sequence per plane.
                        Note: For HDR content, grain can significantly affect brightness — consider reducing or disabling.
    :param planes:      Which planes to process. Defaults to all planes.
    :param iterations:  Number of debanding steps to perform per sample.
                        More iterations yield stronger effect but quickly lose efficiency beyond 4.
    :return:            Debanded and optionally grained clip.
    """

    assert check_variable(clip, placebo_deband)

    thr = normalize_param_planes(clip, normalize_seq(thr), planes, 0, placebo_deband)
    ngrain = normalize_param_planes(clip, normalize_seq(grain), planes, 0, placebo_deband)

    def _placebo(
        clip: ConstantFormatVideoNode, threshold: float, grain_val: float, planes: Sequence[int]
    ) -> ConstantFormatVideoNode:
        plane = 0

        if threshold == grain_val == 0:
            return clip

        for p in planes:
            plane |= pow(2, p)

        return clip.placebo.Deband(
            plane, iterations, threshold, radius, grain_val * (1 << 5) * 0.8, **kwargs
        )

    set_grn = set(ngrain)

    if set_grn == {0} or clip.format.num_planes == 1:
        debs = [_placebo(p, t, ngrain[0], [0]) for p, t in zip(split(clip), thr)]

        if len(debs) == 1:
            return debs[0]

        return join(debs, clip.format.color_family)

    plane_map = {
        tuple(i for i in range(clip.format.num_planes) if ngrain[i] == x): x for x in set_grn - {0}
    }

    debanded = clip

    for planes, grain_val in plane_map.items():
        if len(set(thr[p] for p in planes)) == 1:
            debanded = _placebo(debanded, thr[planes[0]], grain_val, planes)
        else:
            for p in planes:
                debanded = _placebo(debanded, thr[p], grain_val, planes)

    return debanded


_Nb = TypeVar("_Nb", int, float, contravariant=True)


class _DebanderFunc(Protocol[_Nb]):
    """
    Protocol for debanding functions.
    """

    def __call__(
        self,
        clip: vs.VideoNode,
        radius: int = ...,
        thr: _Nb | Sequence[_Nb] = ...,
        grain: float | Sequence[float] = ...,
        planes: PlanesT = ...,
    ) -> vs.VideoNode: ...


def mdb_bilateral(
    clip: vs.VideoNode,
    radius: int = 16,
    thr: float = 260,
    lthr: int | tuple[int, int] = (153, 0),
    elast: float = 3.0,
    bright_thr: int | None = None,
    debander: _DebanderFunc[Any] = f3k_deband
) -> vs.VideoNode:
    """
    Multi stage debanding, bilateral-esque filter.

    This function is more of a last resort for extreme banding.

    Example usage:
    ```py
    from vsdeband import mdb_bilateral, f3k_deband
    from functools import partial

    debanded = mdb_bilateral(clip, 17, 320, debander=partial(f3k_deband, split_planes=True))
    ```

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param lthr:        Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :param elast:       Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :param bright_thr:  Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :param debander:    Specifies what debander callable to use.

    :return:            Debanded clip.
    """
    assert check_variable(clip, mdb_bilateral)

    clip, bits = expect_bits(clip, 16)

    rad1, rad2, rad3 = round(radius * 4 / 3), round(radius * 2 / 3), round(radius / 3)

    db1 = debander(clip, rad1, [max(1, th // 2) for th in to_arr(thr)], 0.0)
    db2 = debander(db1, rad2, thr, 0)
    db3 = debander(db2, rad3, thr, 0)

    limit = limit_filter(db3, db2, clip, thr=lthr, elast=elast, bright_thr=bright_thr)

    return depth(limit, bits)


class _SupportPlanesParam(Protocol):
    """
    Protocol for functions that support planes parameter.
    """

    def __call__(self, clip: vs.VideoNode, *, planes: PlanesT = ..., **kwargs: Any) -> vs.VideoNode:
        ...


def pfdeband(
    clip: vs.VideoNode,
    radius: int = 16,
    thr: float | Sequence[float] = 96,
    lthr: float | tuple[float, float] = 0.5,
    elast: float = 1.5,
    bright_thr: int | None = None,
    prefilter: PrefilterLike | _SupportPlanesParam = gauss_blur,
    debander: _DebanderFunc[Any] = f3k_deband,
    planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Prefilter and deband a clip.

    :param clip:        Input clip.
    :param radius:      Banding detection range.
    :param thr:         Banding detection thr(s) for planes.
    :param lthr:        Threshold of the limiting. Refer to `vsrgtools.limit_filter`.
    :param elast:       Elasticity of the limiting. Refer to `vsrgtools.limit_filter`.
    :param bright_thr:  Limiting over the bright areas. Refer to `vsrgtools.limit_filter`.
    :param prefilter:   Prefilter used to blur the clip before debanding.
    :param debander:    Specifies what debander callable to use.
    :param planes:      Planes to process

    :return:              Debanded clip.
    """
    clip, bits = expect_bits(clip, 16)

    blur = prefilter(clip, planes=planes, **kwargs)
    diff = clip.std.MakeDiff(blur, planes=planes)

    deband = debander(blur, radius, thr, planes=planes)

    merge = deband.std.MergeDiff(diff, planes=planes)
    limit = limit_filter(merge, clip, thr=lthr, elast=elast, bright_thr=bright_thr)

    return depth(limit, bits)
