from __future__ import annotations

from inspect import Signature
from types import NoneType
from typing import Any, Callable, Generic, Literal, Protocol, Sequence, overload

from jetpytools import CustomValueError, P, R, to_arr
from vsrgtools import limit_filter

from vstools import (
    ConstantFormatVideoNode, CustomIntEnum, CustomRuntimeError, InvalidColorFamilyError, PlanesT, check_variable, core,
    depth, expect_bits, join, normalize_param_planes, normalize_seq, split, vs
)


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

        MEAN_DIFF = 5
        """
        Similar to COL_ROW_MEAN, but includes configurable maximum and median 
        difference thresholds for finer control.
        """

        @overload
        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF],
            y1: int | None = None, cb1: int | None = None, cr1: int | None = None,
            y2: int | None = None, cb2: int | None = None, cr2: int | None = None,
            /,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF]:
            """
            Configure MEAN_DIFF using direct values.

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
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF],
            thr_max: Sequence[int | None] | None = None,
            thr_mid: Sequence[int | None] | None = None,
            /,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF]:
            """
            Configure MEAN_DIFF using threshold sequences.

            :param thr_max:     maxDif thresholds for Y, Cb, and Cr.
            :param thr_mid:     midDif thresholds for Y, Cb, and Cr.
            :return:            The configured enum.
            """

        def __call__(  # type: ignore[misc]
            self: Literal[F3KDeband.SampleMode.MEAN_DIFF],  # pyright: ignore[reportGeneralTypeIssues]
            y1_or_thr_max: int | None | Sequence[int | None] = None,
            cb1_or_thr_mid: int | None | Sequence[int | None] = None,
            cr1: int | None = None,
            y2: int | None = None, cb2: int | None = None, cr2: int | None = None,
        ) -> Literal[F3KDeband.SampleMode.MEAN_DIFF]:
            """
            Configure `MEAN_DIFF` with either individual values or sequences.

            You can use either:
            - Individual thresholds: ``y1, cb1, cr1, y2, cb2, cr2``
            - Sequences: ``thr_max`` and ``thr_mid``

            :param y1_or_thr_max:   maxDif threshold for luma (Y) or sequence of max thresholds.
            :param cb1_or_thr_mid:  maxDif threshold for blue-difference chroma (U) or sequence of mid thresholds.
            :param cr1:             maxDif threshold for red-difference chroma (V).
            :param y2:              midDif threshold for luma (Y).
            :param cb2:             midDif threshold for blue-difference chroma (U).
            :param cr2:             midDif threshold for red-difference chroma (V).
            :return:                The configured enum.
            """
            assert self is F3KDeband.SampleMode.MEAN_DIFF

            if isinstance(y1_or_thr_max, Sequence) and isinstance(cb1_or_thr_mid, Sequence):
                y1, cb1, cr1 = normalize_seq(y1_or_thr_max, 3)
                y2, cb2, cr2 = normalize_seq(cb1_or_thr_mid, 3)
            elif isinstance(y1_or_thr_max, (int, NoneType)) and isinstance(cb1_or_thr_mid, (int, NoneType)):
                y1, cb1 = y1_or_thr_max, cb1_or_thr_mid
            else:
                raise TypeError

            new_enum = CustomIntEnum(self.__class__.__name__, F3KDeband.SampleMode.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.kwargs = dict(y1=y1, cb1=cb1, cr1=cr1, y2=y2, cb2=cb2, cr2=cr2)

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
    split_planes: bool = False,
    sample_mode: F3KDeband.SampleMode = F3KDeband.SampleMode.SQUARE,
    dynamic_grain: bool = False,
    blur_first: bool = True,
    seed: int | None = None,
    random: F3KDeband.RandomAlgo | tuple[F3KDeband.RandomAlgo, F3KDeband.RandomAlgo] = F3KDeband.RandomAlgo.UNIFORM,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Debanding filter wrapper using the `neo_f3kdb` plugin.

    :param clip:            Input clip.
    :param radius:          Radius used for banding detection. Valid range is [1, 31].
    :param thr:             Banding detection threshold(s) for each plane (Y, Cb, Cr).
                            A pixel is considered banded if the difference with its reference pixel(s)
                            is less than the corresponding threshold.
    :param grain:           Amount of grain to add after debanding. Accepts a float or sequence of floats.
    :param planes:          Specifies which planes to process. Default is all planes.
    :param split_planes:    If True, processes each plane separately and returns a joined clip.
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

    :raises CustomRuntimeError:         If the installed `neo_f3kdb` version is outdated.
    :raises CustomValueError:           If inconsistent grain parameters are provided across chroma planes.
    :raises InvalidColorFamilyError:    If input clip is not YUV or GRAY.

    :return:                Debanded and optionally grained clip.
    """

    if 'y_2' not in Signature.from_callable(core.neo_f3kdb.Deband).parameters:
        raise CustomRuntimeError(
            "Your version of neo_f3kdb is outdated. "
            "Please update the plugin to the latest version.",
            f3k_deband
    )

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

    if split_planes:
        return join(
            f3k_deband(
                p,
                radius,
                t,
                g,
                0,
                sample_mode=sample_mode,
                dynamic_grain=dynamic_grain,
                blur_first=blur_first,
                seed=seed,
                random=(random_ref, random_grain),
                **kwargs
            )
            for p, t, g in zip(split(clip), [y, cb, cr], [grainy] + ngrainc)
        ).std.CopyFrameProps(clip)

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


class PlaceboDeband(Generic[P, R]):
    """
    Class decorator that wraps the [placebo_deband][vsdeband.debanders.placebo_deband] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, placebo_deband: Callable[P, R]) -> None:
        self._func = placebo_deband

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Dither(CustomIntEnum):
        NONE = -1
        """
        No dithering.
        """

        BLUE_NOISE = 0
        """
        Dither with blue noise. Very high quality, but requires the use of a LUT.

        Warning: Computing a blue noise texture with a large size can be
        very slow, however this only needs to be performed once.
        Even so, using this with a `lut_size` greater than 6 is generally ill-advised.
        This is the preferred/default dither method.
        """

        ORDERED_LUT = 1
        """
        Dither with an ordered (bayer) dither matrix, using a LUT.

        Low quality, and since this also uses a LUT, there's generally no advantage to picking
        this instead of `BLUE_NOISE`.

        It's mainly there for testing.
        """

        ORDERED_FIXED = 2
        """
        The same as `ORDERED_LUT`, but uses fixed function math instead of a LUT.

        This is faster, but only supports a fixed dither matrix size of 16x16 (equal to a `lut_size` of 4).
        """

        WHITE_NOISE = 3
        """
        Dither with white noise.

        This does not require a LUT and is fairly cheap to compute.
        Unlike the other modes it doesn't show any repeating patterns either spatially or temporally,
        but the downside is that this is visually fairly jarring due to the presence of low frequencies
        in the noise spectrum.

        Used as a fallback when the above methods are not available.
        """

        @property
        def kwargs(self) -> dict[str, Any]:
            """Get arguments you must pass to .placebo.Debander for this dither mode."""
            if self is PlaceboDeband.Dither.NONE:
                return dict(dither=False, dither_algo=0)
            return dict(dither=True, dither_algo=self.value)


@PlaceboDeband
def placebo_deband(
    clip: vs.VideoNode,
    radius: float = 16.0,
    thr: float | Sequence[float] = 3.0,
    grain: float | Sequence[float] = 0.0,
    planes: PlanesT = None,
    *,
    iterations: int = 4,
    dither: PlaceboDeband.Dither = PlaceboDeband.Dither.BLUE_NOISE,
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
    :param dither:      Type of dithering algorithm to apply. See [Dither][vsdeband.debanders.PlaceboDeband.Dither].
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
            plane, iterations, threshold, radius, grain_val * (1 << 5) * 0.8, **dither.kwargs | kwargs
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
