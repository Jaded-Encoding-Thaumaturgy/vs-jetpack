from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, NoReturn, Self, Union, overload

from typing_extensions import deprecated

from vstools import (
    CustomIntEnum, CustomRuntimeError, FuncExceptT, FunctionUtil, PlanesT, core, fallback, inject_self, normalize_seq,
    vs
)

from .abstract import Debander

__all__ = [
    'SampleMode',
    'RandomAlgo',
    'F3kdb'
]


@deprecated('"SampleMode" is deprecated, use "f3k_deband.SampleMode" instead.', category=DeprecationWarning)
class SampleMode(CustomIntEnum):
    COLUMN = 1
    """Take 2 pixels as reference pixel. Reference pixels are in the same column of current pixel."""

    SQUARE = 2
    """Take 4 pixels as reference pixel. Reference pixels are in the square around current pixel."""

    ROW = 3
    """Take 2 pixels as reference pixel. Reference pixels are in the same row of current pixel."""

    COL_ROW_MEAN = 4
    """Arithmetic mean of COLUMN and ROW. Reference points are randomly picked within the range."""

    MEAN_DIFF = 5
    """Similar to COL_ROW_MEAN, adds max/mid diff thresholds."""

    @overload
    def __call__(  # type: ignore
        self: Union[
            Literal[SampleMode.COLUMN],
            Literal[SampleMode.SQUARE],
            Literal[SampleMode.ROW],
            Literal[SampleMode.COL_ROW_MEAN],
        ]
    ) -> NoReturn:
        ...

    @overload
    def __call__(  # type: ignore
        self: Literal[SampleMode.MEAN_DIFF], thr_mid: int | list[int], thr_max: int | list[int], /,
    ) -> SampleModeMidDiffInfo:
        ...

    def __call__(self, *args: Any) -> Any:
        if self != SampleMode.MEAN_DIFF:
            raise TypeError

        return SampleModeMidDiffInfo(self, *args)


class SampleModeMidDiffInfo(NamedTuple):
    sample_mode: SampleMode
    thr_mid: int | list[int]
    thr_max: int | list[int]


@deprecated('"RandomAlgo" is deprecated, use "f3k_deband.RandomAlgo" instead.', category=DeprecationWarning)
class RandomAlgo(CustomIntEnum):
    """Random number algorithm for reference positions / grains."""

    OLD = 0
    """Algorithm in old versions"""

    UNIFORM = 1
    """Uniform distribution"""

    GAUSSIAN = 2
    """Gaussian distribution"""

    @overload
    def __call__(self: Literal[RandomAlgo.OLD] | Literal[RandomAlgo.UNIFORM]) -> NoReturn:  # type: ignore
        ...

    @overload
    def __call__(self: Literal[RandomAlgo.GAUSSIAN], sigma: float, /,) -> RandomAlgoWithInfo:  # type: ignore
        """
        StdDev (sigma).
        Only values in [-1.0, 1.0] is used for multiplication, numbers outside this range are simply ignored)
        """
        ...

    def __call__(self, *args: Any) -> Any:
        if self != RandomAlgo.GAUSSIAN:
            return TypeError

        return RandomAlgoWithInfo(self, *args)


class RandomAlgoWithInfo(int):
    sigma: float

    def __new__(cls, x: int, sigma: float) -> Self:
        instance = super().__new__(cls, x)
        instance.sigma = sigma

        return instance


RandomAlgoT = RandomAlgo | RandomAlgoWithInfo


@deprecated('"F3kdb" is deprecated, use "f3k_deband" instead.', category=DeprecationWarning)
@dataclass
class F3kdb(Debander):
    """Debander wrapper around the f3kdb plugin."""

    radius: int | None = None
    thr: int | list[int] | None = None
    grain: int | list[int] | None = None

    sample_mode: SampleMode | SampleModeMidDiffInfo | None = None

    seed: int | None = None
    dynamic_grain: int | None = None

    blur_first: bool | None = None

    @deprecated('"F3kdb.deband" is deprecated, use "f3k_deband" instead.', category=DeprecationWarning)
    @inject_self
    def deband(  # type: ignore[override]
        self, clip: vs.VideoNode,
        radius: int = 16,
        thr: int | list[int] = 96,
        grain: float | list[float] = 0.0,
        sample_mode: SampleMode | SampleModeMidDiffInfo = SampleMode.SQUARE,
        dynamic_grain: bool = False,
        blur_first: bool | None = None,
        seed: int | None = None,
        random: RandomAlgoT | tuple[RandomAlgoT, RandomAlgoT] = RandomAlgo.UNIFORM,
        planes: PlanesT = None,
        _func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """
        :param clip:            Input clip.
        :param radius:          Banding detection range.
        :param thr:             Banding detection threshold for respective plane.
                                If difference between current pixel and reference pixel is less than threshold,
                                it will be considered as banded
        :param grain:           Specifies amount of grains added in the last debanding stage.
        :param sample_mode:     Determines how pixels are taken as reference.
        :param dynamic_grain:   Use different grain pattern for each frame.
        :param blur_first:      If True current pixel is compared with average value of all pixels.
                                If False current pixel is compared with all pixels.
                                The pixel is considered as banded pixel only if all differences are less than threshold.
        :param seed:            Seed for random number generation
        :param random:          Random number algorithm for reference positions / grains.
        :param planes:          Which planes to process.
        """
        func = FunctionUtil(clip, _func or self.deband, planes, (vs.GRAY, vs.YUV), (8, 16))

        if not hasattr(core, 'neo_f3kdb'):
            raise CustomRuntimeError('You are missing the neo_f3kdb plugin!', func.func)

        if 'y_2' not in core.neo_f3kdb.Deband.__signature__.parameters:  # type: ignore
            raise CustomRuntimeError('You are using an outdated version of neo_f3kdb, upgrade now!', func.func)

        radius = fallback(self.radius, radius)

        y, cb, cr = normalize_seq(fallback(self.thr, thr), 3)
        gry, grc = normalize_seq(fallback(self.grain, grain), 2)

        sample_mode = fallback(self.sample_mode, sample_mode)  # type: ignore

        random_ref, random_grain = normalize_seq(random, 2)

        if isinstance(random_ref, RandomAlgoWithInfo):
            random_algo_ref = int(random_ref)
            random_param_ref = random_ref.sigma
        else:
            random_algo_ref = int(random_ref)
            random_param_ref = 1.0

        if isinstance(random_grain, RandomAlgoWithInfo):
            random_algo_grain = int(random_grain)
            random_param_grain = random_grain.sigma
        else:
            random_algo_grain = int(random_grain)
            random_param_grain = 1.0

        y1 = cb1 = cr1 = y2 = cb2 = cr2 = None

        if isinstance(sample_mode, SampleModeMidDiffInfo):
            y1, cb1, cr1 = func.norm_seq(sample_mode.thr_max, 0)
            y2, cb2, cr2 = func.norm_seq(sample_mode.thr_mid, 0)
            sample_mode = sample_mode.sample_mode

        blur_first = fallback(self.blur_first or blur_first, max(y, cb, cr) < 2048)  # type: ignore

        debanded = core.neo_f3kdb.Deband(
            func.work_clip, radius, y, cb, cr, gry * 255 * 0.8, grc * 255 * 0.8,  # type: ignore
            sample_mode.value, self.seed or seed, blur_first, self.dynamic_grain or dynamic_grain,
            None, None, None, False, 16, random_algo_ref, random_algo_grain,
            random_param_ref, random_param_grain, None, y1, cb1, cr1, y2, cb2, cr2, True
        )

        return func.return_clip(debanded)
