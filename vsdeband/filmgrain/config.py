import enum
import math
from dataclasses import dataclass, field, replace
from typing import Callable, Self

from vstools import (
    scale_value,
    vs,
)

from ..noise import Grainer

__all__ = [
    "FilmGrainConfig",
    "FilmGrainThresholds",
    "GrainConfig",
    "MergeStrengths",
    "MergeStrengths",
    "MergeType",
    "Multiplier",
    "PlaneStrength",
    "ResolutionScalingPolicy",
]


@dataclass(frozen=True, slots=True)
class GrainConfig:
    strength: float | tuple[float, float]
    sharpness: int
    size: float
    grainer: Grainer | Callable
    temporal_blur: float = 0.0
    bump: float = 0.0
    blur: bool = True


@dataclass(frozen=True, slots=True)
class Multiplier:
    dark: float
    mid: float
    bright: float


class MergeType(enum.Enum):
    GAMMA = "gamma"
    LINEAR = "linear"
    LOG = "log"

    def get_strength_multipliers(self, dark: float, mid: float, bright: float) -> tuple[float, float, float]:

        match self:
            case MergeType.GAMMA:
                return dark * 1.5, mid * 1.0, bright * 0.8
            case MergeType.LINEAR:
                return dark * 1.0, mid * 1.0, bright * 1.0
            case MergeType.LOG:
                return dark * 0.7, mid * 1.2, bright * 1.0


@dataclass(frozen=True, slots=True)
class FilmGrainThresholds:

    th1: int = 45
    th2: int = 85
    th3: int = 140
    th4: int = 200

    def to_scaled(self, clip: vs.VideoNode) -> tuple[float, float, float, float]:
        fmt = clip.format
        assert fmt, "Clip format must be defined"
        return tuple(
            scale_value(
                x,
                8,
                fmt.bits_per_sample, # type: ignore
                vs.ColorRange.RANGE_FULL
            )
            for x in (self.th1, self.th2, self.th3, self.th4)
        )


@dataclass(frozen=True, slots=True)
class PlaneStrength:
    y: float = 1.0
    u: float = 1.0
    v: float = 1.0


@dataclass(frozen=True, slots=True)
class MergeStrengths:

    dark: PlaneStrength = field(default_factory=PlaneStrength)
    mid: PlaneStrength = field(default_factory=PlaneStrength)
    bright: PlaneStrength = field(default_factory=PlaneStrength)

    @property
    def max_strength_y(self) -> float:
        return max(self.dark.y, self.mid.y, self.bright.y)

    def scale(self, factor: float) -> Self:
        return replace(
            self,
            dark=replace(self.dark, y=self.dark.y * factor, u=self.dark.u * factor, v=self.dark.v * factor),
            mid=replace(self.mid, y=self.mid.y * factor, u=self.mid.u * factor, v=self.mid.v * factor),
            bright=replace(self.bright, y=self.bright.y * factor, u=self.bright.u * factor, v=self.bright.v * factor),
        )


@dataclass(frozen=True, slots=True)
class ResolutionScalingPolicy:
    enable: bool = True

    def strength(self, width: int) -> float:
        # ~1.0 @ 1920, ~1.79 @ 3840
        return 0.409 * (width * 0.001) + 0.214

    def size(self, width: int) -> float:
        # ~1.0 @ 1920, ~1.33 @ 3840
        return 0.173 * (width * 0.001) + 0.667

    def sharpness(self, width: int) -> float:
        # ~1.0 @ 1920, ~0.34 @ 3840
        return max(min(-0.347 * (width * 0.001) + 1.667, 1.0), 0.2)

    @dataclass(frozen=True, slots=True)
    class Result:
        strengths: MergeStrengths
        size: float
        sharpness: int
        bump: float
        gen_strength: tuple[float, float]

    def apply(
        self,
        clip: vs.VideoNode,
        strengths: MergeStrengths,
        size: float,
        sharpness: int,
        bump: float,
        gen_strength: tuple[float, float],
        s16mm: bool = False,
    ) -> Result:

        if not self.enable:
            return ResolutionScalingPolicy.Result(strengths, size, sharpness, bump, gen_strength)

        w = clip.width

        strength_factor = self.strength(w)

        if s16mm:
            strength_factor *= 1.25

        if bump > 1.0:
            strength_factor /= math.sqrt(bump)

        scaled_strengths: MergeStrengths = strengths.scale(strength_factor)

        scaled_size = size * self.size(w)

        if s16mm:
            scaled_size = 1.5 * scaled_size + 0.5

        sharp01 = max(0.0, min(1.0, sharpness / 100.0))
        sharp01 = max(0.0, min(1.0, sharp01 * self.sharpness(w)))

        if s16mm:
            sharp01 *= 0.90

        scaled_sharpness = round(max(0.0, min(1.0, sharp01)) * 100)

        gen_y, gen_c = gen_strength
        scaled_gen = (gen_y * strength_factor, gen_c * strength_factor)

        return ResolutionScalingPolicy.Result(
            strengths=scaled_strengths,
            size=scaled_size,
            sharpness=scaled_sharpness,
            bump=bump,
            gen_strength=scaled_gen,
        )


@dataclass
class FilmGrainConfig:
    strength: tuple[float, float] = (1.2, 0.5)
    merge: MergeStrengths = field(default_factory=MergeStrengths)
    size: float = 1.2
    sharpness: int = 66
    preblur: float = 1.0
    temporal_blur: float | tuple[float, int] = (0.0, 0)
    bump: float = 0.0
    grainer: Grainer | Callable = Grainer.GAUSS
    merge_type: MergeType = MergeType.GAMMA
    deterministic: bool = True
    thresholds: FilmGrainThresholds = field(default_factory=FilmGrainThresholds)
    resolution_policy: ResolutionScalingPolicy = field(default_factory=ResolutionScalingPolicy)
    seed: int = 17
    fast: bool = False
    s16mm: bool = False


    def apply_preset(self: Self, preset: Self) -> Self:
        return replace(self, **vars(preset))

    def with_overrides(self, param_mapping: dict, **kwargs) -> Self:

        updates = {}
        for key, value in kwargs.items():
            if mapper := param_mapping.get(key):
                updates.update(mapper(value))
            elif hasattr(self, key):
                updates[key] = value

        return replace(self, **updates)
