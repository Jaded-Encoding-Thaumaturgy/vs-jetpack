from dataclasses import replace
from typing import Callable

from vsexprtools import inline_expr
from vsrgtools import remove_grain
from vstools import (
    core,
    get_neutral_value,
    get_peak_value,
    scale_value,
    vs,
)

from ..noise import Grainer
from .config import (
    FilmGrainConfig,
    FilmGrainThresholds,
    GrainConfig,
    MergeStrengths,
    MergeType,
    PlaneStrength,
    ResolutionScalingPolicy,
)
from .generator import FilmGrainProcessor, GrainLayerFactory
from .helpers import Signature
from .presets import FilmGrainPreset

__all__ = [
    "film_grain_plus",
    "film_grain_plus_cfg",
    "grain_factory",
]


def film_grain_plus_cfg(clip: vs.VideoNode, cfg: FilmGrainConfig, debug: bool = False) -> vs.VideoNode:
    processor = FilmGrainProcessor(cfg)
    return processor.process(clip, debug)


def film_grain_plus(
    clip: vs.VideoNode,
    strength: tuple[float, float] = (0.2, 0.1),
    merge: MergeStrengths = MergeStrengths(
        dark=PlaneStrength(1.0, 1.0, 1.0),
        mid=PlaneStrength(1.0, 1.0, 1.0),
        bright=PlaneStrength(1.0, 1.0, 1.0),
    ),
    size: float = 1.2,
    sharpness: int = 66,
    bump: float = 0.0,
    grainer: Grainer | Callable = Grainer.GAUSS,
    seed: int = 17,
    preset: FilmGrainPreset | None = None,
    mode: MergeType = MergeType.GAMMA,
    thresholds: FilmGrainThresholds = FilmGrainThresholds(th1=45, th2=85, th3=140, th4=200),
    resolution_policy: bool = True,
    temporal_blur: float | tuple[float, int] = (0.0, 0),
    preblur: float | bool = False,
    deterministic: bool = False,
    fast: bool = False,
    debug: bool = False,
    s16mm: bool = False,
) -> vs.VideoNode:

    settings = FilmGrainConfig(
        strength=strength,
        size=size,
        sharpness=sharpness,
        bump=bump,
        grainer=grainer,
        seed=seed,
        temporal_blur=temporal_blur,
        fast=fast,
        thresholds=thresholds,
        merge=merge,
        merge_type=mode,
        resolution_policy=ResolutionScalingPolicy(enable=resolution_policy),
        deterministic=deterministic,
        preblur=preblur,
        s16mm=s16mm,
    )

    sig = Signature.generate(film_grain_plus, settings)
    user_overrides = sig.changed

    if preset is not None:
        settings = settings.apply_preset(preset.value)

    settings = replace(settings, **user_overrides)

    return FilmGrainProcessor(settings).process(clip, debug)


def grain_factory(
    clip: vs.VideoNode,
    dark_grain: GrainConfig = GrainConfig(strength=7.0, sharpness=60, size=1.5, grainer=Grainer.GAUSS),
    mid_grain: GrainConfig = GrainConfig(strength=5.0, sharpness=66, size=1.2, grainer=Grainer.GAUSS),
    bright_grain: GrainConfig = GrainConfig(strength=3.0, sharpness=80, size=0.9, grainer=Grainer.GAUSS),
    seed: int = 17,
    th1: int = 24,
    th2: int = 56,
    th3: int = 128,
    th4: int = 160,
    blur: bool = True,
) -> vs.VideoNode:
    neutral = get_neutral_value(clip)
    peak = get_peak_value(clip)

    th1, th2, th3, th4 = [
        scale_value(
            x,
            8,
            clip.format.bits_per_sample,  # type: ignore
            vs.ColorRange.RANGE_FULL,
        )
        for x in [th1, th2, th3, th4]
    ]

    factory = GrainLayerFactory()
    grain = [
        factory.create(
            clip,
            grain,
            seed,
            True,
            i + 1
        )
        for i, grain in enumerate([dark_grain, mid_grain, bright_grain])
    ]

    with inline_expr([clip, *grain]) as ie:
        x, grain1, grain2, grain3 = ie.vars

        ramp1 = (peak / (th2 - th1)) * (x - th1)
        m1 = ie.op.tern(x < th1, 0, ie.op.tern(x > th2, peak, ramp1))

        ramp2 = (peak / (th4 - th3)) * (x - th3)
        m2 = ie.op.tern(x < th3, 0, ie.op.tern(x > th4, peak, ramp2))

        fm = grain1 + ((grain2 - grain1) * m1 + peak / 2) / peak
        blended = fm + ((grain3 - fm) * m2 + peak / 2) / peak

        if blur:
            ie.out = x - blended

        ie.out = x - blended + neutral

    result = ie.clip

    if blur:
        # diff = core.std.MakeDiff(clip, result)
        soft = remove_grain(result, 12)
        result = core.std.MergeDiff(clip, soft)

    return result
