from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, MutableMapping, Self, TypedDict, overload

from vstools import T1, T2, KwargsT, PlanesT, SupportsKeysAndGetItem, VSFunction, classproperty, vs

from .enums import FlowMode, MaskMode, MotionMode, SmoothMode, PenaltyMode, RFilterMode, SADMode, SearchMode, SharpMode
from ..prefilters import prefilter_to_full_range


__all__ = [
    "SuperArgs",
    "AnalyzeArgs",
    "RecalculateArgs",
    "CompensateArgs",
    "FlowArgs",
    "DegrainArgs",
    "FlowInterpolateArgs",
    "FlowFpsArgs",
    "BlockFpsArgs",
    "FlowBlurArgs",
    "MaskArgs",
    "ScDetectionArgs",
    "MVToolsPreset",
    "MVToolsPresets"
]


class SuperArgs(TypedDict, total=False):
    levels: int | None
    sharp: SharpMode | None
    rfilter: RFilterMode | None
    pelclip: vs.VideoNode | VSFunction | None


class AnalyzeArgs(TypedDict, total=False):
    blksize: int | None
    blksizev: int | None
    levels: int | None
    search: SearchMode | None
    searchparam: int | None
    pelsearch: int | None
    lambda_: int | None
    truemotion: MotionMode | None
    lsad: int | None
    plevel: PenaltyMode | None
    global_: bool | None
    pnew: int | None
    pzero: int | None
    pglobal: int | None
    overlap: int | None
    overlapv: int | None
    divide: bool | None
    badsad: int | None
    badrange: int | None
    meander: bool | None
    trymany: bool | None
    dct: SADMode | None


class RecalculateArgs(TypedDict, total=False):
    smooth: SmoothMode | None
    thsad: int | None
    blksize: int | None
    blksizev: int | None
    search: SearchMode | None
    searchparam: int | None
    lambda_: int | None
    truemotion: MotionMode | None
    pnew: int | None
    overlap: int | None
    overlapv: int | None
    divide: bool | None
    meander: bool | None
    dct: SADMode | None


class CompensateArgs(TypedDict, total=False):
    scbehavior: bool | None
    thsad: int | None
    thsad2: int | None
    time: float | None
    thscd1: int | None
    thscd2: int | None


class FlowArgs(TypedDict, total=False):
    time: float | None
    mode: FlowMode | None
    thscd1: int | None
    thscd2: int | None


class DegrainArgs(TypedDict, total=False):
    thsad: int | None
    thsadc: int | None
    thsad2: int | None
    limit: int | None
    limitc: int | None
    thscd1: int | None
    thscd2: int | None


class FlowInterpolateArgs(TypedDict, total=False):
    time: float | None
    ml: float | None
    blend: bool | None
    thscd1: int | None
    thscd2: int | None


class FlowFpsArgs(TypedDict, total=False):
    mask: int | None
    ml: float | None
    blend: bool | None
    thscd1: int | None
    thscd2: int | None
    num: int
    den: int


class BlockFpsArgs(TypedDict, total=False):
    mode: int | None
    ml: float | None
    blend: bool | None
    thscd1: int | None
    thscd2: int | None
    num: int
    den: int


class FlowBlurArgs(TypedDict, total=False):
    blur: float | None
    prec: int | None
    thscd1: int | None
    thscd2: int | None


class MaskArgs(TypedDict, total=False):
    ml: float | None
    gamma: float | None
    kind: MaskMode | None
    time: float | None
    ysc: int | None
    thscd1: int | None
    thscd2: int | None


class ScDetectionArgs(TypedDict, total=False):
    thscd1: int | None
    thscd2: int | None


@dataclass(kw_only=True)
class MVToolsPreset(MutableMapping[str, Any]):
    search_clip: VSFunction | None
    tr: int | None = None
    pel: int | None = None
    pad: int | tuple[int | None, int | None] | None = None
    planes: PlanesT | None = None
    super_args: SuperArgs | KwargsT | None = None
    analyze_args: AnalyzeArgs | KwargsT | None = None
    recalculate_args: RecalculateArgs | KwargsT | None = None
    compensate_args: CompensateArgs | KwargsT | None = None
    flow_args: FlowArgs | KwargsT | None = None
    degrain_args: DegrainArgs | KwargsT | None = None
    flow_interpolate_args: FlowInterpolateArgs | KwargsT | None = None
    flow_fps_args: FlowFpsArgs | KwargsT | None = None
    block_fps_args: BlockFpsArgs | KwargsT | None = None
    flow_blur_args: FlowBlurArgs | KwargsT | None = None
    mask_args: MaskArgs | KwargsT | None = None
    sc_detection_args: ScDetectionArgs | KwargsT | None = None

    def __post_init__(self) -> None:
        for k, v in self.__dict__.copy().items():
            if v is None:
                del self.__dict__[k]

    def __getitem__(self, key: str) -> Any:
        return self.__dict__.__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.__dict__.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self.__dict__.__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        return self.__dict__.__iter__()

    def __len__(self) -> int:
        return self.__dict__.__len__()

    def __or__(self, value: MutableMapping[str, Any], /) -> MVToolsPreset:
        return self.__class__(**self.__dict__ | dict(value))

    @overload
    def __ror__(self, value: MutableMapping[str, Any], /) -> dict[str, Any]:
        ...

    @overload
    def __ror__(self, value: MutableMapping[T1, T2], /) -> dict[str | T1, Any | T2]:
        ...

    def __ror__(self, value: Any, /) -> Any:
        return self.__class__(**dict(value) | self.__dict__)

    @overload  # type: ignore[misc]
    def __ior__(self, value: SupportsKeysAndGetItem[str, Any], /) -> Self:
        ...

    @overload
    def __ior__(self, value: Iterable[tuple[str, Any]], /) -> Self:
        ...

    def __ior__(self, value: Any, /) -> Self:  # type: ignore[misc]
        self.__dict__ |= dict[str, Any](value)
        return self


class MVToolsPresets:
    """Presets for MVTools analyzing/refining."""

    @classproperty
    def HQ_COHERENCE(self) -> MVToolsPreset:
        return MVToolsPreset(
            search_clip=prefilter_to_full_range,
            pel=2,
            super_args=SuperArgs(
                sharp=SharpMode.WIENER,
            ),
            analyze_args=AnalyzeArgs(
                blksize=16,
                overlap=8,
                search=SearchMode.HEXAGON,
                dct=SADMode.ADAPTIVE_SPATIAL_MIXED,
            ),
            recalculate_args=RecalculateArgs(
                blksize=8,
                overlap=4,
                search=SearchMode.HEXAGON,
                dct=SADMode.ADAPTIVE_SATD_MIXED,
            )
        )

    @classproperty
    def HQ_SAD(self) -> MVToolsPreset:
        return MVToolsPreset(
            search_clip=prefilter_to_full_range,
            pel=2,
            super_args=SuperArgs(
                sharp=SharpMode.WIENER,
            ),
            analyze_args=AnalyzeArgs(
                blksize=16,
                overlap=8,
                search=SearchMode.HEXAGON,
                dct=SADMode.ADAPTIVE_SPATIAL_MIXED,
                truemotion=MotionMode.SAD,
            ),
            recalculate_args=RecalculateArgs(
                blksize=8,
                overlap=4,
                search=SearchMode.HEXAGON,
                dct=SADMode.ADAPTIVE_SATD_MIXED,
                truemotion=MotionMode.SAD,
            )
        )
