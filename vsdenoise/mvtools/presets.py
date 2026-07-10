from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Required, Self, TypedDict

from jetpytools import KwargsNotNone, classproperty

from vstools import VSFunctionNoArgs, VSObjectABC, vs

from ..prefilters import prefilter_to_full_range
from .enums import PenaltyMode, RFilterMode, SearchMode, SharpMode

__all__ = [
    "AnalyzeArgs",
    "CompensateArgs",
    "DegrainArgs",
    "FlowArgs",
    "FlowBlurArgs",
    "FlowFpsArgs",
    "FlowInterpolateArgs",
    "MVToolsPreset",
    "MaskArgs",
    "RecalculateArgs",
    "ScDetectionArgs",
    "SuperArgs",
]


class SuperArgs(TypedDict, total=False):
    onelevel: bool | None
    sharp: SharpMode | None
    rfilter: RFilterMode | None
    pelclip: vs.VideoNode | VSFunctionNoArgs | None


class AnalyzeArgs(TypedDict, total=False):
    blksize: Required[int | tuple[int, int]]
    overlap_div: Required[int | tuple[int, int]]
    levels: int | None
    search: SearchMode | None
    searchparam: int | None
    pelsearch: int | None
    mvlambda: int | None
    lsad: int | None
    plevel: PenaltyMode | None
    globalmv: bool | None
    pnew: int | None
    pzero: int | None
    pglobal: int | None
    badsad: int | None
    badrange: int | None
    meander: bool | None
    trymany: bool | None
    satd: bool | None


class RecalculateArgs(TypedDict, total=False):
    blksize: Required[int | tuple[int, int]]
    overlap_div: Required[int | tuple[int, int]]
    thsad: int | None
    search: SearchMode | None
    searchparam: int | None
    mvlambda: int | None
    pnew: int | None
    meander: bool | None
    satd: bool | None


class CompensateArgs(TypedDict, total=False):
    thsad: int | None
    time: float | None
    thscd1: int | None
    thscd2: float | None


class FlowArgs(TypedDict, total=False):
    time: float | None
    thscd1: int | None
    thscd2: float | None


class DegrainArgs(TypedDict, total=False):
    thsad: int | tuple[int, int] | None
    limit: int | tuple[int, int] | None
    thscd1: int | None
    thscd2: float | None
    weights: Sequence[int] | None


class FlowInterpolateArgs(TypedDict, total=False):
    time: float | None
    ml: float | None
    blend: bool | None
    thscd1: int | None
    thscd2: float | None


class FlowFpsArgs(TypedDict, total=False):
    extramask: bool | None
    ml: float | None
    blend: bool | None
    thscd1: int | None
    thscd2: float | None
    num: int
    den: int


class FlowBlurArgs(TypedDict, total=False):
    blur: float | None
    prec: int | None
    thscd1: int | None
    thscd2: float | None


class MaskArgs(TypedDict, total=False):
    ml: float | None
    gamma: float | None
    time: float | None
    scval: float | None
    thscd1: int | None
    thscd2: float | None


class ScDetectionArgs(TypedDict, total=False):
    thscd1: int | None
    thscd2: float | None


class MVToolsPreset(VSObjectABC, Mapping[str, Any]):
    search_clip: vs.VideoNode | VSFunctionNoArgs
    pel: int
    pad: int | tuple[int | None, int | None]
    chroma: bool
    super_args: SuperArgs
    analyze_args: AnalyzeArgs
    recalculate_args: RecalculateArgs
    compensate_args: CompensateArgs
    flow_args: FlowArgs
    degrain_args: DegrainArgs
    flow_interpolate_args: FlowInterpolateArgs
    flow_fps_args: FlowFpsArgs
    flow_blur_args: FlowBlurArgs
    mask_args: MaskArgs
    sc_detection_args: ScDetectionArgs

    def __init__(
        self,
        *,
        search_clip: vs.VideoNode | VSFunctionNoArgs | None = None,
        pel: int | None = None,
        pad: int | tuple[int | None, int | None] | None = None,
        chroma: bool | None = None,
        super_args: SuperArgs | None = None,
        analyze_args: AnalyzeArgs | None = None,
        recalculate_args: RecalculateArgs | None = None,
        compensate_args: CompensateArgs | None = None,
        flow_args: FlowArgs | None = None,
        degrain_args: DegrainArgs | None = None,
        flow_interpolate_args: FlowInterpolateArgs | None = None,
        flow_fps_args: FlowFpsArgs | None = None,
        flow_blur_args: FlowBlurArgs | None = None,
        mask_args: MaskArgs | None = None,
        sc_detection_args: ScDetectionArgs | None = None,
    ) -> None:
        self._dict = KwargsNotNone(
            search_clip=search_clip,
            pel=pel,
            pad=pad,
            chroma=chroma,
            super_args=super_args,
            analyze_args=analyze_args,
            recalculate_args=recalculate_args,
            compensate_args=compensate_args,
            flow_args=flow_args,
            degrain_args=degrain_args,
            flow_interpolate_args=flow_interpolate_args,
            flow_fps_args=flow_fps_args,
            flow_blur_args=flow_blur_args,
            mask_args=mask_args,
            sc_detection_args=sc_detection_args,
        )

    def __str__(self) -> str:
        return self._dict.__str__()

    def __getattr__(self, name: str) -> Any:
        d = self._dict if name in self.__annotations__ else self.__dict__

        try:
            return d[name]
        except KeyError:
            pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'") from None

    def __getitem__(self, key: str) -> Any:
        return self._dict.__getitem__(key)

    def __iter__(self) -> Iterator[str]:
        return self._dict.__iter__()

    def __len__(self) -> int:
        return self._dict.__len__()

    def __or__(self, value: Mapping[str, Any], /) -> dict[str, Any]:
        return self._dict | dict(value)

    def __ror__(self, value: Mapping[str, Any], /) -> dict[str, Any]:
        return dict(value) | self._dict

    def copy(self) -> dict[str, Any]:
        """Return a shallow copy of the preset."""
        return self._dict.copy()

    @classproperty
    @classmethod
    def HQ_COHERENCE(cls) -> Self:  # noqa: N802
        return cls(
            search_clip=prefilter_to_full_range,
            analyze_args=AnalyzeArgs(blksize=16, overlap_div=2),
            recalculate_args=RecalculateArgs(blksize=8, overlap_div=2, satd=True),
        )

    @classproperty
    @classmethod
    def HQ_SAD(cls) -> Self:  # noqa: N802
        return cls(
            search_clip=prefilter_to_full_range,
            analyze_args=AnalyzeArgs(blksize=16, overlap_div=2, mvlambda=0),
            recalculate_args=RecalculateArgs(blksize=8, overlap_div=2, satd=True, mvlambda=0),
        )
