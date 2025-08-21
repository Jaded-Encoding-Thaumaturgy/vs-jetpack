from __future__ import annotations

from typing import Any, Iterable, Iterator, MutableMapping, TypedDict, overload

from typing_extensions import Self, deprecated

from vstools import T1, T2, KwargsT, PlanesT, SupportsKeysAndGetItem, VSFunctionNoArgs, classproperty, vs, vs_object

from ..prefilters import prefilter_to_full_range
from .enums import FlowMode, MaskMode, MotionMode, PenaltyMode, RFilterMode, SADMode, SearchMode, SharpMode, SmoothMode

__all__ = [
    "AnalyzeArgs",
    "BlockFpsArgs",
    "CompensateArgs",
    "DegrainArgs",
    "FlowArgs",
    "FlowBlurArgs",
    "FlowFpsArgs",
    "FlowInterpolateArgs",
    "MVToolsPreset",
    "MVToolsPresets",
    "MaskArgs",
    "RecalculateArgs",
    "ScDetectionArgs",
    "SuperArgs",
]


class SuperArgs(TypedDict, total=False):
    levels: int | None
    sharp: SharpMode | None
    rfilter: RFilterMode | None
    pelclip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None


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


class MVToolsPreset(MutableMapping[str, Any], vs_object):
    search_clip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode]
    tr: int
    pel: int
    pad: int | tuple[int | None, int | None]
    planes: PlanesT
    super_args: SuperArgs | KwargsT
    analyze_args: AnalyzeArgs | KwargsT
    recalculate_args: RecalculateArgs | KwargsT
    compensate_args: CompensateArgs | KwargsT
    flow_args: FlowArgs | KwargsT
    degrain_args: DegrainArgs | KwargsT
    flow_interpolate_args: FlowInterpolateArgs | KwargsT
    flow_fps_args: FlowFpsArgs | KwargsT
    block_fps_args: BlockFpsArgs | KwargsT
    flow_blur_args: FlowBlurArgs | KwargsT
    mask_args: MaskArgs | KwargsT
    sc_detection_args: ScDetectionArgs | KwargsT

    def __init__(
        self,
        *,
        search_clip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
        tr: int | None = None,
        pel: int | None = None,
        pad: int | tuple[int | None, int | None] | None = None,
        planes: PlanesT | None = None,
        super_args: SuperArgs | KwargsT | None = None,
        analyze_args: AnalyzeArgs | KwargsT | None = None,
        recalculate_args: RecalculateArgs | KwargsT | None = None,
        compensate_args: CompensateArgs | KwargsT | None = None,
        flow_args: FlowArgs | KwargsT | None = None,
        degrain_args: DegrainArgs | KwargsT | None = None,
        flow_interpolate_args: FlowInterpolateArgs | KwargsT | None = None,
        flow_fps_args: FlowFpsArgs | KwargsT | None = None,
        block_fps_args: BlockFpsArgs | KwargsT | None = None,
        flow_blur_args: FlowBlurArgs | KwargsT | None = None,
        mask_args: MaskArgs | KwargsT | None = None,
        sc_detection_args: ScDetectionArgs | KwargsT | None = None,
    ) -> None:
        self._dict = dict[str, Any](
            search_clip=search_clip,
            tr=tr,
            pel=pel,
            pad=pad,
            planes=planes,
            super_args=super_args,
            analyze_args=analyze_args,
            recalculate_args=recalculate_args,
            compensate_args=compensate_args,
            flow_args=flow_args,
            degrain_args=degrain_args,
            flow_interpolate_args=flow_interpolate_args,
            flow_fps_args=flow_fps_args,
            block_fps_args=block_fps_args,
            flow_blur_args=flow_blur_args,
            mask_args=mask_args,
            sc_detection_args=sc_detection_args,
        )

    def __str__(self) -> str:
        return self.__clean_dict__().__str__()

    def __clean_dict__(self) -> dict[str, Any]:
        return {k: v for k, v in self._dict.items() if v is not None}

    def __getattr__(self, name: str) -> Any:
        if name in self.__annotations__:
            try:
                return self.__clean_dict__()[name]
            except KeyError as e:
                raise AttributeError from e

        return self.__dict__[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__annotations__:
            self._dict[name] = value

        self.__dict__[name] = value

    def __getitem__(self, key: str) -> Any:
        return self.__clean_dict__().__getitem__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._dict.__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._dict.__delitem__(key)

    def __iter__(self) -> Iterator[str]:
        return self.__clean_dict__().__iter__()

    def __len__(self) -> int:
        return self.__clean_dict__().__len__()

    def __or__(self, value: MutableMapping[str, Any], /) -> MVToolsPreset:
        return self.__class__(**self._dict | dict(value))

    @overload
    def __ror__(self, value: MutableMapping[str, Any], /) -> dict[str, Any]: ...

    @overload
    def __ror__(self, value: MutableMapping[T1, T2], /) -> dict[str | T1, Any | T2]: ...

    def __ror__(self, value: Any, /) -> Any:
        return self.__class__(**dict(value) | self._dict)

    @overload  # type: ignore[misc]
    def __ior__(self, value: SupportsKeysAndGetItem[str, Any], /) -> Self: ...

    @overload
    def __ior__(self, value: Iterable[tuple[str, Any]], /) -> Self: ...

    def __ior__(self, value: Any, /) -> Self:  # type: ignore[misc]
        self._dict |= dict[str, Any](value)
        return self

    def __vs_del__(self, core_id: int) -> None:
        self.clear()

    @classproperty
    @classmethod
    def HQ_COHERENCE(cls) -> Self:  # noqa: N802
        return cls(
            search_clip=prefilter_to_full_range,
            analyze_args=AnalyzeArgs(
                blksize=16,
                overlap=8,
            ),
            recalculate_args=RecalculateArgs(
                blksize=8,
                overlap=4,
                dct=SADMode.SATD,
            ),
        )

    @classproperty
    @classmethod
    def HQ_SAD(cls) -> Self:  # noqa: N802
        return cls(
            search_clip=prefilter_to_full_range,
            analyze_args=AnalyzeArgs(
                blksize=16,
                overlap=8,
                truemotion=MotionMode.SAD,
            ),
            recalculate_args=RecalculateArgs(
                blksize=8,
                overlap=4,
                dct=SADMode.SATD,
                truemotion=MotionMode.SAD,
            ),
        )


@deprecated(
    "MVToolsPresets class is deprecated and will be removed in a future version. Use MVToolsPreset instead.",
    category=DeprecationWarning,
)
class MVToolsPresets:
    """
    Presets for arguments passed to MVTools functions.
    """

    @classproperty
    @classmethod
    def HQ_COHERENCE(cls) -> MVToolsPreset:  # noqa: N802
        return MVToolsPreset.HQ_COHERENCE

    @classproperty
    @classmethod
    def HQ_SAD(cls) -> MVToolsPreset:  # noqa: N802
        return MVToolsPreset.HQ_SAD
