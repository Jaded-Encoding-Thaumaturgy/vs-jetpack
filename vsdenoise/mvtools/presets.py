from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Self

from jetpytools import KwargsNotNone, classproperty

from vstools import VSFunctionNoArgs, VSObjectABC, vs

from ..prefilters import prefilter_to_full_range
from .enums import MotionMode, SADMode
from .types import (
    AnalyzeArgs,
    BlockFpsArgs,
    CompensateArgs,
    DegrainArgs,
    FlowArgs,
    FlowBlurArgs,
    FlowFpsArgs,
    FlowInterpolateArgs,
    MaskArgs,
    RecalculateArgs,
    ScDetectionArgs,
    SuperArgs,
)

__all__ = [
    "MVToolsPreset",
]


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
    block_fps_args: BlockFpsArgs
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
        block_fps_args: BlockFpsArgs | None = None,
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
            block_fps_args=block_fps_args,
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
