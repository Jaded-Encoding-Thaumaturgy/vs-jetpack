from typing import TypedDict

from vstools import VSFunctionNoArgs, vs

from .enums import FlowMode, MaskMode, MotionMode, PenaltyMode, RFilterMode, SADMode, SearchMode, SharpMode

__all__ = [
    "AnalyzeArgs",
    "BlockFpsArgs",
    "CompensateArgs",
    "DegrainArgs",
    "FlowArgs",
    "FlowBlurArgs",
    "FlowFpsArgs",
    "FlowInterpolateArgs",
    "MaskArgs",
    "RecalculateArgs",
    "ScDetectionArgs",
    "SuperArgs",
]


class SuperArgs(TypedDict, total=False):
    levels: int | None
    sharp: SharpMode | None
    rfilter: RFilterMode | None
    pelclip: vs.VideoNode | VSFunctionNoArgs | None


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
    limit: int | None
    limitc: int | None
    thscd1: int | None
    thscd2: int | None
    plane: int | None


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
