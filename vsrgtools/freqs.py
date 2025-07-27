from __future__ import annotations

from enum import auto
from typing import Any, Literal, overload

from vsexprtools import ExprOp, ExprVars, combine, norm_expr
from vstools import (
    ConstantFormatVideoNode,
    CustomIntEnum,
    FuncExceptT,
    PlanesT,
    StrList,
    VideoNodeIterableT,
    check_variable_format,
    flatten_vnodes,
    vs,
)

__all__ = ["MeanMode"]


class MeanMode(CustomIntEnum):
    POWER = auto()

    LEHMER = auto()

    HARMONIC = -1

    GEOMETRIC = 0

    ARITHMETIC = 1

    RMS = 2

    CUBIC = 3

    MINIMUM = auto()

    MAXIMUM = auto()

    CONTRAHARMONIC = auto()

    MEDIAN = auto()

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[MeanMode.POWER, MeanMode.LEHMER],
        *_clips: VideoNodeIterableT[vs.VideoNode],
        p: float = ...,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
    ) -> ConstantFormatVideoNode: ...

    @overload
    def __call__(
        self,
        *_clips: VideoNodeIterableT[vs.VideoNode],
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
    ) -> ConstantFormatVideoNode: ...

    def __call__(
        self,
        *_clips: VideoNodeIterableT[vs.VideoNode],
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        assert check_variable_format(clips, func)

        n_clips = len(clips)
        all_clips = ExprVars(n_clips)

        if n_clips < 2:
            return next(iter(clips))

        match self:
            case MeanMode.POWER:
                p = kwargs.get("p", -1)
                return combine(
                    clips,
                    ExprOp.ADD,
                    f"neutral - {p} pow",
                    expr_suffix=f"{n_clips} / {1 / p} pow neutral +",
                    planes=planes,
                    func=func,
                )

            case MeanMode.LEHMER:
                p = kwargs.get("p", 2)
                counts = range(n_clips)

                expr = StrList([[f"{clip} neutral - D{i}!" for i, clip in zip(counts, all_clips)]])
                for x in range(2):
                    expr.extend([[f"D{i}@ {p - x} pow" for i in counts], ExprOp.ADD * (n_clips - 1)])

                return norm_expr(clips, f"{expr} / neutral +", planes, func=func)

            case MeanMode.HARMONIC | MeanMode.GEOMETRIC | MeanMode.RMS | MeanMode.CUBIC:
                return MeanMode.POWER(clips, p=self.value, planes=planes, func=func)

            case MeanMode.CONTRAHARMONIC:
                return MeanMode.LEHMER(clips, p=2, planes=planes, func=func)

            case MeanMode.ARITHMETIC:
                return combine(clips, ExprOp.ADD, expr_suffix=f"{n_clips} /", planes=planes, func=func)

            case MeanMode.MINIMUM:
                return ExprOp.MIN(clips, planes=planes, func=func)

            case MeanMode.MAXIMUM:
                return ExprOp.MAX(clips, planes=planes, func=func)

            case MeanMode.MEDIAN:
                n_op = (n_clips - 1) // 2

                mean = "" if n_clips % 2 else "+ 2 /"

                return norm_expr(
                    clips, f"{all_clips} sort{n_clips} drop{n_op} {mean} swap{n_op} drop{n_op}", planes, func=func
                )
