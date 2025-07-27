from __future__ import annotations

from enum import auto

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
    HARMONIC = -1

    GEOMETRIC = 0

    ARITHMETIC = 1

    RMS = 2

    CUBIC = 3

    MINIMUM = auto()

    MAXIMUM = auto()

    CONTRAHARMONIC = auto()

    MEDIAN = auto()

    POWER = auto()

    LEHMER = auto()

    def __call__(
        self, *_clips: VideoNodeIterableT[vs.VideoNode], planes: PlanesT = None, func: FuncExceptT | None = None
    ) -> ConstantFormatVideoNode:
        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        assert check_variable_format(clips, func)

        n_clips = len(clips)

        if n_clips < 2:
            return next(iter(clips))

        match self:
            case MeanMode.ARITHMETIC:
                return combine(clips, ExprOp.ADD, expr_suffix=(n_clips, ExprOp.DIV), planes=planes, func=func)

            case MeanMode.MINIMUM:
                return ExprOp.MIN(clips, planes=planes, func=func)

            case MeanMode.MAXIMUM:
                return ExprOp.MAX(clips, planes=planes, func=func)

            case MeanMode.MEDIAN:
                all_clips = ExprVars(n_clips)
                n_op = (n_clips - 1) // 2

                mean = "" if n_clips % 2 else "+ 2 /"

                return norm_expr(
                    clips, f"{all_clips} sort{n_clips} drop{n_op} {mean} swap{n_op} drop{n_op}", planes, func=func
                )

            case MeanMode.HARMONIC | MeanMode.GEOMETRIC | MeanMode.RMS | MeanMode.CUBIC:
                return combine(
                    clips,
                    ExprOp.ADD,
                    f"{self.value} {ExprOp.POW}",
                    expr_suffix=(n_clips, ExprOp.DIV, 1 / self.value, ExprOp.POW),
                    planes=planes,
                    func=func,
                )

            case MeanMode.CONTRAHARMONIC:
                all_clips = ExprVars(n_clips)

                expr = StrList()
                for x in range(2):
                    expr.extend([[f"{clip} {2 - x} {ExprOp.POW}" for clip in all_clips], ExprOp.ADD * (n_clips - 1)])

                return norm_expr(clips, f"{expr} {ExprOp.DIV}", planes, func=func)

            case MeanMode.POWER:
                p = -1
                return combine(
                    clips,
                    ExprOp.ADD,
                    f"{p} {ExprOp.POW}",
                    expr_suffix=(n_clips, ExprOp.DIV, 1 / p, ExprOp.POW),
                    planes=planes,
                    func=func,
                )

            case MeanMode.LEHMER:
                p = 2
                all_clips = ExprVars(n_clips)

                expr = StrList()
                for x in range(2):
                    expr.extend([[f"{clip} {p - x} {ExprOp.POW}" for clip in all_clips], ExprOp.ADD * (n_clips - 1)])

                return norm_expr(clips, f"{expr} {ExprOp.DIV}", planes, func=func)
