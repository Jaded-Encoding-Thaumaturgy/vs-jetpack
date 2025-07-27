from __future__ import annotations

from enum import auto

from vsexprtools import ExprOp, ExprVars, combine, norm_expr
from vstools import (
    ConstantFormatVideoNode,
    CustomIntEnum,
    CustomNotImplementedError,
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
    ARITHMETIC = auto()

    MEDIAN = auto()

    MINIMUM = auto()

    MAXIMUM = auto()

    LEHMER = auto()

    RMS = auto()

    def __call__(
        self,
        *_clips: VideoNodeIterableT[vs.VideoNode],
        q: int = 3,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
    ) -> ConstantFormatVideoNode:
        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        assert check_variable_format(clips, func)

        n_clips = len(clips)
        n_op = n_clips - 1

        if n_clips < 2:
            return next(iter(clips))

        match self:
            case MeanMode.ARITHMETIC:
                return combine(clips, ExprOp.ADD, expr_suffix=(n_clips, ExprOp.DIV), planes=planes, func=func)

            case MeanMode.MEDIAN:
                all_clips = ExprVars(n_clips)
                n_ops = n_op // 2

                mean = "" if n_clips % 2 else "+ 2 /"

                return norm_expr(
                    clips, f"{all_clips} sort{n_clips} drop{n_ops} {mean} swap{n_ops} drop{n_ops}", planes, func=func
                )

            case MeanMode.MINIMUM:
                return ExprOp.MIN(clips, planes=planes, func=func)

            case MeanMode.MAXIMUM:
                return ExprOp.MAX(clips, planes=planes, func=func)

            case MeanMode.LEHMER:
                all_clips = ExprVars(n_clips)

                expr = StrList()
                for p in range(2):
                    expr.extend([[f"{clip} {q - p} {ExprOp.POW}" for clip in all_clips], ExprOp.ADD * n_op])

                return norm_expr(clips, f"{expr} /", planes, func=func)

            case MeanMode.RMS:
                return combine(
                    clips,
                    ExprOp.ADD,
                    f"2 {ExprOp.POW}",
                    expr_suffix=(n_clips, ExprOp.DIV, ExprOp.SQRT),
                    planes=planes,
                    func=func,
                )

        raise CustomNotImplementedError
