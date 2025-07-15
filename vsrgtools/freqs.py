from __future__ import annotations

from itertools import count

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
    MINIMUM = -2
    HARMONIC = -1
    GEOMETRIC = 0

    ARITHMETIC = 1

    RMS = 2
    CUBIC = 3
    MAXIMUM = 4

    LEHMER = 10

    MINIMUM_ABS = 20
    MAXIMUM_ABS = 21

    MEDIAN = 30

    def __call__(
        self, *_clips: VideoNodeIterableT[vs.VideoNode], planes: PlanesT = None, func: FuncExceptT | None = None
    ) -> ConstantFormatVideoNode:
        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        assert check_variable_format(clips, func)

        n_clips = len(clips)
        n_op = n_clips - 1

        if n_clips < 2:
            return next(iter(clips))

        if self == MeanMode.MINIMUM:
            return ExprOp.MIN(clips, planes=planes, func=func)

        if self == MeanMode.MAXIMUM:
            return ExprOp.MAX(clips, planes=planes, func=func)

        if self == MeanMode.GEOMETRIC:
            return combine(clips, ExprOp.MUL, None, None, [1 / n_clips, ExprOp.POW], planes=planes, func=func)

        if self == MeanMode.LEHMER:
            counts = range(n_clips)
            clip_vars = ExprVars(n_clips)

            expr = StrList([[f"{clip} neutral - D{i}!" for i, clip in zip(counts, clip_vars)]])

            for y in range(2):
                expr.extend([[f"D{i}@ {3 - y} pow" for i in counts], ExprOp.ADD * n_op, f"P{y + 1}!"])

            expr.append("P2@ P1@ P2@ / 0 ? neutral +")

            return norm_expr(clips, expr, planes=planes, func=func)

        if self in {MeanMode.RMS, MeanMode.ARITHMETIC, MeanMode.CUBIC, MeanMode.HARMONIC}:
            return combine(
                clips,
                ExprOp.ADD,
                f"{self.value} {ExprOp.POW}",
                None,
                [n_clips, ExprOp.DIV, 1 / self, ExprOp.POW],
                planes=planes,
                func=func,
            )

        if self in {MeanMode.MINIMUM_ABS, MeanMode.MAXIMUM_ABS}:
            operator = ExprOp.MIN if self is MeanMode.MINIMUM_ABS else ExprOp.MAX

            expr_string = ""
            for src in ExprVars(n_clips):
                expr_string += f"{src} neutral - abs {src.upper()}D! "

            for i, src, srcn in zip(count(), ExprVars(n_clips), ExprVars(1, n_clips)):
                expr_string += f"{src.upper()}D@ {srcn.upper()}D@ {operator} {src} "

                if i == n_clips - 2:
                    expr_string += f"{srcn} "

            expr_string += "? " * n_op

            return norm_expr(clips, expr_string, planes=planes, func=func)

        if self == MeanMode.MEDIAN:
            all_clips = str(ExprVars(1, n_clips))

            n_ops = n_clips - 2

            yzmin, yzmax = [all_clips + f" {op}" * n_ops for op in (ExprOp.MIN, ExprOp.MAX)]

            return norm_expr(
                clips,
                f"{yzmin} YZMIN! {yzmax} YZMAX! x YZMIN@ min x = YZMIN@ x YZMAX@ max x = YZMAX@ x ? ?",
                planes,
                func=func,
            )

        raise CustomNotImplementedError
