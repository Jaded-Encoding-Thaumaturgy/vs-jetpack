from __future__ import annotations

from enum import auto
from typing import Any, Literal, overload

from vsexprtools import ExprOp, ExprVars, norm_expr
from vstools import (
    ConstantFormatVideoNode,
    CustomEnum,
    FuncExceptT,
    PlanesT,
    StrList,
    VideoNodeIterableT,
    check_variable_format,
    flatten_vnodes,
    vs,
)

__all__ = ["MeanMode"]


class MeanMode(CustomEnum):
    """
    Enum of different mean for combining clips.
    """

    HARMONIC = 0
    """Harmonic mean, implemented as a Lehmer mean with p=0."""

    GEOMETRIC = 0.5
    """Geometric mean, implemented as a Lehmer mean with p=0.5."""

    ARITHMETIC = 1
    """Arithmetic mean."""

    CONTRAHARMONIC = 2
    """Contraharmonic mean, implemented as a Lehmer mean withs p=2"""

    LEHMER = 3
    """
    Lehmer mean, configurable with parameter `p`.

    Note: An odd number for `p` is preferred as it will avoid negative inputs.
    """

    MINIMUM = auto()
    """Minimum value across all clips"""

    MAXIMUM = auto()
    """Maximum value across all clips"""

    MEDIAN = auto()
    """Median value across all clips"""

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[MeanMode.LEHMER],
        *_clips: VideoNodeIterableT[vs.VideoNode],
        p: float = 3,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
    ) -> ConstantFormatVideoNode:
        """
        Combine clips using the Lehmer mean with a configurable exponent.

        Args:
            *_clips: Input clips to combine.
            p: Exponent for the Lehmer mean calculation.
            planes: Which planes to process.
            func: An optional function to use for error handling.

        Returns:
            A new clip containing the combined frames.
        """

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
        """
        Applies the selected mean to one or more video clips.

        Args:
            *_clips: Input clips to combine.
            planes: Which planes to process.
           func: An optional function to use for error handling.
            **kwargs: Additional keyword arguments for certain modes.

                   - p (float): Exponent for `LEHMER` mode. Defaults to 3.

        Returns:
            A new clip containing the combined frames.
        """

        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        assert check_variable_format(clips, func)

        n_clips = len(clips)
        all_clips = ExprVars(n_clips)

        if n_clips < 2:
            return next(iter(clips))

        match self:
            case MeanMode.LEHMER:
                p = kwargs.get("p", self.value)
                counts = range(n_clips)

                expr = StrList([[f"{clip} neutral - D{i}!" for i, clip in zip(counts, all_clips)]])
                for x in range(2):
                    expr.extend([[f"D{i}@ {p - x} pow" for i in counts], ExprOp.ADD * (n_clips - 1), f"P{x}!"])

                return norm_expr(clips, f"{expr} P1@ 0 = 0 P0@ P1@ / ? neutral +", planes, func=func)

            case MeanMode.HARMONIC | MeanMode.GEOMETRIC | MeanMode.CONTRAHARMONIC:
                return MeanMode.LEHMER(clips, p=self.value, planes=planes, func=func)

            case MeanMode.ARITHMETIC:
                return ExprOp.ADD(clips, expr_suffix=f"{n_clips} /", planes=planes, func=func)

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
