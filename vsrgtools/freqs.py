from __future__ import annotations

from enum import auto
from typing import Any, Iterable, Literal, Sequence, SupportsIndex, overload

from jetpytools import CustomEnum, CustomValueError, FuncExcept, SupportsString, normalize_seq

from vsexprtools import ExprList, ExprOp, ExprVars, combine_expr, norm_expr
from vstools import (
    ConvMode,
    HoldsVideoFormat,
    Planes,
    VideoFormatLike,
    VideoNodeIterable,
    flatten_vnodes,
    shift_clip_multi,
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
    """Minimum value across all clips."""

    MAXIMUM = auto()
    """Maximum value across all clips."""

    MEDIAN = auto()
    """Median value across all clips."""

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[MeanMode.LEHMER],
        *_clips: VideoNodeIterable,
        p: float = 3,
        planes: Planes = None,
        func: FuncExcept | None = None,
    ) -> vs.VideoNode:
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
        *_clips: VideoNodeIterable,
        planes: Planes = None,
        func: FuncExcept | None = None,
    ) -> vs.VideoNode: ...

    def __call__(
        self,
        *_clips: VideoNodeIterable,
        planes: Planes = None,
        func: FuncExcept | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Applies the selected mean to multiple clips.

        Args:
            *_clips: Input clips to combine.
            planes: Which planes to process.
            func: An optional function to use for error handling.
            **kwargs: Additional keyword arguments for certain modes.

                   - p (float): Exponent for `LEHMER` mode. Defaults to 3.
        Raises:
            CustomValueError: If there is no clip.

        Returns:
            A new clip containing the combined frames.
        """

        func = func or self.__class__

        clips = flatten_vnodes(_clips)

        if not clips:
            raise CustomValueError("There is no clip to evaluate.", func)

        if (n_clips := len(clips)) < 2:
            return clips[0]

        return self.expr(n_clips, **kwargs)(clips, planes=planes, func=func)

    def single(
        self,
        clip: vs.VideoNode,
        radius: int | Sequence[int] = 1,
        mode: ConvMode = ConvMode.SQUARE,
        exclude: Iterable[tuple[int, int]] | None = None,
        include: Iterable[tuple[int, int]] | None = None,
        planes: Planes = None,
        func: FuncExcept | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Applies the selected mean to one clip, spatially or temporally.

        Args:
            clip: Input clip.
            radius: The radius per plane (Sequence) or uniform radius (int). Only int is allowed in temporal mode.
            mode: The convolution mode. Defaults to ConvMode.SQUARE.
            exclude: Optional set of (x, y) coordinates to exclude from the matrix.
            include: Optional set of (x, y) coordinates to include in the matrix.
            planes: Which planes to process.
            func: An optional function to use for error handling.
            **kwargs: Additional keyword arguments for certain modes.

                   - p (float): Exponent for `LEHMER` mode. Defaults to 3.

        Raises:
            CustomValueError: If a list is passed for radius in temporal mode, which is unsupported.

        Returns:
            A new clip with the mean applied.
        """
        func = func or f"{self!s}.single"

        if mode == ConvMode.TEMPORAL:
            if not isinstance(radius, int):
                raise CustomValueError("A list of radius isn't supported for ConvMode.TEMPORAL!", func, radius)

            clips = shift_clip_multi(clip, (-radius, radius))

            (ops,) = ExprOp.matrix(ExprVars(len(clips)), radius, mode, exclude, include)

            return self.expr(ops, **kwargs)(*clips, planes=planes, func=func)

        radius = normalize_seq(radius, clip.format.num_planes)
        expr_plane = list[list[ExprList]]()

        for r in radius:
            expr_passes = list[ExprList]()

            for mat in ExprOp.matrix("x", r, mode, exclude, include):
                expr_passes.append(self.expr(mat, **kwargs))

            expr_plane.append(expr_passes)

        for e in zip(*expr_plane):
            clip = norm_expr(clip, e, planes, func=func)

        return clip

    def expr(
        self,
        n: SupportsIndex | Sequence[SupportsString] | HoldsVideoFormat | VideoFormatLike,
        eps: float = 1e-7,
        **kwargs: Any,
    ) -> ExprList:
        """
        Builds a mean expression using a specified mode.

        Args:
            n: Object from which to infer the variables.
            eps: Small constant to avoid division by zero. Defaults to 1e-7.
            **kwargs: Additional keyword arguments for certain modes.

                   - p (float): Exponent for `LEHMER` mode. Defaults to 3.

        Returns:
            Mean expression.
        """
        evars = n if isinstance(n, Sequence) else ExprVars(n)

        if (n_len := len(evars)) < 2:
            return ExprList([evars[0]])

        match self:
            case MeanMode.LEHMER:
                p = kwargs.pop("p", self.value)

                expr = ExprList((f"{v} neutral - {eps} + D{i}!" for i, v in enumerate(evars)))

                for x in range(2):
                    expr.extend([[f"D{i}@ {p - x} pow" for i in range(n_len)], ExprOp.ADD * (n_len - 1), f"P{x}!"])

                expr.append("P1@ 0 = 0 P0@ P1@ / ? neutral +")

                return expr
            case MeanMode.HARMONIC | MeanMode.GEOMETRIC | MeanMode.CONTRAHARMONIC:
                return MeanMode.LEHMER.expr(n, eps, p=self.value)
            case MeanMode.MEDIAN:
                n_op = (n_len - 1) // 2
                mean = "" if n_len % 2 else "+ 2 /"

                return ExprList(
                    [
                        f"{' '.join(str(v) for v in evars)}",
                        f"sort{n_len}",
                        f"drop{n_op}",
                        f"{mean}",
                        f"swap{n_op}",
                        f"drop{n_op}",
                    ]
                )
            case MeanMode.ARITHMETIC:
                op = ExprOp.ADD
                kwargs.update(expr_suffix=f"{n_len} /")
            case MeanMode.MINIMUM:
                op = ExprOp.MIN
            case MeanMode.MAXIMUM:
                op = ExprOp.MAX

        return combine_expr(evars, op, **kwargs)
